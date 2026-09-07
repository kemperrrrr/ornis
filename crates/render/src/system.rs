//! Typed plan systems + single declaration registry — S2 (PLAN.md,
//! Appendix C; IDEAS §28.1) and the d3 consolidation (formerly shared
//! with `FramePlan`, which is now folded into this module after d4).
//!
//! A pass declares resource accesses **in types** via ZST markers
//! (`Read<R>` / `Write<R>` / `WriteClear<R, C>` in tuples), and the scheduler
//! derives the graph wiring from them (reads/writes → lifetime → pool). A resource is
//! a type implementing [`FrameResource`]; the `type → ResourceId` mapping is
//! held by [`SystemSet`]. No strings and no syn parsing: resource identity is
//! the type (lesson from `smart_pipeline` brittleness, see Appendix C anti-goals).
//!
//! [`SystemSet`] is also the single declaration registry (d3): resource
//! and pass declarations live here together with the declaration
//! generation, explicit ordering edges and the [`Budget`] value.
//! Compiling declarations into an executable layout is the
//! registry-owned [`TransientPool`](crate::transient_pool::TransientPool)'s
//! job (cold paths), while the frame hot path compiles through the
//! executor-owned pool (`FrameExecutor::ensure_layout`). The imperative
//! builder (`add_pass().read/write`) remains for tests, tools and the
//! `scheduler_parity` oracle; production declares via types.
//!
//! S2 boundaries: access sets are static. Passes whose accesses depend on
//! configuration (depth ownership in forward, bloom input selection, blending in
//! composite) remain on the imperative path in `frame_exec.rs` until the S2b
//! decision (variant types vs. registration-as-selection).
//!
//! Pass order remains registration order (insertion order);
//! `.before()`/`.after()` and auto-parallelism are S5.

use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Mutex;

use ornis_schedule::{OrderError, resolve_named_edge, validate_indexed_edge};

use crate::frame_exec::PassViews;
use crate::mesh::Mesh;
use crate::renderer::Renderer3D;
use crate::transient_pool::{
    Budget, BudgetExceeded, FrameLayout, PassId, PassNode, PoolInput, ResourceId, ResourceNode,
    TextureSpec, TransientPool,
};

/// How a resource enters the registry (see
/// [`SystemSet::{create_resource, import_resource, external_output}`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    /// Created and owned by the plan (transient, pooled).
    FrameOwned,
    /// Imported external input, read-only (e.g. an uploaded shadow map).
    Imported,
    /// Externally backed output (e.g. the swapchain view); never pooled.
    ExternalOutput,
}

/// A typed plan resource: a ZST marker type carrying the resource's
/// identity. One type = one resource per plan.
pub trait FrameResource: 'static {
    /// Unique debug name (layout dumps, panics). Must be unique per plan.
    const NAME: &'static str;
    /// How the resource is registered in [`SystemSet`].
    fn kind() -> ResourceKind;
    /// Texture spec; `surface_format` feeds resources that mirror the
    /// surface format (e.g. the HDR layer).
    fn spec(surface_format: wgpu::TextureFormat) -> TextureSpec;
}

/// A clear value attached to a [`WriteClear`] access.
pub trait ClearValue {
    /// The color written when the attached target is cleared.
    const COLOR: wgpu::Color;
}

/// Clear to opaque black (HDR layers, bloom chain).
/// Clear to opaque black (HDR layers, bloom chain).
pub struct ClearBlack;
impl ClearValue for ClearBlack {
    const COLOR: wgpu::Color = wgpu::Color::BLACK;
}

/// Clear to opaque white (depth when the pass owns it).
/// Clear to opaque white (depth when the pass owns it).
pub struct ClearWhite;
impl ClearValue for ClearWhite {
    const COLOR: wgpu::Color = wgpu::Color::WHITE;
}

/// Clear to fully transparent (forward HDR layer).
/// Clear to fully transparent (forward HDR layer).
pub struct ClearTransparent;
impl ClearValue for ClearTransparent {
    const COLOR: wgpu::Color = wgpu::Color::TRANSPARENT;
}

/// One declared access: which resource, read or write, optional clear.
pub trait Access {
    /// The resource being accessed.
    type Resource: FrameResource;
    /// `true` for write accesses.
    const IS_WRITE: bool;
    /// Clear value applied on first write, if any.
    fn clear() -> Option<wgpu::Color>;
}

/// Read access marker (ZST): the pass only samples the resource's contents.
pub struct Read<R>(PhantomData<fn() -> R>);

/// Write access marker without an explicit clear (ZST); the pass fully
/// overwrites or continues the previous content.
pub struct Write<R>(PhantomData<fn() -> R>);

/// Write access marker carrying the clear value `C` (ZST), e.g. depth
/// owned by the gbuffer pass (`WriteClear<Hdr, ClearBlack>`).
pub struct WriteClear<R, C: ClearValue>(PhantomData<fn() -> (R, C)>);

impl<R: FrameResource> Access for Read<R> {
    type Resource = R;
    const IS_WRITE: bool = false;
    fn clear() -> Option<wgpu::Color> {
        None
    }
}

impl<R: FrameResource> Access for Write<R> {
    type Resource = R;
    const IS_WRITE: bool = true;
    fn clear() -> Option<wgpu::Color> {
        None
    }
}

impl<R: FrameResource, C: ClearValue> Access for WriteClear<R, C> {
    type Resource = R;
    const IS_WRITE: bool = true;
    fn clear() -> Option<wgpu::Color> {
        Some(C::COLOR)
    }
}

/// Runtime projection of one access (used to wire the pass into the plan).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccessDesc {
    /// Type identity of [`FrameResource`].
    pub resource: TypeId,
    /// Debug name from [`FrameResource::NAME`].
    pub name: &'static str,
    /// Write vs read access.
    pub write: bool,
    /// Clear color carried by a `WriteClear` access.
    pub clear: Option<wgpu::Color>,
}

/// A type-level set of accesses: tuples of access markers, e.g.
/// `(Read<Albedo>, Write<Hdr>)`. Collect order = declaration order.
/// Arity is supported up to 6 (one `impl_access_tuple!` line below adds
/// more if a pass ever needs it).
pub trait AccessSet {
    /// Push this set's accesses onto `out`, in declaration order.
    fn collect_accesses(out: &mut Vec<AccessDesc>);
}

impl AccessSet for () {
    fn collect_accesses(_out: &mut Vec<AccessDesc>) {}
}

/// Every access resolves to a texture view of the same lifetime; the helper
/// trait exists so tuple `Views` types can be built per-element in a macro.
pub trait AccessView<'a> {
    /// Resolved view type for one access (always `&'a wgpu::TextureView`).
    type View;
}

impl<'a, A: Access> AccessView<'a> for A {
    type View = &'a wgpu::TextureView;
}

/// Resolves the views for an access set at execution time.
pub trait ViewsFor<'a>: AccessSet {
    /// A tuple of `&wgpu::TextureView` matching the access set's arity.
    type Views;
    /// Fetches the views from the frame resolver.
    fn fetch(resolver: &Resolver<'a>) -> Self::Views;
}

impl<'a> ViewsFor<'a> for () {
    type Views = ();
    fn fetch(_resolver: &Resolver<'a>) -> Self::Views {}
}

macro_rules! impl_access_tuple {
    ($($name:ident),+) => {
        impl<$($name: Access),+> AccessSet for ($($name,)+) {
            fn collect_accesses(out: &mut Vec<AccessDesc>) {
                $(
                    out.push(AccessDesc {
                        resource: TypeId::of::<$name::Resource>(),
                        name: <$name::Resource as FrameResource>::NAME,
                        write: $name::IS_WRITE,
                        clear: $name::clear(),
                    });
                )+
            }
        }

        impl<'a, $($name: Access),+> ViewsFor<'a> for ($($name,)+) {
            type Views = ($(<$name as AccessView<'a>>::View,)+);
            fn fetch(resolver: &Resolver<'a>) -> Self::Views {
                ($(resolver.view::<$name::Resource>(),)+)
            }
        }
    };
}

impl_access_tuple!(A);
impl_access_tuple!(A, B);
impl_access_tuple!(A, B, C);
impl_access_tuple!(A, B, C, D);
impl_access_tuple!(A, B, C, D, E);
impl_access_tuple!(A, B, C, D, E, F);

/// Resolves typed resources to live texture views during pass execution.
#[derive(Clone, Copy)]
pub struct Resolver<'a> {
    views: &'a PassViews<'a>,
    ids: &'a HashMap<TypeId, ResourceId>,
}

impl<'a> Resolver<'a> {
    /// The view backing resource `R` on the current pass.
    ///
    /// # Panics
    /// Panics if `R` was never registered in the [`SystemSet`], or if the
    /// resource is not alive on this pass (the declared access set makes
    /// the latter a wiring bug, not a runtime state). Debug builds also
    /// panic when `R` sits outside the pass's declared reads/writes —
    /// ground-truth enforcement in `PassViews::view_of` (backlog #6).
    pub fn view<R: FrameResource>(&self) -> &'a wgpu::TextureView {
        let id = self
            .ids
            .get(&TypeId::of::<R>())
            .unwrap_or_else(|| panic!("typed resource '{}' is not registered", R::NAME));
        self.views.view_of(*id)
    }
}

/// Per-frame execution context handed to every system: the render context
/// parts plus the frame's draw inputs.
pub struct Frame<'a> {
    /// Logical device owning pipeline/buffers.
    pub device: &'a wgpu::Device,
    /// Upload queue for uniform data.
    pub queue: &'a wgpu::Queue,
    /// Encoder to record passes onto.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The renderer providing pipelines and buffers.
    pub renderer: &'a Renderer3D,
    /// Mesh drawn this frame.
    pub mesh: &'a Mesh,
    /// Number of uploaded instances to draw.
    pub instance_count: u32,
}

/// A pass declared through its signature: `Reads`/`Writes` type-level sets
/// drive the plan wiring, `run` receives the resolved typed views.
///
/// `Send` (S5b): the erased runner may execute on rayon threads when the
/// plan records passes in parallel.
pub trait FramePass: Send + 'static {
    /// Type-level tuple of read accesses, e.g. `(Read<GBufferAlbedo>,)`.
    type Reads: AccessSet + for<'a> ViewsFor<'a>;
    /// Type-level tuple of write accesses.
    type Writes: AccessSet + for<'a> ViewsFor<'a>;
    /// Pass name in the layout (insertion order defines execution order).
    fn name(&self) -> &'static str;
    /// Execute the pass against the resolved views and the frame context.
    ///
    /// `where Self: Sized`: `SystemViews<'_, Self>` is a by-value
    /// parameter, and `Self` in a trait is not implicitly sized.
    fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>)
    where
        Self: Sized;
}

/// The typed views for one pass execution: one `&TextureView` per declared
/// access, in declaration order, plus the resolver for type-based fetch.
/// The typed views for one pass execution: one `&TextureView` per declared
/// access, in declaration order, plus the resolver for type-based fetch.
pub struct SystemViews<'a, P: FramePass> {
    /// Resolved views of [`FramePass::Reads`] in declaration order.
    pub reads: <<P as FramePass>::Reads as ViewsFor<'a>>::Views,
    /// Resolved views of [`FramePass::Writes`] in declaration order.
    pub writes: <<P as FramePass>::Writes as ViewsFor<'a>>::Views,
    resolver: Resolver<'a>,
}

impl<'a, P: FramePass> SystemViews<'a, P> {
    fn new(resolver: &Resolver<'a>) -> Self {
        Self {
            reads: <P::Reads as ViewsFor<'a>>::fetch(resolver),
            writes: <P::Writes as ViewsFor<'a>>::fetch(resolver),
            resolver: *resolver,
        }
    }

    /// The view of a declared resource, fetched by type — no positional
    /// coupling to the access tuple shape. One `get` (not read/write
    /// variants) on purpose: the same resource may be read-declared in one
    /// mode of a pass family and write-declared in another, while the
    /// shared body needs it either way (wgpu views do not distinguish).
    ///
    /// # Panics (debug)
    /// Panics in debug builds when `R` is outside both declared sets —
    /// the same guarantee the compiler enforces for positional tuples.
    pub fn get<R: FrameResource>(&self) -> &'a wgpu::TextureView {
        debug_assert!(
            declared::<P::Reads, R>() || declared::<P::Writes, R>(),
            "pass {} accesses resource '{}' outside its declared sets",
            std::any::type_name::<P>(),
            R::NAME
        );
        self.resolver.view::<R>()
    }
}

/// Whether access set `A` contains resource `R` (debug checks of
/// [`SystemViews::get`]; compile-time membership is blocked by coherence).
fn declared<A: AccessSet, R: FrameResource>() -> bool {
    let mut out = Vec::new();
    A::collect_accesses(&mut out);
    out.iter().any(|d| d.resource == TypeId::of::<R>())
}

/// Registered typed resources and systems: the `types → ResourceId` map
/// plus the type-erased system runners — and (d3) the single declaration
/// registry: resource/pass declarations, the declaration generation,
/// explicit ordering edges and the budget value.
#[derive(Default)]
pub struct SystemSet {
    ids: HashMap<TypeId, ResourceId>,
    /// PassId-keyed runners; a Mutex per system makes `run_pass(&self, …)`
    /// callable from several recording threads at once (S5b) — different
    /// passes lock different mutexes, so there is no contention.
    systems: Vec<(PassId, Mutex<SystemEntry>)>,
    /// Resource declarations in `ResourceId` order.
    resources: Vec<ResourceNode>,
    /// Pass declarations in `PassId` order (typed systems and imperative
    /// builder passes share one id space).
    passes: Vec<PassNode>,
    /// Original `&'static` pass names in `PassId` order — `PassNode::name`
    /// owns a `String` for layout dumps, but the E1 schedule projection
    /// (`schedule_bridge`) must hand `'static` names to
    /// `core::System::name`.
    pass_names: Vec<&'static str>,
    /// Surface size feeding `SizePolicy::MatchSurface`.
    surface_size: (u32, u32),
    /// Registry-owned transient allocator serving `layout()`/`build()`.
    /// The executor keeps a separate instance for the hot path; both
    /// memoize against [`SystemSet::generation`].
    pool: TransientPool,
    /// Monotonic declaration generation; bumped by every mutation
    /// (including [`SystemSet::invalidate`]). Both pool instances memoize
    /// their `Arc` snapshots against this instead of re-cloning per frame.
    generation: u64,
    /// S5c: explicit ordering edges (registration PassId i < j) on top
    /// of access-derived dependencies — for hidden dependencies (shared
    /// renderer queue buffers) invisible in the access sets.
    ordering: Vec<(PassId, PassId)>,
    /// S4 memory budget; unbounded by default.
    budget: Budget,
}

/// Type-erased system runner: resolves the typed views for the access
/// sets and executes the pass body. `Send` — parallel recording (S5b)
/// dispatches systems on rayon threads.
type RunFn = Box<dyn FnMut(&Resolver<'_>, &mut Frame<'_>) + Send>;

struct SystemEntry {
    #[allow(dead_code)] // printed in dispatch diagnostics
    name: &'static str,
    run: RunFn,
}

impl std::fmt::Debug for SystemSet {
    /// Manual `Debug` for [`SystemSet`]: the `TypeId` keys in `ids`/`ids_rev`
    /// and the `Box<dyn FnMut>` runners in `systems` do not implement
    /// `Debug`, so we surface the registry shape and skip the dispatch
    /// internals (the latter are intentionally opaque in tests).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemSet")
            .field("typed_resources", &self.ids.len())
            .field("systems", &self.systems.len())
            .field("resources", &self.resources)
            .field("passes", &self.passes)
            .field("surface_size", &self.surface_size)
            .field("pool", &self.pool)
            .field("generation", &self.generation)
            .field("ordering", &self.ordering)
            .field("budget", &self.budget)
            .finish()
    }
}

impl SystemSet {
    /// Create an empty registry.
    ///
    /// The surface size defaults to `(0, 0)`: call
    /// [`set_surface_size`](Self::set_surface_size) before compiling a
    /// layout that resolves `SizePolicy::MatchSurface` resources.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bump the declaration generation and drop the memoized layout (the
    /// pool recompiles on the next [`layout`](Self::layout)).
    fn touch(&mut self) {
        self.pool.invalidate();
        self.generation = self.generation.wrapping_add(1);
    }

    /// Current declaration generation (the pool memoization key).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Declares a frame-owned resource; returns its `ResourceId`.
    pub fn create_resource(&mut self, name: &'static str, spec: TextureSpec) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceNode {
            name: name.to_owned(),
            spec,
            imported: false,
            external: false,
        });
        self.touch();
        id
    }

    /// Declares an imported (read-only, externally backed) resource.
    pub fn import_resource(&mut self, name: &'static str, spec: TextureSpec) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceNode {
            name: name.to_owned(),
            spec,
            imported: true,
            external: false,
        });
        self.touch();
        id
    }

    /// Declares an externally backed output (e.g. the swapchain view);
    /// never pooled.
    pub fn external_output(&mut self, name: &'static str) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceNode {
            name: name.to_owned(),
            spec: TextureSpec::external(),
            imported: false,
            external: true,
        });
        self.touch();
        id
    }

    /// Resolves an imperative resource name to its `ResourceId`.
    ///
    /// # Panics
    /// Panics if no resource was declared under `name`.
    pub fn resolve_resource(&self, name: &'static str) -> ResourceId {
        self.resources
            .iter()
            .position(|r| r.name == name)
            .map(|i| ResourceId(i as u32))
            .unwrap_or_else(|| panic!("unknown resource '{name}'"))
    }

    /// Starts declaring an imperative pass (tests, tools,
    /// `scheduler_parity` oracle); production declares via types
    /// ([`add_system`](Self::add_system)).
    pub fn add_pass(&mut self, name: &'static str) -> PassBuilder<'_> {
        let id = PassId(self.passes.len() as u32);
        self.passes.push(PassNode {
            name: name.to_owned(),
            reads: Vec::new(),
            writes: Vec::new(),
            enabled: true,
        });
        self.pass_names.push(name);
        self.touch();
        PassBuilder { set: self, id }
    }

    /// Update the surface size (window resize).
    pub fn set_surface_size(&mut self, size: (u32, u32)) {
        self.surface_size = size;
        self.touch();
    }

    /// Declare that pass `a` must run before pass `b` (S5c; hidden
    /// dependencies invisible in the access sets).
    ///
    /// # Panics
    /// Panics if either endpoint is out of range.
    pub fn order_before(&mut self, before: PassId, after: PassId) {
        self.try_order_before(before, after)
            .unwrap_or_else(|error| panic!("order_before({before:?}, {after:?}): {error}"));
    }

    /// Fallible [`order_before`](Self::order_before): returns
    /// [`OrderError`] on error instead of panicking. Also validates both
    /// `PassId`s (an edge with an unknown id is an error, not a silent
    /// no-op).
    ///
    /// # Errors
    /// Returns [`OrderError`] if either endpoint is out of range.
    pub fn try_order_before(&mut self, before: PassId, after: PassId) -> Result<(), OrderError> {
        validate_indexed_edge(before.0 as usize, after.0 as usize, |i| {
            self.passes.get(i).map(|node| node.name.clone())
        })?;
        if !self.ordering.contains(&(before, after)) {
            self.ordering.push((before, after));
        }
        self.touch();
        Ok(())
    }

    /// S5c: name-based [`order_before`](Self::order_before) (pass name from
    /// `add_pass`).
    ///
    /// # Panics
    /// Panics on unknown name or reverse registration order.
    pub fn order_before_named(&mut self, before: &str, after: &str) {
        self.try_order_before_named(before, after)
            .unwrap_or_else(|error| panic!("order_before_named('{before}', '{after}'): {error}"));
    }

    /// Fallible [`order_before_named`](Self::order_before_named).
    ///
    /// # Errors
    /// Returns [`OrderError`] on unknown name or reverse registration order.
    pub fn try_order_before_named(&mut self, before: &str, after: &str) -> Result<(), OrderError> {
        let (b, a) = resolve_named_edge(before, after, |name| {
            self.passes.iter().position(|p| p.name == name)
        })?;
        self.try_order_before(PassId(b as u32), PassId(a as u32))
    }

    /// Enables/disables a pass (culling): a disabled pass is dropped from
    /// the layout, and its resources get no slots unless used elsewhere.
    ///
    /// # Panics
    /// Panics if the pass is unknown.
    pub fn set_pass_enabled(&mut self, id: PassId, enabled: bool) {
        let node = self
            .passes
            .get_mut(id.0 as usize)
            .unwrap_or_else(|| panic!("unknown pass {id:?}"));
        node.enabled = enabled;
        self.touch();
    }

    /// Set the transient-pool memory budget (S4).
    pub fn set_budget(&mut self, budget: Budget) {
        self.budget = budget;
        self.touch();
    }

    /// Current transient-pool memory budget.
    pub fn budget(&self) -> Budget {
        self.budget
    }

    /// Borrowed declaration snapshot for the pool compiler (see
    /// `TransientPool::ensure`). Field borrows only, so the caller can
    /// hold the input while mutably driving either pool instance.
    pub(crate) fn pool_input(&self) -> PoolInput<'_> {
        PoolInput {
            resources: &self.resources,
            passes: &self.passes,
            ordering: &self.ordering,
            surface_size: self.surface_size,
            budget: self.budget,
        }
    }

    /// Pass declarations in `PassId` order (E1 projection input).
    pub(crate) fn pass_nodes(&self) -> &[PassNode] {
        &self.passes
    }

    /// The original `&'static` name of a declared pass.
    pub(crate) fn pass_name(&self, id: PassId) -> &'static str {
        self.pass_names[id.0 as usize]
    }

    /// Reverse registry lookup: the `FrameResource` type backing a
    /// `ResourceId`, if the resource was registered with a type
    /// (`register_resource::<R>`). `None` for resources declared only
    /// through the imperative `create_resource`/`import_resource` API.
    pub(crate) fn resource_type(&self, id: ResourceId) -> Option<TypeId> {
        self.ids
            .iter()
            .find(|(_, &rid)| rid == id)
            .map(|(&tid, _)| tid)
    }

    /// Explicit ordering edges in `PassId` pairs (E1 projection input).
    pub(crate) fn ordering_edges(&self) -> &[(PassId, PassId)] {
        &self.ordering
    }

    /// Returns the frame layout (lifetimes + pool slots), recomputing it
    /// only when the declarations changed since the last call.
    ///
    /// # Panics
    /// Panics on first-touch or read-before-write invariant violations, and
    /// when the transient pool exceeds [`budget`](Self::budget) (see
    /// [`try_layout`](Self::try_layout) for the fallible path).
    pub fn layout(&mut self) -> &FrameLayout {
        self.try_layout()
            .unwrap_or_else(|e| panic!("SystemSet transient pool budget exceeded ({e})"))
    }

    /// Fallible [`layout`](Self::layout).
    ///
    /// # Errors
    /// Returns [`BudgetExceeded`] when the planned pool exceeds
    /// [`budget`](Self::budget); the pool stays unmodified.
    ///
    /// # Panics
    /// Panics on first-touch or read-before-write invariant violations
    /// (programmer errors, not fallible conditions).
    pub fn try_layout(&mut self) -> Result<&FrameLayout, BudgetExceeded> {
        // Field borrows (not `self.pool_input()`): `input` must not hold
        // the whole `&self` while `self.pool` is driven mutably.
        let generation = self.generation;
        let input = PoolInput {
            resources: &self.resources,
            passes: &self.passes,
            ordering: &self.ordering,
            surface_size: self.surface_size,
            budget: self.budget,
        };
        self.pool.ensure(generation, &input)?;
        Ok(self.pool.cached().expect("pool ensured a layout above"))
    }

    /// Compute the layout snapshot (parity oracle, debug tools). Shares
    /// the memoized computation with [`layout`](Self::layout).
    pub fn build(&mut self) -> FrameLayout {
        self.layout().clone()
    }

    /// Force recomputation on the next [`layout`](Self::layout), keeping
    /// the generation (debug invalidation only — every mutation already
    /// invalidates).
    pub fn invalidate(&mut self) {
        self.pool.invalidate();
        self.generation = self.generation.wrapping_add(1);
    }

    /// How many times the layout has been computed over this registry's
    /// lifetime.
    pub fn layout_computations(&self) -> u32 {
        self.pool.layout_computations()
    }

    /// Registers resource `R` and remembers its `ResourceId`.
    pub fn register_resource<R: FrameResource>(
        &mut self,
        surface_format: wgpu::TextureFormat,
    ) -> ResourceId {
        let id = match R::kind() {
            ResourceKind::FrameOwned => self.create_resource(R::NAME, R::spec(surface_format)),
            ResourceKind::Imported => self.import_resource(R::NAME, R::spec(surface_format)),
            ResourceKind::ExternalOutput => self.external_output(R::NAME),
        };
        self.ids.insert(TypeId::of::<R>(), id);
        id
    }

    /// The `ResourceId` of a registered resource.
    ///
    /// # Panics
    /// Panics if `R` was not registered.
    pub fn resource_id<R: FrameResource>(&self) -> ResourceId {
        *self
            .ids
            .get(&TypeId::of::<R>())
            .unwrap_or_else(|| panic!("typed resource '{}' is not registered", R::NAME))
    }

    /// Adds a pass, wiring reads/writes from `P::Reads`/`P::Writes`.
    ///
    /// # Panics
    /// Panics if a declared resource was not registered (with the resource
    /// name in the message), or on registry invariants (read-before-write).
    pub fn add_system<P: FramePass>(&mut self, pass: P) -> PassId {
        let mut reads = Vec::new();
        P::Reads::collect_accesses(&mut reads);
        let mut writes = Vec::new();
        P::Writes::collect_accesses(&mut writes);

        // Resolve `ResourceId`s against `self.ids` while we only hold an
        // immutable borrow; `add_pass` below takes `&mut self`, so borrowing
        // through the builder after that point would be a use-after-move.
        let read_ids: Vec<ResourceId> = reads
            .iter()
            .map(|d| Self::resolve_in(&self.ids, d))
            .collect();
        let write_ids: Vec<(ResourceId, Option<wgpu::Color>)> = writes
            .iter()
            .map(|d| (Self::resolve_in(&self.ids, d), d.clear))
            .collect();

        let mut builder = self.add_pass(pass.name());
        for id in read_ids {
            builder = builder.read(id);
        }
        for (id, clear) in write_ids {
            builder = match clear {
                Some(c) => builder.write_clear(id, c),
                None => builder.write(id),
            };
        }
        let pass_id = builder.id();

        let mut pass = pass;
        let name = pass.name();
        self.systems.push((
            pass_id,
            Mutex::new(SystemEntry {
                name,
                run: Box::new(move |resolver: &Resolver<'_>, frame: &mut Frame<'_>| {
                    let views = SystemViews::<P>::new(resolver);
                    pass.run(views, frame);
                }),
            }),
        ));
        pass_id
    }

    fn resolve_in(ids: &HashMap<TypeId, ResourceId>, d: &AccessDesc) -> ResourceId {
        *ids.get(&d.resource)
            .unwrap_or_else(|| panic!("system resource '{}' is not registered", d.name))
    }

    /// Runs the system registered for `pass_id`, if any. Returns `false`
    /// when the pass is not a typed system (imperative fallback).
    pub fn run_pass(&self, pass_id: PassId, views: &PassViews<'_>, frame: &mut Frame<'_>) -> bool {
        let ids = &self.ids;
        let Some((_, entry)) = self.systems.iter().find(|(id, _)| *id == pass_id) else {
            return false;
        };
        let mut entry = entry.lock().expect("system entry lock");
        let resolver = Resolver { views, ids };
        (entry.run)(&resolver, frame);
        true
    }
}

/// Incremental pass declaration: `.read()`/`.write()` calls accumulate
/// accesses (each bumps [`SystemSet::generation`] via `touch()`).
/// Returned by [`SystemSet::add_pass`]; for tests, tools and the
/// `scheduler_parity` oracle.
#[derive(Debug)]
pub struct PassBuilder<'a> {
    set: &'a mut SystemSet,
    id: PassId,
}

impl PassBuilder<'_> {
    /// Id of the pass being declared.
    pub fn id(&self) -> PassId {
        self.id
    }

    /// Declares a resource as read by the pass.
    ///
    /// # Panics
    /// Panics on an unknown resource or a read-before-write violation
    /// (detected at `layout()`/`build()`).
    pub fn read(self, id: ResourceId) -> Self {
        let index = self.id.0 as usize;
        if self.set.resources.get(id.0 as usize).is_none() {
            panic!(
                "unknown resource {id:?} in pass '{}'",
                self.set.passes[index].name
            );
        }
        self.set.passes[index].reads.push(id);
        self.set.touch();
        self
    }

    /// Declares a resource as written by the pass (no clear).
    pub fn write(self, id: ResourceId) -> Self {
        let index = self.id.0 as usize;
        if self.set.resources.get(id.0 as usize).is_none() {
            panic!(
                "unknown resource {id:?} in pass '{}'",
                self.set.passes[index].name
            );
        }
        self.set.passes[index].writes.push((id, None));
        self.set.touch();
        self
    }

    /// Declares a resource as written by the pass with a clear value
    /// (typically the frame background).
    pub fn write_clear(self, id: ResourceId, clear: wgpu::Color) -> Self {
        let index = self.id.0 as usize;
        if self.set.resources.get(id.0 as usize).is_none() {
            panic!(
                "unknown resource {id:?} in pass '{}'",
                self.set.passes[index].name
            );
        }
        self.set.passes[index].writes.push((id, Some(clear)));
        self.set.touch();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transient_pool::SizePolicy;

    struct ResA;
    impl FrameResource for ResA {
        const NAME: &'static str = "a";
        fn kind() -> ResourceKind {
            ResourceKind::FrameOwned
        }
        fn spec(_: wgpu::TextureFormat) -> TextureSpec {
            TextureSpec {
                format: wgpu::TextureFormat::Rgba8Unorm,
                samples: 1,
                size: SizePolicy::MatchSurface,
            }
        }
    }

    struct ResB;
    impl FrameResource for ResB {
        const NAME: &'static str = "b";
        fn kind() -> ResourceKind {
            ResourceKind::FrameOwned
        }
        fn spec(_: wgpu::TextureFormat) -> TextureSpec {
            TextureSpec {
                format: wgpu::TextureFormat::Rgba16Float,
                samples: 1,
                size: SizePolicy::MatchSurface,
            }
        }
    }

    struct ExtC;
    impl FrameResource for ExtC {
        const NAME: &'static str = "c";
        fn kind() -> ResourceKind {
            ResourceKind::ExternalOutput
        }
        fn spec(_: wgpu::TextureFormat) -> TextureSpec {
            TextureSpec {
                format: wgpu::TextureFormat::Rgba8Unorm,
                samples: 1,
                size: SizePolicy::MatchSurface,
            }
        }
    }

    fn collect<A: AccessSet>() -> Vec<(String, bool, Option<wgpu::Color>)> {
        let mut out = Vec::new();
        A::collect_accesses(&mut out);
        out.into_iter()
            .map(|d| (d.name.to_string(), d.write, d.clear))
            .collect()
    }

    #[test]
    fn access_set_collects_in_declaration_order() {
        let set = collect::<(Read<ResA>, Write<ResB>, Read<ResA>)>();
        assert_eq!(
            set,
            vec![
                ("a".into(), false, None),
                ("b".into(), true, None),
                ("a".into(), false, None),
            ]
        );
    }

    #[test]
    fn write_clear_carries_its_color() {
        let set = collect::<(WriteClear<ResB, ClearBlack>, Write<ResA>)>();
        assert_eq!(
            set,
            vec![
                ("b".into(), true, Some(wgpu::Color::BLACK)),
                ("a".into(), true, None),
            ]
        );
    }

    #[test]
    fn typed_wiring_matches_imperative_builder() {
        // Same plan wired imperatively and through a typed system: the
        // layouts (lifetimes, slots) must be identical.
        let fmt = wgpu::TextureFormat::Rgba8Unorm;

        let mut imperative = SystemSet::new();
        imperative.set_surface_size((320, 240));
        let a = imperative.create_resource("a", ResA::spec(fmt));
        let b = imperative.create_resource("b", ResB::spec(fmt));
        imperative.add_pass("p0").write(a);
        imperative.add_pass("p1").read(a).write(b);

        struct P0;
        impl FramePass for P0 {
            type Reads = ();
            type Writes = (Write<ResA>,);
            fn name(&self) -> &'static str {
                "p0"
            }
            fn run(&mut self, _views: SystemViews<'_, Self>, _frame: &mut Frame<'_>) {
                unreachable!("layout parity test does not execute systems");
            }
        }

        struct P1;
        impl FramePass for P1 {
            type Reads = (Read<ResA>,);
            type Writes = (Write<ResB>,);
            fn name(&self) -> &'static str {
                "p1"
            }
            fn run(&mut self, _views: SystemViews<'_, Self>, _frame: &mut Frame<'_>) {
                unreachable!("layout parity test does not execute systems");
            }
        }

        let mut systems = SystemSet::new();
        systems.set_surface_size((320, 240));
        systems.register_resource::<ResA>(fmt);
        systems.register_resource::<ResB>(fmt);
        systems.add_system(P0);
        systems.add_system(P1);

        assert_eq!(
            systems.build().debug_dump(),
            imperative.build().debug_dump(),
            "typed wiring must produce the same layout as the builder"
        );
    }

    #[test]
    fn external_output_kind_uses_external_wiring() {
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let mut systems = SystemSet::new();
        systems.set_surface_size((320, 240));
        let a = systems.register_resource::<ResA>(fmt);
        let c = systems.register_resource::<ExtC>(fmt);
        // Layout: external output is never pooled.
        systems.add_pass("p0").write(a);
        systems.add_pass("p1").read(a).write(c);
        let layout = systems.build();
        assert!(layout.resources[c.0 as usize].external);
        assert_eq!(layout.resources[c.0 as usize].slot, None);
        assert_eq!(layout.slots.len(), 1, "only 'a' gets a pool slot");
    }

    #[test]
    fn system_set_is_sync() {
        // S5b: parallel recording dispatches systems from rayon threads.
        fn assert_sync<T: Sync>() {}
        assert_sync::<SystemSet>();
    }

    #[test]
    fn declared_membership_matches_access_set() {
        assert!(declared::<(Read<ResA>,), ResA>());
        assert!(declared::<(Write<ResB>, Write<ResA>), ResA>());
        assert!(!declared::<(Read<ResA>,), ResB>());
        assert!(!declared::<(), ResA>());
    }

    #[test]
    #[should_panic(expected = "is not registered")]
    fn unregistered_resource_panics_with_name() {
        let mut systems = SystemSet::new();
        systems.set_surface_size((320, 240));
        systems.register_resource::<ResA>(wgpu::TextureFormat::Rgba8Unorm);

        struct P;
        impl FramePass for P {
            type Reads = (Read<ResB>,); // never registered
            type Writes = ();
            fn name(&self) -> &'static str {
                "p"
            }
            fn run(&mut self, _views: SystemViews<'_, Self>, _frame: &mut Frame<'_>) {
                unreachable!();
            }
        }
        systems.add_system(P);
    }

    // ── d4: builder-level invariants lifted from the dissolved FramePlan ─

    /// `add_pass().read(unknown)` panics with a clear message at the
    /// declaration site (no silent garbage in the layout tables).
    #[test]
    #[should_panic(expected = "unknown resource")]
    fn unknown_resource_panics() {
        let mut set = SystemSet::new();
        set.set_surface_size((320, 240));
        set.add_pass("p0").read(ResourceId(99));
    }

    /// Explicit ordering edges must respect registration order —
    /// a backward edge is a programmer error and panics.
    #[test]
    #[should_panic(expected = "registered")]
    fn explicit_ordering_rejects_backward() {
        let mut set = SystemSet::new();
        set.set_surface_size((64, 64));
        let a = set.create_resource("a", ResA::spec(wgpu::TextureFormat::Rgba8Unorm));
        let b = set.create_resource("b", ResB::spec(wgpu::TextureFormat::Rgba8Unorm));
        let first = set.add_pass("first").write(a).id();
        let second = set.add_pass("second").write(b).id();
        set.order_before(second, first);
    }

    /// Named ordering panics on an unknown target.
    #[test]
    #[should_panic(expected = "no node named")]
    fn explicit_ordering_unknown_name() {
        let mut set = SystemSet::new();
        set.set_surface_size((64, 64));
        let a = set.create_resource("a", ResA::spec(wgpu::TextureFormat::Rgba8Unorm));
        set.add_pass("real").write(a);
        set.order_before_named("real", "ghost");
    }

    /// `try_order_before` is the fallible variant: backward / unknown
    /// errors are returned, not panicked. After a successful forward
    /// edge the levels reflect the constraint.
    #[test]
    fn try_order_before_reports_errors_without_panicking() {
        use ornis_schedule::OrderError;

        let mut set = SystemSet::new();
        set.set_surface_size((64, 64));
        let a = set.create_resource("a", ResA::spec(wgpu::TextureFormat::Rgba8Unorm));
        let b = set.create_resource("b", ResB::spec(wgpu::TextureFormat::Rgba8Unorm));
        let first = set.add_pass("first").write(a).id();
        let second = set.add_pass("second").write(b).id();
        assert!(matches!(
            set.try_order_before(second, first),
            Err(OrderError::BackwardEdge { .. })
        ));
        assert_eq!(
            set.try_order_before_named("first", "ghost").map(|_| ()),
            Err(OrderError::UnknownNode {
                name: "ghost".to_owned(),
            })
        );
        // Ids outside the registry are an error, not a silent garbage edge.
        assert!(matches!(
            set.try_order_before(
                crate::transient_pool::PassId(99),
                crate::transient_pool::PassId(100)
            ),
            Err(OrderError::UnknownNode { .. })
        ));
        assert_eq!(set.build().levels(), vec![vec![0, 1]]);
        assert!(set.try_order_before(first, second).is_ok());
        assert_eq!(set.build().levels(), vec![vec![0], vec![1]]);
    }

    // ── d4: debug-only access enforcement (was on FramePlan) ──────────

    /// Backlog #6 (audit §4.1, Phase B exit criterion "sneaky pass"):
    /// a pass requesting a resource outside its declared reads/writes
    /// panics in debug builds with pass and resource names — mirrors
    /// the `sneaky` system test in `core::Schedule`.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "'sneaky' (index 1) accesses resource 'b'")]
    fn sneaky_pass_undeclared_access_panics() {
        use crate::transient_pool::assert_pass_access_declared;

        let mut set = SystemSet::new();
        set.set_surface_size((320, 240));
        let a = set.create_resource("a", ResA::spec(wgpu::TextureFormat::Rgba8Unorm));
        let b = set.create_resource("b", ResB::spec(wgpu::TextureFormat::Rgba8Unorm));
        set.add_pass("writer").write(a).write(b);
        set.add_pass("sneaky").read(a);
        let layout = set.build();
        // Pass 1 declared only read(a); peeking at `b` is out of set.
        assert_pass_access_declared(&layout, 1, b);
    }

    /// Honest pass: read declaration covers the view; write declaration
    /// also covers reading its own write (own-write read), as in
    /// `core::Schedule::declared_access_passes_enforcement` — both
    /// checks stay silent.
    #[test]
    #[cfg(debug_assertions)]
    fn declared_pass_access_passes_enforcement() {
        use crate::transient_pool::assert_pass_access_declared;

        let mut set = SystemSet::new();
        set.set_surface_size((320, 240));
        let x = set.create_resource("x", ResA::spec(wgpu::TextureFormat::Rgba8Unorm));
        set.add_pass("writer").write(x);
        set.add_pass("reader").read(x);
        let layout = set.build();
        assert_pass_access_declared(&layout, 0, x);
        assert_pass_access_declared(&layout, 1, x);
    }
}
