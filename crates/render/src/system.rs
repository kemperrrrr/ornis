//! Typed graph systems — S2 (PLAN.md, Приложение C; IDEAS §28.1).
//!
//! Пасс объявляет доступы к ресурсам **в типах** через ZST-маркеры
//! (`Read<R>` / `Write<R>` / `WriteClear<R, C>` в кортежах), а планировщик
//! выводит из них проводку графа (reads/writes → lifetime → пул). Ресурс —
//! это тип, реализующий [`GraphResource`]; соответствие «тип → ResourceId»
//! держит [`SystemSet`]. Никаких строк и syn-разбора: идентичность ресурса —
//! это тип (урок хрупкости `smart_pipeline`, см. анти-цели Приложения C).
//!
//! Границы S2: множества доступа статичны. Пассы, чьи доступы зависят от
//! конфигурации (владение depth у forward, выбор входа блума, смешивание в
//! composite), остаются на императивном пути в `graph_frame.rs` до решения
//! S2b (вариантные типы против регистрации-как-выбора).
//!
//! Порядок пассов остаётся порядком регистрации (insertion order);
//! `.before()`/`.after()` и автопараллелизм — S5.

use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Mutex;

use crate::graph_frame::PassViews;
use crate::mesh::Mesh;
use crate::render_graph::{PassId, RenderGraph, ResourceId, TextureSpec};
use crate::renderer::Renderer3D;

/// How a resource enters the graph (see `RenderGraph::{create_resource,
/// import_resource, external_output}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    /// Created and owned by the graph (transient, pooled).
    GraphOwned,
    /// Imported external input, read-only (e.g. an uploaded shadow map).
    Imported,
    /// Externally backed output (e.g. the swapchain view); never pooled.
    ExternalOutput,
}

/// A typed graph resource: a ZST marker type carrying the resource's
/// identity. One type = one resource per graph.
pub trait GraphResource: 'static {
    /// Unique debug name (layout dumps, panics). Must be unique per graph.
    const NAME: &'static str;
    /// How the resource is registered in [`RenderGraph`].
    fn kind() -> ResourceKind;
    /// Texture spec; `surface_format` feeds resources that mirror the
    /// surface format (e.g. the HDR layer).
    fn spec(surface_format: wgpu::TextureFormat) -> TextureSpec;
}

/// A clear value attached to a [`WriteClear`] access.
pub trait ClearValue {
    const COLOR: wgpu::Color;
}

/// Clear to opaque black (HDR layers, bloom chain).
pub struct ClearBlack;
impl ClearValue for ClearBlack {
    const COLOR: wgpu::Color = wgpu::Color::BLACK;
}

/// Clear to opaque white (depth when the pass owns it).
pub struct ClearWhite;
impl ClearValue for ClearWhite {
    const COLOR: wgpu::Color = wgpu::Color::WHITE;
}

/// Clear to fully transparent (forward HDR layer).
pub struct ClearTransparent;
impl ClearValue for ClearTransparent {
    const COLOR: wgpu::Color = wgpu::Color::TRANSPARENT;
}

/// One declared access: which resource, read or write, optional clear.
pub trait Access {
    type Resource: GraphResource;
    const IS_WRITE: bool;
    fn clear() -> Option<wgpu::Color>;
}

/// Read access marker (ZST).
pub struct Read<R>(PhantomData<fn() -> R>);

/// Write access marker, no clear (ZST).
pub struct Write<R>(PhantomData<fn() -> R>);

/// Write access marker with a clear value (ZST).
pub struct WriteClear<R, C: ClearValue>(PhantomData<fn() -> (R, C)>);

impl<R: GraphResource> Access for Read<R> {
    type Resource = R;
    const IS_WRITE: bool = false;
    fn clear() -> Option<wgpu::Color> {
        None
    }
}

impl<R: GraphResource> Access for Write<R> {
    type Resource = R;
    const IS_WRITE: bool = true;
    fn clear() -> Option<wgpu::Color> {
        None
    }
}

impl<R: GraphResource, C: ClearValue> Access for WriteClear<R, C> {
    type Resource = R;
    const IS_WRITE: bool = true;
    fn clear() -> Option<wgpu::Color> {
        Some(C::COLOR)
    }
}

/// Runtime projection of one access (used to wire the pass into the graph).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccessDesc {
    pub resource: TypeId,
    pub name: &'static str,
    pub write: bool,
    pub clear: Option<wgpu::Color>,
}

/// A type-level set of accesses: tuples of access markers, e.g.
/// `(Read<Albedo>, Write<Hdr>)`. Collect order = declaration order.
/// Arity is supported up to 6 (one `impl_access_tuple!` line below adds
/// more if a pass ever needs it).
pub trait AccessSet {
    fn collect_accesses(out: &mut Vec<AccessDesc>);
}

impl AccessSet for () {
    fn collect_accesses(_out: &mut Vec<AccessDesc>) {}
}

/// Every access resolves to a texture view of the same lifetime; the helper
/// trait exists so tuple `Views` types can be built per-element in a macro.
pub trait AccessView<'a> {
    type View;
}

impl<'a, A: Access> AccessView<'a> for A {
    type View = &'a wgpu::TextureView;
}

/// Resolves the views for an access set at execution time.
pub trait ViewsFor<'a>: AccessSet {
    /// A tuple of `&wgpu::TextureView` matching the access set's arity.
    type Views;
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
                        name: <$name::Resource as GraphResource>::NAME,
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
    /// the latter a wiring bug, not a runtime state).
    pub fn view<R: GraphResource>(&self) -> &'a wgpu::TextureView {
        let id = self
            .ids
            .get(&TypeId::of::<R>())
            .unwrap_or_else(|| panic!("typed resource '{}' is not registered", R::NAME));
        self.views.view_of(*id)
    }
}

/// Per-frame execution context handed to every system: the render context
/// parts (`device`/`queue`/`encoder`) plus the frame's draw inputs.
pub struct Frame<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub renderer: &'a Renderer3D,
    pub mesh: &'a Mesh,
    pub instance_count: u32,
}

/// A pass declared through its signature: `Reads`/`Writes` type-level sets
/// drive the graph wiring, `run` receives the resolved typed views.
///
/// `Send` (S5b): the erased runner may execute on rayon threads when the
/// graph records passes in parallel.
pub trait GraphPass: Send + 'static {
    type Reads: AccessSet + for<'a> ViewsFor<'a>;
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
pub struct SystemViews<'a, P: GraphPass> {
    pub reads: <<P as GraphPass>::Reads as ViewsFor<'a>>::Views,
    pub writes: <<P as GraphPass>::Writes as ViewsFor<'a>>::Views,
    resolver: Resolver<'a>,
}

impl<'a, P: GraphPass> SystemViews<'a, P> {
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
    pub fn get<R: GraphResource>(&self) -> &'a wgpu::TextureView {
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
fn declared<A: AccessSet, R: GraphResource>() -> bool {
    let mut out = Vec::new();
    A::collect_accesses(&mut out);
    out.iter().any(|d| d.resource == TypeId::of::<R>())
}

/// Registered typed resources and systems: the `types → ResourceId` map
/// plus the type-erased system runners, parallel to the graph's passes.
#[derive(Default)]
pub struct SystemSet {
    ids: HashMap<TypeId, ResourceId>,
    /// PassId-keyed runners; a Mutex per system makes `run_pass(&self, …)`
    /// callable from several recording threads at once (S5b) — different
    /// passes lock different mutexes, so there is no contention.
    systems: Vec<(PassId, Mutex<SystemEntry>)>,
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

impl SystemSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers resource `R` in the graph and remembers its `ResourceId`.
    pub fn register_resource<R: GraphResource>(
        &mut self,
        graph: &mut RenderGraph,
        surface_format: wgpu::TextureFormat,
    ) -> ResourceId {
        let id = match R::kind() {
            ResourceKind::GraphOwned => graph.create_resource(R::NAME, R::spec(surface_format)),
            ResourceKind::Imported => graph.import_resource(R::NAME, R::spec(surface_format)),
            ResourceKind::ExternalOutput => graph.external_output(R::NAME),
        };
        self.ids.insert(TypeId::of::<R>(), id);
        id
    }

    /// The `ResourceId` of a registered resource.
    ///
    /// # Panics
    /// Panics if `R` was not registered.
    pub fn resource_id<R: GraphResource>(&self) -> ResourceId {
        *self
            .ids
            .get(&TypeId::of::<R>())
            .unwrap_or_else(|| panic!("typed resource '{}' is not registered", R::NAME))
    }

    /// Adds a pass to `graph`, wiring reads/writes from `P::Reads`/`P::Writes`.
    ///
    /// # Panics
    /// Panics if a declared resource was not registered (with the resource
    /// name in the message), or on graph invariants (read-before-write).
    pub fn add_system<P: GraphPass>(&mut self, graph: &mut RenderGraph, pass: P) -> PassId {
        let mut reads = Vec::new();
        P::Reads::collect_accesses(&mut reads);
        let mut writes = Vec::new();
        P::Writes::collect_accesses(&mut writes);

        let mut builder = graph.add_pass(pass.name());
        for d in &reads {
            builder = builder.read(self.resolve(d));
        }
        for d in &writes {
            let id = self.resolve(d);
            builder = match d.clear {
                Some(clear) => builder.write_clear(id, clear),
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

    fn resolve(&self, d: &AccessDesc) -> ResourceId {
        *self
            .ids
            .get(&d.resource)
            .unwrap_or_else(|| panic!("system resource '{}' is not registered", d.name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::SizePolicy;

    struct ResA;
    impl GraphResource for ResA {
        const NAME: &'static str = "a";
        fn kind() -> ResourceKind {
            ResourceKind::GraphOwned
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
    impl GraphResource for ResB {
        const NAME: &'static str = "b";
        fn kind() -> ResourceKind {
            ResourceKind::GraphOwned
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
    impl GraphResource for ExtC {
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
        // Same graph wired imperatively and through a typed system: the
        // layouts (lifetimes, slots) must be identical.
        let fmt = wgpu::TextureFormat::Rgba8Unorm;

        let mut imperative = RenderGraph::new((320, 240));
        let a = imperative.create_resource("a", ResA::spec(fmt));
        let b = imperative.create_resource("b", ResB::spec(fmt));
        imperative.add_pass("p0").write(a);
        imperative.add_pass("p1").read(a).write(b);

        struct P0;
        impl GraphPass for P0 {
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
        impl GraphPass for P1 {
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
        let mut typed = RenderGraph::new((320, 240));
        systems.register_resource::<ResA>(&mut typed, fmt);
        systems.register_resource::<ResB>(&mut typed, fmt);
        systems.add_system(&mut typed, P0);
        systems.add_system(&mut typed, P1);

        assert_eq!(
            typed.build().debug_dump(),
            imperative.build().debug_dump(),
            "typed wiring must produce the same layout as the builder"
        );
    }

    #[test]
    fn external_output_kind_uses_external_wiring() {
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let mut systems = SystemSet::new();
        let mut graph = RenderGraph::new((320, 240));
        let a = systems.register_resource::<ResA>(&mut graph, fmt);
        let c = systems.register_resource::<ExtC>(&mut graph, fmt);
        // Layout: external output is never pooled.
        graph.add_pass("p0").write(a);
        graph.add_pass("p1").read(a).write(c);
        let layout = graph.build();
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
        let mut graph = RenderGraph::new((320, 240));
        systems.register_resource::<ResA>(&mut graph, wgpu::TextureFormat::Rgba8Unorm);

        struct P;
        impl GraphPass for P {
            type Reads = (Read<ResB>,); // never registered
            type Writes = ();
            fn name(&self) -> &'static str {
                "p"
            }
            fn run(&mut self, _views: SystemViews<'_, Self>, _frame: &mut Frame<'_>) {
                unreachable!();
            }
        }
        systems.add_system(&mut graph, P);
    }
}
