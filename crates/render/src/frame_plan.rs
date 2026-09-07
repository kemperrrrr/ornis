//! Frame plan — pass declaration registry (formerly "render graph", Phase 0).
//!
//! An immediate-mode registry in the spirit of Frostbite FrameGraph and
//! Ponies&Light: passes are declared in execution order, and each pass
//! declares which resources it reads and writes. The registry owns
//! declarations plus the declaration generation; compiling them into an
//! executable layout (lifetimes, pool slots, budget) is the
//! [`TransientPool`](crate::transient_pool::TransientPool)'s job, and the
//! computed [`FrameLayout`](crate::transient_pool::FrameLayout) lives
//! there — this is the d2 split of the `FramePlan` dissolution.
//!
//! Model:
//! - declaring resources/passes (plus `generation`, explicit `ordering`
//!   edges and the [`Budget`](crate::transient_pool::Budget) value) stays
//!   here; `FramePlan::layout()` is the cold-path accessor over the
//!   registry-owned pool instance (tests, tools, dumps), while the frame
//!   hot path uses the executor-owned pool via
//!   `FrameExecutor::ensure_layout`;
//! - creating real `wgpu::Texture` objects per slot is the executor's job
//!   (Phase 1). On wgpu, barriers and layout transitions are handled by
//!   wgpu itself, so the plan owns declarations, not synchronization.
//!
//! Invariants (panic with a clear message when violated):
//! - a resource must not be read before it is written (imported resources
//!   are exempt);
//! - unknown resource/pass ids are errors;
//! - within a single pass, no two live resources may share a pool slot
//!   (guaranteed by construction).

use ornis_schedule::{OrderError, resolve_named_edge, validate_indexed_edge};

use crate::transient_pool::{
    Budget, BudgetExceeded, FrameLayout, PassId, PassNode, PoolInput, ResourceId, ResourceNode,
    SizePolicy, TextureSpec, TransientPool,
};

/// The pass plan being assembled.
///
/// Declarations (resources/passes) live here; compiling them into an
/// executable layout is the registry-owned [`TransientPool`]'s job (cold
/// paths: `layout()`/`build()`), while the frame hot path compiles
/// through the executor-owned pool (`FrameExecutor::ensure_layout`).
/// Every mutation bumps [`FramePlan::generation`], which both pools use
/// to invalidate their shared snapshots.
#[derive(Debug)]
pub struct FramePlan {
    resources: Vec<ResourceNode>,
    passes: Vec<PassNode>,
    surface_size: (u32, u32),
    /// Registry-owned transient allocator serving `layout()`/`build()`.
    /// The executor keeps a separate instance for the hot path; both
    /// memoize against [`FramePlan::generation`].
    pool: TransientPool,
    /// Monotonic declaration generation; bumped by every mutation (including
    /// [`FramePlan::invalidate`]). The executor memoizes its `Arc` snapshot
    /// against this instead of re-cloning per frame.
    generation: u64,
    /// S5c: explicit ordering edges (registration PassId i < j) on top
    /// of access-derived dependencies — for hidden dependencies (shared
    /// renderer queue buffers) invisible in the access sets.
    ordering: Vec<(PassId, PassId)>,
    /// S4 memory budget; unbounded by default.
    budget: Budget,
}

/// Executes the plan: for each pass in layout order, `run` is invoked
/// with a [`PassViews`] resolver over the compiled layout (see
/// `FrameExecutor`).
impl FramePlan {
    /// Creates an empty plan; `surface_size` feeds `SizePolicy::MatchSurface`.
    pub fn new(surface_size: (u32, u32)) -> Self {
        Self {
            resources: Vec::new(),
            passes: Vec::new(),
            surface_size,
            pool: TransientPool::new(),
            generation: 0,
            ordering: Vec::new(),
            budget: Budget::unbounded(),
        }
    }

    /// Marks the plan dirty: drops the registry-owned pool snapshot and
    /// bumps the declaration generation so executor-held `Arc` snapshots
    /// refresh.
    fn touch(&mut self) {
        self.pool.invalidate();
        self.generation = self.generation.wrapping_add(1);
    }

    /// Declaration generation for executor snapshot memoization (see
    /// `FrameExecutor::ensure_layout`).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Updates the surface size (window resize) before the next `build()`.
    pub fn set_surface_size(&mut self, width: u32, height: u32) {
        self.surface_size = (width, height);
        self.touch();
    }

    /// Sets the S4 memory budget; invalidates the cached layout.
    pub fn set_budget(&mut self, budget: Budget) {
        self.budget = budget;
        self.touch();
    }

    /// The configured budget.
    pub fn budget(&self) -> Budget {
        self.budget
    }

    /// Registers a plan-owned resource (texture).
    pub fn create_resource(&mut self, name: impl Into<String>, spec: TextureSpec) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceNode {
            name: name.into(),
            spec,
            imported: false,
            external: false,
        });
        self.touch();
        id
    }

    /// Registers an imported (external) resource that passes only read
    /// (e.g. an uploaded shadow map). The "first touch must be a write"
    /// rule does not apply to it.
    pub fn import_resource(&mut self, name: impl Into<String>, spec: TextureSpec) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceNode {
            name: name.into(),
            spec,
            imported: true,
            external: false,
        });
        self.touch();
        id
    }

    /// Registers an externally backed output (e.g. the swapchain image):
    /// passes may write it, but the plan never pools it — the executor
    /// must provide the view via `FrameExecutor::set_external_view`.
    pub fn external_output(&mut self, name: impl Into<String>) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceNode {
            name: name.into(),
            spec: TextureSpec {
                format: wgpu::TextureFormat::Rgba8Unorm,
                samples: 1,
                size: SizePolicy::MatchSurface,
            },
            imported: true,
            external: true,
        });
        self.touch();
        id
    }

    /// Starts declaring a pass; passes execute in insertion order.
    ///
    /// Compatibility shim (S3): production passes are declared as typed
    /// systems — `impl FramePass` + `SystemSet::add_system` — and the
    /// builder remains for tests, tools and the migration period.
    pub fn add_pass(&mut self, name: impl Into<String>) -> PassBuilder<'_> {
        let id = PassId(self.passes.len() as u32);
        self.passes.push(PassNode {
            name: name.into(),
            reads: Vec::new(),
            writes: Vec::new(),
            enabled: true,
        });
        self.touch();
        PassBuilder { plan: self, id }
    }

    /// S5c: declares that pass `before` must execute before pass
    /// `after`, even when their accesses do not conflict (hidden
    /// dependency, e.g. a shared queue-written uniform buffer). Affects
    /// only parallel level partitioning ([`FrameLayout::levels`]);
    /// execution order is registration order.
    ///
    /// # Panics
    /// Panics if `after` was registered before `before` (execution
    /// order is immutable) or if a pass is unknown.
    pub fn order_before(&mut self, before: PassId, after: PassId) {
        self.try_order_before(before, after)
            .unwrap_or_else(|error| panic!("order_before({before:?}, {after:?}): {error}"));
    }

    /// Fallible [`FramePlan::order_before`]: returns [`OrderError`] on
    /// error instead of panicking. Also validates both `PassId`s
    /// (previously an edge with an unknown id was added silently and
    /// ignored during level computation).
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

    /// S5c: name-based [`FramePlan::order_before`] (pass name from
    /// `add_pass`).
    ///
    /// # Panics
    /// Panics on unknown name or reverse registration order.
    pub fn order_before_named(&mut self, before: &str, after: &str) {
        self.try_order_before_named(before, after)
            .unwrap_or_else(|error| panic!("order_before_named('{before}', '{after}'): {error}"));
    }

    /// Fallible [`FramePlan::order_before_named`].
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

    fn resolve_resource(&self, id: ResourceId, pass_name: &str) -> &ResourceNode {
        self.resources
            .get(id.0 as usize)
            .unwrap_or_else(|| panic!("unknown resource {id:?} in pass '{pass_name}'"))
    }

    /// Borrowed declaration snapshot for the pool compiler (see
    /// [`TransientPool::ensure`]). Field borrows only, so the caller can
    /// hold the input while mutably driving either pool instance.
    ///
    /// Parallel to [`SystemSet::pool_input`](crate::system::SystemSet::pool_input)
    /// by design: `FramePlan` and `SystemSet` are the two registry fronts
    /// (imperative parity vs. typed S2), and both feed `TransientPool`.
    /// Currently unused because the executor's hot path goes through
    /// `SystemSet`; left in place so a future cold-path caller
    /// (parity-frontend tool, dump) can drive a `TransientPool` directly
    /// without round-tripping through `SystemSet`. Keep the docstring
    /// in sync if a real consumer appears — or remove together with
    /// `FramePlan` if the d3 consolidation is finalized.
    #[allow(dead_code)]
    pub(crate) fn pool_input(&self) -> PoolInput<'_> {
        PoolInput {
            resources: &self.resources,
            passes: &self.passes,
            ordering: &self.ordering,
            surface_size: self.surface_size,
            budget: self.budget,
        }
    }

    /// Returns the frame layout (lifetimes + pool slots), compiling it
    /// through the registry-owned pool only when the declarations changed
    /// since the last call. This is the cold-path accessor (tests, tools,
    /// dumps); `RenderFrame3D::render` compiles through the
    /// executor-owned pool instead.
    ///
    /// # Panics
    /// Panics if invariants are violated (read-before-write, etc.) — the
    /// panic fires on the first compilation after the offending mutation,
    /// not at the mutation site.
    pub fn layout(&mut self) -> &FrameLayout {
        self.try_layout()
            .unwrap_or_else(|e| panic!("frame plan budget exceeded: {e}"))
    }

    /// Like [`FramePlan::layout`], but a budget violation is a returned
    /// error instead of a panic (editors/tools; S4).
    ///
    /// # Errors
    /// Returns [`BudgetExceeded`] when the pool does not fit the
    /// configured [`Budget`]; nothing is cached in that case.
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
        // Filled by the successful `ensure` above (or by an earlier call
        // at the same generation).
        Ok(self
            .pool
            .cached()
            .expect("pool cache is filled by successful ensure"))
    }

    /// Snapshot of the layout as an owned value. Equivalent to cloning
    /// [`FramePlan::layout`]; prefer `layout()` on cold paths — this
    /// clones the pass/resource/slot vectors.
    ///
    /// # Panics
    /// Same as [`FramePlan::layout`].
    pub fn build(&mut self) -> FrameLayout {
        self.layout().clone()
    }

    /// Forces the next [`FramePlan::layout`] to recompile. Mutating
    /// methods do this automatically; this is for benchmarks and tests
    /// that drive recompilation explicitly.
    pub fn invalidate(&mut self) {
        self.touch();
    }

    /// How many times the layout has been compiled over this plan's
    /// lifetime (S1 cache diagnostics: stays flat while the cache holds).
    /// Counts registry-owned pool compilations only — the executor-owned
    /// pool tracks its own (see `FrameExecutor::ensure_layout`).
    pub fn layout_computations(&self) -> u32 {
        self.pool.layout_computations()
    }
}

/// Builder for declaring a pass.
///
/// Test/parity funnel (S3, dissolution stage 3): production passes declare
/// accesses as types (`impl FramePass` + `SystemSet::add_system`, the sole
/// prod wiring path); the builder stays for unit/integration tests and the
/// `scheduler_parity` oracle.
#[derive(Debug)]
pub struct PassBuilder<'a> {
    plan: &'a mut FramePlan,
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
        self.plan
            .resolve_resource(id, &self.plan.passes[self.id.0 as usize].name);
        self.plan.passes[self.id.0 as usize].reads.push(id);
        self.plan.touch();
        self
    }

    /// Declares a resource as written by the pass (no clear).
    pub fn write(self, id: ResourceId) -> Self {
        self.plan
            .resolve_resource(id, &self.plan.passes[self.id.0 as usize].name);
        self.plan.passes[self.id.0 as usize].writes.push((id, None));
        self.plan.touch();
        self
    }

    /// Declares a resource as written by the pass with a clear value
    /// (typically the frame background).
    pub fn write_clear(self, id: ResourceId, clear: wgpu::Color) -> Self {
        self.plan
            .resolve_resource(id, &self.plan.passes[self.id.0 as usize].name);
        self.plan.passes[self.id.0 as usize]
            .writes
            .push((id, Some(clear)));
        self.plan.touch();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transient_pool::assert_pass_access_declared;

    fn spec(format: wgpu::TextureFormat, samples: u32) -> TextureSpec {
        TextureSpec {
            format,
            samples,
            size: SizePolicy::MatchSurface,
        }
    }

    #[test]
    fn lifetime_window_basic() {
        let mut g = FramePlan::new((1920, 1080));
        let albedo = g.create_resource("albedo", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let hdr = g.create_resource("hdr", spec(wgpu::TextureFormat::Rgba16Float, 1));
        let depth = g.create_resource("depth", spec(wgpu::TextureFormat::Depth32Float, 1));

        g.add_pass("gbuffer").write(albedo).write(depth);
        g.add_pass("lighting").read(albedo).read(depth).write(hdr);

        let layout = g.build();
        assert_eq!(layout.passes.len(), 2);
        let a = &layout.resources[albedo.0 as usize];
        assert_eq!(
            (a.first_use, a.last_use),
            (0, 1),
            "albedo: gbuffer → lighting"
        );
        let h = &layout.resources[hdr.0 as usize];
        assert_eq!(
            (h.first_use, h.last_use),
            (1, 1),
            "hdr lives only on lighting"
        );
        let d = &layout.resources[depth.0 as usize];
        assert_eq!((d.first_use, d.last_use), (0, 1));
        // Different formats → different slots.
        assert_ne!(a.slot, h.slot);
        assert_eq!(layout.slots.len(), 3);
    }

    #[test]
    fn transient_slot_reuse_same_spec() {
        // a lives [0,1], b lives [2,3], same spec → one slot (aliasing).
        let mut g = FramePlan::new((320, 240));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));

        g.add_pass("p0").write(a);
        g.add_pass("p1").read(a);
        g.add_pass("p2").write(b);
        g.add_pass("p3").read(b);

        let layout = g.build();
        assert_eq!(
            layout.slots.len(),
            1,
            "non-overlapping windows share a slot"
        );
        assert_eq!(layout.slots[0].resources, vec![a, b]);
        assert_eq!(layout.resources[a.0 as usize].slot, Some(0));
        assert_eq!(layout.resources[b.0 as usize].slot, Some(0));
    }

    #[test]
    fn overlapping_resources_need_distinct_slots() {
        // a [0,1], b [1,2] — overlap on pass 1 → two slots.
        let mut g = FramePlan::new((320, 240));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));

        g.add_pass("p0").write(a);
        g.add_pass("p1").read(a).write(b);
        g.add_pass("p2").read(b);

        let layout = g.build();
        assert_eq!(layout.slots.len(), 2);
        assert_ne!(
            layout.resources[a.0 as usize].slot,
            layout.resources[b.0 as usize].slot
        );
    }

    #[test]
    #[should_panic(expected = "before any write")]
    fn read_before_write_panics() {
        let mut g = FramePlan::new((320, 240));
        let x = g.create_resource("x", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        g.add_pass("p0").read(x);
        g.add_pass("p1").write(x);
        g.build();
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "'sneaky' (index 1) accesses resource 'b'")]
    fn sneaky_pass_undeclared_access_panics() {
        // Backlog #6 (audit §4.1, Phase B exit criterion "sneaky pass"):
        // pass requests a resource outside its declared reads/writes →
        // debug panic with pass and resource names (mirrors the
        // `sneaky` system in `core::Schedule`).
        let mut g = FramePlan::new((320, 240));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        g.add_pass("writer").write(a).write(b);
        g.add_pass("sneaky").read(a);
        let layout = g.build();
        // Pass 1 declared only read(a); peeking at `b` is out of set.
        assert_pass_access_declared(&layout, 1, b);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn declared_pass_access_passes_enforcement() {
        // Honest pass: read declaration covers the view; write declaration
        // also covers reading its own write (own-write read), as in
        // core `declared_access_passes_enforcement` — both checks stay silent.
        let mut g = FramePlan::new((320, 240));
        let x = g.create_resource("x", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        g.add_pass("writer").write(x);
        g.add_pass("reader").read(x);
        let layout = g.build();
        assert_pass_access_declared(&layout, 0, x);
        assert_pass_access_declared(&layout, 1, x);
    }

    #[test]
    #[should_panic(expected = "unknown resource")]
    fn unknown_resource_panics() {
        let mut g = FramePlan::new((320, 240));
        g.add_pass("p0").read(ResourceId(99));
    }

    #[test]
    fn imported_resource_may_be_read_first() {
        let mut g = FramePlan::new((320, 240));
        let shadow = g.import_resource("shadow", spec(wgpu::TextureFormat::R32Float, 1));
        g.add_pass("p0").read(shadow);
        g.add_pass("p1").read(shadow);
        let layout = g.build(); // does not panic
        let rl = &layout.resources[shadow.0 as usize];
        assert_eq!((rl.first_use, rl.last_use), (0, 1));
        assert_eq!(rl.slot, Some(0));
    }

    #[test]
    fn disabled_pass_culls_its_resources() {
        let mut g = FramePlan::new((320, 240));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let p1 = g.add_pass("p1").write(a).id();
        g.add_pass("p2").write(b);

        g.set_pass_enabled(p1, false);
        let layout = g.build();
        assert_eq!(layout.passes.len(), 1);
        assert_eq!(layout.passes[0].name, "p2");
        let ra = &layout.resources[a.0 as usize];
        assert_eq!(
            ra.first_use,
            usize::MAX,
            "a is not used by any enabled pass"
        );
        assert_eq!(ra.slot, None);
        assert_eq!(layout.slots.len(), 1, "only b gets a slot");
    }

    #[test]
    fn execute_delivers_live_resources_and_slots() {
        let mut g = FramePlan::new((640, 480));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba16Float, 1));
        g.add_pass("gbuffer").write(a);
        g.add_pass("lighting").read(a).write(b);
        g.add_pass("composite").read(b);

        let layout = g.build();
        // Per-pass walk over the layout tables (insertion order; disabled
        // passes are already dropped from `passes`).
        let visits: Vec<(usize, Vec<ResourceId>, Option<usize>)> = layout
            .passes
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let slot_a = layout.resources[a.0 as usize]
                    .alive_at(index)
                    .then(|| layout.resources[a.0 as usize].slot)
                    .flatten();
                (index, layout.pass_alive[index].clone(), slot_a)
            })
            .collect();

        assert_eq!(visits[0], (0, vec![a], Some(0)), "gbuffer: a alive");
        assert_eq!(
            visits[1],
            (1, vec![a, b], Some(0)),
            "lighting: a and b alive"
        );
        assert_eq!(visits[2], (2, vec![b], None), "composite: a is dead");
        assert_eq!(
            ctx_pass_names(&layout),
            vec!["gbuffer", "lighting", "composite"]
        );
    }

    fn ctx_pass_names(layout: &FrameLayout) -> Vec<String> {
        layout.passes.iter().map(|p| p.name.clone()).collect()
    }

    #[test]
    fn independent_branches_share_levels() {
        // p0→p1 (a→b) and p2→p3 (c→d) share no resources: levels [p0,p2], [p1,p3].
        let mut g = FramePlan::new((64, 64));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let c = g.create_resource("c", spec(wgpu::TextureFormat::Rg16Float, 1));
        let d = g.create_resource("d", spec(wgpu::TextureFormat::Rg16Float, 1));
        g.add_pass("p0").write(a);
        g.add_pass("p1").read(a).write(b);
        g.add_pass("p2").write(c);
        g.add_pass("p3").read(c).write(d);
        let layout = g.build();
        assert_eq!(layout.levels(), vec![vec![0, 2], vec![1, 3]]);
    }

    #[test]
    fn explicit_ordering_splits_shared_level() {
        // p0→p1 and p2→p3 are independent: [[0,2],[1,3]]; edge p0→p2 splits
        // the first level (hidden dependency without access conflict).
        let mut g = FramePlan::new((64, 64));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let c = g.create_resource("c", spec(wgpu::TextureFormat::Rg16Float, 1));
        let d = g.create_resource("d", spec(wgpu::TextureFormat::Rg16Float, 1));
        let p0 = g.add_pass("p0").write(a).id();
        g.add_pass("p1").read(a).write(b);
        let p2 = g.add_pass("p2").write(c).id();
        g.add_pass("p3").read(c).write(d);
        assert_eq!(g.build().levels(), vec![vec![0, 2], vec![1, 3]]);
        g.order_before(p0, p2);
        assert_eq!(
            g.build().levels(),
            vec![vec![0], vec![1, 2], vec![3]],
            "explicit edge lifts p2 without touching p1's level"
        );
    }

    #[test]
    #[should_panic(expected = "registered")]
    fn explicit_ordering_rejects_backward() {
        let mut g = FramePlan::new((64, 64));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let first = g.add_pass("first").write(a).id();
        let second = g.add_pass("second").write(b).id();
        g.order_before(second, first);
    }

    #[test]
    #[should_panic(expected = "no node named")]
    fn explicit_ordering_unknown_name() {
        let mut g = FramePlan::new((64, 64));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        g.add_pass("real").write(a);
        g.order_before_named("real", "ghost");
    }

    #[test]
    fn try_order_before_reports_errors_without_panicking() {
        let mut g = FramePlan::new((64, 64));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let first = g.add_pass("first").write(a).id();
        let second = g.add_pass("second").write(b).id();
        assert!(matches!(
            g.try_order_before(second, first),
            Err(OrderError::BackwardEdge { .. })
        ));
        assert_eq!(
            g.try_order_before_named("first", "ghost").map(|_| ()),
            Err(OrderError::UnknownNode {
                name: "ghost".to_owned(),
            })
        );
        // Id outside the registry is an error, not a silent garbage edge.
        assert!(matches!(
            g.try_order_before(PassId(99), PassId(100)),
            Err(OrderError::UnknownNode { .. })
        ));
        assert_eq!(g.build().levels(), vec![vec![0, 1]]);
        assert!(g.try_order_before(first, second).is_ok());
        assert_eq!(g.build().levels(), vec![vec![0], vec![1]]);
    }

    #[test]
    fn mermaid_is_a_valid_projection() {
        // S6: graph as debug projection — levels as subgraphs,
        // resources as nodes, flows as edges; GitHub renders natively.
        let mut g = FramePlan::new((64, 64));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rg16Float, 1));
        g.add_pass("p0").write(a);
        g.add_pass("p1").read(a).write(b);

        let m = g.build().mermaid();
        assert!(m.starts_with("flowchart TD\n"), "head: {m}");
        assert!(m.contains("subgraph L0[\"level 0\"]"), "levels: {m}");
        assert!(m.contains("P0[\"p0\"]"), "pass nodes: {m}");
        assert!(m.contains("R0[\"a Rgba8Unorm\"]"), "resource nodes: {m}");
        assert!(m.contains("P0 --> R0"), "write edges: {m}");
        assert!(m.contains("R0 --> P1"), "read edges: {m}");
        // Dead resources are excluded from the projection.
        let mut g2 = FramePlan::new((64, 64));
        let dead = g2.create_resource("dead", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let live = g2.create_resource("live", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        g2.add_pass("only").write(live);
        let m2 = g2.build().mermaid();
        assert!(!m2.contains("dead"), "dead resource hidden: {m2}");
        assert!(!m2.contains(format!("R{}", dead.0).as_str()));
    }

    #[test]
    fn debug_dump_lists_structure() {
        let mut g = FramePlan::new((1280, 720));
        let albedo = g.create_resource("albedo", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let hdr = g.create_resource("hdr", spec(wgpu::TextureFormat::Rgba16Float, 1));
        g.add_pass("gbuffer").write(albedo);
        g.add_pass("lighting").read(albedo).write(hdr);

        let dump = g.build().debug_dump();
        assert!(dump.contains("2 passes"), "dump: {dump}");
        assert!(dump.contains("'gbuffer'"), "dump: {dump}");
        assert!(dump.contains("'hdr'"), "dump: {dump}");
        assert!(dump.contains("pool slots"), "dump: {dump}");
        assert!(dump.contains("albedo"), "dump: {dump}");
    }

    #[test]
    fn clear_value_is_carried_to_layout() {
        let mut g = FramePlan::new((640, 480));
        let hdr = g.create_resource("hdr", spec(wgpu::TextureFormat::Rgba16Float, 1));
        let pid = {
            let builder = g.add_pass("lighting");
            let pid = builder.id();
            builder.write_clear(hdr, wgpu::Color::BLACK);
            pid
        };
        let layout = g.build();
        assert_eq!(
            layout.passes[pid.0 as usize].writes,
            vec![(hdr, Some(wgpu::Color::BLACK))]
        );
    }

    // ── S1: FrameLayout cache ────────────────────────────────────────

    fn two_pass_plan() -> (FramePlan, ResourceId, ResourceId) {
        let mut g = FramePlan::new((320, 240));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba16Float, 1));
        g.add_pass("p0").write(a);
        g.add_pass("p1").read(a).write(b);
        (g, a, b)
    }

    #[test]
    fn layout_is_cached_until_mutation() {
        let (mut g, _, _) = two_pass_plan();
        assert_eq!(g.layout_computations(), 0, "nothing computed yet");
        let _ = g.build();
        let _ = g.build();
        let _ = g.layout();
        let _ = g.layout();
        assert_eq!(
            g.layout_computations(),
            1,
            "repeated access without mutations must be a cache hit"
        );
    }

    #[test]
    fn every_mutation_invalidates_cache() {
        let (mut g, a, _) = two_pass_plan();
        let _ = g.layout();
        assert_eq!(g.layout_computations(), 1);

        g.set_surface_size(640, 480);
        let _ = g.layout();
        assert_eq!(g.layout_computations(), 2, "resize invalidates");

        // The builder chain covers both `add_pass` and `read`.
        g.add_pass("p2").read(a);
        let _ = g.layout();
        assert_eq!(g.layout_computations(), 3, "add_pass/read invalidates");

        let p2 = PassId(2);
        g.set_pass_enabled(p2, false);
        let _ = g.layout();
        assert_eq!(g.layout_computations(), 4, "pass toggle invalidates");

        g.create_resource("c", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let _ = g.layout();
        assert_eq!(g.layout_computations(), 5, "create_resource invalidates");

        g.import_resource("ext", spec(wgpu::TextureFormat::R32Float, 1));
        let _ = g.layout();
        assert_eq!(g.layout_computations(), 6, "import_resource invalidates");

        g.invalidate();
        let _ = g.layout();
        assert_eq!(g.layout_computations(), 7, "explicit invalidate works");
    }

    #[test]
    fn build_snapshot_matches_cached_layout() {
        let (mut g, a, b) = two_pass_plan();
        let cached = g.layout().debug_dump();
        let snapshot = g.build().debug_dump();
        assert_eq!(cached, snapshot, "build() must mirror the cached layout");
        assert_eq!(g.layout_computations(), 1);
        // Snapshot ids are the same stable ResourceIds the plan handed out.
        let snapshot = g.build();
        assert_eq!(snapshot.resources[a.0 as usize].name, "a");
        assert_eq!(snapshot.resources[b.0 as usize].name, "b");
    }
}
