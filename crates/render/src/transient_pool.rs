//! Transient pool — the dynamic half of the dissolved `FramePlan` (d2).
//!
//! [`Schedule`] (and its render-side declaration registry) answers "what
//! runs in which order" with static keys; the pool answers "where in
//! memory, and does it fit" with runtime keys. It compiles a borrowed
//! declaration snapshot ([`PoolInput`]) into a shared [`FrameLayout`]:
//! resource lifetimes (`first_use..last_use`), interval-partitioned pool
//! slots over matching [`TextureSpec`]s, per-pass live sets, parallel
//! levels, and the [`Budget`] check. Results memoize by declaration
//! generation, so steady-state frames share one `Arc` without recompute.
//!
//! The pool assumes frame-quantized, upfront declarations with a
//! non-realtime compile budget (push model). Domains that violate this —
//! latency-driven streaming such as audio — must not use it; see the
//! static/dynamic split in `docs/rendering/unified-scheduler.md`.
//!
//! Two instances coexist by design: the declaration registry keeps one
//! for cold paths (`layout()`/`build()` in tests and tools), while
//! [`FrameExecutor`](crate::frame_exec::FrameExecutor) owns one for the
//! frame hot path (`ensure_layout`). Both key off the same generation,
//! so a mutation refreshes each at most once.

use std::collections::HashMap;
use std::sync::Arc;

use ornis_schedule::{MermaidDiagram, bitset_level_plan};

/// Bytes per pixel for the texture formats used by the engine's renderer.
pub fn format_bytes_per_pixel(format: wgpu::TextureFormat) -> u32 {
    match format {
        wgpu::TextureFormat::Rgba8Unorm
        | wgpu::TextureFormat::Rgba8UnormSrgb
        | wgpu::TextureFormat::Bgra8UnormSrgb
        | wgpu::TextureFormat::R32Uint
        | wgpu::TextureFormat::Rg16Float
        | wgpu::TextureFormat::Depth32Float
        | wgpu::TextureFormat::Depth24Plus => 4,
        wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rg32Float => 8,
        wgpu::TextureFormat::Rgba32Float => 16,
        other => panic!("format_bytes_per_pixel: unsupported format {other:?}"),
    }
}

/// Builds the actionable budget error: top slots by bytes.
pub(crate) fn budget_exceeded(budget: u64, required: u64, layout: &FrameLayout) -> BudgetExceeded {
    let mut slots: Vec<&PoolSlot> = layout.slots.iter().collect();
    slots.sort_by_key(|s| std::cmp::Reverse(slot_bytes(s, layout.surface_size)));
    let offenders = slots
        .iter()
        .take(3)
        .map(|s| {
            let (w, h) = s.spec.size.resolve(layout.surface_size);
            let names: Vec<&str> = s
                .resources
                .iter()
                .map(|&id| layout.resources[id.0 as usize].name.as_str())
                .collect();
            format!(
                "{} ({:?} {}x{}, {} B)",
                names.join("/"),
                s.spec.format,
                w,
                h,
                slot_bytes(s, layout.surface_size)
            )
        })
        .collect();
    BudgetExceeded {
        budget,
        required,
        offenders,
    }
}

fn slot_bytes(slot: &PoolSlot, surface: (u32, u32)) -> u64 {
    let (w, h) = slot.spec.size.resolve(surface);
    format_bytes_per_pixel(slot.spec.format) as u64 * w as u64 * h as u64
}

/// Texture size policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizePolicy {
    /// Size matches the surface (swapchain) size.
    MatchSurface,
    /// Surface size divided by a power-of-two divisor (mip chains, e.g.
    /// bloom at 1/2, 1/4, 1/8). The result is floored and clamped to 1.
    Fraction(u32),
    /// Fixed size.
    Fixed {
        /// Absolute width in texels.
        width: u32,
        /// Absolute height in texels.
        height: u32,
    },
}

impl SizePolicy {
    /// Resolves the policy to a concrete (width, height) for a surface size.
    pub fn resolve(&self, surface: (u32, u32)) -> (u32, u32) {
        match *self {
            SizePolicy::MatchSurface => surface,
            SizePolicy::Fraction(divisor) => {
                let divisor = divisor.max(1);
                ((surface.0 / divisor).max(1), (surface.1 / divisor).max(1))
            }
            SizePolicy::Fixed { width, height } => (width, height),
        }
    }
}

/// Texture specification — the pool reuse key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSpec {
    /// Texture format; resources of different formats never share slots.
    pub format: wgpu::TextureFormat,
    /// MSAA sample count.
    pub samples: u32,
    /// Size policy (resolved against the surface at allocation time).
    pub size: SizePolicy,
}

impl TextureSpec {
    /// Placeholder spec for an externally backed output (e.g. a swapchain
    /// view): the executor never pools the resource, so the format/size are
    /// only used for plan dumps and consistency checks.
    ///
    /// Kept separate from `FrameOwned` defaults so the `external` flag on
    /// the resource (and `ResourceLayout::external`) is the single source
    /// of truth for "skip pooling"; the spec exists only to satisfy the
    /// `Hash` slot-key contract.
    pub fn external() -> Self {
        Self {
            format: wgpu::TextureFormat::Rgba8Unorm,
            samples: 1,
            size: SizePolicy::MatchSurface,
        }
    }
}

/// Logical resource handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(pub u32);

/// Logical pass handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassId(pub u32);

/// Per-resource information in a layout.
#[derive(Debug, Clone)]
pub struct ResourceLayout {
    /// Unique resource identifier.
    pub id: ResourceId,
    /// Human-readable name for debugging and plan dumps.
    pub name: String,
    /// Texture format/size/usage specification.
    pub spec: TextureSpec,
    /// Index of the first pass that uses the resource; `usize::MAX` if unused.
    pub first_use: usize,
    /// Index of the last pass that uses the resource; `0` if unused.
    pub last_use: usize,
    /// Pool slot (`None` — the resource is not used by any enabled pass,
    /// or it is external).
    pub slot: Option<usize>,
    /// Backed by an externally provided view (swapchain or similar);
    /// never pooled.
    pub external: bool,
}

impl ResourceLayout {
    /// Whether the resource is alive at the pass with `pass_index`.
    pub fn alive_at(&self, pass_index: usize) -> bool {
        self.first_use != usize::MAX && self.first_use <= pass_index && pass_index <= self.last_use
    }
}

/// Pool slot: a group of resources with the same [`TextureSpec`] whose
/// lifetime windows do not overlap.
#[derive(Debug, Clone)]
pub struct PoolSlot {
    /// Slot index used by the executor to key its texture pool.
    pub index: usize,
    /// Shared spec — the pool reuse key.
    pub spec: TextureSpec,
    /// Resources sharing the slot (non-overlapping windows).
    pub resources: Vec<ResourceId>,
    /// Index of the first pass using the slot.
    pub first_pass: usize,
    /// Index of the last pass using the slot.
    pub last_pass: usize,
}

/// A pass in the executable layout.
#[derive(Debug, Clone)]
pub struct PassLayout {
    /// Identifier of the pass.
    pub id: PassId,
    /// Human-readable pass name for debugging and plan dumps.
    pub name: String,
    /// Resources read by the pass.
    pub reads: Vec<ResourceId>,
    /// Resources written by the pass; `Some(Color)` carries a clear value.
    pub writes: Vec<(ResourceId, Option<wgpu::Color>)>,
}

/// Result of compiling a [`PoolInput`] — the computed frame layout.
#[derive(Debug, Clone)]
pub struct FrameLayout {
    pub(crate) surface_size: (u32, u32),
    /// Passes in execution order (insertion order, disabled passes dropped).
    pub(crate) passes: Vec<PassLayout>,
    /// Resources (parallel to the registry's declaration order).
    pub(crate) resources: Vec<ResourceLayout>,
    /// Pool slots.
    pub(crate) slots: Vec<PoolSlot>,
    /// Live resources per pass (by index into `passes`).
    pub(crate) pass_alive: Vec<Vec<ResourceId>>,
    /// Parallel execution levels (bitset plan via `ornis-schedule`),
    /// computed once per build and cached in the layout (audit §4.3 —
    /// no recomputation per `levels()` call).
    pub(crate) levels: Vec<Vec<usize>>,
}

impl FrameLayout {
    /// Total bytes the pool will allocate at this layout's surface size —
    /// the device-free counterpart of `FrameExecutor::texture_budget`
    /// (golden tests, S0 metrics, the S4 budget check).
    pub fn planned_pool_bytes(&self) -> u64 {
        self.slots
            .iter()
            .map(|slot| {
                let (w, h) = slot.spec.size.resolve(self.surface_size);
                format_bytes_per_pixel(slot.spec.format) as u64 * w as u64 * h as u64
            })
            .sum()
    }

    /// Parallel execution levels (S5b planning data): passes whose
    /// declared accesses do not conflict share a level; levels are
    /// ordered by dependencies (read-after-write, write-after-read,
    /// write-after-write), passes within a level are independent and
    /// safe to record in parallel. Deterministic — derived from the
    /// registration order and the declared accesses, exactly like the
    /// core `ornis_core::schedule::Schedule`. Computed once per build
    /// (bitset plan from `ornis-schedule`, audit §4.3) and cached in
    /// this layout; the accessor clones the vec, as before.
    pub fn levels(&self) -> Vec<Vec<usize>> {
        self.levels.clone()
    }

    /// Mermaid diagram of this layout — the debug projection (S6):
    /// passes grouped into parallel-level subgraphs, resources as nodes,
    /// write/read flows as edges. GitHub renders ```mermaid blocks
    /// natively, so a layout drop pasted into a PR review becomes a
    /// picture of the frame pipeline.
    ///
    /// Slice 1b (toward graph elimination): rendered by the shared
    /// [`MermaidDiagram`] projector — same byte format pinned by the
    /// `mermaid_is_a_valid_projection` test; the same diagram is
    /// available from the top-level scheduler (`Schedule::mermaid`).
    pub fn mermaid(&self) -> String {
        let mut d = MermaidDiagram::new();
        for (li, level) in self.levels().iter().enumerate() {
            let nodes: Vec<(String, String)> = level
                .iter()
                .map(|&pi| (format!("P{pi}"), self.passes[pi].name.clone()))
                .collect();
            d.level(&format!("L{li}"), &format!("level {li}"), &nodes);
        }
        for rl in &self.resources {
            if rl.first_use == usize::MAX {
                continue;
            }
            d.node(
                &format!("R{}", rl.id.0),
                &format!("{} {:?}", rl.name, rl.spec.format),
            );
        }
        for (pi, pass) in self.passes.iter().enumerate() {
            for rid in &pass.reads {
                d.edge(&format!("R{}", rid.0), &format!("P{pi}"));
            }
            for (rid, _) in &pass.writes {
                d.edge(&format!("P{pi}"), &format!("R{}", rid.0));
            }
        }
        d.render()
    }

    /// Textual layout dump for debugging/reporting.
    pub fn debug_dump(&self) -> String {
        let mut s = format!(
            "frame plan: {} passes, {} resources, {} pool slots (surface {:?})\n",
            self.passes.len(),
            self.resources.len(),
            self.slots.len(),
            self.surface_size
        );
        for (i, pass) in self.passes.iter().enumerate() {
            let reads: Vec<&str> = pass
                .reads
                .iter()
                .map(|&r| self.resources[r.0 as usize].name.as_str())
                .collect();
            let writes: Vec<&str> = pass
                .writes
                .iter()
                .map(|&(r, _)| self.resources[r.0 as usize].name.as_str())
                .collect();
            s += &format!(
                "  pass {i} '{}' read[{}] write[{}]\n",
                pass.name,
                reads.join(", "),
                writes.join(", ")
            );
        }
        for rl in &self.resources {
            if rl.first_use == usize::MAX {
                s += &format!("  resource '{}' UNUSED\n", rl.name);
            } else {
                s += &format!(
                    "  resource '{}' ({:?}) passes {}..={} slot {:?}\n",
                    rl.name, rl.spec, rl.first_use, rl.last_use, rl.slot
                );
            }
        }
        for slot in &self.slots {
            let names: Vec<&str> = slot
                .resources
                .iter()
                .map(|&r| self.resources[r.0 as usize].name.as_str())
                .collect();
            s += &format!(
                "  slot #{} {:?} passes {}..={}: {}\n",
                slot.index,
                slot.spec,
                slot.first_pass,
                slot.last_pass,
                names.join(", ")
            );
        }
        s
    }
}

/// Debug enforcement of declared pass accesses — boundary of
/// `PassViews::view_of` (backlog #6, audit §4.1): a pass requesting a view
/// for a `ResourceId` outside its declared reads/writes panics with the
/// pass and resource names. Such access is an out-of-schedule step: it
/// may race with a pass at the same parallel level
/// (`FrameExecutor::execute_parallel`). Pass-level analogue of
/// `assert_access_declared` for `core::Schedule` systems; a write
/// declaration also covers reading its own write (see
/// `Forward<OwnsDepth>` — reads its own cleared depth), as in core.
/// Debug-only: compiled out in release.
#[cfg(debug_assertions)]
pub(crate) fn assert_pass_access_declared(layout: &FrameLayout, pass_index: usize, id: ResourceId) {
    let pass = &layout.passes[pass_index];
    let declared =
        pass.reads.contains(&id) || pass.writes.iter().any(|(written, _)| *written == id);
    if !declared {
        let resource = &layout.resources[id.0 as usize];
        panic!(
            "pass '{}' (index {pass_index}) accesses resource '{}' ({id:?}) that is not \
             declared in its access set (PassBuilder::read/write) — undeclared access breaks \
             the frame-plan scheduling contract",
            pass.name, resource.name
        );
    }
}

/// GPU memory budget for the transient pool (S4, IDEAS §28.3).
///
/// The pool either fits into the budget or refuses with an actionable
/// [`BudgetExceeded`]; `unbounded()` restores the S3 behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Byte cap for the pooled transient textures; `None` = unbounded.
    pub gpu_textures: Option<u64>,
}

impl Budget {
    /// No cap: the S3 behavior (any pool size passes).
    pub fn unbounded() -> Self {
        Self { gpu_textures: None }
    }

    /// Cap the transient texture pool at `bytes`.
    pub fn gpu_textures(bytes: u64) -> Self {
        Self {
            gpu_textures: Some(bytes),
        }
    }
}

/// The transient pool does not fit the configured [`Budget`] (S4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetExceeded {
    /// The configured cap.
    pub budget: u64,
    /// What the pool needs at this plan configuration.
    pub required: u64,
    /// Largest slots (bytes desc): what to shrink or disable first.
    pub offenders: Vec<String>,
}

impl Default for Budget {
    /// Unbounded pool — the pre-budget behavior.
    fn default() -> Self {
        Self::unbounded()
    }
}

impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "transient pool needs {} ({:.1} MiB), budget {} ({:.1} MiB); largest slots: {}",
            self.required,
            self.required as f64 / (1024.0 * 1024.0),
            self.budget,
            self.budget as f64 / (1024.0 * 1024.0),
            self.offenders.join("; ")
        )?;
        if !self.offenders.is_empty() {
            write!(f, " — reduce resource sizes or disable passes (e.g. bloom)")?;
        }
        Ok(())
    }
}

/// A declared resource, as seen by the pool compiler: name, spec, and
/// the import/external flags that shape validation and slot assignment.
#[derive(Debug)]
pub(crate) struct ResourceNode {
    pub name: String,
    pub spec: TextureSpec,
    /// Imported (external) resource: the "first touch must be a write"
    /// rule does not apply.
    pub imported: bool,
    /// Resource backed by an externally provided view (e.g. the swapchain):
    /// never pooled, `slot` is always `None`.
    pub external: bool,
}

/// A declared pass, as seen by the pool compiler.
#[derive(Debug)]
pub(crate) struct PassNode {
    pub name: String,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<(ResourceId, Option<wgpu::Color>)>,
    pub enabled: bool,
}

/// Borrowed declaration snapshot for one [`TransientPool::ensure`]
/// compilation. The registry owns the declarations (`FramePlan` today,
/// `SystemSet` after the d3 consolidation); the pool only borrows them
/// for the duration of the compile.
#[derive(Debug)]
pub(crate) struct PoolInput<'a> {
    pub resources: &'a [ResourceNode],
    pub passes: &'a [PassNode],
    pub ordering: &'a [(PassId, PassId)],
    pub surface_size: (u32, u32),
    pub budget: Budget,
}

/// Layout projections of the enabled passes.
fn collect_enabled_passes(nodes: &[PassNode]) -> Vec<PassLayout> {
    nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.enabled)
        .map(|(i, node)| PassLayout {
            // `PassId` is positional in the full declaration order
            // (disabled passes keep their indices) — `layout_levels`
            // translates explicit edges through these ids.
            id: PassId(i as u32),
            name: node.name.clone(),
            reads: node.reads.clone(),
            writes: node.writes.clone(),
        })
        .collect()
}

fn init_resource_layout(nodes: &[ResourceNode]) -> Vec<ResourceLayout> {
    nodes
        .iter()
        .enumerate()
        .map(|(i, node)| ResourceLayout {
            id: ResourceId(i as u32),
            name: node.name.clone(),
            spec: node.spec,
            first_use: usize::MAX,
            last_use: 0,
            slot: None,
            external: node.external,
        })
        .collect()
}

/// Lifetimes over enabled passes.
fn compute_resource_lifetimes(passes: &[PassLayout], resources: &mut [ResourceLayout]) {
    for (pi, pass) in passes.iter().enumerate() {
        for rid in pass.reads.iter().chain(pass.writes.iter().map(|(r, _)| r)) {
            let rl = &mut resources[rid.0 as usize];
            rl.first_use = rl.first_use.min(pi);
            rl.last_use = rl.last_use.max(pi);
        }
    }
}

/// "First touch must be a write" rule (imported resources exempt).
fn validate_first_touch_is_write(
    passes: &[PassLayout],
    nodes: &[ResourceNode],
    resources: &[ResourceLayout],
) {
    for (pi, pass) in passes.iter().enumerate() {
        for &rid in &pass.reads {
            let node = &nodes[rid.0 as usize];
            let rl = &resources[rid.0 as usize];
            if !node.imported && rl.first_use == pi {
                let written_earlier = pass.writes.iter().any(|(w, _)| *w == rid);
                let first_write = passes[..pi]
                    .iter()
                    .any(|p| p.writes.iter().any(|(w, _)| *w == rid));
                if !written_earlier && !first_write {
                    panic!(
                        "resource '{}' is read in pass '{}' (index {pi}) before any write; \
                         use import_resource() for external inputs, or write it in an earlier pass",
                        node.name, pass.name
                    );
                }
            }
        }
    }
}

/// Interval partitioning: greedy first-fit over slots with a free window
/// and a matching spec. External resources are never pooled.
fn assign_pool_slots(resources: &mut [ResourceLayout]) -> Vec<PoolSlot> {
    let mut used: Vec<ResourceId> = resources
        .iter()
        .filter(|rl| rl.first_use != usize::MAX && !rl.external)
        .map(|rl| rl.id)
        .collect();
    used.sort_by_key(|&id| {
        (
            resources[id.0 as usize].first_use,
            resources[id.0 as usize].last_use,
        )
    });

    let mut slots: Vec<PoolSlot> = Vec::new();
    for id in used {
        let (spec, first_use, last_use) = {
            let rl = &resources[id.0 as usize];
            (rl.spec, rl.first_use, rl.last_use)
        };
        match slots
            .iter()
            .position(|s| s.spec == spec && s.last_pass < first_use)
        {
            Some(i) => {
                slots[i].resources.push(id);
                slots[i].last_pass = last_use;
                resources[id.0 as usize].slot = Some(i);
            }
            None => {
                let i = slots.len();
                slots.push(PoolSlot {
                    index: i,
                    spec,
                    resources: vec![id],
                    first_pass: first_use,
                    last_pass: last_use,
                });
                resources[id.0 as usize].slot = Some(i);
            }
        }
    }
    slots
}

fn live_resources_per_pass(
    passes: &[PassLayout],
    resources: &[ResourceLayout],
) -> Vec<Vec<ResourceId>> {
    (0..passes.len())
        .map(|pi| {
            resources
                .iter()
                .filter(|rl| rl.alive_at(pi))
                .map(|rl| rl.id)
                .collect()
        })
        .collect()
}

/// Internal invariant check: a slot must not be shared within one pass.
fn validate_no_slot_aliasing(pass_alive: &[Vec<ResourceId>], resources: &[ResourceLayout]) {
    for (pi, alive) in pass_alive.iter().enumerate() {
        let mut seen: HashMap<usize, ResourceId> = HashMap::new();
        for &rid in alive {
            let rl = &resources[rid.0 as usize];
            let Some(slot) = rl.slot else {
                continue;
            };
            if let Some(prev) = seen.insert(slot, rid) {
                panic!(
                    "layout bug: pass {pi} aliases slot #{slot} for resources {prev:?} and {rid:?}"
                );
            }
        }
    }
}

/// Parallel levels of a built layout: bitset plan (`ornis-schedule`)
/// over per-pass access slices plus translated explicit edges
/// (registration PassId → layout index; disabled passes are not in the
/// layout, so their edges drop out).
fn layout_levels(passes: &[PassLayout], ordering: &[(PassId, PassId)]) -> Vec<Vec<usize>> {
    let reads: Vec<Vec<ResourceId>> = passes.iter().map(|p| p.reads.clone()).collect();
    let writes: Vec<Vec<ResourceId>> = passes
        .iter()
        .map(|p| p.writes.iter().map(|(id, _)| *id).collect())
        .collect();
    let index_of = |id: PassId| passes.iter().position(|p| p.id == id);
    let edges: Vec<(usize, usize)> = ordering
        .iter()
        .filter_map(|(b, a)| Some((index_of(*b)?, index_of(*a)?)))
        .collect();
    bitset_level_plan(&reads, &writes, &edges)
}

/// Transient GPU-memory allocator: compiles declaration snapshots into
/// shared layout snapshots.
///
/// The pool holds no declarations — only the memoized result keyed by
/// the registry's declaration generation. [`TransientPool::ensure`]
/// recompiles on a generation miss (or an empty cache) and enforces the
/// snapshot's [`Budget`]; every other call is an `Arc` clone.
#[derive(Debug, Default)]
pub struct TransientPool {
    /// Shared layout snapshot + the declaration generation it was built from.
    cached: Option<(u64, Arc<FrameLayout>)>,
    /// How many times a layout has been compiled over this pool's
    /// lifetime. Diagnostics for the S1 cache (tests, benches, probes).
    layout_computations: u32,
}

impl TransientPool {
    /// Creates an empty pool (no cached layout).
    pub fn new() -> Self {
        Self::default()
    }

    /// Drops the memoized layout snapshot without touching GPU objects.
    /// (GPU textures live in the executor's object pool, not here.)
    pub fn invalidate(&mut self) {
        self.cached = None;
    }

    /// How many times a layout has been compiled over this pool's
    /// lifetime (S1 cache diagnostics: stays flat while the cache holds).
    pub fn layout_computations(&self) -> u32 {
        self.layout_computations
    }

    /// Shared snapshot of the compiled layout for `generation`.
    ///
    /// In steady state (same generation as the last call) this is a
    /// cache hit: no recomputation, just an `Arc` clone. On a miss the
    /// snapshot is rebuilt from `input` and memoized.
    ///
    /// # Panics
    ///
    /// Panics if declaration invariants are violated
    /// (read-before-write, slot aliasing) — the panic fires on the first
    /// compilation after the offending mutation, not at the mutation site.
    ///
    /// # Errors
    ///
    /// Returns [`BudgetExceeded`] when the compiled pool does not fit
    /// `input.budget`; nothing is cached in that case.
    pub(crate) fn ensure(
        &mut self,
        generation: u64,
        input: &PoolInput<'_>,
    ) -> Result<Arc<FrameLayout>, BudgetExceeded> {
        if let Some((cached_generation, layout)) = &self.cached
            && *cached_generation == generation
        {
            return Ok(Arc::clone(layout));
        }
        let layout = self.compile(input);
        if let Some(cap) = input.budget.gpu_textures {
            let planned = layout.planned_pool_bytes();
            if planned > cap {
                return Err(budget_exceeded(cap, planned, &layout));
            }
        }
        let layout = Arc::new(layout);
        self.cached = Some((generation, Arc::clone(&layout)));
        self.layout_computations += 1;
        Ok(layout)
    }

    /// The memoized snapshot, if any (set by the last successful [`TransientPool::ensure`]).
    pub(crate) fn cached(&self) -> Option<&Arc<FrameLayout>> {
        self.cached.as_ref().map(|(_, layout)| layout)
    }

    fn compile(&self, input: &PoolInput<'_>) -> FrameLayout {
        let passes = collect_enabled_passes(input.passes);
        let mut resources = init_resource_layout(input.resources);

        compute_resource_lifetimes(&passes, &mut resources);
        validate_first_touch_is_write(&passes, input.resources, &resources);

        let slots = assign_pool_slots(&mut resources);
        let pass_alive = live_resources_per_pass(&passes, &resources);
        validate_no_slot_aliasing(&pass_alive, &resources);

        let levels = layout_levels(&passes, input.ordering);
        FrameLayout {
            surface_size: input.surface_size,
            passes,
            resources,
            slots,
            pass_alive,
            levels,
        }
    }
}
