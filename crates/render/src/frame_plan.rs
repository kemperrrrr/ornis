//! Frame plan — pass orchestration layer (бывший «render graph», Phase 0).
//!
//! An immediate-mode plan in the spirit of Frostbite FrameGraph and
//! Ponies&Light: passes are declared in execution order, and each pass
//! declares which resources it reads and writes. The plan computes
//! resource lifetimes (transient windows `[first_use, last_use]`) and
//! assigns them to pool slots so that non-overlapping resources with the
//! same specification share one slot (object-level aliasing).
//!
//! Model:
//! - `FramePlan::layout()` → cached `&FrameLayout` — pure logic, no GPU
//!   needed; recomputed only after a mutation (`build()` is the owned
//!   snapshot of the same cache);
//! - `FramePlan::execute()` yields one [`PassContext`] per pass
//!   (insertion order; disabled passes are skipped);
//! - creating real `wgpu::Texture` objects per slot is the executor's job
//!   (Phase 1). On wgpu, barriers and layout transitions are handled by
//!   wgpu itself, so the plan owns lifetimes and pooling, not
//!   synchronization.
//!
//! Invariants (panic with a clear message when violated):
//! - a resource must not be read before it is written (imported resources
//!   are exempt);
//! - unknown resource/pass ids are errors;
//! - within a single pass, no two live resources may share a pool slot
//!   (guaranteed by construction).

use std::collections::HashMap;

use ornis_schedule::{OrderError, bitset_level_plan, resolve_named_edge, validate_indexed_edge};

/// Bytes per pixel for the texture formats used by the engine's renderer.
pub fn format_bytes_per_pixel(format: wgpu::TextureFormat) -> u32 {
    match format {
        wgpu::TextureFormat::Rgba8Unorm
        | wgpu::TextureFormat::Rgba8UnormSrgb
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
fn budget_exceeded(budget: u64, required: u64, layout: &FrameLayout) -> BudgetExceeded {
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
    Fixed { width: u32, height: u32 },
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
    pub format: wgpu::TextureFormat,
    pub samples: u32,
    pub size: SizePolicy,
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
    pub id: ResourceId,
    pub name: String,
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
    pub index: usize,
    pub spec: TextureSpec,
    /// Resources sharing the slot (non-overlapping windows).
    pub resources: Vec<ResourceId>,
    pub first_pass: usize,
    pub last_pass: usize,
}

/// A pass in the executable layout.
#[derive(Debug, Clone)]
pub struct PassLayout {
    pub id: PassId,
    pub name: String,
    /// Resources read by the pass.
    pub reads: Vec<ResourceId>,
    /// Resources written by the pass; `Some(Color)` carries a clear value.
    pub writes: Vec<(ResourceId, Option<wgpu::Color>)>,
}

/// Result of `FramePlan::build()` — the computed frame layout.
#[derive(Debug, Clone)]
pub struct FrameLayout {
    pub(crate) surface_size: (u32, u32),
    /// Passes in execution order (insertion order, disabled passes dropped).
    pub(crate) passes: Vec<PassLayout>,
    /// Resources (parallel to `FramePlan::resources`).
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
    pub fn mermaid(&self) -> String {
        let mut out = String::from("flowchart TD\n");
        for (li, level) in self.levels().iter().enumerate() {
            out.push_str(&format!("  subgraph L{li}[\"level {li}\"]\n"));
            for &pi in level {
                out.push_str(&format!("    P{pi}[\"{}\"]\n", self.passes[pi].name));
            }
            out.push_str("  end\n");
        }
        for rl in &self.resources {
            if rl.first_use == usize::MAX {
                continue;
            }
            out.push_str(&format!(
                "  R{}[\"{} {:?}\"]\n",
                rl.id.0, rl.name, rl.spec.format
            ));
        }
        for (pi, pass) in self.passes.iter().enumerate() {
            for rid in &pass.reads {
                out.push_str(&format!("  R{} --> P{pi}\n", rid.0));
            }
            for (rid, _) in &pass.writes {
                out.push_str(&format!("  P{pi} --> R{}\n", rid.0));
            }
        }
        out
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

/// Debug-enforcement объявленных доступов пасса — граница
/// `PassViews::view_of` (бэклог #6, аудит §4.1): пасс запрашивает view по
/// `ResourceId` вне своих declared reads/writes → паника с именем пасса и
/// ресурса. Такой доступ — шаг вне расписания: он может гоняться с пассом
/// того же уровня параллельной записи (`FrameExecutor::execute_parallel`).
/// Пасс-аналог `assert_access_declared` для систем `core::Schedule`;
/// write-декларация покрывает и чтение собственной записи (см.
/// `Forward<OwnsDepth>` — читает собственно очищенный depth), как и в
/// core. Только debug: в release проверка скомпилирована в ноль.
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

#[derive(Debug)]
struct ResourceNode {
    name: String,
    spec: TextureSpec,
    /// Imported (external) resource: the "first touch must be a write"
    /// rule does not apply.
    imported: bool,
    /// Resource backed by an externally provided view (e.g. the swapchain):
    /// never pooled, `slot` is always `None`.
    external: bool,
}

#[derive(Debug)]
struct PassNode {
    name: String,
    reads: Vec<ResourceId>,
    writes: Vec<(ResourceId, Option<wgpu::Color>)>,
    enabled: bool,
}

/// GPU memory budget for the transient pool (S4, IDEAS §28.3).
///
/// The scheduler either fits the pool into the budget or refuses with an
/// actionable [`BudgetExceeded`]; `unbounded()` restores the S3 behavior.
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

/// The pass plan being assembled.
#[derive(Debug)]
pub struct FramePlan {
    resources: Vec<ResourceNode>,
    passes: Vec<PassNode>,
    surface_size: (u32, u32),
    /// Cached layout; `None` means dirty — the next [`FramePlan::layout`]
    /// recomputes. Every mutation resets this (S1: `compute_layout` must
    /// stay off the per-frame hot path).
    cached: Option<FrameLayout>,
    /// S5c: явные рёбра порядка (PassId регистрации i < j) поверх
    /// зависимостей из доступов — для скрытых зависимостей (общие
    /// queue-буферы рендерера), невидимых в множествах доступа.
    ordering: Vec<(PassId, PassId)>,
    /// S4 memory budget; unbounded by default.
    budget: Budget,
    /// How many times the layout has been computed over this plan's
    /// lifetime. Diagnostics for the S1 cache (tests, benches, probes).
    layout_computations: u32,
}

impl FramePlan {
    /// Creates an empty plan; `surface_size` feeds `SizePolicy::MatchSurface`.
    pub fn new(surface_size: (u32, u32)) -> Self {
        Self {
            resources: Vec::new(),
            passes: Vec::new(),
            surface_size,
            cached: None,
            ordering: Vec::new(),
            budget: Budget::unbounded(),
            layout_computations: 0,
        }
    }

    /// Updates the surface size (window resize) before the next `build()`.
    pub fn set_surface_size(&mut self, width: u32, height: u32) {
        self.surface_size = (width, height);
        self.cached = None;
    }

    /// Sets the S4 memory budget; invalidates the cached layout.
    pub fn set_budget(&mut self, budget: Budget) {
        self.budget = budget;
        self.cached = None;
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
        self.cached = None;
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
        self.cached = None;
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
        self.cached = None;
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
        self.cached = None;
        PassBuilder { plan: self, id }
    }

    /// S5c: объявляет, что пасс `before` обязан выполниться раньше пасса
    /// `after`, даже если их доступы не конфликтуют (скрытая зависимость,
    /// например общий queue-записанный uniform-буфер). Влияет только на
    /// разбиение на уровни параллельности ([`FrameLayout::levels`]);
    /// порядок исполнения — порядок регистрации.
    ///
    /// # Panics
    /// Паникует, если `after` зарегистрирован раньше `before` (порядок
    /// исполнения неизменяем) или пасс неизвестен.
    pub fn order_before(&mut self, before: PassId, after: PassId) {
        self.try_order_before(before, after)
            .unwrap_or_else(|error| panic!("order_before({before:?}, {after:?}): {error}"));
    }

    /// Мягкая [`FramePlan::order_before`]: ошибка — возвращаемый
    /// [`OrderError`], не паника. Заодно валидирует оба `PassId`
    /// (раньше ребро с неизвестным id добавлялось молча и игнорировалось
    /// при подсчёте уровней).
    pub fn try_order_before(&mut self, before: PassId, after: PassId) -> Result<(), OrderError> {
        validate_indexed_edge(before.0 as usize, after.0 as usize, |i| {
            self.passes.get(i).map(|node| node.name.clone())
        })?;
        if !self.ordering.contains(&(before, after)) {
            self.ordering.push((before, after));
        }
        self.cached = None;
        Ok(())
    }

    /// S5c: name-based [`FramePlan::order_before`] (имя пасса из
    /// `add_pass`).
    ///
    /// # Panics
    /// Паникует при неизвестном имени или обратном порядке регистрации.
    pub fn order_before_named(&mut self, before: &str, after: &str) {
        self.try_order_before_named(before, after)
            .unwrap_or_else(|error| panic!("order_before_named('{before}', '{after}'): {error}"));
    }

    /// Мягкая [`FramePlan::order_before_named`].
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
        self.cached = None;
    }

    fn resolve_resource(&self, id: ResourceId, pass_name: &str) -> &ResourceNode {
        self.resources
            .get(id.0 as usize)
            .unwrap_or_else(|| panic!("unknown resource {id:?} in pass '{pass_name}'"))
    }

    /// Returns the frame layout (lifetimes + pool slots), recomputing it
    /// only when the plan changed since the last call. This is the hot-path
    /// accessor: `RenderFrame3D::render` calls it every frame, and in
    /// steady state (no resizes, no pass toggles) it is a cache hit.
    ///
    /// # Panics
    /// Panics if invariants are violated (read-before-write, etc.) — the
    /// panic fires on the first recomputation after the offending mutation,
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
        if self.cached.is_none() {
            let layout = self.compute_layout();
            if let Some(cap) = self.budget.gpu_textures {
                let planned = layout.planned_pool_bytes();
                if planned > cap {
                    return Err(budget_exceeded(cap, planned, &layout));
                }
            }
            self.cached = Some(layout);
            self.layout_computations += 1;
        }
        // Filled by the branch above (or by an earlier call).
        Ok(self.cached.as_ref().expect("layout cache is filled above"))
    }

    /// Snapshot of the cached layout as an owned value. Equivalent to
    /// cloning [`FramePlan::layout`]; prefer `layout()` on hot paths —
    /// this clones the pass/resource/slot vectors.
    ///
    /// # Panics
    /// Same as [`FramePlan::layout`].
    pub fn build(&mut self) -> FrameLayout {
        self.layout().clone()
    }

    /// Forces the next [`FramePlan::layout`] to recompute. Mutating
    /// methods do this automatically; this is for benchmarks and tests
    /// that drive recomputation explicitly.
    pub fn invalidate(&mut self) {
        self.cached = None;
    }

    /// How many times the layout has been computed over this plan's
    /// lifetime (S1 cache diagnostics: stays flat while the cache holds).
    pub fn layout_computations(&self) -> u32 {
        self.layout_computations
    }

    fn compute_layout(&self) -> FrameLayout {
        let enabled: Vec<usize> = self
            .passes
            .iter()
            .enumerate()
            .filter(|(_, p)| p.enabled)
            .map(|(i, _)| i)
            .collect();

        let passes: Vec<PassLayout> = enabled
            .iter()
            .map(|&i| {
                let node = &self.passes[i];
                PassLayout {
                    id: PassId(i as u32),
                    name: node.name.clone(),
                    reads: node.reads.clone(),
                    writes: node.writes.clone(),
                }
            })
            .collect();

        let mut resources: Vec<ResourceLayout> = self
            .resources
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
            .collect();

        // Lifetimes over enabled passes.
        for (pi, pass) in passes.iter().enumerate() {
            for rid in pass.reads.iter().chain(pass.writes.iter().map(|(r, _)| r)) {
                let rl = &mut resources[rid.0 as usize];
                rl.first_use = rl.first_use.min(pi);
                rl.last_use = rl.last_use.max(pi);
            }
        }

        // "First touch must be a write" rule (imported resources exempt).
        for (pi, pass) in passes.iter().enumerate() {
            for &rid in &pass.reads {
                let node = &self.resources[rid.0 as usize];
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

        // Interval partitioning: greedy first-fit over slots with a free
        // window and a matching spec. External resources are never pooled.
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
            let rl = &resources[id.0 as usize];
            match slots
                .iter()
                .position(|s| s.spec == rl.spec && s.last_pass < rl.first_use)
            {
                Some(i) => {
                    slots[i].resources.push(id);
                    slots[i].last_pass = rl.last_use;
                    resources[id.0 as usize].slot = Some(i);
                }
                None => {
                    let i = slots.len();
                    slots.push(PoolSlot {
                        index: i,
                        spec: rl.spec,
                        resources: vec![id],
                        first_pass: rl.first_use,
                        last_pass: rl.last_use,
                    });
                    resources[id.0 as usize].slot = Some(i);
                }
            }
        }

        // Live resources per pass.
        let pass_alive: Vec<Vec<ResourceId>> = (0..passes.len())
            .map(|pi| {
                resources
                    .iter()
                    .filter(|rl| rl.alive_at(pi))
                    .map(|rl| rl.id)
                    .collect()
            })
            .collect();

        // Internal invariant check: a slot must not be shared within one pass.
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

        let levels = layout_levels(&passes, &self.ordering);
        FrameLayout {
            surface_size: self.surface_size,
            passes,
            resources,
            slots,
            pass_alive,
            levels,
        }
    }

    /// Executes the plan: for each pass in layout order, `run` is invoked
    /// with a [`PassContext`] (live resources and their slots).
    ///
    /// # Panics
    /// Panics if the layout has not been computed yet (call `build()` first).
    pub fn execute(&self, layout: &FrameLayout, mut run: impl FnMut(PassContext<'_>)) {
        for index in 0..layout.passes.len() {
            run(PassContext { layout, index });
        }
    }
}

/// Context of the pass being executed: live resources and their pool slots.
#[derive(Debug)]
pub struct PassContext<'a> {
    layout: &'a FrameLayout,
    index: usize,
}

impl<'a> PassContext<'a> {
    /// The pass in execution order.
    pub fn pass(&self) -> &'a PassLayout {
        &self.layout.passes[self.index]
    }

    /// Index of the pass within the layout.
    pub fn pass_index(&self) -> usize {
        self.index
    }

    /// Resources alive on this pass.
    pub fn alive(&self) -> &'a [ResourceId] {
        &self.layout.pass_alive[self.index]
    }

    /// Resource metadata.
    pub fn resource(&self, id: ResourceId) -> &'a ResourceLayout {
        &self.layout.resources[id.0 as usize]
    }

    /// Pool slot for the resource, if it is alive on this pass.
    pub fn slot_of(&self, id: ResourceId) -> Option<usize> {
        let rl = self.resource(id);
        if rl.alive_at(self.index) {
            rl.slot
        } else {
            None
        }
    }
}

/// Builder for declaring a pass.
///
/// Compatibility shim (S3): prefer declaring passes as typed systems
/// (`impl FramePass` + `SystemSet::add_system`); the builder stays for
/// tests, tools and the migration period.
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
        self.plan.cached = None;
        self
    }

    /// Declares a resource as written by the pass (no clear).
    pub fn write(self, id: ResourceId) -> Self {
        self.plan
            .resolve_resource(id, &self.plan.passes[self.id.0 as usize].name);
        self.plan.passes[self.id.0 as usize]
            .writes
            .push((id, None));
        self.plan.cached = None;
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
        self.plan.cached = None;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Бэклог #6 (аудит §4.1, критерий выхода Фазы B «sneaky pass»):
        // пасс запрашивает ресурс вне своих declared reads/writes →
        // debug-паника с именем пасса и ресурса (симметрия со
        // `sneaky`-системой `core::Schedule`).
        let mut g = FramePlan::new((320, 240));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        g.add_pass("writer").write(a).write(b);
        g.add_pass("sneaky").read(a);
        let layout = g.build();
        // Пасс 1 декларировал только read(a); peek в `b` — вне набора.
        assert_pass_access_declared(&layout, 1, b);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn declared_pass_access_passes_enforcement() {
        // Честный пасс: read-декларация покрывает view; write-декларация
        // покрывает и чтение собственной записи (own-write read), как и в
        // core `declared_access_passes_enforcement` — обе проверки молчат.
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
        let mut visits: Vec<(usize, Vec<ResourceId>, Option<usize>)> = Vec::new();
        g.execute(&layout, |ctx| {
            let slot_a = ctx.slot_of(a);
            visits.push((ctx.pass_index(), ctx.alive().to_vec(), slot_a));
        });

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
        // p0→p1 (a→b) и p2→p3 (c→d) не делят ресурсов: уровни [p0,p2], [p1,p3].
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
        // p0→p1 и p2→p3 независимы: [[0,2],[1,3]]; ребро p0→p2 разводит
        // первый уровень (скрытая зависимость без конфликта доступов).
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
        // Id вне реестра — ошибка, а не молчаливое мусорное ребро.
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
        // S6: граф как отладочная проекция — уровни подграфами,
        // ресурсы узлами, потоки рёбрами; GitHub рендерит нативно.
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
        // Мёртвые ресурсы в проекцию не входят.
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

    fn two_pass_graph() -> (FramePlan, ResourceId, ResourceId) {
        let mut g = FramePlan::new((320, 240));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba16Float, 1));
        g.add_pass("p0").write(a);
        g.add_pass("p1").read(a).write(b);
        (g, a, b)
    }

    #[test]
    fn layout_is_cached_until_mutation() {
        let (mut g, _, _) = two_pass_graph();
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
        let (mut g, a, _) = two_pass_graph();
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
        let (mut g, a, b) = two_pass_graph();
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
