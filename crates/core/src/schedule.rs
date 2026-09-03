//! Unified system scheduler — S5a foundation (PLAN App. C, IDEAS §28.2).
//!
//! One scheduler for everything: systems declare accesses to singleton
//! resources (the "world" in [`Resources`]), the scheduler derives conflicts
//! (read-after-write, write-after-read, write-after-write), arranges
//! systems into parallelism levels and executes levels sequentially,
//! while systems within a level run in parallel (rayon).
//!
//! Determinism (Strong Confluence): levels are derived from registration
//! order; within a level systems do not conflict on *declared* accesses
//! (writes are disjoint from others' reads/writes), and any two conflicting
//! systems run in registration order. With honest declarations this
//! guarantees absence of data races, but **not** bit-identical parallel
//! and sequential runs — that requires the commutativity contract below.
//!
//! # Commutativity contract
//!
//! A `reads` access allows shared use of a resource by systems on the
//! same level, and interior mutability (`Mutex`/atomics) makes it
//! **memory-safe but non-deterministic**: the order in which level
//! systems acquire a shared `Mutex<Vec>` depends on thread scheduling
//! and varies between runs. Therefore a resource declared as `reads`
//! and mutated from within must be a **commutative accumulator**: the
//! observable result must not depend on operation order (atomic counters
//! with commutative ops; append multisets compared up to permutation).
//! Non-commutative mutation ("last writer wins", order-dependent `push`,
//! stack) must be reflected as `writes` — then the conflict is visible
//! to the scheduler and systems are split across levels in registration
//! order. The boundary is illustrated by the `parallel_matches_sequential`
//! test: events are compared sorted, so permutation within a level is an
//! allowed part of the semantics.
//!
//! Registration order is the tie-break, as with render-graph passes:
//! any two conflicting processes run in insertion order.
//!
//! # Enforcement of declared accesses
//!
//! The "do not touch others' resources" condition is no longer just
//! an honor system: while a system runs, [`Resources::get`]/
//! [`Resources::contains`] verify that the resource is declared in that
//! system's `access()` (read or write; own write covers read). A
//! violation panics with the system name and resource type. The check
//! is a thread-local stack of active declarations (RAII, correctly
//! unwound on panics and nested schedulers); outside [`Schedule::run`]
//! access is unrestricted. Enabled by default in debug builds and
//! disabled in release ([`Schedule::set_enforce_accesses`] overrides).
//!
//! # `SmartStore` lane granularity (audit §3.6, backlog #5)
//!
//! The component store is a plain singleton in [`Resources`]; the
//! `reads::<SmartStore>()` declaration gives systems the store itself
//! (shared read, no conflict). Component granularity is declared
//! separately: [`SystemAccess::reads_lane`]/[`SystemAccess::writes_lane`]
//! by component `TypeId` — conflicts are derived per lane, systems over
//! disjoint lanes run in parallel, manual `order_before` is unnecessary.
//! Resources and lanes are separate scheduler key namespaces (the same
//! type can be a resource in one system and a component in another
//! without a false conflict). Enforcement is at the
//! `SmartStore::read_lane/write_lane` boundary: reads are covered by
//! `reads_lane` or `writes_lane`, writes strictly by `writes_lane`
//! (the write guard proves intent to write — unlike resources with
//! their invisible interior mutability). Cold and lock-free lanes are
//! separate namespaces and not covered by this protocol.
//!
//! # Parallelism inside systems (audit §3.3, backlog #7)
//!
//! The TLS stack is active in the thread executing `System::run`; child
//! tasks of the system start with an empty stack. The gap is closed by
//! frame capture: [`capture_access_frame`] before entering the parallel
//! section and [`AccessFrameCapture::install`] in each child task. The
//! `#[smart_pipeline]` macro generates this pair around `par_iter` bodies
//! automatically (the engine's main pattern is covered). Documented
//! limitation: manual parallelism (`rayon::scope`/`spawn`) inside a
//! system **without** capture is still unchecked, and enforcement is
//! debug-only by default (both limits are the "document it"
//! requirement from §3.3).
//!
//! Level and conflict mechanics (including the bitset plan and plan
//! cache with diagnostics), the unified [`OrderError`] and the level
//! executor live in the `ornis-schedule` crate (Phase A, audit §7);
//! what remains here is the ECS frontend: [`Resources`],
//! [`SystemAccess`], [`System`] and TLS enforcement of declared accesses.

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;

use ornis_schedule::{
    MermaidDiagram, PlanCache, bitset_level_plan, resolve_named_edge, run_levels,
};
pub use ornis_schedule::{OrderError, compute_levels};

/// Singleton resource container ("world"): one value per type.
/// Mutation via interior mutability (`Mutex<T>`, atomics) —
/// parallel systems receive `&Resources`.
#[derive(Default)]
pub struct Resources {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Resources {
    /// Creates an empty resource map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts/replaces the singleton of type `R`; returns the previous one if any.
    pub fn insert<R: Any + Send + Sync>(&mut self, resource: R) -> Option<R> {
        self.map
            .insert(TypeId::of::<R>(), Box::new(resource))
            .and_then(|boxed| boxed.downcast::<R>().ok())
            .map(|boxed| *boxed)
    }

    /// Whether a singleton of type `R` exists.
    ///
    /// # Panics
    /// Panics if called from a system that did not declare `R`
    /// (access enforcement enabled; see module docs).
    pub fn contains<R: Any + Send + Sync>(&self) -> bool {
        assert_access_declared::<R>();
        self.map.contains_key(&TypeId::of::<R>())
    }

    /// Shared access to the singleton of type `R`.
    ///
    /// # Panics
    /// Panics if called from a system that did not declare `R`
    /// (read or write; access enforcement enabled).
    pub fn get<R: Any + Send + Sync>(&self) -> Option<&R> {
        assert_access_declared::<R>();
        self.map
            .get(&TypeId::of::<R>())
            .and_then(|boxed| boxed.downcast_ref::<R>())
    }

    /// Mutable access (only between scheduler runs, not from parallel systems).
    pub fn get_mut<R: Any + Send + Sync>(&mut self) -> Option<&mut R> {
        self.map
            .get_mut(&TypeId::of::<R>())
            .and_then(|boxed| boxed.downcast_mut::<R>())
    }

    /// Removes the singleton of type `R`.
    pub fn remove<R: Any + Send + Sync>(&mut self) -> Option<R> {
        self.map
            .remove(&TypeId::of::<R>())
            .and_then(|boxed| boxed.downcast::<R>().ok())
            .map(|boxed| *boxed)
    }

    /// Number of singleton resources.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the world is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Declared system accesses: singleton resources (`reads`/`writes`) and
/// hot `SmartStore` component lanes (`reads_lanes`/`writes_lanes`) by
/// type. Previously called `Access` — renamed to avoid clashing with
/// the typed `ornis_render::Access` (graph ZST markers `Read`/`Write`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemAccess {
    /// Resource types the system reads (`&R`).
    pub reads: Vec<TypeId>,
    /// Resource types the system writes (`&mut R`).
    pub writes: Vec<TypeId>,
    /// Component lanes read by the system (`TypeId` of the component).
    pub reads_lanes: Vec<TypeId>,
    /// Component lanes written by the system (`TypeId` of the component).
    pub writes_lanes: Vec<TypeId>,
}

impl SystemAccess {
    /// Creates an empty access set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `R` to the read set.
    pub fn reads<R: Any + Send + Sync>(mut self) -> Self {
        self.reads.push(TypeId::of::<R>());
        self
    }

    /// Adds `R` to the write set.
    pub fn writes<R: Any + Send + Sync>(mut self) -> Self {
        self.writes.push(TypeId::of::<R>());
        self
    }

    /// Adds the component lane `T` to the read set.
    pub fn reads_lane<T: 'static + Send + Sync>(mut self) -> Self {
        self.reads_lanes.push(TypeId::of::<T>());
        self
    }

    /// Adds the component lane `T` to the write set.
    pub fn writes_lane<T: 'static + Send + Sync>(mut self) -> Self {
        self.writes_lanes.push(TypeId::of::<T>());
        self
    }

    /// [`SystemAccess::reads_lane`] by known `TypeId` — for dynamic
    /// frontends via the registry ([`ComponentMeta::type_id`]).
    ///
    /// [`ComponentMeta::type_id`]: crate::ComponentMeta::type_id
    pub fn reads_lane_id(mut self, id: TypeId) -> Self {
        self.reads_lanes.push(id);
        self
    }

    /// [`SystemAccess::writes_lane`] by known `TypeId`.
    pub fn writes_lane_id(mut self, id: TypeId) -> Self {
        self.writes_lanes.push(id);
        self
    }

    /// Whether a write conflicts with (read ∪ write) of another access.
    #[cfg(test)]
    fn writes_touch(&self, other: &SystemAccess) -> bool {
        self.writes
            .iter()
            .any(|w| other.reads.contains(w) || other.writes.contains(w))
    }
}

/// Level scheduler key: singleton resources and component lanes are two
/// separate namespaces. The same type can be a resource in one system and
/// a component in another; without separate namespaces this would cause
/// a false conflict (audit §3.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AccessKey {
    Resource(TypeId),
    Lane(TypeId),
}

impl SystemAccess {
    /// Projects reads into scheduler keys (resources + lanes).
    fn read_keys(&self) -> Vec<AccessKey> {
        self.reads
            .iter()
            .map(|&id| AccessKey::Resource(id))
            .chain(self.reads_lanes.iter().map(|&id| AccessKey::Lane(id)))
            .collect()
    }

    /// Projects writes into scheduler keys (resources + lanes).
    fn write_keys(&self) -> Vec<AccessKey> {
        self.writes
            .iter()
            .map(|&id| AccessKey::Resource(id))
            .chain(self.writes_lanes.iter().map(|&id| AccessKey::Lane(id)))
            .collect()
    }
}

/// Unified scheduler system: accesses are data, execution is over the world.
pub trait System: Send + Sync {
    /// Unique system name for diagnostics and ordering.
    fn name(&self) -> &'static str;
    /// Declares the resources and lanes this system touches.
    fn access(&self) -> SystemAccess;
    /// Runs the system against the shared resources.
    fn run(&self, resources: &Resources);
}

/// Conflict of two systems (in registration order i < j): ordered if
/// i writes what j reads/writes (RaW/WaW), or j writes what i reads
/// (anti-dependency, WaR).
#[cfg(test)]
fn conflicts(a: &SystemAccess, b: &SystemAccess) -> bool {
    a.writes_touch(b) || b.writes_touch(a)
}

/// Parallelism levels: systems without conflicts share a level;
/// levels are ordered by dependencies. Deterministic by registration order.
///
/// Reference Vec implementation: the production [`Schedule`] path uses a
/// bitset projection of the same conflicts; the
/// `bitset_plan_matches_reference_model` test pins equivalence.
#[cfg(test)]
fn reference_level_groups(
    accesses: &[SystemAccess],
    ordering: &[(usize, usize)],
) -> Vec<Vec<usize>> {
    compute_levels(accesses.len(), |i, j| {
        conflicts(&accesses[i], &accesses[j]) || ordering.contains(&(i, j))
    })
}

/// Active access declaration of the executing system (enforcement).
///
/// `Clone` — the top frame snapshot is carried into child parallel tasks
/// of the system ([`AccessFrameCapture`], audit §3.3, backlog #7).
#[derive(Clone)]
struct AccessFrame {
    system: &'static str,
    reads: Vec<TypeId>,
    writes: Vec<TypeId>,
    reads_lanes: Vec<TypeId>,
    writes_lanes: Vec<TypeId>,
}

thread_local! {
    /// Stack of active declarations: empty outside `Schedule::run` (or when
    /// enforcement is off) — then `Resources` is unrestricted.
    static ACCESS_FRAMES: RefCell<Vec<AccessFrame>> = const { RefCell::new(Vec::new()) };
}

/// RAII push of the declaration onto the thread-local `ACCESS_FRAMES`
/// stack; popped on scope exit, including panic unwinding (rayon reuses
/// threads — without RAII the stack would go stale).
///
/// The guard belongs to scheduler enforcement: it is created only at the
/// two push sites (`Schedule::run_system`, [`AccessFrameCapture::install`]);
/// manual construction would break push/pop pairing.
pub struct AccessFrameGuard;

impl AccessFrameGuard {
    fn push(system: &'static str, access: &SystemAccess) -> Self {
        ACCESS_FRAMES.with(|frames| {
            frames.borrow_mut().push(AccessFrame {
                system,
                reads: access.reads.clone(),
                writes: access.writes.clone(),
                reads_lanes: access.reads_lanes.clone(),
                writes_lanes: access.writes_lanes.clone(),
            });
        });
        AccessFrameGuard
    }
}

impl Drop for AccessFrameGuard {
    fn drop(&mut self) {
        ACCESS_FRAMES.with(|frames| {
            frames.borrow_mut().pop();
        });
    }
}

/// Snapshot of the top active access declaration for carrying enforcement
/// into a system's parallel child tasks (audit §3.3, backlog #7).
///
/// `#[smart_pipeline]` generates the capture and install around
/// `par_iter` loop bodies automatically. For manual parallelism inside
/// `System::run` (`rayon::scope`/`spawn`, `std::thread::scope`): capture
/// before entering the parallel section, install in each child task;
/// then the child thread inherits the system declaration and an
/// undeclared access panics just as in the system thread.
///
/// `Send + Sync` (&'static str + TypeId): the snapshot is shared by
/// reference in rayon `for_each` closures. Empty outside
/// [`Schedule::run`] (or with access enforcement off) — then
/// [`install`](Self::install) returns `None` and child thread behavior
/// is unchanged.
pub struct AccessFrameCapture(Option<AccessFrame>);

/// Captures the top active access declaration — the entry point
/// described at [`AccessFrameCapture`].
pub fn capture_access_frame() -> AccessFrameCapture {
    ACCESS_FRAMES.with(|frames| AccessFrameCapture(frames.borrow().last().cloned()))
}

impl AccessFrameCapture {
    /// Installs the captured declaration into the TLS of the current
    /// (worker) thread; the guard removes it on scope exit, including
    /// panic unwinding. Empty snapshot → `None`: one TLS load, zero
    /// allocations.
    pub fn install(&self) -> Option<AccessFrameGuard> {
        let frame = self.0.clone()?;
        ACCESS_FRAMES.with(|frames| frames.borrow_mut().push(frame));
        Some(AccessFrameGuard)
    }
}

/// Checks `R` against the declaration of the currently executing system;
/// outside [`Schedule::run`] — no-op. A violation is a breach of the
/// determinism contract: an undeclared access can race with systems on
/// the same parallel level.
fn assert_access_declared<R: Any + Send + Sync>() {
    ACCESS_FRAMES.with(|frames| {
        let frames = frames.borrow();
        let Some(frame) = frames.last() else {
            return;
        };
        let id = TypeId::of::<R>();
        let declared = frame.reads.contains(&id) || frame.writes.contains(&id);
        if !declared {
            panic!(
                "system '{}' reads resource '{}' that is not declared in its access set \
                 (SystemAccess::reads/writes) — undeclared access breaks the deterministic \
                 schedule contract",
                frame.system,
                std::any::type_name::<R>()
            );
        }
    });
}

/// Checks the component lane `T` against the declaration of the
/// currently executing system — the `SmartStore::read_lane`/`write_lane`
/// boundary (audit §3.6). Outside [`Schedule::run`] (and with enforcement
/// off the stack is empty) — no-op. Reads are covered by `reads_lane` or
/// `writes_lane`; writes strictly by `writes_lane`: the write guard
/// proves intent to write, unlike resources where mutation happens via
/// invisible interior mutability. Cold and lock-free lanes are separate
/// namespaces and not covered by this protocol (experimental mechanisms).
pub(crate) fn assert_lane_access_declared<T: 'static>(for_write: bool) {
    ACCESS_FRAMES.with(|frames| {
        let frames = frames.borrow();
        let Some(frame) = frames.last() else {
            return;
        };
        let id = TypeId::of::<T>();
        let declared = if for_write {
            frame.writes_lanes.contains(&id)
        } else {
            frame.reads_lanes.contains(&id) || frame.writes_lanes.contains(&id)
        };
        if !declared {
            let verb = if for_write { "writes" } else { "reads" };
            panic!(
                "system '{}' {} lane '{}' that is not declared in its access set \
                 (SystemAccess::reads_lane/writes_lane) — undeclared access breaks the \
                 deterministic schedule contract",
                frame.system,
                verb,
                std::any::type_name::<T>()
            );
        }
    });
}

/// System schedule: registration order is the conflict tie-break.
pub struct Schedule {
    systems: Vec<Box<dyn System>>,
    accesses: Vec<SystemAccess>,
    /// S5c: explicit ordering edges (registration indices i < j) on top
    /// of inferred access edges — e.g., for hidden dependencies (shared
    /// queue buffers) invisible in access sets.
    ordering: Vec<(usize, usize)>,
    parallel: bool,
    enforce_accesses: bool,
    /// Level plan cache with diagnostics (mirrors the render
    /// `FrameLayout` S1 cache): recomputed only after
    /// `add_system`/`order_before`.
    plan: PlanCache,
}

impl Default for Schedule {
    fn default() -> Self {
        Self::new()
    }
}

impl Schedule {
    /// Creates an empty schedule with parallel execution enabled.
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            accesses: Vec::new(),
            ordering: Vec::new(),
            parallel: true,
            enforce_accesses: cfg!(debug_assertions),
            plan: PlanCache::new(),
        }
    }

    /// Adds a system (order = priority on conflicts).
    pub fn add_system<S: System + 'static>(&mut self, system: S) -> &mut Self {
        self.accesses.push(system.access());
        self.systems.push(Box::new(system));
        self.invalidate_plan();
        self
    }

    /// Inserts a system before all already registered systems.
    ///
    /// Used by platform adapters that need to place domain systems before
    /// an already installed render/extract pass. Explicit edges are
    /// automatically shifted by one index; registration order of the
    /// remaining systems is preserved.
    pub fn prepend_system<S: System + 'static>(&mut self, system: S) -> &mut Self {
        self.accesses.insert(0, system.access());
        self.systems.insert(0, Box::new(system));
        self.ordering = self
            .ordering
            .iter()
            .map(|&(before, after)| (before.saturating_add(1), after.saturating_add(1)))
            .collect();
        self.invalidate_plan();
        self
    }

    fn invalidate_plan(&mut self) {
        self.plan.invalidate();
    }

    /// S5c: declares that system `before` must execute before system
    /// `after`, even if their accesses do not conflict (hidden
    /// dependency). Both are looked up by `name()`.
    ///
    /// # Panics
    /// Panics if a name is not found (name uniqueness is the caller's
    /// responsibility) or `after` is registered before `before`:
    /// execution order is registration order (S3), explicit edges only
    /// split levels.
    pub fn order_before(&mut self, before: &str, after: &str) -> &mut Self {
        self.try_order_before(before, after)
            .unwrap_or_else(|error| panic!("order_before('{before}', '{after}'): {error}"))
    }

    /// Fallible [`Schedule::order_before`]: returns [`OrderError`] on
    /// failure instead of panicking (for phase-6 dynamic frontends).
    pub fn try_order_before(&mut self, before: &str, after: &str) -> Result<&mut Self, OrderError> {
        let (b, a) = resolve_named_edge(before, after, |name| self.try_system_index(name))?;
        if !self.ordering.contains(&(b, a)) {
            self.ordering.push((b, a));
            self.invalidate_plan();
        }
        Ok(self)
    }

    /// S5c: mirror of [`Schedule::order_before`].
    pub fn order_after(&mut self, after: &str, before: &str) -> &mut Self {
        self.order_before(before, after)
    }

    /// Fallible [`Schedule::order_after`].
    pub fn try_order_after(&mut self, after: &str, before: &str) -> Result<&mut Self, OrderError> {
        self.try_order_before(before, after)
    }

    fn try_system_index(&self, name: &str) -> Option<usize> {
        self.systems.iter().position(|sys| sys.name() == name)
    }

    /// Parallel (true, default) or strictly sequential (bit-identical
    /// registration order) execution.
    pub fn set_parallel(&mut self, parallel: bool) -> &mut Self {
        self.parallel = parallel;
        self
    }

    /// Enforcement of declared accesses (see module docs): enabled by
    /// default in debug builds, disabled in release.
    pub fn set_enforce_accesses(&mut self, enforce: bool) -> &mut Self {
        self.enforce_accesses = enforce;
        self
    }

    /// Number of systems.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// Whether the schedule is empty.
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// Parallelism levels (system indices in registration order).
    /// One concept under one name on both engine frontends (mirrors
    /// render's `FrameLayout::levels`; canonical, backlog #19).
    pub fn levels(&self) -> Vec<Vec<usize>> {
        self.cached_levels()
    }

    /// How many times the level plan was recomputed — cache diagnostics
    /// (does not grow in steady state).
    pub fn level_computations(&self) -> usize {
        self.plan.computations()
    }

    /// Mermaid projection of the level plan — debug diagram over the
    /// shared [`MermaidDiagram`] projector (slice 1b of the graph-
    /// elimination approach, PLAN.md App. C): systems as `S{i}` nodes in
    /// level subgraphs, explicit `order_before` edges as flow arrows.
    /// The same picture `FrameLayout::mermaid` draws for render passes;
    /// GitHub renders ```mermaid natively, so a schedule dump in review
    /// becomes an image of the system pipeline.
    pub fn mermaid(&self) -> String {
        let mut d = MermaidDiagram::new();
        for (li, level) in self.levels().iter().enumerate() {
            let nodes: Vec<(String, String)> = level
                .iter()
                .map(|&si| (format!("S{si}"), self.systems[si].name().to_string()))
                .collect();
            d.level(&format!("L{li}"), &format!("level {li}"), &nodes);
        }
        for &(a, b) in &self.ordering {
            d.edge(&format!("S{a}"), &format!("S{b}"));
        }
        d.render()
    }

    fn cached_levels(&self) -> Vec<Vec<usize>> {
        let reads: Vec<Vec<AccessKey>> =
            self.accesses.iter().map(SystemAccess::read_keys).collect();
        let writes: Vec<Vec<AccessKey>> =
            self.accesses.iter().map(SystemAccess::write_keys).collect();
        self.plan
            .get_or_compute(|| bitset_level_plan(&reads, &writes, &self.ordering))
    }

    /// Executes the schedule over the world.
    pub fn run(&self, resources: &Resources) {
        // Sequential mode does not compute the plan at all (as before).
        let levels = if self.parallel {
            self.cached_levels()
        } else {
            Vec::new()
        };
        run_levels(&levels, self.systems.len(), self.parallel, |i| {
            self.run_system(i, resources);
        });
    }

    fn run_system(&self, i: usize, resources: &Resources) {
        let system = &self.systems[i];
        if self.enforce_accesses {
            let _frame = AccessFrameGuard::push(system.name(), &self.accesses[i]);
            system.run(resources);
        } else {
            system.run(resources);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct A;
    struct B;
    struct C;

    fn reads_writes<R: Any + Send + Sync>(reads: bool, writes: bool) -> SystemAccess {
        let mut a = SystemAccess::new();
        if reads {
            a = a.reads::<R>();
        }
        if writes {
            a = a.writes::<R>();
        }
        a
    }

    /// Test stub system with a fixed name and accesses.
    struct NamedNoop(&'static str, SystemAccess);

    impl System for NamedNoop {
        fn name(&self) -> &'static str {
            self.0
        }
        fn access(&self) -> SystemAccess {
            self.1.clone()
        }
        fn run(&self, _: &Resources) {}
    }

    #[test]
    fn resources_singleton_per_type() {
        let mut res = Resources::new();
        assert!(res.is_empty());
        res.insert(7u32);
        assert_eq!(res.insert(9u32), Some(7));
        assert_eq!(res.get::<u32>(), Some(&9));
        *res.get_mut::<u32>().unwrap() = 11;
        assert_eq!(res.remove::<u32>(), Some(11));
        assert!(!res.contains::<u32>());
    }

    #[test]
    fn diamond_yields_parallel_level() {
        // A writes X; B and C read X (write Y/Z); D reads Y and Z.
        let accesses = vec![
            reads_writes::<A>(false, true),
            reads_writes::<B>(false, true),
            reads_writes::<C>(false, true),
        ];
        // independent writers → one level
        assert_eq!(reference_level_groups(&accesses, &[]), vec![vec![0, 1, 2]]);

        let chain = vec![
            SystemAccess::new().writes::<A>(),
            SystemAccess::new().reads::<A>().writes::<B>(),
            SystemAccess::new().reads::<B>().writes::<C>(),
        ];
        assert_eq!(
            reference_level_groups(&chain, &[]),
            vec![vec![0], vec![1], vec![2]]
        );
    }

    #[test]
    fn anti_dependency_orders_reader_first() {
        // i reads X, j writes X → i before j (anti-dependency).
        let accesses = vec![
            SystemAccess::new().reads::<A>(),
            SystemAccess::new().writes::<A>(),
        ];
        assert_eq!(
            reference_level_groups(&accesses, &[]),
            vec![vec![0], vec![1]]
        );
    }

    struct Bump {
        name: &'static str,
        access: SystemAccess,
        target: &'static str,
    }

    impl System for Bump {
        fn name(&self) -> &'static str {
            self.name
        }
        fn access(&self) -> SystemAccess {
            self.access.clone()
        }
        fn run(&self, resources: &Resources) {
            let counter = resources
                .get::<Mutex<Vec<&'static str>>>()
                .expect("log resource");
            let mut log = counter.lock().expect("log lock");
            log.push(self.target);
        }
    }

    /// Log resource type for tests below (declared as a read).
    type Log = Mutex<Vec<&'static str>>;

    #[test]
    fn sequential_mode_is_registration_order() {
        let mut res = Resources::new();
        res.insert(Log::default());
        let mut sched = Schedule::new();
        sched
            .add_system(Bump {
                name: "w",
                access: SystemAccess::new().writes::<A>().reads::<Log>(),
                target: "w",
            })
            .add_system(Bump {
                name: "r",
                access: SystemAccess::new()
                    .reads::<A>()
                    .writes::<B>()
                    .reads::<Log>(),
                target: "r",
            });
        sched.set_parallel(false);
        sched.run(&res);
        let log = res.get::<Log>().unwrap();
        assert_eq!(*log.lock().unwrap(), vec!["w", "r"]);
    }

    /// Strong Confluence: a parallel run produces the same result as a
    /// sequential one, regardless of thread count.
    #[test]
    fn parallel_matches_sequential() {
        #[derive(Default)]
        struct Counters {
            sum: AtomicU64,
            events: Mutex<Vec<&'static str>>,
        }
        struct Add {
            access: SystemAccess,
            name: &'static str,
        }
        impl System for Add {
            fn name(&self) -> &'static str {
                self.name
            }
            fn access(&self) -> SystemAccess {
                self.access.clone()
            }
            fn run(&self, resources: &Resources) {
                let c = resources.get::<Counters>().expect("counters");
                c.sum.fetch_add(1, Ordering::Relaxed);
                c.events.lock().unwrap().push(self.name);
            }
        }

        let build = |parallel: bool| -> (u64, Vec<&'static str>) {
            let mut res = Resources::new();
            res.insert(Counters::default());
            let mut sched = Schedule::new();
            // diamond: source → two independent nodes → sink
            sched
                .add_system(Add {
                    access: SystemAccess::new().writes::<A>().reads::<Counters>(),
                    name: "src",
                })
                .add_system(Add {
                    access: SystemAccess::new()
                        .reads::<A>()
                        .writes::<B>()
                        .reads::<Counters>(),
                    name: "left",
                })
                .add_system(Add {
                    access: SystemAccess::new()
                        .reads::<A>()
                        .writes::<C>()
                        .reads::<Counters>(),
                    name: "right",
                })
                .add_system(Add {
                    access: SystemAccess::new()
                        .reads::<B>()
                        .reads::<C>()
                        .writes::<A>()
                        .reads::<Counters>(),
                    name: "sink",
                });
            sched.set_parallel(parallel);
            sched.run(&res);
            let c = res.remove::<Counters>().unwrap();
            let mut events = c.events.into_inner().unwrap();
            events.sort();
            (c.sum.into_inner(), events)
        };

        let (seq_sum, seq_events) = build(false);
        let (par_sum, par_events) = build(true);
        assert_eq!(seq_sum, 4);
        assert_eq!(par_sum, 4);
        // Event multisets coincide; order within a level is
        // non-deterministic — hence we compare sorted.
        assert_eq!(seq_events, par_events);
    }

    #[test]
    fn prepend_system_preserves_explicit_edges_and_registration_order() {
        let mut sched = Schedule::new();
        sched
            .add_system(NamedNoop("source", SystemAccess::new().writes::<A>()))
            .add_system(NamedNoop("sink", SystemAccess::new().reads::<A>()));
        sched.order_before("source", "sink");
        sched.prepend_system(NamedNoop("prefix", SystemAccess::new().writes::<B>()));

        assert_eq!(sched.levels(), vec![vec![0, 1], vec![2]]);
        assert!(sched.try_order_before("prefix", "source").is_ok());
        assert_eq!(sched.levels(), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn explicit_ordering_splits_a_level() {
        // Two independent writers share a level; an explicit edge separates them.
        let mut sched = Schedule::new();
        sched
            .add_system(NamedNoop("first", SystemAccess::new().writes::<A>()))
            .add_system(NamedNoop("second", SystemAccess::new().writes::<B>()));
        assert_eq!(sched.levels(), vec![vec![0, 1]]);
        sched.order_before("first", "second");
        assert_eq!(sched.levels(), vec![vec![0], vec![1]]);
    }

    #[test]
    #[should_panic(expected = "registered")]
    fn explicit_ordering_rejects_backward_direction() {
        let mut sched = Schedule::new();
        sched
            .add_system(NamedNoop("a", SystemAccess::new()))
            .add_system(NamedNoop("b", SystemAccess::new()));
        sched.order_before("b", "a");
    }

    #[test]
    #[should_panic(expected = "no node named")]
    fn explicit_ordering_unknown_name_panics() {
        let mut sched = Schedule::new();
        sched.add_system(NamedNoop("only", SystemAccess::new()));
        sched.order_before("only", "ghost");
    }

    #[test]
    fn try_order_before_reports_errors_without_panicking() {
        let mut sched = Schedule::new();
        sched
            .add_system(NamedNoop("a", SystemAccess::new().writes::<A>()))
            .add_system(NamedNoop("b", SystemAccess::new().writes::<B>()));
        assert!(matches!(
            sched.try_order_before("b", "a"),
            Err(OrderError::BackwardEdge { .. })
        ));
        assert_eq!(
            sched.try_order_before("a", "ghost").map(|_| ()),
            Err(OrderError::UnknownNode {
                name: "ghost".to_owned(),
            })
        );
        // The plan is untouched by errors; a successful edge splits the level.
        assert_eq!(sched.levels(), vec![vec![0, 1]]);
        assert!(sched.try_order_before("a", "b").is_ok());
        assert_eq!(sched.levels(), vec![vec![0], vec![1]]);
    }

    #[test]
    fn schedule_levels_diamond() {
        let mut sched = Schedule::new();
        sched
            .add_system(NamedNoop("src", SystemAccess::new().writes::<A>()))
            .add_system(NamedNoop(
                "left",
                SystemAccess::new().reads::<A>().writes::<B>(),
            ))
            .add_system(NamedNoop(
                "right",
                SystemAccess::new().reads::<A>().writes::<C>(),
            ))
            .add_system(NamedNoop(
                "sink",
                SystemAccess::new().reads::<B>().reads::<C>().writes::<A>(),
            ));
        assert_eq!(sched.levels(), vec![vec![0], vec![1, 2], vec![3]]);
    }

    #[test]
    fn mermaid_projects_levels_and_order_edges() {
        // Slice 1b: projection over the shared ornis_schedule::MermaidDiagram —
        // systems as nodes in level subgraphs, order_before edges as arrows.
        let mut sched = Schedule::new();
        sched
            .add_system(NamedNoop("writer", SystemAccess::new().writes::<A>()))
            .add_system(NamedNoop("free", SystemAccess::new().writes::<B>()))
            .add_system(NamedNoop("reader", SystemAccess::new().reads::<A>()));
        assert!(sched.try_order_before("free", "reader").is_ok());
        // writer ∥ free on level zero, reader on level one — by conflict
        // (RaW from writer) and by explicit edge (from free).
        assert_eq!(sched.levels(), vec![vec![0, 1], vec![2]]);
        let m = sched.mermaid();
        assert!(m.starts_with("flowchart TD\n"), "head: {m}");
        assert!(m.contains("subgraph L0[\"level 0\"]"), "levels: {m}");
        assert!(m.contains("S0[\"writer\"]"), "system nodes: {m}");
        assert!(m.contains("S2[\"reader\"]"), "system nodes: {m}");
        assert!(m.contains("S1 --> S2"), "order edges: {m}");
    }

    #[test]
    fn level_plan_is_cached_until_mutation() {
        let mut sched = Schedule::new();
        for name in ["a", "b", "c"] {
            sched.add_system(NamedNoop(name, SystemAccess::new().writes::<A>()));
        }
        let res = Resources::new();
        assert_eq!(sched.level_computations(), 0);
        sched.run(&res);
        assert_eq!(sched.level_computations(), 1);
        sched.run(&res);
        sched.run(&res);
        assert_eq!(
            sched.level_computations(),
            1,
            "steady state reuses the cached plan"
        );
        sched.add_system(NamedNoop("d", SystemAccess::new().reads::<A>()));
        sched.run(&res);
        assert_eq!(
            sched.level_computations(),
            2,
            "add_system invalidates the plan"
        );
        assert_eq!(sched.levels(), vec![vec![0], vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn bitset_plan_matches_reference_model() {
        // Pseudo-random access sets (LCG): the bitset plan must match the
        // reference Vec implementation (conflicts → levels).
        let mut lcg = 0x5EED_600Du64;
        let mut next = move || {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lcg
        };
        const NAMES: [&str; 12] = [
            "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
        ];
        let mut sched = Schedule::new();
        let mut accesses: Vec<SystemAccess> = Vec::new();
        for &name in NAMES.iter() {
            let mut access = SystemAccess::new();
            if next() % 2 == 0 {
                access = access.reads::<A>();
            }
            if next() % 3 == 0 {
                access = access.reads::<B>();
            }
            if next() % 2 == 0 {
                access = access.writes::<C>();
            }
            if next() % 3 == 0 {
                access = access.writes::<A>();
            }
            sched.add_system(NamedNoop(name, access.clone()));
            accesses.push(access);
        }
        assert_eq!(
            sched.levels(),
            reference_level_groups(&accesses, &[]),
            "bitset plan must match the reference model without explicit edges"
        );
        sched.order_before("s2", "s8");
        assert_eq!(
            sched.levels(),
            reference_level_groups(&accesses, &[(2, 8)]),
            "bitset plan must match the reference model with an explicit edge"
        );
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn undeclared_read_panics_under_enforcement() {
        struct Sneaky;
        impl System for Sneaky {
            fn name(&self) -> &'static str {
                "sneaky"
            }
            fn access(&self) -> SystemAccess {
                SystemAccess::new().writes::<A>()
            }
            fn run(&self, resources: &Resources) {
                // Reads B without declaring it — contract violation.
                let _ = resources.get::<B>();
            }
        }
        let mut res = Resources::new();
        res.insert(B);
        let mut sched = Schedule::new();
        sched.add_system(Sneaky).set_enforce_accesses(true);
        sched.run(&res);
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn undeclared_access_in_child_thread_panics_with_captured_frame() {
        // Backlog #7 (audit §3.3): a system child thread starts with an
        // empty TLS stack — a historic enforcement gap; capture plus
        // frame install close it. `std::thread::scope` guarantees a
        // foreign thread (a rayon task could run on the caller where
        // TLS is already populated); join catches the child panic and
        // `resume_unwind` re-raises its payload with the original message.
        struct SneakySpawn;
        impl System for SneakySpawn {
            fn name(&self) -> &'static str {
                "sneaky_spawn"
            }
            fn access(&self) -> SystemAccess {
                SystemAccess::new().writes::<A>()
            }
            fn run(&self, resources: &Resources) {
                let frame = capture_access_frame();
                std::thread::scope(|scope| {
                    let handle = scope.spawn(|| {
                        let _guard = frame.install();
                        // Reads B without declaring it — from the child thread.
                        let _ = resources.get::<B>();
                    });
                    // Auto-join scope would re-panic with the wrapper "a scoped
                    // thread panicked" — we re-raise the original payload so
                    // should_panic sees the cause, not the wrapper.
                    if let Err(payload) = handle.join() {
                        std::panic::resume_unwind(payload);
                    }
                });
            }
        }
        let mut res = Resources::new();
        res.insert(B);
        let mut sched = Schedule::new();
        sched.add_system(SneakySpawn).set_enforce_accesses(true);
        sched.run(&res);
    }

    #[test]
    fn declared_access_in_child_thread_passes_with_captured_frame() {
        // Frame carry is not a violation: own-write read is declared,
        // the child thread passes the check (carry does not create false
        // positives).
        struct HonestSpawn;
        impl System for HonestSpawn {
            fn name(&self) -> &'static str {
                "honest_spawn"
            }
            fn access(&self) -> SystemAccess {
                SystemAccess::new().writes::<A>()
            }
            fn run(&self, resources: &Resources) {
                let frame = capture_access_frame();
                std::thread::scope(|scope| {
                    scope.spawn(|| {
                        let _guard = frame.install();
                        assert!(resources.get::<A>().is_some());
                    });
                });
            }
        }
        let mut res = Resources::new();
        res.insert(A);
        let mut sched = Schedule::new();
        sched.add_system(HonestSpawn).set_enforce_accesses(true);
        sched.run(&res);
    }

    #[test]
    fn capture_outside_schedule_run_is_noop() {
        // Outside `Schedule::run` (and with enforcement off) the stack is empty:
        // snapshot is empty, install → None, child thread behavior is
        // unchanged — the documented "off" path.
        let frame = capture_access_frame();
        assert!(frame.install().is_none());
    }

    #[test]
    fn declared_access_passes_enforcement() {
        struct Honest;
        impl System for Honest {
            fn name(&self) -> &'static str {
                "honest"
            }
            fn access(&self) -> SystemAccess {
                SystemAccess::new().reads::<A>().writes::<B>()
            }
            fn run(&self, resources: &Resources) {
                // B is declared as a write — own read is allowed.
                assert!(resources.get::<A>().is_some());
                assert!(resources.get::<B>().is_some());
            }
        }
        let mut res = Resources::new();
        res.insert(A);
        res.insert(B);
        let mut sched = Schedule::new();
        sched.add_system(Honest).set_enforce_accesses(true);
        sched.run(&res);
    }

    #[test]
    fn resources_are_unrestricted_outside_schedule() {
        let mut res = Resources::new();
        res.insert(B);
        // Outside Schedule::run there are no active declarations — access is unrestricted.
        assert!(res.contains::<B>());
        assert!(res.get::<B>().is_some());
    }
}
