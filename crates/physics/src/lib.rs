//! Ornis sequential-impulse rigid-body physics: dynamics, collision detection, contact
//! and joint solving.
//!
//! The crate is organized as a small CPU pipeline shared by the engine trait
//! ([`engine::PhysicsEngine`]) and its reference implementation
//! ([`engine::SequentialImpulseEngine`]):
//!
//! - [`body`] — rigid bodies and their handles/mass model.
//! - [`shape`] — collision shapes (sphere, box, capsule, cylinder, cone,
//!   convex hull, heightfield terrain, triangle-soup [`shape::TriMesh`]
//!   under a median-split AABB BVH) with AABB projection and inertia
//!   tensors; cylinder/cone/hull pairs resolve through the GJK/EPA
//!   fallback (`gjk` module).
//! - [`math`] — geometric queries used by broadphase and raycasts.
//! - `broadphase` — candidate-pair backends and benchmark diagnostics.
//! - [`joint`] — persistent equality constraints (ball, revolute,
//!   prismatic, fixed, distance, wheel, gear, six-DOF) with limits/motors.
//! - [`engine`] — the sequential-impulse step pipeline: broadphase → narrowphase → island
//!   partitioning → substepped velocity/position solving, with optional
//!   SIMD-wide (`wide` module) and GPU (`gpu` feature) solver paths.
//! - [`avbd`] — second engine behind the same seam: position-level AVBD
//!   sweep (single-thread) over exact `distance` queries.
//! - `migration`/`split` — scene snapshots and deterministic island
//!   ownership for solver switches and [`RoutingKind::Islands`] routing.
//! - [`Engine`] — solver orchestrator over [`SolverKind`]: one engine in
//!   [`RoutingKind::Single`], both engines under [`RoutingKind::Islands`];
//!   migrates scenes 1:1 and runs the cross-solver fracture pass.
#![warn(missing_docs)]

/// Midpoint / half-extent scale (tests and examples).
const HALF: f32 = 0.5;

/// Mesh/collider recipes → solver bodies (the single projection point).
pub mod colliders;
/// Collision detection: broadphase backends, shapes and distance queries.
pub mod collision;
/// Shared solver thresholds (effective-mass floor, degenerate length, …).
pub(crate) mod constants;
mod contact_math;
/// Typed physics failures (point 5: thiserror hierarchies).
pub mod errors;
/// Typed replacements for legacy `bool` flags (point 3: bool -> enum).
pub mod flags;
/// Scene snapshots and joint-remap helpers for solver migration.
pub mod migration;
mod split;

#[cfg(test)]
mod engine_policy_tests;

/// AVBD rigid-body engine: second [`engine::PhysicsEngine`] implementation
/// (M1, Genesis-style engine-level modularity).
pub mod avbd;
/// Rigid bodies: [`RigidBody`], mass model and body handles/types.
pub mod body;
/// Analytic closest-point and distance queries between convex shapes.
pub use collision::distance;
/// The physics step pipeline and the [`crate::engine::PhysicsEngine`] trait.
pub mod engine;
/// GJK/EPA fallback narrow phase for cylinder, cone and convex hull pairs.
pub(crate) use collision::gjk;
#[cfg(feature = "gpu")]
/// GPU sequential-impulse accelerator for wide contact batches (G7).
// The `#[gpu_pipeline]` macro emits undocumented `pub mod`s for its kernels
// (outer attributes do not propagate into the expansion), so the docs gate
// is relaxed for this subtree; every hand-written public item stays documented.
#[allow(missing_docs)]
pub mod gpu;
/// Invariant-preserving newtypes (mass, unit directions, capped manifolds).
pub mod invariants;
pub mod joint;
pub mod math;
/// Sequential-impulse solver internals (Genesis-style `solvers/rigid/` box).
pub mod sequential_impulse;
/// Collision shapes with AABB projection and inertia tensors.
pub use collision::shape;
/// Deformable bodies (PLAN B2/D1): particles and distance topologies.
pub mod soft;
/// Render-only tube soup for rope/chain soft bodies (PLAN B2/D1 leftover #3).
pub mod soft_render;
/// Particle self-collision for soft bodies (PLAN B2/D1 leftover #1).
pub(crate) mod soft_self;
/// Trigger overlap event types emitted by the sequential-impulse physics engine.
pub mod trigger;
/// SIMD-wide solver for single-point contact batches.
pub mod wide;
/// XPBD rigid-body engine: standalone [`engine::PhysicsEngine`]
/// implementation (Small-Steps substepping over compliant constraints).
pub mod xpbd;

use migration::{JointSnapshot, SceneSnapshot};
use split::{SplitBody, SplitJoint, SplitOwner, SplitState};
// Internal paths kept stable after the `collision/` move: `broadphase` and
// `broadphase_tree` were (and stay) crate-private, reachable as before.
pub(crate) use collision::{broadphase, broadphase_tree};

pub use avbd::AvbdEngine;
pub use body::{BodyHandle, BodyType, LocalAvbdBody, LocalSiBody, RigidBody};
pub use collision::broadphase::{BroadPhaseKind, BroadPhaseStats, StepBudget, StepTiming};
pub use engine::{PhysicsEngine, SequentialImpulseEngine};
pub use errors::{ColliderError, JointError, MeshError, QueryError};
pub use flags::{
    AxisStatus, BodyRole, CachePolicy, CoordKind, Dispatch, HitKind, LimitSide, Order,
    RestitutionGate, RollAxis, RoutePhase, SolvePath, SolverSide, StructuralState,
};
pub use invariants::{
    Capped4, FrictionFrame, FrictionFrameError, HeightfieldError, Mass, MassKind, Meters,
    NonEmpty4, PositiveF32, Radians, UnitVec3, validate_heightfield,
};
pub use joint::{
    AxisConfig, CrossRowKind, JointHandle, JointKind, LocalAvbdJoint, LocalSiJoint, PrismaticLimit,
    PrismaticMotor, ResolvedJoint, RevoluteLimit, RevoluteMotor, WheelSuspension, cross_row_kind,
    resolve_joint,
};
pub use math::{AABB, Ray, RaycastHit};
pub use shape::{ConvexHull, Heightfield, PairSupport, Shape, TriIndex, TriMesh, Triangle};
pub use soft::{
    ClothPin, DeformConstraint, DeformKind, Particle, ParticleIdx, SoftBody, SoftHandle,
};
pub use soft_render::{tube_indices, tube_positions};
pub use trigger::{
    CONTACT_BEGIN_SLOP, CONTACT_HIT_THRESHOLD, ContactEvent, ContactEventKind, FractureEvent,
    TriggerEvent, TriggerEventKind,
};
pub use xpbd::XpbdEngine;

/// Selectable constraint solver (M2 intra-engine modularity, Genesis
/// style): the sequential-impulse engine, the AVBD engine, or the XPBD
/// engine (which additionally owns soft bodies — see [`Engine`]).
/// Same [`PhysicsEngine`] seam, same scenes — the orchestrator
/// ([`Engine`]) migrates bodies and joints across the switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    /// Sequential-impulse engine with islands, sleep and substeps.
    SequentialImpulse,
    /// AVBD engine (position-level, single-thread sweep).
    Avbd,
    /// XPBD engine (Small-Steps substepping over compliant constraints).
    /// The only path that steps soft bodies: a world with soft bodies
    /// migrates wholly onto it, so rigid + soft share one substep loop and
    /// soft↔rigid coupling actually runs. Rigid-only worlds are unaffected
    /// (they never touch this variant unless the host asks for it).
    Xpbd,
}

/// Routing policy for the M3 multi-solver scene (validated by spikes
/// 001–004, see `spikes/`): one solver per contact island, never
/// teleported cross-solver mirrors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingKind {
    /// One engine owns the whole scene (M1/M2 behavior, default).
    Single,
    /// Contact islands route between SequentialImpulse and AVBD with hysteresis
    /// (unanimous-island calm migrates down, any fast body wakes up).
    Islands,
}

/// Wall-clock cost of the last Islands host step (diagnostic only).
#[derive(Debug, Clone, Copy, Default)]
pub struct SplitTiming {
    /// Conservative island discovery and ownership decisions.
    pub routing: std::time::Duration,
    /// Body/joint/event-baseline reconstruction.
    pub rebuild: std::time::Duration,
    /// Time inside AVBD, summed over completed fixed substeps.
    pub avbd: std::time::Duration,
    /// Time inside the sequential-impulse solver, summed over completed fixed substeps.
    pub si: std::time::Duration,
    /// Time inside the XPBD solver, summed over completed fixed substeps
    /// (zero while no body is XPBD-pinned — the engine is then not stepped).
    pub xpbd: std::time::Duration,
}

/// How one registry joint is solved under [`RoutingKind::Islands`]
/// routing (see [`Engine::cross_joint_status`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossJointStatus {
    /// Both ends in one solver (or [`RoutingKind::Single`] mode): solved
    /// natively by the owning engine.
    Native,
    /// Ends in different solvers, structural kind (ball/distance): solved
    /// by the post-step coupling pass ([`CrossRowKind`]).
    Coupled(CrossRowKind),
    /// Not solved anywhere: a cross joint without a structural row, an
    /// XPBD-unsupported native kind, or an unresolvable gear. The detail
    /// names the cause — such joints are never silently dropped.
    Unsupported {
        /// Human-readable cause (joint kind + solver placement).
        detail: String,
    },
}

/// M3 coupling metrics: the coupling tax, in the books per PLAN M3.
/// `None` from [`Engine::split_metrics`] outside Islands routing.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplitMetrics {
    /// Rebuild migrations triggered by routing decisions (structural
    /// add/remove rebuilds are not counted).
    pub migrations: u64,
    /// Dynamic bodies currently owned by AVBD.
    pub avbd_bodies: usize,
    /// Dynamic bodies currently owned by SequentialImpulse.
    pub si_bodies: usize,
    /// Dynamic bodies currently owned by XPBD (host-pinned only —
    /// hysteresis never assigns this side).
    pub xpbd_bodies: usize,
    /// All rebuilds, including structural edits and initial construction.
    pub rebuilds: u64,
    /// Number of bodies whose owner changed (not just rebuild count).
    pub migrated_bodies: u64,
    /// Completed common 1/60 s simulation steps since Islands was enabled.
    pub simulation_steps: u64,
    /// Last host call's measured cost; never an input to routing.
    pub timing: SplitTiming,
}

/// Solver orchestrator: owns one engine behind the [`PhysicsEngine`] seam
/// and migrates the full scene (bodies 1:1 in handle order, then joints)
/// across [`SolverKind`] switches, so handles cached by the host stay
/// valid. Warm-start state (contact lambdas, island sleep) does not
/// migrate — only poses, velocities and joint specs do.
///
/// The orchestrator also owns cross-solver policy: the [`FractureEvent`]
/// pass runs inside [`PhysicsEngine::step`] (uniform for both solvers)
/// and its reports wait in [`Engine::drain_fracture_events`].
///
/// M3 adds an opt-in second mode ([`RoutingKind::Islands`]): both engines
/// stay alive and whole contact islands route between them with
/// hysteresis. Global [`BodyHandle`]s stay stable (fixed-order registry);
/// solver-local handles are an internal detail.
///
/// Cross-solver joints (v1): a joint whose dynamic ends are owned by
/// different solvers. Hysteresis itself never splits a joint — joint edges
/// still pin their island into one solver, so unpinned scenes behave
/// exactly as M3 and routing cannot thrash on a cross edge. A split arises
/// only through [`Engine::pin_body_solver`]: a pinned body keeps its owner
/// across routing ticks, so a joint between differently-pinned bodies stays
/// cross-solver until the host unpins. Each tick, after every engine steps
/// and before the registry sync, the coupling pass solves each cross ball
/// or distance joint as one positional projection plus one velocity row
/// directly between the live mirrors, in canonical global-joint order,
/// with mass-restore on both sides (sleep must not read as infinite mass).
/// Every other cross kind is never half-solved: it stays unmirrored and
/// [`Engine::cross_joint_status`] reports it as [`CrossJointStatus::Unsupported`]
/// with a named cause. Out of scope for v1: per-substep coupling frequency
/// and cross-solver contacts (pairs across tables would need an O(n²)
/// cross-AABB broadphase — routing keeps contact islands whole instead).
///
/// Soft bodies (D1, design (b): whole-world XPBD migration) live in the
/// [`SolverKind::Xpbd`] engine and step in its substep loop together with
/// the rigid bodies, so soft↔rigid coupling runs in one solver — there is
/// no second engine and no double-step of the same particles. Outside the
/// XPBD path (SequentialImpulse, AVBD, or Islands routing) soft bodies park
/// in the orchestrator in dense handle order: [`Engine::get_soft_body`]
/// keeps serving them and handles stay stable, but they do not step, sleep
/// or emit events until the world migrates back. A world gains the XPBD
/// path through [`Engine::set_solver_kind`]; the host from then on drives
/// soft bodies through the [`Engine::add_soft_body`] family, and reads
/// sleep/events through [`Engine::is_soft_asleep`] /
/// [`Engine::drain_soft_contact_events`].
pub struct Engine {
    inner: EngineInner,
    fracture_events: Vec<FractureEvent>,
    /// Contact events drained from the inner engine during `step`
    /// (the fracture pass consumes the inner queue, so the orchestrator
    /// re-serves them here — see `drain_contact_events`).
    contact_events: Vec<ContactEvent>,
    /// Globally remapped trigger transitions, independent of rebuilt locals.
    trigger_events: Vec<TriggerEvent>,
    /// Soft↔rigid begin/end transitions drained from the XPBD engine during
    /// `step` (see [`Engine::drain_soft_contact_events`]). Parked soft
    /// bodies emit nothing; structural edits clear the queue like the
    /// rigid one.
    soft_contact_events: Vec<xpbd::SoftContactEvent>,
    /// Soft bodies parked outside the XPBD path, in dense [`SoftHandle`]
    /// order (empty while the XPBD engine owns them live). Order is the
    /// handle: add appends, remove swap-removes, migrations preserve 1:1.
    parked_soft: Vec<SoftBody>,
    /// Parked soft↔rigid touch triples (see
    /// [`migration::SceneSnapshot::soft_touch`]). Cleared — never remapped —
    /// on any structural edit, mirroring the engines' own invalidation.
    parked_soft_touch: std::collections::BTreeSet<(usize, usize, usize)>,
    /// Soft-side Coulomb coefficient, applied to every fresh XPBD engine
    /// (solver tuning never migrates, so the orchestrator carries it).
    soft_friction: f32,
    /// Solver used in [`RoutingKind::Single`] (and the collapse target
    /// when leaving Islands). Unchanged by routing.
    single_kind: SolverKind,
    /// Active routing policy (M3). `Single` behaves exactly as M1/M2.
    routing: RoutingKind,
    /// Gravity this orchestrator was built (or switched) with.
    gravity: glam::Vec3,
    /// Live M3 registry, present only under Islands routing.
    split: Option<Box<SplitState>>,
    /// Structural changes (add/remove) pending engine rebuild. Single
    /// mode applies them immediately, so this only ever sets in Islands.
    structural_dirty: StructuralState,
    /// Globals touched through `get_body_mut` since the last step; the
    /// owner engine wakes them before stepping (host edits do not wake
    /// sleeping solvers by themselves).
    wake_set: std::collections::BTreeSet<BodyHandle>,
    /// Routing-triggered rebuilds since the last routing change.
    migrations: u64,
}

/// Active solver behind the seam (boxed: both states dwarf the policy).
enum EngineInner {
    /// Sequential-impulse engine with islands, sleep and substeps.
    SequentialImpulse(Box<SequentialImpulseEngine>),
    /// AVBD engine (position-level, single-thread sweep).
    Avbd(Box<AvbdEngine>),
    /// XPBD engine (Small-Steps compliant solve; owns soft bodies live).
    Xpbd(Box<XpbdEngine>),
}

impl Engine {
    /// Empty orchestrator with the given solver and world-space gravity.
    pub fn new(kind: SolverKind, gravity: glam::Vec3) -> Self {
        let inner = match kind {
            SolverKind::SequentialImpulse => {
                EngineInner::SequentialImpulse(Box::new(SequentialImpulseEngine::new(gravity)))
            }
            SolverKind::Avbd => EngineInner::Avbd(Box::new(AvbdEngine::new(gravity))),
            SolverKind::Xpbd => EngineInner::Xpbd(Box::new(XpbdEngine::new(gravity))),
        };
        Self {
            inner,
            fracture_events: Vec::new(),
            contact_events: Vec::new(),
            trigger_events: Vec::new(),
            soft_contact_events: Vec::new(),
            parked_soft: Vec::new(),
            parked_soft_touch: std::collections::BTreeSet::new(),
            soft_friction: HALF,
            gravity,
            single_kind: kind,
            routing: RoutingKind::Single,
            split: None,
            structural_dirty: StructuralState::Clean,
            wake_set: std::collections::BTreeSet::new(),
            migrations: 0,
        }
    }

    /// Active solver kind for [`RoutingKind::Single`] (and the collapse
    /// target when leaving Islands). Routing does not change it.
    pub fn kind(&self) -> SolverKind {
        self.single_kind
    }

    /// Active routing policy.
    pub fn routing(&self) -> RoutingKind {
        self.routing
    }

    /// M3 coupling metrics, or `None` outside Islands routing.
    pub fn split_metrics(&self) -> Option<SplitMetrics> {
        let s = self.split.as_ref()?;
        let (mut av, mut si, mut xp) = (0, 0, 0);
        for b in &s.bodies {
            match b.owner {
                SplitOwner::Avbd => av += 1,
                SplitOwner::SequentialImpulse => si += 1,
                SplitOwner::Xpbd => xp += 1,
                SplitOwner::Static => {}
            }
        }
        Some(SplitMetrics {
            migrations: self.migrations,
            avbd_bodies: av,
            si_bodies: si,
            xpbd_bodies: xp,
            rebuilds: s.rebuilds,
            migrated_bodies: s.migrated_bodies,
            simulation_steps: s.steps,
            timing: s.timing,
        })
    }

    /// Switch the active solver, migrating bodies and joints 1:1 (handles
    /// stay valid). Physical rest state, driver baselines and pending events
    /// survive; numerical warm starts are discarded. No-op for the same
    /// kind and gravity. Under Islands
    /// routing this collapses the split registry into a fresh `Single`
    /// engine (global order, then joints).
    ///
    /// Soft bodies migrate with the world: onto [`SolverKind::Xpbd`] they
    /// unpark into the fresh engine in handle order (with their touch
    /// baseline, so no manufactured begins); off it they park in the
    /// orchestrator, order-preserving but frozen. Sleep does not migrate —
    /// bodies re-sleep on the new path. Migrating onto XPBD drops joints
    /// the XPBD engine cannot solve (wheel/gear/six-DOF, see
    /// [`xpbd::XpbdEngine::add_joint`]): survivors keep their relative
    /// order, joint handles past a dropped joint compact.
    pub fn set_solver_kind(&mut self, kind: SolverKind, gravity: glam::Vec3) {
        if self.routing == RoutingKind::Single && self.kind() == kind && self.gravity == gravity {
            return;
        }
        self.collapse_to_single(kind, gravity);
    }

    /// Switch the routing policy. `Single` collapses any live split
    /// registry into the current [`Engine::kind`] engine; `Islands`
    /// builds the registry from the current single engine (bodies in
    /// handle order, then joints) and starts both engines. Global
    /// handles stay valid across the switch. No-op when unchanged.
    /// Soft bodies park for the duration of Islands routing (the split
    /// engines are sequential-impulse/AVBD and do not step particles) and
    /// unpark on the way back to `Single(Xpbd)`.
    pub fn set_routing(&mut self, routing: RoutingKind) {
        if self.routing == routing {
            return;
        }
        if routing == RoutingKind::Single {
            self.collapse_to_single(self.single_kind, self.gravity);
            return;
        }
        // Islands engines never step particles: park live soft first so the
        // split snapshot (which reads the park) cannot drop bodies.
        if let EngineInner::Xpbd(e) = &mut self.inner {
            let (bodies, touch) = e.drain_soft_registry();
            self.parked_soft = bodies;
            self.parked_soft_touch = touch;
        }
        let snapshot = self.snapshot();
        let triggers = self.drain_trigger_events();
        let mut split = Box::new(SplitState::new(self.gravity));
        for (h, body) in snapshot.bodies.into_iter().enumerate() {
            let mut record = SplitBody::new(body, self.single_kind);
            record.previous = snapshot.previous[h];
            split.bodies.push(record);
        }
        split.joints = snapshot.joints.into_iter().map(SplitJoint::new).collect();
        split.events = snapshot.events;
        split.route(split::DT, RoutePhase::Probe, &self.wake_set);
        split.rebuild();
        self.split = Some(split);
        self.routing = RoutingKind::Islands;
        self.migrations = 0;
        self.structural_dirty = StructuralState::Clean;
        self.wake_set.clear();
        self.trigger_events.extend(triggers);
    }

    /// Collapse Islands routing into a fresh `Single(kind)` engine from
    /// the registry (global order, then joints with verbatim specs —
    /// globals equal dense locals on a fresh engine). Soft bodies follow
    /// the snapshot: onto XPBD they unpark in order, off it they park.
    fn collapse_to_single(&mut self, kind: SolverKind, gravity: glam::Vec3) {
        // Drain first: the snapshot borrows, and the XPBD queue must move
        // into the orchestrator stash before the old engine is dropped.
        self.drain_xpbd_soft_queue();
        let mut snapshot = self.snapshot();
        let triggers = self.drain_trigger_events();
        if kind == SolverKind::Xpbd {
            xpbd::retain_supported_joints(&mut snapshot.joints);
        }
        let mut next = Self::new(kind, gravity);
        next.soft_friction = self.soft_friction;
        for body in snapshot.bodies {
            next.add_body(body);
        }
        next.restore_joints(snapshot.joints);
        for (h, pose) in snapshot.previous.into_iter().enumerate() {
            let h = BodyHandle::from(h);
            match &mut next.inner {
                EngineInner::SequentialImpulse(e) => e.restore_body_baseline(h, pose),
                EngineInner::Avbd(e) => e.restore_body_baseline(h, pose),
                EngineInner::Xpbd(e) => e.restore_body_baseline(h, pose),
            }
        }
        match &mut next.inner {
            EngineInner::SequentialImpulse(e) => e.restore_event_state(snapshot.events),
            EngineInner::Avbd(e) => e.restore_event_state(snapshot.events),
            EngineInner::Xpbd(e) => e.restore_event_state(snapshot.events),
        }
        match &mut next.inner {
            EngineInner::Xpbd(e) => {
                e.set_soft_friction(next.soft_friction);
                for body in snapshot.soft_bodies {
                    e.add_soft_body(body);
                }
                e.restore_soft_touch_state(snapshot.soft_touch);
            }
            EngineInner::SequentialImpulse(_) | EngineInner::Avbd(_) => {
                next.parked_soft = snapshot.soft_bodies;
                next.parked_soft_touch = snapshot.soft_touch;
            }
        }
        next.contact_events = std::mem::take(&mut self.contact_events);
        next.fracture_events = std::mem::take(&mut self.fracture_events);
        next.soft_contact_events = std::mem::take(&mut self.soft_contact_events);
        next.trigger_events = triggers;
        *self = next;
    }

    /// Move the XPBD engine's queued soft↔rigid transitions into the
    /// orchestrator stash. No-op off the XPBD path (parked bodies emit
    /// nothing). Called by `step` and before any snapshot that drops the
    /// inner engine.
    fn drain_xpbd_soft_queue(&mut self) {
        if let EngineInner::Xpbd(e) = &mut self.inner {
            self.soft_contact_events
                .extend(e.drain_soft_contact_events());
        }
    }

    /// Number of registered bodies. Removing one swaps the last into its slot.
    pub fn body_count(&self) -> usize {
        if let Some(s) = &self.split {
            return s.bodies.len();
        }
        match &self.inner {
            EngineInner::SequentialImpulse(e) => e.body_count(),
            EngineInner::Avbd(e) => e.body_count(),
            EngineInner::Xpbd(e) => e.body_count(),
        }
    }

    /// Number of live joints, including gears, in global handle order.
    pub fn joint_count(&self) -> usize {
        if let Some(s) = &self.split {
            return s.joints.len();
        }
        match &self.inner {
            EngineInner::SequentialImpulse(e) => e.joint_count(),
            EngineInner::Avbd(e) => e.joint_count(),
            EngineInner::Xpbd(e) => e.joint_count(),
        }
    }

    /// Current owner of a dynamic body. Non-dynamics mirror into every
    /// split engine and return `None`, as do invalid handles.
    pub fn body_solver(&self, handle: BodyHandle) -> Option<SolverKind> {
        if let Some(s) = &self.split {
            return match s.bodies.get(handle.index())?.owner {
                SplitOwner::Static => None,
                SplitOwner::Avbd => Some(SolverKind::Avbd),
                SplitOwner::SequentialImpulse => Some(SolverKind::SequentialImpulse),
                SplitOwner::Xpbd => Some(SolverKind::Xpbd),
            };
        }
        (self.get_body(handle)?.body_type == BodyType::Dynamic).then_some(self.single_kind)
    }

    /// Pin a body to one solver under Islands routing (`Some`), or release
    /// it back to hysteresis (`None`). A pinned dynamic body keeps its
    /// owner across routing ticks, so a joint between differently-pinned
    /// bodies becomes a cross-solver joint held by the post-step coupling
    /// pass (see [`CrossJointStatus`]); unpinning lets the island reunite
    /// on the next calm/active decision. Takes effect at the next step
    /// (marks a rebuild, wakes the body). No-op for invalid or
    /// non-dynamic handles (statics mirror into every engine natively)
    /// and outside Islands routing (one engine owns everything there).
    pub fn pin_body_solver(&mut self, handle: BodyHandle, solver: Option<SolverKind>) {
        let Some(s) = self.split.as_mut() else {
            return;
        };
        let Some(b) = s.bodies.get_mut(handle.index()) else {
            return;
        };
        if b.body.body_type != BodyType::Dynamic {
            return;
        }
        if b.pinned == solver {
            return;
        }
        b.pinned = solver;
        self.structural_dirty = StructuralState::Dirty;
        self.wake_set.insert(handle);
    }

    /// How one registry joint is solved: [`CrossJointStatus::Native`] when
    /// both ends share a solver (always, under [`RoutingKind::Single`]),
    /// [`CrossJointStatus::Coupled`] for a cross-solver ball/distance row,
    /// [`CrossJointStatus::Unsupported`] with a named cause otherwise.
    /// `None` for an invalid joint handle.
    pub fn cross_joint_status(&self, handle: JointHandle) -> Option<CrossJointStatus> {
        if let Some(s) = &self.split {
            return s.cross_status(handle);
        }
        (handle.index() < self.joint_count()).then_some(CrossJointStatus::Native)
    }

    fn snapshot(&self) -> SceneSnapshot {
        if let Some(s) = &self.split {
            return SceneSnapshot {
                bodies: s.bodies.iter().map(|b| b.body.clone()).collect(),
                joints: s.joint_snapshots(),
                previous: s.bodies.iter().map(|b| b.previous).collect(),
                events: s.events.clone(),
                soft_bodies: self.parked_soft.clone(),
                soft_touch: self.parked_soft_touch.clone(),
            };
        }
        match &self.inner {
            EngineInner::SequentialImpulse(e) => SceneSnapshot {
                bodies: e.bodies_snapshot(),
                joints: e.joint_snapshots(),
                previous: e.body_baselines(),
                events: e.event_state(),
                soft_bodies: self.parked_soft.clone(),
                soft_touch: self.parked_soft_touch.clone(),
            },
            EngineInner::Avbd(e) => SceneSnapshot {
                bodies: e.bodies_snapshot(),
                joints: e.joint_snapshots(),
                previous: e.body_baselines(),
                events: e.event_state(),
                soft_bodies: self.parked_soft.clone(),
                soft_touch: self.parked_soft_touch.clone(),
            },
            EngineInner::Xpbd(e) => SceneSnapshot {
                bodies: e.bodies_snapshot(),
                joints: e.joint_snapshots(),
                previous: e.body_baselines(),
                events: e.event_state(),
                soft_bodies: e.soft_bodies_snapshot(),
                soft_touch: e.soft_touch_state(),
            },
        }
    }

    fn restore_joints(&mut self, joints: Vec<JointSnapshot>) {
        let mut remap = vec![None; joints.len()];
        for (old, mut j) in joints.into_iter().enumerate() {
            migration::remap_gear(&mut j.spec, &remap);
            let Ok(h) = self.add_joint(j.a, j.b, j.spec) else {
                continue;
            };
            match &mut self.inner {
                EngineInner::SequentialImpulse(e) => e.restore_joint_reference(h, j.reference),
                EngineInner::Avbd(e) => e.restore_joint_reference(h, j.reference),
                EngineInner::Xpbd(e) => e.restore_joint_reference(h, j.reference),
            }
            remap[old] = Some(h);
        }
    }

    /// Number of registered soft bodies, live or parked (dense
    /// [`SoftHandle`] order — see [`Engine::add_soft_body`]).
    pub fn soft_body_count(&self) -> usize {
        if self.split.is_some() {
            return self.parked_soft.len();
        }
        match &self.inner {
            EngineInner::Xpbd(e) => e.soft_body_count(),
            EngineInner::SequentialImpulse(_) | EngineInner::Avbd(_) => self.parked_soft.len(),
        }
    }

    /// Register a soft body and return its handle. On the XPBD path the
    /// body steps in the engine's substep loop; off it (SequentialImpulse,
    /// AVBD, Islands) the body parks in the orchestrator in dense order —
    /// readable but frozen — until the world migrates onto XPBD. Handles
    /// are dense indices: removal swaps the last body into the freed slot.
    pub fn add_soft_body(&mut self, body: SoftBody) -> SoftHandle {
        if self.split.is_some() {
            self.parked_soft.push(body);
            return SoftHandle::from(self.parked_soft.len() - 1);
        }
        match &mut self.inner {
            EngineInner::Xpbd(e) => e.add_soft_body(body),
            EngineInner::SequentialImpulse(_) | EngineInner::Avbd(_) => {
                self.parked_soft.push(body);
                SoftHandle::from(self.parked_soft.len() - 1)
            }
        }
    }

    /// Remove a soft body, swapping the last into its slot (invalid handles
    /// are a no-op). Queued soft events and the touch baseline are cleared —
    /// indices shift, like the engines' own invalidation on removal.
    pub fn remove_soft_body(&mut self, handle: SoftHandle) {
        if handle.index() >= self.soft_body_count() {
            return;
        }
        self.soft_contact_events.clear();
        if self.split.is_some() {
            self.parked_soft.swap_remove(handle.index());
            self.parked_soft_touch.clear();
            return;
        }
        match &mut self.inner {
            EngineInner::Xpbd(e) => e.remove_soft_body(handle),
            EngineInner::SequentialImpulse(_) | EngineInner::Avbd(_) => {
                self.parked_soft.swap_remove(handle.index());
                self.parked_soft_touch.clear();
            }
        }
    }

    /// Read-only access to a soft body (live or parked), or `None` for an
    /// invalid handle.
    pub fn get_soft_body(&self, handle: SoftHandle) -> Option<&SoftBody> {
        if self.split.is_some() {
            return self.parked_soft.get(handle.index());
        }
        match &self.inner {
            EngineInner::Xpbd(e) => e.get_soft_body(handle),
            EngineInner::SequentialImpulse(_) | EngineInner::Avbd(_) => {
                self.parked_soft.get(handle.index())
            }
        }
    }

    /// Mutable access to a soft body (live or parked), or `None` for an
    /// invalid handle. Direct particle edits take effect at the next
    /// [`PhysicsEngine::step`]; after teleporting a live body, call
    /// [`Engine::wake_soft_body`] so no velocity spike is derived from the
    /// jump (same discipline as [`XpbdEngine::wake_soft_body`]). Unlike the
    /// Islands rigid path this does not auto-wake: a host that only reads
    /// must not keep bodies awake.
    pub fn get_soft_body_mut(&mut self, handle: SoftHandle) -> Option<&mut SoftBody> {
        if self.split.is_some() {
            return self.parked_soft.get_mut(handle.index());
        }
        match &mut self.inner {
            EngineInner::Xpbd(e) => e.get_soft_body_mut(handle),
            EngineInner::SequentialImpulse(_) | EngineInner::Avbd(_) => {
                self.parked_soft.get_mut(handle.index())
            }
        }
    }

    /// Whether the soft body is currently sleeping (frozen with zeroed
    /// velocities). Parked bodies do not step and report `false`, as do
    /// invalid handles.
    pub fn is_soft_asleep(&self, handle: SoftHandle) -> bool {
        if self.split.is_some() {
            return false;
        }
        match &self.inner {
            EngineInner::Xpbd(e) => e.is_soft_asleep(handle),
            EngineInner::SequentialImpulse(_) | EngineInner::Avbd(_) => false,
        }
    }

    /// Wake a sleeping soft body without moving it (see
    /// [`XpbdEngine::wake_soft_body`]). No-op for parked or invalid
    /// handles — call it after direct pose edits through
    /// [`Engine::get_soft_body_mut`].
    pub fn wake_soft_body(&mut self, handle: SoftHandle) {
        if self.split.is_some() {
            return;
        }
        if let EngineInner::Xpbd(e) = &mut self.inner {
            e.wake_soft_body(handle);
        }
    }

    /// Drain soft↔rigid begin/end transitions produced by completed steps
    /// (see [`xpbd::SoftContactEvent`]). Queued until drained; body
    /// removals clear the queue. Parked bodies emit nothing.
    pub fn drain_soft_contact_events(&mut self) -> Vec<xpbd::SoftContactEvent> {
        self.drain_xpbd_soft_queue();
        std::mem::take(&mut self.soft_contact_events)
    }

    /// Coulomb coefficient on the soft side (≥ 0, default 0.5): each
    /// soft↔rigid pair uses `sqrt(soft_friction · body.friction)` (see
    /// [`XpbdEngine::set_soft_friction`]). Applies live on the XPBD path;
    /// off it the value is carried by the orchestrator and applied to the
    /// next fresh XPBD engine.
    pub fn set_soft_friction(&mut self, mu: f32) {
        self.soft_friction = if mu.is_finite() { mu.max(0.0) } else { 0.0 };
        if let EngineInner::Xpbd(e) = &mut self.inner {
            e.set_soft_friction(self.soft_friction);
        }
    }

    /// Current soft-side Coulomb coefficient (see
    /// [`Engine::set_soft_friction`]).
    pub fn soft_friction(&self) -> f32 {
        self.soft_friction
    }

    /// Drain fracture reports since the last call (see [`FractureEvent`]).
    pub fn drain_fracture_events(&mut self) -> Vec<FractureEvent> {
        std::mem::take(&mut self.fracture_events)
    }

    /// Rebuild split engines from the registry when structural changes
    /// (add/remove) are pending. Cheap flag check on the hot path.
    fn split_ensure_built(&mut self) {
        if self.structural_dirty.is_dirty() {
            if let Some(s) = &mut self.split {
                s.route(split::DT, RoutePhase::Probe, &self.wake_set);
                s.rebuild();
            }
            self.structural_dirty = StructuralState::Clean;
        }
    }

    /// Accumulate host time; each common tick routes swept islands first,
    /// steps AVBD and SequentialImpulse, then captures events before fracture/rebuild.
    fn split_step(&mut self, dt: f32) {
        let Some(s) = &mut self.split else { return };
        s.timing = SplitTiming::default();
        s.time_debt =
            (s.time_debt + f64::from(dt)).min(f64::from(split::DT) * split::MAX_STEPS as f64);
        let mut steps = 0;
        while self
            .split
            .as_ref()
            .is_some_and(|s| s.time_debt >= f64::from(split::DT))
            && steps < split::MAX_STEPS
        {
            self.split_ensure_built();
            let edited = std::mem::take(&mut self.wake_set);
            let Some(s) = self.split.as_mut() else {
                break;
            };
            s.time_debt -= f64::from(split::DT);
            if s.route(split::DT, RoutePhase::Tick, &edited) > 0 {
                s.rebuild();
                self.migrations += 1;
            }
            for h in edited {
                s.push_global(h);
            }
            let (contacts, triggers) = s.step();
            self.trigger_events.extend(triggers);
            self.fracture_split(&contacts);
            self.contact_events.extend(contacts);
            steps += 1;
        }
    }

    /// Fracture uses globally mapped hits captured before any rebuild.
    fn fracture_split(&mut self, contacts: &[ContactEvent]) {
        let Some(s) = self.split.as_mut() else { return };
        let mut candidates = std::collections::BTreeSet::new();
        for event in contacts {
            if let ContactEventKind::Hit { approach_speed, .. } = event.kind {
                for h in [event.body_a, event.body_b] {
                    if let Some(body) = s.bodies.get(h.index()).map(|b| &b.body)
                        && body.body_type == BodyType::Dynamic
                        && approach_speed >= body.fracture_impact_speed
                    {
                        candidates.insert(h);
                    }
                }
            }
        }
        let start = self.fracture_events.len();
        for parent in candidates.into_iter().rev() {
            let Some(body) = s.bodies.get(parent.index()).map(|b| b.body.clone()) else {
                continue;
            };
            let Some(halves) = Self::split_box(&body) else {
                continue;
            };
            let last = BodyHandle::from(s.bodies.len() - 1);
            Self::remap_fracture_pieces(&mut self.fracture_events[start..], parent, last);
            // Halves inherit the parent's solver placement (owner + pin):
            // a cross-joint end that shatters must not silently reunite or
            // strand its joint on another solver.
            let inherit = s
                .bodies
                .get(parent.index())
                .map(|b| (b.owner, b.pinned))
                .filter(|(owner, _)| *owner != SplitOwner::Static);
            Self::split_remove_body(s, parent);
            for half in halves {
                let mut record = SplitBody::new(half, SolverKind::Avbd);
                if let Some((owner, pinned)) = inherit {
                    record.owner = owner;
                    record.pinned = pinned;
                }
                s.bodies.push(record);
            }
            let n = s.bodies.len();
            self.fracture_events.push(FractureEvent {
                parent,
                pieces: [BodyHandle::from(n - 2), BodyHandle::from(n - 1)],
            });
            self.structural_dirty = StructuralState::Dirty;
        }
    }

    /// Registry body removal with joint remap (swap_remove discipline,
    /// same as the engines: the tail moves into the hole, refs are patched).
    fn split_remove_body(s: &mut SplitState, handle: BodyHandle) {
        if handle.index() >= s.bodies.len() {
            return;
        }
        let last = BodyHandle::from(s.bodies.len() - 1);
        let removed = s
            .joints
            .iter()
            .map(|j| j.state.a == handle || j.state.b == handle)
            .collect();
        Self::split_retain_joints(s, removed);
        let map = |h| if h == last { handle } else { h };
        for j in &mut s.joints {
            j.state.a = map(j.state.a);
            j.state.b = map(j.state.b);
        }
        s.bodies.swap_remove(handle.index());
        let remap = |set: &std::collections::BTreeSet<(BodyHandle, BodyHandle)>| {
            set.iter()
                .filter_map(|&(a, b)| {
                    if a == handle || b == handle {
                        None
                    } else {
                        Some((map(a).min(map(b)), map(a).max(map(b))))
                    }
                })
                .collect()
        };
        s.events.contacts = remap(&s.events.contacts);
        s.events.triggers = remap(&s.events.triggers);
    }

    /// Registry joint removal with gear-ref remap.
    fn split_remove_joint(s: &mut SplitState, handle: JointHandle) {
        if handle.index() >= s.joints.len() {
            return;
        }
        let mut removed = vec![false; s.joints.len()];
        removed[handle.index()] = true;
        Self::split_retain_joints(s, removed);
    }

    fn split_retain_joints(s: &mut SplitState, removed: Vec<bool>) {
        let kinds: Vec<_> = s.joints.iter().map(|j| j.state.spec).collect();
        let remap = migration::joint_remap(&kinds, removed);
        let mut old = 0;
        s.joints.retain_mut(|j| {
            let keep = remap[old].is_some();
            old += 1;
            if keep {
                migration::remap_gear(&mut j.state.spec, &remap);
            }
            keep
        });
    }

    fn remap_fracture_pieces(events: &mut [FractureEvent], removed: BodyHandle, last: BodyHandle) {
        for event in events {
            for h in &mut event.pieces {
                if *h == last {
                    *h = removed;
                }
            }
        }
    }

    /// Split a dynamic box along its longest axis into two halves with
    /// conserved mass, velocity and material (halves inherit the fracture
    /// threshold, so rubble may fracture again). Returns `None` for
    /// non-boxes, non-dynamics and massless bodies.
    fn split_box(parent: &RigidBody) -> Option<[RigidBody; 2]> {
        let Shape::Box { half_extents: h } = parent.shape else {
            return None;
        };
        if parent.body_type != BodyType::Dynamic || parent.mass <= 0.0 {
            return None;
        }
        let (axis, ext) = if h.x >= h.y && h.x >= h.z {
            (glam::Vec3::X, h.x)
        } else if h.y >= h.z {
            (glam::Vec3::Y, h.y)
        } else {
            (glam::Vec3::Z, h.z)
        };
        let mut h2 = h.to_array();
        h2[if axis == glam::Vec3::X {
            0
        } else if axis == glam::Vec3::Y {
            1
        } else {
            2
        }] = ext * HALF;
        let h2 = glam::Vec3::from_array(h2);
        let off = (parent.orientation * axis).normalize_or(axis) * (ext * HALF);
        let mut halves = [parent.clone(), parent.clone()];
        for (half, s) in halves.iter_mut().zip([-1.0, 1.0]) {
            half.shape = Shape::Box { half_extents: h2 };
            half.position = parent.position + off * s;
            half.velocity = parent.velocity + parent.angular_velocity.cross(off * s);
            if let Some(m) = crate::invariants::PositiveF32::try_new(parent.mass * HALF) {
                half.set_mass_kind(crate::invariants::MassKind::Free(m));
            } else {
                half.set_mass_kind(crate::invariants::MassKind::Fixed);
            }
        }
        Some(halves)
    }

    /// Fracture pass: every [`ContactEventKind::Hit`] whose approach
    /// speed reaches a dynamic box's `fracture_impact_speed` splits it.
    /// Candidates are removed in DESCENDING handle order — `swap_remove`
    /// only ever moves the tail into the removed slot, so smaller handles
    /// (processed later) are never invalidated. Joints on the parent die
    /// with the removal on both solvers; halves start joint-free, awake,
    /// with zeroed quiet timers. The XPBD path produces no rigid contact
    /// events, so fracture never fires there.
    fn fracture_pass(&mut self) {
        use ContactEventKind::Hit;
        let mut candidates = std::collections::BTreeSet::new();
        let event_start = self.fracture_events.len();
        let inner = &mut self.inner;
        // The inner queue is consumed here, so every event is stashed for
        // re-serve: fracture must never swallow the host's contact stream.
        let drained: Vec<ContactEvent> = match inner {
            EngineInner::SequentialImpulse(e) => e.drain_contact_events(),
            EngineInner::Avbd(e) => e.drain_contact_events(),
            EngineInner::Xpbd(e) => e.drain_contact_events(),
        };
        self.contact_events.extend(drained.iter().cloned());
        let hits: Vec<(BodyHandle, BodyHandle, f32)> = drained
            .into_iter()
            .filter_map(|ev| match ev.kind {
                Hit { approach_speed, .. } => Some((ev.body_a, ev.body_b, approach_speed)),
                _ => None,
            })
            .collect();
        let get = |inner: &EngineInner, h: BodyHandle| -> Option<RigidBody> {
            match inner {
                EngineInner::SequentialImpulse(e) => e.get_body(h).cloned(),
                EngineInner::Avbd(e) => e.get_body(h).cloned(),
                EngineInner::Xpbd(e) => e.get_body(h).cloned(),
            }
        };
        for (a, b, approach) in hits {
            for h in [a, b] {
                if let Some(body) = get(inner, h)
                    && body.body_type == BodyType::Dynamic
                    && approach >= body.fracture_impact_speed
                {
                    candidates.insert(h);
                }
            }
        }
        let mut ordered: Vec<BodyHandle> = candidates.into_iter().collect();
        ordered.sort_unstable_by(|a, b| b.cmp(a));
        for parent in ordered {
            let Some(body) = get(inner, parent) else {
                continue;
            };
            let Some([pa, pb]) = Self::split_box(&body) else {
                continue;
            };
            let last = match inner {
                EngineInner::SequentialImpulse(e) => BodyHandle::from(e.body_count() - 1),
                EngineInner::Avbd(e) => BodyHandle::from(e.body_count() - 1),
                EngineInner::Xpbd(e) => BodyHandle::from(e.body_count() - 1),
            };
            Self::remap_fracture_pieces(&mut self.fracture_events[event_start..], parent, last);
            match inner {
                EngineInner::SequentialImpulse(e) => e.remove_body(parent),
                EngineInner::Avbd(e) => e.remove_body(parent),
                EngineInner::Xpbd(e) => e.remove_body(parent),
            }
            let (ha, hb) = match inner {
                EngineInner::SequentialImpulse(e) => (e.add_body(pa), e.add_body(pb)),
                EngineInner::Avbd(e) => (e.add_body(pa), e.add_body(pb)),
                EngineInner::Xpbd(e) => (e.add_body(pa), e.add_body(pb)),
            };
            self.fracture_events.push(FractureEvent {
                parent,
                pieces: [ha, hb],
            });
        }
    }
}

impl PhysicsEngine for Engine {
    fn step(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        if self.routing == RoutingKind::Islands {
            self.split_step(dt);
            return;
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.step(dt),
            EngineInner::Avbd(e) => e.step(dt),
            EngineInner::Xpbd(e) => e.step(dt),
        }
        self.drain_xpbd_soft_queue();
        self.fracture_pass();
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        if let Some(s) = &mut self.split {
            let h = BodyHandle::from(s.bodies.len());
            s.bodies.push(SplitBody::new(body, SolverKind::Avbd));
            self.structural_dirty = StructuralState::Dirty;
            return h;
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.add_body(body),
            EngineInner::Avbd(e) => e.add_body(body),
            EngineInner::Xpbd(e) => e.add_body(body),
        }
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        if handle.index() >= self.body_count() {
            return;
        }
        self.contact_events.clear();
        // Rigid indices shift under soft touch triples too: the live engine
        // clears its own caches, the parked baseline is dropped (a later
        // Begin re-establishes it — same policy as the queue clear above).
        self.soft_contact_events.clear();
        self.parked_soft_touch.clear();
        if let Some(s) = &mut self.split {
            let last = BodyHandle::from(s.bodies.len() - 1);
            for &(a, b) in &s.events.triggers {
                if a == handle || b == handle {
                    self.trigger_events.push(TriggerEvent {
                        body_a: a,
                        body_b: b,
                        kind: TriggerEventKind::Exited,
                    });
                }
            }
            Self::split_remove_body(s, handle);
            self.wake_set.remove(&handle);
            if handle != last && self.wake_set.remove(&last) {
                self.wake_set.insert(handle);
            }
            self.structural_dirty = StructuralState::Dirty;
            return;
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.remove_body(handle),
            EngineInner::Avbd(e) => e.remove_body(handle),
            EngineInner::Xpbd(e) => e.remove_body(handle),
        }
    }

    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        if self.routing == RoutingKind::Islands {
            return self
                .split
                .as_ref()?
                .bodies
                .get(handle.index())
                .map(|r| &r.body);
        }
        match &self.inner {
            EngineInner::SequentialImpulse(e) => e.get_body(handle),
            EngineInner::Avbd(e) => e.get_body(handle),
            EngineInner::Xpbd(e) => e.get_body(handle),
        }
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        if self.routing == RoutingKind::Islands {
            let s = self.split.as_mut()?;
            let r = s.bodies.get_mut(handle.index())?;
            // Host edit: wake the owner's copy before the next step (an
            // edit alone must not leave a stale sleeping copy behind).
            self.wake_set.insert(handle);
            return Some(&mut r.body);
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.get_body_mut(handle),
            EngineInner::Avbd(e) => e.get_body_mut(handle),
            EngineInner::Xpbd(e) => e.get_body_mut(handle),
        }
    }

    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Result<JointHandle, JointError> {
        if let Some(s) = &mut self.split {
            let h = s.add_joint(body_a, body_b, kind)?;
            self.structural_dirty = StructuralState::Dirty;
            self.wake_set.insert(body_a);
            self.wake_set.insert(body_b);
            return Ok(h);
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.add_joint(body_a, body_b, kind),
            EngineInner::Avbd(e) => e.add_joint(body_a, body_b, kind),
            EngineInner::Xpbd(e) => e.add_joint(body_a, body_b, kind),
        }
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        if handle.index() >= self.joint_count() {
            return;
        }
        if let Some(s) = &mut self.split {
            let j = s.joints[handle.index()].state;
            self.wake_set.insert(j.a);
            self.wake_set.insert(j.b);
            Self::split_remove_joint(s, handle);
            self.structural_dirty = StructuralState::Dirty;
            return;
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.remove_joint(handle),
            EngineInner::Avbd(e) => e.remove_joint(handle),
            EngineInner::Xpbd(e) => e.remove_joint(handle),
        }
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Result<Option<RaycastHit>, QueryError> {
        if let Some(s) = &self.split {
            return s.raycast(ray, max_dist);
        }
        match &self.inner {
            EngineInner::SequentialImpulse(e) => e.raycast(ray, max_dist),
            EngineInner::Avbd(e) => e.raycast(ray, max_dist),
            EngineInner::Xpbd(e) => e.raycast(ray, max_dist),
        }
    }

    fn shapecast(&self, shape: &Shape, from: glam::Vec3, to: glam::Vec3) -> Option<RaycastHit> {
        if let Some(s) = &self.split {
            return s.shapecast(shape, from, to);
        }
        match &self.inner {
            EngineInner::SequentialImpulse(e) => e.shapecast(shape, from, to),
            EngineInner::Avbd(e) => e.shapecast(shape, from, to),
            EngineInner::Xpbd(e) => e.shapecast(shape, from, to),
        }
    }

    fn drain_trigger_events(&mut self) -> Vec<TriggerEvent> {
        let mut events = std::mem::take(&mut self.trigger_events);
        if self.routing == RoutingKind::Single {
            events.extend(match &mut self.inner {
                EngineInner::SequentialImpulse(e) => e.drain_trigger_events(),
                EngineInner::Avbd(e) => e.drain_trigger_events(),
                EngineInner::Xpbd(e) => e.drain_trigger_events(),
            });
        }
        events
    }

    fn drain_contact_events(&mut self) -> Vec<ContactEvent> {
        // Served from the orchestrator stash (filled by `step`): the
        // fracture pass sits between the inner queue and the host.
        std::mem::take(&mut self.contact_events)
    }

    fn wake_body(&mut self, handle: BodyHandle) {
        if let Some(s) = &mut self.split {
            if handle.index() < s.bodies.len() {
                self.wake_set.insert(handle);
                s.wake_global(handle);
            }
            return;
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.wake_body(handle),
            EngineInner::Avbd(e) => e.wake_body(handle),
            EngineInner::Xpbd(e) => e.wake_body(handle),
        }
    }
}
