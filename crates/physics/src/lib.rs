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

mod broadphase;
mod broadphase_tree;
mod contact_math;
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
pub mod distance;
/// The physics step pipeline and the [`crate::engine::PhysicsEngine`] trait.
pub mod engine;
/// GJK/EPA fallback narrow phase for cylinder, cone and convex hull pairs.
pub(crate) mod gjk;
#[cfg(feature = "gpu")]
/// GPU sequential-impulse accelerator for wide contact batches (G7).
// The `#[gpu_pipeline]` macro emits undocumented `pub mod`s for its kernels
// (outer attributes do not propagate into the expansion), so the docs gate
// is relaxed for this subtree; every hand-written public item stays documented.
#[allow(missing_docs)]
pub mod gpu;
pub mod joint;
pub mod math;
/// Sequential-impulse solver internals (Genesis-style `solvers/rigid/` box).
pub mod sequential_impulse;
/// Collision shapes with AABB projection and inertia tensors.
pub mod shape;
/// Trigger overlap event types emitted by the sequential-impulse physics engine.
pub mod trigger;
/// SIMD-wide solver for single-point contact batches.
pub mod wide;

use migration::{JointSnapshot, SceneSnapshot};
use split::{SplitBody, SplitJoint, SplitOwner, SplitState};

pub use avbd::AvbdEngine;
pub use body::{BodyHandle, BodyType, RigidBody};
pub use broadphase::{BroadPhaseKind, BroadPhaseStats, StepBudget, StepTiming};
pub use engine::{PhysicsEngine, SequentialImpulseEngine};
pub use joint::{
    AxisConfig, JointHandle, JointKind, PrismaticLimit, PrismaticMotor, ResolvedJoint,
    RevoluteLimit, RevoluteMotor, WheelSuspension, resolve_joint,
};
pub use math::{AABB, Ray, RaycastHit};
pub use shape::{ConvexHull, Heightfield, Shape, TriMesh};
pub use trigger::{
    CONTACT_BEGIN_SLOP, CONTACT_HIT_THRESHOLD, ContactEvent, ContactEventKind, FractureEvent,
    TriggerEvent, TriggerEventKind,
};

/// Selectable constraint solver (M2 intra-engine modularity, Genesis
/// style): the sequential-impulse engine or the AVBD engine.
/// Same [`PhysicsEngine`] seam, same scenes — the orchestrator
/// ([`Engine`]) migrates bodies and joints across the switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    /// Sequential-impulse engine with islands, sleep and substeps.
    SequentialImpulse,
    /// AVBD engine (position-level, single-thread sweep).
    Avbd,
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
pub struct Engine {
    inner: EngineInner,
    fracture_events: Vec<FractureEvent>,
    /// Contact events drained from the inner engine during `step`
    /// (the fracture pass consumes the inner queue, so the orchestrator
    /// re-serves them here — see `drain_contact_events`).
    contact_events: Vec<ContactEvent>,
    /// Globally remapped trigger transitions, independent of rebuilt locals.
    trigger_events: Vec<TriggerEvent>,
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
    structural_dirty: bool,
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
}

impl Engine {
    /// Empty orchestrator with the given solver and world-space gravity.
    pub fn new(kind: SolverKind, gravity: glam::Vec3) -> Self {
        let inner = match kind {
            SolverKind::SequentialImpulse => {
                EngineInner::SequentialImpulse(Box::new(SequentialImpulseEngine::new(gravity)))
            }
            SolverKind::Avbd => EngineInner::Avbd(Box::new(AvbdEngine::new(gravity))),
        };
        Self {
            inner,
            fracture_events: Vec::new(),
            contact_events: Vec::new(),
            trigger_events: Vec::new(),
            gravity,
            single_kind: kind,
            routing: RoutingKind::Single,
            split: None,
            structural_dirty: false,
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
        let (mut av, mut si) = (0, 0);
        for b in &s.bodies {
            match b.owner {
                SplitOwner::Avbd => av += 1,
                SplitOwner::SequentialImpulse => si += 1,
                SplitOwner::Static => {}
            }
        }
        Some(SplitMetrics {
            migrations: self.migrations,
            avbd_bodies: av,
            si_bodies: si,
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
    pub fn set_routing(&mut self, routing: RoutingKind) {
        if self.routing == routing {
            return;
        }
        if routing == RoutingKind::Single {
            self.collapse_to_single(self.single_kind, self.gravity);
            return;
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
        split.route(split::DT, false, &self.wake_set);
        split.rebuild();
        self.split = Some(split);
        self.routing = RoutingKind::Islands;
        self.migrations = 0;
        self.structural_dirty = false;
        self.wake_set.clear();
        self.trigger_events.extend(triggers);
    }

    /// Collapse Islands routing into a fresh `Single(kind)` engine from
    /// the registry (global order, then joints with verbatim specs —
    /// globals equal dense locals on a fresh engine).
    fn collapse_to_single(&mut self, kind: SolverKind, gravity: glam::Vec3) {
        let snapshot = self.snapshot();
        let triggers = self.drain_trigger_events();
        let mut next = Self::new(kind, gravity);
        for body in snapshot.bodies {
            next.add_body(body);
        }
        next.restore_joints(snapshot.joints);
        for (h, pose) in snapshot.previous.into_iter().enumerate() {
            match &mut next.inner {
                EngineInner::SequentialImpulse(e) => e.restore_body_baseline(h, pose),
                EngineInner::Avbd(e) => e.restore_body_baseline(h, pose),
            }
        }
        match &mut next.inner {
            EngineInner::SequentialImpulse(e) => e.restore_event_state(snapshot.events),
            EngineInner::Avbd(e) => e.restore_event_state(snapshot.events),
        }
        next.contact_events = std::mem::take(&mut self.contact_events);
        next.fracture_events = std::mem::take(&mut self.fracture_events);
        next.trigger_events = triggers;
        *self = next;
    }

    /// Number of registered bodies. Removing one swaps the last into its slot.
    pub fn body_count(&self) -> usize {
        if let Some(s) = &self.split {
            return s.bodies.len();
        }
        match &self.inner {
            EngineInner::SequentialImpulse(e) => e.body_count(),
            EngineInner::Avbd(e) => e.body_count(),
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
        }
    }

    /// Current owner of a dynamic body. Non-dynamics live in both solvers
    /// under Islands routing and return `None`, as do invalid handles.
    pub fn body_solver(&self, handle: BodyHandle) -> Option<SolverKind> {
        if let Some(s) = &self.split {
            return match s.bodies.get(handle)?.owner {
                SplitOwner::Static => None,
                SplitOwner::Avbd => Some(SolverKind::Avbd),
                SplitOwner::SequentialImpulse => Some(SolverKind::SequentialImpulse),
            };
        }
        (self.get_body(handle)?.body_type == BodyType::Dynamic).then_some(self.single_kind)
    }

    fn snapshot(&self) -> SceneSnapshot {
        if let Some(s) = &self.split {
            return SceneSnapshot {
                bodies: s.bodies.iter().map(|b| b.body.clone()).collect(),
                joints: s.joint_snapshots(),
                previous: s.bodies.iter().map(|b| b.previous).collect(),
                events: s.events.clone(),
            };
        }
        match &self.inner {
            EngineInner::SequentialImpulse(e) => SceneSnapshot {
                bodies: e.bodies_snapshot(),
                joints: e.joint_snapshots(),
                previous: e.body_baselines(),
                events: e.event_state(),
            },
            EngineInner::Avbd(e) => SceneSnapshot {
                bodies: e.bodies_snapshot(),
                joints: e.joint_snapshots(),
                previous: e.body_baselines(),
                events: e.event_state(),
            },
        }
    }

    fn restore_joints(&mut self, joints: Vec<JointSnapshot>) {
        let mut remap = vec![None; joints.len()];
        for (old, mut j) in joints.into_iter().enumerate() {
            migration::remap_gear(&mut j.spec, &remap);
            let h = self
                .add_joint(j.a, j.b, j.spec)
                .expect("validated migrating joint");
            match &mut self.inner {
                EngineInner::SequentialImpulse(e) => e.restore_joint_reference(h, j.reference),
                EngineInner::Avbd(e) => e.restore_joint_reference(h, j.reference),
            }
            remap[old] = Some(h);
        }
    }

    /// Drain fracture reports since the last call (see [`FractureEvent`]).
    pub fn drain_fracture_events(&mut self) -> Vec<FractureEvent> {
        std::mem::take(&mut self.fracture_events)
    }

    /// Rebuild split engines from the registry when structural changes
    /// (add/remove) are pending. Cheap flag check on the hot path.
    fn split_ensure_built(&mut self) {
        if self.structural_dirty {
            if let Some(s) = &mut self.split {
                s.route(split::DT, false, &self.wake_set);
                s.rebuild();
            }
            self.structural_dirty = false;
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
            let s = self.split.as_mut().expect("Islands state");
            s.time_debt -= f64::from(split::DT);
            if s.route(split::DT, true, &edited) > 0 {
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
                    if let Some(body) = s.bodies.get(h).map(|b| &b.body)
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
            let Some(body) = s.bodies.get(parent).map(|b| b.body.clone()) else {
                continue;
            };
            let Some(halves) = Self::split_box(&body) else {
                continue;
            };
            let last = s.bodies.len() - 1;
            Self::remap_fracture_pieces(&mut self.fracture_events[start..], parent, last);
            Self::split_remove_body(s, parent);
            for half in halves {
                s.bodies.push(SplitBody::new(half, SolverKind::Avbd));
            }
            let n = s.bodies.len();
            self.fracture_events.push(FractureEvent {
                parent,
                pieces: [n - 2, n - 1],
            });
            self.structural_dirty = true;
        }
    }

    /// Registry body removal with joint remap (swap_remove discipline,
    /// same as the engines: the tail moves into the hole, refs are patched).
    fn split_remove_body(s: &mut SplitState, handle: BodyHandle) {
        if handle >= s.bodies.len() {
            return;
        }
        let last = s.bodies.len() - 1;
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
        s.bodies.swap_remove(handle);
        let remap = |set: &std::collections::BTreeSet<(usize, usize)>| {
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
    fn split_remove_joint(s: &mut SplitState, handle: usize) {
        if handle >= s.joints.len() {
            return;
        }
        let mut removed = vec![false; s.joints.len()];
        removed[handle] = true;
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

    fn remap_fracture_pieces(events: &mut [FractureEvent], removed: usize, last: usize) {
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
        }] = ext * 0.5;
        let h2 = glam::Vec3::from_array(h2);
        let off = (parent.orientation * axis).normalize_or(axis) * (ext * 0.5);
        let mut halves = [parent.clone(), parent.clone()];
        for (half, s) in halves.iter_mut().zip([-1.0, 1.0]) {
            half.shape = Shape::Box { half_extents: h2 };
            half.position = parent.position + off * s;
            half.velocity = parent.velocity + parent.angular_velocity.cross(off * s);
            half.mass = parent.mass * 0.5;
            half.inv_mass = 1.0 / half.mass;
            half.inertia = half.shape.inertia(half.mass);
        }
        Some(halves)
    }

    /// Fracture pass: every [`ContactEventKind::Hit`] whose approach
    /// speed reaches a dynamic box's `fracture_impact_speed` splits it.
    /// Candidates are removed in DESCENDING handle order — `swap_remove`
    /// only ever moves the tail into the removed slot, so smaller handles
    /// (processed later) are never invalidated. Joints on the parent die
    /// with the removal on both solvers; halves start joint-free, awake,
    /// with zeroed quiet timers.
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
                EngineInner::SequentialImpulse(e) => e.body_count() - 1,
                EngineInner::Avbd(e) => e.body_count() - 1,
            };
            Self::remap_fracture_pieces(&mut self.fracture_events[event_start..], parent, last);
            match inner {
                EngineInner::SequentialImpulse(e) => e.remove_body(parent),
                EngineInner::Avbd(e) => e.remove_body(parent),
            }
            let (ha, hb) = match inner {
                EngineInner::SequentialImpulse(e) => (e.add_body(pa), e.add_body(pb)),
                EngineInner::Avbd(e) => (e.add_body(pa), e.add_body(pb)),
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
        }
        self.fracture_pass();
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        if let Some(s) = &mut self.split {
            let h = s.bodies.len();
            s.bodies.push(SplitBody::new(body, SolverKind::Avbd));
            self.structural_dirty = true;
            return h;
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.add_body(body),
            EngineInner::Avbd(e) => e.add_body(body),
        }
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        if handle >= self.body_count() {
            return;
        }
        self.contact_events.clear();
        if let Some(s) = &mut self.split {
            let last = s.bodies.len() - 1;
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
            self.structural_dirty = true;
            return;
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.remove_body(handle),
            EngineInner::Avbd(e) => e.remove_body(handle),
        }
    }

    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        if self.routing == RoutingKind::Islands {
            return self.split.as_ref()?.bodies.get(handle).map(|r| &r.body);
        }
        match &self.inner {
            EngineInner::SequentialImpulse(e) => e.get_body(handle),
            EngineInner::Avbd(e) => e.get_body(handle),
        }
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        if self.routing == RoutingKind::Islands {
            let s = self.split.as_mut()?;
            let r = s.bodies.get_mut(handle)?;
            // Host edit: wake the owner's copy before the next step (an
            // edit alone must not leave a stale sleeping copy behind).
            self.wake_set.insert(handle);
            return Some(&mut r.body);
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.get_body_mut(handle),
            EngineInner::Avbd(e) => e.get_body_mut(handle),
        }
    }

    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Option<JointHandle> {
        if let Some(s) = &mut self.split {
            let h = s.add_joint(body_a, body_b, kind)?;
            self.structural_dirty = true;
            self.wake_set.insert(body_a);
            self.wake_set.insert(body_b);
            return Some(h);
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.add_joint(body_a, body_b, kind),
            EngineInner::Avbd(e) => e.add_joint(body_a, body_b, kind),
        }
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        if handle >= self.joint_count() {
            return;
        }
        if let Some(s) = &mut self.split {
            let j = s.joints[handle].state;
            self.wake_set.insert(j.a);
            self.wake_set.insert(j.b);
            Self::split_remove_joint(s, handle);
            self.structural_dirty = true;
            return;
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.remove_joint(handle),
            EngineInner::Avbd(e) => e.remove_joint(handle),
        }
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit> {
        if let Some(s) = &self.split {
            return s.raycast(ray, max_dist);
        }
        match &self.inner {
            EngineInner::SequentialImpulse(e) => e.raycast(ray, max_dist),
            EngineInner::Avbd(e) => e.raycast(ray, max_dist),
        }
    }

    fn shapecast(&self, shape: &Shape, from: glam::Vec3, to: glam::Vec3) -> Option<RaycastHit> {
        if let Some(s) = &self.split {
            return s.shapecast(shape, from, to);
        }
        match &self.inner {
            EngineInner::SequentialImpulse(e) => e.shapecast(shape, from, to),
            EngineInner::Avbd(e) => e.shapecast(shape, from, to),
        }
    }

    fn drain_trigger_events(&mut self) -> Vec<TriggerEvent> {
        let mut events = std::mem::take(&mut self.trigger_events);
        if self.routing == RoutingKind::Single {
            events.extend(match &mut self.inner {
                EngineInner::SequentialImpulse(e) => e.drain_trigger_events(),
                EngineInner::Avbd(e) => e.drain_trigger_events(),
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
            if handle < s.bodies.len() {
                self.wake_set.insert(handle);
                s.wake_global(handle);
            }
            return;
        }
        match &mut self.inner {
            EngineInner::SequentialImpulse(e) => e.wake_body(handle),
            EngineInner::Avbd(e) => e.wake_body(handle),
        }
    }
}
