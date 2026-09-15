//! Ornis builtin rigid-body physics: dynamics, collision detection, contact
//! and joint solving.
//!
//! The crate is organized as a small CPU pipeline shared by the engine trait
//! ([`engine::PhysicsEngine`]) and its reference implementation
//! ([`engine::BuiltinPhysicsEngine`]):
//!
//! - [`body`] — rigid bodies and their handles/mass model.
//! - [`shape`] — convex primitives with AABB projection and inertia tensors.
//! - [`math`] — geometric queries used by broadphase and raycasts.
//! - `broadphase` — candidate-pair backends and benchmark diagnostics.
//! - [`joint`] — persistent equality constraints (ball/revolute).
//! - [`engine`] — the step pipeline: broadphase → narrowphase → island
//!   partitioning → substepped velocity/position solving, with optional
//!   SIMD-wide (`wide` module) and GPU (`gpu` feature) solver paths.
#![warn(missing_docs)]

mod broadphase;
mod broadphase_tree;

/// AVBD rigid-body engine: second [`engine::PhysicsEngine`] implementation
/// (M1, Genesis-style engine-level modularity).
pub mod avbd;
/// Rigid bodies: [`RigidBody`], mass model and body handles/types.
pub mod body;
pub(crate) mod distance;
/// The physics step pipeline and the [`crate::engine::PhysicsEngine`] trait.
pub mod engine;
/// GJK/EPA fallback narrow phase for cylinder, cone and convex hull pairs.
pub(crate) mod gjk;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
pub mod joint;
pub mod math;
/// Collision shapes with AABB projection and inertia tensors.
pub mod shape;
/// Trigger overlap event types emitted by the builtin physics engine.
pub mod trigger;
pub(crate) mod wide;

pub use avbd::AvbdEngine;
pub use body::{BodyHandle, BodyType, RigidBody};
pub use broadphase::{BroadPhaseKind, BroadPhaseStats, StepBudget, StepTiming};
pub use engine::{BuiltinPhysicsEngine, PhysicsEngine};
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
/// style): the builtin sequential-impulse engine or the AVBD engine.
/// Same [`PhysicsEngine`] seam, same scenes — the orchestrator
/// ([`Engine`]) migrates bodies and joints across the switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    /// Sequential-impulse engine with islands, sleep and substeps.
    Builtin,
    /// AVBD engine (position-level, single-thread sweep).
    Avbd,
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
pub struct Engine {
    inner: EngineInner,
    fracture_events: Vec<FractureEvent>,
    /// Contact events drained from the inner engine during `step`
    /// (the fracture pass consumes the inner queue, so the orchestrator
    /// re-serves them here — see `drain_contact_events`).
    contact_events: Vec<ContactEvent>,
}

/// Active solver behind the seam (boxed: both states dwarf the policy).
enum EngineInner {
    /// Sequential-impulse engine with islands, sleep and substeps.
    Builtin(Box<BuiltinPhysicsEngine>),
    /// AVBD engine (position-level, single-thread sweep).
    Avbd(Box<AvbdEngine>),
}

impl Engine {
    /// Empty orchestrator with the given solver and world-space gravity.
    pub fn new(kind: SolverKind, gravity: glam::Vec3) -> Self {
        let inner = match kind {
            SolverKind::Builtin => {
                EngineInner::Builtin(Box::new(BuiltinPhysicsEngine::new(gravity)))
            }
            SolverKind::Avbd => EngineInner::Avbd(Box::new(AvbdEngine::new(gravity))),
        };
        Self {
            inner,
            fracture_events: Vec::new(),
            contact_events: Vec::new(),
        }
    }

    /// Active solver kind.
    pub fn kind(&self) -> SolverKind {
        match self.inner {
            EngineInner::Builtin(_) => SolverKind::Builtin,
            EngineInner::Avbd(_) => SolverKind::Avbd,
        }
    }

    /// Switch the active solver, migrating bodies and joints 1:1 (handles
    /// stay valid). Pending trigger/contact/fracture events are dropped
    /// with the old engine. No-op when already on `kind`.
    pub fn set_solver_kind(&mut self, kind: SolverKind, gravity: glam::Vec3) {
        if self.kind() == kind {
            return;
        }
        let (bodies, joints) = match &self.inner {
            EngineInner::Builtin(e) => (e.bodies_snapshot(), e.joint_specs()),
            EngineInner::Avbd(e) => (e.bodies_snapshot(), e.joint_specs()),
        };
        let mut next = Self::new(kind, gravity);
        for body in bodies {
            next.add_body(body);
        }
        for (a, b, spec) in joints {
            // Gears reference joint handles, which are dense 0..n on both
            // sides here (re-added in order), so specs migrate verbatim.
            // A spec the target solver rejects (AVBD has no row model gap
            // left at M1-close, but future kinds may) is skipped, never
            // fatal — same discipline as stale gear references.
            let _ = next.add_joint(a, b, spec);
        }
        *self = next;
    }

    /// Drain fracture reports since the last call (see [`FractureEvent`]).
    pub fn drain_fracture_events(&mut self) -> Vec<FractureEvent> {
        std::mem::take(&mut self.fracture_events)
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
        let inner = &mut self.inner;
        // The inner queue is consumed here, so every event is stashed for
        // re-serve: fracture must never swallow the host's contact stream.
        let drained: Vec<ContactEvent> = match inner {
            EngineInner::Builtin(e) => e.drain_contact_events(),
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
                EngineInner::Builtin(e) => e.get_body(h).cloned(),
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
            match inner {
                EngineInner::Builtin(e) => e.remove_body(parent),
                EngineInner::Avbd(e) => e.remove_body(parent),
            }
            let (ha, hb) = match inner {
                EngineInner::Builtin(e) => (e.add_body(pa), e.add_body(pb)),
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
        match &mut self.inner {
            EngineInner::Builtin(e) => e.step(dt),
            EngineInner::Avbd(e) => e.step(dt),
        }
        self.fracture_pass();
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        match &mut self.inner {
            EngineInner::Builtin(e) => e.add_body(body),
            EngineInner::Avbd(e) => e.add_body(body),
        }
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        match &mut self.inner {
            EngineInner::Builtin(e) => e.remove_body(handle),
            EngineInner::Avbd(e) => e.remove_body(handle),
        }
    }

    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        match &self.inner {
            EngineInner::Builtin(e) => e.get_body(handle),
            EngineInner::Avbd(e) => e.get_body(handle),
        }
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        match &mut self.inner {
            EngineInner::Builtin(e) => e.get_body_mut(handle),
            EngineInner::Avbd(e) => e.get_body_mut(handle),
        }
    }

    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Option<JointHandle> {
        match &mut self.inner {
            EngineInner::Builtin(e) => e.add_joint(body_a, body_b, kind),
            EngineInner::Avbd(e) => e.add_joint(body_a, body_b, kind),
        }
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        match &mut self.inner {
            EngineInner::Builtin(e) => e.remove_joint(handle),
            EngineInner::Avbd(e) => e.remove_joint(handle),
        }
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit> {
        match &self.inner {
            EngineInner::Builtin(e) => e.raycast(ray, max_dist),
            EngineInner::Avbd(e) => e.raycast(ray, max_dist),
        }
    }

    fn shapecast(&self, shape: &Shape, from: glam::Vec3, to: glam::Vec3) -> Option<RaycastHit> {
        match &self.inner {
            EngineInner::Builtin(e) => e.shapecast(shape, from, to),
            EngineInner::Avbd(e) => e.shapecast(shape, from, to),
        }
    }

    fn drain_trigger_events(&mut self) -> Vec<TriggerEvent> {
        match &mut self.inner {
            EngineInner::Builtin(e) => e.drain_trigger_events(),
            EngineInner::Avbd(e) => e.drain_trigger_events(),
        }
    }

    fn drain_contact_events(&mut self) -> Vec<ContactEvent> {
        // Served from the orchestrator stash (filled by `step`): the
        // fracture pass sits between the inner queue and the host.
        std::mem::take(&mut self.contact_events)
    }
}
