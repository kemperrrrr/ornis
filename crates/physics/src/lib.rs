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
    CONTACT_BEGIN_SLOP, CONTACT_HIT_THRESHOLD, ContactEvent, ContactEventKind, TriggerEvent,
    TriggerEventKind,
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
pub enum Engine {
    /// Active builtin solver (boxed: the builtin state dwarfs AVBD's).
    Builtin(Box<BuiltinPhysicsEngine>),
    /// Active AVBD solver (boxed: keeps the seam enum pointer-sized).
    Avbd(Box<AvbdEngine>),
}

impl Engine {
    /// Empty orchestrator with the given solver and world-space gravity.
    pub fn new(kind: SolverKind, gravity: glam::Vec3) -> Self {
        match kind {
            SolverKind::Builtin => Self::Builtin(Box::new(BuiltinPhysicsEngine::new(gravity))),
            SolverKind::Avbd => Self::Avbd(Box::new(AvbdEngine::new(gravity))),
        }
    }

    /// Active solver kind.
    pub fn kind(&self) -> SolverKind {
        match self {
            Self::Builtin(_) => SolverKind::Builtin,
            Self::Avbd(_) => SolverKind::Avbd,
        }
    }

    /// Switch the active solver, migrating bodies and joints 1:1 (handles
    /// stay valid). Pending trigger/contact events are dropped with the
    /// old engine. No-op when already on `kind`.
    pub fn set_solver_kind(&mut self, kind: SolverKind, gravity: glam::Vec3) {
        if self.kind() == kind {
            return;
        }
        let (bodies, joints) = match self {
            Self::Builtin(e) => (e.bodies_snapshot(), e.joint_specs()),
            Self::Avbd(e) => (e.bodies_snapshot(), e.joint_specs()),
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
}

impl PhysicsEngine for Engine {
    fn step(&mut self, dt: f32) {
        match self {
            Self::Builtin(e) => e.step(dt),
            Self::Avbd(e) => e.step(dt),
        }
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        match self {
            Self::Builtin(e) => e.add_body(body),
            Self::Avbd(e) => e.add_body(body),
        }
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        match self {
            Self::Builtin(e) => e.remove_body(handle),
            Self::Avbd(e) => e.remove_body(handle),
        }
    }

    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        match self {
            Self::Builtin(e) => e.get_body(handle),
            Self::Avbd(e) => e.get_body(handle),
        }
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        match self {
            Self::Builtin(e) => e.get_body_mut(handle),
            Self::Avbd(e) => e.get_body_mut(handle),
        }
    }

    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Option<JointHandle> {
        match self {
            Self::Builtin(e) => e.add_joint(body_a, body_b, kind),
            Self::Avbd(e) => e.add_joint(body_a, body_b, kind),
        }
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        match self {
            Self::Builtin(e) => e.remove_joint(handle),
            Self::Avbd(e) => e.remove_joint(handle),
        }
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit> {
        match self {
            Self::Builtin(e) => e.raycast(ray, max_dist),
            Self::Avbd(e) => e.raycast(ray, max_dist),
        }
    }

    fn shapecast(&self, shape: &Shape, from: glam::Vec3, to: glam::Vec3) -> Option<RaycastHit> {
        match self {
            Self::Builtin(e) => e.shapecast(shape, from, to),
            Self::Avbd(e) => e.shapecast(shape, from, to),
        }
    }

    fn drain_trigger_events(&mut self) -> Vec<TriggerEvent> {
        match self {
            Self::Builtin(e) => e.drain_trigger_events(),
            Self::Avbd(e) => e.drain_trigger_events(),
        }
    }

    fn drain_contact_events(&mut self) -> Vec<ContactEvent> {
        match self {
            Self::Builtin(e) => e.drain_contact_events(),
            Self::Avbd(e) => e.drain_contact_events(),
        }
    }
}
