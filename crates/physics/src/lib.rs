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
    AxisConfig, JointHandle, JointKind, PrismaticLimit, PrismaticMotor, RevoluteLimit,
    RevoluteMotor, WheelSuspension,
};
pub use math::{AABB, Ray, RaycastHit};
pub use shape::{ConvexHull, Heightfield, Shape, TriMesh};
pub use trigger::{
    CONTACT_BEGIN_SLOP, CONTACT_HIT_THRESHOLD, ContactEvent, ContactEventKind, TriggerEvent,
    TriggerEventKind,
};
