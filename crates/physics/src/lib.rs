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
//! - [`joint`] — persistent equality constraints (ball/revolute).
//! - [`engine`] — the step pipeline: broadphase → narrowphase → island
//!   partitioning → substepped velocity/position solving, with optional
//!   SIMD-wide (`wide` module) and GPU (`gpu` feature) solver paths.
#![warn(missing_docs)]

mod broadphase;

/// Rigid bodies: [`RigidBody`], mass model and body handles/types.
pub mod body;
pub(crate) mod distance;
/// The physics step pipeline and the [`crate::engine::PhysicsEngine`] trait.
pub mod engine;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
pub mod joint;
pub mod math;
/// Collision shapes with AABB projection and inertia tensors.
pub mod shape;
/// Trigger overlap event types emitted by the builtin physics engine.
pub mod trigger;
pub(crate) mod wide;

pub use body::{BodyHandle, BodyType, RigidBody};
pub use broadphase::BroadPhaseKind;
pub use engine::{BuiltinPhysicsEngine, PhysicsEngine};
pub use joint::{JointHandle, JointKind};
pub use math::{AABB, Ray, RaycastHit};
pub use shape::Shape;
pub use trigger::{TriggerEvent, TriggerEventKind};
