//! Sequential-impulse (projected Gauss-Seidel, Catto-Box2D-Jolt class)
//! physics engine: trait + CPU implementation.
//!
//! [`PhysicsEngine`] defines a single simulation step (broadphase →
//! narrowphase → island partitioning → substepped contact/joint solving →
//! integration; `dt` must be positive and finite) plus body/joint
//! management and ray/shape cast queries. [`SequentialImpulseEngine`] is the
//! reference implementation: velocity iterations + friction/restitution +
//! Baumgarte positional correction, parallelized with rayon, with optional GPU
//! contact solving behind the `gpu` feature.

use glam::Vec3;

use crate::body::{BodyHandle, RigidBody};
use crate::joint::{JointHandle, JointKind};
use crate::math::{Ray, RaycastHit};
use crate::shape::Shape;
use crate::trigger::{ContactEvent, TriggerEvent};

/// Physics engine trait: a single step of simulation, plus body/joint management
/// and queries. Implementations may be CPU or GPU-based, single-threaded or multi-threaded.
pub trait PhysicsEngine: Send + Sync {
    /// Advance the simulation by `dt` seconds: broadphase → narrowphase →
    /// island partitioning → substepped velocity/position solving (contacts,
    /// friction, joints) → integration. `dt` must be > 0 and finite.
    fn step(&mut self, dt: f32);
    /// Register a body and return its stable handle.
    fn add_body(&mut self, body: RigidBody) -> BodyHandle;
    /// Remove a body, swapping the final body into its slot. The moved body's
    /// handle changes; joints on the removed body and their dependent gears
    /// are destroyed. Invalid handles are a no-op.
    fn remove_body(&mut self, handle: BodyHandle);
    /// Read-only access to a body, or `None` for an invalid handle.
    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody>;
    /// Mutable access to a body, or `None` for an invalid handle. Direct
    /// pose edits take effect at the next [`PhysicsEngine::step`].
    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody>;
    /// Create a joint between two existing, distinct bodies (G5).
    /// Returns None on invalid handles or a self-joint.
    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Option<JointHandle>;
    /// Destroy a joint by handle; no-op for an invalid handle.
    fn remove_joint(&mut self, handle: JointHandle);
    /// Closest exact shape hit of `ray` against registered bodies within
    /// `max_dist` (in units of the ray direction's length), or `None` if
    /// nothing is hit. Pass a normalized direction for world-distance units.
    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit>;
    /// Sweep `shape` along the segment `from → to` and report the first body
    /// hit (hit distance measured along the sweep direction), or `None`.
    fn shapecast(&self, shape: &Shape, from: Vec3, to: Vec3) -> Option<RaycastHit>;
    /// Drain trigger enter/exit transitions produced by completed steps.
    ///
    /// Engines without trigger support may keep the default empty result;
    /// the sequential-impulse engine reports canonical body-handle pairs in deterministic
    /// order.
    fn drain_trigger_events(&mut self) -> Vec<TriggerEvent> {
        Vec::new()
    }
    /// Drain solid-contact begin/end/hit transitions produced by completed
    /// steps (Box3D `b3ContactEvents` parity). Empty by default; the sequential-impulse
    /// engine reports them in deterministic pair order.
    fn drain_contact_events(&mut self) -> Vec<ContactEvent> {
        Vec::new()
    }
    /// Wake a sleeping body/island (default: no-op). Host edits through
    /// [`PhysicsEngine::get_body_mut`] do not wake every solver by
    /// themselves — the M3 orchestrator wakes edited bodies explicitly
    /// before stepping. Must be a no-op for invalid handles.
    fn wake_body(&mut self, _handle: BodyHandle) {}
}

pub(crate) struct Contact {
    pub(crate) normal: Vec3,
    pub(crate) penetration: f32,
    pub(crate) contact_point: Vec3,
}

/// A single contact point inside a manifold (G2).
#[derive(Clone, Copy, Debug)]
pub struct ManifoldPoint {
    /// Contact point in world space.
    pub world_point: Vec3,
    /// Penetration depth at the point.
    pub penetration: f32,
}

/// Contact manifold: one normal + up to 4 points per body pair.
#[derive(Clone, Debug)]
pub struct Manifold {
    /// First body handle.
    pub body_a: BodyHandle,
    /// Second body handle.
    pub body_b: BodyHandle,
    /// Contact normal (body A to body B).
    pub normal: Vec3,
    /// Active point count (1..=4).
    pub point_count: usize,
    /// Contact points (only the first `point_count` are live).
    pub points: [ManifoldPoint; 4],
}

impl Manifold {
    pub(crate) fn single(body_a: BodyHandle, body_b: BodyHandle, c: Contact) -> Self {
        let mut points = [ManifoldPoint {
            world_point: Vec3::ZERO,
            penetration: 0.0,
        }; 4];
        points[0] = ManifoldPoint {
            world_point: c.contact_point,
            penetration: c.penetration,
        };
        Self {
            body_a,
            body_b,
            normal: c.normal,
            point_count: 1,
            points,
        }
    }
}
/// Joint constraint kernels (hinge twist, coordinates, sub-solvers) shared
/// by the CPU island path and the GPU batch path.
pub use crate::sequential_impulse::joints;

pub(crate) use crate::sequential_impulse::raycast_shape_hit;
/// Solver surface re-exported from the `sequential_impulse` subsystem
/// (Genesis-style `solvers/rigid/` box): the engine, narrowphase, caches,
/// solver-state and query kernels. [`PhysicsEngine`], [`Contact`] and
/// [`Manifold`] stay defined here.
pub use crate::sequential_impulse::{
    ContinuousHit, ManifoldState, NarrowShardPool, SatCache, SatCacheEntry,
    SequentialImpulseEngine, apply_impulse, box_manifold, ccd_impact_velocity,
    detect_collisions_into, effective_mass, find_angular_continuous_hit, inv_inertia_axis,
    kinematic_cast, mul_inv_inertia, obb_sat, point_velocity, remove_angular_approach,
    solve_normal_block, solve_small, sweep_gap,
};
