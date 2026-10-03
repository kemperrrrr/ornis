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
use crate::errors::{JointError, QueryError};
use crate::joint::{JointHandle, JointKind};
use crate::math::{Ray, RaycastHit};
use crate::sequential_impulse::MAX_MANIFOLD_POINTS;
use crate::shape::Shape;
use crate::trigger::{ContactEvent, ContactForceEvent, TriggerEvent};

/// Physics engine trait: a single step of simulation, plus body/joint management
/// and queries. Implementations may be CPU or GPU-based, single-threaded or multi-threaded.
pub trait PhysicsEngine: Send + Sync {
    /// Advance the simulation by `dt` seconds: broadphase → narrowphase →
    /// island partitioning → substepped velocity/position solving (contacts,
    /// friction, joints) → integration. `dt` must be > 0 and finite.
    fn step(&mut self, dt: f32);
    /// Typed step entry point: advances by `dt` ([`ornis_core::units::Seconds`]).
    /// Non-positive or non-finite deltas are a no-op (same policy as the
    /// orchestrator's raw guard).
    fn step_seconds(&mut self, dt: ornis_core::units::Seconds) {
        use ornis_core::units::SecondsExt;
        if dt.is_valid_step() {
            self.step(dt.get());
        }
    }
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
    ///
    /// # Errors
    ///
    /// [`JointError`] naming the flaw: bad handles, self-joint,
    /// non-finite/bad bounds/axes, unknown gear refs, or a kind the
    /// solver does not support.
    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Result<JointHandle, JointError>;
    /// Destroy a joint by handle; no-op for an invalid handle.
    fn remove_joint(&mut self, handle: JointHandle);
    /// Closest exact shape hit of `ray` against registered bodies within
    /// `max_dist` (in units of the ray direction's length).
    /// `Ok(None)` is a clean miss; `Err` is an invalid input
    /// (non-finite/zero direction, bad `max_dist`). Pass a normalized
    /// direction for world-distance units.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidInput`] on degenerate queries.
    fn raycast(&self, ray: Ray, max_dist: f32) -> Result<Option<RaycastHit>, QueryError>;
    /// Typed raycast entry point: cutoff as [`ornis_core::units::Meters`].
    /// Negative or non-finite cutoffs report no hit.
    fn raycast_distance(
        &self,
        ray: Ray,
        max_dist: ornis_core::units::Meters,
    ) -> Option<RaycastHit> {
        let d = max_dist.get();
        if d.is_finite() && d >= 0.0 {
            self.raycast(ray, d).unwrap_or(None)
        } else {
            None
        }
    }
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
    /// Drain per-pair contact-force reports produced by completed steps
    /// (Rapier `CONTACT_FORCE_EVENTS` parity). Empty by default and empty
    /// unless a body opts in via its contact-force threshold; the
    /// sequential-impulse engine reports them in deterministic canonical
    /// pair order.
    fn drain_contact_force_events(&mut self) -> Vec<ContactForceEvent> {
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

/// Contact manifold: one normal + up to [`MAX_MANIFOLD_POINTS`] points per body pair.
#[derive(Clone, Debug)]
pub struct Manifold {
    /// First body handle.
    pub body_a: BodyHandle,
    /// Second body handle.
    pub body_b: BodyHandle,
    /// Contact normal (body A to body B).
    pub normal: Vec3,
    /// Active point count (`1..=MAX_MANIFOLD_POINTS`). Enforced by the checked
    /// constructors ([`Manifold::from_parts`], [`Manifold::from_nonempty`],
    /// [`Manifold::try_from_points`]); direct writes bypass the invariant.
    pub point_count: usize,
    /// Contact points (only the first `point_count` are live).
    pub points: [ManifoldPoint; MAX_MANIFOLD_POINTS],
}

impl Manifold {
    pub(crate) fn single(body_a: BodyHandle, body_b: BodyHandle, c: Contact) -> Self {
        let point = ManifoldPoint {
            world_point: c.contact_point,
            penetration: c.penetration,
        };
        Self::from_nonempty(
            body_a,
            body_b,
            c.normal,
            crate::invariants::NonEmpty4::single(point),
        )
    }

    /// Checked constructor from a non-empty capped point set: the
    /// `point_count`/`points` pair is built in one place, so the
    /// count-invariant cannot drift.
    pub fn from_nonempty(
        body_a: BodyHandle,
        body_b: BodyHandle,
        normal: Vec3,
        points: crate::invariants::NonEmpty4<ManifoldPoint>,
    ) -> Self {
        let mut buf = [ManifoldPoint {
            world_point: Vec3::ZERO,
            penetration: 0.0,
        }; MAX_MANIFOLD_POINTS];
        for (k, p) in points.iter().enumerate() {
            buf[k] = p;
        }
        Self {
            body_a,
            body_b,
            normal,
            point_count: points.len(),
            points: buf,
        }
    }

    /// Checked constructor from a slice: `None` when empty or longer than
    /// [`MAX_MANIFOLD_POINTS`].
    pub fn try_from_points(
        body_a: BodyHandle,
        body_b: BodyHandle,
        normal: Vec3,
        points: &[ManifoldPoint],
    ) -> Option<Self> {
        crate::invariants::NonEmpty4::try_from_slice(points)
            .map(|v| Self::from_nonempty(body_a, body_b, normal, v))
    }

    /// Checked constructor from a raw buffer plus count: `None` unless
    /// `count` is in `1..=MAX_MANIFOLD_POINTS`. Single validation site for
    /// narrow-phase code that fills the buffer by hand (only the first
    /// `count` entries are live).
    pub fn from_parts(
        body_a: BodyHandle,
        body_b: BodyHandle,
        normal: Vec3,
        points: [ManifoldPoint; MAX_MANIFOLD_POINTS],
        count: usize,
    ) -> Option<Self> {
        if !(1..=MAX_MANIFOLD_POINTS).contains(&count) {
            return None;
        }
        Some(Self {
            body_a,
            body_b,
            normal,
            point_count: count,
            points,
        })
    }

    /// Live contact points (`points[..point_count]`).
    pub fn points_slice(&self) -> &[ManifoldPoint] {
        &self.points[..self.point_count.min(MAX_MANIFOLD_POINTS)]
    }

    /// Typed view of the live points as a [`crate::invariants::NonEmpty4`]:
    /// `None` when the `point_count` invariant is broken.
    pub fn nonempty(&self) -> Option<crate::invariants::NonEmpty4<ManifoldPoint>> {
        crate::invariants::NonEmpty4::try_from_slice(self.points_slice())
    }

    /// Whether `point_count` satisfies the 1..=[`MAX_MANIFOLD_POINTS`] invariant.
    pub fn has_valid_count(&self) -> bool {
        (1..=MAX_MANIFOLD_POINTS).contains(&self.point_count)
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
    ContactHooks, ContactPointView, ContactView, ContinuousHit, ManifoldState, ModifyContext,
    NarrowShardPool, OneWayPlatform, PairFilterContext, SatCache, SatCacheEntry,
    SequentialImpulseEngine, SolverFlags, apply_impulse, box_manifold, ccd_impact_velocity,
    detect_collisions_into, effective_mass, find_angular_continuous_hit, inv_inertia_axis,
    kinematic_cast, mul_inv_inertia, obb_sat, point_velocity, remove_angular_approach,
    solve_normal_block, solve_small, sweep_gap,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32) -> ManifoldPoint {
        ManifoldPoint {
            world_point: Vec3::new(x, 0.0, 0.0),
            penetration: 0.01,
        }
    }

    /// `from_parts` admits exactly `1..=4` and the typed view mirrors it.
    #[test]
    fn from_parts_enforces_count_invariant() {
        let buf = [point(0.0), point(1.0), point(2.0), point(3.0)];
        let (a, b) = (BodyHandle::from_raw(0), BodyHandle::from_raw(1));
        assert!(Manifold::from_parts(a, b, Vec3::Y, buf, 0).is_none());
        assert!(Manifold::from_parts(a, b, Vec3::Y, buf, 5).is_none());
        let m = Manifold::from_parts(a, b, Vec3::Y, buf, 2).expect("1..=4 builds");
        assert_eq!(m.point_count, 2);
        assert_eq!(m.points_slice().len(), 2);
        assert_eq!(m.nonempty().expect("valid count").len(), 2);
        assert!(m.has_valid_count());
    }

    /// The typed view catches a drifted `point_count` instead of trusting it.
    #[test]
    fn nonempty_rejects_drifted_count() {
        let mut m = Manifold::single(
            BodyHandle::from_raw(0),
            BodyHandle::from_raw(1),
            Contact {
                normal: Vec3::Y,
                penetration: 0.01,
                contact_point: Vec3::ZERO,
            },
        );
        assert!(m.nonempty().is_some());
        m.point_count = 0;
        assert!(!m.has_valid_count());
        assert!(m.nonempty().is_none());
        assert!(m.points_slice().is_empty());
    }
}
