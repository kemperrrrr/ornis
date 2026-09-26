//! Axis-aligned bounding boxes and rays — the geometric query primitives
//! shared by broadphase culling, sleep-region checks and raycasts.
//!
//! `AABB` is the unit used for swept broadphase (`broadphase::swept_aabbs`,
//! incremental `UniformGrid` with `body_cells`/`prev_meta`), `Ray`/`RaycastHit`
//! back the exact sphere/OBB/capsule ray intersections and the analytic
//! `distance::cast_shape` TOI path.

use crate::errors::MeshError;

pub use glam::Vec3;

/// World-space axis-aligned bounding box, stored by its two extreme corners.
///
/// Invariant: every component of `min` is `<=` the matching component of
/// `max` (a degenerate zero-volume box is valid and used as a point seed).
/// All containment/overlap tests are boundary-inclusive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    /// Corner with the smallest coordinates on every axis.
    pub min: Vec3,
    /// Corner with the largest coordinates on every axis.
    pub max: Vec3,
}

impl AABB {
    /// Box spanning exactly the two given corners (components are swapped
    /// per-axis only if the caller passes them out of order — no normalization).
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Degenerate zero-volume box at a single point; a seed for [`AABB::expand`].
    pub fn from_point(point: Vec3) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    /// Smallest box containing all points.
    ///
    /// Legacy wrapper over [`AABB::try_from_points`]: panics on an empty
    /// slice so existing call sites stay bit-identical; new code should
    /// match on the typed error instead (deprecated — do not use in new
    /// code, kept only for compat).
    ///
    /// # Panics
    ///
    /// Panics when `points` is empty (see [`AABB::try_from_points`]
    /// for the fallible canonical path and its `# Errors`).
    pub fn from_points(points: &[Vec3]) -> Self {
        Self::try_from_points(points).expect("AABB::from_points needs at least one point")
    }

    /// Fallible smallest box containing all points.
    ///
    /// # Errors
    ///
    /// [`MeshError::EmptyPoints`] when `points` is empty.
    pub fn try_from_points(points: &[Vec3]) -> Result<Self, MeshError> {
        let [first, rest @ ..] = points else {
            return Err(MeshError::EmptyPoints);
        };
        let mut aabb = Self::from_point(*first);
        for p in rest {
            aabb.expand(*p);
        }
        Ok(aabb)
    }

    /// Grow the box to include `point` (per-axis min/max); never shrinks.
    pub fn expand(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Midpoint of the two corners.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half of the full extent on each axis (`(max - min) * 0.5`).
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Whether the boxes intersect (touching faces count as overlapping).
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Boundary-inclusive containment test: a point exactly on a face counts.
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }
}

/// An infinite half-line: `origin + t * direction` for `t >= 0`.
///
/// Contract: callers are expected to pass a normalized `direction`, since
/// [`RaycastHit::distance`] is measured in units of `direction` length.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// Starting point of the ray.
    pub origin: Vec3,
    /// Travel direction (conventionally unit length).
    pub direction: Vec3,
}

impl Ray {
    /// Ray from `origin` toward `direction`.
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    /// World-space point at ray parameter `t` (`t = 0` is the origin).
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Result of a ray or shape cast that hit a body.
#[derive(Debug, Clone, Copy)]
pub struct RaycastHit {
    /// Handle of the body that was hit.
    pub handle: crate::body::BodyHandle,
    /// World-space point of first contact.
    pub point: Vec3,
    /// Surface normal at the hit point, pointing out of the surface
    /// (against the ray direction).
    pub normal: Vec3,
    /// Distance along the cast direction from the origin to the hit,
    /// in units of the direction vector's length.
    pub distance: f32,
}

/// Deterministic tangent frame for a unit normal: `t1` is the normal
/// crossed with a fixed reference axis (no exact-equality branches, so
/// near-axis normals don't flicker), `t2` completes the frame. The single
/// canonical copy — the engine, AVBD and GPU-batch solvers previously
/// carried their own identical versions.
pub fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    let axis = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let t1 = n.cross(axis).normalize_or(Vec3::Z);
    (t1, t1.cross(n))
}

/// Tangent frame for a statically checked unit normal.
///
/// Takes [`crate::invariants::UnitVec3`] so the unit-length precondition
/// is enforced at construction time instead of assumed at the call site.
pub fn tangent_basis_unit(n: crate::invariants::UnitVec3) -> (Vec3, Vec3) {
    tangent_basis(n.get())
}

/// Explicit fallible tangent frame: `None` for zero/non-finite/non-unit
/// input instead of silently substituting a default axis.
pub fn tangent_basis_checked(n: Vec3) -> Option<(Vec3, Vec3)> {
    crate::invariants::UnitVec3::try_from(n).map(tangent_basis_unit)
}

/// Wheel/joint axle orthogonalized against its reference axis (SI
/// creation rule): a near-parallel axle gets a deterministic
/// perpendicular fallback from [`tangent_basis`], never a NaN.
pub fn orthogonalize_axle(suspension: Vec3, axle: Vec3) -> Vec3 {
    let a = axle - suspension * axle.dot(suspension);
    if a.length_squared() < 1e-6 {
        tangent_basis(suspension).0
    } else {
        a.normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `math.rs` had zero unit tests (night gate, 2026-08-24 — 22 missed
    /// mutants). Pins the exact arithmetic (`min`/`max`, `+`/`-`/`*0.5`,
    /// `<=`/`>=` boundary comparisons) rather than just smoke-testing.
    const EPS: f32 = 1e-6;

    #[test]
    fn aabb_from_point_is_a_degenerate_box() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let aabb = AABB::from_point(p);
        assert_eq!(aabb.min, p);
        assert_eq!(aabb.max, p);
    }

    #[test]
    fn aabb_expand_grows_min_and_max_independently() {
        let mut aabb = AABB::from_point(Vec3::new(0.0, 0.0, 0.0));
        aabb.expand(Vec3::new(-1.0, 5.0, 0.0));
        assert_eq!(aabb.min, Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(aabb.max, Vec3::new(0.0, 5.0, 0.0));
        // A point already inside must not change the bounds.
        aabb.expand(Vec3::new(-0.5, 2.0, 0.0));
        assert_eq!(aabb.min, Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(aabb.max, Vec3::new(0.0, 5.0, 0.0));
    }

    #[test]
    fn aabb_from_points_folds_expand_over_the_slice() {
        let pts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, -1.0, 0.0),
            Vec3::new(-3.0, 4.0, 1.0),
        ];
        let aabb = AABB::from_points(&pts);
        assert_eq!(aabb.min, Vec3::new(-3.0, -1.0, 0.0));
        assert_eq!(aabb.max, Vec3::new(2.0, 4.0, 1.0));
    }

    #[test]
    fn aabb_center_and_half_extents_match_min_max() {
        let aabb = AABB::new(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(3.0, 4.0, 5.0));
        assert!((aabb.center() - Vec3::new(1.0, 1.0, 1.0)).length() < EPS);
        assert!((aabb.half_extents() - Vec3::new(2.0, 3.0, 4.0)).length() < EPS);
    }

    #[test]
    fn aabb_overlaps_is_symmetric_and_boundary_inclusive() {
        let a = AABB::new(Vec3::ZERO, Vec3::splat(1.0));
        let touching = AABB::new(Vec3::splat(1.0), Vec3::splat(2.0));
        let separated = AABB::new(Vec3::splat(1.001), Vec3::splat(2.0));
        // Touching at exactly one face counts as overlapping (`<=`/`>=`,
        // not `<`/`>`) — this is the boundary a `<=`<->`<` mutant flips.
        assert!(a.overlaps(&touching));
        assert!(touching.overlaps(&a), "overlaps must be symmetric");
        assert!(!a.overlaps(&separated));
        assert!(!separated.overlaps(&a));
    }

    #[test]
    fn aabb_contains_point_boundary_inclusive() {
        let aabb = AABB::new(Vec3::ZERO, Vec3::splat(2.0));
        assert!(aabb.contains_point(Vec3::new(1.0, 1.0, 1.0)));
        // On every face exactly — must count as contained.
        assert!(aabb.contains_point(Vec3::new(0.0, 1.0, 1.0)));
        assert!(aabb.contains_point(Vec3::new(2.0, 1.0, 1.0)));
        // Just outside on a single axis must NOT be contained.
        assert!(!aabb.contains_point(Vec3::new(2.0001, 1.0, 1.0)));
        assert!(!aabb.contains_point(Vec3::new(1.0, -0.0001, 1.0)));
    }

    #[test]
    fn aabb_try_from_points_rejects_empty_slice() {
        assert_eq!(
            AABB::try_from_points(&[]),
            Err(crate::errors::MeshError::EmptyPoints)
        );
        let pts = [Vec3::new(1.0, 2.0, 3.0)];
        assert_eq!(AABB::try_from_points(&pts), Ok(AABB::from_point(pts[0])));
    }

    #[test]
    fn ray_point_at_moves_along_direction_scaled_by_t() {
        let ray = Ray::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0));
        assert_eq!(ray.point_at(0.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(ray.point_at(1.0), Vec3::new(1.0, 2.0, 0.0));
        assert_eq!(ray.point_at(0.5), Vec3::new(1.0, 1.0, 0.0));
    }
}
