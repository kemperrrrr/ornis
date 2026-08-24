pub use glam::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_point(point: Vec3) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    pub fn from_points(points: &[Vec3]) -> Self {
        let mut aabb = Self::from_point(points[0]);
        for p in &points[1..] {
            aabb.expand(*p);
        }
        aabb
    }

    pub fn expand(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    pub fn point_at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RaycastHit {
    pub handle: usize,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
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
    fn ray_point_at_moves_along_direction_scaled_by_t() {
        let ray = Ray::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0));
        assert_eq!(ray.point_at(0.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(ray.point_at(1.0), Vec3::new(1.0, 2.0, 0.0));
        assert_eq!(ray.point_at(0.5), Vec3::new(1.0, 1.0, 0.0));
    }
}
