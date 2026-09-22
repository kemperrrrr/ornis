//! Deterministic AABB/Ray invariant checks (ported from
//! `crates/core/tests/property_tests.rs`, which pulled the whole
//! `ornis-physics` crate into `ornis-core` dev-dependencies just for these
//! four cases).
//!
//! The originals were `proptest` cases (64 random inputs each); this crate
//! has no `proptest` dev-dependency, so the same invariants run over a
//! fixed LCG stream with identical input ranges (`±1e4` coordinates,
//! `±1e3` ray parameters, 1..32 points per cloud). No randomness, no new
//! dependencies — `cargo test -p ornis-physics` covers them.

use glam::Vec3;
use ornis_physics::math::{AABB, Ray};

/// Minimal LCG (`pcg`-free, `no_std`-friendly): enough for deterministic
/// pseudo-random coordinates without pulling `proptest`/`rand`.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn next_f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        // 24-bit mantissa worth of uniformity — plenty for containment/
        // commutativity/linearity invariants.
        let bits = (self.next_u64() >> 11) as f32 / (u64::MAX >> 11) as f32;
        lo + bits * (hi - lo)
    }

    fn next_vec3(&mut self) -> Vec3 {
        Vec3::new(
            self.next_f32_in(-1e4, 1e4),
            self.next_f32_in(-1e4, 1e4),
            self.next_f32_in(-1e4, 1e4),
        )
    }
}

const CASES: usize = 64;

#[test]
fn aabb_from_points_contains_all() {
    let mut rng = Lcg(0x1EAF_AE5E_9B3D_1C77);
    for _ in 0..CASES {
        let n = 1 + (rng.next_u64() % 32) as usize;
        let points: Vec<Vec3> = (0..n).map(|_| rng.next_vec3()).collect();
        let aabb = AABB::from_points(&points);
        for p in &points {
            assert!(aabb.contains_point(*p), "from_points must contain {p:?}");
        }
    }
}

#[test]
fn aabb_expand_keeps_contents() {
    let mut rng = Lcg(0x5EED_1234_ABCD_0001);
    for _ in 0..CASES {
        let n = 1 + (rng.next_u64() % 32) as usize;
        let points: Vec<Vec3> = (0..n).map(|_| rng.next_vec3()).collect();
        let extra = rng.next_vec3();
        let mut aabb = AABB::from_points(&points);
        aabb.expand(extra);
        for p in points.iter().chain(std::iter::once(&extra)) {
            assert!(aabb.contains_point(*p), "expand must keep {p:?}");
        }
    }
}

#[test]
fn aabb_overlaps_is_commutative() {
    let mut rng = Lcg(0xC0DE_11FE_0000_0001);
    for _ in 0..CASES {
        let a_min = rng.next_vec3();
        let a = AABB::new(a_min, a_min + rng.next_vec3().abs());
        let b_min = rng.next_vec3();
        let b = AABB::new(b_min, b_min + rng.next_vec3().abs());
        assert_eq!(
            a.overlaps(&b),
            b.overlaps(&a),
            "overlaps must commute for {a:?} vs {b:?}"
        );
    }
}

#[test]
fn ray_point_at_is_linear() {
    let mut rng = Lcg(0x8A4A_9E10_0000_0007);
    for _ in 0..CASES {
        let origin = rng.next_vec3();
        let direction = rng.next_vec3();
        let t = rng.next_f32_in(-1e3, 1e3);
        let ray = Ray::new(origin, direction);
        let expected = origin + direction * t;
        let got = ray.point_at(t);
        let eps = 1e-3 * (1.0 + expected.length());
        assert!(
            (got - expected).length() <= eps,
            "point_at({t}) drifted: got {got:?}, expected {expected:?}"
        );
    }
}
