//! Analytic closest-point / distance queries between the engine's convex
//! shapes (G6). These power the honest `shapecast` (conservative advancement)
//! and the TOI pass: every query returns the surface-to-surface distance and
//! witness points. Exact (not sampled) — casts can never tunnel through
//! geometry thinner than the step length.

use glam::{Quat, Vec3};

use crate::shape::Shape;

/// A placed shape: geometry plus world transform.
#[derive(Clone, Copy)]
pub(crate) struct ShapeRef<'a> {
    pub shape: &'a Shape,
    pub pos: Vec3,
    pub rot: Quat,
}

/// Result of a pairwise distance query: surface distance and witness points
/// on each shape (`point_a` lies on the first shape's surface).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Distance {
    pub dist: f32,
    pub point_a: Vec3,
    pub point_b: Vec3,
}

/// Closest point on segment [a, b] to point `p`.
fn point_segment_closest(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-12 {
        return a;
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + ab * t
}

/// Closest points between segments [a0, a1] and [b0, b1] (Ericson's
/// two-pass clamped parametrization).
fn seg_seg_closest(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> (Vec3, Vec3) {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    // Both segments degenerate to points.
    if a < 1e-12 && e < 1e-12 {
        return (a0, b0);
    }
    if a < 1e-12 {
        let t = (f / e).clamp(0.0, 1.0);
        return (a0, b0 + d2 * t);
    }
    let c = d1.dot(r);
    if e < 1e-12 {
        let t = (-c / a).clamp(0.0, 1.0);
        return (a0 + d1 * t, b0);
    }
    let b = d1.dot(d2);
    let denom = a * e - b * b;
    let mut s = if denom.abs() > 1e-12 {
        ((b * f - c * e) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = if e > 1e-12 { (b * s + f) / e } else { 0.0 };
    if !(0.0..=1.0).contains(&t) {
        t = t.clamp(0.0, 1.0);
        s = ((b * t - c) / a).clamp(0.0, 1.0);
    }
    (a0 + d1 * s, b0 + d2 * t)
}

/// Closest point on an OBB to world point `p`. Interior points are pushed
/// out to the nearest face (distance then reads as penetration depth).
fn point_obb_closest(p: Vec3, pos: Vec3, half: Vec3, rot: Quat) -> Vec3 {
    let l = rot.inverse() * (p - pos);
    let clamped = l.clamp(-half, half);
    let q = if clamped == l {
        // Interior: snap the axis with the smallest face distance.
        let dx = half.x - l.x.abs();
        let dy = half.y - l.y.abs();
        let dz = half.z - l.z.abs();
        let mut q = l;
        if dx <= dy && dx <= dz {
            q.x = half.x.copysign(l.x);
        } else if dy <= dz {
            q.y = half.y.copysign(l.y);
        } else {
            q.z = half.z.copysign(l.z);
        }
        q
    } else {
        clamped
    };
    pos + rot * q
}

/// Eight world-space corners of an OBB. Index bits: bit2 = x, bit1 = y,
/// bit0 = z (0 = positive side).
fn obb_corners(pos: Vec3, half: Vec3, rot: Quat) -> [Vec3; 8] {
    let x = rot * (Vec3::X * half.x);
    let y = rot * (Vec3::Y * half.y);
    let z = rot * (Vec3::Z * half.z);
    [
        pos + x + y + z,
        pos + x + y - z,
        pos + x - y + z,
        pos + x - y - z,
        pos - x + y + z,
        pos - x + y - z,
        pos - x - y + z,
        pos - x - y - z,
    ]
}

/// The 12 edges of a box as corner-index pairs (differ in exactly one bit).
const OBB_EDGES: [(usize, usize); 12] = [
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7), // along X
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7), // along Y
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7), // along Z
];

/// Capsule core segment endpoints (world space).
fn capsule_segment(pos: Vec3, half_height: f32, rot: Quat) -> (Vec3, Vec3) {
    let axis = rot * Vec3::Y;
    (pos - axis * half_height, pos + axis * half_height)
}

fn sphere_sphere(a: ShapeRef, ra: f32, b: ShapeRef, rb: f32) -> Distance {
    let d_vec = b.pos - a.pos;
    let center_dist = d_vec.length();
    let n = d_vec.normalize_or(Vec3::X);
    Distance {
        dist: center_dist - ra - rb,
        point_a: a.pos + n * ra,
        point_b: b.pos - n * rb,
    }
}

/// Sphere (a) vs OBB (b).
fn sphere_obb(a: ShapeRef, r: f32, b: ShapeRef, half: Vec3) -> Distance {
    let pb = point_obb_closest(a.pos, b.pos, half, b.rot);
    let d_vec = a.pos - pb;
    let core = d_vec.length();
    let n = d_vec.normalize_or(Vec3::X); // from box surface toward sphere
    Distance {
        dist: core - r,
        point_a: a.pos - n * r,
        point_b: pb,
    }
}

/// Sphere (a) vs capsule (b).
fn sphere_capsule(a: ShapeRef, r: f32, b: ShapeRef, cr: f32, hh: f32) -> Distance {
    let (s0, s1) = capsule_segment(b.pos, hh, b.rot);
    let core_b = point_segment_closest(a.pos, s0, s1);
    let d_vec = a.pos - core_b;
    let core = d_vec.length();
    let n = d_vec.normalize_or(Vec3::X);
    Distance {
        dist: core - r - cr,
        point_a: a.pos - n * r,
        point_b: core_b + n * cr,
    }
}

/// Capsule (a) vs capsule (b).
fn capsule_capsule(a: ShapeRef, ra: f32, ha: f32, b: ShapeRef, rb: f32, hb: f32) -> Distance {
    let (a0, a1) = capsule_segment(a.pos, ha, a.rot);
    let (b0, b1) = capsule_segment(b.pos, hb, b.rot);
    let (ca, cb) = seg_seg_closest(a0, a1, b0, b1);
    let d_vec = cb - ca;
    let core = d_vec.length();
    let n = d_vec.normalize_or(Vec3::X); // from a toward b
    Distance {
        dist: core - ra - rb,
        point_a: ca + n * ra,
        point_b: cb - n * rb,
    }
}

/// OBB (a) vs OBB (b): exact convex-polyhedra distance via the complete
/// feature set — vertex→face both ways plus all edge-edge pairs.
fn obb_obb(a: ShapeRef, ha: Vec3, b: ShapeRef, hb: Vec3) -> Distance {
    let ca = obb_corners(a.pos, ha, a.rot);
    let cb = obb_corners(b.pos, hb, b.rot);
    let mut best = f32::MAX;
    let (mut pa, mut pb) = (Vec3::ZERO, Vec3::ZERO);
    let mut consider = |x: Vec3, y: Vec3| {
        let d = (y - x).length_squared();
        if d < best {
            best = d;
            pa = x;
            pb = y;
        }
    };
    // Vertex → face (both directions).
    for &c in &ca {
        let q = point_obb_closest(c, b.pos, hb, b.rot);
        consider(c, q);
    }
    for &c in &cb {
        let q = point_obb_closest(c, a.pos, ha, a.rot);
        consider(q, c);
    }
    // Edge → edge.
    for &(i0, i1) in &OBB_EDGES {
        for &(j0, j1) in &OBB_EDGES {
            let (x, y) = seg_seg_closest(ca[i0], ca[i1], cb[j0], cb[j1]);
            consider(x, y);
        }
    }
    Distance {
        dist: best.sqrt(),
        point_a: pa,
        point_b: pb,
    }
}

/// OBB (a) vs capsule (b): box features vs the capsule core segment,
/// radius subtracted at the end.
fn obb_capsule(a: ShapeRef, ha: Vec3, b: ShapeRef, r: f32, hh: f32) -> Distance {
    let (s0, s1) = capsule_segment(b.pos, hh, b.rot);
    let corners = obb_corners(a.pos, ha, a.rot);
    let mut best = f32::MAX;
    let (mut pa, mut pb_core) = (Vec3::ZERO, Vec3::ZERO);
    let mut consider = |x: Vec3, y: Vec3| {
        let d = (y - x).length_squared();
        if d < best {
            best = d;
            pa = x;
            pb_core = y;
        }
    };
    // Capsule endpoints → box, box corners → capsule core segment.
    for &p in &[s0, s1] {
        let q = point_obb_closest(p, a.pos, ha, a.rot);
        consider(q, p);
    }
    for &c in &corners {
        let q = point_segment_closest(c, s0, s1);
        consider(c, q);
    }
    // Box edges vs the capsule core segment.
    for &(i0, i1) in &OBB_EDGES {
        let (x, y) = seg_seg_closest(corners[i0], corners[i1], s0, s1);
        consider(x, y);
    }
    let d_vec = pb_core - pa;
    let core = d_vec.length();
    let n = d_vec.normalize_or(Vec3::X); // from box toward capsule core
    Distance {
        dist: core - r,
        point_a: pa,
        point_b: pb_core - n * r,
    }
}

/// Exact surface-to-surface distance between two placed shapes. Negative
/// distance means penetration (witnesses then are best-effort).
pub(crate) fn shape_distance(a: ShapeRef, b: ShapeRef) -> Distance {
    match (a.shape, b.shape) {
        (Shape::Sphere { radius: ra }, Shape::Sphere { radius: rb }) => {
            sphere_sphere(a, *ra, b, *rb)
        }
        (Shape::Sphere { radius: r }, Shape::Box { half_extents: h }) => sphere_obb(a, *r, b, *h),
        (Shape::Box { half_extents: h }, Shape::Sphere { radius: r }) => {
            let d = sphere_obb(b, *r, a, *h);
            Distance {
                dist: d.dist,
                point_a: d.point_b,
                point_b: d.point_a,
            }
        }
        (
            Shape::Sphere { radius: r },
            Shape::Capsule {
                radius: cr,
                half_height: hh,
            },
        ) => sphere_capsule(a, *r, b, *cr, *hh),
        (
            Shape::Capsule {
                radius: cr,
                half_height: hh,
            },
            Shape::Sphere { radius: r },
        ) => {
            let d = sphere_capsule(b, *r, a, *cr, *hh);
            Distance {
                dist: d.dist,
                point_a: d.point_b,
                point_b: d.point_a,
            }
        }
        (Shape::Box { half_extents: ha }, Shape::Box { half_extents: hb }) => {
            obb_obb(a, *ha, b, *hb)
        }
        (
            Shape::Box { half_extents: h },
            Shape::Capsule {
                radius: r,
                half_height: hh,
            },
        ) => obb_capsule(a, *h, b, *r, *hh),
        (
            Shape::Capsule {
                radius: r,
                half_height: hh,
            },
            Shape::Box { half_extents: h },
        ) => {
            let d = obb_capsule(b, *h, a, *r, *hh);
            Distance {
                dist: d.dist,
                point_a: d.point_b,
                point_b: d.point_a,
            }
        }
        (
            Shape::Capsule {
                radius: ra,
                half_height: ha,
            },
            Shape::Capsule {
                radius: rb,
                half_height: hb,
            },
        ) => capsule_capsule(a, *ra, *ha, b, *rb, *hb),
    }
}

/// Result of a swept cast: ABSOLUTE distance traveled along the sweep, world
/// hit point on the target surface, outward normal (pointing back toward the
/// mover), and the target's handle.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CastHit {
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub handle: usize,
}

/// Conservative advancement of `mover` along `delta` against `targets`.
/// Guaranteed tunnel-free: each advance is bounded by the exact current
/// distance, so no geometry can be crossed mid-step. Rotation is treated as
/// fixed during the sweep (linear cast, Jolt's LinearCast motion quality).
///
/// A mover already touching a target at t=0 does NOT report a hit for it —
/// resting/sliding contact is the discrete solver's job, not the sweep's.
pub(crate) fn cast_shape<'t>(
    mover: ShapeRef,
    delta: Vec3,
    targets: impl Iterator<Item = (usize, ShapeRef<'t>)>,
) -> Option<CastHit> {
    let len = delta.length();
    if len < 1e-9 {
        return None;
    }
    let dir = delta / len;
    /// Gap at which shapes count as touching.
    const TOUCH: f32 = 1e-3;
    const MAX_ITERS: usize = 24;

    let mut best: Option<CastHit> = None;
    for (handle, target) in targets {
        let mut t = 0.0f32;
        let mut hit: Option<CastHit> = None;
        for _ in 0..MAX_ITERS {
            let pos = mover.pos + dir * t;
            let d = shape_distance(
                ShapeRef {
                    shape: mover.shape,
                    pos,
                    rot: mover.rot,
                },
                target,
            );
            if d.dist <= TOUCH {
                if t > 0.0 {
                    let n = (d.point_a - d.point_b).normalize_or(-dir);
                    hit = Some(CastHit {
                        t,
                        point: d.point_b,
                        normal: n,
                        handle,
                    });
                }
                break;
            }
            // Advance by slightly less than the exact gap: no shape can be
            // reached in less than `dist` along ANY direction.
            t += d.dist - TOUCH * 0.5;
            if t >= len {
                break;
            }
        }
        if let Some(h) = hit
            && best.is_none_or(|b| h.t < b.t)
        {
            best = Some(h);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn sphere(radius: f32) -> Shape {
        Shape::Sphere { radius }
    }

    fn cuboid(half: Vec3) -> Shape {
        Shape::Box { half_extents: half }
    }

    fn capsule(radius: f32, half_height: f32) -> Shape {
        Shape::Capsule {
            radius,
            half_height,
        }
    }

    fn at(shape: &Shape, pos: Vec3) -> ShapeRef<'_> {
        ShapeRef {
            shape,
            pos,
            rot: Quat::IDENTITY,
        }
    }

    fn at_rot(shape: &Shape, pos: Vec3, rot: Quat) -> ShapeRef<'_> {
        ShapeRef { shape, pos, rot }
    }

    fn assert_vec3_close(got: Vec3, want: Vec3) {
        assert!((got - want).length() < EPS, "got {got:?}, want {want:?}");
    }

    // ── segment helpers ────────────────────────────────────────────────

    #[test]
    fn point_segment_closest_clamps_to_endpoints() {
        let a = Vec3::ZERO;
        let b = Vec3::new(2.0, 0.0, 0.0);
        assert_vec3_close(
            point_segment_closest(Vec3::new(1.0, 5.0, 0.0), a, b),
            Vec3::X,
        );
        // Past the ends: clamped to the endpoint.
        assert_vec3_close(point_segment_closest(Vec3::new(-3.0, 1.0, 0.0), a, b), a);
        assert_vec3_close(point_segment_closest(Vec3::new(9.0, 0.0, 0.0), a, b), b);
    }

    #[test]
    fn point_segment_closest_degenerate_segment() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        assert_vec3_close(point_segment_closest(Vec3::ZERO, a, a), a);
    }

    #[test]
    fn seg_seg_closest_parallel_segments() {
        // Two unit segments side by side along Y, 3 apart in X.
        let (pa, pb) = seg_seg_closest(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(3.0, -1.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
        );
        assert!((pa - pb).length() - 3.0 < EPS);
    }

    #[test]
    fn seg_seg_closest_degenerate_both() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 2.0, 0.0);
        let (pa, pb) = seg_seg_closest(a, a, b, b);
        assert_vec3_close(pa, a);
        assert_vec3_close(pb, b);
    }

    // ── sphere vs sphere ───────────────────────────────────────────────

    #[test]
    fn sphere_sphere_separated() {
        let (sa, sb) = (sphere(1.0), sphere(1.0));
        let d = shape_distance(at(&sa, Vec3::ZERO), at(&sb, Vec3::new(4.0, 0.0, 0.0)));
        assert!((d.dist - 2.0).abs() < EPS);
        assert_vec3_close(d.point_a, Vec3::X);
        assert_vec3_close(d.point_b, Vec3::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn sphere_sphere_touching() {
        let (sa, sb) = (sphere(1.0), sphere(0.5));
        let d = shape_distance(at(&sa, Vec3::ZERO), at(&sb, Vec3::new(1.5, 0.0, 0.0)));
        assert!(d.dist.abs() < EPS);
    }

    #[test]
    fn sphere_sphere_overlapping() {
        let (sa, sb) = (sphere(1.0), sphere(1.0));
        let d = shape_distance(at(&sa, Vec3::ZERO), at(&sb, Vec3::new(1.5, 0.0, 0.0)));
        assert!((d.dist - (-0.5)).abs() < EPS);
    }

    #[test]
    fn sphere_sphere_concentric() {
        // Zero-length direction: falls back to +X, distance is -(ra + rb).
        let (sa, sb) = (sphere(1.0), sphere(1.0));
        let d = shape_distance(at(&sa, Vec3::ZERO), at(&sb, Vec3::ZERO));
        assert!((d.dist - (-2.0)).abs() < EPS);
        assert_vec3_close(d.point_a, Vec3::X);
        assert_vec3_close(d.point_b, -Vec3::X);
    }

    // ── sphere vs box ──────────────────────────────────────────────────

    #[test]
    fn sphere_box_face() {
        let (s, b) = (sphere(0.5), cuboid(Vec3::ONE));
        let d = shape_distance(at(&s, Vec3::new(3.0, 0.0, 0.0)), at(&b, Vec3::ZERO));
        assert!((d.dist - 1.5).abs() < EPS);
        assert_vec3_close(d.point_a, Vec3::new(2.5, 0.0, 0.0));
        assert_vec3_close(d.point_b, Vec3::X);
    }

    #[test]
    fn sphere_box_touching() {
        let (s, b) = (sphere(1.0), cuboid(Vec3::ONE));
        let d = shape_distance(at(&s, Vec3::new(2.0, 0.0, 0.0)), at(&b, Vec3::ZERO));
        assert!(d.dist.abs() < EPS);
    }

    #[test]
    fn sphere_box_corner() {
        // Sphere on the box's space diagonal: closest feature is the corner.
        let (s, b) = (sphere(0.5), cuboid(Vec3::ONE));
        let corner = Vec3::ONE;
        let center = corner * 3.0;
        let d = shape_distance(at(&s, center), at(&b, Vec3::ZERO));
        let want = (center - corner).length() - 0.5;
        assert!((d.dist - want).abs() < EPS);
        assert_vec3_close(d.point_b, corner);
    }

    #[test]
    fn sphere_box_center_inside() {
        // Interior point is pushed to the nearest face, so the query reports
        // (face distance - radius) — positive here, i.e. depth, not a signed
        // penetration. Documents current behavior for CCD.
        let (s, b) = (sphere(0.25), cuboid(Vec3::ONE));
        let d = shape_distance(at(&s, Vec3::new(0.5, 0.0, 0.0)), at(&b, Vec3::ZERO));
        assert!((d.dist - 0.25).abs() < EPS);
        assert_vec3_close(d.point_b, Vec3::X);
    }

    #[test]
    fn sphere_box_rotated() {
        // Box rotated 45° about Z: its +Y-facing corner sits at (0, sqrt(2)).
        let (s, b) = (sphere(0.0), cuboid(Vec3::ONE));
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let d = shape_distance(
            at(&s, Vec3::new(0.0, 3.0, 0.0)),
            at_rot(&b, Vec3::ZERO, rot),
        );
        let want = 3.0 - 2.0f32.sqrt();
        assert!((d.dist - want).abs() < EPS);
        assert_vec3_close(d.point_b, Vec3::new(0.0, 2.0f32.sqrt(), 0.0));
    }

    #[test]
    fn box_sphere_swapped_witnesses() {
        // The Box/Sphere arm must mirror the Sphere/Box one, witnesses swapped.
        let (s, b) = (sphere(0.5), cuboid(Vec3::ONE));
        let d = shape_distance(at(&b, Vec3::ZERO), at(&s, Vec3::new(3.0, 0.0, 0.0)));
        assert!((d.dist - 1.5).abs() < EPS);
        assert_vec3_close(d.point_a, Vec3::X);
        assert_vec3_close(d.point_b, Vec3::new(2.5, 0.0, 0.0));
    }

    // ── sphere vs capsule ──────────────────────────────────────────────

    #[test]
    fn sphere_capsule_side() {
        let (s, c) = (sphere(0.5), capsule(0.5, 1.0));
        let d = shape_distance(at(&s, Vec3::new(3.0, 0.0, 0.0)), at(&c, Vec3::ZERO));
        assert!((d.dist - 2.0).abs() < EPS);
        assert_vec3_close(d.point_a, Vec3::new(2.5, 0.0, 0.0));
        assert_vec3_close(d.point_b, Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn sphere_capsule_cap() {
        // Above the cap: closest core point is the segment endpoint.
        let (s, c) = (sphere(0.5), capsule(0.5, 1.0));
        let d = shape_distance(at(&s, Vec3::new(0.0, 3.0, 0.0)), at(&c, Vec3::ZERO));
        assert!((d.dist - 1.0).abs() < EPS);
        assert_vec3_close(d.point_b, Vec3::new(0.0, 1.5, 0.0));
    }

    #[test]
    fn sphere_capsule_touching() {
        let (s, c) = (sphere(0.5), capsule(0.5, 1.0));
        let d = shape_distance(at(&s, Vec3::new(1.0, 0.0, 0.0)), at(&c, Vec3::ZERO));
        assert!(d.dist.abs() < EPS);
    }

    #[test]
    fn capsule_sphere_swapped_witnesses() {
        let (s, c) = (sphere(0.5), capsule(0.5, 1.0));
        let d = shape_distance(at(&c, Vec3::ZERO), at(&s, Vec3::new(3.0, 0.0, 0.0)));
        assert!((d.dist - 2.0).abs() < EPS);
        assert_vec3_close(d.point_a, Vec3::new(0.5, 0.0, 0.0));
        assert_vec3_close(d.point_b, Vec3::new(2.5, 0.0, 0.0));
    }

    // ── capsule vs capsule ─────────────────────────────────────────────

    #[test]
    fn capsule_capsule_parallel() {
        let (ca, cb) = (capsule(0.5, 1.0), capsule(0.5, 1.0));
        let d = shape_distance(at(&ca, Vec3::ZERO), at(&cb, Vec3::new(3.0, 0.0, 0.0)));
        assert!((d.dist - 2.0).abs() < EPS);
        assert_vec3_close(d.point_a, Vec3::new(0.5, d.point_a.y, 0.0));
        assert_vec3_close(d.point_b, Vec3::new(2.5, d.point_b.y, 0.0));
    }

    #[test]
    fn capsule_capsule_perpendicular() {
        // a along Y at the origin, b along X two units above in Z.
        let (ca, cb) = (capsule(0.5, 1.0), capsule(0.5, 1.0));
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let d = shape_distance(
            at(&ca, Vec3::ZERO),
            at_rot(&cb, Vec3::new(0.0, 0.0, 2.0), rot),
        );
        assert!((d.dist - 1.0).abs() < EPS);
        assert_vec3_close(d.point_a, Vec3::new(0.0, 0.0, 0.5));
        assert_vec3_close(d.point_b, Vec3::new(0.0, 0.0, 1.5));
    }

    #[test]
    fn capsule_capsule_overlapping() {
        let (ca, cb) = (capsule(0.5, 1.0), capsule(0.5, 1.0));
        let d = shape_distance(at(&ca, Vec3::ZERO), at(&cb, Vec3::new(0.5, 0.0, 0.0)));
        assert!((d.dist - (-0.5)).abs() < EPS);
    }

    #[test]
    fn capsule_capsule_degenerate_zero_half_height() {
        // Zero half-height capsules behave as spheres.
        let (ca, cb) = (capsule(0.5, 0.0), capsule(0.5, 0.0));
        let d = shape_distance(at(&ca, Vec3::ZERO), at(&cb, Vec3::new(2.0, 0.0, 0.0)));
        assert!((d.dist - 1.0).abs() < EPS);
    }

    // ── box vs box ─────────────────────────────────────────────────────

    #[test]
    fn box_box_face_to_face() {
        let (a, b) = (cuboid(Vec3::ONE), cuboid(Vec3::ONE));
        let d = shape_distance(at(&a, Vec3::ZERO), at(&b, Vec3::new(3.0, 0.0, 0.0)));
        assert!((d.dist - 1.0).abs() < EPS);
        // Witnesses are the facing surfaces (tie-breaking picks a corner).
        assert!((d.point_a.x - 1.0).abs() < EPS);
        assert!((d.point_b.x - 2.0).abs() < EPS);
        assert!((d.point_b.y - d.point_a.y).abs() < EPS);
        assert!((d.point_b.z - d.point_a.z).abs() < EPS);
    }

    #[test]
    fn box_box_corner_to_corner() {
        let (a, b) = (cuboid(Vec3::ONE), cuboid(Vec3::ONE));
        let d = shape_distance(at(&a, Vec3::ZERO), at(&b, Vec3::new(3.0, 3.0, 0.0)));
        // Nearest corners (1,1,z) and (2,2,z): distance sqrt(2).
        assert!((d.dist - 2.0f32.sqrt()).abs() < EPS);
    }

    #[test]
    fn box_box_touching() {
        let (a, b) = (cuboid(Vec3::ONE), cuboid(Vec3::new(0.5, 0.5, 0.5)));
        let d = shape_distance(at(&a, Vec3::ZERO), at(&b, Vec3::new(1.5, 0.0, 0.0)));
        assert!(d.dist.abs() < EPS);
    }

    #[test]
    fn box_box_overlapping() {
        // Overlap: vertex-inside-face snaps to the surface, distance bottoms
        // out at zero (penetration witnesses are documented as best-effort).
        let (a, b) = (cuboid(Vec3::ONE), cuboid(Vec3::ONE));
        let d = shape_distance(at(&a, Vec3::ZERO), at(&b, Vec3::new(1.0, 0.0, 0.0)));
        assert!(d.dist.abs() < EPS);
    }

    // ── box vs capsule ─────────────────────────────────────────────────

    #[test]
    fn box_capsule_side() {
        let (b, c) = (cuboid(Vec3::ONE), capsule(0.5, 1.0));
        let d = shape_distance(at(&b, Vec3::ZERO), at(&c, Vec3::new(3.0, 0.0, 0.0)));
        // Face x=1 to core x=3 is 2, minus capsule radius.
        assert!((d.dist - 1.5).abs() < EPS);
        assert_vec3_close(d.point_a, Vec3::new(1.0, d.point_a.y, 0.0));
        assert_vec3_close(d.point_b, Vec3::new(2.5, d.point_b.y, 0.0));
    }

    #[test]
    fn box_capsule_above_cap() {
        let (b, c) = (cuboid(Vec3::ONE), capsule(0.5, 1.0));
        let d = shape_distance(at(&b, Vec3::ZERO), at(&c, Vec3::new(0.0, 4.0, 0.0)));
        // Core endpoint (0,3), box face y=1: gap 2 minus radius 0.5.
        assert!((d.dist - 1.5).abs() < EPS);
    }

    #[test]
    fn capsule_box_swapped_witnesses() {
        let (b, c) = (cuboid(Vec3::ONE), capsule(0.5, 1.0));
        let d = shape_distance(at(&c, Vec3::new(3.0, 0.0, 0.0)), at(&b, Vec3::ZERO));
        assert!((d.dist - 1.5).abs() < EPS);
        assert_vec3_close(d.point_a, Vec3::new(2.5, d.point_a.y, 0.0));
        assert_vec3_close(d.point_b, Vec3::new(1.0, d.point_b.y, 0.0));
    }

    // ── conservative advancement ───────────────────────────────────────

    #[test]
    fn cast_sphere_onto_box_stops_at_surface() {
        let (s, b) = (sphere(1.0), cuboid(Vec3::ONE));
        let hit = cast_shape(
            at(&s, Vec3::ZERO),
            Vec3::new(10.0, 0.0, 0.0),
            [(7usize, at(&b, Vec3::new(5.0, 0.0, 0.0)))].into_iter(),
        )
        .expect("must hit");
        // Gap at t=0 is 3 (sphere surface x=1, box face x=4).
        assert!((hit.t - 3.0).abs() < 2e-3, "t = {}", hit.t);
        assert!((hit.point.x - 4.0).abs() < 2e-3);
        assert!((hit.normal.x - (-1.0)).abs() < 1e-3);
        assert_eq!(hit.handle, 7);
    }

    #[test]
    fn cast_away_from_target_misses() {
        let (s, b) = (sphere(1.0), cuboid(Vec3::ONE));
        let hit = cast_shape(
            at(&s, Vec3::ZERO),
            Vec3::new(-10.0, 0.0, 0.0),
            [(0usize, at(&b, Vec3::new(5.0, 0.0, 0.0)))].into_iter(),
        );
        assert!(hit.is_none());
    }

    #[test]
    fn cast_already_touching_reports_no_hit() {
        // Resting contact at t=0 is the discrete solver's job, not the sweep's.
        let (s, b) = (sphere(1.0), cuboid(Vec3::ONE));
        let hit = cast_shape(
            at(&s, Vec3::ZERO),
            Vec3::new(10.0, 0.0, 0.0),
            [(0usize, at(&b, Vec3::new(2.0, 0.0, 0.0)))].into_iter(),
        );
        assert!(hit.is_none());
    }

    #[test]
    fn cast_zero_delta_is_none() {
        let (s, b) = (sphere(1.0), cuboid(Vec3::ONE));
        let hit = cast_shape(
            at(&s, Vec3::ZERO),
            Vec3::ZERO,
            [(0usize, at(&b, Vec3::new(5.0, 0.0, 0.0)))].into_iter(),
        );
        assert!(hit.is_none());
    }

    #[test]
    fn cast_picks_nearest_target() {
        let (s, b) = (sphere(1.0), cuboid(Vec3::ONE));
        let hit = cast_shape(
            at(&s, Vec3::ZERO),
            Vec3::new(20.0, 0.0, 0.0),
            [
                (1usize, at(&b, Vec3::new(9.0, 0.0, 0.0))),
                (2usize, at(&b, Vec3::new(5.0, 0.0, 0.0))),
            ]
            .into_iter(),
        )
        .expect("must hit");
        assert_eq!(hit.handle, 2);
        assert!((hit.t - 3.0).abs() < 2e-3, "t = {}", hit.t);
    }
}
