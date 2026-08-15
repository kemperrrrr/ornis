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
