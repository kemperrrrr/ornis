//! Signed box queries: SAT for overlap, exact feature distance when separated,
//! and piecewise-quadratic segment/AABB distance for capsule spines.

use glam::{Mat3, Quat, Vec3};

use super::{Distance, OBB_EDGES, ShapeRef, obb_corners, point_obb_closest, seg_seg_closest};

/// Numerical zero for degeneracy guards: squared SAT cross-axis lengths, segment direction components and squared core distances at or below this magnitude are treated as exactly zero (parallel face/edge axes with no separating direction, collapsed slab crossings, touching spine cores) — values here are f32 dust on O(1) geometry, not features.
const DEGENERATE_EPS: f32 = 1e-12;
/// Face-axis significance for support-face collapse: local direction components above this snap the query onto the opposing face pair, while smaller components are orientation dust rather than a facing axis — without the gate a grazing normal would collapse the wrong extent and teleport the witnesses.
const FACE_AXIS_EPS: f32 = 1e-6;
/// Spine-to-normal alignment for capsule endpoint selection: axis projections above this commit the penetration query to the leading cap, while smaller projections read as broadside so the whole spine segment stays in play — the value matches `FACE_AXIS_EPS` bit-for-bit but guards alignment, not facing, hence a separate const.
const SPINE_ALIGNMENT_EPS: f32 = 1e-6;

fn separating_axis(a: ShapeRef, ha: Vec3, b: ShapeRef, hb: Vec3) -> (f32, Vec3) {
    let ra = Mat3::from_quat(a.rot);
    let rb = Mat3::from_quat(b.rot);
    let aa = [ra.x_axis, ra.y_axis, ra.z_axis];
    let bb = [rb.x_axis, rb.y_axis, rb.z_axis];
    let mut axes = [Vec3::ZERO; 15];
    axes[..3].copy_from_slice(&aa);
    axes[3..6].copy_from_slice(&bb);
    for i in 0..3 {
        for j in 0..3 {
            let cross = aa[i].cross(bb[j]);
            if cross.length_squared() > DEGENERATE_EPS {
                axes[6 + 3 * i + j] = cross.normalize();
            }
        }
    }
    let delta = b.pos - a.pos;
    let mut result = (f32::NEG_INFINITY, Vec3::Y);
    for n in axes {
        if n == Vec3::ZERO {
            continue;
        }
        let ea = ha.x * n.dot(aa[0]).abs() + ha.y * n.dot(aa[1]).abs() + ha.z * n.dot(aa[2]).abs();
        let eb = hb.x * n.dot(bb[0]).abs() + hb.y * n.dot(bb[1]).abs() + hb.z * n.dot(bb[2]).abs();
        let projection = delta.dot(n);
        let gap = projection.abs() - ea - eb;
        if gap > result.0 {
            result = (gap, if projection >= 0.0 { n } else { -n });
        }
    }
    result
}

pub(crate) fn box_box_signed_gap(
    a_pos: Vec3,
    a_rot: Quat,
    ha: Vec3,
    b_pos: Vec3,
    b_rot: Quat,
    hb: Vec3,
) -> f32 {
    let a_shape = crate::Shape::Box { half_extents: ha };
    let b_shape = crate::Shape::Box { half_extents: hb };
    separating_axis(
        ShapeRef {
            shape: &a_shape,
            pos: a_pos,
            rot: a_rot,
        },
        ha,
        ShapeRef {
            shape: &b_shape,
            pos: b_pos,
            rot: b_rot,
        },
        hb,
    )
    .0
}

fn support_face<'a>(body: ShapeRef<'a>, half: Vec3, direction: Vec3) -> (ShapeRef<'a>, Vec3) {
    let local = body.rot.conjugate() * direction;
    let mut offset = Vec3::ZERO;
    let mut extent = half;
    for i in 0..3 {
        if local[i].abs() > FACE_AXIS_EPS {
            offset[i] = local[i].signum() * half[i];
            extent[i] = 0.0;
        }
    }
    (
        ShapeRef {
            pos: body.pos + body.rot * offset,
            ..body
        },
        extent,
    )
}

fn feature_distance(a: ShapeRef, ha: Vec3, b: ShapeRef, hb: Vec3) -> Distance {
    let ca = obb_corners(a.pos, ha, a.rot);
    let cb = obb_corners(b.pos, hb, b.rot);
    let mut result = Distance {
        dist: f32::INFINITY,
        point_a: a.pos,
        point_b: b.pos,
    };
    let mut best = f32::INFINITY;
    let mut consider = |x: Vec3, y: Vec3| {
        let squared = (y - x).length_squared();
        if squared < best {
            best = squared;
            result = Distance {
                dist: squared.sqrt(),
                point_a: x,
                point_b: y,
            };
        }
    };
    for &c in &ca {
        consider(c, point_obb_closest(c, b.pos, hb, b.rot));
    }
    for &c in &cb {
        consider(point_obb_closest(c, a.pos, ha, a.rot), c);
    }
    for &(a0, a1) in &OBB_EDGES {
        for &(b0, b1) in &OBB_EDGES {
            let (x, y) = seg_seg_closest(ca[a0], ca[a1], cb[b0], cb[b1]);
            consider(x, y);
        }
    }
    result
}

pub(super) fn box_box(a: ShapeRef, ha: Vec3, b: ShapeRef, hb: Vec3) -> Distance {
    let (gap, normal) = separating_axis(a, ha, b, hb);
    if gap > 0.0 {
        return feature_distance(a, ha, b, hb);
    }
    // Closest points of the opposing support features retain a coherent
    // penetration normal, including fully contained/concentric boxes.
    let (fa, ea) = support_face(a, ha, normal);
    let (fb, eb) = support_face(b, hb, -normal);
    let mut result = feature_distance(fa, ea, fb, eb);
    result.dist = gap;
    result
}

/// Distance to the solid box, not just its edges: face-interior crossings
/// and fully contained segments have zero core distance.
fn segment_box(start: Vec3, end: Vec3, lo: Vec3, hi: Vec3) -> (Vec3, Vec3, f32) {
    let delta = end - start;
    let mut cuts = vec![0.0f32, 1.0];
    for i in 0..3 {
        if delta[i].abs() > DEGENERATE_EPS {
            for boundary in [lo[i], hi[i]] {
                let t = (boundary - start[i]) / delta[i];
                if t > 0.0 && t < 1.0 {
                    cuts.push(t);
                }
            }
        }
    }
    cuts.sort_by(f32::total_cmp);
    let mut result = (start, start.clamp(lo, hi), f32::INFINITY);
    for range in cuts.windows(2) {
        let middle = (range[0] + range[1]) * 0.5;
        let p = start + middle * delta;
        let (mut numerator, mut denominator) = (0.0, 0.0);
        for i in 0..3 {
            let boundary = if p[i] < lo[i] {
                Some(lo[i])
            } else if p[i] > hi[i] {
                Some(hi[i])
            } else {
                None
            };
            if let Some(boundary) = boundary {
                numerator += delta[i] * (start[i] - boundary);
                denominator += delta[i] * delta[i];
            }
        }
        let stationary = if denominator > 0.0 {
            (-numerator / denominator).clamp(range[0], range[1])
        } else {
            middle
        };
        for t in [stationary, range[0], range[1]] {
            let point = start + t * delta;
            let closest = point.clamp(lo, hi);
            let squared = (point - closest).length_squared();
            if squared < result.2 {
                result = (point, closest, squared);
            }
        }
    }
    result
}

pub(super) fn box_capsule(
    a: ShapeRef,
    half: Vec3,
    b: ShapeRef,
    radius: f32,
    height: f32,
) -> Distance {
    let inv = a.rot.conjugate();
    let center = inv * (b.pos - a.pos);
    let axis = inv * (b.rot * Vec3::Y);
    let (start, end) = (center - axis * height, center + axis * height);
    let (mut core, mut surface, squared) = segment_box(start, end, -half, half);
    let (distance, normal) = if squared > DEGENERATE_EPS {
        let length = squared.sqrt();
        (length - radius, (core - surface) / length)
    } else {
        let (depth, normal) = spine_penetration(center, axis, half, radius, height);
        let (mut lo, mut hi) = (-half, half);
        for i in 0..3 {
            if normal[i].abs() > FACE_AXIS_EPS {
                lo[i] = normal[i].signum() * half[i];
                hi[i] = lo[i];
            }
        }
        let projection = axis.dot(normal);
        let endpoints = if projection > SPINE_ALIGNMENT_EPS {
            (start, start)
        } else if projection < -SPINE_ALIGNMENT_EPS {
            (end, end)
        } else {
            (start, end)
        };
        (core, surface, _) = segment_box(endpoints.0, endpoints.1, lo, hi);
        (depth, normal)
    };
    Distance {
        dist: distance,
        point_a: a.pos + a.rot * surface,
        point_b: a.pos + a.rot * (core - normal * radius),
    }
}

/// The box-minus-segment zonotope has face normals from its generator
/// cross-products. For a spine intersecting the box, offset each face by r.
fn spine_penetration(
    center: Vec3,
    axis: Vec3,
    half: Vec3,
    radius: f32,
    height: f32,
) -> (f32, Vec3) {
    let mut result = (f32::NEG_INFINITY, Vec3::X);
    for e in [Vec3::X, Vec3::Y, Vec3::Z] {
        for n in [e, axis.cross(e).normalize_or_zero()] {
            if n == Vec3::ZERO {
                continue;
            }
            let projection = center.dot(n);
            let gap = projection.abs() - half.dot(n.abs()) - height * axis.dot(n).abs() - radius;
            if gap > result.0 {
                result = (gap, if projection >= 0.0 { n } else { -n });
            }
        }
    }
    result
}
