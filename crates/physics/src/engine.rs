use std::collections::HashMap;

use glam::{Quat, Vec3};

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::math::{AABB, Ray, RaycastHit};
use crate::shape::Shape;

pub trait PhysicsEngine: Send + Sync {
    fn step(&mut self, dt: f32);
    fn add_body(&mut self, body: RigidBody) -> BodyHandle;
    fn remove_body(&mut self, handle: BodyHandle);
    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody>;
    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody>;
    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit>;
    fn shapecast(&self, shape: &Shape, from: Vec3, to: Vec3) -> Option<RaycastHit>;
}

struct Contact {
    normal: Vec3,
    penetration: f32,
    contact_point: Vec3,
}

/// A single contact point inside a manifold (G2).
#[derive(Clone, Copy, Debug)]
struct ManifoldPoint {
    world_point: Vec3,
    penetration: f32,
}

/// Contact manifold: one normal + up to 4 points per body pair.
#[derive(Clone, Debug)]
struct Manifold {
    body_a: BodyHandle,
    body_b: BodyHandle,
    normal: Vec3,
    point_count: usize,
    points: [ManifoldPoint; 4],
}

impl Manifold {
    fn single(body_a: BodyHandle, body_b: BodyHandle, c: Contact) -> Self {
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

#[inline]
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

#[inline]
fn inv_inertia_axis(i: f32) -> f32 {
    if i > 0.0 { 1.0 / i } else { 0.0 }
}

/// Apply the inverse world-space inertia tensor (body axes rotated to world).
fn mul_inv_inertia(inertia: Vec3, orientation: glam::Quat, v: Vec3) -> Vec3 {
    let body = orientation.inverse() * v;
    Vec3::new(
        inv_inertia_axis(inertia.x) * body.x,
        inv_inertia_axis(inertia.y) * body.y,
        inv_inertia_axis(inertia.z) * body.z,
    )
}

/// Effective inverse mass along direction `dir` at contact points with
/// levers `ra`/`rb` (linear + rotational terms, world-space inertia).
fn effective_mass(bodies: &[RigidBody], i: usize, j: usize, dir: Vec3, ra: Vec3, rb: Vec3) -> f32 {
    let ra_d = ra.cross(dir);
    let rb_d = rb.cross(dir);
    bodies[i].inv_mass
        + bodies[j].inv_mass
        + ra_d.dot(mul_inv_inertia(
            bodies[i].inertia,
            bodies[i].orientation,
            ra_d,
        ))
        + rb_d.dot(mul_inv_inertia(
            bodies[j].inertia,
            bodies[j].orientation,
            rb_d,
        ))
}

/// Entry of the contact normal "K matrix" (G4 block solver): how a unit
/// normal impulse applied at point `l` changes the normal relative velocity
/// measured at point `k`. Symmetric in exact arithmetic.
#[allow(clippy::too_many_arguments)]
fn k_entry(
    bodies: &[RigidBody],
    i: usize,
    j: usize,
    n: Vec3,
    ra_k: Vec3,
    rb_k: Vec3,
    ra_l: Vec3,
    rb_l: Vec3,
) -> f32 {
    let a = &bodies[i];
    let b = &bodies[j];
    let ia = mul_inv_inertia(a.inertia, a.orientation, ra_l.cross(n));
    let ib = mul_inv_inertia(b.inertia, b.orientation, rb_l.cross(n));
    a.inv_mass + b.inv_mass + ia.cross(ra_k).dot(n) + ib.cross(rb_k).dot(n)
}

/// Solve a small (≤4×4) dense linear system by Gaussian elimination with
/// partial pivoting. Returns None on a (near-)singular matrix.
// Index loops are kept for matrix-math clarity (rows/cols, not elements).
#[allow(clippy::needless_range_loop)]
fn solve_small(a: &[[f32; 4]; 4], b: &[f32; 4], n: usize) -> Option<[f32; 4]> {
    let mut m = *a;
    let mut x = *b;
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        for r in (col + 1)..n {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        if piv != col {
            m.swap(piv, col);
            x.swap(piv, col);
        }
        let d = m[col][col];
        for r in (col + 1)..n {
            let f = m[r][col] / d;
            for c in col..n {
                m[r][c] -= f * m[col][c];
            }
            x[r] -= f * x[col];
        }
    }
    let mut out = [0.0f32; 4];
    for r in (0..n).rev() {
        let mut s = x[r];
        for c in (r + 1)..n {
            s -= m[r][c] * out[c];
        }
        out[r] = s / m[r][r];
    }
    Some(out)
}

/// Apply an impulse at a contact point to the body pair (velocity + angular).
/// The contact normal points from body `i` to body `j`; a positive impulse
/// pushes `j` along it and `i` against it.
fn apply_impulse(bodies: &mut [RigidBody], i: usize, j: usize, imp: Vec3, ra: Vec3, rb: Vec3) {
    debug_assert!(i != j);
    let (lo, hi, swapped) = if i < j { (i, j, false) } else { (j, i, true) };
    let (head, tail) = bodies.split_at_mut(hi);
    let (a, b) = if swapped {
        // j < i: body i is in tail, body j is in head.
        (&mut tail[0], &mut head[lo])
    } else {
        (&mut head[lo], &mut tail[0])
    };
    // `a` is body i, `b` is body j.
    let (ia, oa) = (a.inertia, a.orientation);
    let (ib, ob) = (b.inertia, b.orientation);
    a.velocity -= imp * a.inv_mass;
    b.velocity += imp * b.inv_mass;
    a.angular_velocity -= mul_inv_inertia(ia, oa, ra.cross(imp));
    b.angular_velocity += mul_inv_inertia(ib, ob, rb.cross(imp));
}

/// Velocity of a body at a world-space contact point (linear + angular part).
#[inline]
fn point_velocity(body: &RigidBody, r: Vec3) -> Vec3 {
    body.velocity + body.angular_velocity.cross(r)
}

/// CFM-regularized effective mass (b3MakeSoft analog): `cfm` > 0 softens the
/// constraint — the same position/velocity error produces a smaller impulse,
/// spread across iterations instead of a one-shot rigid correction.
#[inline]
fn make_soft(k: f32, cfm: f32) -> f32 {
    k + cfm
}

/// Any unit vector perpendicular to `n` — the fixed tangent frame for
/// friction. Deterministic so warm-start and iterations stay consistent.
fn tangent_basis(n: Vec3) -> Vec3 {
    let axis = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    n.cross(axis).normalize_or(Vec3::Z)
}

/// Apply a positional (pseudo) impulse to the body pair: moves positions and
/// orientations WITHOUT touching real velocities (split impulse / NGS).
fn apply_positional_impulse(
    bodies: &mut [RigidBody],
    i: usize,
    j: usize,
    jp: Vec3,
    ra: Vec3,
    rb: Vec3,
) {
    debug_assert!(i != j);
    let (lo, hi, swapped) = if i < j { (i, j, false) } else { (j, i, true) };
    let (head, tail) = bodies.split_at_mut(hi);
    let (a, b) = if swapped {
        (&mut tail[0], &mut head[lo])
    } else {
        (&mut head[lo], &mut tail[0])
    };
    // `a` is body i, `b` is body j.
    let (ia, oa) = (a.inertia, a.orientation);
    let (ib, ob) = (b.inertia, b.orientation);
    a.position -= jp * a.inv_mass;
    b.position += jp * b.inv_mass;
    let da = mul_inv_inertia(ia, oa, ra.cross(-jp));
    let db = mul_inv_inertia(ib, ob, rb.cross(jp));
    if da != Vec3::ZERO {
        a.orientation = (Quat::from_scaled_axis(da) * a.orientation).normalize();
    }
    if db != Vec3::ZERO {
        b.orientation = (Quat::from_scaled_axis(db) * b.orientation).normalize();
    }
}

/// Exact LCP block solve of the normal direction for one manifold (G4).
/// Generalizes the Box2D b2ContactSolver case tree to 3-4 points: enumerate
/// active sets (largest first), solve the K system for the new accumulated
/// impulses directly, and take the first set satisfying complementarity
/// (acc' ≥ 0 on active, vn' ≥ 0 on inactive). Scalar per-point Gauss-Seidel
/// oscillates between coupled points of one manifold (the rocking pump from
/// G3); the block solve finds the exact active set in one shot.
fn solve_normal_block(
    bodies: &mut [RigidBody],
    i: usize,
    j: usize,
    n: Vec3,
    pts: &[Vec3; 4],
    acc: &mut [f32; 4],
    count: usize,
) {
    let pa = bodies[i].position;
    let pb = bodies[j].position;
    let mut ras = [Vec3::ZERO; 4];
    let mut rbs = [Vec3::ZERO; 4];
    for k in 0..count {
        ras[k] = pts[k] - pa;
        rbs[k] = pts[k] - pb;
    }
    let mut k_mat = [[0.0f32; 4]; 4];
    for k in 0..count {
        for l in 0..count {
            k_mat[k][l] = k_entry(bodies, i, j, n, ras[k], rbs[k], ras[l], rbs[l]);
        }
    }
    let mut vn = [0.0f32; 4];
    for k in 0..count {
        vn[k] = (point_velocity(&bodies[j], rbs[k]) - point_velocity(&bodies[i], ras[k])).dot(n);
    }

    let total = 1usize << count;
    for pop in (1..=count).rev() {
        for mask in 1..total {
            if mask.count_ones() as usize != pop {
                continue;
            }
            let mut idx = [0usize; 4];
            let mut ns = 0;
            for k in 0..count {
                if (mask >> k) & 1 == 1 {
                    idx[ns] = k;
                    ns += 1;
                }
            }
            // Solve K_S · acc'_S = -vn_S + K_{S,all} · acc (new accumulated
            // impulses directly, so zeroing the inactive set is consistent).
            let mut ks = [[0.0f32; 4]; 4];
            let mut bs = [0.0f32; 4];
            for a in 0..ns {
                for b in 0..ns {
                    ks[a][b] = k_mat[idx[a]][idx[b]];
                }
                let mut r = -vn[idx[a]];
                for m in 0..count {
                    r += k_mat[idx[a]][m] * acc[m];
                }
                bs[a] = r;
            }
            let Some(ap) = solve_small(&ks, &bs, ns) else {
                continue;
            };
            if ap.iter().take(ns).any(|&v| v < -1e-6) {
                continue;
            }
            // vn' on the inactive set must stay non-negative.
            let mut ok = true;
            for t in 0..count {
                if (mask >> t) & 1 == 1 {
                    continue;
                }
                let mut v = vn[t];
                for a in 0..ns {
                    v += k_mat[t][idx[a]] * (ap[a] - acc[idx[a]]);
                }
                v -= k_mat[t][t] * acc[t]; // this point's impulse is removed
                for m in 0..count {
                    if (mask >> m) & 1 == 0 && m != t {
                        v -= k_mat[t][m] * acc[m];
                    }
                }
                if v < -1e-5 {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            // Commit: apply the deltas and store the new accumulated impulses.
            for a in 0..ns {
                let k = idx[a];
                let d = ap[a] - acc[k];
                if d.abs() > 1e-12 {
                    apply_impulse(bodies, i, j, n * d, ras[k], rbs[k]);
                }
                acc[k] = ap[a];
            }
            for t in 0..count {
                if (mask >> t) & 1 == 0 && acc[t].abs() > 1e-12 {
                    let d = -acc[t];
                    apply_impulse(bodies, i, j, n * d, ras[t], rbs[t]);
                    acc[t] = 0.0;
                } else if (mask >> t) & 1 == 0 {
                    acc[t] = 0.0;
                }
            }
            return;
        }
    }
    // No valid active set (numerically degenerate) — keep the warm-started
    // state; the next outer iteration will retry from updated velocities.
}

fn compute_aabb(body: &RigidBody) -> AABB {
    body.shape.aabb(body.position, body.orientation)
}

// ---- Narrow-phase: world-frame analytic contact tests (oriented shapes) ----

fn sphere_vs_sphere(pos_a: Vec3, radius_a: f32, pos_b: Vec3, radius_b: f32) -> Option<Contact> {
    let diff = pos_b - pos_a;
    let dist_sq = diff.length_squared();
    let radius_sum = radius_a + radius_b;
    if dist_sq > radius_sum * radius_sum || dist_sq < 1e-10 {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = diff / dist;
    let penetration = radius_sum - dist;
    Some(Contact {
        normal,
        penetration,
        contact_point: pos_a + normal * (radius_a - penetration * 0.5),
    })
}

/// Sphere vs an oriented box (OBB), resolved in the box's local frame.
fn sphere_vs_obb(
    sphere_pos: Vec3,
    sphere_radius: f32,
    box_pos: Vec3,
    half_extents: Vec3,
    box_rot: Quat,
) -> Option<Contact> {
    let local = box_rot.inverse() * (sphere_pos - box_pos);
    let clamped = local.clamp(-half_extents, half_extents);
    let delta = clamped - local;
    let dist_sq = delta.length_squared();
    if dist_sq > sphere_radius * sphere_radius || dist_sq < 1e-10 {
        return None;
    }
    let dist = dist_sq.sqrt();
    // Normal points from the box toward the sphere, in world space.
    let normal = box_rot * (delta / dist);
    let penetration = sphere_radius - dist;
    // Contact point: where the sphere touches the box (midway on the normal).
    let contact_point = (local + clamped) * 0.5;
    Some(Contact {
        normal,
        penetration,
        contact_point: box_pos + box_rot * contact_point,
    })
}

/// Axis used for the OBB overlap test: returns `radius_a + radius_b - separation`.
#[allow(clippy::too_many_arguments)]
fn obb_overlap_on(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    axis: Vec3,
) -> f32 {
    let aa = rot_a * Vec3::X;
    let ab = rot_a * Vec3::Y;
    let ac = rot_a * Vec3::Z;
    let ba = rot_b * Vec3::X;
    let bb = rot_b * Vec3::Y;
    let bc = rot_b * Vec3::Z;
    let ra = half_a.x * axis.dot(aa).abs()
        + half_a.y * axis.dot(ab).abs()
        + half_a.z * axis.dot(ac).abs();
    let rb = half_b.x * axis.dot(ba).abs()
        + half_b.y * axis.dot(bb).abs()
        + half_b.z * axis.dot(bc).abs();
    let center_dist = (pos_b - pos_a).dot(axis);
    ra + rb - center_dist.abs()
}

#[allow(clippy::too_many_arguments)]
fn obb_sat(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
) -> Option<(Vec3, f32)> {
    // SAT: the 3 face normals of each box plus the cross products of their axes.
    let aa = [rot_a * Vec3::X, rot_a * Vec3::Y, rot_a * Vec3::Z];
    let ba = [rot_b * Vec3::X, rot_b * Vec3::Y, rot_b * Vec3::Z];

    // Face normals first; an edge-edge axis may replace a face axis only if
    // it beats it by a margin. Otherwise micro-tilts at face contacts make
    // SAT pick noisy cross-product axes and the normal flickers.
    const FACE_PREFERENCE: f32 = 1e-3;
    // Speculative margin on the "separated" verdict: a pair closer than this
    // is still reported as touching so the manifold never blinks off for one
    // substep (which would drop the warm-start cache and pump energy).
    const SAT_SEPARATION_EPS: f32 = 2e-3;

    let mut best_overlap = f32::MAX;
    let mut best_axis = Vec3::X;

    for u in aa.into_iter().chain(ba) {
        let overlap = obb_overlap_on(pos_a, half_a, rot_a, pos_b, half_b, rot_b, u);
        // Separated along any axis -> no contact. A hair of negative
        // tolerance turns near-touching into a speculative contact instead
        // of a blink (the solver treats zero-depth points as harmless).
        if overlap <= -SAT_SEPARATION_EPS {
            return None;
        }
        if overlap < best_overlap {
            best_overlap = overlap;
            best_axis = u;
        }
    }
    for ai in &aa {
        for bi in &ba {
            let c = ai.cross(*bi);
            // Near-parallel edge pairs give a numerically noisy axis; the
            // face axes cover that case. A too-small threshold here lets
            // float noise produce false "separated" verdicts on micro-tilts
            // and the manifold blinks on/off — a warm-start energy pump.
            if c.length() < 1e-3 {
                continue;
            }
            let u = c.normalize();
            let overlap = obb_overlap_on(pos_a, half_a, rot_a, pos_b, half_b, rot_b, u);
            if overlap <= -SAT_SEPARATION_EPS {
                return None;
            }
            if overlap < best_overlap - FACE_PREFERENCE {
                best_overlap = overlap;
                best_axis = u;
            }
        }
    }

    // Orient the normal so it points from A to B.
    let normal = if best_axis.dot(pos_b - pos_a) < 0.0 {
        -best_axis
    } else {
        best_axis
    };
    Some((normal, best_overlap))
}

#[allow(clippy::too_many_arguments)]
fn box_vs_box(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
) -> Option<Contact> {
    let (normal, penetration) = obb_sat(pos_a, half_a, rot_a, pos_b, half_b, rot_b)?;
    Some(Contact {
        normal,
        penetration,
        contact_point: (pos_a + pos_b) * 0.5,
    })
}

/// Eight world-space corners of an oriented box.
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

/// Multi-point manifold for OBB-OBB: keeps up to 4 vertices that lie inside the
/// opposing box's face (vertex-face contacts), falling back to a single point.
#[allow(clippy::too_many_arguments)]
fn box_manifold(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
) -> Option<Manifold> {
    let (n, _pen) = obb_sat(pos_a, half_a, rot_a, pos_b, half_b, rot_b)?;

    let aa = [rot_a * Vec3::X, rot_a * Vec3::Y, rot_a * Vec3::Z];
    let ba = [rot_b * Vec3::X, rot_b * Vec3::Y, rot_b * Vec3::Z];
    // Half-width of each box projected onto the contact normal.
    let hwn_a = half_a.x * aa[0].dot(n).abs()
        + half_a.y * aa[1].dot(n).abs()
        + half_a.z * aa[2].dot(n).abs();
    let hwn_b = half_b.x * ba[0].dot(n).abs()
        + half_b.y * ba[1].dot(n).abs()
        + half_b.z * ba[2].dot(n).abs();

    // Contact-region tolerance: a corner counts as touching the opposing face
    // when it is within this distance along the (negated) contact normal —
    // a small speculative margin keeps near-touching corners in the manifold.
    const DEPTH_TOL: f32 = 0.01;
    // Tangential slack beyond the face rectangle: corners slightly outside the
    // face edge (micro-tilts at face contacts) must still generate points,
    // otherwise the manifold collapses to the single-point fallback and the
    // body starts rocking on a corner.
    let tangent_slack = 0.05;

    let mut cand: Vec<(Vec3, f32)> = Vec::new();
    // B's corners touching A's face (the face most anti-parallel to `n`).
    for c in obb_corners(pos_b, half_b, rot_b) {
        let local = rot_a.inverse() * (c - pos_a);
        // Depth of the corner relative to A's surface along the normal.
        let d = hwn_a - (c - pos_a).dot(n);
        if d < -DEPTH_TOL {
            continue;
        }
        // Tangential containment inside the face rectangle (with slack).
        if local.x.abs() <= half_a.x + tangent_slack
            && local.y.abs() <= half_a.y + tangent_slack
            && local.z.abs() <= half_a.z + tangent_slack
        {
            cand.push((c, d));
        }
    }
    // A's corners touching B's face.
    for c in obb_corners(pos_a, half_a, rot_a) {
        let local = rot_b.inverse() * (c - pos_b);
        let d = hwn_b - (c - pos_b).dot(-n);
        if d < -DEPTH_TOL {
            continue;
        }
        if local.x.abs() <= half_b.x + tangent_slack
            && local.y.abs() <= half_b.y + tangent_slack
            && local.z.abs() <= half_b.z + tangent_slack
        {
            cand.push((c, d));
        }
    }

    // Deduplicate in the tangent plane: the same contact region appears once
    // from each box's corners, offset along the normal by the penetration
    // depth. Merge by tangential distance, keep the deeper representative —
    // this yields a stable 4-point manifold instead of a flickering mix.
    let mut uniq: Vec<(Vec3, f32)> = Vec::new();
    for (p, d) in cand {
        let mut merged = false;
        for (q, qd) in uniq.iter_mut() {
            let tangential = (p - *q) - n * (p - *q).dot(n);
            // 5 cm: near-coincident points make the constraint system
            // near-singular and PGS oscillates into runaway impulses.
            if tangential.length() < 0.05 {
                if d > *qd {
                    *q = p;
                    *qd = d;
                }
                merged = true;
                break;
            }
        }
        if !merged {
            uniq.push((p, d));
        }
    }
    uniq.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut points = [ManifoldPoint {
        world_point: Vec3::ZERO,
        penetration: 0.0,
    }; 4];
    let mut count = 0;
    for (p, d) in uniq.into_iter().take(4) {
        points[count] = ManifoldPoint {
            world_point: p,
            penetration: d.max(0.0),
        };
        count += 1;
    }

    if count == 0 {
        return box_vs_box(pos_a, half_a, rot_a, pos_b, half_b, rot_b)
            .map(|c| Manifold::single(0, 0, c));
    }
    Some(Manifold {
        body_a: 0,
        body_b: 0,
        normal: n,
        point_count: count,
        points,
    })
}

/// Sphere vs an oriented capsule: closest point on the capsule's segment.
#[allow(clippy::too_many_arguments)]
fn sphere_vs_capsule(
    sphere_pos: Vec3,
    sphere_radius: f32,
    cap_pos: Vec3,
    cap_radius: f32,
    cap_half_height: f32,
    cap_rot: Quat,
) -> Option<Contact> {
    let axis = cap_rot * Vec3::Y;
    let bottom = cap_pos - axis * cap_half_height;
    let seg = axis * (2.0 * cap_half_height);
    let t = (sphere_pos - bottom).dot(seg) / seg.length_squared();
    let t = t.clamp(0.0, 1.0);
    let closest = bottom + seg * t;
    let to_sphere = sphere_pos - closest;
    let d = to_sphere.length();
    let rr = cap_radius + sphere_radius;
    if d >= rr || d < 1e-10 {
        return None;
    }
    // Normal points from the capsule toward the sphere.
    let n = to_sphere / d;
    let penetration = rr - d;
    let contact_point = closest + n * (cap_radius - penetration * 0.5);
    Some(Contact {
        normal: n,
        penetration,
        contact_point,
    })
}

/// Capsule-capsule: both segment axes are rotated by the body orientation.
#[allow(clippy::too_many_arguments)]
fn capsule_vs_capsule(
    pos_a: Vec3,
    radius_a: f32,
    half_height_a: f32,
    rot_a: Quat,
    pos_b: Vec3,
    radius_b: f32,
    half_height_b: f32,
    rot_b: Quat,
) -> Option<Contact> {
    let ax = rot_a * Vec3::Y;
    let bx = rot_b * Vec3::Y;
    let bot_a = pos_a - ax * half_height_a;
    let bot_b = pos_b - bx * half_height_b;

    let seg_a = ax * (2.0 * half_height_a);
    let seg_b = bx * (2.0 * half_height_b);
    let diff = bot_b - bot_a;
    let a = seg_a.dot(seg_a);
    let b = seg_a.dot(seg_b);
    let c = seg_b.dot(seg_b);
    let d = seg_a.dot(diff);
    let e = seg_b.dot(diff);
    let det = a * c - b * b;

    let (t_a, t_b) = if det.abs() < 1e-10 {
        (0.0, if c > 0.0 { e / c } else { 0.0 })
    } else {
        ((b * e - c * d) / det, (a * e - b * d) / det)
    };
    let t_a = clamp01(t_a);
    let t_b = clamp01(t_b);

    let closest_a = bot_a + seg_a * t_a;
    let closest_b = bot_b + seg_b * t_b;
    let diff2 = closest_b - closest_a;
    let dist_sq = diff2.length_squared();
    let radius_sum = radius_a + radius_b;
    if dist_sq > radius_sum * radius_sum || dist_sq < 1e-10 {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = diff2 / dist;
    let penetration = radius_sum - dist;
    Some(Contact {
        normal,
        penetration,
        contact_point: (closest_a + closest_b) * 0.5,
    })
}

fn detect_collisions(bodies: &[RigidBody], active: &[(usize, usize)]) -> Vec<Manifold> {
    let mut manifolds = Vec::new();
    for &(i, j) in active {
        let a = &bodies[i];
        let b = &bodies[j];
        if a.body_type == BodyType::Static && b.body_type == BodyType::Static {
            continue;
        }

        let manifold = match (&a.shape, &b.shape) {
            (&Shape::Sphere { radius: ra }, &Shape::Sphere { radius: rb }) => {
                sphere_vs_sphere(a.position, ra, b.position, rb).map(|c| Manifold::single(i, j, c))
            }
            (&Shape::Sphere { radius: ra }, &Shape::Box { half_extents: hb }) => {
                sphere_vs_obb(a.position, ra, b.position, hb, b.orientation)
                    .map(|c| Manifold::single(i, j, c))
            }
            (&Shape::Box { half_extents: ha }, &Shape::Sphere { radius: rb }) => {
                sphere_vs_obb(b.position, rb, a.position, ha, a.orientation).map(|c| {
                    Manifold::single(
                        i,
                        j,
                        Contact {
                            normal: -c.normal,
                            penetration: c.penetration,
                            contact_point: c.contact_point,
                        },
                    )
                })
            }
            (&Shape::Box { half_extents: ha }, &Shape::Box { half_extents: hb }) => {
                box_manifold(a.position, ha, a.orientation, b.position, hb, b.orientation)
            }
            (
                &Shape::Capsule {
                    radius: ra,
                    half_height: ha,
                },
                &Shape::Capsule {
                    radius: rb,
                    half_height: hb,
                },
            ) => capsule_vs_capsule(
                a.position,
                ra,
                ha,
                a.orientation,
                b.position,
                rb,
                hb,
                b.orientation,
            )
            .map(|c| Manifold::single(i, j, c)),
            (
                &Shape::Sphere { radius: r },
                &Shape::Capsule {
                    radius: cr,
                    half_height: hh,
                },
            ) => sphere_vs_capsule(a.position, r, b.position, cr, hh, b.orientation)
                .map(|c| Manifold::single(i, j, c)),
            (
                &Shape::Capsule {
                    radius: cr,
                    half_height: hh,
                },
                &Shape::Sphere { radius: r },
            ) => sphere_vs_capsule(b.position, r, a.position, cr, hh, a.orientation).map(|c| {
                Manifold::single(
                    i,
                    j,
                    Contact {
                        normal: -c.normal,
                        penetration: c.penetration,
                        contact_point: c.contact_point,
                    },
                )
            }),
            _ => None,
        };

        if let Some(mut m) = manifold {
            m.body_a = i;
            m.body_b = j;
            manifolds.push(m);
        }
    }
    manifolds
}

struct SweepAndPrune {
    aabbs: Vec<AABB>,
    active: Vec<(usize, usize)>,
    sort_axis: usize,
}

impl SweepAndPrune {
    fn new() -> Self {
        Self {
            aabbs: Vec::new(),
            active: Vec::new(),
            sort_axis: 0,
        }
    }

    fn update(&mut self, bodies: &[RigidBody]) {
        self.aabbs = bodies.iter().map(compute_aabb).collect();
        self.sort_axis = (self.sort_axis + 1) % 3;
        self.active.clear();

        let n = self.aabbs.len();
        let mut starts: Vec<(f32, usize)> = self
            .aabbs
            .iter()
            .enumerate()
            .map(|(i, aabb)| match self.sort_axis {
                0 => (aabb.min.x, i),
                1 => (aabb.min.y, i),
                _ => (aabb.min.z, i),
            })
            .collect();
        starts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for sweep_pos in 0..n {
            let i = starts[sweep_pos].1;
            let sweep_aabb = &self.aabbs[i];
            let end = match self.sort_axis {
                0 => sweep_aabb.max.x,
                1 => sweep_aabb.max.y,
                _ => sweep_aabb.max.z,
            };
            for &(_pos, j) in &starts[(sweep_pos + 1)..] {
                let start_j = match self.sort_axis {
                    0 => self.aabbs[j].min.x,
                    1 => self.aabbs[j].min.y,
                    _ => self.aabbs[j].min.z,
                };
                if start_j > end {
                    break;
                }
                if i < j && sweep_aabb.overlaps(&self.aabbs[j]) {
                    self.active.push((i, j));
                }
            }
        }
    }
}

/// Cached contact point for warm starting (G2b): world-space point plus the
/// accumulated normal impulse from the previous substep.
#[derive(Clone, Copy, Debug)]
struct WarmPoint {
    /// Body-frame anchors of the contact point on both bodies. Unlike the
    /// world position, these are stable while the same surface feature stays
    /// in contact (Jolt persists contacts by feature id the same way).
    la: Vec3,
    lb: Vec3,
    /// Contact normal at cache time: a corner rolling from one face to the
    /// next is a DIFFERENT feature and must not inherit the impulse.
    normal: Vec3,
    impulse: f32,
}

/// Warm-start cache: per body pair, up to 4 matched contact points.
type WarmCache = HashMap<(usize, usize), ([WarmPoint; 4], usize)>;

pub struct BuiltinPhysicsEngine {
    bodies: Vec<RigidBody>,
    broadphase: SweepAndPrune,
    gravity: Vec3,
    substeps: u32,
    velocity_iterations: u32,
    position_iterations: u32,
    /// CFM softness scale for the positional pass (G3): 0 = rigid, larger =
    /// softer, smoother corrections spread over more iterations.
    contact_softness: f32,
    /// Accumulated normal impulses per matched contact point, keyed by sorted
    /// body pair. Applied in a dedicated WarmStart stage (G2b).
    warm_impulses: WarmCache,
    /// Island (constraint-graph component) per body, rebuilt every step from
    /// the contact graph (G4). Sleep and wake are island-coherent, exactly
    /// like Jolt/Box3D: a resting stack can only sleep as a whole, otherwise
    /// an awake neighbour's contact immediately re-wakes a per-body sleeper.
    island: Vec<u32>,
    /// Per-island sleep timers, keyed by island root handle.
    island_timers: HashMap<u32, f32>,
    asleep: Vec<bool>,
}

impl BuiltinPhysicsEngine {
    pub fn new(gravity: Vec3) -> Self {
        Self {
            bodies: Vec::new(),
            broadphase: SweepAndPrune::new(),
            gravity,
            substeps: 12,
            velocity_iterations: 8,
            position_iterations: 4,
            contact_softness: 0.0,
            warm_impulses: HashMap::new(),
            island: Vec::new(),
            island_timers: HashMap::new(),
            asleep: Vec::new(),
        }
    }

    pub fn set_substeps(&mut self, n: u32) {
        self.substeps = n;
    }

    pub fn set_velocity_iterations(&mut self, n: u32) {
        self.velocity_iterations = n;
    }

    pub fn set_position_iterations(&mut self, n: u32) {
        self.position_iterations = n;
    }

    pub fn set_contact_softness(&mut self, softness: f32) {
        self.contact_softness = softness;
    }

    fn integrate(&mut self, dt: f32) {
        for (h, body) in self.bodies.iter_mut().enumerate() {
            if body.body_type != BodyType::Dynamic || self.asleep.get(h).copied().unwrap_or(false) {
                continue;
            }
            // Linear integrate (semi-implicit).
            body.velocity += self.gravity * dt;
            body.position += body.velocity * dt;

            // Angular integrate from applied torque.
            if body.torque != Vec3::ZERO {
                body.angular_velocity +=
                    mul_inv_inertia(body.inertia, body.orientation, body.torque * dt);
                body.torque = Vec3::ZERO;
            }

            // Rotation: exact small-step quaternion update (exp of angular velocity * dt).
            if body.angular_velocity != Vec3::ZERO {
                let dwq = Quat::from_scaled_axis(body.angular_velocity * dt);
                body.orientation = (dwq * body.orientation).normalize();
            }
        }
    }

    /// Rebuild the constraint-graph islands (union-find over dynamic bodies
    /// connected by a contact manifold). Static bodies never join islands —
    /// they anchor them, like in Jolt.
    fn rebuild_islands(&mut self, manifolds: &[Manifold]) {
        let n = self.bodies.len();
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        for m in manifolds {
            let (a, b) = (m.body_a, m.body_b);
            if self.bodies[a].body_type == BodyType::Dynamic
                && self.bodies[b].body_type == BodyType::Dynamic
            {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[rb] = ra;
                }
            }
        }
        // A fully sleeping island keeps its composition even if the contact
        // detection blinks for a step: its members are not integrated, so
        // their relative geometry cannot change — dissolving the island
        // would let one member wake while its support stays asleep.
        for a in 0..n {
            if !self.asleep.get(a).copied().unwrap_or(false) {
                continue;
            }
            for b in (a + 1)..n {
                if self.asleep.get(b).copied().unwrap_or(false)
                    && self.island[a] == self.island[b]
                    && self.island[a] != u32::MAX
                {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    if ra != rb {
                        parent[rb] = ra;
                    }
                }
            }
        }
        for h in 0..n {
            self.island[h] = if self.bodies[h].body_type == BodyType::Dynamic {
                find(&mut parent, h) as u32
            } else {
                u32::MAX
            };
        }
        // Drop timers of roots that no longer exist.
        let roots: std::collections::HashSet<u32> = self.island.iter().copied().collect();
        self.island_timers.retain(|r, _| roots.contains(r));
    }

    /// Island-coherent sleep bookkeeping, run once per step (G4): an island
    /// whose bodies ALL stay slow for SLEEP_TIME seconds is frozen as a
    /// whole; islands are woken as a whole by contact with an awake body
    /// (see resolve_manifolds).
    fn update_sleep(&mut self, dt: f32) {
        // Well below anything gameplay-visible, above the solver's settled
        // jitter floor (~0.01-0.03).
        const LIN_SLEEP: f32 = 0.15;
        const ANG_SLEEP: f32 = 0.15;
        const SLEEP_TIME: f32 = 0.5;
        let n = self.bodies.len();
        // An island is quiet only if every awake member is slow.
        let mut quiet: HashMap<u32, bool> = HashMap::new();
        for h in 0..n {
            if self.island[h] == u32::MAX || self.asleep[h] {
                continue;
            }
            let b = &self.bodies[h];
            let slow = b.velocity.length() < LIN_SLEEP && b.angular_velocity.length() < ANG_SLEEP;
            quiet
                .entry(self.island[h])
                .and_modify(|q| *q &= slow)
                .or_insert(slow);
        }
        let mut to_sleep: Vec<u32> = Vec::new();
        for (root, q) in quiet {
            let timer = self.island_timers.entry(root).or_insert(0.0);
            if q {
                *timer += dt;
                if *timer >= SLEEP_TIME {
                    to_sleep.push(root);
                }
            } else {
                *timer = 0.0;
            }
        }
        for root in to_sleep {
            for h in 0..n {
                if self.island[h] == root {
                    self.asleep[h] = true;
                    let b = &mut self.bodies[h];
                    b.velocity = Vec3::ZERO;
                    b.angular_velocity = Vec3::ZERO;
                }
            }
        }
    }

    /// Wake the whole island containing body `h` (contact with an awake body
    /// propagates motion through the island, so partial wake is incoherent).
    fn wake_island(&mut self, h: usize) {
        let root = self.island[h];
        for b in 0..self.bodies.len() {
            if self.island[b] == root {
                self.asleep[b] = false;
            }
        }
        self.island_timers.insert(root, 0.0);
    }

    // Solver loops index several parallel per-point arrays (manifold points,
    // warm cache, accumulators); range loops are the clearest form here.
    #[allow(clippy::needless_range_loop)]
    fn resolve_manifolds(&mut self, manifolds: &[Manifold], allow_restitution: bool) {
        // G2b: warm-start cache matches points by proximity, not by index —
        // manifold point order changes frame to frame (sorted by depth).
        const MATCH_TOL_SQ: f32 = 0.05 * 0.05;
        // Below this approach speed restitution is skipped (avoids jitter).
        const RESTITUTION_THRESHOLD: f32 = 1.0;
        // Restitution fires only on genuine impacts = shallow penetration.
        // A deep, spinning, penetrating contact is solver-recovery state
        // (NGS is still extracting the body); restituting there converts the
        // approach speed into a huge angular kick at the corner lever, and
        // the spin presents as even faster approach next step — an energy
        // pump. Deep contacts recover purely inelastically (dissipative).
        const RESTITUTION_MAX_PEN: f32 = 0.05;

        // Per-manifold solver state, index-aligned via `mi`.
        struct State {
            mi: usize,
            i: usize,
            j: usize,
            count: usize,
            acc: [f32; 4],
            acc_friction: [f32; 4],
            acc_friction2: [f32; 4],
            bias: [f32; 4],
            mu: f32,
            // Fixed tangent basis (Box2D-style): friction is solved along
            // directions derived from the contact normal ONCE, not from the
            // instantaneous slip velocity — velocity-aligned friction walks
            // the contact and lets resting stacks drift sideways.
            t1: Vec3,
            t2: Vec3,
            // G3: body-frame anchors and detection-time penetration per point,
            // so the positional pass can re-measure live separation.
            la: [Vec3; 4],
            lb: [Vec3; 4],
            pen0: [f32; 4],
        }

        let mut states: Vec<State> = Vec::with_capacity(manifolds.len());
        for (mi, m) in manifolds.iter().enumerate() {
            let (i, j) = (m.body_a, m.body_b);
            let total_inv = self.bodies[i].inv_mass + self.bodies[j].inv_mass;
            if total_inv < 1e-10 {
                continue;
            }
            // Sleep: a contact needs work only if at least one side is an
            // AWAKE DYNAMIC body. Static geometry never wakes anything (a
            // body asleep on the floor must stay asleep); an awake dynamic
            // partner wakes the sleeper's whole island.
            let ai = self.asleep[i] || self.bodies[i].body_type != BodyType::Dynamic;
            let aj = self.asleep[j] || self.bodies[j].body_type != BodyType::Dynamic;
            if ai && aj {
                continue;
            }
            if self.asleep[i] && !aj {
                self.wake_island(i);
            }
            if self.asleep[j] && !ai {
                self.wake_island(j);
            }
            let n = m.normal;
            let key = (i.min(j), i.max(j));
            let count = m.point_count;

            // --- Body-frame anchors first: matching and G3 both need them ---
            let mut la = [Vec3::ZERO; 4];
            let mut lb = [Vec3::ZERO; 4];
            let mut pen0 = [0.0f32; 4];
            for k in 0..count {
                let p = m.points[k].world_point;
                la[k] = self.bodies[i].orientation.inverse() * (p - self.bodies[i].position);
                lb[k] = self.bodies[j].orientation.inverse() * (p - self.bodies[j].position);
                pen0[k] = m.points[k].penetration;
            }

            // --- Match cached impulses by body-frame anchors (feature
            // persistence, Jolt-style): stable while the same surface feature
            // stays in contact, even when the bodies move fast in world space.
            let mut warm = [0.0f32; 4];
            let mut matched = [false; 4];
            if let Some((cached_points, cached_count)) = self.warm_impulses.get(&key) {
                let mut used = [false; 4];
                for k in 0..count {
                    let mut best: Option<(usize, f32)> = None;
                    for (c, cp) in cached_points.iter().enumerate().take(*cached_count) {
                        if used[c] {
                            continue;
                        }
                        // Feature compatibility: same surface region AND a
                        // compatible contact normal (rolling over an edge
                        // changes the feature, dot < 0.7 => no match).
                        if cp.normal.dot(n) < 0.7 {
                            continue;
                        }
                        let d2 =
                            (cp.la - la[k]).length_squared() + (cp.lb - lb[k]).length_squared();
                        if d2 < MATCH_TOL_SQ && best.is_none_or(|(_, bd)| d2 < bd) {
                            best = Some((c, d2));
                        }
                    }
                    if let Some((c, _)) = best {
                        used[c] = true;
                        warm[k] = cached_points[c].impulse;
                        matched[k] = true;
                    }
                }
            }

            // Restitution bias from the pre-solve approach velocity — only on
            // the first substep of a step and only for NEW (unmatched) points:
            // one bounce per impact event. A persistent contact must never
            // re-restitute — the NGS position pass would feed it fresh
            // approach velocity every step and the bounce becomes an energy
            // pump (Box3D applies restitution as a one-shot, never cached).
            let e = self.bodies[i].restitution.min(self.bodies[j].restitution);
            let mu = self.bodies[i].friction.max(self.bodies[j].friction);
            let mut bias = [0.0f32; 4];
            if allow_restitution {
                for k in 0..count {
                    if matched[k] || pen0[k] > RESTITUTION_MAX_PEN {
                        continue;
                    }
                    let p = m.points[k].world_point;
                    let ra = p - self.bodies[i].position;
                    let rb = p - self.bodies[j].position;
                    let vn0 = (point_velocity(&self.bodies[j], rb)
                        - point_velocity(&self.bodies[i], ra))
                    .dot(n);
                    if vn0 < -RESTITUTION_THRESHOLD {
                        bias[k] = -e * vn0;
                    }
                }
            }

            // --- WarmStart stage: apply cached impulses once (Box2D pattern) ---
            // Capped so the warm impulse can never push the pair APART faster
            // than they currently approach: a stale cached impulse applied to
            // a separating (or nearly static) contact is pure energy
            // injection, repeated 240×/s (this was the high-spin pump).
            let mut warm_applied = warm;
            for k in 0..count {
                if warm[k] > 0.0 {
                    let p = m.points[k].world_point;
                    let ra = p - self.bodies[i].position;
                    let rb = p - self.bodies[j].position;
                    let k_eff = effective_mass(&self.bodies, i, j, n, ra, rb);
                    if k_eff < 1e-10 {
                        warm_applied[k] = 0.0;
                        continue;
                    }
                    let vn_pre = (point_velocity(&self.bodies[j], rb)
                        - point_velocity(&self.bodies[i], ra))
                    .dot(n);
                    let applied = warm[k].min((-vn_pre / k_eff).max(0.0));
                    warm_applied[k] = applied;
                    if applied > 0.0 {
                        apply_impulse(&mut self.bodies, i, j, n * applied, ra, rb);
                    }
                }
            }

            states.push(State {
                mi,
                i,
                j,
                count,
                acc: warm_applied,
                acc_friction: [0.0; 4],
                acc_friction2: [0.0; 4],
                bias,
                mu,
                t1: tangent_basis(n),
                t2: tangent_basis(n).cross(n),
                la,
                lb,
                pen0,
            });
        }

        // --- Velocity solve: Gauss-Seidel iterations over ALL manifolds ---
        for _ in 0..self.velocity_iterations {
            for st in states.iter_mut() {
                let m = &manifolds[st.mi];
                let (i, j) = (st.i, st.j);
                let n = m.normal;
                let total_inv = self.bodies[i].inv_mass + self.bodies[j].inv_mass;

                // ---- Normal direction ----
                // G4: multi-point manifolds are solved as an exact LCP block
                // (scalar per-point GS oscillates between coupled points of
                // one manifold — the rocking pump); single points keep the
                // scalar projected update.
                if st.count >= 2 {
                    let mut pts = [Vec3::ZERO; 4];
                    for k in 0..st.count {
                        pts[k] = m.points[k].world_point;
                    }
                    solve_normal_block(&mut self.bodies, i, j, n, &pts, &mut st.acc, st.count);
                } else {
                    let k = 0;
                    let p = m.points[k].world_point;
                    let ra = p - self.bodies[i].position;
                    let rb = p - self.bodies[j].position;
                    let k_eff = effective_mass(&self.bodies, i, j, n, ra, rb);
                    if k_eff >= 1e-10 {
                        let rel = point_velocity(&self.bodies[j], rb)
                            - point_velocity(&self.bodies[i], ra);
                        let vn = rel.dot(n);
                        // Inelastic contact: restitution is a separate
                        // one-shot stage (below), never accumulated.
                        let lambda = -vn / k_eff;
                        let new_acc = (st.acc[k] + lambda).max(0.0);
                        let delta = new_acc - st.acc[k];
                        st.acc[k] = new_acc;
                        if delta.abs() > 1e-12 {
                            apply_impulse(&mut self.bodies, i, j, n * delta, ra, rb);
                        }
                    }
                }

                // ---- Friction (Coulomb) along the FIXED tangent basis ----
                // Accumulated 2D friction, clamped as a vector to µ·λn.
                // (Re-deriving the tangent from slip velocity every
                // iteration makes the contact "walk" — stacks drift.)
                for k in 0..st.count {
                    let p = m.points[k].world_point;
                    let ra = p - self.bodies[i].position;
                    let rb = p - self.bodies[j].position;
                    let rel =
                        point_velocity(&self.bodies[j], rb) - point_velocity(&self.bodies[i], ra);
                    let max_friction = st.mu * st.acc[k];
                    let mut f_imp = Vec3::ZERO;
                    for axis in 0..2 {
                        let t = if axis == 0 { st.t1 } else { st.t2 };
                        let ra_t = ra.cross(t);
                        let rb_t = rb.cross(t);
                        let k_t = total_inv
                            + ra_t.dot(mul_inv_inertia(
                                self.bodies[i].inertia,
                                self.bodies[i].orientation,
                                ra_t,
                            ))
                            + rb_t.dot(mul_inv_inertia(
                                self.bodies[j].inertia,
                                self.bodies[j].orientation,
                                rb_t,
                            ));
                        if k_t < 1e-10 {
                            continue;
                        }
                        let vt = rel.dot(t);
                        let lambda_t = -vt / k_t;
                        let (cur, other) = if axis == 0 {
                            (st.acc_friction[k], st.acc_friction2[k])
                        } else {
                            (st.acc_friction2[k], st.acc_friction[k])
                        };
                        let new_t = cur + lambda_t;
                        // Circular clamp of the combined friction vector.
                        let len = (new_t * new_t + other * other).sqrt();
                        let new_t = if len > max_friction && len > 1e-12 {
                            new_t * (max_friction / len)
                        } else {
                            new_t
                        };
                        if axis == 0 {
                            f_imp += t * (new_t - st.acc_friction[k]);
                            st.acc_friction[k] = new_t;
                        } else {
                            f_imp += t * (new_t - st.acc_friction2[k]);
                            st.acc_friction2[k] = new_t;
                        }
                    }
                    if f_imp.length_squared() > 1e-24 {
                        apply_impulse(&mut self.bodies, i, j, f_imp, ra, rb);
                    }
                }
            }
        }

        // --- Restitution stage (Box3D b3SolverStage_Restitution analog) ---
        // One-shot per step: push the normal point velocity up to the stored
        // bounce target. NOT accumulated, NOT warm-started — this is what
        // keeps spinning bodies from pumping energy through the bounce.
        if allow_restitution {
            for st in &states {
                let m = &manifolds[st.mi];
                let (i, j) = (st.i, st.j);
                let n = m.normal;
                let total_inv = self.bodies[i].inv_mass + self.bodies[j].inv_mass;
                for k in 0..st.count {
                    if st.bias[k] <= 0.0 {
                        continue;
                    }
                    let p = m.points[k].world_point;
                    let ra = p - self.bodies[i].position;
                    let rb = p - self.bodies[j].position;
                    let ra_n = ra.cross(n);
                    let rb_n = rb.cross(n);
                    let k_eff = total_inv
                        + ra_n.dot(mul_inv_inertia(
                            self.bodies[i].inertia,
                            self.bodies[i].orientation,
                            ra_n,
                        ))
                        + rb_n.dot(mul_inv_inertia(
                            self.bodies[j].inertia,
                            self.bodies[j].orientation,
                            rb_n,
                        ));
                    if k_eff < 1e-10 {
                        continue;
                    }
                    let vn = (point_velocity(&self.bodies[j], rb)
                        - point_velocity(&self.bodies[i], ra))
                    .dot(n);
                    let lambda = (st.bias[k] - vn) / k_eff;
                    if lambda > 0.0 {
                        apply_impulse(&mut self.bodies, i, j, n * lambda, ra, rb);
                    }
                }
            }
        }

        // ---- G3 split impulse: iterated NGS over ALL manifolds ----
        // Position errors are corrected with pseudo-motion only — real
        // velocities are never touched. Each iteration re-measures the LIVE
        // separation at the stored body-frame anchors, so corrections
        // distribute evenly across the whole manifold set (stacks) instead
        // of one-shot rigid pushes, which seeded the micro-tilts from G2b.
        // β is kept low (0.2): stronger pseudo-correction resonates with the
        // velocity solve on rocking contacts and pumps the rock mode.
        const SLOP: f32 = 0.02;
        const MAX_CORRECTION: f32 = 0.25;
        const BETA_POS: f32 = 0.2;
        for _ in 0..self.position_iterations {
            for st in &states {
                let m = &manifolds[st.mi];
                let (i, j) = (st.i, st.j);
                let n = m.normal;
                let inv_mass_a = self.bodies[i].inv_mass;
                let inv_mass_b = self.bodies[j].inv_mass;
                let total_inv = inv_mass_a + inv_mass_b;
                let cfm = self.contact_softness * total_inv;
                for k in 0..st.count {
                    let (pos_a, rot_a) = (self.bodies[i].position, self.bodies[i].orientation);
                    let (pos_b, rot_b) = (self.bodies[j].position, self.bodies[j].orientation);
                    // Live world anchors; at detection they coincided, so the
                    // separation along n started at -pen0.
                    let wa = pos_a + rot_a * st.la[k];
                    let wb = pos_b + rot_b * st.lb[k];
                    let separation = (wb - wa).dot(n) - st.pen0[k];
                    let c = (-separation - SLOP).clamp(0.0, MAX_CORRECTION);
                    if c <= 0.0 {
                        continue;
                    }
                    let ra = wa - pos_a;
                    let rb = wb - pos_b;
                    let ra_n = ra.cross(n);
                    let rb_n = rb.cross(n);
                    let k_pos = total_inv
                        + ra_n.dot(mul_inv_inertia(self.bodies[i].inertia, rot_a, ra_n))
                        + rb_n.dot(mul_inv_inertia(self.bodies[j].inertia, rot_b, rb_n));
                    let k_soft = make_soft(k_pos, cfm);
                    if k_soft < 1e-10 {
                        continue;
                    }
                    let lam = BETA_POS * c / k_soft;
                    apply_positional_impulse(&mut self.bodies, i, j, n * lam, ra, rb);
                }
            }
        }

        // --- Persist the cache for the next substep/step ---
        let mut next: WarmCache = HashMap::new();
        for st in &states {
            let m = &manifolds[st.mi];
            let mut pts = [WarmPoint {
                la: Vec3::ZERO,
                lb: Vec3::ZERO,
                normal: Vec3::ZERO,
                impulse: 0.0,
            }; 4];
            for k in 0..st.count {
                pts[k] = WarmPoint {
                    la: st.la[k],
                    lb: st.lb[k],
                    normal: m.normal,
                    impulse: st.acc[k],
                };
            }
            next.insert((st.i.min(st.j), st.i.max(st.j)), (pts, st.count));
        }
        self.warm_impulses = next;
    }

    fn raycast_body(&self, ray: &Ray, handle: usize, max_dist: f32) -> Option<RaycastHit> {
        let body = &self.bodies[handle];
        let aabb = body.shape.aabb(body.position, body.orientation);
        let inv_dir = Vec3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );

        let t1 = (aabb.min - ray.origin) * inv_dir;
        let t2 = (aabb.max - ray.origin) * inv_dir;
        let t_min = t1.min(t2);
        let t_max = t1.max(t2);
        let enter = t_min.x.max(t_min.y.max(t_min.z));
        let exit = t_max.x.min(t_max.y.min(t_max.z));

        if enter > exit || exit < 0.0 || enter > max_dist {
            return None;
        }

        let t = enter.max(0.0);
        let point = ray.point_at(t);
        // Approximate surface normal pointing away from the shape centre.
        let normal = (point - body.position).normalize_or_zero();
        if normal.length_squared() < 0.5 {
            return None;
        }
        Some(RaycastHit {
            handle,
            point,
            normal,
            distance: t,
        })
    }
}

impl PhysicsEngine for BuiltinPhysicsEngine {
    fn step(&mut self, dt: f32) {
        let sub_dt = dt / self.substeps as f32;
        let mut last_manifolds = Vec::new();
        for s in 0..self.substeps {
            self.integrate(sub_dt);
            self.broadphase.update(&self.bodies);
            let manifolds = detect_collisions(&self.bodies, &self.broadphase.active);
            // Restitution is one-shot per step, evaluated on the first substep.
            self.resolve_manifolds(&manifolds, s == 0);
            last_manifolds = manifolds;
        }
        self.rebuild_islands(&last_manifolds);
        self.update_sleep(dt);
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        let handle = self.bodies.len();
        let island_id = if body.body_type == BodyType::Dynamic {
            handle as u32
        } else {
            u32::MAX
        };
        self.bodies.push(body);
        self.island.push(island_id);
        self.asleep.push(false);
        handle
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        if handle < self.bodies.len() {
            self.bodies.swap_remove(handle);
            self.island.swap_remove(handle);
            self.asleep.swap_remove(handle);
            // swap_remove shifts the last body's index; warm-start keys are
            // body indices, so the cache is no longer valid.
            self.warm_impulses.clear();
        }
    }

    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        self.bodies.get(handle)
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        self.bodies.get_mut(handle)
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit> {
        let mut closest: Option<RaycastHit> = None;
        for handle in 0..self.bodies.len() {
            if let Some(hit) = self.raycast_body(&ray, handle, max_dist) {
                match &closest {
                    Some(best) if hit.distance < best.distance => closest = Some(hit),
                    None => closest = Some(hit),
                    _ => {}
                }
            }
        }
        closest
    }

    fn shapecast(&self, shape: &Shape, from: Vec3, to: Vec3) -> Option<RaycastHit> {
        // Conservative sweep by reusing the narrow-phase over a probe body.
        let delta = to - from;
        if delta.length_squared() < 1e-9 {
            return None;
        }
        const STEPS: usize = 64;
        for s in 1..=STEPS {
            let pos = from + delta * (s as f32 / STEPS as f32);
            let probe = RigidBody {
                position: pos,
                orientation: Quat::IDENTITY,
                velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
                mass: 1.0,
                inv_mass: 1.0,
                inertia: shape.inertia(1.0),
                torque: Vec3::ZERO,
                restitution: 0.0,
                friction: 0.0,
                shape: shape.clone(),
                body_type: BodyType::Dynamic,
            };
            for (handle, body) in self.bodies.iter().enumerate() {
                let pair = [probe.clone(), body.clone()];
                if !detect_collisions(&pair, &[(0, 1)]).is_empty() {
                    return Some(RaycastHit {
                        handle,
                        point: pos,
                        normal: Vec3::ZERO,
                        distance: (pos - from).length(),
                    });
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_falls() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let sphere = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 10.0, 0.0), 1.0, 1.0));
        physics.step(1.0 / 60.0);
        let body = physics.get_body(sphere).unwrap();
        assert!(body.position.y < 10.0);
    }

    #[test]
    fn static_body_does_not_fall() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let ground = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(10.0, 1.0, 10.0),
            0.0,
        ));
        physics.step(1.0 / 60.0);
        let body = physics.get_body(ground).unwrap();
        assert_eq!(body.position.y, -1.0);
    }

    #[test]
    fn sphere_vs_sphere_collision() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let a = physics.add_body(RigidBody::new_sphere(Vec3::new(-0.4, 0.0, 0.0), 0.5, 1.0));
        let b = physics.add_body(RigidBody::new_sphere(Vec3::new(0.4, 0.0, 0.0), 0.5, 1.0));
        physics.step(1.0 / 60.0);
        let body_a = physics.get_body(a).unwrap();
        let body_b = physics.get_body(b).unwrap();
        let dist = (body_a.position - body_b.position).length();
        assert!(dist < 1.1);
    }

    #[test]
    fn raycast_hits_sphere() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0, 1.0));
        let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        let hit = physics.raycast(ray, 10.0);
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert!((hit.distance - 4.0).abs() < 0.01);
    }

    #[test]
    fn box_vs_box_collision() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let a = physics.add_body(RigidBody::new_box(
            Vec3::new(-0.4, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            1.0,
        ));
        let b = physics.add_body(RigidBody::new_box(
            Vec3::new(0.4, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            1.0,
        ));
        physics.step(1.0 / 60.0);
        let body_a = physics.get_body(a).unwrap();
        let body_b = physics.get_body(b).unwrap();
        let dist = (body_a.position - body_b.position).length();
        assert!(dist < 1.1);
    }

    // ---- G1: orientation + angular dynamics ----

    #[test]
    fn angular_velocity_rotates_body() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let handle = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0));
        physics
            .get_body_mut(handle)
            .unwrap()
            .set_angular_velocity(Vec3::new(0.0, 2.0, 0.0));
        physics.step(1.0 / 60.0);
        let body = physics.get_body(handle).unwrap();
        // Orientation must have changed and remain a unit quaternion.
        assert!(
            body.orientation.to_axis_angle().1.abs() > 1e-4,
            "should have rotated about Y"
        );
        assert!(
            (body.orientation.length() - 1.0).abs() < 1e-4,
            "unit quaternion preserved"
        );
    }

    #[test]
    fn torque_turns_body() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let sphere = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0));
        // Apply torque around Z -> angular velocity must appear.
        let w_after = {
            physics
                .get_body_mut(sphere)
                .unwrap()
                .apply_torque(Vec3::new(0.0, 0.0, 1.0));
            physics.step(1.0 / 60.0);
            physics.get_body(sphere).unwrap().angular_velocity
        };
        assert!(
            w_after.z.abs() > 1e-5,
            "torque must produce angular velocity, got {w_after:?}"
        );
    }

    #[test]
    fn oriented_boxes_collide() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        // Same center, rotated 45° about Y, box-ish units: OBB-OBB should separate.
        let half = Vec3::new(0.5, 0.5, 0.5);
        let a = physics.add_body(
            RigidBody::new_box(Vec3::new(0.0, 0.0, 0.0), half, 1.0)
                .with_orientation(Quat::from_rotation_z(0.0)),
        );
        let b = physics.add_body(
            RigidBody::new_box(Vec3::new(0.4, 0.0, 0.0), half, 1.0)
                .with_orientation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        );
        physics.step(1.0 / 60.0);
        let body_a = physics.get_body(a).unwrap();
        let body_b = physics.get_body(b).unwrap();
        // Resting separation for two half-0.5 cubes is exactly 1.0 (touching).
        let dist = (body_a.position - body_b.position).length();
        assert!(dist <= 1.05, "oriented boxes should resolve, dist={dist}");
    }

    #[test]
    fn obb_aabb_respects_rotation() {
        let half = Vec3::new(1.0, 1.0, 1.0);
        let q = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let aabb = Shape::Box { half_extents: half }.aabb(Vec3::ZERO, q);
        // The rotated corner (1,1,1) -> (0, 1.414, 1) must be inside the bounding AABB.
        let corner = q * Vec3::splat(1.0);
        assert!(
            aabb.contains_point(corner),
            "corner {corner:?} not inside {aabb:?}"
        );
        // Y half-extent grows to sqrt(2) after the 45° rotation.
        let half_y = (aabb.max.y - aabb.min.y) * 0.5;
        assert!(
            (half_y - 2f32.sqrt()).abs() < 1e-3,
            "OBB->AABB y-extent, got {half_y}"
        );
    }

    #[test]
    fn sphere_capsule_collision() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let sphere = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.0, 0.0), 0.5, 1.0));
        let capsule = physics.add_body(
            RigidBody::new_capsule(Vec3::new(0.6, 0.0, 0.0), 0.5, 1.0, 1.0)
                .with_orientation(glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        );
        physics.step(1.0 / 60.0);
        let d = (physics.get_body(sphere).unwrap().position
            - physics.get_body(capsule).unwrap().position)
            .length();
        // Sphere radius 0.5 + capsule radius 0.5 -> resting center distance ~1.0.
        assert!(d <= 1.05, "sphere/capsule should resolve on contact, d={d}");
    }

    #[test]
    fn shapecast_hits_body() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0, 0.0));
        // Cast a small sphere from origin toward the static target.
        let shape = Shape::Sphere { radius: 0.1 };
        let hit = physics.shapecast(&shape, Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0));
        assert!(hit.is_some(), "conservative shapecast should hit");
        let hit = hit.unwrap();
        assert_eq!(hit.handle, 0);
        assert!(
            hit.distance > 3.0 && hit.distance < 10.0,
            "hit distance={}",
            hit.distance
        );
    }

    #[test]
    fn box_manifold_produces_four_points() {
        // Two equal half-0.5 boxes, overlapping by 0.25 along +Y: the resting
        // face yields 4 manifold points (vertex-face contact), not one.
        let half = Vec3::new(0.5, 0.5, 0.5);
        let m = box_manifold(
            Vec3::new(0.0, 0.0, 0.0),
            half,
            Quat::IDENTITY,
            Vec3::new(0.0, 0.75, 0.0),
            half,
            Quat::IDENTITY,
        )
        .expect("boxes overlap");
        assert_eq!(m.point_count, 4, "expected a 4-point manifold");
        assert!((m.normal - Vec3::Y).length() < 1e-3, "normal should be +Y");
        for k in 0..m.point_count {
            assert!(
                m.points[k].penetration > 0.0,
                "point {k} has positive penetration"
            );
        }
    }

    #[test]
    fn box_rests_on_static_floor() {
        // A box in free fall must settle on a static floor (G2b target).
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let top = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.8, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            1.0,
        ));
        for _ in 0..240 {
            physics.step(1.0 / 60.0);
        }
        let b = physics.get_body(top).unwrap();
        assert!(
            b.position.y > 0.40 && b.position.y < 0.55,
            "box should rest at y≈0.5, got {}",
            b.position.y
        );
        assert!(
            b.velocity.length() < 0.05,
            "settled velocity: {:?}",
            b.velocity
        );
        assert!(
            b.angular_velocity.length() < 0.05,
            "no jitter: {:?}",
            b.angular_velocity
        );
    }

    #[test]
    fn sphere_rests_on_static_floor() {
        // G2 gate: a sphere dropped on a static floor settles and stays.
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let ball = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 2.0, 0.0), 0.5, 1.0));
        for _ in 0..240 {
            physics.step(1.0 / 60.0);
        }
        let b = physics.get_body(ball).unwrap();
        assert!(
            b.position.y > 0.40 && b.position.y < 0.55,
            "sphere should rest at y≈0.5, got {}",
            b.position.y
        );
        assert!(
            b.velocity.length() < 0.05,
            "settled velocity: {:?}",
            b.velocity
        );
    }

    #[test]
    fn two_box_stack_stays_stable() {
        // G2 gate: a 2-box stack stands for 5 seconds without drift or toppling.
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let lower = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            1.0,
        ));
        let upper = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 1.55, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            1.0,
        ));
        for _ in 0..300 {
            physics.step(1.0 / 60.0);
        }
        let lo = physics.get_body(lower).unwrap();
        let hi = physics.get_body(upper).unwrap();
        assert!(
            (lo.position.y - 0.5).abs() < 0.05,
            "lower box rest height, got {}",
            lo.position.y
        );
        assert!(
            (hi.position.y - 1.5).abs() < 0.08,
            "upper box rest height, got {}",
            hi.position.y
        );
        // No horizontal drift: the stack must stay centred.
        assert!(
            lo.position.x.abs() < 0.05 && lo.position.z.abs() < 0.05,
            "lower box drifted: {:?}",
            lo.position
        );
        assert!(
            hi.position.x.abs() < 0.08 && hi.position.z.abs() < 0.08,
            "upper box drifted: {:?}",
            hi.position
        );
        assert!(
            lo.velocity.length() < 0.05 && hi.velocity.length() < 0.05,
            "stack not settled: {:?} / {:?}",
            lo.velocity,
            hi.velocity
        );
        assert!(
            lo.angular_velocity.length() < 0.05 && hi.angular_velocity.length() < 0.05,
            "stack spinning: {:?} / {:?}",
            lo.angular_velocity,
            hi.angular_velocity
        );
    }

    #[test]
    fn four_box_stack_stays_stable() {
        // G3 gate: a 4-box stack stands for 5 seconds without drift or topple.
        // Taller stacks need the iterated cross-manifold position solve —
        // per-manifold nested correction cannot balance the chain.
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let mut handles = Vec::new();
        for level in 0..4 {
            handles.push(physics.add_body(RigidBody::new_box(
                Vec3::new(0.0, 0.5 + level as f32 * 1.02, 0.0),
                Vec3::new(0.5, 0.5, 0.5),
                1.0,
            )));
        }
        for _ in 0..300 {
            physics.step(1.0 / 60.0);
        }
        for (level, &h) in handles.iter().enumerate() {
            let b = physics.get_body(h).unwrap();
            let expected_y = 0.5 + level as f32;
            assert!(
                (b.position.y - expected_y).abs() < 0.1,
                "box {level} rest height ≈{expected_y}, got {}",
                b.position.y
            );
            assert!(
                b.position.x.abs() < 0.15 && b.position.z.abs() < 0.15,
                "box {level} drifted: {:?}",
                b.position
            );
            assert!(
                b.velocity.length() < 0.08,
                "box {level} not settled: {:?}",
                b.velocity
            );
            assert!(
                b.angular_velocity.length() < 0.08,
                "box {level} spinning: {:?}",
                b.angular_velocity
            );
        }
    }

    #[test]
    fn tilted_box_falls_flat() {
        // G3 gate: a box dropped at a 20° tilt lands on an edge, tips over,
        // and comes to rest flat on the floor (4-point face manifold).
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let tilt = Quat::from_rotation_z(20.0f32.to_radians());
        let top = physics.add_body(
            RigidBody::new_box(Vec3::new(0.0, 1.2, 0.0), Vec3::new(0.5, 0.5, 0.5), 1.0)
                .with_orientation(tilt),
        );
        for _ in 0..300 {
            physics.step(1.0 / 60.0);
        }
        let b = physics.get_body(top).unwrap();
        // Resting flat: the box's local +Y axis must align with world ±Y.
        let up = b.orientation * Vec3::Y;
        assert!(
            up.dot(Vec3::Y).abs() > 0.99,
            "box should lie flat, up={up:?}"
        );
        assert!(
            (b.position.y - 0.5).abs() < 0.08,
            "flat rest height ≈0.5, got {}",
            b.position.y
        );
        assert!(
            b.velocity.length() < 0.05 && b.angular_velocity.length() < 0.05,
            "not settled: {:?} / {:?}",
            b.velocity,
            b.angular_velocity
        );
    }
}
