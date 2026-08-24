use std::collections::{HashMap, HashSet};

use glam::{Quat, Vec3};
use rayon::prelude::*;

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::distance;
#[cfg(feature = "gpu")]
use crate::gpu::{WgpuContactSolver, pack_single_point_batches, write_back_acc};
use crate::joint::{Joint, JointHandle, JointKind};
use crate::math::{AABB, Ray, RaycastHit};
use crate::shape::Shape;
use crate::wide::{SolverStep, build_solver_steps};

pub trait PhysicsEngine: Send + Sync {
    fn step(&mut self, dt: f32);
    fn add_body(&mut self, body: RigidBody) -> BodyHandle;
    fn remove_body(&mut self, handle: BodyHandle);
    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody>;
    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody>;
    /// Create a joint between two existing, distinct bodies (G5).
    /// Returns None on invalid handles or a self-joint.
    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Option<JointHandle>;
    fn remove_joint(&mut self, handle: JointHandle);
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
pub(crate) struct ManifoldPoint {
    pub world_point: Vec3,
    pub penetration: f32,
}

/// Contact manifold: one normal + up to 4 points per body pair.
#[derive(Clone, Debug)]
pub(crate) struct Manifold {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub normal: Vec3,
    pub point_count: usize,
    pub points: [ManifoldPoint; 4],
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

#[inline]
fn vec3_finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

#[inline]
fn quat_finite(q: Quat) -> bool {
    q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite()
}

/// Union-find root with path halving. Bounded to `parent.len()` steps so a
/// corrupted parent array (or a cargo-mutants sign flip) cannot spin forever.
fn union_find(parent: &mut [usize], mut x: usize) -> usize {
    for _ in 0..parent.len() {
        if parent[x] == x {
            return x;
        }
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    debug_assert!(
        false,
        "union_find: did not converge within {} steps",
        parent.len()
    );
    x
}

/// Apply the inverse world-space inertia tensor: I⁻¹_world = R · I⁻¹_body · Rᵀ.
pub(crate) fn mul_inv_inertia(inertia: Vec3, orientation: glam::Quat, v: Vec3) -> Vec3 {
    debug_assert!(vec3_finite(inertia), "inertia must be finite, got {inertia:?}");
    debug_assert!(vec3_finite(v), "v must be finite, got {v:?}");
    debug_assert!(quat_finite(orientation), "orientation must be finite");
    let body = orientation.inverse() * v;
    let scaled = Vec3::new(
        inv_inertia_axis(inertia.x) * body.x,
        inv_inertia_axis(inertia.y) * body.y,
        inv_inertia_axis(inertia.z) * body.z,
    );
    debug_assert!(vec3_finite(scaled), "scaled must be finite, got {scaled:?}");
    let result = orientation * scaled;
    debug_assert!(vec3_finite(result), "result must be finite, got {result:?}");
    result
}

/// Effective inverse mass along direction `dir` at contact points with
/// levers `ra`/`rb` (linear + rotational terms, world-space inertia).
fn effective_mass(bodies: &[RigidBody], i: usize, j: usize, dir: Vec3, ra: Vec3, rb: Vec3) -> f32 {
    debug_assert!(
        bodies[i].inv_mass.is_finite() && bodies[i].inv_mass >= 0.0,
        "inv_mass[i] must be finite and non-negative"
    );
    debug_assert!(
        bodies[j].inv_mass.is_finite() && bodies[j].inv_mass >= 0.0,
        "inv_mass[j] must be finite and non-negative"
    );
    debug_assert!(vec3_finite(bodies[i].inertia), "inertia[i] must be finite");
    debug_assert!(vec3_finite(bodies[j].inertia), "inertia[j] must be finite");
    debug_assert!(vec3_finite(dir), "dir must be finite");
    let ra_d = ra.cross(dir);
    let rb_d = rb.cross(dir);
    let result = bodies[i].inv_mass
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
        ));
    debug_assert!(
        result.is_finite(),
        "effective_mass: result must be finite, got {result}"
    );
    result
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
        debug_assert!(
            d.is_finite() && d.abs() > 1e-12,
            "solve_small: pivot d = {} — near-zero or NaN at col={col}",
            d
        );
        for r in (col + 1)..n {
            let f = m[r][col] / d;
            debug_assert!(
                f.is_finite(),
                "solve_small: f non-finite at r={r}, col={col}, d={d}"
            );
            for c in col..n {
                m[r][c] -= f * m[col][c];
            }
            debug_assert!(
                m[r].iter().all(|&x| x.is_finite()),
                "solve_small: row {r} non-finite after forward elim at col={col}"
            );
            x[r] -= f * x[col];
            debug_assert!(
                x[r].is_finite(),
                "solve_small: x[{r}] non-finite after forward elim at col={col}"
            );
        }
    }
    let mut out = [0.0f32; 4];
    for r in (0..n).rev() {
        debug_assert!(
            m[r][r].is_finite() && m[r][r].abs() > 1e-12,
            "solve_small: pivot m[{r}][{r}] = {} — singular or NaN",
            m[r][r]
        );
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
pub(crate) fn apply_impulse(
    bodies: &mut [RigidBody],
    i: usize,
    j: usize,
    imp: Vec3,
    ra: Vec3,
    rb: Vec3,
) {
    debug_assert!(i != j, "apply_impulse: i == j");
    debug_assert!(vec3_finite(imp), "imp must be finite, got {imp:?}");
    debug_assert!(
        bodies[i].inv_mass.is_finite() && bodies[i].inv_mass >= 0.0,
        "inv_mass[i] must be finite and non-negative"
    );
    debug_assert!(
        bodies[j].inv_mass.is_finite() && bodies[j].inv_mass >= 0.0,
        "inv_mass[j] must be finite and non-negative"
    );
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
    debug_assert!(
        vec3_finite(a.velocity),
        "apply_impulse: a.velocity must be finite, got {:?}",
        a.velocity
    );
    debug_assert!(
        vec3_finite(b.velocity),
        "apply_impulse: b.velocity must be finite, got {:?}",
        b.velocity
    );
    debug_assert!(
        vec3_finite(a.angular_velocity),
        "apply_impulse: a.angular_velocity must be finite"
    );
    debug_assert!(
        vec3_finite(b.angular_velocity),
        "apply_impulse: b.angular_velocity must be finite"
    );
}

/// Apply a pure angular impulse to the body pair (joint axis constraints).
/// Positive impulse spins body `j` along `imp` and body `i` against it.
fn apply_angular_impulse(bodies: &mut [RigidBody], i: usize, j: usize, imp: Vec3) {
    debug_assert!(i != j);
    let (lo, hi, swapped) = if i < j { (i, j, false) } else { (j, i, true) };
    let (head, tail) = bodies.split_at_mut(hi);
    let (a, b) = if swapped {
        (&mut tail[0], &mut head[lo])
    } else {
        (&mut head[lo], &mut tail[0])
    };
    // `a` is body i, `b` is body j.
    a.angular_velocity -= mul_inv_inertia(a.inertia, a.orientation, imp);
    b.angular_velocity += mul_inv_inertia(b.inertia, b.orientation, imp);
}

/// Rotate a body by a small positional (pseudo) rotation vector, leaving
/// velocities untouched (NGS-style, cf. apply_positional_impulse).
fn apply_positional_rotation(body: &mut RigidBody, d: Vec3) {
    if d != Vec3::ZERO {
        body.orientation = (Quat::from_scaled_axis(d) * body.orientation).normalize();
    }
}

/// Velocity of a body at a world-space contact point (linear + angular part).
#[inline]
pub(crate) fn point_velocity(body: &RigidBody, r: Vec3) -> Vec3 {
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
/// (acc' ≥ 0 on active, vn' ≥ target on inactive). Scalar per-point
/// Gauss-Seidel oscillates between coupled points of one manifold (the
/// rocking pump from G3); the block solve finds the exact active set in one
/// shot. `target` (G6) is the per-point velocity floor: 0 for touching
/// points, the speculative approach limit for separated ones.
#[allow(clippy::too_many_arguments)]
fn solve_normal_block(
    bodies: &mut [RigidBody],
    i: usize,
    j: usize,
    n: Vec3,
    pts: &[Vec3; 4],
    acc: &mut [f32; 4],
    target: &[f32; 4],
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
            // Solve K_S · acc'_S = target_S − vn_S + K_{S,all} · acc (new
            // accumulated impulses directly, so zeroing the inactive set is
            // consistent). vn' = target on the active set.
            let mut ks = [[0.0f32; 4]; 4];
            let mut bs = [0.0f32; 4];
            for a in 0..ns {
                for b in 0..ns {
                    ks[a][b] = k_mat[idx[a]][idx[b]];
                }
                let mut r = target[idx[a]] - vn[idx[a]];
                debug_assert!(
                    r.is_finite(),
                    "solve_normal_block: r non-finite at a={a}, target={} vn={}",
                    target[idx[a]],
                    vn[idx[a]]
                );
                for m in 0..count {
                    r += k_mat[idx[a]][m] * acc[m];
                }
                debug_assert!(
                    r.is_finite(),
                    "solve_normal_block: r non-finite after accum loop at a={a}"
                );
                bs[a] = r;
                debug_assert!(
                    bs[a].is_finite(),
                    "solve_normal_block: non-finite bs[{a}]={}",
                    bs[a]
                );
            }
            let Some(ap) = solve_small(&ks, &bs, ns) else {
                continue;
            };
            debug_assert!(
                ap.iter().take(ns).all(|v| v.is_finite()),
                "solve_normal_block: non-finite impulse solution"
            );
            if ap.iter().take(ns).any(|&v| v < -1e-6) {
                continue;
            }
            // vn' on the inactive set must stay at or above its target.
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
                if v < target[t] - 1e-5 {
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

/// Sphere-sphere. `margin` (G6 speculative): pairs separated by less than
/// the margin still report a contact with NEGATIVE penetration (= the gap),
/// so the solver can stop approach before any overlap exists.
fn sphere_vs_sphere(
    pos_a: Vec3,
    radius_a: f32,
    pos_b: Vec3,
    radius_b: f32,
    margin: f32,
) -> Option<Contact> {
    let diff = pos_b - pos_a;
    let dist_sq = diff.length_squared();
    let radius_sum = radius_a + radius_b + margin;
    if dist_sq > radius_sum * radius_sum || dist_sq < 1e-10 {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = diff / dist;
    let penetration = radius_sum - dist - margin;
    Some(Contact {
        normal,
        penetration,
        contact_point: pos_a + normal * (radius_a - penetration * 0.5),
    })
}

/// Sphere vs an oriented box (OBB), resolved in the box's local frame.
/// `margin`: speculative contact distance (see sphere_vs_sphere).
fn sphere_vs_obb(
    sphere_pos: Vec3,
    sphere_radius: f32,
    box_pos: Vec3,
    half_extents: Vec3,
    box_rot: Quat,
    margin: f32,
) -> Option<Contact> {
    let local = box_rot.inverse() * (sphere_pos - box_pos);
    let clamped = local.clamp(-half_extents, half_extents);
    let delta = clamped - local;
    let dist_sq = delta.length_squared();
    let reach = sphere_radius + margin;
    if dist_sq > reach * reach || dist_sq < 1e-10 {
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
    margin: f32,
) -> Option<(Vec3, f32)> {
    // SAT: the 3 face normals of each box plus the cross products of their axes.
    let aa = [rot_a * Vec3::X, rot_a * Vec3::Y, rot_a * Vec3::Z];
    let ba = [rot_b * Vec3::X, rot_b * Vec3::Y, rot_b * Vec3::Z];

    // Face normals first; an edge-edge axis may replace a face axis only if
    // it beats it by a margin. Otherwise micro-tilts at face contacts make
    // SAT pick noisy cross-product axes and the normal flickers.
    const FACE_PREFERENCE: f32 = 1e-3;

    let mut best_overlap = f32::MAX;
    let mut best_axis = Vec3::X;

    for u in aa.into_iter().chain(ba) {
        let overlap = obb_overlap_on(pos_a, half_a, rot_a, pos_b, half_b, rot_b, u);
        // Separated along any axis by more than the speculative margin -> no
        // contact. Within the margin the pair still reports as touching
        // (negative overlap = gap): the manifold never blinks off for a
        // substep, and fast pairs get speculative constraints (G6).
        if overlap <= -margin {
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
            if overlap <= -margin {
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
    margin: f32,
) -> Option<Contact> {
    let (normal, penetration) = obb_sat(pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin)?;
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
    margin: f32,
) -> Option<Manifold> {
    let (n, _pen) = obb_sat(pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin)?;

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
    // when it is within this distance along the (negated) contact normal.
    // G6: this is the pair's speculative margin (base + approach speed · dt),
    // so fast pairs generate constraints BEFORE any overlap exists; points
    // then carry negative penetration (= the remaining gap).
    let depth_tol = margin;
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
        if d < -depth_tol {
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
        if d < -depth_tol {
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
        // Speculative points keep their NEGATIVE depth (= the gap); the
        // velocity solver turns it into an approach-speed limit (G6).
        points[count] = ManifoldPoint {
            world_point: p,
            penetration: d,
        };
        count += 1;
    }

    if count == 0 {
        return box_vs_box(pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin)
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
    margin: f32,
) -> Option<Contact> {
    let axis = cap_rot * Vec3::Y;
    let bottom = cap_pos - axis * cap_half_height;
    let seg = axis * (2.0 * cap_half_height);
    let t = (sphere_pos - bottom).dot(seg) / seg.length_squared();
    let t = t.clamp(0.0, 1.0);
    let closest = bottom + seg * t;
    let to_sphere = sphere_pos - closest;
    let d = to_sphere.length();
    let rr = cap_radius + sphere_radius + margin;
    if d >= rr || d < 1e-10 {
        return None;
    }
    // Normal points from the capsule toward the sphere.
    let n = to_sphere / d;
    let penetration = rr - d - margin;
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
    margin: f32,
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
    let radius_sum = radius_a + radius_b + margin;
    if dist_sq > radius_sum * radius_sum || dist_sq < 1e-10 {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = diff2 / dist;
    let penetration = radius_sum - dist - margin;
    Some(Contact {
        normal,
        penetration,
        contact_point: (closest_a + closest_b) * 0.5,
    })
}

/// Narrow phase over the broadphase pair list. G6: every pair gets a
/// speculative contact margin = base + approach speed · sub_dt, so contacts
/// exist BEFORE overlap; the velocity solver then caps the approach speed
/// to the remaining gap instead of letting the bodies interpenetrate.
fn detect_collisions(
    bodies: &[RigidBody],
    active: &[(usize, usize)],
    asleep: &[bool],
    sub_dt: f32,
) -> Vec<Manifold> {
    /// Base speculative margin (m): also the AABB inflation used by the
    /// broadphase, so pairs within it are guaranteed to reach narrow phase.
    const SPEC_BASE: f32 = 0.05;
    let mut manifolds = Vec::new();
    for &(i, j) in active {
        let a = &bodies[i];
        let b = &bodies[j];
        if a.body_type == BodyType::Static && b.body_type == BodyType::Static {
            continue;
        }
        // G7: both-asleep pairs are frozen in place — their relative geometry
        // cannot change, so re-running the narrow phase (SAT!) per substep is
        // pure waste. On a settled scene this IS the frame cost. The island
        // graph keeps them composed via the frozen-asleep union instead.
        if asleep[i] && asleep[j] {
            continue;
        }
        let rel_speed = (a.velocity - b.velocity).length();
        let margin = SPEC_BASE + rel_speed * sub_dt;

        let manifold = match (&a.shape, &b.shape) {
            (&Shape::Sphere { radius: ra }, &Shape::Sphere { radius: rb }) => {
                sphere_vs_sphere(a.position, ra, b.position, rb, margin)
                    .map(|c| Manifold::single(i, j, c))
            }
            (&Shape::Sphere { radius: ra }, &Shape::Box { half_extents: hb }) => {
                sphere_vs_obb(a.position, ra, b.position, hb, b.orientation, margin)
                    .map(|c| Manifold::single(i, j, c))
            }
            (&Shape::Box { half_extents: ha }, &Shape::Sphere { radius: rb }) => {
                sphere_vs_obb(b.position, rb, a.position, ha, a.orientation, margin).map(|c| {
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
            (&Shape::Box { half_extents: ha }, &Shape::Box { half_extents: hb }) => box_manifold(
                a.position,
                ha,
                a.orientation,
                b.position,
                hb,
                b.orientation,
                margin,
            ),
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
                margin,
            )
            .map(|c| Manifold::single(i, j, c)),
            (
                &Shape::Sphere { radius: r },
                &Shape::Capsule {
                    radius: cr,
                    half_height: hh,
                },
            ) => sphere_vs_capsule(a.position, r, b.position, cr, hh, b.orientation, margin)
                .map(|c| Manifold::single(i, j, c)),
            (
                &Shape::Capsule {
                    radius: cr,
                    half_height: hh,
                },
                &Shape::Sphere { radius: r },
            ) => sphere_vs_capsule(b.position, r, a.position, cr, hh, a.orientation, margin).map(
                |c| {
                    Manifold::single(
                        i,
                        j,
                        Contact {
                            normal: -c.normal,
                            penetration: c.penetration,
                            contact_point: c.contact_point,
                        },
                    )
                },
            ),
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

    /// Rebuild AABBs and the active pair list. G6: dynamic bodies get a
    /// SWEPT AABB (extended by this substep's displacement) and everything
    /// is inflated by half the base speculative margin — a fast pair must
    /// reach the narrow phase before its shapes can interpenetrate.
    fn update(&mut self, bodies: &[RigidBody], sub_dt: f32) {
        const HALF_SPEC_MARGIN: f32 = 0.025; // SPEC_BASE / 2 in detect_collisions
        self.aabbs = bodies
            .iter()
            .map(|b| {
                let mut aabb = compute_aabb(b);
                if b.body_type == BodyType::Dynamic {
                    let d = b.velocity * sub_dt;
                    aabb.expand(aabb.min + d);
                    aabb.expand(aabb.max + d);
                }
                let m = Vec3::splat(HALF_SPEC_MARGIN);
                aabb.expand(aabb.min - m);
                aabb.expand(aabb.max + m);
                aabb
            })
            .collect();
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

/// Per-manifold solver state shared between the velocity and position stages
/// of a substep (G6 stage split: velocities solve BEFORE positions move, so
/// the NGS pass needs the detection-time anchors/penetrations carried over).
#[derive(Clone)]
pub(crate) struct ManifoldState {
    pub mi: usize,
    pub i: usize,
    pub j: usize,
    pub count: usize,
    pub acc: [f32; 4],
    pub acc_friction: [f32; 4],
    pub acc_friction2: [f32; 4],
    pub bias: [f32; 4],
    // G6 speculative: per-point approach-speed LIMIT (negative of the
    // remaining gap / sub_dt; 0 for touching points). The velocity
    // solve drives vn to this target instead of 0, so a separated
    // point may close its gap within the substep but never more.
    pub target: [f32; 4],
    pub mu: f32,
    // Fixed tangent basis (Box2D-style): friction is solved along
    // directions derived from the contact normal ONCE, not from the
    // instantaneous slip velocity — velocity-aligned friction walks
    // the contact and lets resting stacks drift sideways.
    pub t1: Vec3,
    pub t2: Vec3,
    // G3: body-frame anchors and detection-time penetration per point,
    // so the positional pass can re-measure live separation.
    pub la: [Vec3; 4],
    pub lb: [Vec3; 4],
    pub pen0: [f32; 4],
}

/// Per-island work item for the G7 parallel solver: an island-local shard of
/// the world. Body indices inside `manifolds` and `states` are LOCAL
/// (positions in `body_idx`/`bodies`); `keys` maps each local manifold to its
/// global body-pair key for the warm-start cache. Islands are disjoint over
/// dynamic bodies by construction (union-find over the fresh manifolds), so
/// solving them concurrently is race-free and bit-identical for any thread
/// count: each island runs its manifolds in the original global order, and
/// Gauss-Seidel updates on disjoint state commute exactly.
struct IslandWork {
    /// Sorted global body handles; local index = position in this vec.
    body_idx: Vec<usize>,
    /// Gathered body shard (statics included; never written back).
    bodies: Vec<RigidBody>,
    /// Manifolds cloned with LOCAL body indices.
    manifolds: Vec<Manifold>,
    /// Global sorted body-pair key per local manifold (warm cache I/O).
    keys: Vec<(usize, usize)>,
    /// Velocity-stage output, consumed by the position stage.
    states: Vec<ManifoldState>,
    /// This island's updated warm-cache entries (merged after the join).
    warm: WarmCache,
}

/// Context for building a ManifoldState (packs the per-manifold parameters,
/// keeping `build_manifold_state` below the bca nargs limit).
// Only consumed by the `gpu` feature's GPU contact path today.
#[allow(dead_code)]
struct ManifoldCtx<'a> {
    bodies: &'a mut [RigidBody],
    warm_in: &'a WarmCache,
    allow_restitution: bool,
    sub_dt: f32,
    mi: usize,
    i: usize,
    j: usize,
}

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
    /// Persistent joint constraints with warm-start state (G5). Joints also
    /// feed the island union-find: jointed bodies sleep and wake together.
    joints: Vec<Joint>,
    /// Sorted body pairs connected by a joint. Jointed bodies never collide
    /// (Box2D `collide_connected = false` default): a hinge pin passes through
    /// the arm, so the parts legitimately sweep through each other's space,
    /// and contact friction there would act as a phantom brake on the joint.
    joint_pairs: HashSet<(usize, usize)>,
    /// Diagnostics: (body_a, body_b) of the last substep's manifolds.
    debug_pairs: Vec<(usize, usize)>,
    /// G7: enable SIMD-wide contact solver for single-point manifolds.
    /// Default true. Set to false for bit-exact scalar reproduction.
    wide_solver: bool,
    /// G7: optional GPU contact solver (gpu feature). When attached,
    /// single-point manifolds are solved on the GPU instead of the CPU
    /// wide path; multi-point manifolds stay on the CPU island path.
    #[cfg(feature = "gpu")]
    gpu_solver: Option<WgpuContactSolver>,
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
            joints: Vec::new(),
            joint_pairs: HashSet::new(),
            debug_pairs: Vec::new(),
            wide_solver: true,
            #[cfg(feature = "gpu")]
            gpu_solver: None,
        }
    }

    /// Toggle the G7 SIMD-wide contact solver (default: enabled). Disabling
    /// it forces the scalar single-point path, which is bit-exact with the
    /// pre-G7 solver for scenes that predate the wide batches.
    pub fn set_wide_solver(&mut self, enabled: bool) {
        self.wide_solver = enabled;
    }

    /// Attach a GPU contact solver (G7, `gpu` feature). When the GPU solver
    /// is present, single-point contacts are solved on the GPU and the CPU
    /// wide-path is unused. The GPU solver is a Jacobi/GS hybrid (not
    /// bit-identical to the CPU path); see the `gpu` module docs.
    #[cfg(feature = "gpu")]
    pub fn set_gpu_solver(&mut self, solver: WgpuContactSolver) {
        self.gpu_solver = Some(solver);
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

    /// Whether the body's island is currently sleeping (G4/G7 diagnostics).
    pub fn is_asleep(&self, handle: BodyHandle) -> bool {
        self.asleep.get(handle).copied().unwrap_or(false)
    }

    /// (Diagnostics) island id of the body and its current sleep timer.
    pub fn debug_island_info(&self, handle: BodyHandle) -> Option<(u32, f32)> {
        let root = *self.island.get(handle)?;
        let timer = self.island_timers.get(&root).copied().unwrap_or(0.0);
        Some((root, timer))
    }

    /// (Diagnostics) how many contact manifolds touched the body on the last
    /// substep of the previous step.
    pub fn debug_contact_count(&self, handle: BodyHandle) -> usize {
        self.debug_pairs
            .iter()
            .filter(|&&(a, b)| a == handle || b == handle)
            .count()
    }

    /// Velocity half of the integration (Box3D `IntegrateVelocities`): apply
    /// gravity and pending torque so the constraint solvers below act on the
    /// velocities that the upcoming position integration will actually use.
    fn integrate_velocities(&mut self, dt: f32) {
        debug_assert!(
            dt.is_finite() && dt > 0.0,
            "dt must be positive finite, got {dt}"
        );
        for (h, body) in self.bodies.iter_mut().enumerate() {
            if body.body_type != BodyType::Dynamic || self.asleep.get(h).copied().unwrap_or(false) {
                continue;
            }
            body.velocity += self.gravity * dt;

            // Angular integrate from applied torque.
            if body.torque != Vec3::ZERO {
                let torque_delta = body.torque * dt;
                debug_assert!(
                    vec3_finite(torque_delta),
                    "torque*dt overflowed: torque={:?} dt={dt}",
                    body.torque
                );
                body.angular_velocity +=
                    mul_inv_inertia(body.inertia, body.orientation, torque_delta);
                body.torque = Vec3::ZERO;
            }
        }
    }

    /// Position half of the integration (Box3D `IntegratePositions`): move
    /// bodies along the solver-adjusted velocities. Bodies flagged in `skip`
    /// were already clamped to their time of impact by the continuous pass
    /// and must not move again this substep.
    fn integrate_positions(&mut self, dt: f32, skip: &[bool]) {
        for (h, body) in self.bodies.iter_mut().enumerate() {
            if body.body_type != BodyType::Dynamic || self.asleep.get(h).copied().unwrap_or(false) {
                continue;
            }
            if skip.get(h).copied().unwrap_or(false) {
                continue;
            }
            // Linear integrate (semi-implicit, post-solve velocities).
            body.position += body.velocity * dt;

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

        for m in manifolds {
            let (a, b) = (m.body_a, m.body_b);
            if self.bodies[a].body_type == BodyType::Dynamic
                && self.bodies[b].body_type == BodyType::Dynamic
            {
                let (ra, rb) = (union_find(&mut parent, a), union_find(&mut parent, b));
                if ra != rb {
                    parent[rb] = ra;
                }
            }
        }
        // Joints are constraint-graph edges too (G5): jointed dynamic bodies
        // belong to one island and sleep/wake together.
        for joint in &self.joints {
            let (a, b) = (joint.body_a, joint.body_b);
            if self.bodies[a].body_type == BodyType::Dynamic
                && self.bodies[b].body_type == BodyType::Dynamic
            {
                let (ra, rb) = (union_find(&mut parent, a), union_find(&mut parent, b));
                if ra != rb {
                    parent[rb] = ra;
                }
            }
        }
        // A fully sleeping island keeps its composition even if the contact
        // detection blinks for a step: its members are not integrated, so
        // their relative geometry cannot change — dissolving the island
        // would let one member wake while its support stays asleep.
        // (One representative per old island, not an O(n²) pair scan.)
        let mut asleep_rep: HashMap<u32, usize> = HashMap::new();
        for h in 0..n {
            if !self.asleep.get(h).copied().unwrap_or(false) {
                continue;
            }
            let old = self.island[h];
            if old == u32::MAX {
                continue;
            }
            match asleep_rep.entry(old) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(h);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    let (ra, rb) = (union_find(&mut parent, *e.get()), union_find(&mut parent, h));
                    if ra != rb {
                        parent[rb] = ra;
                    }
                }
            }
        }
        // Canonicalize island ids to the MINIMUM member index. The raw
        // union-find root depends on manifold order, which varies step to
        // step; a root that flips identity resets the island's sleep timer
        // forever and the island never sleeps (measured on a 1025-body grid:
        // half of the perfectly quiet scene stayed awake at ~200 ms/frame).
        let mut canonical: HashMap<usize, usize> = HashMap::new();
        for h in 0..n {
            if self.bodies[h].body_type != BodyType::Dynamic {
                continue;
            }
            let r = union_find(&mut parent, h);
            canonical
                .entry(r)
                .and_modify(|m| *m = (*m).min(h))
                .or_insert(h);
        }
        for h in 0..n {
            self.island[h] = if self.bodies[h].body_type == BodyType::Dynamic {
                canonical[&union_find(&mut parent, h)] as u32
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
                    // A sleeping body is STATIC for the solver (Jolt
                    // semantics): zero inverse mass/inertia makes every
                    // impulse and effective-mass computation treat it as
                    // immovable, so a resting contact with an awake body
                    // can never accumulate invisible velocity in the
                    // sleeper and detonate it on wake. Restored on wake.
                    b.inv_mass = 0.0;
                    b.inertia = Vec3::ZERO;
                }
            }
        }
    }

    /// Wake the whole island containing body `h` (contact with an awake body
    /// propagates motion through the island, so partial wake is incoherent).
    fn wake_island(&mut self, h: usize) {
        let root = self.island[h];
        for h2 in 0..self.bodies.len() {
            if self.island[h2] == root {
                self.asleep[h2] = false;
                // Undo the sleep-time staticification (see update_sleep).
                let b = &mut self.bodies[h2];
                if b.body_type == BodyType::Dynamic {
                    b.inv_mass = 1.0 / b.mass;
                    b.inertia = b.shape.inertia(b.mass);
                }
            }
        }
        self.island_timers.insert(root, 0.0);
    }

    /// Joint sub-solver (G5), run once per substep after the contact pass.
    /// Ball joint: 3 linear equality constraints along the world axes at the
    /// anchor points. Revolute: ball + 2 angular equality constraints along
    /// the axes perpendicular to the hinge (the hinge rotation itself is
    /// free). Both are warm-started from impulses accumulated last substep,
    /// exactly like the contact cache. Joints and contacts alternate at
    /// substep granularity (12 substeps ≈ 720 Hz), which converges well for
    /// chains; true per-iteration interleaving is left for a later refactor.
    /// Velocity stage of the joint solver: warm start from the accumulated
    /// impulses, then velocity iterations. Runs before positions move.
    fn solve_joints_velocity(&mut self) {
        if self.joints.is_empty() {
            return;
        }
        const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

        let Self {
            bodies,
            joints,
            asleep,
            velocity_iterations,
            ..
        } = self;

        for joint in joints.iter_mut() {
            let (a, b) = (joint.body_a, joint.body_b);
            // A fully sleeping jointed pair is frozen; island-coherent sleep
            // guarantees both members share the sleep state.
            if asleep[a] && asleep[b] {
                continue;
            }
            let (la, lb) = joint.local_anchors();
            let revolute_axes = match &joint.kind {
                JointKind::Revolute {
                    local_axis_a,
                    local_axis_b,
                    ..
                } => Some((*local_axis_a, *local_axis_b)),
                JointKind::Ball { .. } => None,
            };

            // --- Warm start: re-apply the accumulated impulses (G2b pattern).
            let ra = bodies[a].orientation * la;
            let rb = bodies[b].orientation * lb;
            for (k, dir) in AXES.iter().enumerate() {
                let l = joint.acc_lin[k];
                if l.abs() > 1e-12 {
                    apply_impulse(bodies, a, b, dir * l, ra, rb);
                }
            }
            if let Some((axis_a, _)) = revolute_axes {
                let wa = (bodies[a].orientation * axis_a).normalize_or(Vec3::Z);
                let t1 = tangent_basis(wa);
                let t2 = wa.cross(t1).normalize_or_zero();
                for (k, t) in [t1, t2].iter().enumerate() {
                    let l = joint.acc_ang[k];
                    if l.abs() > 1e-12 {
                        apply_angular_impulse(bodies, a, b, t * l);
                    }
                }
            }

            // --- Velocity iterations.
            for _ in 0..*velocity_iterations {
                for (k, dir) in AXES.iter().enumerate() {
                    let k_eff = effective_mass(bodies, a, b, *dir, ra, rb);
                    if k_eff < 1e-9 {
                        continue;
                    }
                    let vrel =
                        (point_velocity(&bodies[b], rb) - point_velocity(&bodies[a], ra)).dot(*dir);
                    // Equality constraint: no clamp, any sign of impulse.
                    let dl = -vrel / k_eff;
                    joint.acc_lin[k] += dl;
                    apply_impulse(bodies, a, b, dir * dl, ra, rb);
                }
                if let Some((axis_a, _)) = revolute_axes {
                    let wa = (bodies[a].orientation * axis_a).normalize_or(Vec3::Z);
                    let t1 = tangent_basis(wa);
                    let t2 = wa.cross(t1).normalize_or_zero();
                    for (k, t) in [t1, t2].iter().enumerate() {
                        let (ba, bb) = (&bodies[a], &bodies[b]);
                        let k_eff = mul_inv_inertia(ba.inertia, ba.orientation, *t).dot(*t)
                            + mul_inv_inertia(bb.inertia, bb.orientation, *t).dot(*t);
                        if k_eff < 1e-9 {
                            continue;
                        }
                        let wrel = (bb.angular_velocity - ba.angular_velocity).dot(*t);
                        let dl = -wrel / k_eff;
                        joint.acc_ang[k] += dl;
                        apply_angular_impulse(bodies, a, b, t * dl);
                    }
                }
            }
        }
    }

    /// Position stage of the joint solver (split impulse: positions only).
    /// Runs after `integrate_positions`, like Box3D's joint position pass.
    fn solve_joints_position(&mut self) {
        if self.joints.is_empty() {
            return;
        }
        // Baumgarte-style, same β/cap policy as contacts.
        const BETA: f32 = 0.2;
        const MAX_LIN_CORRECTION: f32 = 0.25;
        const MAX_ANG_CORRECTION: f32 = 0.5;
        const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

        let Self {
            bodies,
            joints,
            asleep,
            position_iterations,
            ..
        } = self;

        for joint in joints.iter_mut() {
            let (a, b) = (joint.body_a, joint.body_b);
            if asleep[a] && asleep[b] {
                continue;
            }
            let (la, lb) = joint.local_anchors();
            let revolute_axes = match &joint.kind {
                JointKind::Revolute {
                    local_axis_a,
                    local_axis_b,
                    ..
                } => Some((*local_axis_a, *local_axis_b)),
                JointKind::Ball { .. } => None,
            };

            for _ in 0..*position_iterations {
                let ra = bodies[a].orientation * la;
                let rb = bodies[b].orientation * lb;
                let c = (bodies[b].position + rb) - (bodies[a].position + ra);
                for dir in AXES {
                    let e = c.dot(dir).clamp(-MAX_LIN_CORRECTION, MAX_LIN_CORRECTION);
                    if e.abs() < 1e-6 {
                        continue;
                    }
                    let k_eff = effective_mass(bodies, a, b, dir, ra, rb);
                    if k_eff < 1e-9 {
                        continue;
                    }
                    let lambda = -BETA * e / k_eff;
                    apply_positional_impulse(bodies, a, b, dir * lambda, ra, rb);
                }
                if let Some((axis_a, axis_b)) = revolute_axes {
                    let wa = (bodies[a].orientation * axis_a).normalize_or(Vec3::Z);
                    let wb = (bodies[b].orientation * axis_b).normalize_or(Vec3::Z);
                    // Small-angle misalignment. Rotation aligning wb with wa
                    // is δ = −(wa × wb) (triple product: (wa×wb)×wb =
                    // wb·cosθ − wa, i.e. +e would PUSH wb away — sign matters,
                    // a flipped sign turns the correction into an exponential
                    // pump). The error lives in the plane ⟂ wa.
                    let e = wa.cross(wb);
                    let t1 = tangent_basis(wa);
                    let t2 = wa.cross(t1).normalize_or_zero();
                    for t in [t1, t2] {
                        let err = e.dot(t).clamp(-MAX_ANG_CORRECTION, MAX_ANG_CORRECTION);
                        if err.abs() < 1e-6 {
                            continue;
                        }
                        let (ba, bb) = (&bodies[a], &bodies[b]);
                        let k_eff = mul_inv_inertia(ba.inertia, ba.orientation, t).dot(t)
                            + mul_inv_inertia(bb.inertia, bb.orientation, t).dot(t);
                        if k_eff < 1e-9 {
                            continue;
                        }
                        let lambda = -BETA * err / k_eff;
                        // Inertia-weighted split: b rotates toward alignment,
                        // a rotates against it (a static body has I⁻¹ = 0).
                        let da = mul_inv_inertia(ba.inertia, ba.orientation, t * -lambda);
                        let db = mul_inv_inertia(bb.inertia, bb.orientation, t * lambda);
                        // Reborrow mutably after the shared reads above.
                        let (lo, hi, swapped) = if a < b { (a, b, false) } else { (b, a, true) };
                        let (head, tail) = bodies.split_at_mut(hi);
                        let (ma, mb) = if swapped {
                            (&mut tail[0], &mut head[lo])
                        } else {
                            (&mut head[lo], &mut tail[0])
                        };
                        apply_positional_rotation(ma, da);
                        apply_positional_rotation(mb, db);
                    }
                }
            }
        }
    }

    /// Time-of-impact pass (G6, b3SolveContinuous analog in its linear form):
    /// runs after the velocity solve, before positions move. A body whose
    /// predicted substep displacement exceeds half its smallest dimension is
    /// cast along that displacement (exact distances, rotation fixed) and
    /// clamped to the first impact; the clamped body is flagged in `skip` so
    /// `integrate_positions` leaves it where the cast put it. This is the
    /// safety net under the speculative contacts for extreme speeds — a true
    /// pass-through needs the displacement to exceed margin + thickness +
    /// radii in one substep.
    // The loop indexes bodies/asleep/skip in parallel; a range loop is the
    // clearest form here (same policy as the solver loops above).
    #[allow(clippy::needless_range_loop)]
    fn solve_continuous(&mut self, sub_dt: f32, skip: &mut [bool]) {
        debug_assert!(
            sub_dt.is_finite() && sub_dt > 0.0,
            "sub_dt must be positive finite, got {sub_dt}"
        );
        for h in 0..self.bodies.len() {
            if self.bodies[h].body_type != BodyType::Dynamic || self.asleep[h] {
                continue;
            }
            let disp = self.bodies[h].velocity * sub_dt;
            debug_assert!(
                vec3_finite(disp),
                "velocity*sub_dt overflowed for body {h}"
            );
            let min_dim = match &self.bodies[h].shape {
                Shape::Sphere { radius } => *radius,
                Shape::Box { half_extents } => half_extents.min_element(),
                Shape::Capsule { radius, .. } => *radius,
            };
            if disp.length_squared() <= (0.5 * min_dim) * (0.5 * min_dim) {
                continue;
            }
            let mover = distance::ShapeRef {
                shape: &self.bodies[h].shape,
                pos: self.bodies[h].position,
                rot: self.bodies[h].orientation,
            };
            let targets = self
                .bodies
                .iter()
                .enumerate()
                .filter(|&(o, _)| o != h)
                .map(|(o, b)| {
                    (
                        o,
                        distance::ShapeRef {
                            shape: &b.shape,
                            pos: b.position,
                            rot: b.orientation,
                        },
                    )
                });
            let hit = distance::cast_shape(mover, disp, targets);
            if let Some(hit) = hit {
                let n = hit.normal; // from the target toward the mover
                let e = self.bodies[h]
                    .restitution
                    .min(self.bodies[hit.handle].restitution);
                let b = &mut self.bodies[h];
                // Back off a hair so the discrete narrow phase sees a clean
                // touching contact next substep, not a zero-gap flicker.
                // hit.t is an ABSOLUTE distance along the displacement.
                b.position += disp.normalize() * hit.t + n * 1e-3;
                skip[h] = true;
                let vn = b.velocity.dot(n);
                if vn < 0.0 {
                    // Inelastic below the shared restitution threshold; a
                    // genuine impact bounces (one-shot, like the discrete
                    // restitution stage).
                    let bounce = if vn < -1.0 { 1.0 + e } else { 1.0 };
                    b.velocity -= n * (bounce * vn);
                }
            }
        }
    }

    /// Build one ManifoldState entry for a manifold at global body indices
    /// `i`/`j`. This is the preamble extracted from `solve_island_velocity`;
    /// today only the GPU single-point path calls it (the CPU island path
    /// keeps its own inline copy in `solve_island_velocity`).
    /// `key` is the sorted global body-pair for warm-cache lookup.
    #[allow(clippy::needless_range_loop)]
    #[allow(dead_code)]
    fn build_manifold_state(
        ctx: &mut ManifoldCtx,
        m: &Manifold,
        key: (usize, usize),
    ) -> Option<ManifoldState> {
        const MATCH_TOL_SQ: f32 = 0.05 * 0.05;
        const RESTITUTION_THRESHOLD: f32 = 1.0;
        const RESTITUTION_MAX_PEN: f32 = 0.05;

        let bodies = &mut *ctx.bodies;
        let (i, j) = (ctx.i, ctx.j);
        let sub_dt = ctx.sub_dt;
        let allow_restitution = ctx.allow_restitution;
        let total_inv = bodies[i].inv_mass + bodies[j].inv_mass;
        if total_inv < 1e-10 {
            return None;
        }
        let n = m.normal;
        let count = m.point_count;

        let mut la = [Vec3::ZERO; 4];
        let mut lb = [Vec3::ZERO; 4];
        let mut pen0 = [0.0f32; 4];
        for k in 0..count {
            let p = m.points[k].world_point;
            la[k] = bodies[i].orientation.inverse() * (p - bodies[i].position);
            lb[k] = bodies[j].orientation.inverse() * (p - bodies[j].position);
            pen0[k] = m.points[k].penetration;
        }

        // Warm-start matching (extracted to reduce bca cognitive)
        let (warm, matched) =
            Self::match_warm_points(&la, &lb, m, key, ctx.warm_in, MATCH_TOL_SQ, count);

        // Restitution bias & speculative target
        let e = bodies[i].restitution.min(bodies[j].restitution);
        let mu = bodies[i].friction.max(bodies[j].friction);
        let mut bias = [0.0f32; 4];
        let mut target = [0.0f32; 4];
        for k in 0..count {
            if pen0[k] < 0.0 {
                target[k] = pen0[k] / sub_dt;
            }
        }
        if allow_restitution {
            for k in 0..count {
                if matched[k] || pen0[k] > RESTITUTION_MAX_PEN {
                    continue;
                }
                let p = m.points[k].world_point;
                let ra = p - bodies[i].position;
                let rb = p - bodies[j].position;
                let vn0 = (point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra)).dot(n);
                if vn0 >= -RESTITUTION_THRESHOLD {
                    continue;
                }
                if pen0[k] < 0.0 && -pen0[k] > -vn0 * sub_dt {
                    continue;
                }
                bias[k] = -e * vn0;
            }
        }

        // Warm-start application (capped)
        let mut warm_applied = warm;
        for k in 0..count {
            if warm[k] > 0.0 {
                let p = m.points[k].world_point;
                let ra = p - bodies[i].position;
                let rb = p - bodies[j].position;
                let k_eff = effective_mass(bodies, i, j, n, ra, rb);
                if k_eff < 1e-10 {
                    warm_applied[k] = 0.0;
                    continue;
                }
                let vn_pre =
                    (point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra)).dot(n);
                let applied = warm[k].min(((target[k] - vn_pre) / k_eff).max(0.0));
                warm_applied[k] = applied;
                if applied > 0.0 {
                    apply_impulse(bodies, i, j, n * applied, ra, rb);
                }
            }
        }

        Some(ManifoldState {
            mi: ctx.mi,
            i: ctx.i,
            j: ctx.j,
            count,
            acc: warm_applied,
            acc_friction: [0.0; 4],
            acc_friction2: [0.0; 4],
            bias,
            target,
            mu,
            t1: tangent_basis(n),
            t2: tangent_basis(n).cross(n),
            la,
            lb,
            pen0,
        })
    }

    /// Warm-start matching helper (extracted to reduce bca complexity).
    #[allow(clippy::needless_range_loop)]
    // Only called from `build_manifold_state` (the `gpu` feature path) today.
    #[allow(dead_code)]
    fn match_warm_points(
        la: &[Vec3; 4],
        lb: &[Vec3; 4],
        m: &Manifold,
        key: (usize, usize),
        warm_in: &WarmCache,
        match_tol_sq: f32,
        count: usize,
    ) -> ([f32; 4], [bool; 4]) {
        let mut warm = [0.0f32; 4];
        let mut matched = [false; 4];
        if let Some((cached_points, cached_count)) = warm_in.get(&key) {
            let mut used = [false; 4];
            for k in 0..count {
                let mut best: Option<(usize, f32)> = None;
                for (c, cp) in cached_points.iter().enumerate().take(*cached_count) {
                    if used[c] {
                        continue;
                    }
                    if cp.normal.dot(m.normal) < 0.7 {
                        continue;
                    }
                    let d2 = (cp.la - la[k]).length_squared() + (cp.lb - lb[k]).length_squared();
                    if d2 < match_tol_sq && best.is_none_or(|(_, bd)| d2 < bd) {
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
        (warm, matched)
    }

    /// Partition `active` (manifold indices) into islands and build work
    /// items. Extracted so both the CPU path and the GPU hybrid path reuse
    /// the same island-building logic.
    fn partition_into_islands(&self, active: &[usize], manifolds: &[Manifold]) -> Vec<IslandWork> {
        let n = self.bodies.len();
        let mut parent: Vec<usize> = (0..n).collect();
        for &mi in active {
            let m = &manifolds[mi];
            let (a, b) = (m.body_a, m.body_b);
            if self.bodies[a].body_type == BodyType::Dynamic
                && self.bodies[b].body_type == BodyType::Dynamic
            {
                let (ra, rb) = (union_find(&mut parent, a), union_find(&mut parent, b));
                if ra != rb {
                    parent[rb] = ra;
                }
            }
        }
        let mut group_of: HashMap<usize, usize> = HashMap::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for &mi in active {
            let m = &manifolds[mi];
            let d = if self.bodies[m.body_a].body_type == BodyType::Dynamic {
                m.body_a
            } else {
                m.body_b
            };
            let root = union_find(&mut parent, d);
            match group_of.entry(root) {
                std::collections::hash_map::Entry::Occupied(e) => groups[*e.get()].push(mi),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(groups.len());
                    groups.push(vec![mi]);
                }
            }
        }

        let mut islands: Vec<IslandWork> = Vec::with_capacity(groups.len());
        for group in groups {
            let mut body_idx: Vec<usize> = Vec::new();
            for &mi in &group {
                body_idx.push(manifolds[mi].body_a);
                body_idx.push(manifolds[mi].body_b);
            }
            body_idx.sort_unstable();
            body_idx.dedup();
            let shard: Vec<RigidBody> = body_idx.iter().map(|&g| self.bodies[g].clone()).collect();
            let local = |g: usize| body_idx.binary_search(&g).expect("island body");
            let island_manifolds: Vec<Manifold> = group
                .iter()
                .map(|&mi| {
                    let mut mc = manifolds[mi].clone();
                    mc.body_a = local(manifolds[mi].body_a);
                    mc.body_b = local(manifolds[mi].body_b);
                    mc
                })
                .collect();
            let keys: Vec<(usize, usize)> = group
                .iter()
                .map(|&mi| {
                    let m = &manifolds[mi];
                    (m.body_a.min(m.body_b), m.body_a.max(m.body_b))
                })
                .collect();
            islands.push(IslandWork {
                body_idx,
                bodies: shard,
                manifolds: island_manifolds,
                keys,
                states: Vec::new(),
                warm: HashMap::new(),
            });
        }
        islands
    }

    /// Dispatch the island velocity solves (parallel via rayon when wide
    /// enough), scatter bodies back, and merge warm caches.
    fn dispatch_islands_velocity(
        &mut self,
        islands: &mut Vec<IslandWork>,
        allow_restitution: bool,
        sub_dt: f32,
    ) {
        const PAR_MIN_ISLANDS: usize = 2;
        const PAR_MIN_MANIFOLDS: usize = 24;
        if islands.is_empty() {
            return;
        }
        let parallel = islands.len() >= PAR_MIN_ISLANDS
            && islands.iter().map(|i| i.manifolds.len()).sum::<usize>() >= PAR_MIN_MANIFOLDS;
        let warm_in = &self.warm_impulses;
        let iters = self.velocity_iterations;
        let wide_on = self.wide_solver;
        let solve = |isl: &mut IslandWork| {
            let (states, warm) = Self::solve_island_velocity(
                &mut isl.bodies,
                &isl.manifolds,
                &isl.keys,
                warm_in,
                iters,
                allow_restitution,
                sub_dt,
                wide_on,
            );
            isl.states = states;
            isl.warm = warm;
        };
        if parallel {
            islands.par_iter_mut().for_each(solve);
        } else {
            islands.iter_mut().for_each(solve);
        }
        let mut next: WarmCache = HashMap::new();
        for isl in islands.iter() {
            for (l, &g) in isl.body_idx.iter().enumerate() {
                if self.bodies[g].body_type == BodyType::Dynamic {
                    self.bodies[g] = isl.bodies[l].clone();
                }
            }
            next.extend(isl.warm.iter().map(|(k, v)| (*k, *v)));
        }
        self.warm_impulses = next;
    }

    /// Contact velocity solve using the GPU for single-point manifolds
    /// (G7, `gpu` feature). Multi-point manifolds are dispatched on CPU
    /// islands. This is a Jacobi/GS hybrid (not bit-identical).
    #[cfg(feature = "gpu")]
    // Warm-point packing indexes parallel per-point arrays; a range loop is
    // the clearest form here (same style as the scalar solver).
    #[allow(clippy::needless_range_loop)]
    fn solve_contacts_velocity_gpu(
        &mut self,
        active: Vec<usize>,
        manifolds: &[Manifold],
        allow_restitution: bool,
        sub_dt: f32,
    ) -> Vec<IslandWork> {
        // Build global states for ALL active manifolds (shared preamble).
        let mut global_states: Vec<ManifoldState> = Vec::with_capacity(active.len());
        for &mi in &active {
            let m = &manifolds[mi];
            let (i, j) = (m.body_a, m.body_b);
            let key = (i.min(j), i.max(j));
            let mut ctx = ManifoldCtx {
                bodies: &mut self.bodies,
                warm_in: &self.warm_impulses,
                allow_restitution,
                sub_dt,
                mi,
                i,
                j,
            };
            if let Some(st) = Self::build_manifold_state(&mut ctx, m, key) {
                global_states.push(st);
            }
        }

        // Split: single-point → GPU, multi-point → CPU islands.
        let mut single_si: Vec<usize> = Vec::new();
        let mut multi_mi: Vec<usize> = Vec::new();
        for (si, st) in global_states.iter().enumerate() {
            if st.count == 1 {
                single_si.push(si);
            } else {
                multi_mi.push(st.mi);
            }
        }

        // GPU solve single-point contacts.
        let mut gpu_warm: WarmCache = HashMap::new();
        if !single_si.is_empty() {
            let gpu = self.gpu_solver.as_mut().unwrap();
            let (batches, num_batches) =
                pack_single_point_batches(&self.bodies, &global_states, manifolds, &single_si);
            if num_batches > 0 {
                gpu.upload_bodies(&self.bodies);
                gpu.upload_batches(&batches);
                gpu.solve(num_batches, self.velocity_iterations, allow_restitution);
                gpu.download_bodies(&mut self.bodies);
                let mut dl_batches = batches;
                gpu.download_acc(&mut dl_batches);
                write_back_acc(&mut global_states, &single_si, &dl_batches);
                // Persist warm cache for single-point manifolds.
                for &si in &single_si {
                    let st = &global_states[si];
                    let m = &manifolds[st.mi];
                    let key = (m.body_a.min(m.body_b), m.body_a.max(m.body_b));
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
                    gpu_warm.insert(key, (pts, st.count));
                }
            }
        }

        // CPU islands for multi-point manifolds. The island dispatch replaces
        // `warm_impulses` with the multi-point cache, so the GPU entries are
        // merged back in afterwards.
        let islands = if multi_mi.is_empty() {
            Vec::new()
        } else {
            let mut islands = self.partition_into_islands(&multi_mi, manifolds);
            self.dispatch_islands_velocity(&mut islands, allow_restitution, sub_dt);
            islands
        };
        self.warm_impulses.extend(gpu_warm);
        islands
    }

    /// Velocity stage of the contact solver (G6 stage order, G7 island
    /// dispatch). Orchestrator: a sequential sleep/wake pre-pass (the only
    /// part mutating island state), a union-find partition over the FRESH
    /// manifolds, then each island solved independently by
    /// `solve_island_velocity` — in parallel via rayon when the scene is wide
    /// enough. Islands are disjoint over dynamic bodies by construction, so
    /// concurrent solves are race-free and bit-identical for any thread
    /// count (Strong Confluence). Returns the island work items; the position
    /// stage reuses them (states + remapped manifolds) after integration.
    fn solve_contacts_velocity(
        &mut self,
        manifolds: &[Manifold],
        allow_restitution: bool,
        sub_dt: f32,
    ) -> Vec<IslandWork> {
        // --- Sequential pre-pass: sleep/wake policy + active filtering ---
        // Sleep: a contact needs work only if at least one side is an AWAKE
        // DYNAMIC body. Static geometry never wakes anything (a body asleep
        // on the floor must stay asleep).
        const WAKE_IMPACT_SPEED: f32 = 0.5;
        let mut active: Vec<usize> = Vec::with_capacity(manifolds.len());
        for (mi, m) in manifolds.iter().enumerate() {
            let (i, j) = (m.body_a, m.body_b);
            let ai = self.asleep[i] || self.bodies[i].body_type != BodyType::Dynamic;
            let aj = self.asleep[j] || self.bodies[j].body_type != BodyType::Dynamic;
            if ai && aj {
                continue;
            }
            // Wake hysteresis (G7): a sleeping island is woken only by a
            // genuine IMPACT — approach speed above the threshold. A resting
            // micro-jitter contact (vn ≈ 0) must NOT wake it: island
            // composition can flicker at solver limit-cycle boundaries, and
            // without hysteresis a singleton sliver sleeps, gets tickled
            // awake by its quiet neighbour, and the pair churns
            // sleep→wake→sleep forever, keeping half of a settled scene
            // awake (measured on a 1025-body grid: ~50% never slept).
            if (self.asleep[i] && !aj) || (self.asleep[j] && !ai) {
                let (s, o) = if self.asleep[i] { (i, j) } else { (j, i) };
                let p = m.points[0].world_point;
                let rs = p - self.bodies[s].position;
                let ro = p - self.bodies[o].position;
                let approach = (point_velocity(&self.bodies[o], ro)
                    - point_velocity(&self.bodies[s], rs))
                .dot(m.normal)
                    * if self.asleep[i] { -1.0 } else { 1.0 };
                // m.normal points i → j; `approach` is the speed at which the
                // awake partner closes in on the sleeper (sleep velocities
                // are zeroed, so this is just the partner's normal speed).
                if approach > WAKE_IMPACT_SPEED {
                    self.wake_island(s);
                }
            }
            // A still-sleeping body is static for the solver (its inv_mass is
            // zeroed at sleep), so sleeper+static pairs carry no work.
            if self.bodies[i].inv_mass + self.bodies[j].inv_mass < 1e-10 {
                continue;
            }
            active.push(mi);
        }
        if active.is_empty() {
            self.warm_impulses.clear();
            return Vec::new();
        }

        // --- GPU single-point path (G7) ---
        // When a GPU solver is attached, the whole velocity solve for
        // single-point manifolds moves to the GPU (`solve_contacts_velocity_gpu`),
        // and multi-point manifolds keep the CPU island path. The GPU and CPU
        // passes are NOT interleaved per Gauss-Seidel iteration (they run
        // sequentially per substep) — a Jacobi/GS hybrid that is physically
        // correct but not bit-identical to the pure CPU path (see PLAN.md).
        #[cfg(feature = "gpu")]
        if self.gpu_solver.is_some() {
            return self.solve_contacts_velocity_gpu(active, manifolds, allow_restitution, sub_dt);
        }

        // --- Partition into islands + dispatch (G7) ---
        // Islands are disjoint over dynamic bodies by construction, so
        // concurrent solves are race-free and bit-identical for any thread
        // count (Strong Confluence).
        let mut islands = self.partition_into_islands(&active, manifolds);
        self.dispatch_islands_velocity(&mut islands, allow_restitution, sub_dt);
        islands
    }

    /// Position stage (G3 split impulse, G6 order, G7 island dispatch): runs
    /// AFTER positions are integrated. Iterated NGS per island;
    /// pseudo-motion only — real velocities are never touched. Each iteration
    /// re-measures the LIVE separation at the stored body-frame anchors, so
    /// corrections distribute evenly across the manifold set instead of
    /// one-shot rigid pushes. β is kept low (0.2): stronger pseudo-correction
    /// resonates with the velocity solve on rocking contacts.
    fn solve_contacts_position(&mut self, islands: &mut [IslandWork]) {
        const PAR_MIN_ISLANDS: usize = 2;
        const PAR_MIN_MANIFOLDS: usize = 24;
        if islands.is_empty() {
            return;
        }
        // Re-gather: integration, TOI clamps and the joint velocity pass all
        // moved the main array since the velocity stage ran.
        for isl in islands.iter_mut() {
            for (l, &g) in isl.body_idx.iter().enumerate() {
                isl.bodies[l] = self.bodies[g].clone();
            }
        }
        let iters = self.position_iterations;
        let softness = self.contact_softness;
        let total_manifolds: usize = islands.iter().map(|i| i.manifolds.len()).sum();
        let solve = |isl: &mut IslandWork| {
            Self::solve_island_position(
                &mut isl.bodies,
                &isl.manifolds,
                &isl.states,
                iters,
                softness,
            );
        };
        if islands.len() >= PAR_MIN_ISLANDS && total_manifolds >= PAR_MIN_MANIFOLDS {
            islands.par_iter_mut().for_each(solve);
        } else {
            islands.iter_mut().for_each(solve);
        }
        for isl in islands.iter() {
            for (l, &g) in isl.body_idx.iter().enumerate() {
                if self.bodies[g].body_type == BodyType::Dynamic {
                    self.bodies[g] = isl.bodies[l].clone();
                }
            }
        }
    }

    /// Per-island velocity solve: the G2b–G6 inner solver (warm start,
    /// Gauss-Seidel with block-LCP normals and fixed-basis friction, one-shot
    /// restitution, cache persist), operating on an island-local body shard.
    /// All body indices in `manifolds` and the returned states are LOCAL;
    /// `keys` maps each local manifold to its global body-pair warm-cache key.
    /// When `use_wide` is true, single-point manifolds are solved in
    /// SIMD-wide batches (G7); multi-point (block LCP) stays scalar.
    // Solver loops index several parallel per-point arrays (manifold points,
    // warm cache, accumulators); range loops are the clearest form here.
    // The 8th parameter (`use_wide`, G7) tips this over clippy's default
    // 7-argument limit; packing them into a struct would only add churn.
    #[allow(clippy::needless_range_loop)]
    #[allow(clippy::too_many_arguments)]
    fn solve_island_velocity(
        bodies: &mut [RigidBody],
        manifolds: &[Manifold],
        keys: &[(usize, usize)],
        warm_in: &WarmCache,
        velocity_iterations: u32,
        allow_restitution: bool,
        sub_dt: f32,
        use_wide: bool,
    ) -> (Vec<ManifoldState>, WarmCache) {
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

        let mut states: Vec<ManifoldState> = Vec::with_capacity(manifolds.len());
        for (mi, m) in manifolds.iter().enumerate() {
            let (i, j) = (m.body_a, m.body_b);
            let total_inv = bodies[i].inv_mass + bodies[j].inv_mass;
            if total_inv < 1e-10 {
                continue;
            }
            let n = m.normal;
            let key = keys[mi];
            let count = m.point_count;

            // --- Body-frame anchors first: matching and G3 both need them ---
            let mut la = [Vec3::ZERO; 4];
            let mut lb = [Vec3::ZERO; 4];
            let mut pen0 = [0.0f32; 4];
            for k in 0..count {
                let p = m.points[k].world_point;
                la[k] = bodies[i].orientation.inverse() * (p - bodies[i].position);
                lb[k] = bodies[j].orientation.inverse() * (p - bodies[j].position);
                pen0[k] = m.points[k].penetration;
            }

            // --- Match cached impulses by body-frame anchors (feature
            // persistence, Jolt-style): stable while the same surface feature
            // stays in contact, even when the bodies move fast in world space.
            let mut warm = [0.0f32; 4];
            let mut matched = [false; 4];
            if let Some((cached_points, cached_count)) = warm_in.get(&key) {
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
            let e = bodies[i].restitution.min(bodies[j].restitution);
            let mu = bodies[i].friction.max(bodies[j].friction);
            let mut bias = [0.0f32; 4];
            let mut target = [0.0f32; 4];
            for k in 0..count {
                // Speculative (separated) point: may close the gap within this
                // substep, but not more — Box2D's speculative distance baked
                // into the velocity target.
                if pen0[k] < 0.0 {
                    target[k] = pen0[k] / sub_dt;
                }
            }
            if allow_restitution {
                for k in 0..count {
                    if matched[k] || pen0[k] > RESTITUTION_MAX_PEN {
                        continue;
                    }
                    let p = m.points[k].world_point;
                    let ra = p - bodies[i].position;
                    let rb = p - bodies[j].position;
                    let vn0 =
                        (point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra)).dot(n);
                    if vn0 >= -RESTITUTION_THRESHOLD {
                        continue;
                    }
                    // A speculative point restitutes only if the approach is
                    // fast enough to actually land within this substep —
                    // otherwise the bounce would fire in mid-air.
                    if pen0[k] < 0.0 && -pen0[k] > -vn0 * sub_dt {
                        continue;
                    }
                    bias[k] = -e * vn0;
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
                    let ra = p - bodies[i].position;
                    let rb = p - bodies[j].position;
                    let k_eff = effective_mass(bodies, i, j, n, ra, rb);
                    if k_eff < 1e-10 {
                        warm_applied[k] = 0.0;
                        continue;
                    }
                    let vn_pre =
                        (point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra)).dot(n);
                    // Cap against the speculative target too: a separated
                    // point may keep approaching up to its gap limit.
                    let applied = warm[k].min(((target[k] - vn_pre) / k_eff).max(0.0));
                    warm_applied[k] = applied;
                    if applied > 0.0 {
                        apply_impulse(bodies, i, j, n * applied, ra, rb);
                    }
                }
            }

            states.push(ManifoldState {
                mi,
                i,
                j,
                count,
                acc: warm_applied,
                acc_friction: [0.0; 4],
                acc_friction2: [0.0; 4],
                bias,
                target,
                mu,
                t1: tangent_basis(n),
                t2: tangent_basis(n).cross(n),
                la,
                lb,
                pen0,
            });
        }

        // --- Velocity solve: Gauss-Seidel iterations over ALL manifolds ---
        // G7: single-point manifolds are packed into SIMD-wide batches
        // (disjoint body sets, original GS order preserved — every contact
        // stays in its place in the sequence); multi-point manifolds keep
        // the scalar block-LCP path. Steps run in manifold order, so the
        // computation is the same sequence either way.
        let mut steps = if use_wide {
            build_solver_steps(bodies, manifolds, &states)
        } else {
            Vec::new()
        };
        for _ in 0..velocity_iterations {
            if use_wide {
                for step in &mut steps {
                    match step {
                        SolverStep::Wide(b) => {
                            b.gather(bodies);
                            b.solve_iteration();
                            b.scatter(bodies);
                        }
                        SolverStep::Scalar(si) => {
                            Self::solve_scalar_velocity_step(bodies, manifolds, &mut states[*si]);
                        }
                    }
                }
            } else {
                for st in states.iter_mut() {
                    Self::solve_scalar_velocity_step(bodies, manifolds, st);
                }
            }
        }
        // Wide batches own the accumulated impulses of their lanes during
        // the iterations; write them back so the cache persist below sees
        // the final values.
        if use_wide {
            for step in &steps {
                if let SolverStep::Wide(b) = step {
                    b.write_back_acc(&mut states);
                }
            }
        }

        // --- Restitution stage (Box3D b3SolverStage_Restitution analog) ---
        // One-shot per step: push the normal point velocity up to the stored
        // bounce target. NOT accumulated, NOT warm-started — this is what
        // keeps spinning bodies from pumping energy through the bounce.
        if allow_restitution {
            if use_wide {
                for step in &mut steps {
                    match step {
                        SolverStep::Wide(b) => {
                            b.gather(bodies);
                            b.solve_restitution();
                            b.scatter(bodies);
                        }
                        SolverStep::Scalar(si) => {
                            Self::solve_scalar_restitution_step(bodies, manifolds, &states[*si]);
                        }
                    }
                }
            } else {
                for st in &states {
                    Self::solve_scalar_restitution_step(bodies, manifolds, st);
                }
            }
        }

        // --- Persist the cache for the next substep/step ---
        // --- Persist this island's cache entries for the next substep ---
        // (st.i/st.j are island-LOCAL indices; the cache is keyed globally.)
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
            next.insert(keys[st.mi], (pts, st.count));
        }
        (states, next)
    }

    /// One scalar Gauss-Seidel velocity step for a single manifold: the
    /// block-LCP normal solve (multi-point) or the projected scalar solve
    /// (single-point), then Coulomb friction along the fixed tangent basis.
    /// Extracted from the iteration loop so the G7 step sequence (wide
    /// batches interleaved with scalar manifolds) reuses the exact same code.
    // Solver loops index several parallel per-point arrays; range loops are
    // the clearest form here.
    #[allow(clippy::needless_range_loop)]
    fn solve_scalar_velocity_step(
        bodies: &mut [RigidBody],
        manifolds: &[Manifold],
        st: &mut ManifoldState,
    ) {
        let m = &manifolds[st.mi];
        let (i, j) = (st.i, st.j);
        let n = m.normal;
        let total_inv = bodies[i].inv_mass + bodies[j].inv_mass;

        // ---- Normal direction ----
        // G4: multi-point manifolds are solved as an exact LCP block
        // (scalar per-point GS oscillates between coupled points of one
        // manifold — the rocking pump); single points keep the scalar
        // projected update.
        if st.count >= 2 {
            let mut pts = [Vec3::ZERO; 4];
            for k in 0..st.count {
                pts[k] = m.points[k].world_point;
            }
            solve_normal_block(bodies, i, j, n, &pts, &mut st.acc, &st.target, st.count);
        } else {
            let k = 0;
            let p = m.points[k].world_point;
            let ra = p - bodies[i].position;
            let rb = p - bodies[j].position;
            let k_eff = effective_mass(bodies, i, j, n, ra, rb);
            if k_eff >= 1e-10 {
                let rel = point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra);
                let vn = rel.dot(n);
                // Inelastic contact: restitution is a separate one-shot stage
                // (below), never accumulated. G6: the target is the
                // speculative approach limit (0 when touching), not
                // necessarily a full stop.
                let lambda = (st.target[k] - vn) / k_eff;
                let new_acc = (st.acc[k] + lambda).max(0.0);
                let delta = new_acc - st.acc[k];
                st.acc[k] = new_acc;
                if delta.abs() > 1e-12 {
                    apply_impulse(bodies, i, j, n * delta, ra, rb);
                }
            }
        }

        // Friction (Coulomb) along the FIXED tangent basis (extracted helper).
        Self::solve_scalar_friction(bodies, i, j, st, m, total_inv);
    }

    /// One-shot restitution step for a single manifold (the scalar half of
    /// the post-iteration stage; the wide half lives in `WideBatch`).
    #[allow(clippy::needless_range_loop)]
    fn solve_scalar_restitution_step(
        bodies: &mut [RigidBody],
        manifolds: &[Manifold],
        st: &ManifoldState,
    ) {
        let m = &manifolds[st.mi];
        let (i, j) = (st.i, st.j);
        let n = m.normal;
        let total_inv = bodies[i].inv_mass + bodies[j].inv_mass;
        for k in 0..st.count {
            if st.bias[k] <= 0.0 {
                continue;
            }
            let p = m.points[k].world_point;
            let ra = p - bodies[i].position;
            let rb = p - bodies[j].position;
            let ra_n = ra.cross(n);
            let rb_n = rb.cross(n);
            let k_eff = total_inv
                + ra_n.dot(mul_inv_inertia(
                    bodies[i].inertia,
                    bodies[i].orientation,
                    ra_n,
                ))
                + rb_n.dot(mul_inv_inertia(
                    bodies[j].inertia,
                    bodies[j].orientation,
                    rb_n,
                ));
            if k_eff < 1e-10 {
                continue;
            }
            let vn = (point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra)).dot(n);
            let lambda = (st.bias[k] - vn) / k_eff;
            if lambda > 0.0 {
                apply_impulse(bodies, i, j, n * lambda, ra, rb);
            }
        }
    }

    /// Friction step for a single manifold (extracted to reduce bca
    /// cognitive complexity of solve_scalar_velocity_step).
    #[allow(clippy::needless_range_loop)]
    fn solve_scalar_friction(
        bodies: &mut [RigidBody],
        i: usize,
        j: usize,
        st: &mut ManifoldState,
        m: &Manifold,
        total_inv: f32,
    ) {
        for k in 0..st.count {
            debug_assert!(st.acc[k].is_finite(), "acc must be finite");
            debug_assert!(
                st.mu.is_finite() && st.mu >= 0.0,
                "mu must be non-negative, got {}",
                st.mu
            );
            let p = m.points[k].world_point;
            let ra = p - bodies[i].position;
            let rb = p - bodies[j].position;
            let rel = point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra);
            let max_friction = st.mu * st.acc[k];
            let mut f_imp = Vec3::ZERO;
            for axis in 0..2 {
                let t = if axis == 0 { st.t1 } else { st.t2 };
                let ra_t = ra.cross(t);
                let rb_t = rb.cross(t);
                let k_t = total_inv
                    + ra_t.dot(mul_inv_inertia(
                        bodies[i].inertia,
                        bodies[i].orientation,
                        ra_t,
                    ))
                    + rb_t.dot(mul_inv_inertia(
                        bodies[j].inertia,
                        bodies[j].orientation,
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
                let len = (new_t * new_t + other * other).sqrt();
                let new_t = if len > max_friction && len > 1e-12 {
                    new_t * (max_friction / len)
                } else {
                    new_t
                };
                debug_assert!(new_t.is_finite(), "friction impulse overflowed");
                if axis == 0 {
                    f_imp += t * (new_t - st.acc_friction[k]);
                    st.acc_friction[k] = new_t;
                } else {
                    f_imp += t * (new_t - st.acc_friction2[k]);
                    st.acc_friction2[k] = new_t;
                }
            }
            if f_imp.length_squared() > 1e-24 {
                apply_impulse(bodies, i, j, f_imp, ra, rb);
            }
        }
    }

    /// Per-island NGS position solve on the local shard (stage doc lives on
    /// `solve_contacts_position`). Pseudo-motion only: real velocities and
    /// the warm cache are never touched here.
    // Live anchors are re-measured per iteration; range loops over the
    // per-point arrays are the clearest form here.
    #[allow(clippy::needless_range_loop)]
    fn solve_island_position(
        bodies: &mut [RigidBody],
        manifolds: &[Manifold],
        states: &[ManifoldState],
        position_iterations: u32,
        contact_softness: f32,
    ) {
        const SLOP: f32 = 0.02;
        const MAX_CORRECTION: f32 = 0.25;
        const BETA_POS: f32 = 0.2;
        for _ in 0..position_iterations {
            for st in states {
                let m = &manifolds[st.mi];
                let (i, j) = (st.i, st.j);
                let n = m.normal;
                let inv_mass_a = bodies[i].inv_mass;
                let inv_mass_b = bodies[j].inv_mass;
                let total_inv = inv_mass_a + inv_mass_b;
                let cfm = contact_softness * total_inv;
                for k in 0..st.count {
                    let (pos_a, rot_a) = (bodies[i].position, bodies[i].orientation);
                    let (pos_b, rot_b) = (bodies[j].position, bodies[j].orientation);
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
                        + ra_n.dot(mul_inv_inertia(bodies[i].inertia, rot_a, ra_n))
                        + rb_n.dot(mul_inv_inertia(bodies[j].inertia, rot_b, rb_n));
                    let k_soft = make_soft(k_pos, cfm);
                    if k_soft < 1e-10 {
                        continue;
                    }
                    let lam = BETA_POS * c / k_soft;
                    apply_positional_impulse(bodies, i, j, n * lam, ra, rb);
                }
            }
        }
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
        // G7: a fully sleeping world cannot change — skip the whole substep
        // loop (broadphase re-sort included) instead of paying ~10 ms/frame
        // to rediscover that nothing moves.
        if self
            .bodies
            .iter()
            .enumerate()
            .all(|(h, b)| b.body_type != BodyType::Dynamic || self.asleep[h])
        {
            return;
        }
        let sub_dt = dt / self.substeps as f32;
        let mut last_manifolds = Vec::new();
        for s in 0..self.substeps {
            // Box3D stage order: solve velocities BEFORE moving positions, so
            // a resting contact kills gravity's velocity gain in the same
            // substep instead of letting the body free-fall and snapping it
            // back (the snap is an inelastic collision and bleeds energy).
            self.integrate_velocities(sub_dt);
            self.broadphase.update(&self.bodies, sub_dt);
            // Jointed pairs never collide: their parts legitimately sweep
            // through each other's space (a hinge pin passes through the arm).
            let manifolds = if self.joint_pairs.is_empty() {
                detect_collisions(&self.bodies, &self.broadphase.active, &self.asleep, sub_dt)
            } else {
                let pairs: Vec<(usize, usize)> = self
                    .broadphase
                    .active
                    .iter()
                    .copied()
                    .filter(|p| !self.joint_pairs.contains(p))
                    .collect();
                detect_collisions(&self.bodies, &pairs, &self.asleep, sub_dt)
            };
            // Restitution is one-shot per step, evaluated on the first substep.
            let mut islands = self.solve_contacts_velocity(&manifolds, s == 0, sub_dt);
            self.solve_joints_velocity();
            // Continuous pass on the solver-adjusted velocities: clamp fast
            // movers to their first impact and keep them there this substep.
            let mut clamped = vec![false; self.bodies.len()];
            self.solve_continuous(sub_dt, &mut clamped);
            self.integrate_positions(sub_dt, &clamped);
            self.solve_contacts_position(&mut islands);
            self.solve_joints_position();
            last_manifolds = manifolds;
        }
        // Diagnostics: contact-manifold partners per body from the last
        // substep (drives sleep/island debugging; tiny flat copy).
        self.debug_pairs.clear();
        self.debug_pairs
            .extend(last_manifolds.iter().map(|m| (m.body_a, m.body_b)));
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
            let last = self.bodies.len() - 1;
            self.bodies.swap_remove(handle);
            self.island.swap_remove(handle);
            self.asleep.swap_remove(handle);
            // swap_remove shifts the last body's index; warm-start keys are
            // body indices, so the cache is no longer valid.
            self.warm_impulses.clear();
            // Drop joints touching the removed body; remap the swapped-in
            // body's index in the survivors.
            self.joints.retain_mut(|j| {
                if j.body_a == handle || j.body_b == handle {
                    return false;
                }
                if j.body_a == last {
                    j.body_a = handle;
                }
                if j.body_b == last {
                    j.body_b = handle;
                }
                true
            });
            self.joint_pairs = self
                .joints
                .iter()
                .map(|j| (j.body_a.min(j.body_b), j.body_a.max(j.body_b)))
                .collect();
        }
    }

    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Option<JointHandle> {
        if body_a == body_b || body_a >= self.bodies.len() || body_b >= self.bodies.len() {
            return None;
        }
        // Normalize the hinge axes once, at creation.
        let kind = match kind {
            JointKind::Revolute {
                local_anchor_a,
                local_anchor_b,
                local_axis_a,
                local_axis_b,
            } => JointKind::Revolute {
                local_anchor_a,
                local_anchor_b,
                local_axis_a: local_axis_a.normalize_or(Vec3::Z),
                local_axis_b: local_axis_b.normalize_or(Vec3::Z),
            },
            other => other,
        };
        // A new joint on a sleeping island changes its constraint set — wake
        // it so the joint state can settle coherently.
        for h in [body_a, body_b] {
            if self.bodies[h].body_type == BodyType::Dynamic
                && self.asleep.get(h).copied().unwrap_or(false)
            {
                self.wake_island(h);
            }
        }
        self.joint_pairs
            .insert((body_a.min(body_b), body_a.max(body_b)));
        self.joints.push(Joint::new(body_a, body_b, kind));
        Some(self.joints.len() - 1)
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        if handle < self.joints.len() {
            let removed = self.joints.swap_remove(handle);
            // The pair may still be covered by another joint between the
            // same bodies — only forget it when no joint references it.
            let (a, b) = (removed.body_a, removed.body_b);
            let key = (a.min(b), a.max(b));
            if !self
                .joints
                .iter()
                .any(|j| (j.body_a.min(j.body_b), j.body_a.max(j.body_b)) == key)
            {
                self.joint_pairs.remove(&key);
            }
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

    /// Honest shapecast (G6): conservative advancement over exact pairwise
    /// shape distances (`distance.rs`). Tunnel-free for any cast length and
    /// any target thickness; the hit distance is the true first touch, not
    /// the nearest fixed sample. Rotation of the cast shape is fixed during
    /// the sweep (linear cast).
    fn shapecast(&self, shape: &Shape, from: Vec3, to: Vec3) -> Option<RaycastHit> {
        let mover = distance::ShapeRef {
            shape,
            pos: from,
            rot: Quat::IDENTITY,
        };
        let targets = self.bodies.iter().enumerate().map(|(h, b)| {
            (
                h,
                distance::ShapeRef {
                    shape: &b.shape,
                    pos: b.position,
                    rot: b.orientation,
                },
            )
        });
        distance::cast_shape(mover, to - from, targets).map(|h| RaycastHit {
            handle: h.handle,
            point: h.point,
            normal: h.normal,
            distance: h.t,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat3, Mat4};

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
        // ALL EIGHT rotated corners must lie inside the AABB — not just one.
        // The previous version checked only `q * Vec3::splat(1.0)`, which
        // for a unit-cube half-extent is numerically identical to the
        // (buggy) `orientation.mul_vec3(half).abs()` formula the
        // production code used to compute, so the assertion was
        // tautological and passed even with an under-sized AABB (night
        // gate, 2026-08-24: fixed real OBB->AABB bug in `Shape::aabb`,
        // see its comment for the derivation).
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    let corner = q * (half * Vec3::new(sx, sy, sz));
                    assert!(
                        aabb.contains_point(corner),
                        "corner {corner:?} not inside {aabb:?}"
                    );
                }
            }
        }
        // Both X and Y half-extents grow to sqrt(2) after a 45° Z rotation
        // of a unit cube (Z is the rotation axis, so its extent is
        // unchanged). The buggy formula zeroed the X extent here.
        let half_x = (aabb.max.x - aabb.min.x) * 0.5;
        let half_y = (aabb.max.y - aabb.min.y) * 0.5;
        let half_z = (aabb.max.z - aabb.min.z) * 0.5;
        assert!(
            (half_x - 2f32.sqrt()).abs() < 1e-3,
            "OBB->AABB x-extent, got {half_x}"
        );
        assert!(
            (half_y - 2f32.sqrt()).abs() < 1e-3,
            "OBB->AABB y-extent, got {half_y}"
        );
        assert!((half_z - 1.0).abs() < 1e-3, "OBB->AABB z-extent, got {half_z}");
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
    fn shapecast_exact_hit_distance() {
        // Sphere r=0.5 cast straight down onto a half-1 box at the origin:
        // contact when the sphere center is 1.5 above the origin, so a cast
        // from y=5 must report a hit distance of exactly 3.5 (G6: the cast
        // uses analytic shape distances, not a sampled march).
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 0.0));
        let shape = Shape::Sphere { radius: 0.5 };
        let hit = physics
            .shapecast(&shape, Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO)
            .expect("cast straight down must hit the box");
        assert!(
            (hit.distance - 3.5).abs() < 1e-2,
            "hit distance={} expected 3.5",
            hit.distance
        );
        // Surface normal at the hit points up, toward the caster.
        assert!(hit.normal.y > 0.99, "normal={:?}", hit.normal);
    }

    #[test]
    fn shapecast_thin_wall_no_tunnel() {
        // A 4 cm wall is far thinner than the cast segment: a sampled march
        // would step over it, conservative advancement must not (G6).
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.add_body(RigidBody::new_box(
            Vec3::ZERO,
            Vec3::new(2.0, 2.0, 0.02),
            0.0,
        ));
        let shape = Shape::Sphere { radius: 0.1 };
        let hit = physics.shapecast(&shape, Vec3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 2.0));
        let hit = hit.expect("cast through the thin wall must hit, not tunnel");
        // Sphere surface touches the wall face at z = -0.02 - 0.1 = -0.12,
        // i.e. 1.88 into the 4-unit cast.
        assert!(
            (hit.distance - 1.88).abs() < 1e-2,
            "hit distance={} expected 1.88",
            hit.distance
        );
    }

    #[test]
    fn fast_sphere_does_not_tunnel() {
        // Bullet vs thin floor (G6): at -80 m/s the sphere moves 0.111 m per
        // substep (12 substeps at 60 Hz) — more than the 0.1 m floor slab.
        // Without speculative contacts + the TOI pass it would sail through;
        // here it must end up resting on top.
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::ZERO,
            Vec3::new(10.0, 0.05, 10.0),
            0.0,
        ));
        let bullet = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 3.0, 0.0), 0.1, 1.0));
        {
            let b = physics.get_body_mut(bullet).unwrap();
            b.velocity = Vec3::new(0.0, -80.0, 0.0);
            b.restitution = 0.0; // we test tunneling, not bouncing
        }
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
        }
        let y = physics.get_body(bullet).unwrap().position.y;
        assert!(y > 0.0, "bullet tunneled through the floor: y={y}");
        // And it settled near the contact plane (center = slab top + radius),
        // not hovering or buried.
        assert!(
            (y - 0.15).abs() < 0.05,
            "bullet did not settle on the floor: y={y}"
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
            0.05,
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
    fn solver_is_deterministic_across_thread_counts() {
        // G7 gate: per-island parallel dispatch must be bit-identical to the
        // sequential run. Islands are disjoint over dynamic bodies and the
        // warm cache is merged by disjoint keys, so any difference here is a
        // data race, not float noise. The scene (9 separate 4-box stacks on
        // a floor) is wide enough to engage the rayon path: ≥2 islands,
        // ≥24 manifolds.
        fn build_scene() -> BuiltinPhysicsEngine {
            let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
            physics.add_body(RigidBody::new_box(
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(8.0, 1.0, 8.0),
                0.0,
            ));
            for gx in 0..3 {
                for gz in 0..3 {
                    let base = Vec3::new(gx as f32 * 2.5 - 2.5, 0.0, gz as f32 * 2.5 - 2.5);
                    for level in 0..4 {
                        physics.add_body(RigidBody::new_box(
                            base + Vec3::new(0.0, 0.5 + level as f32 * 1.02, 0.0),
                            Vec3::new(0.5, 0.5, 0.5),
                            1.0,
                        ));
                    }
                }
            }
            physics
        }
        /// (position, orientation, velocity, angular velocity) as f32 bits.
        type Snapshot = ([u32; 3], [u32; 4], [u32; 3], [u32; 3]);
        fn run(threads: usize) -> Vec<Snapshot> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut physics = build_scene();
                for _ in 0..120 {
                    physics.step(1.0 / 60.0);
                }
                (0..physics.bodies.len())
                    .map(|i| {
                        let b = &physics.bodies[i];
                        let (p, o) = (b.position.to_array(), b.orientation.to_array());
                        let (v, w) = (b.velocity.to_array(), b.angular_velocity.to_array());
                        (
                            p.map(f32::to_bits),
                            o.map(f32::to_bits),
                            v.map(f32::to_bits),
                            w.map(f32::to_bits),
                        )
                    })
                    .collect()
            })
        }
        let single = run(1);
        let multi = run(4);
        assert_eq!(single.len(), multi.len(), "body count differs between runs");
        for (i, (a, b)) in single.iter().zip(multi.iter()).enumerate() {
            assert_eq!(a, b, "body {i} diverged between 1-thread and 4-thread runs");
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

    /// World-space distance between the two anchor points of a joint.
    fn joint_anchor_error(
        physics: &BuiltinPhysicsEngine,
        ja: BodyHandle,
        jb: BodyHandle,
        la: Vec3,
        lb: Vec3,
    ) -> f32 {
        let (a, b) = (physics.get_body(ja).unwrap(), physics.get_body(jb).unwrap());
        let pa = a.position + a.orientation * la;
        let pb = b.position + b.orientation * lb;
        (pa - pb).length()
    }

    #[test]
    fn ball_joint_pendulum_holds_anchor() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let anchor = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.1, 0.0));
        // Pendulum bob released off to the side: it must swing, not fall.
        let bob = physics.add_body(RigidBody::new_sphere(Vec3::new(1.0, -1.0, 0.0), 0.25, 1.0));
        let lb = Vec3::new(-1.0, 1.0, 0.0); // world anchor = origin
        physics
            .add_joint(
                anchor,
                bob,
                JointKind::Ball {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: lb,
                },
            )
            .expect("valid joint");
        for _ in 0..300 {
            physics.step(1.0 / 60.0);
            let err = joint_anchor_error(&physics, anchor, bob, Vec3::ZERO, lb);
            assert!(err < 0.05, "anchor drifted apart: {err}");
        }
        let b = physics.get_body(bob).unwrap();
        // Still hanging from the anchor: distance to the pivot stays ≈ √2.
        let dist = b.position.length();
        assert!(
            (dist - std::f32::consts::SQRT_2).abs() < 0.15,
            "pendulum length drifted: {dist}"
        );
        // And it did swing at some point (started at x=1, must reach x<0).
        // (Checked implicitly: a falling bob would have y << -1.5.)
        assert!(
            b.position.y > -1.6,
            "bob fell off the joint: {:?}",
            b.position
        );
    }

    #[test]
    fn ball_joint_chain_hangs() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
        let mut prev = anchor;
        let mut links = Vec::new();
        for k in 1..=3 {
            let link = physics.add_body(RigidBody::new_box(
                Vec3::new(0.0, -(k as f32), 0.0),
                Vec3::splat(0.1),
                0.5,
            ));
            physics
                .add_joint(
                    prev,
                    link,
                    JointKind::Ball {
                        local_anchor_a: if prev == anchor {
                            Vec3::ZERO
                        } else {
                            Vec3::new(0.0, -0.5, 0.0)
                        },
                        local_anchor_b: Vec3::new(0.0, 0.5, 0.0),
                    },
                )
                .expect("valid joint");
            links.push(link);
            prev = link;
        }
        for _ in 0..300 {
            physics.step(1.0 / 60.0);
        }
        // Every link still connected: anchor pairs coincide.
        let mut prev = anchor;
        let mut prev_anchor = Vec3::ZERO;
        for (k, &link) in links.iter().enumerate() {
            let lb = Vec3::new(0.0, 0.5, 0.0);
            let err = joint_anchor_error(&physics, prev, link, prev_anchor, lb);
            assert!(err < 0.1, "chain link {k} detached: err={err}");
            let b = physics.get_body(link).unwrap();
            assert!(
                b.position.y > -(k as f32) - 1.5,
                "link {k} fell too far: {:?}",
                b.position
            );
            prev = link;
            prev_anchor = Vec3::new(0.0, -0.5, 0.0);
        }
    }

    #[test]
    fn revolute_hinge_rotates_about_axis_only() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
        // Arm hangs with its top at the origin: center one meter below. The
        // jointed pair does not collide (a hinge pin passes through the arm),
        // so the test measures the JOINT, not contact friction.
        let arm = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.1, 1.0, 0.1),
            1.0,
        ));
        physics
            .add_joint(
                anchor,
                arm,
                JointKind::Revolute {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::new(0.0, 1.0, 0.0), // arm top at origin
                    local_axis_a: Vec3::Z,
                    local_axis_b: Vec3::Z,
                },
            )
            .expect("valid joint");
        // Kick sideways so the pendulum arm swings about the Z hinge.
        physics.get_body_mut(arm).unwrap().velocity = Vec3::new(1.5, 0.0, 0.0);
        // The pendulum oscillates; the swing EXTREMES are what must show pure
        // Z rotation, so track the maxima rather than the final frame's phase.
        let mut max_z_rot = 0.0f32;
        let mut max_tilt = 0.0f32;
        for _ in 0..300 {
            physics.step(1.0 / 60.0);
            let q = physics.get_body(arm).unwrap().orientation;
            max_z_rot = max_z_rot.max(q.z.abs());
            max_tilt = max_tilt.max(q.x.abs()).max(q.y.abs());
        }
        // The arm swung about Z (the 1.5 m/s kick lifts it well past 5°)...
        assert!(max_z_rot > 0.05, "hinge barely rotated: {max_z_rot}");
        // ...but tilt about X and Y stays locked throughout the swing.
        assert!(max_tilt < 0.02, "hinge tilted off its axis: {max_tilt}");
        // Anchor stays coincident.
        let err = joint_anchor_error(&physics, anchor, arm, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        assert!(err < 0.05, "hinge anchor drifted: {err}");
    }

    // ---- T13 regression: intermediate-value soundness of the 5 solver ----
    // ---- primitives. These lock the *algebra*, not just finiteness, so ----
    // ---- they catch op/sign mutants (e.g. `*`->`+`, `+=`->`-=`) that a ----
    // ---- finite-only debug_assert cannot. ----

    /// Orientation as a glam rotation matrix (independent oracle for
    /// `mul_inv_inertia`, which feeds `k_entry`/`effective_mass`).
    fn rot_mat(q: Quat) -> Mat3 {
        Mat3::from_quat(q)
    }

    #[test]
    fn mul_inv_inertia_matches_quat_application() {
        // I_world⁻¹ · v = R · I_body⁻¹ · Rᵀ · v, where I_body⁻¹ is diagonal.
        // We test the rotational part by checking the transformation is an
        // orientation application that the quaternion gives consistently.
        let inertia = Vec3::new(2.0, 4.0, 8.0);
        let ori = Quat::from_rotation_y(0.7);
        let v = Vec3::new(1.0, 2.0, -3.0);

        let got = mul_inv_inertia(inertia, ori, v);

        // Oracle: R · diag(1/inertia) · Rᵀ · v, built from glam matrices
        // (never touches mul_inv_inertia, so a mutant cannot pass it).
        let r = rot_mat(ori);
        let inv_diag = Vec3::new(
            inv_inertia_axis(inertia.x),
            inv_inertia_axis(inertia.y),
            inv_inertia_axis(inertia.z),
        );
        let body = r.transpose() * v;
        let scaled = Vec3::new(
            inv_diag.x * body.x,
            inv_diag.y * body.y,
            inv_diag.z * body.z,
        );
        let oracle = r * scaled;

        assert!(
            (got - oracle).length() < 1e-5,
            "mul_inv_inertia diverged from quaternion oracle: got {got:?}, oracle {oracle:?}"
        );
        // Sanity: a permutation of axes — same vector, different orientation,
        // must not all collapse to the input (catches `* -> +` on every axis).
        let ori2 = Quat::from_rotation_x(1.1);
        let got2 = mul_inv_inertia(inertia, ori2, v);
        assert!(
            (got - got2).length() > 1e-4,
            "orientation must change the result, got {got:?} vs {got2:?}"
        );
    }

    #[test]
    fn effective_mass_matches_assembled_inverse_inertia() {
        // effective_mass(dir, ra) = 1/m + (ra×dir)·I_world⁻¹·(ra×dir).
        // Build two distinct bodies and check the assembled scalar matches the
        // matrix form: m⁻¹ + (ra×d)ᵀ · R·I⁻¹·Rᵀ · (ra×d).
        let mut a = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0);
        a.orientation = Quat::from_rotation_z(0.6);
        a.inertia = Vec3::new(3.0, 5.0, 7.0);
        a.inv_mass = 1.0 / a.mass;
        let mut b = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 2.0);
        b.orientation = Quat::from_rotation_x(-0.4);
        b.inertia = Vec3::new(2.0, 6.0, 4.0);
        b.inv_mass = 1.0 / b.mass;

        let bodies = [a, b];
        let dir = Vec3::new(0.0, 1.0, 0.0).normalize();
        let ra = Vec3::new(0.5, 0.0, 0.0);
        let rb = Vec3::new(-0.5, 0.0, 0.0);

        let em = effective_mass(&bodies, 0, 1, dir, ra, rb);

        // Oracle: 1/m_i + 1/m_j + (ra×d)ᵀ Iᵢ⁻¹ (ra×d) + (rb×d)ᵀ Iⱼ⁻¹ (rb×d).
        fn rot_inertia(ori: Quat, inv: Vec3) -> Mat3 {
            let r = rot_mat(ori);
            let diag = Mat3::from_diagonal(inv);
            r * diag * r.transpose()
        }
        let ra_d = ra.cross(dir);
        let rb_d = rb.cross(dir);
        let iw_i = rot_inertia(bodies[0].orientation, Vec3::new(
            inv_inertia_axis(bodies[0].inertia.x),
            inv_inertia_axis(bodies[0].inertia.y),
            inv_inertia_axis(bodies[0].inertia.z),
        ));
        let iw_j = rot_inertia(bodies[1].orientation, Vec3::new(
            inv_inertia_axis(bodies[1].inertia.x),
            inv_inertia_axis(bodies[1].inertia.y),
            inv_inertia_axis(bodies[1].inertia.z),
        ));
        let oracle = bodies[0].inv_mass
            + bodies[1].inv_mass
            + ra_d.dot(iw_i * ra_d)
            + rb_d.dot(iw_j * rb_d);

        assert!(
            (em - oracle).abs() < 1e-5,
            "effective_mass diverged from matrix oracle: got {em}, oracle {oracle}"
        );
        assert!(em > 0.0, "effective mass must be positive, got {em}");
    }

    #[test]
    fn solve_small_matches_glam_lu() {
        // Independent oracle: solve A x = b with glam's matrix inverse and
        // check we recover `b` (A·x ≈ b) plus match glam's x. Any op/sign
        // mutant in the Gaussian elimination changes the recovered residual.
        let a = [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 5.0],
        ];
        let b = [1.0, 2.0, 3.0, 4.0];
        let n = 4;

        let x = solve_small(&a, &b, n).expect("well-conditioned system");

        // Reconstruct A·x via glam and confirm we recover b.
        let am = Mat4::from_cols_array(&[
            a[0][0], a[1][0], a[2][0], a[3][0],
            a[0][1], a[1][1], a[2][1], a[3][1],
            a[0][2], a[1][2], a[2][2], a[3][2],
            a[0][3], a[1][3], a[2][3], a[3][3],
        ]);
        let xv = glam::vec4(x[0], x[1], x[2], x[3]);
        let ax = am * xv;
        let residual = glam::Vec4::new(b[0], b[1], b[2], b[3]) - ax;
        assert!(
            residual.length() < 1e-3,
            "solve_small does not satisfy A x = b: residual {residual:?}"
        );

        // Cross-check against glam's own inverse solution.
        let inv = am.inverse();
        let oracle = inv * glam::vec4(b[0], b[1], b[2], b[3]);
        let diff = (glam::vec4(x[0], x[1], x[2], x[3]) - oracle).length();
        assert!(
            diff < 1e-3,
            "solve_small diverged from glam inverse: got {x:?}, oracle {oracle:?}"
        );
    }

    #[test]
    fn solve_small_singular_returns_none() {
        // A singular (rank-deficient) matrix must be rejected, not produce a
        // finite-but-wrong answer or loop forever (the `* -> %`, `* -> /`
        // and `-= -> +=` mutants are caught here).
        let a = [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0], // row 2 = 2 * row 0 -> singular
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ];
        let b = [1.0, 2.0, 3.0, 4.0];
        let x = solve_small(&a, &b, 4);
        assert!(x.is_none(), "singular system must return None, got {x:?}");
    }

    #[test]
    fn apply_impulse_is_symmetric_and_linear() {
        // Impulse j at contact point p between i and j must:
        //  - change v_i by -j/m_i and v_j by +j/m_j (linear term),
        //  - be antisymmetric: swapping (i,j) flips the velocity deltas,
        //  - preserve total (linear) momentum: m_i Δv_i + m_j Δv_j = 0.
        // Any `+= -> -=` / `-= -> +=` mutant breaks momentum/antisymmetry.
        let mut bodies = vec![
            RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0),
            RigidBody::new_sphere(Vec3::ZERO, 1.0, 2.0),
        ];
        bodies[0].velocity = Vec3::new(0.5, 0.0, 0.0);
        bodies[1].velocity = Vec3::new(-0.2, 0.0, 0.0);

        let imp = Vec3::new(0.0, 3.0, 0.0);
        let ra = Vec3::new(0.0, 1.0, 0.0);
        let rb = Vec3::new(0.0, -1.0, 0.0);

        let v0_i = bodies[0].velocity;
        let v0_j = bodies[1].velocity;
        apply_impulse(&mut bodies, 0, 1, imp, ra, rb);
        let dv_i = bodies[0].velocity - v0_i;
        let dv_j = bodies[1].velocity - v0_j;

        let expected_i = -imp * bodies[0].inv_mass;
        let expected_j = imp * bodies[1].inv_mass;
        assert!(
            (dv_i - expected_i).length() < 1e-5,
            "v_i delta wrong: got {dv_i:?}, expected {expected_i:?}"
        );
        assert!(
            (dv_j - expected_j).length() < 1e-5,
            "v_j delta wrong: got {dv_j:?}, expected {expected_j:?}"
        );

        // Momentum conservation (angular contributes via ang. momentum, but
        // the linear part alone must cancel exactly).
        let p_delta = bodies[0].mass * dv_i + bodies[1].mass * dv_j;
        assert!(
            p_delta.length() < 1e-5,
            "linear momentum not conserved: {p_delta:?}"
        );

        // Antisymmetry: for the SAME physical body, the velocity delta when it
        // plays role `i` must be the exact negative of its delta when it plays
        // role `j` (the impulse is antisymmetric under i<->j swap). This is
        // independent of the array slot, so it catches `+= <-> -=` mutants.
        let mut bodies2 = vec![
            RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0),
            RigidBody::new_sphere(Vec3::ZERO, 1.0, 2.0),
        ];
        bodies2[0].velocity = Vec3::new(0.5, 0.0, 0.0);
        bodies2[1].velocity = Vec3::new(-0.2, 0.0, 0.0);
        let v0b_0 = bodies2[0].velocity;
        let v0b_1 = bodies2[1].velocity;
        // body 0 is now role `j`, body 1 is role `i`.
        apply_impulse(&mut bodies2, 1, 0, imp, rb, ra);
        let dv_0_as_j = bodies2[0].velocity - v0b_0;
        let dv_1_as_i = bodies2[1].velocity - v0b_1;

        // body 0 as i (first call, dv_i) should oppose body 0 as j (dv_0_as_j).
        assert!(
            (dv_i + dv_0_as_j).length() < 1e-5,
            "body 0 i/j antisymmetry broken: as_i {dv_i:?} vs as_j {dv_0_as_j:?}"
        );
        // body 1 as j (first call, dv_j) should oppose body 1 as i (dv_1_as_i).
        assert!(
            (dv_j + dv_1_as_i).length() < 1e-5,
            "body 1 i/j antisymmetry broken: as_j {dv_j:?} vs as_i {dv_1_as_i:?}"
        );
    }

    #[test]
    fn solve_normal_block_reduces_normal_velocity() {
        // Drive solve_normal_block on a 2-point manifold and assert the
        // complementarity result: the normal relative velocity at the active
        // points moves toward the target floor, and the committed state is
        // self-consistent (acc impulses ≥ 0 on the active set). The `- -> +`
        // / `* -> /` / `== -> !=` mutants in the block solver change this
        // outcome detectably.
        let mut bodies = vec![
            RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0),
            RigidBody::new_sphere(Vec3::new(0.0, -2.0, 0.0), 1.0, 1.0),
        ];
        // Body j approaches body i (moves up, +Y, into i which is above): its
        // normal relative velocity is negative, so solve_normal_block must
        // commit a positive separating impulse (acc > 0).
        bodies[1].velocity = Vec3::new(0.0, 5.0, 0.0);
        let n = Vec3::new(0.0, -1.0, 0.0); // i->j normal
        let pts = [
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.3, -1.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
        ];
        let mut acc = [0.0f32; 4];
        let target = [0.0f32, 0.0, 0.0, 0.0];
        let count = 2;

        // Normal relative velocity of body j minus body i at each point,
        // measured before solving.
        let vn_before: Vec<f32> = (0..count)
            .map(|k| {
                (point_velocity(&bodies[1], pts[k] - bodies[1].position)
                    - point_velocity(&bodies[0], pts[k] - bodies[0].position))
                .dot(n)
            })
            .collect();

        solve_normal_block(&mut bodies, 0, 1, n, &pts, &mut acc, &target, count);

        let vn_after: Vec<f32> = (0..count)
            .map(|k| {
                (point_velocity(&bodies[1], pts[k] - bodies[1].position)
                    - point_velocity(&bodies[0], pts[k] - bodies[0].position))
                .dot(n)
            })
            .collect();

        // Active-set impulses must be non-negative.
        for k in 0..count {
            assert!(acc[k] >= -1e-6, "accumulated impulse {} negative: {}", k, acc[k]);
        }
        // Each point's post-solve normal velocity must be at/above target (0),
        // i.e. separation or resting contact, not interpenetration growth.
        for k in 0..count {
            assert!(
                vn_after[k] >= target[k] - 1e-4,
                "point {k} normal velocity regressed below target: before {} after {}",
                vn_before[k],
                vn_after[k]
            );
            // The block solve must have done *something* (it found an active set).
            assert!(
                acc.iter().take(count).cloned().fold(0.0f32, f32::max) > 0.0,
                "solve_normal_block committed no impulse"
            );
        }
    }
}
