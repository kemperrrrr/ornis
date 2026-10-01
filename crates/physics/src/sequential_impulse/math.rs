//! Small solver-math helpers shared by the sequential-impulse stages:
//! clamping/finiteness guards, union-find, inertia and effective-mass
//! kernels, impulse application and the block-LCP active-set solve.

use glam::{Quat, Vec3};

use super::MAX_MANIFOLD_POINTS;
use crate::body::RigidBody;
use crate::constants::{DEGENERATE_LEN2, POS_CORRECTION_EPS};

/// Velocity-feasibility slop for inactive-set checks (m/s).
const FEASIBLE_VEL_SLOP: f32 = 1e-5;

#[inline]
pub(crate) fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// Reciprocal inertia axis with zero-guard: `1/i` for positive `i`, else 0.
#[inline]
pub fn inv_inertia_axis(i: f32) -> f32 {
    if i > 0.0 { 1.0 / i } else { 0.0 }
}

#[inline]
pub(crate) fn vec3_finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

#[inline]
pub(crate) fn quat_finite(q: Quat) -> bool {
    q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite()
}

/// Union-find root with path halving. Bounded to `parent.len()` steps so a
/// corrupted parent array (or a cargo-mutants sign flip) cannot spin forever.
pub(crate) fn union_find(parent: &mut [usize], mut x: usize) -> usize {
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
pub fn mul_inv_inertia(inertia: Vec3, orientation: glam::Quat, v: Vec3) -> Vec3 {
    debug_assert!(
        vec3_finite(inertia),
        "inertia must be finite, got {inertia:?}"
    );
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
pub fn effective_mass(
    bodies: &[RigidBody],
    i: usize,
    j: usize,
    dir: Vec3,
    ra: Vec3,
    rb: Vec3,
) -> f32 {
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
pub(crate) fn k_entry(
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
pub fn solve_small(
    a: &[[f32; MAX_MANIFOLD_POINTS]; MAX_MANIFOLD_POINTS],
    b: &[f32; MAX_MANIFOLD_POINTS],
    n: usize,
) -> Option<[f32; MAX_MANIFOLD_POINTS]> {
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
        if m[piv][col].abs() < DEGENERATE_LEN2 {
            return None;
        }
        if piv != col {
            m.swap(piv, col);
            x.swap(piv, col);
        }
        let d = m[col][col];
        debug_assert!(
            d.is_finite() && d.abs() > DEGENERATE_LEN2,
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
    let mut out = [0.0f32; MAX_MANIFOLD_POINTS];
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
pub fn apply_impulse(bodies: &mut [RigidBody], i: usize, j: usize, imp: Vec3, ra: Vec3, rb: Vec3) {
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
pub(crate) fn apply_angular_impulse(bodies: &mut [RigidBody], i: usize, j: usize, imp: Vec3) {
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
pub(crate) fn apply_positional_rotation(body: &mut RigidBody, d: Vec3) {
    if d != Vec3::ZERO {
        body.orientation = (Quat::from_scaled_axis(d) * body.orientation).normalize();
    }
}

/// Velocity of a body at a world-space contact point (linear + angular part).
#[inline]
pub fn point_velocity(body: &RigidBody, r: Vec3) -> Vec3 {
    body.velocity + body.angular_velocity.cross(r)
}

/// CFM-regularized effective mass (b3MakeSoft analog): `cfm` > 0 softens the
/// constraint — the same position/velocity error produces a smaller impulse,
/// spread across iterations instead of a one-shot rigid correction.
#[inline]
pub(crate) fn make_soft(k: f32, cfm: f32) -> f32 {
    k + cfm
}

/// Apply a positional (pseudo) impulse to the body pair: moves positions and
/// orientations WITHOUT touching real velocities (split impulse / NGS).
pub(crate) fn apply_positional_impulse(
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

/// Projected block solve for a multi-point normal manifold: commits the
/// separating impulse that zeroes the approach velocity across `count` points.
#[allow(clippy::too_many_arguments)]
pub fn solve_normal_block(
    bodies: &mut [RigidBody],
    i: usize,
    j: usize,
    n: Vec3,
    pts: &[Vec3; MAX_MANIFOLD_POINTS],
    acc: &mut [f32; MAX_MANIFOLD_POINTS],
    target: &[f32; MAX_MANIFOLD_POINTS],
    count: usize,
) {
    let pa = bodies[i].position;
    let pb = bodies[j].position;
    let mut ras = [Vec3::ZERO; MAX_MANIFOLD_POINTS];
    let mut rbs = [Vec3::ZERO; MAX_MANIFOLD_POINTS];
    for k in 0..count {
        ras[k] = pts[k] - pa;
        rbs[k] = pts[k] - pb;
    }
    let mut k_mat = [[0.0f32; MAX_MANIFOLD_POINTS]; MAX_MANIFOLD_POINTS];
    for k in 0..count {
        for l in 0..count {
            k_mat[k][l] = k_entry(bodies, i, j, n, ras[k], rbs[k], ras[l], rbs[l]);
        }
    }
    let mut vn = [0.0f32; MAX_MANIFOLD_POINTS];
    for k in 0..count {
        vn[k] = (point_velocity(&bodies[j], rbs[k]) - point_velocity(&bodies[i], ras[k])).dot(n);
    }
    let geom = BlockGeom { ras, rbs };

    let total = 1u32 << count;
    for pop in (1..=count).rev() {
        for mask in 1..total {
            if mask.count_ones() as usize != pop {
                continue;
            }
            let set = active_set_indices(mask, count);
            let Some(ap) = try_active_set(&k_mat, &vn, acc, target, count, &set) else {
                continue;
            };
            commit_active_set(bodies, i, j, n, &geom, acc, &set, &ap);
            return;
        }
    }
    // No valid active set (numerically degenerate) — keep the warm-started
    // state; the next outer iteration will retry from updated velocities.
}

/// One candidate active set of the block-LCP enumeration: bitmask plus the
/// unpacked point indices (`idx[..ns]`).
pub(crate) struct ActiveSet {
    mask: u32,
    idx: [usize; MAX_MANIFOLD_POINTS],
    ns: usize,
    count: usize,
}

/// Contact-point lever arms of one manifold (body-relative anchors).
pub(crate) struct BlockGeom {
    ras: [Vec3; MAX_MANIFOLD_POINTS],
    rbs: [Vec3; MAX_MANIFOLD_POINTS],
}

pub(crate) fn active_set_indices(mask: u32, count: usize) -> ActiveSet {
    let mut idx = [0usize; MAX_MANIFOLD_POINTS];
    // count is carried so helpers do not need it as a separate argument.
    let mut ns = 0;
    for k in 0..count {
        if (mask >> k) & 1 == 1 {
            idx[ns] = k;
            ns += 1;
        }
    }
    ActiveSet {
        mask,
        idx,
        ns,
        count,
    }
}

/// Try one candidate active set: solve the reduced K system and verify
/// complementarity (acc' >= 0 on the active set, vn' >= target elsewhere).
/// Returns the new accumulated impulses on success.
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)] // mirrors the K-matrix block layout
pub(crate) fn try_active_set(
    k_mat: &[[f32; MAX_MANIFOLD_POINTS]; MAX_MANIFOLD_POINTS],
    vn: &[f32; MAX_MANIFOLD_POINTS],
    acc: &[f32; MAX_MANIFOLD_POINTS],
    target: &[f32; MAX_MANIFOLD_POINTS],
    count: usize,
    set: &ActiveSet,
) -> Option<[f32; MAX_MANIFOLD_POINTS]> {
    let ActiveSet { idx, ns, .. } = *set;
    let mut ks = [[0.0f32; MAX_MANIFOLD_POINTS]; MAX_MANIFOLD_POINTS];
    let mut bs = [0.0f32; MAX_MANIFOLD_POINTS];
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
    let ap = solve_small(&ks, &bs, ns)?;
    debug_assert!(
        ap.iter().take(ns).all(|v| v.is_finite()),
        "solve_normal_block: non-finite impulse solution"
    );
    if ap.iter().take(ns).any(|&v| v < -POS_CORRECTION_EPS) {
        return None;
    }
    if !inactive_feasible(k_mat, vn, acc, target, count, set, &ap) {
        return None;
    }
    Some(ap)
}

/// Check that removing the impulses of the inactive set keeps every inactive
/// point's normal velocity at or above its target.
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)] // mirrors the K-matrix block layout
pub(crate) fn inactive_feasible(
    k_mat: &[[f32; MAX_MANIFOLD_POINTS]; MAX_MANIFOLD_POINTS],
    vn: &[f32; MAX_MANIFOLD_POINTS],
    acc: &[f32; MAX_MANIFOLD_POINTS],
    target: &[f32; MAX_MANIFOLD_POINTS],
    count: usize,
    set: &ActiveSet,
    ap: &[f32; MAX_MANIFOLD_POINTS],
) -> bool {
    let ActiveSet { mask, idx, ns, .. } = *set;
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
        if v < target[t] - FEASIBLE_VEL_SLOP {
            return false;
        }
    }
    true
}

/// Commit a solved active set: apply the impulse deltas, store the new
/// accumulated impulses, and zero out the inactive set's impulses.
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)] // mirrors the K-matrix block layout
pub(crate) fn commit_active_set(
    bodies: &mut [RigidBody],
    i: usize,
    j: usize,
    n: Vec3,
    geom: &BlockGeom,
    acc: &mut [f32; MAX_MANIFOLD_POINTS],
    set: &ActiveSet,
    ap: &[f32; MAX_MANIFOLD_POINTS],
) {
    let ActiveSet {
        mask,
        idx,
        ns,
        count,
    } = *set;
    let BlockGeom { ras, rbs } = geom;
    for a in 0..ns {
        let k = idx[a];
        let d = ap[a] - acc[k];
        if d.abs() > DEGENERATE_LEN2 {
            apply_impulse(bodies, i, j, n * d, ras[k], rbs[k]);
        }
        acc[k] = ap[a];
    }
    for t in 0..count {
        if (mask >> t) & 1 == 0 && acc[t].abs() > DEGENERATE_LEN2 {
            let d = -acc[t];
            apply_impulse(bodies, i, j, n * d, ras[t], rbs[t]);
            acc[t] = 0.0;
        } else if (mask >> t) & 1 == 0 {
            acc[t] = 0.0;
        }
    }
}
