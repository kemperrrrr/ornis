//! Physics engine trait and the builtin CPU implementation.
//!
//! [`PhysicsEngine`] defines a single simulation step (broadphase →
//! narrowphase → island partitioning → substepped contact/joint solving →
//! integration; `dt` must be positive and finite) plus body/joint
//! management and ray/shape cast queries. [`BuiltinPhysicsEngine`] is the
//! reference implementation: parallelized with rayon, with optional GPU
//! contact solving behind the `gpu` feature.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use glam::{Quat, Vec3};
use rayon::prelude::*;

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::broadphase::{
    BroadPhase, BroadPhaseBackend, BroadPhaseKind, BroadPhaseStats, StepTiming,
};
use crate::distance;
#[cfg(feature = "gpu")]
use crate::gpu::WgpuContactSolver;
use crate::joint::{Joint, JointHandle, JointKind};
use crate::math::{Ray, RaycastHit};
use crate::shape::Shape;
use crate::trigger::{TriggerEvent, TriggerEventKind};
use crate::wide::{SolverStep, build_solver_steps};

/// Physics engine trait: a single step of simulation, plus body/joint management
/// and queries. Implementations may be CPU or GPU-based, single-threaded or multi-threaded.
pub trait PhysicsEngine: Send + Sync {
    /// Advance the simulation by `dt` seconds: broadphase → narrowphase →
    /// island partitioning → substepped velocity/position solving (contacts,
    /// friction, joints) → integration. `dt` must be > 0 and finite.
    fn step(&mut self, dt: f32);
    /// Register a body and return its stable handle.
    fn add_body(&mut self, body: RigidBody) -> BodyHandle;
    /// Remove a body; handles of later bodies shift down, so cached handles
    /// may become stale. Joints touching the removed body are destroyed too.
    fn remove_body(&mut self, handle: BodyHandle);
    /// Read-only access to a body, or `None` for an invalid handle.
    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody>;
    /// Mutable access to a body, or `None` for an invalid handle. Direct
    /// pose edits take effect at the next [`PhysicsEngine::step`].
    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody>;
    /// Create a joint between two existing, distinct bodies (G5).
    /// Returns None on invalid handles or a self-joint.
    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Option<JointHandle>;
    /// Destroy a joint by handle; no-op for an invalid handle.
    fn remove_joint(&mut self, handle: JointHandle);
    /// Closest exact shape hit of `ray` against registered bodies within
    /// `max_dist` (in units of the ray direction's length), or `None` if
    /// nothing is hit. Pass a normalized direction for world-distance units.
    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit>;
    /// Sweep `shape` along the segment `from → to` and report the first body
    /// hit (hit distance measured along the sweep direction), or `None`.
    fn shapecast(&self, shape: &Shape, from: Vec3, to: Vec3) -> Option<RaycastHit>;
    /// Drain trigger enter/exit transitions produced by completed steps.
    ///
    /// Engines without trigger support may keep the default empty result;
    /// the builtin engine reports canonical body-handle pairs in deterministic
    /// order.
    fn drain_trigger_events(&mut self) -> Vec<TriggerEvent> {
        Vec::new()
    }
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
struct ActiveSet {
    mask: u32,
    idx: [usize; 4],
    ns: usize,
    count: usize,
}

/// Contact-point lever arms of one manifold (body-relative anchors).
struct BlockGeom {
    ras: [Vec3; 4],
    rbs: [Vec3; 4],
}

fn active_set_indices(mask: u32, count: usize) -> ActiveSet {
    let mut idx = [0usize; 4];
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
fn try_active_set(
    k_mat: &[[f32; 4]; 4],
    vn: &[f32; 4],
    acc: &[f32; 4],
    target: &[f32; 4],
    count: usize,
    set: &ActiveSet,
) -> Option<[f32; 4]> {
    let ActiveSet { idx, ns, .. } = *set;
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
    let ap = solve_small(&ks, &bs, ns)?;
    debug_assert!(
        ap.iter().take(ns).all(|v| v.is_finite()),
        "solve_normal_block: non-finite impulse solution"
    );
    if ap.iter().take(ns).any(|&v| v < -1e-6) {
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
fn inactive_feasible(
    k_mat: &[[f32; 4]; 4],
    vn: &[f32; 4],
    acc: &[f32; 4],
    target: &[f32; 4],
    count: usize,
    set: &ActiveSet,
    ap: &[f32; 4],
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
        if v < target[t] - 1e-5 {
            return false;
        }
    }
    true
}

/// Commit a solved active set: apply the impulse deltas, store the new
/// accumulated impulses, and zero out the inactive set's impulses.
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)] // mirrors the K-matrix block layout
fn commit_active_set(
    bodies: &mut [RigidBody],
    i: usize,
    j: usize,
    n: Vec3,
    geom: &BlockGeom,
    acc: &mut [f32; 4],
    set: &ActiveSet,
    ap: &[f32; 4],
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
}

/// Detect actual (not speculative) overlaps for pairs containing a trigger.
///
/// Trigger geometry uses the exact distance oracle rather than the contact
/// margin: a nearby but non-overlapping body must not emit `Entered`. The
/// broadphase has already applied the mutual layer masks, so this pass only
/// performs the shape-level check.
fn detect_trigger_overlaps(bodies: &[RigidBody], active: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut overlaps = Vec::new();
    for &(i, j) in active {
        let a = &bodies[i];
        let b = &bodies[j];
        if !(a.is_trigger || b.is_trigger) || !a.can_collide_with(b) {
            continue;
        }
        let distance = distance::shape_distance(
            distance::ShapeRef {
                shape: &a.shape,
                pos: a.position,
                rot: a.orientation,
            },
            distance::ShapeRef {
                shape: &b.shape,
                pos: b.position,
                rot: b.orientation,
            },
        );
        if distance.dist <= 0.0 {
            overlaps.push((i, j));
        }
    }
    overlaps
}

/// Reconcile the current overlap set with the previous step and queue sorted
/// enter/exit events. Sorting keeps event order independent of broadphase
/// sweep-axis rotation and hash-set iteration order.
fn update_trigger_events(
    previous: &HashSet<(usize, usize)>,
    current: Vec<(usize, usize)>,
    events: &mut Vec<TriggerEvent>,
) -> HashSet<(usize, usize)> {
    let current_set: HashSet<(usize, usize)> = current.into_iter().collect();
    let mut entered: Vec<_> = current_set.difference(previous).copied().collect();
    let mut exited: Vec<_> = previous.difference(&current_set).copied().collect();
    entered.sort_unstable();
    exited.sort_unstable();
    events.extend(entered.into_iter().map(|(body_a, body_b)| TriggerEvent {
        body_a,
        body_b,
        kind: TriggerEventKind::Entered,
    }));
    events.extend(exited.into_iter().map(|(body_a, body_b)| TriggerEvent {
        body_a,
        body_b,
        kind: TriggerEventKind::Exited,
    }));
    current_set
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

    // B's corners touching A's face (the face most anti-parallel to `n`),
    // then A's corners touching B's face.
    let mut cand: Vec<(Vec3, f32)> = Vec::new();
    cand.extend(collect_face_corners(
        &obb_corners(pos_b, half_b, rot_b),
        &FaceProbe {
            pos: pos_a,
            half: half_a,
            rot: rot_a,
            hwn: hwn_a,
            dir: n,
            depth_tol,
            slack: tangent_slack,
        },
    ));
    cand.extend(collect_face_corners(
        &obb_corners(pos_a, half_a, rot_a),
        &FaceProbe {
            pos: pos_b,
            half: half_b,
            rot: rot_b,
            hwn: hwn_b,
            dir: -n,
            depth_tol,
            slack: tangent_slack,
        },
    ));

    // Deduplicate in the tangent plane, then keep the deepest four points.
    let mut uniq = dedupe_contact_points(cand, n);
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

/// One opposing face of an OBB to test the other box's corners against.
struct FaceProbe {
    pos: Vec3,
    half: Vec3,
    rot: Quat,
    /// Half-width of the probed face's box along the contact normal.
    hwn: f32,
    /// Direction pointing INTO the face along the contact normal.
    dir: Vec3,
    depth_tol: f32,
    slack: f32,
}

/// Collect the corners of one box that touch the probed face of the other:
/// depth along the normal within tolerance AND tangential containment inside
/// the face rectangle (with slack).
fn collect_face_corners(corners: &[Vec3; 8], probe: &FaceProbe) -> Vec<(Vec3, f32)> {
    let mut out: Vec<(Vec3, f32)> = Vec::new();
    for c in corners {
        let local = probe.rot.inverse() * (*c - probe.pos);
        // Depth of the corner relative to the surface along the normal.
        let d = probe.hwn - (*c - probe.pos).dot(probe.dir);
        if d < -probe.depth_tol {
            continue;
        }
        if local.x.abs() <= probe.half.x + probe.slack
            && local.y.abs() <= probe.half.y + probe.slack
            && local.z.abs() <= probe.half.z + probe.slack
        {
            out.push((*c, d));
        }
    }
    out
}

/// Merge near-coincident contact candidates in the tangent plane: the same
/// contact region appears once from each box's corners, offset along the
/// normal by the penetration depth. Keep the deeper representative — a stable
/// 4-point manifold instead of a flickering mix.
fn dedupe_contact_points(cand: Vec<(Vec3, f32)>, n: Vec3) -> Vec<(Vec3, f32)> {
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
    uniq
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

/// Capsule collision parameters (keeps `capsule_vs_capsule` within the structural gate's
/// argument-count limit).
struct CapsuleShape {
    pos: Vec3,
    radius: f32,
    half_height: f32,
    rot: Quat,
}

/// Capsule-capsule: both segment axes are rotated by the body orientation.
fn capsule_vs_capsule(a: &CapsuleShape, b: &CapsuleShape, margin: f32) -> Option<Contact> {
    let ax = a.rot * Vec3::Y;
    let bx = b.rot * Vec3::Y;
    let bot_a = a.pos - ax * a.half_height;
    let bot_b = b.pos - bx * b.half_height;

    let seg_a = ax * (2.0 * a.half_height);
    let seg_b = bx * (2.0 * b.half_height);
    let diff = bot_b - bot_a;
    let q = seg_a.dot(seg_a);
    let r = seg_a.dot(seg_b);
    let c = seg_b.dot(seg_b);
    let d = seg_a.dot(diff);
    let e = seg_b.dot(diff);
    let det = q * c - r * r;

    let (t_a, t_b) = if det.abs() < 1e-10 {
        (0.0, if c > 0.0 { e / c } else { 0.0 })
    } else {
        ((r * e - c * d) / det, (q * e - r * d) / det)
    };
    let t_a = clamp01(t_a);
    let t_b = clamp01(t_b);

    let closest_a = bot_a + seg_a * t_a;
    let closest_b = bot_b + seg_b * t_b;
    let diff2 = closest_b - closest_a;
    let dist_sq = diff2.length_squared();
    let radius_sum = a.radius + b.radius + margin;
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
        if !a.can_collide_with(b) || a.is_trigger || b.is_trigger {
            continue;
        }
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
                &CapsuleShape {
                    pos: a.position,
                    radius: ra,
                    half_height: ha,
                    rot: a.orientation,
                },
                &CapsuleShape {
                    pos: b.position,
                    radius: rb,
                    half_height: hb,
                    rot: b.orientation,
                },
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
/// keeping `build_manifold_state` below the structural gate's nargs limit).
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

mod contacts;
mod islands;
mod joints;

/// The CPU reference physics engine: sequential-impulse solver with a
/// selectable broadphase, manifold generation, island-coherent sleeping,
/// warm-started contacts and joints, and optional SIMD-wide / GPU contact
/// solving. Sweep-and-Prune is the default; UniformGrid is opt-in while its
/// workload tradeoffs are benchmarked.
///
/// Step pipeline per [`PhysicsEngine::step`]: rebuild AABBs and broadphase
/// pairs → narrowphase manifolds → union-find islands (contacts + joints) →
/// `substeps` × (warm start, velocity iterations with friction/restitution,
/// positional Baumgarte pass) → integration. Bodies outside active islands
/// sleep as a whole island and wake together.
pub struct BuiltinPhysicsEngine {
    bodies: Vec<RigidBody>,
    broadphase: BroadPhaseBackend,
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
    /// Trigger pairs overlapping on the previous completed step.
    trigger_pairs: HashSet<(usize, usize)>,
    /// Wall-clock breakdown of the last completed `step` (diagnostics only).
    last_step_timing: StepTiming,
    /// Enter/exit transitions waiting for the caller to drain.
    trigger_events: Vec<TriggerEvent>,
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
    /// Empty engine with the default tuning: 12 substeps, 8 velocity
    /// iterations, 4 position iterations, rigid contacts, SIMD-wide solver
    /// on, no gravity until set here. `gravity` is a constant world-space
    /// acceleration (m/s²) applied to dynamic bodies each step.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            bodies: Vec::new(),
            broadphase: BroadPhaseBackend::new(BroadPhaseKind::UniformGrid),
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
            trigger_pairs: HashSet::new(),
            last_step_timing: StepTiming::default(),
            trigger_events: Vec::new(),
            wide_solver: true,
            #[cfg(feature = "gpu")]
            gpu_solver: None,
        }
    }

    /// Select the broadphase candidate-pair backend.
    ///
    /// The default is [`BroadPhaseKind::UniformGrid`] (wins the local 10k-body
    /// scene matrix: tiled / giant_floor / sparse / islands / heterogeneous).
    /// [`BroadPhaseKind::SweepAndPrune`] is retained as the compatibility
    /// baseline; [`BroadPhaseKind::DynamicAabbTree`] is experimental.
    pub fn set_broadphase(&mut self, kind: BroadPhaseKind) {
        if self.broadphase.kind() != kind {
            self.broadphase = BroadPhaseBackend::new(kind);
            self.warm_impulses.clear();
        }
    }

    /// Returns the currently selected broadphase backend.
    pub fn broadphase_kind(&self) -> BroadPhaseKind {
        self.broadphase.kind()
    }

    /// Selects the uniform-grid backend and configures its cell size.
    ///
    /// Smaller cells reduce false candidate pairs at the cost of more cell
    /// bookkeeping. The default grid size is 2.0 world units. This method
    /// resets the warm-start cache because changing the backend is a
    /// diagnostic/configuration boundary between simulation runs.
    pub fn set_uniform_grid_cell_size(&mut self, cell_size: f32) {
        self.broadphase = BroadPhaseBackend::uniform_grid(cell_size);
        self.warm_impulses.clear();
    }

    /// Returns counters from the latest broadphase update.
    ///
    /// The values are diagnostics for tuning and benchmarks; they are not
    /// part of the simulation contract.
    pub fn broadphase_stats(&self) -> BroadPhaseStats {
        self.broadphase.stats()
    }

    /// Wall-clock breakdown of the last completed `step`.
    ///
    /// Diagnostic only: per-substep phases are summed across the substep loop.
    /// Zeroed until the first step runs.
    pub fn step_timing(&self) -> StepTiming {
        self.last_step_timing
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

    /// Number of sub-iterations the solver splits each `step(dt)` into
    /// (default 12). More substeps = more stable stacks, linearly more cost.
    pub fn set_substeps(&mut self, n: u32) {
        self.substeps = n;
    }

    /// Sequential-impulse velocity iterations per substep (default 8).
    pub fn set_velocity_iterations(&mut self, n: u32) {
        self.velocity_iterations = n;
    }

    /// Adaptive substep count for this step: keep each sub-dt at or below a
    /// target so fast bodies get enough solver passes to settle without
    /// tunnelling, while resting/low-speed scenes drop to `MIN_SUBSTEPS` so the
    /// world can sleep cheaply. `self.substeps` is the upper bound (set by the
    /// caller); the solver never runs more than that.
    ///
    /// ponytail: global heuristic clamped to a fixed min — per-island or
    /// penetration-driven substepping is the upgrade path when heterogeneous
    /// scenes show regressions.
    fn effective_substeps(&self, dt: f32) -> u32 {
        const MIN_SUBSTEPS: u32 = 4;
        const SUB_DT_TARGET: f32 = 1.0 / 240.0;
        let mut max_speed = 0.0f32;
        for (h, b) in self.bodies.iter().enumerate() {
            if b.body_type == BodyType::Dynamic && !self.asleep[h] {
                max_speed = max_speed.max(b.velocity.length());
            }
        }
        // Adaptive substepping only *lowers* the caller's cap (self.substeps)
        // on low-speed scenes so the world can sleep cheaply; it never raises
        // above the cap (so explicit set_substeps(1) for CCD tests is kept).
        let lower = MIN_SUBSTEPS.min(self.substeps);
        let wanted = (max_speed * dt / SUB_DT_TARGET).ceil() as u32;
        wanted.clamp(lower, self.substeps)
    }

    /// Per-island iteration scaling: fast islands get the full budget,
    /// slow islands are solved cheaply. Derived from the same target sub-dt
    /// as `effective_substeps` so the two heuristics stay coherent.
    ///
    /// ponytail: 2 iters minimum for velocity, 1 for position — per-island
    /// sub-dt splitting (instead of iteration scaling) is the upgrade path
    /// if heterogeneous scenes still show solver jitter.
    #[allow(dead_code)]
    pub(super) fn adaptive_iters_for_island(
        &self,
        max_speed: f32,
        dt: f32,
        base_iters: u32,
    ) -> u32 {
        self.adaptive_iters_for_island_with_pen(max_speed, 0.0, dt, base_iters)
    }

    pub(super) fn adaptive_iters_for_island_with_pen(
        &self,
        max_speed: f32,
        max_pen: f32,
        dt: f32,
        base_iters: u32,
    ) -> u32 {
        const MIN_SUBSTEPS: u32 = 4;
        const SUB_DT_TARGET: f32 = 1.0 / 240.0;
        const PEN_SLOP: f32 = 0.01;
        // Keep at least 2 velocity / 1 position iteration so even resting
        // islands still correct residual penetration.
        let min_iters = if base_iters > 4 { 2 } else { 1 };
        let max_sub = self.substeps.max(1);
        let lower = MIN_SUBSTEPS.min(max_sub);
        let wanted_vel = (max_speed * dt / SUB_DT_TARGET).ceil() as u32;
        let wanted_pen = (max_pen / PEN_SLOP).ceil() as u32;
        let wanted = wanted_vel.max(wanted_pen).clamp(lower, max_sub);
        let scaled = ((wanted as f32 / max_sub as f32) * base_iters as f32).ceil() as u32;
        scaled.clamp(min_iters, base_iters)
    }

    /// Baumgarte positional-correction iterations per substep (default 4).
    pub fn set_position_iterations(&mut self, n: u32) {
        self.position_iterations = n;
    }

    /// CFM softness scale for the positional pass: 0 (default) = rigid,
    /// larger values spread corrections over more iterations for smoother
    /// but softer penetration recovery.
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

    /// Time-of-impact pass (G6, b3SolveContinuous analog): runs after the
    /// velocity solve, before positions move. Linear movers use conservative
    /// advancement; rotating boxes/capsules use the bounded angular sweep
    /// helper. The body is clamped to the first detected impact and flagged in
    /// `skip` so `integrate_positions` does not move it a second time. The
    /// angular path is deliberately a bounded approximation until an analytic
    /// swept-volume solver is available.
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
            debug_assert!(vec3_finite(disp), "velocity*sub_dt overflowed for body {h}");
            let Some(hit) = find_continuous_hit(&self.bodies, h, disp, sub_dt) else {
                continue;
            };
            let orientation = swept_orientation(&self.bodies[h], sub_dt, hit.fraction);
            let e = self.bodies[h]
                .restitution
                .min(self.bodies[hit.handle].restitution);
            let b = &mut self.bodies[h];
            // Back off a hair so the discrete narrow phase sees a clean
            // touching contact next substep, not a zero-gap flicker.
            b.position += disp * hit.fraction + hit.normal * 1e-3;
            b.orientation = orientation;
            skip[h] = true;
            if hit.angular {
                // The angular path has stopped the body at the first sampled
                // rotational impact. A later joint/contact pass may provide
                // a more precise angular response.
                b.angular_velocity = Vec3::ZERO;
            }
            let vn = b.velocity.dot(hit.normal);
            if vn < 0.0 {
                // Inelastic below the shared restitution threshold; a
                // genuine impact bounces (one-shot, like the discrete
                // restitution stage).
                let bounce = if vn < -1.0 { 1.0 + e } else { 1.0 };
                b.velocity -= hit.normal * (bounce * vn);
            }
        }
    }

    fn raycast_body(&self, ray: &Ray, handle: usize, max_dist: f32) -> Option<RaycastHit> {
        if max_dist.is_nan() || max_dist < 0.0 || !vec3_finite(ray.direction) {
            return None;
        }
        let body = &self.bodies[handle];
        let inverse = body.orientation.inverse();
        let origin = inverse * (ray.origin - body.position);
        let direction = inverse * ray.direction;
        let hit = match &body.shape {
            Shape::Sphere { radius } => {
                ray_sphere_hit(origin, direction, Vec3::ZERO, *radius, max_dist)
            }
            Shape::Box { half_extents } => ray_obb_hit(origin, direction, *half_extents, max_dist),
            Shape::Capsule {
                radius,
                half_height,
            } => ray_capsule_hit(origin, direction, *radius, *half_height, max_dist),
        }?;
        let (distance, local_normal) = hit;
        let point = ray.point_at(distance);
        let normal = (body.orientation * local_normal).normalize_or(Vec3::Y);
        Some(RaycastHit {
            handle,
            point,
            normal,
            distance,
        })
    }
}

/// Candidate returned by the linear or angular continuous collision query.
struct ContinuousHit {
    fraction: f32,
    normal: Vec3,
    handle: usize,
    angular: bool,
}

fn shape_min_dimension(shape: &Shape) -> f32 {
    match shape {
        Shape::Sphere { radius } => *radius,
        Shape::Box { half_extents } => half_extents.min_element(),
        Shape::Capsule { radius, .. } => *radius,
    }
}

fn shape_rotation_sensitive(shape: &Shape) -> bool {
    !matches!(shape, Shape::Sphere { .. })
}

/// Orientation at a fraction of the current substep's angular motion.
fn swept_orientation(body: &RigidBody, sub_dt: f32, fraction: f32) -> Quat {
    (Quat::from_scaled_axis(body.angular_velocity * (sub_dt * fraction)) * body.orientation)
        .normalize()
}

/// Exact shape distance at a pose on the combined linear/angular sweep.
fn swept_distance(
    body: &RigidBody,
    target: distance::ShapeRef<'_>,
    displacement: Vec3,
    sub_dt: f32,
    fraction: f32,
) -> distance::Distance {
    distance::shape_distance(
        distance::ShapeRef {
            shape: &body.shape,
            pos: body.position + displacement * fraction,
            rot: swept_orientation(body, sub_dt, fraction),
        },
        target,
    )
}

/// Conservative overlap predicate for a swept pose. OBB pairs use SAT because
/// the generic OBB distance oracle is unsigned while overlapping boxes need a
/// signed contact decision; the other pairs use their analytic signed distance.
fn swept_shape_overlaps(
    body: &RigidBody,
    target: distance::ShapeRef<'_>,
    displacement: Vec3,
    sub_dt: f32,
    fraction: f32,
) -> bool {
    let position = body.position + displacement * fraction;
    let orientation = swept_orientation(body, sub_dt, fraction);
    match (&body.shape, target.shape) {
        (
            Shape::Box {
                half_extents: half_a,
            },
            Shape::Box {
                half_extents: half_b,
            },
        ) => obb_sat(
            position,
            *half_a,
            orientation,
            target.pos,
            *half_b,
            target.rot,
            1e-5,
        )
        .is_some(),
        _ => {
            let distance = distance::shape_distance(
                distance::ShapeRef {
                    shape: &body.shape,
                    pos: position,
                    rot: orientation,
                },
                target,
            );
            distance.dist <= 1e-5
        }
    }
}

fn find_linear_continuous_hit(
    bodies: &[RigidBody],
    mover_index: usize,
    displacement: Vec3,
) -> Option<ContinuousHit> {
    let body = &bodies[mover_index];
    if body.is_trigger {
        return None;
    }
    let length = displacement.length();
    let min_dimension = shape_min_dimension(&body.shape);
    if length <= 0.5 * min_dimension {
        return None;
    }
    let mover_layer = body.collision_layer;
    let mover_mask = body.collision_mask;
    let mover = distance::ShapeRef {
        shape: &body.shape,
        pos: body.position,
        rot: body.orientation,
    };
    let targets = bodies
        .iter()
        .enumerate()
        .filter(move |&(handle, target)| {
            handle != mover_index
                && !target.is_trigger
                && mover_mask & target.collision_layer != 0
                && target.collision_mask & mover_layer != 0
        })
        .map(|(handle, target)| {
            (
                handle,
                distance::ShapeRef {
                    shape: &target.shape,
                    pos: target.position,
                    rot: target.orientation,
                },
            )
        });
    distance::cast_shape(mover, displacement, targets).map(|hit| ContinuousHit {
        fraction: (hit.t / length).clamp(0.0, 1.0),
        normal: hit.normal,
        handle: hit.handle,
        angular: false,
    })
}

/// Find the first sampled overlap fraction for one target and binary-search
/// that sample interval for a more accurate time of impact.
fn first_angular_overlap_fraction(
    body: &RigidBody,
    target: distance::ShapeRef<'_>,
    displacement: Vec3,
    sub_dt: f32,
    samples: u32,
) -> Option<f32> {
    if swept_shape_overlaps(body, target, displacement, sub_dt, 0.0) {
        return None;
    }
    let mut previous = 0.0;
    for sample in 1..=samples {
        let current = sample as f32 / samples as f32;
        if !swept_shape_overlaps(body, target, displacement, sub_dt, current) {
            previous = current;
            continue;
        }
        let mut low = previous;
        let mut high = current;
        for _ in 0..8 {
            let middle = (low + high) * 0.5;
            if swept_shape_overlaps(body, target, displacement, sub_dt, middle) {
                high = middle;
            } else {
                low = middle;
            }
        }
        return Some(high);
    }
    None
}

/// Sample the angular sweep at no more than five degrees per interval and
/// binary-search the first overlap. The path is a bounded CCD approximation:
/// exact shape distance/SAT decides each sampled pose, while a future
/// analytic swept-volume solver can remove the sampling limit.
fn find_angular_continuous_hit(
    bodies: &[RigidBody],
    mover_index: usize,
    displacement: Vec3,
    sub_dt: f32,
) -> Option<ContinuousHit> {
    let body = &bodies[mover_index];
    if body.is_trigger || !shape_rotation_sensitive(&body.shape) {
        return None;
    }
    const MAX_ANGLE_STEP: f32 = 5.0f32.to_radians();
    // Resting-contact jitter is handled by the discrete solver. Reserve the
    // angular CCD path for genuinely fast rotation (at least 15° per
    // substep), where tunneling is a meaningful risk.
    const MIN_ANGLE: f32 = 15.0f32.to_radians();
    let angle = (body.angular_velocity * sub_dt).length();
    if angle <= MIN_ANGLE {
        return None;
    }
    let samples = (angle / MAX_ANGLE_STEP).ceil().clamp(1.0, 128.0) as u32;
    let mover_layer = body.collision_layer;
    let mover_mask = body.collision_mask;
    let mut best = None;

    for (handle, target) in bodies.iter().enumerate() {
        if handle == mover_index
            || target.is_trigger
            || mover_mask & target.collision_layer == 0
            || target.collision_mask & mover_layer == 0
        {
            continue;
        }
        let target_ref = distance::ShapeRef {
            shape: &target.shape,
            pos: target.position,
            rot: target.orientation,
        };
        let Some(fraction) =
            first_angular_overlap_fraction(body, target_ref, displacement, sub_dt, samples)
        else {
            continue;
        };
        let distance = swept_distance(body, target_ref, displacement, sub_dt, fraction);
        let position = body.position + displacement * fraction;
        let fallback = (position - target.position).normalize_or(Vec3::Y);
        let normal = (distance.point_a - distance.point_b).normalize_or(fallback);
        let candidate = ContinuousHit {
            fraction,
            normal,
            handle,
            angular: true,
        };
        best = choose_continuous_hit(best, Some(candidate));
    }
    best
}

fn choose_continuous_hit(
    best: Option<ContinuousHit>,
    candidate: Option<ContinuousHit>,
) -> Option<ContinuousHit> {
    match (best, candidate) {
        (None, candidate) => candidate,
        (best, None) => best,
        (Some(best), Some(candidate)) => Some(if candidate.fraction < best.fraction {
            candidate
        } else {
            best
        }),
    }
}

/// Find the earliest linear or angular time of impact for one dynamic body.
fn find_continuous_hit(
    bodies: &[RigidBody],
    mover_index: usize,
    displacement: Vec3,
    sub_dt: f32,
) -> Option<ContinuousHit> {
    let linear = find_linear_continuous_hit(bodies, mover_index, displacement);
    let angular = find_angular_continuous_hit(bodies, mover_index, displacement, sub_dt);
    choose_continuous_hit(linear, angular)
}

/// Ray/sphere intersection in the shape's local frame. The returned normal is
/// also local so callers can rotate it back into world space.
fn ray_sphere_hit(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    radius: f32,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let a = direction.length_squared();
    if a <= 1e-12 {
        return None;
    }
    let offset = origin - center;
    let half_b = offset.dot(direction);
    let c = offset.length_squared() - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let mut distance = (-half_b - root) / a;
    if distance < 0.0 {
        distance = (-half_b + root) / a;
    }
    if distance < 0.0 || distance > max_dist {
        return None;
    }
    let point = origin + direction * distance;
    Some((distance, (point - center).normalize_or(Vec3::X)))
}

/// Mutable interval and normal state for a local-space OBB ray query.
struct RayObbState {
    near: f32,
    far: f32,
    near_normal: Vec3,
    far_normal: Vec3,
}

/// Update one slab of a local-space OBB ray intersection.
fn ray_obb_slab(
    origin: f32,
    direction: f32,
    minimum: f32,
    maximum: f32,
    axis: Vec3,
    state: &mut RayObbState,
) -> bool {
    if direction.abs() <= 1e-12 {
        return origin >= minimum && origin <= maximum;
    }
    let (entry, entry_normal, exit, exit_normal) = if direction > 0.0 {
        (
            (minimum - origin) / direction,
            -axis,
            (maximum - origin) / direction,
            axis,
        )
    } else {
        (
            (maximum - origin) / direction,
            axis,
            (minimum - origin) / direction,
            -axis,
        )
    };
    if entry > state.near {
        state.near = entry;
        state.near_normal = entry_normal;
    }
    if exit < state.far {
        state.far = exit;
        state.far_normal = exit_normal;
    }
    state.near <= state.far
}

/// Exact local-space ray/OBB intersection using a three-axis slab test.
fn ray_obb_hit(
    origin: Vec3,
    direction: Vec3,
    half_extents: Vec3,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    if direction.length_squared() <= 1e-12 {
        return None;
    }
    let mut state = RayObbState {
        near: f32::NEG_INFINITY,
        far: max_dist,
        near_normal: Vec3::ZERO,
        far_normal: Vec3::ZERO,
    };
    if !ray_obb_slab(
        origin.x,
        direction.x,
        -half_extents.x,
        half_extents.x,
        Vec3::X,
        &mut state,
    ) || !ray_obb_slab(
        origin.y,
        direction.y,
        -half_extents.y,
        half_extents.y,
        Vec3::Y,
        &mut state,
    ) || !ray_obb_slab(
        origin.z,
        direction.z,
        -half_extents.z,
        half_extents.z,
        Vec3::Z,
        &mut state,
    ) {
        return None;
    }
    if state.far < 0.0 || state.near > max_dist {
        return None;
    }
    if state.near >= 0.0 {
        Some((state.near, state.near_normal))
    } else {
        Some((state.far, state.far_normal))
    }
}

/// Keep the closest candidate hit in a local-space ray query.
fn keep_closest_hit(
    best: Option<(f32, Vec3)>,
    candidate: Option<(f32, Vec3)>,
) -> Option<(f32, Vec3)> {
    match (best, candidate) {
        (None, candidate) => candidate,
        (best, None) => best,
        (Some(best), Some(candidate)) => Some(if candidate.0 < best.0 {
            candidate
        } else {
            best
        }),
    }
}

/// Exact intersection with the cylindrical side of a local Y-axis capsule.
fn ray_capsule_cylinder_hit(
    origin: Vec3,
    direction: Vec3,
    radius: f32,
    half_height: f32,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let a = direction.x * direction.x + direction.z * direction.z;
    if a <= 1e-12 {
        return None;
    }
    let half_b = origin.x * direction.x + origin.z * direction.z;
    let c = origin.x * origin.x + origin.z * origin.z - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let denominator = a;
    let roots = [
        (-half_b - root) / denominator,
        (-half_b + root) / denominator,
    ];
    let mut best = None;
    for distance in roots {
        if distance < 0.0 || distance > max_dist {
            continue;
        }
        let point = origin + direction * distance;
        if point.y < -half_height || point.y > half_height {
            continue;
        }
        let normal = Vec3::new(point.x, 0.0, point.z).normalize_or(Vec3::X);
        best = keep_closest_hit(best, Some((distance, normal)));
    }
    best
}

/// Exact local-space ray/capsule intersection: finite cylinder side plus its
/// two spherical caps. The nearest valid feature is returned.
fn ray_capsule_hit(
    origin: Vec3,
    direction: Vec3,
    radius: f32,
    half_height: f32,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let mut best = ray_capsule_cylinder_hit(origin, direction, radius, half_height, max_dist);
    for center in [
        Vec3::new(0.0, -half_height, 0.0),
        Vec3::new(0.0, half_height, 0.0),
    ] {
        best = keep_closest_hit(
            best,
            ray_sphere_hit(origin, direction, center, radius, max_dist),
        );
    }
    best
}

impl PhysicsEngine for BuiltinPhysicsEngine {
    fn step(&mut self, dt: f32) {
        // G7: a fully sleeping world with no trigger state cannot change —
        // skip the whole substep loop (broadphase re-sort included) instead
        // of paying to rediscover that nothing moves. Trigger-only worlds
        // still run the overlap reconciliation pass below.
        let has_awake_dynamic = self
            .bodies
            .iter()
            .enumerate()
            .any(|(h, b)| b.body_type == BodyType::Dynamic && !self.asleep[h]);
        let has_trigger = self.bodies.iter().any(|body| body.is_trigger);
        if !has_awake_dynamic && !has_trigger && self.trigger_pairs.is_empty() {
            // Fully sleeping world: no substep loop ran, so no phase work
            // happened this step.
            self.last_step_timing = StepTiming::default();
            return;
        }
        let eff_substeps = self.effective_substeps(dt);
        let sub_dt = dt / eff_substeps as f32;
        let mut last_manifolds = Vec::new();
        let mut timing = StepTiming {
            substeps: eff_substeps,
            ..StepTiming::default()
        };
        for s in 0..eff_substeps {
            // Box3D stage order: solve velocities BEFORE moving positions, so
            // a resting contact kills gravity's velocity gain in the same
            // substep instead of letting the body free-fall and snapping it
            // back (the snap is an inelastic collision and bleeds energy).
            self.integrate_velocities(sub_dt);
            let t0 = Instant::now();
            self.broadphase.update(&self.bodies, sub_dt);
            timing.broad_phase_ms += t0.elapsed().as_secs_f64() * 1000.0;
            // Jointed pairs never collide: their parts legitimately sweep
            // through each other's space (a hinge pin passes through the arm).
            let t0 = Instant::now();
            let manifolds = if self.joint_pairs.is_empty() {
                detect_collisions(&self.bodies, self.broadphase.active(), &self.asleep, sub_dt)
            } else {
                let pairs: Vec<(usize, usize)> = self
                    .broadphase
                    .active()
                    .iter()
                    .copied()
                    .filter(|p| !self.joint_pairs.contains(p))
                    .collect();
                detect_collisions(&self.bodies, &pairs, &self.asleep, sub_dt)
            };
            timing.narrow_phase_ms += t0.elapsed().as_secs_f64() * 1000.0;
            // Restitution is one-shot per step, evaluated on the first substep.
            let t0 = Instant::now();
            let mut islands = self.solve_contacts_velocity(&manifolds, s == 0, sub_dt, dt);
            self.solve_joints_velocity();
            // Continuous pass on the solver-adjusted velocities: clamp fast
            // movers to their first impact and keep them there this substep.
            let mut clamped = vec![false; self.bodies.len()];
            self.solve_continuous(sub_dt, &mut clamped);
            self.integrate_positions(sub_dt, &clamped);
            self.solve_contacts_position(&mut islands, dt);
            self.solve_joints_position();
            timing.solver_ms += t0.elapsed().as_secs_f64() * 1000.0;
            last_manifolds = manifolds;
        }
        // Diagnostics: contact-manifold partners per body from the last
        // substep (drives sleep/island debugging; tiny flat copy).
        self.debug_pairs.clear();
        self.debug_pairs
            .extend(last_manifolds.iter().map(|m| (m.body_a, m.body_b)));
        self.rebuild_islands(&last_manifolds);
        self.update_sleep(dt);
        self.last_step_timing = timing;

        // Rebuild the broadphase at the completed poses so trigger events
        // describe the state visible after this whole physics step, not the
        // state from before the final substep's integration.
        self.broadphase.update(&self.bodies, 0.0);
        let current_triggers = detect_trigger_overlaps(&self.bodies, self.broadphase.active());
        let previous_triggers = std::mem::take(&mut self.trigger_pairs);
        self.trigger_pairs = update_trigger_events(
            &previous_triggers,
            current_triggers,
            &mut self.trigger_events,
        );
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
            let previous_triggers = std::mem::take(&mut self.trigger_pairs);
            let mut removed_triggers = Vec::new();
            let mut remapped_triggers = HashSet::new();
            for (body_a, body_b) in previous_triggers {
                if body_a == handle || body_b == handle {
                    removed_triggers.push((body_a, body_b));
                    continue;
                }
                let map = |body: usize| if body == last { handle } else { body };
                let a = map(body_a);
                let b = map(body_b);
                remapped_triggers.insert((a.min(b), a.max(b)));
            }
            removed_triggers.sort_unstable();
            for (body_a, body_b) in removed_triggers {
                self.trigger_events.push(TriggerEvent {
                    body_a,
                    body_b,
                    kind: TriggerEventKind::Exited,
                });
            }
            self.trigger_pairs = remapped_triggers;
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

    fn drain_trigger_events(&mut self) -> Vec<TriggerEvent> {
        std::mem::take(&mut self.trigger_events)
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
    fn broadphase_backend_can_be_selected_explicitly() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        assert_eq!(physics.broadphase_kind(), BroadPhaseKind::UniformGrid);
        physics.set_broadphase(BroadPhaseKind::UniformGrid);
        assert_eq!(physics.broadphase_kind(), BroadPhaseKind::UniformGrid);
        physics.set_uniform_grid_cell_size(1.0);
        assert_eq!(physics.broadphase_kind(), BroadPhaseKind::UniformGrid);
        physics.set_broadphase(BroadPhaseKind::SweepAndPrune);
        assert_eq!(physics.broadphase_kind(), BroadPhaseKind::SweepAndPrune);
    }

    #[test]
    fn adaptive_substeps_scale_with_body_speed() {
        // A fast body needs the full substep cap; a resting scene drops to the
        // minimum so it can sleep cheaply.
        let mut fast = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        fast.add_body(RigidBody::new_box(
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(10.0, 0.5, 10.0),
            0.0,
        ));
        let mut ball = RigidBody::new_box(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
        ball.velocity = Vec3::new(0.0, -40.0, 0.0);
        fast.add_body(ball);
        fast.step(1.0 / 60.0);
        assert_eq!(fast.step_timing().substeps, 12, "fast body uses full cap");

        // Resting grid: after settling, velocities are ~0 -> minimum substeps.
        let mut rest = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        rest.add_body(RigidBody::new_box(
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(100.0, 0.5, 100.0),
            0.0,
        ));
        for i in 0..4 {
            rest.add_body(RigidBody::new_box(
                Vec3::new(0.0, 0.4 + i as f32 * 0.82, 0.0),
                Vec3::splat(0.4),
                1.0,
            ));
        }
        for _ in 0..240 {
            rest.step(1.0 / 60.0);
        }
        assert!(
            rest.step_timing().substeps < 12,
            "resting scene adapts below the 12 cap (got {})",
            rest.step_timing().substeps
        );
    }

    #[test]
    fn per_island_iters_scale_with_speed() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let dt = 1.0 / 60.0;
        // slow island → minimal iters (3 vel from 4/12*8), fast → full cap
        assert_eq!(physics.adaptive_iters_for_island(0.0, dt, 8), 3);
        assert_eq!(physics.adaptive_iters_for_island(0.1, dt, 8), 3);
        assert!(physics.adaptive_iters_for_island(2.0, dt, 8) > 3);
        assert!(physics.adaptive_iters_for_island(2.0, dt, 8) < 8);
        assert_eq!(physics.adaptive_iters_for_island(40.0, dt, 8), 8);
        assert_eq!(physics.adaptive_iters_for_island(40.0, dt, 4), 4);
        // penetration drives iters even when speed is zero
        assert_eq!(
            physics.adaptive_iters_for_island_with_pen(0.0, 0.12, dt, 8),
            8
        );
        assert!(physics.adaptive_iters_for_island_with_pen(0.0, 0.06, dt, 8) > 3);
        // respects substeps cap — still returns scaled within base
        physics.set_substeps(1);
        assert_eq!(physics.adaptive_iters_for_island(40.0, dt, 8), 8);
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
    fn collision_filter_blocks_broadphase_and_narrowphase() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let a = physics.add_body(
            RigidBody::new_sphere(Vec3::new(-0.4, 0.0, 0.0), 0.5, 1.0)
                .with_collision_filter(0b0001, 0b0010),
        );
        let b = physics.add_body(
            RigidBody::new_sphere(Vec3::new(0.4, 0.0, 0.0), 0.5, 1.0)
                .with_collision_filter(0b0010, 0b0100),
        );

        physics.step(1.0 / 60.0);

        assert_eq!(physics.debug_contact_count(a), 0);
        assert_eq!(physics.debug_contact_count(b), 0);
        assert_eq!(physics.get_body(a).unwrap().position.x, -0.4);
        assert_eq!(physics.get_body(b).unwrap().position.x, 0.4);
    }

    #[test]
    fn collision_filter_allows_mutual_layer_match() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let a = physics.add_body(
            RigidBody::new_sphere(Vec3::new(-0.4, 0.0, 0.0), 0.5, 1.0)
                .with_collision_filter(0b0001, 0b0010),
        );
        let b = physics.add_body(
            RigidBody::new_sphere(Vec3::new(0.4, 0.0, 0.0), 0.5, 1.0)
                .with_collision_filter(0b0010, 0b0001),
        );

        physics.step(1.0 / 60.0);

        assert!(physics.debug_contact_count(a) > 0);
        assert!(physics.debug_contact_count(b) > 0);
    }

    #[test]
    fn collision_filter_applies_to_continuous_cast() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.add_body(
            RigidBody::new_box(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.05, 10.0), 0.0)
                .with_collision_filter(0b0010, 0b0010),
        );
        let bullet = physics.add_body(
            RigidBody::new_sphere(Vec3::new(0.0, 3.0, 0.0), 0.1, 1.0)
                .with_collision_filter(0b0001, 0b0001),
        );
        physics.get_body_mut(bullet).unwrap().velocity = Vec3::new(0.0, -80.0, 0.0);

        for _ in 0..60 {
            physics.step(1.0 / 60.0);
        }

        assert_eq!(physics.debug_contact_count(bullet), 0);
        assert!(
            physics.get_body(bullet).unwrap().position.y < -0.1,
            "filtered bullet should pass through the floor"
        );
    }

    #[test]
    fn trigger_emits_enter_and_exit_without_solving_contact() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let mut trigger_body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 0.0);
        trigger_body.set_trigger(true);
        let trigger = physics.add_body(trigger_body);
        let mover = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.8, 0.0), 0.5, 1.0));

        physics.step(1.0 / 60.0);
        assert_eq!(
            physics.drain_trigger_events(),
            vec![TriggerEvent {
                body_a: trigger.min(mover),
                body_b: trigger.max(mover),
                kind: TriggerEventKind::Entered,
            }]
        );
        assert_eq!(physics.debug_contact_count(mover), 0);
        assert_eq!(
            physics.get_body(mover).unwrap().position,
            Vec3::new(0.0, 0.8, 0.0)
        );

        physics.step(1.0 / 60.0);
        assert!(physics.drain_trigger_events().is_empty());

        physics.get_body_mut(mover).unwrap().position = Vec3::new(0.0, 3.0, 0.0);
        physics.step(1.0 / 60.0);
        assert_eq!(
            physics.drain_trigger_events(),
            vec![TriggerEvent {
                body_a: trigger.min(mover),
                body_b: trigger.max(mover),
                kind: TriggerEventKind::Exited,
            }]
        );
    }

    #[test]
    fn removing_trigger_body_queues_exit_and_clears_pair_state() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let mut trigger_body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 0.0);
        trigger_body.set_trigger(true);
        let trigger = physics.add_body(trigger_body);
        let first = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0));
        let second = physics.add_body(RigidBody::new_sphere(Vec3::new(5.0, 0.0, 0.0), 0.5, 1.0));
        physics.step(1.0 / 60.0);
        assert_eq!(physics.drain_trigger_events().len(), 1);

        physics.remove_body(trigger);
        assert_eq!(
            physics.drain_trigger_events(),
            vec![TriggerEvent {
                body_a: trigger,
                body_b: first,
                kind: TriggerEventKind::Exited,
            }]
        );
        physics.get_body_mut(second - 1).unwrap().position = Vec3::ZERO;
        physics.step(1.0 / 60.0);
        let events = physics.drain_trigger_events();
        assert!(events.is_empty());
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
    fn raycast_obb_uses_exact_surface_and_normal() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        physics.add_body(
            RigidBody::new_box(Vec3::ZERO, Vec3::new(1.0, 0.25, 0.25), 0.0)
                .with_orientation(rotation),
        );

        let ray = Ray::new(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = physics
            .raycast(ray, 10.0)
            .expect("ray must hit the rotated box");
        let expected_distance = 2.0 - 0.25 * std::f32::consts::SQRT_2;
        let expected_normal = rotation * Vec3::Y;
        assert!((hit.distance - expected_distance).abs() < 1e-4);
        assert!(hit.normal.dot(expected_normal) > 0.999);
        assert!((hit.point - ray.point_at(expected_distance)).length() < 1e-4);
    }

    #[test]
    fn raycast_capsule_uses_spherical_cap_normal() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.add_body(RigidBody::new_capsule(Vec3::ZERO, 0.5, 1.0, 0.0));

        let ray = Ray::new(Vec3::new(0.4, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = physics
            .raycast(ray, 10.0)
            .expect("ray must hit the capsule cap");
        let expected_distance = 2.0 - (1.0 + 0.3);
        let expected_normal = Vec3::new(0.8, 0.6, 0.0);
        assert!((hit.distance - expected_distance).abs() < 1e-4);
        assert!(hit.normal.dot(expected_normal) > 0.999);
        assert!((hit.point - Vec3::new(0.4, 1.3, 0.0)).length() < 1e-4);
    }

    #[test]
    fn raycast_ignores_zero_length_rays() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 1.0, 0.0));
        assert!(
            physics
                .raycast(Ray::new(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO), 10.0)
                .is_none()
        );
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
        assert!(
            (half_z - 1.0).abs() < 1e-3,
            "OBB->AABB z-extent, got {half_z}"
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
    fn angular_sweep_finds_rotating_box_impact() {
        let dt = 1.0 / 60.0;
        let mut mover = RigidBody::new_box(Vec3::ZERO, Vec3::new(1.5, 0.1, 0.1), 1.0);
        mover.angular_velocity = Vec3::Z * (std::f32::consts::FRAC_PI_2 / dt);
        let target = RigidBody::new_box(Vec3::new(0.0, 1.1, 0.0), Vec3::new(0.2, 0.05, 0.2), 0.0);
        let bodies = [mover, target];

        let hit = find_angular_continuous_hit(&bodies, 0, Vec3::ZERO, dt)
            .expect("angular sweep must find the rotating box impact");
        assert_eq!(hit.handle, 1);
        assert!(hit.angular);
        assert!(hit.fraction > 0.0 && hit.fraction < 1.0);
    }

    #[test]
    fn angular_continuous_motion_stops_at_first_impact() {
        let dt = 1.0 / 60.0;
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.set_substeps(1);
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 1.1, 0.0),
            Vec3::new(0.2, 0.05, 0.2),
            0.0,
        ));
        let mover = physics.add_body(RigidBody::new_box(
            Vec3::ZERO,
            Vec3::new(1.5, 0.1, 0.1),
            1.0,
        ));
        physics.get_body_mut(mover).unwrap().angular_velocity =
            Vec3::Z * (std::f32::consts::FRAC_PI_2 / dt);

        physics.step(dt);

        let body = physics.get_body(mover).expect("mover remains alive");
        assert!(
            body.angular_velocity.length() < 1e-5,
            "angular CCD must stop the rotating body"
        );
        assert_ne!(
            body.orientation,
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            "rotating body must not jump through the target"
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
        let iw_i = rot_inertia(
            bodies[0].orientation,
            Vec3::new(
                inv_inertia_axis(bodies[0].inertia.x),
                inv_inertia_axis(bodies[0].inertia.y),
                inv_inertia_axis(bodies[0].inertia.z),
            ),
        );
        let iw_j = rot_inertia(
            bodies[1].orientation,
            Vec3::new(
                inv_inertia_axis(bodies[1].inertia.x),
                inv_inertia_axis(bodies[1].inertia.y),
                inv_inertia_axis(bodies[1].inertia.z),
            ),
        );
        let oracle =
            bodies[0].inv_mass + bodies[1].inv_mass + ra_d.dot(iw_i * ra_d) + rb_d.dot(iw_j * rb_d);

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
            a[0][0], a[1][0], a[2][0], a[3][0], a[0][1], a[1][1], a[2][1], a[3][1], a[0][2],
            a[1][2], a[2][2], a[3][2], a[0][3], a[1][3], a[2][3], a[3][3],
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
        for (k, impulse) in acc.iter().enumerate().take(count) {
            assert!(
                *impulse >= -1e-6,
                "accumulated impulse {} negative: {}",
                k,
                impulse
            );
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
