//! Physics engine trait and the builtin CPU implementation.
//!
//! [`PhysicsEngine`] defines a single simulation step (broadphase →
//! narrowphase → island partitioning → substepped contact/joint solving →
//! integration; `dt` must be positive and finite) plus body/joint
//! management and ray/shape cast queries. [`BuiltinPhysicsEngine`] is the
//! reference implementation: parallelized with rayon, with optional GPU
//! contact solving behind the `gpu` feature.

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::sync::Mutex;
use std::time::Instant;

use dashmap::DashMap;
use ornis_schedule::run_levels;

use glam::{Quat, Vec3};

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::broadphase::{
    BroadPhase, BroadPhaseBackend, BroadPhaseKind, BroadPhaseStats, StepBudget, StepTiming,
};
use crate::distance;
#[cfg(feature = "gpu")]
use crate::gpu::WgpuContactSolver;
use crate::joint::{Joint, JointHandle, JointKind};
use crate::math::{Ray, RaycastHit};
use crate::shape::Shape;
use crate::trigger::{
    CONTACT_BEGIN_SLOP, ContactEvent, ContactEventKind, TriggerEvent, TriggerEventKind,
};
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
    /// Drain solid-contact begin/end/hit transitions produced by completed
    /// steps (Box3D `b3ContactEvents` parity). Empty by default; the builtin
    /// engine reports them in deterministic pair order.
    fn drain_contact_events(&mut self) -> Vec<ContactEvent> {
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
    previous: &FxHashSet<(usize, usize)>,
    current: Vec<(usize, usize)>,
    events: &mut Vec<TriggerEvent>,
) -> FxHashSet<(usize, usize)> {
    let current_set: FxHashSet<(usize, usize)> = current.into_iter().collect();
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
    // Contact point: the sphere's surface point pushed halfway into the
    // overlap (same convention as `sphere_vs_sphere`). The old code used
    // the midpoint of (center, closest box point), which sits at half the
    // radius depth for a touching sphere — halving every friction lever
    // and torque arm. Symptom (measured): a rolling ball converged to a
    // phantom "half-rolling" equilibrium v = ω·r/2 with live slip, because
    // the solver saw zero slip at the half-depth point.
    let dir = delta / dist; // box frame, sphere center toward box surface
    let contact_point = local + dir * (sphere_radius + penetration * 0.5);
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

/// Box vs capsule via the shared analytic `shape_distance` (G6).
/// Thin wrapper: converts the exact distance witness into a speculative
/// contact (`dist <= margin`), keeping the narrowphase zero-alloc.
#[allow(clippy::too_many_arguments)]
fn box_vs_capsule(
    box_pos: Vec3,
    half_extents: Vec3,
    box_rot: Quat,
    cap_pos: Vec3,
    cap_radius: f32,
    cap_half_height: f32,
    cap_rot: Quat,
    margin: f32,
) -> Option<Contact> {
    let d = crate::distance::shape_distance(
        crate::distance::ShapeRef {
            shape: &Shape::Box { half_extents },
            pos: box_pos,
            rot: box_rot,
        },
        crate::distance::ShapeRef {
            shape: &Shape::Capsule {
                radius: cap_radius,
                half_height: cap_half_height,
            },
            pos: cap_pos,
            rot: cap_rot,
        },
    );
    if d.dist > margin {
        return None;
    }
    let ab = d.point_b - d.point_a;
    let len = ab.length();
    if len < 1e-10 {
        return None;
    }
    let normal = ab / len; // box → capsule
    let penetration = -d.dist;
    let contact_point = (d.point_a + d.point_b) * 0.5;
    Some(Contact {
        normal,
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
#[allow(dead_code)]
fn detect_collisions(
    bodies: &[RigidBody],
    active: &[(usize, usize)],
    asleep: &[bool],
    sub_dt: f32,
) -> Vec<Manifold> {
    let mut out = Vec::new();
    let mut pool = NarrowShardPool::default();
    detect_collisions_into(
        bodies, active, asleep, sub_dt, &mut out, None, 0, None, &mut pool,
    );
    out
}

/// Base speculative margin (m): also the AABB inflation used by the
/// broadphase, so pairs within it are guaranteed to reach narrow phase.
const SPEC_BASE: f32 = 0.05;

/// Shard count rule for scheduler-dispatched narrowphase: enough coarse
/// tasks to feed every worker without starving (the spike showed 8 shards
/// on 8 threads ~50% slower than flat rayon from imbalance, while 32
/// shards ran ~20% faster). Order-preserving concat keeps results
/// deterministic for any shard count.
fn narrow_shard_count(pairs: usize) -> usize {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (threads * 4).clamp(4, 64).min(pairs.max(1))
}

/// Pooled narrowphase shard buffers for scheduler dispatch: one uncontended
/// `Mutex<Vec>` per shard plus the cached single-group level plan. Owned by
/// the engine and passed down like the other scratch buffers, so the hot
/// loop never allocates. Resized only when the shard count moves.
#[derive(Debug, Default)]
struct NarrowShardPool {
    bufs: Vec<Mutex<Vec<Manifold>>>,
    level: Vec<Vec<usize>>,
}

impl NarrowShardPool {
    fn ensure(&mut self, shards: usize) {
        if self.bufs.len() != shards {
            self.bufs.clear();
            for _ in 0..shards {
                self.bufs.push(Mutex::new(Vec::new()));
            }
            self.level = vec![(0..shards).collect()];
        }
        for b in &self.bufs {
            b.lock().unwrap().clear();
        }
    }
}

/// Per-pair narrowphase kernel shared by the parallel and sequential paths:
/// filters, speculative margin, all 9 shape combos with the unified SAT-cache
/// gate, and canonical body ids on the manifold. Pure over shared borrows
/// (`sat_cache` is lock-free), so any scheduler can shard it — this is the
/// unit the scheduler spike compares against rayon.
#[allow(clippy::too_many_arguments)]
fn narrow_pair(
    bodies: &[RigidBody],
    asleep: &[bool],
    i: usize,
    j: usize,
    body_required: Option<&[u32]>,
    cur_substep: u32,
    sub_dt: f32,
    sat_cache: Option<&SatCache>,
) -> Option<Manifold> {
    let a = &bodies[i];
    let b = &bodies[j];
    if !a.can_collide_with(b) || a.is_trigger || b.is_trigger {
        return None;
    }
    if a.body_type == BodyType::Static && b.body_type == BodyType::Static {
        return None;
    }
    // G7: both-asleep pairs are frozen in place — their relative geometry
    // cannot change, so re-running the narrow phase (SAT!) per substep is
    // pure waste. On a settled scene this IS the frame cost. The island
    // graph keeps them composed via the frozen-asleep union instead.
    if asleep[i] && asleep[j] {
        return None;
    }
    // B: per-body substeps — slow pairs only needed for first
    // MIN substeps; fast pairs need all. Skip extra substeps.
    if let Some(req) = body_required
        && req[i].max(req[j]) <= cur_substep
    {
        return None;
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
        (&Shape::Box { half_extents: ha }, &Shape::Box { half_extents: hb }) => {
            // SAT cache is lock-free (DashMap): shared by both paths; hits
            // reuse the axis only, contacts rebuild from live geometry.
            let use_sat = sat_cache.is_some()
                && cur_substep == 0
                && rel_speed <= 0.5
                && a.angular_velocity.length_squared() <= 0.25
                && b.angular_velocity.length_squared() <= 0.25;
            if use_sat {
                box_manifold_cached(
                    a.position,
                    ha,
                    a.orientation,
                    b.position,
                    hb,
                    b.orientation,
                    margin,
                    Some((i, j)),
                    sat_cache,
                    true,
                )
            } else {
                box_manifold(
                    a.position,
                    ha,
                    a.orientation,
                    b.position,
                    hb,
                    b.orientation,
                    margin,
                )
            }
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
        ) => sphere_vs_capsule(b.position, r, a.position, cr, hh, a.orientation, margin).map(|c| {
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
        (
            &Shape::Box { half_extents: ha },
            &Shape::Capsule {
                radius: cr,
                half_height: hh,
            },
        ) => box_vs_capsule(
            a.position,
            ha,
            a.orientation,
            b.position,
            cr,
            hh,
            b.orientation,
            margin,
        )
        .map(|c| Manifold::single(i, j, c)),
        (
            &Shape::Capsule {
                radius: cr,
                half_height: hh,
            },
            &Shape::Box { half_extents: ha },
        ) => box_vs_capsule(
            b.position,
            ha,
            b.orientation,
            a.position,
            cr,
            hh,
            a.orientation,
            margin,
        )
        .map(|c| {
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
    };
    manifold.map(|mut m| {
        m.body_a = i;
        m.body_b = j;
        m
    })
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::collapsible_if)]
fn detect_collisions_into(
    bodies: &[RigidBody],
    active: &[(usize, usize)],
    asleep: &[bool],
    sub_dt: f32,
    out: &mut Vec<Manifold>,
    body_required: Option<&[u32]>,
    cur_substep: u32,
    sat_cache: Option<&SatCache>,
    pool: &mut NarrowShardPool,
) {
    out.clear();
    // Parallel narrowphase for large candidate sets: SAT/box_manifold is heavy,
    // and bodies/asleep are read-only. Threshold keeps small scenes sequential.
    if active.len() > 256 {
        // Scheduler dispatch (one level, K coarse shards): same kernel, same
        // order-preserving concat as the flat rayon path it replaces, so
        // results are identical for any shard count. Shard buffers come from
        // the engine-owned pool — no allocation on the hot path.
        let shards = narrow_shard_count(active.len());
        pool.ensure(shards);
        let bufs = &pool.bufs;
        run_levels(&pool.level, shards, true, |shard| {
            let lo = shard * active.len() / shards;
            let hi = (shard + 1) * active.len() / shards;
            let mut guard = bufs[shard].lock().unwrap();
            for &(i, j) in &active[lo..hi] {
                if let Some(m) = narrow_pair(
                    bodies,
                    asleep,
                    i,
                    j,
                    body_required,
                    cur_substep,
                    sub_dt,
                    sat_cache,
                ) {
                    guard.push(m);
                }
            }
        });
        out.reserve(active.len());
        for b in &pool.bufs {
            out.extend(b.lock().unwrap().drain(..));
        }
        return;
    }
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
        if let Some(req) = body_required
            && req[i].max(req[j]) <= cur_substep
        {
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
            (&Shape::Box { half_extents: ha }, &Shape::Box { half_extents: hb }) => {
                // SAT cache is lock-free (DashMap): shared by both paths; hits
                // reuse the axis only, contacts rebuild from live geometry.
                let use_sat = sat_cache.is_some()
                    && cur_substep == 0
                    && (a.velocity - b.velocity).length() <= 0.5
                    && a.angular_velocity.length_squared() <= 0.25
                    && b.angular_velocity.length_squared() <= 0.25;
                if use_sat {
                    box_manifold_cached(
                        a.position,
                        ha,
                        a.orientation,
                        b.position,
                        hb,
                        b.orientation,
                        margin,
                        Some((i, j)),
                        sat_cache,
                        true,
                    )
                } else {
                    box_manifold(
                        a.position,
                        ha,
                        a.orientation,
                        b.position,
                        hb,
                        b.orientation,
                        margin,
                    )
                }
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
            (
                &Shape::Box { half_extents: ha },
                &Shape::Capsule {
                    radius: cr,
                    half_height: hh,
                },
            ) => box_vs_capsule(
                a.position,
                ha,
                a.orientation,
                b.position,
                cr,
                hh,
                b.orientation,
                margin,
            )
            .map(|c| Manifold::single(i, j, c)),
            (
                &Shape::Capsule {
                    radius: cr,
                    half_height: hh,
                },
                &Shape::Box { half_extents: ha },
            ) => box_vs_capsule(
                b.position,
                ha,
                b.orientation,
                a.position,
                cr,
                hh,
                a.orientation,
                margin,
            )
            .map(|c| {
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
        };

        if let Some(mut m) = manifold {
            m.body_a = i;
            m.body_b = j;
            out.push(m);
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
type WarmCache = FxHashMap<(usize, usize), ([WarmPoint; 4], usize)>;

/// Narrowphase cache hit test: positions within 0.1mm, rotations within ~0.06°, margin within 0.1mm.
fn narrow_cache_hit(entry: &NarrowCacheEntry, a: &RigidBody, b: &RigidBody, margin: f32) -> bool {
    const POS_EPS_SQ: f32 = 1e-8; // (1e-4)^2
    const ROT_DOT_MIN: f32 = 0.999_999;
    const MARGIN_EPS: f32 = 1e-4;
    if (entry.margin - margin).abs() > MARGIN_EPS {
        return false;
    }
    if (entry.pos_a - a.position).length_squared() > POS_EPS_SQ {
        return false;
    }
    if (entry.pos_b - b.position).length_squared() > POS_EPS_SQ {
        return false;
    }
    if entry.rot_a.dot(a.orientation).abs() < ROT_DOT_MIN {
        return false;
    }
    if entry.rot_b.dot(b.orientation).abs() < ROT_DOT_MIN {
        return false;
    }
    true
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn sat_cache_hit(
    entry: &SatCacheEntry,
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    margin: f32,
) -> bool {
    // Widened tolerances: a hit reuses only the separating axis, while
    // `box_manifold_cached` rebuilds contact points from live geometry, so
    // ~1 mm / ~0.26 deg of pose drift cannot inject stale positions.
    const POS_EPS_SQ: f32 = 1e-6;
    const HALF_EPS_SQ: f32 = 1e-6;
    const ROT_DOT_MIN: f32 = 0.999_99;
    const MARGIN_EPS: f32 = 5e-4;
    if (entry.margin - margin).abs() > MARGIN_EPS {
        return false;
    }
    if (entry.pos_a - pos_a).length_squared() > POS_EPS_SQ {
        return false;
    }
    if (entry.pos_b - pos_b).length_squared() > POS_EPS_SQ {
        return false;
    }
    if (entry.half_a - half_a).length_squared() > HALF_EPS_SQ {
        return false;
    }
    if (entry.half_b - half_b).length_squared() > HALF_EPS_SQ {
        return false;
    }
    if entry.rot_a.dot(rot_a).abs() < ROT_DOT_MIN {
        return false;
    }
    if entry.rot_b.dot(rot_b).abs() < ROT_DOT_MIN {
        return false;
    }
    true
}

#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::collapsible_if)]
fn obb_sat_cached(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    margin: f32,
    key: Option<(usize, usize)>,
    sat_cache: Option<&SatCache>,
    slow_only: bool,
) -> Option<(Vec3, f32)> {
    if slow_only {
        if let (Some(k), Some(cache)) = (key, sat_cache) {
            if let Some(entry) = cache.get(&k) {
                if sat_cache_hit(&entry, pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin) {
                    return entry.result;
                }
            }
        }
    }
    let res = obb_sat(pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin);
    if slow_only {
        if let (Some(k), Some(cache)) = (key, sat_cache) {
            cache.insert(
                k,
                SatCacheEntry {
                    pos_a,
                    half_a,
                    rot_a,
                    pos_b,
                    half_b,
                    rot_b,
                    margin,
                    result: res,
                },
            );
        }
    }
    res
}

#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::collapsible_if)]
fn box_manifold_cached(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    margin: f32,
    key: Option<(usize, usize)>,
    sat_cache: Option<&SatCache>,
    slow_only: bool,
) -> Option<Manifold> {
    let (n, _pen) = obb_sat_cached(
        pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin, key, sat_cache, slow_only,
    )?;
    // Reuse normal from SAT — remainder is face-corner collection (cheap vs 15-axis SAT).
    let aa = [rot_a * Vec3::X, rot_a * Vec3::Y, rot_a * Vec3::Z];
    let ba = [rot_b * Vec3::X, rot_b * Vec3::Y, rot_b * Vec3::Z];
    let hwn_a = half_a.x * aa[0].dot(n).abs()
        + half_a.y * aa[1].dot(n).abs()
        + half_a.z * aa[2].dot(n).abs();
    let hwn_b = half_b.x * ba[0].dot(n).abs()
        + half_b.y * ba[1].dot(n).abs()
        + half_b.z * ba[2].dot(n).abs();
    let depth_tol = margin;
    let tangent_slack = 0.05;
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
    let mut uniq = dedupe_contact_points(cand, n);
    uniq.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut points = [ManifoldPoint {
        world_point: Vec3::ZERO,
        penetration: 0.0,
    }; 4];
    let mut count = 0;
    for (p, d) in uniq.into_iter().take(4) {
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

#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::collapsible_if)]
fn detect_collisions_into_with_cache(
    bodies: &[RigidBody],
    active: &[(usize, usize)],
    asleep: &[bool],
    sub_dt: f32,
    out: &mut Vec<Manifold>,
    body_required: Option<&[u32]>,
    cur_substep: u32,
    cache: &mut FxHashMap<(usize, usize), NarrowCacheEntry>,
    sat_cache: Option<&SatCache>,
    pool: &mut NarrowShardPool,
) {
    // Only cache the first substep: later substeps are filtered to fast bodies,
    // hit rate is near zero and the HashMap overhead dominates.
    if cur_substep != 0 {
        detect_collisions_into(
            bodies,
            active,
            asleep,
            sub_dt,
            out,
            body_required,
            cur_substep,
            sat_cache,
            pool,
        );
        return;
    }
    out.clear();
    // Evict stale entries where bodies were removed.
    cache.retain(|(a, b), _| *a < bodies.len() && *b < bodies.len());
    if let Some(sc) = sat_cache {
        sc.retain(|(a, b), _| *a < bodies.len() && *b < bodies.len());
    }
    if active.is_empty() {
        return;
    }
    // Fast path: small active sets bypass cache overhead (same as sequential threshold).
    // For large sets we do two-phase: hits sequential, misses parallel via the original routine.
    let mut misses: Vec<(usize, usize)> = Vec::with_capacity(active.len());
    let mut fast_misses: Vec<(usize, usize)> = Vec::new();
    for &(i, j) in active {
        let a = &bodies[i];
        let b = &bodies[j];
        // Early rejects identical to the original routine — don't cache trivial rejects.
        if !a.can_collide_with(b) || a.is_trigger || b.is_trigger {
            continue;
        }
        if a.body_type == BodyType::Static && b.body_type == BodyType::Static {
            continue;
        }
        if asleep[i] && asleep[j] {
            continue;
        }
        if let Some(req) = body_required
            && req[i].max(req[j]) <= cur_substep
        {
            continue;
        }
        let rel_speed = (a.velocity - b.velocity).length();
        // Fast-moving pairs have near-zero cache hit rate — bypass HashMap lookup and don't cache.
        if rel_speed > 0.5
            || a.angular_velocity.length_squared() > 0.25
            || b.angular_velocity.length_squared() > 0.25
        {
            fast_misses.push((i, j));
            continue;
        }
        let margin = 0.05 + rel_speed * sub_dt;
        let key = (i, j);
        if let Some(entry) = cache.get(&key) {
            if narrow_cache_hit(entry, a, b, margin) {
                if let Some(m) = &entry.manifold {
                    out.push(m.clone());
                }
                continue;
            }
        }
        misses.push((i, j));
    }
    if misses.is_empty() && fast_misses.is_empty() {
        return;
    }
    // Fast-moving pairs: compute directly without caching.
    if !fast_misses.is_empty() {
        let mut fast_tmp: Vec<Manifold> = Vec::new();
        detect_collisions_into(
            bodies,
            &fast_misses,
            asleep,
            sub_dt,
            &mut fast_tmp,
            body_required,
            cur_substep,
            sat_cache,
            pool,
        );
        out.extend(fast_tmp);
    }
    if misses.is_empty() {
        return;
    }
    // Compute misses with the original (potentially parallel) routine into a temp vec.
    let mut tmp: Vec<Manifold> = Vec::new();
    detect_collisions_into(
        bodies,
        &misses,
        asleep,
        sub_dt,
        &mut tmp,
        body_required,
        cur_substep,
        sat_cache,
        pool,
    );
    // Populate cache for misses.
    let mut tmp_map: FxHashMap<(usize, usize), Option<Manifold>> = FxHashMap::default();
    for m in &tmp {
        tmp_map.insert((m.body_a, m.body_b), Some(m.clone()));
    }
    for &(i, j) in &misses {
        let key = (i, j);
        if tmp_map.contains_key(&key) {
            continue;
        }
        tmp_map.insert(key, None);
    }
    for &(i, j) in &misses {
        let a = &bodies[i];
        let b = &bodies[j];
        let rel_speed = (a.velocity - b.velocity).length();
        let margin = 0.05 + rel_speed * sub_dt;
        let manifold = tmp_map.get(&(i, j)).and_then(|o| o.clone());
        cache.insert(
            (i, j),
            NarrowCacheEntry {
                pos_a: a.position,
                pos_b: b.position,
                rot_a: a.orientation,
                rot_b: b.orientation,
                margin,
                manifold: manifold.clone(),
            },
        );
        if let Some(m) = manifold {
            out.push(m);
        }
    }
}

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
    /// Transverse Coulomb coefficient (ODE `mu2` parity): cap along `t2`
    /// while `mu` caps `t1`. Bitwise-equal to `mu` for all default bodies,
    /// which routes the solve through the legacy circular-cone path —
    /// existing scenes stay bit-identical (snapshot-guarded).
    pub mu2: f32,
    /// Pair rolling-resistance coefficient in meters (`max` of both
    /// bodies; MuJoCo `rolling` parity). Zero disables rolling solve.
    pub mu_roll: f32,
    /// Pair torsional coefficient in meters (`max`; MuJoCo parity).
    pub mu_spin: f32,
    /// Accumulated rolling impulses about `t1`/`t2` per point, capped by
    /// `mu_roll × acc[k]` (torque-cap × normal-impulse, MuJoCo units).
    pub acc_roll: [f32; 4],
    pub acc_roll2: [f32; 4],
    /// Accumulated torsional impulse about `n` per point, capped by
    /// `mu_spin × acc[k]`.
    pub acc_spin: [f32; 4],
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
#[derive(Clone, Debug)]
struct NarrowCacheEntry {
    pos_a: Vec3,
    pos_b: Vec3,
    rot_a: Quat,
    rot_b: Quat,
    margin: f32,
    manifold: Option<Manifold>,
}

#[derive(Clone, Debug)]
struct SatCacheEntry {
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    margin: f32,
    result: Option<(Vec3, f32)>,
}

/// Lock-free SAT axis cache: sorted body pair -> last separating axis.
/// `DashMap` shards internally, so the rayon narrowphase shares it without
/// `try_lock` misses or a parallel bypass. Hits reuse the axis only;
/// `box_manifold_cached` rebuilds contacts from current geometry.
type SatCache = DashMap<(usize, usize), SatCacheEntry, FxBuildHasher>;

#[allow(missing_docs)]
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
    island_timers: FxHashMap<u32, f32>,
    asleep: Vec<bool>,
    /// Persistent joint constraints with warm-start state (G5). Joints also
    /// feed the island union-find: jointed bodies sleep and wake together.
    joints: Vec<Joint>,
    /// Sorted body pairs connected by a joint. Jointed bodies never collide
    /// (Box2D `collide_connected = false` default): a hinge pin passes through
    /// the arm, so the parts legitimately sweep through each other's space,
    /// and contact friction there would act as a phantom brake on the joint.
    joint_pairs: FxHashSet<(usize, usize)>,
    /// Diagnostics: (body_a, body_b) of the last substep's manifolds.
    debug_pairs: Vec<(usize, usize)>,
    /// Trigger pairs overlapping on the previous completed step.
    trigger_pairs: FxHashSet<(usize, usize)>,
    /// Solid-contact pairs touching on the previous completed step
    /// (canonical min/max keys). Frozen pairs retain their state — sleep
    /// emits no transitions.
    contact_touch: FxHashSet<(usize, usize)>,
    /// Contact transitions waiting for the caller to drain (hits recorded
    /// pre-solve every substep with per-step dedupe, begin/end reconciled
    /// at step end).
    contact_events: Vec<ContactEvent>,
    /// Pairs that already emitted a hit this step (dedupe for sustained
    /// crushes); cleared at s==0 of every step.
    scratch_hit_pairs: FxHashSet<(usize, usize)>,
    /// Wall-clock breakdown of the last completed `step` (diagnostics only).
    last_step_timing: StepTiming,
    /// Worst-case budget: deterministic substep shedding (see
    /// [`StepBudget`]). `None` disables shedding. Default: on.
    step_budget: Option<StepBudget>,
    /// Substeps shed by the budget on the last completed `step` (0 when the
    /// full speed-requested count ran). Observable marker for the fallback,
    /// read via [`BuiltinPhysicsEngine::last_substep_shed`].
    last_shed: u32,
    /// Enter/exit transitions waiting for the caller to drain.
    trigger_events: Vec<TriggerEvent>,
    /// G7: enable SIMD-wide contact solver for single-point manifolds.
    /// Default true. Set to false for bit-exact scalar reproduction.
    wide_solver: bool,
    /// Scratch buffers reused across substeps to avoid per-frame allocations.
    scratch_manifolds: Vec<Manifold>,
    scratch_pairs: Vec<(usize, usize)>,
    /// Per-substep bucket boundaries over `scratch_pairs` (stable counting
    /// sort by required substeps, rebuilt once per step when filtering).
    scratch_bucket_edges: Vec<usize>,
    scratch_clamped: Vec<bool>,
    scratch_parent: Vec<usize>,
    /// Pooled narrowphase shard buffers for scheduler dispatch, taken and
    /// restored around the substep loop like the other scratch state.
    scratch_narrow_shards: NarrowShardPool,
    /// G7: optional GPU contact solver (gpu feature). When attached,
    /// single-point manifolds are solved on the GPU instead of the CPU
    /// wide path; multi-point manifolds stay on the CPU island path.
    #[cfg(feature = "gpu")]
    gpu_solver: Option<WgpuContactSolver>,
    narrow_cache: FxHashMap<(usize, usize), NarrowCacheEntry>,
    sat_cache: SatCache,
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
            warm_impulses: FxHashMap::default(),
            island: Vec::new(),
            island_timers: FxHashMap::default(),
            asleep: Vec::new(),
            joints: Vec::new(),
            joint_pairs: FxHashSet::default(),
            debug_pairs: Vec::new(),
            trigger_pairs: FxHashSet::default(),
            contact_touch: FxHashSet::default(),
            contact_events: Vec::new(),
            scratch_hit_pairs: FxHashSet::default(),
            last_step_timing: StepTiming::default(),
            step_budget: Some(StepBudget::default()),
            last_shed: 0,
            trigger_events: Vec::new(),
            wide_solver: true,
            scratch_manifolds: Vec::new(),
            scratch_pairs: Vec::new(),
            scratch_bucket_edges: Vec::new(),
            scratch_clamped: Vec::new(),
            scratch_parent: Vec::new(),
            scratch_narrow_shards: NarrowShardPool::default(),
            narrow_cache: FxHashMap::default(),
            sat_cache: DashMap::default(),
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
    /// [`BroadPhaseKind::Auto`] routes analytically between sweep, grid and
    /// tree with hysteresis; see [`BroadPhaseKind::Auto`] for the contract.
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

    /// Backend that served the latest update when
    /// [`BroadPhaseKind::Auto`] is selected (any of the three backends,
    /// never `Auto` itself); `None` for explicit backend
    /// selections. Diagnostics for tuning and benchmarks, not part of the
    /// simulation contract.
    pub fn auto_active_broadphase(&self) -> Option<BroadPhaseKind> {
        self.broadphase.auto_active_kind()
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

    /// Worst-case step budget (default: on, see [`StepBudget::default`]).
    /// `None` disables shedding and always runs the speed-requested count.
    /// Some budget sheds substeps — never pairs — when the candidate-pair
    /// count times the requested count exceeds the budget. The shed count of
    /// the last step is observable via [`Self::last_substep_shed`].
    pub fn set_step_budget(&mut self, budget: Option<StepBudget>) {
        self.step_budget = budget;
    }

    /// Substeps shed by the budget on the last completed `step`: requested
    /// minus applied (0 when the full count ran, or when the world slept
    /// through the step). Diagnostics for tuning; not part of the
    /// simulation contract.
    pub fn last_substep_shed(&self) -> u32 {
        self.last_shed
    }

    /// Applies the worst-case budget to a speed-requested substep count,
    /// given the broadphase candidate-pair count. Returns
    /// `(applied, shed)`. Pure counts, no wall-clock: deterministic for a
    /// given scene trajectory.
    fn apply_step_budget(&self, requested: u32, pairs: usize) -> (u32, u32) {
        let Some(budget) = self.step_budget else {
            return (requested, 0);
        };
        if requested <= budget.min_substeps || pairs == 0 {
            return (requested, 0);
        }
        let allowed = u32::try_from(budget.max_pair_substeps / pairs).unwrap_or(u32::MAX);
        let applied = requested.min(allowed.max(budget.min_substeps));
        (applied, requested - applied)
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

    /// Per-body required substeps for the current frame (B: per-island
    /// sub-dt splitting). Slow bodies need MIN, fast need up to `self.substeps`.
    pub fn body_required_substeps(&self, dt: f32) -> Vec<u32> {
        const MIN_SUBSTEPS: u32 = 4;
        const SUB_DT_TARGET: f32 = 1.0 / 240.0;
        let lower = MIN_SUBSTEPS.min(self.substeps);
        self.bodies
            .iter()
            .enumerate()
            .map(|(h, b)| {
                if b.body_type != BodyType::Dynamic || self.asleep[h] {
                    lower
                } else {
                    let wanted = (b.velocity.length() * dt / SUB_DT_TARGET).ceil() as u32;
                    wanted.clamp(lower, self.substeps)
                }
            })
            .collect()
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

    /// Whether the body is currently sleeping (G4/G7 diagnostics). Static
    /// bodies report true from birth — they never move, which is exactly
    /// what the frozen-pair skips in the narrow phase rely on.
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
    /// Free spin also gains the gyroscopic correction
    /// ([`apply_gyroscopic`]): without it anisotropic bodies cannot precess,
    /// and spin about the intermediate inertia axis stays (wrongly) stable.
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
            // Gyroscopic correction, torque or not: free spinners need it.
            Self::apply_gyroscopic(
                body.inertia,
                body.orientation,
                &mut body.angular_velocity,
                dt,
            );
        }
    }

    /// Gyroscopic correction, Box3D-`b3IntegrateVelocitiesTask` idea with an own
    /// diagonal-only implementation: Newton-Raphson on
    /// `I·(w2−w1) + h·(w2×(I·w2)) = 0` in the body frame (our inertia is a body
    /// diagonal, theirs a full local 3×3 — same equation, cheaper Jacobian).
    /// Fixed [`GYROSCOPIC_ITERATIONS`] iterations, so the result is a pure
    /// function of the inputs (deterministic); on non-convergence the last
    /// iterate is kept, never NaN (the Jacobian stays diagonally dominant for
    /// small `h`, and `solve_small` reports singularity instead of dividing).
    ///
    /// Skips, both exact: isotropic inertia (the term is identically zero, so
    /// every cube/sphere test scene pays nothing) and zero spin.
    ///
    /// Without this, spin about the intermediate inertia axis is (wrongly)
    /// stable and free asymmetric bodies never tumble — see the Dzhanibekov
    /// test below.
    const GYROSCOPIC_ITERATIONS: usize = 3;

    fn apply_gyroscopic(inertia: Vec3, orientation: Quat, omega: &mut Vec3, h: f32) {
        if *omega == Vec3::ZERO {
            return;
        }
        let imax = inertia.x.max(inertia.y).max(inertia.z);
        if !imax.is_finite() || imax <= 0.0 {
            return;
        }
        let spread = (inertia.x - inertia.y)
            .abs()
            .max((inertia.y - inertia.z).abs())
            .max((inertia.z - inertia.x).abs());
        if spread <= 1e-6 * imax {
            return;
        }
        let (a, b, c) = (inertia.x, inertia.y, inertia.z);
        // To the body frame (diagonal inertia) and back with the same rotation.
        let q = orientation;
        let w1 = q.conjugate() * *omega;
        let mut w2 = w1;
        for _ in 0..Self::GYROSCOPIC_ITERATIONS {
            // u = I·w2; residual r = I·(w2−w1) + h·(w2×u).
            let u = Vec3::new(a * w2.x, b * w2.y, c * w2.z);
            let cross = w2.cross(u);
            let r = Vec3::new(
                a * (w2.x - w1.x) + h * cross.x,
                b * (w2.y - w1.y) + h * cross.y,
                c * (w2.z - w1.z) + h * cross.z,
            );
            // Jacobian J = I + h·(skew(w2)·I − skew(u)), diagonal inertia:
            // J[i][i] is the bare inertia (the motion term has no self-part),
            // off-diagonals mix the spin with the inertia-weighted spin.
            let j = [
                [a, h * (u.z - b * w2.z), h * (c * w2.y - u.y), 0.0],
                [h * (a * w2.z - u.z), b, h * (u.x - c * w2.x), 0.0],
                [h * (u.y - a * w2.y), h * (b * w2.x - u.x), c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let Some(d) = solve_small(&j, &[-r.x, -r.y, -r.z, 0.0], 3) else {
                break;
            };
            w2 += Vec3::new(d[0], d[1], d[2]);
        }
        let w2_world = q * w2;
        debug_assert!(
            vec3_finite(w2_world),
            "gyroscopic correction diverged: inertia={inertia:?} w1={w1:?}"
        );
        *omega = w2_world;
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
    /// advancement; rotating boxes/capsules use the fully analytic swept-volume
    /// conservative advancement (exact distance + `|disp|+r*angle` bound, no
    /// 5° sampling). The body is clamped to the first detected impact and
    /// flagged in `skip` so `integrate_positions` does not move it a second
    /// time.
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
                // Unified contact impulse: contact-point speed (spin counts),
                // restitution-aware, tangential preserved. Subsumes the old
                // split linear-bounce + spin-kill for angular hits.
                let lever = hit.contact.map(|c| c - b.position).unwrap_or(Vec3::ZERO);
                let (v, w) = ccd_impact_velocity(
                    b.velocity,
                    b.angular_velocity,
                    b.inv_mass,
                    b.inertia,
                    b.orientation,
                    lever,
                    hit.normal,
                    e,
                );
                b.velocity = v;
                b.angular_velocity = w;
            } else {
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
    /// World contact point on the mover at the hit fraction (angular hits
    /// only): the response levers the spin off it instead of killing it.
    contact: Option<Vec3>,
}

fn shape_min_dimension(shape: &Shape) -> f32 {
    match shape {
        Shape::Sphere { radius } => *radius,
        Shape::Box { half_extents } => half_extents.min_element(),
        Shape::Capsule { radius, .. } => *radius,
    }
}

fn shape_max_radius(shape: &Shape) -> f32 {
    match shape {
        Shape::Sphere { radius } => *radius,
        Shape::Box { half_extents } => half_extents.length(),
        Shape::Capsule {
            radius,
            half_height,
        } => half_height + radius,
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

/// Energy-neutral cap for a spin correction `delta`: walk back along it to
/// the closed-form neutral point `t = (d·Iω)/E(d)` when the full correction
/// would add rotational energy (stiff anisotropic levers). Pure arithmetic,
/// hence deterministic; degenerate inputs return `omega` unchanged.
fn cap_spin_correction(omega: Vec3, inertia: Vec3, orientation: Quat, delta: Vec3) -> Vec3 {
    let out = omega - delta;
    // Body-frame energies: E(Ω) = ½ΣIᵢwᵢ². Exact correction first; if it
    // would inject energy, walk back along `delta` to the neutral point.
    let qb = orientation.conjugate();
    let wb = qb * omega;
    let db = qb * delta;
    let iw = inertia * wb;
    let e_omega = 0.5 * iw.dot(wb);
    let e_out = 0.5 * (inertia * (wb - db)).dot(wb - db);
    if e_out <= e_omega {
        return out;
    }
    let e_d = 0.5 * (inertia * db).dot(db);
    if !e_d.is_finite() || e_d <= 0.0 {
        return omega;
    }
    let t = db.dot(iw) / e_d;
    if !t.is_finite() || t <= 0.0 {
        return omega;
    }
    omega - delta * t.min(1.0)
}

/// Frictionless spin response for a CCD stop: remove exactly the spin that
/// drives the contact point into the surface, keep the tangential spin.
/// The correction follows a frictionless contact impulse — `Δω = J·I⁻¹m`
/// with the lever `m = r_c×n̂` and `J = −vn/((I⁻¹m)·m)`, `vn = ((ω×r_c)·n̂)`
/// the approach speed (`n̂` points from the target back toward the mover, so
/// approach is `vn < 0`). Consequences: for isotropic inertia this is the
/// minimum-norm projection; for planar motion with an in-plane contact
/// normal it reduces exactly to the old full stop — the fix only changes
/// contacts with a genuine out-of-plane lever (scrapes keep tangential
/// spin instead of dying). No angular restitution here (inelastic) — the
/// restitution-aware path is [`ccd_impact_velocity`]; friction stays with
/// the contact/joint passes.
///
/// The energy-neutral [`cap_spin_correction`] applies (see its docs).
/// Pure arithmetic, hence deterministic. Degenerate lever or a separating
/// contact returns `omega` unchanged (never NaN).
#[cfg(test)]
fn remove_angular_approach(
    omega: Vec3,
    inertia: Vec3,
    orientation: Quat,
    lever: Vec3,
    normal: Vec3,
) -> Vec3 {
    let m = lever.cross(normal);
    let im = mul_inv_inertia(inertia, orientation, m);
    let denom = im.dot(m);
    if !denom.is_finite() || denom <= 1e-12 {
        return omega;
    }
    let vn = omega.cross(lever).dot(normal);
    if !vn.is_finite() || vn >= 0.0 {
        return omega;
    }
    cap_spin_correction(omega, inertia, orientation, im * (vn / denom))
}

/// Unified one-shot impact for an angular CCD hit: the contact-point velocity
/// `v_c = v + ω×r_c` (spin counts toward the approach, not just the center
/// motion), a textbook rigid-body impulse `J = −(1+e)·vn_c/denom` with
/// `denom = inv_mass + (I⁻¹m)·m`, restitution `e` above the shared 1 m/s
/// contact-speed threshold (otherwise inelastic). Tangential surface motion
/// is preserved — friction is the discrete solver's job next substep (it
/// sees a clean touching contact thanks to the clamp+backoff), so CCD never
/// double-applies it. The linear part applies fully (center-mass, bounded);
/// the spin part goes through [`cap_spin_correction`]. Pure arithmetic,
/// hence deterministic.
#[allow(clippy::too_many_arguments)]
fn ccd_impact_velocity(
    velocity: Vec3,
    omega: Vec3,
    inv_mass: f32,
    inertia: Vec3,
    orientation: Quat,
    lever: Vec3,
    normal: Vec3,
    restitution: f32,
) -> (Vec3, Vec3) {
    let contact_vel = velocity + omega.cross(lever);
    let vn_c = contact_vel.dot(normal);
    if !vn_c.is_finite() || vn_c >= 0.0 || !inv_mass.is_finite() || inv_mass <= 0.0 {
        return (velocity, omega);
    }
    let e = if vn_c < -1.0 { restitution } else { 0.0 };
    let m = lever.cross(normal);
    let im = mul_inv_inertia(inertia, orientation, m);
    let denom = inv_mass + im.dot(m);
    if !denom.is_finite() || denom <= 1e-12 {
        return (velocity, omega);
    }
    let impulse = -(1.0 + e) * vn_c / denom;
    let new_velocity = velocity + normal * (impulse * inv_mass);
    let new_omega = cap_spin_correction(omega, inertia, orientation, im * -impulse);
    (new_velocity, new_omega)
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
        contact: None,
    })
}

/// Conservative-advancement first overlap for the combined linear+angular
/// sweep. Uses the exact distance at each pose and the uniform bound
/// `|displacement| + max_radius*angle` per unit fraction.
///
/// Why the uniform bound is enough (and deliberately kept): the mover is
/// rigid and the target frozen, so the gap is Lipschitz in the fraction with
/// exactly this constant — the distance cannot change faster than the
/// fastest surface point. A step of `gap/μ` therefore varies the gap by at
/// most `gap`: a pass-through must land interpenetrating, never straddling,
/// so the per-iterate overlap check plus the binary refine catch every
/// crossing for any feature thickness. Any "tighter" witness-based bound
/// would break this (witness switches mid-step), trading a proof for fewer
/// iterations — not worth it; separated pairs already exit in 1–2 oracle
/// calls, only grazing approaches walk the full 32.
fn first_angular_overlap_fraction(
    body: &RigidBody,
    target: distance::ShapeRef<'_>,
    displacement: Vec3,
    sub_dt: f32,
) -> Option<f32> {
    if swept_shape_overlaps(body, target, displacement, sub_dt, 0.0) {
        return None;
    }
    let angle = (body.angular_velocity * sub_dt).length();
    let bound = displacement.length() + shape_max_radius(&body.shape) * angle;
    if bound < 1e-9 {
        return None;
    }
    const TOUCH: f32 = 1e-5;
    const MAX_ITERS: usize = 32;
    let mut f = 0.0f32;
    let mut prev_f = 0.0f32;
    for _ in 0..MAX_ITERS {
        if f >= 1.0 {
            break;
        }
        if swept_shape_overlaps(body, target, displacement, sub_dt, f) {
            if f <= 0.0 {
                return None;
            }
            // Binary refine the bracket [prev_f, f] for sub-sample precision.
            let mut low = prev_f;
            let mut high = f;
            for _ in 0..10 {
                let mid = (low + high) * 0.5;
                if swept_shape_overlaps(body, target, displacement, sub_dt, mid) {
                    high = mid;
                } else {
                    low = mid;
                }
            }
            return Some(high);
        }
        let d = swept_distance(body, target, displacement, sub_dt, f);
        // `d.dist` is the exact surface gap (positive = separated). Advance
        // by at most the gap over the worst-case point speed.
        let gap = d.dist - TOUCH * 0.5;
        if gap <= 0.0 {
            // Numerically touching — treat as overlap at next fraction.
            let next = (f + 1e-4).min(1.0);
            if swept_shape_overlaps(body, target, displacement, sub_dt, next) {
                return Some(next);
            }
            break;
        }
        let step = (gap / bound).clamp(1e-4, 1.0 - f);
        prev_f = f;
        f += step;
        if f <= prev_f {
            break;
        }
    }
    None
}

/// Fully analytic angular CCD: conservative advancement along the screw
/// motion `pos(t)=pos0+disp*t, rot(t)=slerp(angVel*t)` using the exact
/// distance oracle. No 5° sampling — the bound guarantees zero tunneling
/// for any thin feature, any angle.
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
    // Travel gate, mirror of the linear one (`0.5 * min_dimension` on
    // displacement): rotation alone cannot defeat the discrete phase unless
    // its fastest surface point moves more than half the thinnest feature
    // within the substep. Thin bodies arm CCD at small angles (they tunnel
    // easily); chunky bodies only at large ones — cheaper than a flat angle
    // for cubes, stricter than one for blades.
    let angle = (body.angular_velocity * sub_dt).length();
    if shape_max_radius(&body.shape) * angle <= 0.5 * shape_min_dimension(&body.shape) {
        return None;
    }
    let bound = displacement.length() + shape_max_radius(&body.shape) * angle;
    if bound < 1e-9 {
        return None;
    }
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
        let Some(fraction) = first_angular_overlap_fraction(body, target_ref, displacement, sub_dt)
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
            contact: Some(distance.point_a),
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
        // Driven kinematics are never asleep by construction: a MOVING one
        // (nonzero velocity field) keeps the world awake, otherwise a
        // kinematic platform would ghost through sleepers with the loop
        // skipped. Parked kinematics (zero velocity) cost nothing. Driver
        // contract (Box2D parity): a teleported body MUST carry the matching
        // velocity field — zero-velocity teleports are invisible to wake,
        // margins and CCD alike.
        let has_driven_kinematic = self.bodies.iter().any(|b| {
            b.body_type == BodyType::Kinematic
                && (b.velocity.length_squared() + b.angular_velocity.length_squared() > 0.0)
        });
        let has_trigger = self.bodies.iter().any(|body| body.is_trigger);
        if !has_awake_dynamic
            && !has_driven_kinematic
            && !has_trigger
            && self.trigger_pairs.is_empty()
        {
            // Fully sleeping world: no substep loop ran, so no phase work
            // happened this step.
            self.last_step_timing = StepTiming::default();
            self.last_shed = 0;
            return;
        }
        let eff_substeps_before_budget = self.effective_substeps(dt);
        // Diagnostics: contact-manifold partners per body from the last
        // Reuse scratch buffers across substeps; keep capacity across frames.
        self.scratch_manifolds.clear();
        // last_manifolds will alias scratch_manifolds after loop; keep separate copy for debug
        let mut last_manifolds_snapshot: Vec<Manifold> = Vec::new();
        // Broadphase once per step using full dt swept AABBs (conservative
        // superset for all substeps). This cuts 12x broadphase cost for the
        // tiled 10k case: 70ms -> 6ms. Narrow still per substep because
        // contacts depend on moving poses, but candidate list is reused.
        let t0 = Instant::now();
        self.broadphase.update(&self.bodies, dt);
        let broad_phase_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let broad_active: Vec<(usize, usize)> = self.broadphase.active().to_vec();
        // Worst-case budget (deterministic): shed substeps against the known
        // pair count — never pairs. `timing.substeps` records the applied
        // count; the shed count is kept in `last_shed` for observability.
        let (eff_substeps, shed) =
            self.apply_step_budget(eff_substeps_before_budget, broad_active.len());
        self.last_shed = shed;
        let sub_dt = dt / eff_substeps as f32;
        let mut timing = StepTiming {
            substeps: eff_substeps,
            broad_phase_ms,
            ..StepTiming::default()
        };
        // B: per-body required substeps for filtering extra substeps.
        let body_needed = self.body_required_substeps(dt);
        let body_needed_opt: Option<&[u32]> = {
            let min_needed = body_needed.iter().copied().min().unwrap_or(0);
            let max_needed = body_needed.iter().copied().max().unwrap_or(0);
            if max_needed - min_needed < 4 {
                None
            } else {
                Some(&body_needed)
            }
        };
        let needs_filter = body_needed_opt.is_some() || !self.joint_pairs.is_empty();
        // Per-substep pair buckets: stable counting sort of the candidate
        // pairs by required substeps, once per step. Slow pairs are then
        // visited only on the substeps that need them instead of being
        // re-checked (and re-rejected) 12x per step. Bucket s holds exactly
        // the pairs the old per-substep filter kept at s, in the same
        // relative order — bit-identical narrowphase input per substep.
        let mut sorted_pairs = std::mem::take(&mut self.scratch_pairs);
        sorted_pairs.clear();
        let mut bucket_edges = std::mem::take(&mut self.scratch_bucket_edges);
        bucket_edges.clear();
        if needs_filter {
            let eff = eff_substeps as usize;
            sorted_pairs.reserve(broad_active.len());
            // Count required-substep classes, clamped to eff (over-budget
            // pairs are needed on every substep that runs, like before).
            // bucket_edges doubles as the counter array, then the placement
            // cursors, then the final boundary table — no per-step
            // allocation. Class of a pair: jointed pairs are excluded on
            // every substep (class 0 = never visited); without per-body
            // data every pair needs every substep (class eff).
            bucket_edges.resize(eff + 1, 0);
            // Fast path: no joints — skip the per-pair set lookup entirely.
            if self.joint_pairs.is_empty() {
                if let Some(req) = body_needed_opt {
                    for &(a, b) in &broad_active {
                        bucket_edges[(req[a].max(req[b]) as usize).min(eff)] += 1;
                    }
                } else {
                    bucket_edges[eff] = broad_active.len();
                }
            } else {
                for &(a, b) in &broad_active {
                    let c = if self.joint_pairs.contains(&(a, b)) {
                        0
                    } else if let Some(req) = body_needed_opt {
                        (req[a].max(req[b]) as usize).min(eff)
                    } else {
                        eff
                    };
                    bucket_edges[c] += 1;
                }
            }
            // Exclusive prefix sums: starts[c] = #{k < c}. Forward stable
            // placement advances them into end offsets #{k <= c} — which
            // are exactly the bucket boundaries S_s (bucket s = suffix
            // [S_s..N) of pairs with class > s, original relative order).
            let mut cursor = 0;
            for slot in bucket_edges.iter_mut() {
                let n = *slot;
                *slot = cursor;
                cursor += n;
            }
            sorted_pairs.resize(broad_active.len(), (0, 0));
            if self.joint_pairs.is_empty() {
                if let Some(req) = body_needed_opt {
                    for &(a, b) in &broad_active {
                        let c = (req[a].max(req[b]) as usize).min(eff);
                        let w = bucket_edges[c];
                        sorted_pairs[w] = (a, b);
                        bucket_edges[c] = w + 1;
                    }
                } else {
                    // No per-body data, no joints: every pair needs every
                    // substep. Counting put the whole list in class eff at
                    // offset 0; the prefix starts are already the correct
                    // boundaries (S_s = 0 for s < eff).
                    sorted_pairs.copy_from_slice(&broad_active);
                }
            } else {
                for &(a, b) in &broad_active {
                    let c = if self.joint_pairs.contains(&(a, b)) {
                        0
                    } else if let Some(req) = body_needed_opt {
                        (req[a].max(req[b]) as usize).min(eff)
                    } else {
                        eff
                    };
                    if c == 0 {
                        continue;
                    }
                    let w = bucket_edges[c];
                    sorted_pairs[w] = (a, b);
                    bucket_edges[c] = w + 1;
                }
            }
        }
        // Scheduler narrowphase shard pool, taken once for the whole substep
        // loop and restored after (same discipline as the other scratch).
        let mut narrow_shards = std::mem::take(&mut self.scratch_narrow_shards);
        for s in 0..eff_substeps {
            // Per-step hit dedupe window opens here (see collect_active).
            if s == 0 {
                self.scratch_hit_pairs.clear();
            }
            // Box3D stage order: solve velocities BEFORE moving positions, so
            // a resting contact kills gravity's velocity gain in the same
            // substep instead of letting the body free-fall and snapping it
            // back (the snap is an inelastic collision and bleeds energy).
            self.integrate_velocities(sub_dt);
            // Narrowphase input: the full candidate list on homogeneous
            // scenes, the pre-bucketed class suffix on mixed ones (see the
            // counting sort above the substep loop).
            let t0 = Instant::now();
            let mut manifolds_buf = std::mem::take(&mut self.scratch_manifolds);
            manifolds_buf.clear();
            if needs_filter {
                // Bucket s = suffix of pairs with class > s: exactly the
                // set the old per-substep filter kept, same order.
                let start = bucket_edges[s as usize];
                detect_collisions_into_with_cache(
                    &self.bodies,
                    &sorted_pairs[start..],
                    &self.asleep,
                    sub_dt,
                    &mut manifolds_buf,
                    None,
                    s,
                    &mut self.narrow_cache,
                    Some(&self.sat_cache),
                    &mut narrow_shards,
                );
            } else {
                detect_collisions_into_with_cache(
                    &self.bodies,
                    &broad_active,
                    &self.asleep,
                    sub_dt,
                    &mut manifolds_buf,
                    None,
                    s,
                    &mut self.narrow_cache,
                    Some(&self.sat_cache),
                    &mut narrow_shards,
                );
            }
            timing.narrow_phase_ms += t0.elapsed().as_secs_f64() * 1000.0;
            // Restitution is one-shot per step, evaluated on the first substep.
            let t0 = Instant::now();
            let mut islands = self.solve_contacts_velocity(&manifolds_buf, s == 0, sub_dt, dt);
            self.solve_joints_velocity(sub_dt);
            // Continuous pass on the solver-adjusted velocities: clamp fast
            // movers to their first impact and keep them there this substep.
            // Reuse a local clamped buffer (capacity kept across substeps via scratch).
            let mut clamped_buf = std::mem::take(&mut self.scratch_clamped);
            clamped_buf.clear();
            clamped_buf.resize(self.bodies.len(), false);
            clamped_buf.fill(false);
            self.solve_continuous(sub_dt, &mut clamped_buf);
            self.integrate_positions(sub_dt, &clamped_buf);
            self.solve_contacts_position(&mut islands, dt);
            self.solve_joints_position();
            timing.solver_ms += t0.elapsed().as_secs_f64() * 1000.0;
            // Snapshot last manifolds for island rebuild; clone only once per frame
            if s + 1 == eff_substeps {
                last_manifolds_snapshot.clear();
                last_manifolds_snapshot.extend_from_slice(&manifolds_buf);
            }
            self.scratch_clamped = clamped_buf;
            self.scratch_manifolds = manifolds_buf;
        }
        self.scratch_narrow_shards = narrow_shards;
        self.scratch_pairs = sorted_pairs;
        self.scratch_bucket_edges = bucket_edges;
        // Diagnostics: contact-manifold partners per body from the last
        // substep (drives sleep/island debugging; tiny flat copy).
        self.debug_pairs.clear();
        self.debug_pairs
            .extend(last_manifolds_snapshot.iter().map(|m| (m.body_a, m.body_b)));
        let t_island = Instant::now();
        self.rebuild_islands(&last_manifolds_snapshot);
        self.update_sleep(dt);
        timing.island_ms += t_island.elapsed().as_secs_f64() * 1000.0;
        self.last_step_timing = timing;

        // Rebuild the broadphase at the completed poses so trigger events
        // describe the state visible after this whole physics step, not the
        // state from before the final substep's integration. The rebuild
        // itself always runs (backends key their incremental baseline off
        // it); only the overlap detect + reconcile is gated — triggerless
        // worlds skip it instead of scanning every pair for nothing.
        let t_trigger = Instant::now();
        self.broadphase.update(&self.bodies, 0.0);
        if has_trigger || !self.trigger_pairs.is_empty() {
            let current_triggers = detect_trigger_overlaps(&self.bodies, self.broadphase.active());
            let previous_triggers = std::mem::take(&mut self.trigger_pairs);
            self.trigger_pairs = update_trigger_events(
                &previous_triggers,
                current_triggers,
                &mut self.trigger_events,
            );
        }
        // Solid-contact begin/end reconcile (gameplay events): touching =
        // penetration above the begin slop in the last substep's manifolds.
        // Frozen pairs keep their prior state (sleep emits no transitions); stale handles
        // (removed bodies) are dropped — removal clears the set anyway.
        let n = self.bodies.len();
        let mut current_touch: FxHashSet<(usize, usize)> = FxHashSet::default();
        for m in &last_manifolds_snapshot {
            let (a, b) = (m.body_a, m.body_b);
            if a >= n || b >= n {
                continue;
            }
            let mut touching = false;
            for k in 0..m.point_count {
                if m.points[k].penetration > -CONTACT_BEGIN_SLOP {
                    touching = true;
                    break;
                }
            }
            if touching {
                current_touch.insert((a.min(b), a.max(b)));
            }
        }
        for &(a, b) in &self.contact_touch {
            if a < n && b < n && self.asleep[a] && self.asleep[b] {
                current_touch.insert((a, b));
            }
        }
        let previous_touch = std::mem::replace(&mut self.contact_touch, current_touch);
        let mut begins: Vec<(usize, usize)> = self
            .contact_touch
            .difference(&previous_touch)
            .copied()
            .collect();
        let mut ends: Vec<(usize, usize)> = previous_touch
            .difference(&self.contact_touch)
            .copied()
            .collect();
        begins.sort_unstable();
        ends.sort_unstable();
        for (a, b) in begins {
            self.contact_events.push(ContactEvent {
                body_a: a,
                body_b: b,
                kind: ContactEventKind::Begin,
            });
        }
        for (a, b) in ends {
            self.contact_events.push(ContactEvent {
                body_a: a,
                body_b: b,
                kind: ContactEventKind::End,
            });
        }
        self.contact_events
            .sort_by_key(|e| (e.body_a.min(e.body_b), e.body_a.max(e.body_b)));
        self.last_step_timing.trigger_ms += t_trigger.elapsed().as_secs_f64() * 1000.0;
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        let handle = self.bodies.len();
        let island_id = if body.body_type == BodyType::Dynamic {
            handle as u32
        } else {
            u32::MAX
        };
        // World-scale sleep foundation: statics are born asleep (they never
        // move, so every frozen-pair skip in narrowphase/scheduler applies
        // to them from step one). Kinematics stay awake — driven bodies
        // must wake sleepers on contact, never ghost through them.
        let born_asleep = body.body_type == BodyType::Static;
        self.bodies.push(body);
        self.island.push(island_id);
        self.asleep.push(born_asleep);
        handle
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        if handle < self.bodies.len() {
            let last = self.bodies.len() - 1;
            let previous_triggers = std::mem::take(&mut self.trigger_pairs);
            let mut removed_triggers = Vec::new();
            let mut remapped_triggers = FxHashSet::default();
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
            // Body handles are identities: the swap remaps the tail body's
            // index, so contact state keyed by handles is no longer valid.
            // Drop it (Box2D parity: events may invalidate on destroy) —
            // surviving contacts re-begin on the next step.
            self.contact_touch.clear();
            self.contact_events.clear();
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
        // Normalize the hinge/slide axes once, at creation.
        let kind = match kind {
            JointKind::Revolute {
                local_anchor_a,
                local_anchor_b,
                local_axis_a,
                local_axis_b,
                limit,
                motor,
            } => JointKind::Revolute {
                local_anchor_a,
                local_anchor_b,
                local_axis_a: local_axis_a.normalize_or(Vec3::Z),
                local_axis_b: local_axis_b.normalize_or(Vec3::Z),
                limit,
                motor,
            },
            JointKind::Prismatic {
                local_anchor_a,
                local_anchor_b,
                local_axis_a,
                local_axis_b,
                limit,
                motor,
            } => JointKind::Prismatic {
                local_anchor_a,
                local_anchor_b,
                local_axis_a: local_axis_a.normalize_or(Vec3::Z),
                local_axis_b: local_axis_b.normalize_or(Vec3::Z),
                limit,
                motor,
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
        // Limits/motors measure travel from the assembly pose (Box2D
        // `m_referenceAngle`): capture the hinge twist before the first step.
        let reference_angle = match &kind {
            JointKind::Revolute { local_axis_a, .. } => crate::engine::joints::hinge_twist(
                self.bodies[body_a].orientation,
                self.bodies[body_b].orientation,
                *local_axis_a,
            ),
            JointKind::Ball { .. } | JointKind::Prismatic { .. } => 0.0,
        };
        // Prismatic limits measure anchor separation along the slide axis
        // from the assembly pose.
        let reference_length = match &kind {
            JointKind::Prismatic {
                local_anchor_a,
                local_anchor_b,
                local_axis_a,
                ..
            } => {
                let wa = (self.bodies[body_a].orientation * *local_axis_a).normalize_or(Vec3::Z);
                let ra = self.bodies[body_a].orientation * *local_anchor_a;
                let rb = self.bodies[body_b].orientation * *local_anchor_b;
                ((self.bodies[body_b].position + rb) - (self.bodies[body_a].position + ra)).dot(wa)
            }
            JointKind::Ball { .. } | JointKind::Revolute { .. } => 0.0,
        };
        let mut joint = Joint::new(body_a, body_b, kind);
        joint.reference_angle = reference_angle;
        joint.reference_length = reference_length;
        self.joints.push(joint);
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

    fn drain_contact_events(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.contact_events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::{PrismaticLimit, PrismaticMotor};
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
    fn auto_broadphase_routes_small_scene_to_sweep_and_steps() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.set_broadphase(BroadPhaseKind::Auto);
        assert_eq!(physics.broadphase_kind(), BroadPhaseKind::Auto);
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(10.0, 0.5, 10.0),
            0.0,
        ));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::splat(0.4),
            1.0,
        ));
        physics.step(1.0 / 60.0);
        assert_eq!(
            physics.auto_active_broadphase(),
            Some(BroadPhaseKind::SweepAndPrune)
        );
        let body = physics.get_body(1).unwrap();
        assert!(body.position.y < 5.0, "dynamic body still falls under Auto");

        // Explicit selections report no auto-active backend.
        physics.set_broadphase(BroadPhaseKind::UniformGrid);
        assert_eq!(physics.auto_active_broadphase(), None);
    }

    #[test]
    fn step_budget_shed_arithmetic_is_deterministic() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.set_step_budget(Some(StepBudget {
            max_pair_substeps: 200_000,
            min_substeps: 4,
        }));
        assert_eq!(physics.apply_step_budget(12, 0), (12, 0));
        // Tiled-10k-cold scale stays untouched under the default budget.
        assert_eq!(physics.apply_step_budget(12, 14_161), (12, 0));
        assert_eq!(physics.apply_step_budget(12, 1_000_000), (4, 8));
        // The floor is never shed, even under extreme load.
        assert_eq!(physics.apply_step_budget(4, 1_000_000), (4, 0));
        physics.set_step_budget(None);
        assert_eq!(physics.apply_step_budget(12, 1_000_000), (12, 0));
    }

    fn dense_shedding_scene() -> BuiltinPhysicsEngine {
        // 24 overlapping dynamic boxes (276 candidate pairs) plus one fast
        // body forcing the 12-substep speed request.
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        for i in 0..24 {
            physics.add_body(RigidBody::new_box(
                Vec3::new((i % 5) as f32 * 0.1, (i / 5) as f32 * 0.1, 0.0),
                Vec3::splat(0.4),
                1.0,
            ));
        }
        let mut fast = RigidBody::new_box(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
        fast.velocity = Vec3::new(0.0, -40.0, 0.0);
        physics.add_body(fast);
        physics
    }

    #[test]
    fn step_budget_leaves_typical_scenes_untouched() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(10.0, 0.5, 10.0),
            0.0,
        ));
        let mut fast = RigidBody::new_box(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
        fast.velocity = Vec3::new(0.0, -40.0, 0.0);
        physics.add_body(fast);
        physics.step(1.0 / 60.0);
        assert_eq!(physics.step_timing().substeps, 12);
        assert_eq!(physics.last_substep_shed(), 0);
    }

    #[test]
    fn step_budget_sheds_substeps_but_never_pairs() {
        let mut budgeted = dense_shedding_scene();
        budgeted.set_step_budget(Some(StepBudget {
            max_pair_substeps: 100,
            min_substeps: 4,
        }));
        budgeted.step(1.0 / 60.0);
        assert_eq!(budgeted.step_timing().substeps, 4);
        assert_eq!(budgeted.last_substep_shed(), 8);

        // Same trajectory without a budget: full count runs, and the pair
        // set is identical — shedding never drops contacts.
        let mut unbudgeted = dense_shedding_scene();
        unbudgeted.set_step_budget(None);
        unbudgeted.step(1.0 / 60.0);
        assert_eq!(unbudgeted.step_timing().substeps, 12);
        assert_eq!(unbudgeted.last_substep_shed(), 0);
        assert_eq!(
            budgeted.broadphase_stats().candidate_pairs,
            unbudgeted.broadphase_stats().candidate_pairs
        );
        assert!(budgeted.broadphase_stats().candidate_pairs > 0);
    }

    /// Dzhanibekov discriminant for the gyroscopic correction: half extents
    /// (0.2, 0.6, 0.4) give Ix > Iz > Iy, so body Z is the intermediate
    /// axis — spin about it must tumble end over end. Without the correction
    /// the spin axis stays world-fixed and the body Z rides a ~1.7deg cone
    /// (min dot ≈ 0.998), so a deep flip is unreachable by construction.
    #[test]
    fn gyroscopic_intermediate_axis_spin_tumbles() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let mut body = RigidBody::new_box(Vec3::ZERO, Vec3::new(0.2, 0.6, 0.4), 1.0);
        body.angular_velocity = Vec3::new(0.3, 0.0, 10.0);
        physics.add_body(body);

        let l0 = angular_momentum(&physics.bodies[0]);
        let e0 = rotational_energy(&physics.bodies[0]);
        let mut min_dot = 1.0f32;
        for _ in 0..300 {
            physics.step(1.0 / 60.0);
            let z_now = physics.bodies[0].orientation * Vec3::Z;
            min_dot = min_dot.min(z_now.dot(Vec3::Z));
        }
        let b = &physics.bodies[0];
        let dl = (angular_momentum(b) - l0).length() / l0.length();
        let de = ((rotational_energy(b) - e0) / e0).abs();
        eprintln!("dzhanibekov: min_dot={min_dot:.3} dL/L={dl:.4} dE/E={de:.4}");
        assert!(min_dot < -0.5, "no Dzhanibekov flip, min_dot={min_dot}");
        // Free motion has no torques, so these only drift by discretization
        // (measured 0.055/0.108 at 300 steps; bounds carry ~1.6x margin).
        assert!(dl < 0.09, "angular momentum drifted, dL/L={dl}");
        assert!(de < 0.18, "energy drifted, dE/E={de}");
    }

    /// Guard against overcorrection: spin about the major axis (body X here)
    /// is genuinely stable and must stay aligned.
    #[test]
    fn gyroscopic_major_axis_spin_stays_stable() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let mut body = RigidBody::new_box(Vec3::ZERO, Vec3::new(0.2, 0.6, 0.4), 1.0);
        body.angular_velocity = Vec3::new(10.0, 0.3, 0.0);
        physics.add_body(body);

        let mut min_dot = 1.0f32;
        for _ in 0..600 {
            physics.step(1.0 / 60.0);
            let x_now = physics.bodies[0].orientation * Vec3::X;
            min_dot = min_dot.min(x_now.dot(Vec3::X));
        }
        assert!(min_dot > 0.9, "major-axis spin wandered, min_dot={min_dot}");
    }

    /// Isotropic fast path: a spinning cube must come back bit-identical —
    /// the gyroscopic term is exactly zero there, so the skip gate fires.
    #[test]
    fn gyroscopic_isotropic_spin_is_untouched() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let mut body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0);
        body.angular_velocity = Vec3::new(1.0, 2.0, 3.0);
        physics.add_body(body);

        for _ in 0..60 {
            physics.step(1.0 / 60.0);
        }
        assert_eq!(
            physics.bodies[0].angular_velocity.to_array(),
            [1.0, 2.0, 3.0],
            "isotropic skip gate leaked"
        );
    }

    fn angular_momentum(body: &RigidBody) -> Vec3 {
        let w_body = body.orientation.conjugate() * body.angular_velocity;
        body.orientation * (body.inertia * w_body)
    }

    fn rotational_energy(body: &RigidBody) -> f32 {
        let w = body.orientation.conjugate() * body.angular_velocity;
        0.5 * body.inertia.dot(w * w)
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
    fn sat_cache_is_shared_by_parallel_narrowphase() {
        // 299 overlapping slow box pairs force the rayon path (>256) and the
        // SAT-eligible branch (near-zero speeds, substep 0). The lock-free
        // cache must fill on the first pass and replay identically.
        let mut bodies = Vec::new();
        for i in 0..300 {
            bodies.push(RigidBody::new_box(
                Vec3::new(i as f32 * 0.9, 0.0, 0.0),
                Vec3::splat(0.5),
                1.0,
            ));
        }
        let pairs: Vec<(usize, usize)> = (0..299).map(|i| (i, i + 1)).collect();
        let asleep = vec![false; bodies.len()];
        let cache = SatCache::default();
        let mut first: Vec<Manifold> = Vec::new();
        let mut pool = NarrowShardPool::default();
        detect_collisions_into(
            &bodies,
            &pairs,
            &asleep,
            1.0 / 240.0,
            &mut first,
            None,
            0,
            Some(&cache),
            &mut pool,
        );
        assert_eq!(
            cache.len(),
            pairs.len(),
            "every slow pair seeds the SAT cache"
        );
        assert!(!first.is_empty());
        let mut second: Vec<Manifold> = Vec::new();
        detect_collisions_into(
            &bodies,
            &pairs,
            &asleep,
            1.0 / 240.0,
            &mut second,
            None,
            0,
            Some(&cache),
            &mut pool,
        );
        assert_eq!(second.len(), first.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!((a.body_a, a.body_b), (b.body_a, b.body_b));
            assert!((a.normal - b.normal).length() < 1e-6);
            assert_eq!(a.point_count, b.point_count);
        }
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

    /// Contact events: a dropped box begins touching the floor on impact.
    #[test]
    fn contact_begin_fires_on_touch() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let floor = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let klein = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
        let mut begun = false;
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
            for e in physics.drain_contact_events() {
                if matches!(e.kind, ContactEventKind::Begin)
                    && ((e.body_a == floor && e.body_b == klein)
                        || (e.body_a == klein && e.body_b == floor))
                {
                    begun = true;
                }
            }
        }
        assert!(begun, "touchdown must emit Begin");
    }

    /// Contact events: a speculative near-miss (gap inside the margin, no
    /// touch) emits nothing — gameplay must not see begins without contact.
    #[test]
    fn contact_no_begin_for_speculative_gap() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        // Hovering 2 cm above the floor: inside the 5 cm speculative margin
        // (manifolds exist), zero velocity, zero gravity — never touches.
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.52, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
        for _ in 0..30 {
            physics.step(1.0 / 60.0);
            let events = physics.drain_contact_events();
            assert!(
                events.is_empty(),
                "gap contact must stay silent, got {events:?}"
            );
        }
    }

    /// Contact events: a fast impact records a Hit with the approach speed.
    #[test]
    fn contact_hit_reports_approach_speed() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let mut drop = RigidBody::new_box(Vec3::new(0.0, 6.0, 0.0), Vec3::splat(0.5), 1.0);
        drop.velocity = Vec3::new(0.0, -20.0, 0.0);
        physics.add_body(drop);
        let mut hit_speed = 0.0f32;
        for _ in 0..60 {
            physics.step(1.0 / 60.0);
            for e in physics.drain_contact_events() {
                if let ContactEventKind::Hit {
                    approach_speed,
                    normal,
                    ..
                } = e.kind
                {
                    hit_speed = hit_speed.max(approach_speed);
                    assert!(
                        normal.y.abs() > 0.9,
                        "hit normal must be vertical, got {normal:?}"
                    );
                }
            }
        }
        assert!(
            hit_speed > 10.0,
            "20 m/s impact must record a Hit, max approach {hit_speed}"
        );
    }

    /// Contact events: launching a resting box off the floor emits End.
    #[test]
    fn contact_end_fires_on_separation() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let floor = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let klein = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
        for _ in 0..90 {
            physics.step(1.0 / 60.0);
        }
        let _ = physics.drain_contact_events();
        physics.get_body_mut(klein).unwrap().velocity = Vec3::new(0.0, 10.0, 0.0);
        // Wake it: the test sets velocity directly (a driver would too).
        physics.wake_island(klein);
        let mut ended = false;
        for _ in 0..60 {
            physics.step(1.0 / 60.0);
            for e in physics.drain_contact_events() {
                if matches!(e.kind, ContactEventKind::End)
                    && ((e.body_a == floor && e.body_b == klein)
                        || (e.body_a == klein && e.body_b == floor))
                {
                    ended = true;
                }
            }
        }
        assert!(ended, "liftoff must emit End");
    }

    /// Contact events: a frozen (sleeping) contact emits no churn — sleep
    /// retains touch state silently.
    #[test]
    fn contact_frozen_pair_emits_no_churn() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let klein = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
        for _ in 0..150 {
            physics.step(1.0 / 60.0);
        }
        assert!(physics.is_asleep(klein), "box must settle");
        let _ = physics.drain_contact_events();
        for _ in 0..60 {
            physics.step(1.0 / 60.0);
            let events = physics.drain_contact_events();
            assert!(
                events.is_empty(),
                "frozen contact must stay silent, got {events:?}"
            );
        }
    }

    /// Contact events are deterministic run-to-run on identical scenes.
    #[test]
    fn contact_events_deterministic_across_runs() {
        fn run() -> Vec<ContactEvent> {
            let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
            physics.add_body(RigidBody::new_box(
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(5.0, 1.0, 5.0),
                0.0,
            ));
            let mut drop = RigidBody::new_box(Vec3::new(0.5, 5.0, 0.0), Vec3::splat(0.4), 1.0);
            drop.velocity = Vec3::new(-2.0, -15.0, 1.0);
            physics.add_body(drop);
            let mut all = Vec::new();
            for _ in 0..90 {
                physics.step(1.0 / 60.0);
                all.extend(physics.drain_contact_events());
            }
            all
        }
        assert_eq!(run(), run(), "contact events must be run-deterministic");
    }

    /// Anisotropic floor (ODE fdir1/mu/mu2 parity): slick along X, grippy
    /// along Z. A box kicked diagonally must keep its X slide while the Z
    /// component dies — separate per-axis Coulomb caps, one basis.
    #[test]
    fn aniso_floor_channels_sliding() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let mut floor =
            RigidBody::new_box(Vec3::new(0.0, -1.0, 0.0), Vec3::new(5.0, 1.0, 5.0), 0.0);
        floor.friction = 0.0;
        floor.friction_transverse = 1.0;
        floor.friction_dir = Some(Vec3::X);
        physics.add_body(floor);
        let mut b = RigidBody::new_box(Vec3::new(0.0, 2.0, 0.0), Vec3::splat(0.5), 1.0);
        b.friction = 0.0;
        b.velocity = Vec3::new(3.0, 0.0, 3.0);
        physics.add_body(b);
        // 60 steps ≈ 1 s: lands at ~0.55 s, then ~0.45 s of channeled
        // slide — short enough to stay on the 10 m floor (x ≈ 3 < 5).
        for _ in 0..60 {
            physics.step(1.0 / 60.0);
        }
        let v = physics.bodies[1].velocity;
        let y = physics.bodies[1].position.y;
        eprintln!("ANISO v={v:?} y={y}");
        assert!(v.x > 2.5, "slick axis must preserve slide, got {v:?}");
        assert!(v.z.abs() < 0.4, "grippy axis must kill slide, got {v:?}");
        assert!(
            v.y.abs() < 1.0 && (y - 0.5).abs() < 0.1,
            "box must rest ON the floor, not fall through, got {v:?} y={y}"
        );
    }

    /// Rolling resistance (MuJoCo parity): a ball with rolling friction
    /// must stop; the zero-coefficient control keeps rolling.
    #[test]
    fn rolling_resistance_stops_ball() {
        fn run(rolling: f32) -> (Vec3, f32) {
            let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
            // ±30 m floor: 120 steps at ~5 m/s stay on the slab, so both
            // balls are measured in rolling contact, never in free fall.
            let mut floor =
                RigidBody::new_box(Vec3::new(0.0, -1.0, 0.0), Vec3::new(30.0, 1.0, 30.0), 0.0);
            floor.rolling_friction = rolling;
            physics.add_body(floor);
            let mut ball = RigidBody::new_sphere(Vec3::new(0.0, 0.6, 0.0), 0.5, 1.0);
            ball.rolling_friction = rolling;
            // 240 steps: the damped ball stops (~190 steps at μr = 0.1);
            // the control is in pure rolling (zero slip) and coasts at
            // 8·5/7 ≈ 5.71 indefinitely — x ≈ 23 < 30 stays on the slab.
            ball.velocity = Vec3::new(8.0, 0.0, 0.0);
            physics.add_body(ball);
            for _ in 0..240 {
                physics.step(1.0 / 60.0);
            }
            (physics.bodies[1].velocity, physics.bodies[1].position.y)
        }
        let (stopped, ys) = run(0.1);
        let (rolling, yr) = run(0.0);
        eprintln!("ROLL stopped={stopped:?} y={ys} control={rolling:?} y={yr}");
        assert!(
            stopped.length() < 1.0 && (ys - 0.5).abs() < 0.1,
            "rolling friction must stop the ball ON the floor, got {stopped:?} y={ys}"
        );
        assert!(
            rolling.x > 3.0 && rolling.y.abs() < 1.0 && (yr - 0.5).abs() < 0.1,
            "zero rolling friction must keep it rolling on the floor, got {rolling:?} y={yr}"
        );
    }

    /// Torsional friction (MuJoCo parity): a sphere spinning about the
    /// contact normal (no slip, so slide friction is blind to it) must
    /// lose its spin; the control keeps spinning.
    #[test]
    fn torsion_friction_kills_spin() {
        fn run(torsion: f32) -> Vec3 {
            let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
            let mut floor =
                RigidBody::new_box(Vec3::new(0.0, -1.0, 0.0), Vec3::new(5.0, 1.0, 5.0), 0.0);
            floor.torsion_friction = torsion;
            physics.add_body(floor);
            let mut ball = RigidBody::new_sphere(Vec3::new(0.0, 0.6, 0.0), 0.5, 1.0);
            ball.torsion_friction = torsion;
            ball.angular_velocity = Vec3::new(0.0, 10.0, 0.0);
            physics.add_body(ball);
            for _ in 0..600 {
                physics.step(1.0 / 60.0);
            }
            physics.bodies[1].angular_velocity
        }
        let damped = run(0.05);
        let spinning = run(0.0);
        eprintln!("TORSION damped={damped:?} control={spinning:?}");
        assert!(
            damped.y.abs() < 2.0,
            "torsion friction must kill spin, got {damped:?}"
        );
        assert!(
            spinning.y.abs() > 5.0,
            "zero torsion must preserve spin, got {spinning:?}"
        );
    }

    /// Sphere-vs-box contact point (regression for the half-depth bug):
    /// with slide friction and NO rolling resistance, a rolling ball must
    /// converge to TRUE rolling (contact slip → 0). The old midpoint
    /// contact sat at half the radius depth, so the solver saw zero slip
    /// at a phantom "half-rolling" v = ω·r/2 and held it forever.
    #[test]
    fn rolling_converges_to_true_rolling_not_half() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(30.0, 1.0, 30.0),
            0.0,
        ));
        let mut ball = RigidBody::new_sphere(Vec3::new(0.0, 0.6, 0.0), 0.5, 1.0);
        ball.velocity = Vec3::new(8.0, 0.0, 0.0);
        physics.add_body(ball);
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
        }
        let b = &physics.bodies[1];
        // Still rolling (above the sleep threshold), so the slip below is
        // solver-converged, not frozen.
        assert!(
            b.velocity.x > 3.0,
            "ball must still be rolling, got {:?}",
            b.velocity
        );
        let slip = b.velocity.x + b.angular_velocity.z * 0.5;
        eprintln!(
            "ROLLTRUE v={:?} w={:?} slip={slip}",
            b.velocity, b.angular_velocity
        );
        assert!(
            slip.abs() < 0.2,
            "true rolling means zero contact slip, got {slip} (half-rolling phantom if ~v/2)"
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

    /// Travel gate, thin body: 10°/substep is below the old flat 15° gate, but
    /// R·angle = 1.51·0.175 = 0.26 > 0.5·0.2 = 0.1 arms CCD — blades tunnel
    /// easily, so they get the sweep early. Pre-rotated to 70° so the bar
    /// enters the target box (0, 1.1) mid-substep (separated at 70°,
    /// centerline inside at 80°).
    #[test]
    fn angular_gate_fires_below_15deg_for_thin_bodies() {
        let dt = 1.0 / 60.0;
        let mut mover = RigidBody::new_box(Vec3::ZERO, Vec3::new(1.5, 0.1, 0.1), 1.0);
        mover.orientation = Quat::from_rotation_z(70.0f32.to_radians());
        mover.angular_velocity = Vec3::Z * (10.0f32.to_radians() / dt);
        let target = RigidBody::new_box(Vec3::new(0.0, 1.1, 0.0), Vec3::new(0.2, 0.05, 0.2), 0.0);
        let bodies = [mover, target];

        let hit = find_angular_continuous_hit(&bodies, 0, Vec3::ZERO, dt)
            .expect("thin fast spinner must arm angular CCD below 15°/substep");
        assert!(hit.angular);
        assert!(hit.fraction > 0.0 && hit.fraction < 1.0);
        assert!(hit.contact.is_some(), "angular hit must carry its contact");
    }

    /// Travel gate, chunky body: 20°/substep exceeds the old flat 15° gate,
    /// but R·angle = 0.87·0.35 = 0.30 < 0.5·1.0 = 0.5 exempts the cube —
    /// the discrete phase resolves that travel, CCD would be pure overhead.
    #[test]
    fn angular_gate_spares_slow_chunky_spinners() {
        let dt = 1.0 / 60.0;
        let mut mover = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0);
        mover.angular_velocity = Vec3::Z * (20.0f32.to_radians() / dt);
        let target = RigidBody::new_box(Vec3::new(0.0, 1.1, 0.0), Vec3::new(0.2, 0.05, 0.2), 0.0);
        let bodies = [mover, target];

        assert!(
            find_angular_continuous_hit(&bodies, 0, Vec3::ZERO, dt).is_none(),
            "chunky slow spinner must skip angular CCD"
        );
    }

    /// Frictionless response, true 3D: ω=(10,0,10), lever +Y, normal −Z.
    /// The X-spin drives the approach, the Z-spin is tangential and must
    /// survive exactly; the old full stop would return zero here.
    #[test]
    fn angular_response_keeps_tangential_spin_in_3d() {
        let out = remove_angular_approach(
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::splat(2.0),
            Quat::IDENTITY,
            Vec3::Y,
            Vec3::NEG_Z,
        );
        assert!(
            (out - Vec3::new(0.0, 0.0, 10.0)).length() < 1e-6,
            "tangential spin must survive, got {out:?}"
        );
    }

    /// Frictionless response, planar head-on: exactly the old full stop
    /// (ω ∥ lever×normal always in-plane, so the constraint eats all spin).
    #[test]
    fn angular_response_stops_planar_head_on() {
        let out = remove_angular_approach(
            Vec3::Z * 10.0,
            Vec3::splat(2.0),
            Quat::IDENTITY,
            Vec3::X * 1.5,
            Vec3::NEG_Y,
        );
        assert!(out.length() < 1e-5, "planar head-on must stop, got {out:?}");
    }

    /// Stiff-lever cap: thin-bar inertia (huge I⁻¹ on X) with an off-axis
    /// contact. The exact projection would demand a wall impulse that
    /// injects spin energy (the blender); the cap holds energy neutral and
    /// leaves the rest to the discrete solver.
    #[test]
    fn angular_response_caps_energy_on_stiff_levers() {
        let w = Vec3::Z * 90.0;
        let inertia = Vec3::new(0.0067, 0.753, 0.753);
        let energy = |v: Vec3| 0.5 * inertia.dot(v * v);
        let out = remove_angular_approach(
            w,
            inertia,
            Quat::IDENTITY,
            Vec3::new(1.5, 0.05, 0.05),
            Vec3::NEG_Y,
        );
        assert!(out.is_finite(), "cap must never produce NaN, got {out:?}");
        assert!(
            energy(out) <= energy(w) + 1e-3,
            "cap must not inject energy: {} -> {}",
            energy(w),
            energy(out)
        );
        // ...while still reducing the approach (cap walks back along the
        // correction, never reverses it).
        let approach = |v: Vec3| v.cross(Vec3::new(1.5, 0.05, 0.05)).dot(Vec3::NEG_Y);
        assert!(
            approach(out) >= approach(w),
            "cap must not worsen the approach: {} -> {}",
            approach(w),
            approach(out)
        );
    }

    /// Unified impact, restitution: v=0, ω=(10,0,10), lever +Y, normal −Z,
    /// e=0.5. The tip approaches at −10 m/s, bounces to +5; the impulse
    /// couples into translation (v'=−10ẑ) and trims the driving X-spin
    /// (ω'=(5,0,10)) while the tangential Z-spin survives. Old code: v
    /// untouched, ω zeroed — a spinning bounce was impossible.
    #[test]
    fn ccd_impact_couples_spin_and_bounce() {
        let (v, w) = ccd_impact_velocity(
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 10.0),
            1.0,
            Vec3::splat(2.0),
            Quat::IDENTITY,
            Vec3::Y,
            Vec3::NEG_Z,
            0.5,
        );
        assert!(
            (v - Vec3::new(0.0, 0.0, -10.0)).length() < 1e-5,
            "bounce couples into translation, got {v:?}"
        );
        assert!(
            (w - Vec3::new(5.0, 0.0, 10.0)).length() < 1e-5,
            "driving spin trimmed, tangential kept, got {w:?}"
        );
        // Restitution identity on the contact point: vn' = −e·vn.
        let vn_after = (v + w.cross(Vec3::Y)).dot(Vec3::NEG_Z);
        assert!(
            (vn_after - 5.0).abs() < 1e-4,
            "vn' must be +5, got {vn_after}"
        );
    }

    /// Unified impact, inelastic limit: same setup with e=0 kills the
    /// contact approach exactly (vn' ≈ 0), dissipating total energy.
    #[test]
    fn ccd_impact_inelastic_kills_contact_approach() {
        let (v, w) = ccd_impact_velocity(
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 10.0),
            1.0,
            Vec3::splat(2.0),
            Quat::IDENTITY,
            Vec3::Y,
            Vec3::NEG_Z,
            0.0,
        );
        let vn_after = (v + w.cross(Vec3::Y)).dot(Vec3::NEG_Z);
        assert!(
            vn_after.abs() < 1e-4,
            "inelastic must stop the approach, got {vn_after}"
        );
        assert!(
            (v - Vec3::new(0.0, 0.0, -20.0 / 3.0)).length() < 1e-5,
            "got {v:?}"
        );
    }

    /// Unified impact, stiff lever: thin-bar inertia, e=0. The inv_mass
    /// floor in the denominator regularizes the impulse (no blender);
    /// total energy must drop, approach must vanish.
    #[test]
    fn ccd_impact_stiff_lever_dissipates() {
        let w0 = Vec3::Z * 90.0;
        let inertia = Vec3::new(0.0067, 0.753, 0.753);
        let lever = Vec3::new(1.5, 0.05, 0.05);
        let total = |v: Vec3, w: Vec3| 0.5 * v.dot(v) + 0.5 * inertia.dot(w * w);
        let (v, w) = ccd_impact_velocity(
            Vec3::ZERO,
            w0,
            1.0,
            inertia,
            Quat::IDENTITY,
            lever,
            Vec3::NEG_Y,
            0.0,
        );
        assert!(v.is_finite() && w.is_finite(), "got {v:?} {w:?}");
        assert!(
            total(v, w) <= total(Vec3::ZERO, w0) + 1e-2,
            "inelastic impact must dissipate: {} -> {}",
            total(Vec3::ZERO, w0),
            total(v, w)
        );
        let vn_after = (v + w.cross(lever)).dot(Vec3::NEG_Y);
        assert!(
            vn_after.abs() < 1e-2,
            "approach must vanish, got {vn_after}"
        );
    }

    /// Engine-level spin bounce: cube corner 10 cm above the floor (outside
    /// the 5 cm speculative margin, so no discrete phantom contact
    /// pre-empts the sweep), pure (40,0,40) spin, zero gravity, one
    /// substep. The corner outruns the discrete phase; CCD must clamp (no
    /// penetration) AND convert spin into an upward pop (old code:
    /// velocity stays zero, spin dies).
    #[test]
    fn ccd_spin_bounce_pops_upward() {
        let dt = 1.0 / 60.0;
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.set_substeps(1);
        let floor_pos = Vec3::new(0.0, -1.0, 0.0);
        let floor_half = Vec3::new(5.0, 1.0, 5.0);
        physics.add_body(RigidBody::new_box(floor_pos, floor_half, 0.0));
        let mover = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.6, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
        physics.get_body_mut(mover).unwrap().angular_velocity = Vec3::new(40.0, 0.0, 40.0);

        physics.step(dt);

        let body = physics.get_body(mover).expect("mover remains alive");
        assert!(
            obb_sat(
                body.position,
                Vec3::splat(0.5),
                body.orientation,
                floor_pos,
                floor_half,
                Quat::IDENTITY,
                1e-5
            )
            .is_none(),
            "CCD must not let the corner through the floor"
        );
        assert!(
            body.velocity.y > 2.0,
            "spin bounce must pop the body up, got {:?}",
            body.velocity
        );
    }

    /// Frictionless response, separating scrape: bit-identical passthrough.
    #[test]
    fn angular_response_ignores_separating_scrape() {
        let w = Vec3::Z * 10.0;
        let out =
            remove_angular_approach(w, Vec3::splat(2.0), Quat::IDENTITY, Vec3::X * 1.5, Vec3::Y);
        assert_eq!(out.to_array(), w.to_array());
    }

    /// Engine-level 3D graze: a cube corner outruns the discrete phase
    /// (0.82 m/substep vs a 1 cm gap) with a combined (40,0,40) spin. CCD
    /// must clamp before the wall face AND keep tangential spin (the old
    /// response returns exactly zero here).
    #[test]
    fn angular_graze_keeps_tangential_spin() {
        let dt = 1.0 / 60.0;
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.set_substeps(1);
        let wall_half = Vec3::new(0.49, 2.0, 2.0);
        let wall_pos = Vec3::new(1.0, 0.0, 0.0);
        physics.add_body(RigidBody::new_box(wall_pos, wall_half, 0.0));
        let mover = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
        physics.get_body_mut(mover).unwrap().angular_velocity = Vec3::new(40.0, 0.0, 40.0);

        physics.step(dt);

        let body = physics.get_body(mover).expect("mover remains alive");
        assert!(
            obb_sat(
                body.position,
                Vec3::splat(0.5),
                body.orientation,
                wall_pos,
                wall_half,
                Quat::IDENTITY,
                1e-5
            )
            .is_none(),
            "CCD must not let the corner through the wall"
        );
        assert!(
            body.angular_velocity.length() > 5.0,
            "tangential spin must survive the graze, got {:?}",
            body.angular_velocity
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

        let e_before = rotational_energy(physics.get_body(mover).expect("mover remains alive"));

        physics.step(dt);

        let body = physics.get_body(mover).expect("mover remains alive");
        // New contract (frictionless response + energy cap): the sweep still
        // clamps before tunneling, but a stiff out-of-plane lever no longer
        // dies to zero — the correction is capped at energy-neutral and the
        // discrete solver owns the remainder next substep.
        assert!(
            rotational_energy(body) <= e_before + 1e-3,
            "CCD response must never inject spin energy, before={e_before} after={} {:?}",
            rotational_energy(body),
            body.angular_velocity
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

    /// World-scale sleep: statics are born asleep (they never move), dynamics
    /// are born awake. Every frozen-pair skip keys off this from step one.
    #[test]
    fn statics_are_born_asleep_dynamics_awake() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let floor = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let free = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
        assert!(physics.is_asleep(floor), "static must be born asleep");
        assert!(!physics.is_asleep(free), "dynamic must be born awake");
    }

    /// World-scale sleep: a settled stack on a static floor survives a drop
    /// impact with the floor still asleep (wake_island on statics is a
    /// no-op) while the struck boxes wake and move.
    #[test]
    fn static_floor_stays_asleep_under_impact() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let floor = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let mut stack = Vec::new();
        for i in 0..3 {
            stack.push(physics.add_body(RigidBody::new_box(
                Vec3::new(0.0, 0.5 + i as f32, 0.0),
                Vec3::splat(0.5),
                1.0,
            )));
        }
        for _ in 0..150 {
            physics.step(1.0 / 60.0);
        }
        for &h in &stack {
            assert!(physics.is_asleep(h), "stack must settle before the drop");
        }
        let top_y = physics.get_body(stack[2]).unwrap().position.y;
        let mut drop = RigidBody::new_box(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
        drop.velocity = Vec3::new(0.0, -20.0, 0.0);
        physics.add_body(drop);
        // Rigid resting stacks barely displace under load — the observable
        // is the transient wake while the impact churns through, not the
        // final pose (it re-sleeps where it stood).
        let mut woke = false;
        for _ in 0..90 {
            physics.step(1.0 / 60.0);
            woke |= !physics.is_asleep(stack[2]);
        }
        assert!(
            physics.is_asleep(floor),
            "static floor must stay asleep through the impact"
        );
        let moved = physics.get_body(stack[2]).unwrap().position.y;
        assert!(
            woke,
            "struck stack must transiently wake (top {top_y} -> {moved})"
        );
    }

    /// World-scale sleep: a driven kinematic wall plows into a sleeping box
    /// — the sleeper must wake and be pushed, never ghosted through. (Before
    /// the asleep-flag cleanup, kinematic contacts were dropped in the
    /// active-manifold filter and sleepers were intangible to drivers.)
    #[test]
    fn kinematic_wall_wakes_and_pushes_sleeper() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let sleeper = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
        for _ in 0..90 {
            physics.step(1.0 / 60.0);
        }
        assert!(
            physics.is_asleep(sleeper),
            "box must settle before the plow"
        );
        let mut wall = RigidBody::new_box(Vec3::new(-3.0, 0.5, 0.0), Vec3::new(0.5, 1.0, 1.0), 1.0);
        wall.body_type = BodyType::Kinematic;
        let wall_h = physics.add_body(wall);
        // Driven body, done consistently: the driver sets positions AND the
        // matching velocity field (approach wake, margins and CCD all read
        // velocities — a zero-velocity teleport is invisible to them).
        for _ in 0..120 {
            let w = physics.get_body_mut(wall_h).unwrap();
            w.velocity = Vec3::new(2.0, 0.0, 0.0);
            w.position.x += 2.0 / 60.0;
            physics.step(1.0 / 60.0);
        }
        let pushed = physics.get_body(sleeper).unwrap().position.x;
        assert!(
            pushed > 0.5,
            "kinematic wall must push the sleeper (x={pushed}), not ghost through"
        );
    }

    /// Penetration wake, isolated: a box spawned 5 cm deep inside a sleeper
    /// with zero velocities must still wake it — the approach test is blind
    /// here (no velocity field), overlap is the only signal. (Teleporting
    /// drivers hit this path every frame.)
    #[test]
    fn teleport_overlap_wakes_sleeper_without_velocity() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let sleeper = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
        for _ in 0..30 {
            physics.step(1.0 / 60.0);
        }
        assert!(physics.is_asleep(sleeper), "box must sleep in zero-g");
        // Spawn overlapping: intruder bottom 5 cm inside the sleeper top.
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 1.45, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
        for _ in 0..5 {
            physics.step(1.0 / 60.0);
        }
        assert!(
            !physics.is_asleep(sleeper),
            "deep overlap must wake the sleeper even at zero approach speed"
        );
    }

    /// Kinematic CCD deferred with proof: thin SLEEPING victim (4 cm) vs a
    /// fast kinematic wall (10 m/s) at full 12 substeps. The speculative
    /// margin provably covers travel (margin >= v*dt always), so the
    /// discrete path already carries the victim without tunneling (rode
    /// 7.5 m here) — a dedicated kinematic sweep would only polish
    /// response quality, not close a capability gap.
    #[test]
    fn fast_kinematic_plow_carries_thin_sleeper() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let victim = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.02, 0.5, 0.5),
            1.0,
        ));
        for _ in 0..30 {
            physics.step(1.0 / 60.0);
        }
        assert!(physics.is_asleep(victim), "victim must sleep in zero-g");
        let mut wall = RigidBody::new_box(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(0.5, 1.0, 1.0), 1.0);
        wall.body_type = BodyType::Kinematic;
        let wall_h = physics.add_body(wall);
        for _ in 0..60 {
            let w = physics.get_body_mut(wall_h).unwrap();
            w.velocity = Vec3::new(10.0, 0.0, 0.0);
            w.position.x += 10.0 / 60.0;
            physics.step(1.0 / 60.0);
        }
        let vx = physics.get_body(victim).unwrap().position.x;
        assert!(vx > 1.0, "victim must ride the plow, not tunnel (x={vx})");
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
    fn solver_is_deterministic_across_runs() {
        // Same binary, two fresh engines: every hash map gets a fresh hasher
        // state per instance, so bit-identical snapshots prove iteration
        // order cannot leak into float state (not merely same-seed luck).
        // Uses a small heterogeneous scene (stacked boxes, a resting
        // sphere and a fast drop), so caches, islands and CCD all engage.
        fn build() -> BuiltinPhysicsEngine {
            let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
            physics.add_body(RigidBody::new_box(
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(10.0, 1.0, 10.0),
                0.0,
            ));
            for i in 0..4 {
                physics.add_body(RigidBody::new_box(
                    Vec3::new(0.0, 0.5 + i as f32 * 1.02, 0.0),
                    Vec3::splat(0.5),
                    1.0,
                ));
            }
            physics.add_body(RigidBody::new_sphere(Vec3::new(3.0, 6.0, 0.0), 0.5, 1.0));
            let mut fast = RigidBody::new_box(Vec3::new(-3.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
            fast.velocity = Vec3::new(0.0, -30.0, 0.0);
            physics.add_body(fast);
            physics
        }
        #[allow(clippy::type_complexity)]
        fn snapshot(
            physics: &BuiltinPhysicsEngine,
        ) -> Vec<([u32; 3], [u32; 4], [u32; 3], [u32; 3])> {
            physics
                .bodies
                .iter()
                .map(|b| {
                    (
                        b.position.to_array().map(f32::to_bits),
                        b.orientation.to_array().map(f32::to_bits),
                        b.velocity.to_array().map(f32::to_bits),
                        b.angular_velocity.to_array().map(f32::to_bits),
                    )
                })
                .collect()
        }
        let mut first = build();
        let mut second = build();
        for _ in 0..120 {
            first.step(1.0 / 60.0);
            second.step(1.0 / 60.0);
        }
        assert_eq!(snapshot(&first), snapshot(&second));
    }

    /// Canonical cross-platform determinism snapshot (Box3D-level claim):
    /// a fixed heterogeneous scene (stack, sphere, fast drop, jointed
    /// pendulum) stepped 120 times, compared bit-for-bit against the
    /// checked-in `tests/data/determinism_snapshot.hex` generated on ARM.
    /// CI runs this on x86_64: any float-contraction or codegen drift
    /// (including LLVM fusing mul+add into fma, which stable rustc cannot
    /// disable) fails loudly here instead of silently diverging.
    /// Re-baseline ONLY for intentional solver changes: run
    /// `determinism_snapshot_regenerate` (ignored), inspect the diff, commit.
    fn determinism_snapshot_scene() -> BuiltinPhysicsEngine {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(10.0, 1.0, 10.0),
            0.0,
        ));
        for i in 0..4 {
            physics.add_body(RigidBody::new_box(
                Vec3::new(0.0, 0.5 + i as f32 * 1.02, 0.0),
                Vec3::splat(0.5),
                1.0,
            ));
        }
        physics.add_body(RigidBody::new_sphere(Vec3::new(3.0, 6.0, 0.0), 0.5, 1.0));
        let mut fast = RigidBody::new_box(Vec3::new(-3.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
        fast.velocity = Vec3::new(0.0, -30.0, 0.0);
        physics.add_body(fast);
        let anchor = physics.add_body(RigidBody::new_box(
            Vec3::new(6.0, 3.0, 0.0),
            Vec3::splat(0.5),
            0.0,
        ));
        let arm = physics.add_body(RigidBody::new_box(
            Vec3::new(6.0, 1.0, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
        physics.add_joint(
            anchor,
            arm,
            JointKind::Ball {
                local_anchor_a: Vec3::new(0.0, -1.0, 0.0),
                local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
            },
        );
        physics
    }

    fn determinism_snapshot_render(physics: &BuiltinPhysicsEngine) -> String {
        let mut out = format!(
            "ornis-determinism-snapshot v1 bodies={} steps=120 dt=0.0166667\n",
            physics.bodies.len()
        );
        for b in &physics.bodies {
            let mut first = true;
            for x in b
                .position
                .to_array()
                .into_iter()
                .chain(b.orientation.to_array())
                .chain(b.velocity.to_array())
                .chain(b.angular_velocity.to_array())
            {
                if !first {
                    out.push(' ');
                }
                first = false;
                out.push_str(&format!("{:08x}", x.to_bits()));
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn determinism_snapshot_matches_canonical() {
        let mut physics = determinism_snapshot_scene();
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
        }
        let expected = include_str!("../tests/data/determinism_snapshot.hex");
        assert_eq!(
            determinism_snapshot_render(&physics),
            expected,
            "simulation bits drifted: intentional solver change? re-baseline via \
             determinism_snapshot_regenerate, else float/codegen drift"
        );
    }

    #[test]
    #[ignore]
    fn determinism_snapshot_regenerate() {
        let mut physics = determinism_snapshot_scene();
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
        }
        let path = format!(
            "{}/tests/data/determinism_snapshot.hex",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::write(path, determinism_snapshot_render(&physics)).unwrap();
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
                    limit: None,
                    motor: None,
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

    /// Prismatic slider: block kicked along X on a horizontal rail under
    /// gravity must travel freely along the axis while the point-to-line
    /// constraints carry its weight (no sag) and lock the spin.
    #[test]
    fn prismatic_slider_travels_on_axis_without_sag() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
        let block = physics.add_body(RigidBody::new_box(
            Vec3::ZERO,
            Vec3::new(0.2, 0.2, 0.2),
            1.0,
        ));
        physics
            .add_joint(
                anchor,
                block,
                JointKind::Prismatic {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                    local_axis_a: Vec3::X,
                    local_axis_b: Vec3::X,
                    limit: None,
                    motor: None,
                },
            )
            .expect("valid joint");
        physics.get_body_mut(block).unwrap().velocity = Vec3::new(3.0, 0.0, 0.0);
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
        }
        let b = physics.get_body(block).unwrap();
        assert!(b.position.x > 1.0, "slider must travel, x={}", b.position.x);
        assert!(
            b.position.y.abs() < 0.05 && b.position.z.abs() < 0.05,
            "slider must not sag off axis: {:?}",
            b.position
        );
        let wx = b.orientation * Vec3::X;
        assert!(
            wx.dot(Vec3::X) > 0.995,
            "slider must not twist off axis: {wx:?}"
        );
    }

    /// Prismatic limit: a fast block stops at the window end and stays.
    #[test]
    fn prismatic_limit_blocks_travel_past_bounds() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
        let block = physics.add_body(RigidBody::new_box(
            Vec3::ZERO,
            Vec3::new(0.2, 0.2, 0.2),
            1.0,
        ));
        physics
            .add_joint(
                anchor,
                block,
                JointKind::Prismatic {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                    local_axis_a: Vec3::X,
                    local_axis_b: Vec3::X,
                    limit: Some(PrismaticLimit {
                        min: -1.0,
                        max: 1.0,
                    }),
                    motor: None,
                },
            )
            .expect("valid joint");
        physics.get_body_mut(block).unwrap().velocity = Vec3::new(10.0, 0.0, 0.0);
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
        }
        let b = physics.get_body(block).unwrap();
        assert!(
            (b.position.x - 1.0).abs() < 0.3,
            "slider must stop at the upper bound, x={}",
            b.position.x
        );
        assert!(
            b.velocity.x.abs() < 1.0,
            "limit must kill the slide speed: {:?}",
            b.velocity
        );
    }

    /// Prismatic motor: a resting block spins up to the target slide speed.
    #[test]
    fn prismatic_motor_drives_to_target_speed() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
        let block = physics.add_body(RigidBody::new_box(
            Vec3::ZERO,
            Vec3::new(0.2, 0.2, 0.2),
            1.0,
        ));
        physics
            .add_joint(
                anchor,
                block,
                JointKind::Prismatic {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                    local_axis_a: Vec3::X,
                    local_axis_b: Vec3::X,
                    limit: None,
                    motor: Some(PrismaticMotor {
                        target_speed: 2.0,
                        max_force: 100.0,
                    }),
                },
            )
            .expect("valid joint");
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
        }
        let b = physics.get_body(block).unwrap();
        assert!(
            (b.velocity.x - 2.0).abs() < 0.3,
            "motor must reach target speed: {:?}",
            b.velocity
        );
    }

    /// Prismatic assembly with a twisted block: the axis-alignment pass
    /// pulls the slide axis back parallel.
    #[test]
    fn prismatic_misaligned_axes_realign() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
        let mut block = RigidBody::new_box(Vec3::ZERO, Vec3::new(0.2, 0.2, 0.2), 1.0);
        block.orientation = Quat::from_rotation_y(10.0f32.to_radians());
        let block = physics.add_body(block);
        physics
            .add_joint(
                anchor,
                block,
                JointKind::Prismatic {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                    local_axis_a: Vec3::X,
                    local_axis_b: Vec3::X,
                    limit: None,
                    motor: None,
                },
            )
            .expect("valid joint");
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
        }
        let b = physics.get_body(block).unwrap();
        let wx = b.orientation * Vec3::X;
        assert!(wx.dot(Vec3::X) > 0.998, "slide axes must realign: {wx:?}");
    }

    #[test]
    fn revolute_limit_blocks_travel_past_bounds() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
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
                    local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
                    local_axis_a: Vec3::Z,
                    local_axis_b: Vec3::Z,
                    limit: Some(crate::joint::RevoluteLimit {
                        min: -0.2,
                        max: 0.2,
                    }),
                    motor: None,
                },
            )
            .expect("valid joint");
        // Hard kick: a free hinge would swing to ~1.4 rad.
        physics.get_body_mut(arm).unwrap().velocity = Vec3::new(4.0, 0.0, 0.0);
        let mut max_travel = 0.0f32;
        for _ in 0..300 {
            physics.step(1.0 / 60.0);
            let a = physics.get_body(anchor).unwrap();
            let b = physics.get_body(arm).unwrap();
            let travel = super::joints::hinge_twist(a.orientation, b.orientation, Vec3::Z).abs();
            max_travel = max_travel.max(travel);
        }
        assert!(max_travel > 0.05, "arm never moved: {max_travel}");
        assert!(max_travel < 0.45, "limit failed to hold: {max_travel}");
    }

    #[test]
    fn revolute_motor_spins_up_to_target_speed() {
        // Zero gravity: only the motor drives the hinge.
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
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
                    local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
                    local_axis_a: Vec3::Z,
                    local_axis_b: Vec3::Z,
                    limit: None,
                    motor: Some(crate::joint::RevoluteMotor {
                        target_speed: 3.0,
                        max_torque: 50.0,
                    }),
                },
            )
            .expect("valid joint");
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
        }
        let a = physics.get_body(anchor).unwrap();
        let b = physics.get_body(arm).unwrap();
        let w = (b.angular_velocity - a.angular_velocity).dot(Vec3::Z);
        assert!((w - 3.0).abs() < 0.3, "motor missed target speed: {w}");
        // A starved torque budget must NOT reach the target (clamp binds).
        let mut weak = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let anchor = weak.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
        let arm = weak.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.1, 1.0, 0.1),
            1.0,
        ));
        weak.add_joint(
            anchor,
            arm,
            JointKind::Revolute {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: None,
                motor: Some(crate::joint::RevoluteMotor {
                    target_speed: 3.0,
                    max_torque: 0.05,
                }),
            },
        )
        .expect("valid joint");
        for _ in 0..120 {
            weak.step(1.0 / 60.0);
        }
        let a = weak.get_body(anchor).unwrap();
        let b = weak.get_body(arm).unwrap();
        let w = (b.angular_velocity - a.angular_velocity).dot(Vec3::Z);
        assert!(w < 1.5, "torque clamp did not bind: {w}");
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
