//! SIMD-wide contact solver (G7).
//!
//! Models Box3D's `b3ContactConstraintWide` (bepu-inspired): up to 4
//! single-point contact constraints are packed into one batch in SoA
//! (structure-of-arrays) layout — every component is a contiguous `[f32; 4]`
//! — so the inner solver loop walks 4 contacts with cache-friendly stride-1
//! loads, and LLVM can vectorize the lane-wise arithmetic into SSE/AVX/NEON.
//!
//! The module is the target of the **P2 narrow-cache follow-up**: after the
//! incremental broadphase (P1), the per-pair SAT work it consumes is the
//! remaining ~8 ms bottleneck. A persistent `NarrowCacheEntry`/`SatCacheEntry`
//! cache is in progress; the wide path itself stays opt-in via
//! `set_wide_solver` and the GPU variant via `gpu` feature.
//!
//! # Determinism (Strong Confluence)
//!
//! Batches are formed greedily over **consecutive** single-point manifolds
//! whose body sets are **disjoint**. The global Gauss-Seidel order is
//! therefore preserved exactly: every contact is solved in its original
//! relative position in the sequence, and lanes inside one batch never share
//! a body — the simultaneous lane solve equals the sequential solve, so the
//! wide path is bit-deterministic for any thread count.
//!
//! The wide path is *not* guaranteed bit-identical to the scalar solver:
//! effective masses and inertia applications are precomputed (world-space
//! inverse inertia tensors instead of per-call quaternion rotations), which
//! may round ±1 ulp differently. Physically equivalent, deterministic, but
//! numerically a distinct variant — keep the scalar path
//! (`set_wide_solver(false)`) when bit-exact reproduction of the old
//! numbers is required.

use glam::{Mat3, Vec3};

use crate::body::RigidBody;
use crate::engine::{Manifold, ManifoldState};

// ---------------------------------------------------------------------------
// 4-lane SoA primitives
// ---------------------------------------------------------------------------

/// One 4-wide scalar in SoA layout: lane `l` is `.0[l]`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Fx4(pub [f32; 4]);

impl Fx4 {
    #[inline]
    pub fn zero() -> Self {
        Self([0.0; 4])
    }

    #[inline]
    pub fn lane(&self, l: usize) -> f32 {
        self.0[l]
    }

    #[inline]
    pub fn set_lane(&mut self, l: usize, v: f32) {
        self.0[l] = v;
    }
}

/// One 4-wide Vec3 in SoA layout: lane `l` is `(x[l], y[l], z[l])`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Vec3x4 {
    pub x: Fx4,
    pub y: Fx4,
    pub z: Fx4,
}

impl Vec3x4 {
    #[inline]
    pub fn zero() -> Self {
        Self {
            x: Fx4::zero(),
            y: Fx4::zero(),
            z: Fx4::zero(),
        }
    }

    #[inline]
    pub fn lane(&self, l: usize) -> Vec3 {
        Vec3::new(self.x.lane(l), self.y.lane(l), self.z.lane(l))
    }

    #[inline]
    pub fn set_lane(&mut self, l: usize, v: Vec3) {
        self.x.set_lane(l, v.x);
        self.y.set_lane(l, v.y);
        self.z.set_lane(l, v.z);
    }
}

// ---------------------------------------------------------------------------
// Wide batch
// ---------------------------------------------------------------------------

/// Batch of up to 4 single-point contact constraints with DISJOINT body
/// sets (guaranteed by `build_solver_steps`), solved in SoA lanes.
pub(crate) struct WideBatch {
    /// Active lane count (1..=4); lanes >= count are masked to body 0.
    count: usize,
    /// Manifold-state index per lane, for writing accumulated impulses back.
    state_idx: [usize; 4],
    /// Island-local body indices per lane.
    idx_a: [usize; 4],
    idx_b: [usize; 4],

    // --- Constant per-solve geometry (precomputed at build time) ---
    n: Vec3x4,
    ra: Vec3x4,
    rb: Vec3x4,
    /// Precomputed `ra × n`, `rb × n`.
    ra_n: Vec3x4,
    rb_n: Vec3x4,
    /// Fixed tangent basis per lane (from ManifoldState).
    t1: Vec3x4,
    t2: Vec3x4,

    // --- Constant per-solve scalars ---
    inv_ma: Fx4,
    inv_mb: Fx4,
    /// 1 / effective normal mass (0 for inert lanes).
    inv_k_n: Fx4,
    /// 1 / effective friction mass per tangent axis.
    inv_k_t1: Fx4,
    inv_k_t2: Fx4,
    /// `n * inv_mass` — linear impulse application factors.
    apply_n_a: Vec3x4,
    apply_n_b: Vec3x4,
    /// `I⁻¹_world · (ra×n)` — angular impulse application factors.
    apply_w_a: Vec3x4,
    apply_w_b: Vec3x4,
    /// World-space inverse inertia matrix rows per lane: `w_mat[lane][row]`.
    wa_mat: [[Vec3; 3]; 4],
    wb_mat: [[Vec3; 3]; 4],
    target: Fx4,
    mu: Fx4,
    /// One-shot restitution bias per lane.
    bias: Fx4,

    // --- Solver state (mutated per iteration) ---
    acc: Fx4,
    acc_f1: Fx4,
    acc_f2: Fx4,
    va: Vec3x4,
    vb: Vec3x4,
    wa: Vec3x4,
    wb: Vec3x4,
}

impl WideBatch {
    /// World-space inverse inertia tensor rows for one body:
    /// `I⁻¹_world = R · diag(1/Ix, 1/Iy, 1/Iz) · Rᵀ`, as 3 row vectors.
    #[inline]
    fn world_inertia(rot: glam::Quat, inertia: Vec3) -> [Vec3; 3] {
        let inv = Vec3::new(
            if inertia.x > 0.0 {
                1.0 / inertia.x
            } else {
                0.0
            },
            if inertia.y > 0.0 {
                1.0 / inertia.y
            } else {
                0.0
            },
            if inertia.z > 0.0 {
                1.0 / inertia.z
            } else {
                0.0
            },
        );
        let m = Mat3::from_quat(rot);
        let col_x = m.x_axis * inv.x;
        let col_y = m.y_axis * inv.y;
        let col_z = m.z_axis * inv.z;
        // rows of R · (D·Rᵀ)
        [
            m.x_axis * col_x.x + m.y_axis * col_y.x + m.z_axis * col_z.x,
            m.x_axis * col_x.y + m.y_axis * col_y.y + m.z_axis * col_z.y,
            m.x_axis * col_x.z + m.y_axis * col_y.z + m.z_axis * col_z.z,
        ]
    }

    #[inline]
    fn matvec(m: &[Vec3; 3], v: Vec3) -> Vec3 {
        Vec3::new(m[0].dot(v), m[1].dot(v), m[2].dot(v))
    }

    /// Build a wide batch from 1..=4 single-point manifolds with disjoint
    /// body sets. `items` = (state index, manifold, state) triples.
    /// Lanes beyond `count` are masked to body 0 (a static placeholder).
    pub(crate) fn build(
        items: &[(usize, &Manifold, &ManifoldState)],
        bodies: &[RigidBody],
    ) -> Self {
        let count = items.len();
        debug_assert!((1..=4).contains(&count));
        let mut b = WideBatch {
            count,
            state_idx: [0; 4],
            idx_a: [0; 4],
            idx_b: [0; 4],
            n: Vec3x4::zero(),
            ra: Vec3x4::zero(),
            rb: Vec3x4::zero(),
            ra_n: Vec3x4::zero(),
            rb_n: Vec3x4::zero(),
            t1: Vec3x4::zero(),
            t2: Vec3x4::zero(),
            inv_ma: Fx4::zero(),
            inv_mb: Fx4::zero(),
            inv_k_n: Fx4::zero(),
            inv_k_t1: Fx4::zero(),
            inv_k_t2: Fx4::zero(),
            apply_n_a: Vec3x4::zero(),
            apply_n_b: Vec3x4::zero(),
            apply_w_a: Vec3x4::zero(),
            apply_w_b: Vec3x4::zero(),
            wa_mat: [[Vec3::ZERO; 3]; 4],
            wb_mat: [[Vec3::ZERO; 3]; 4],
            target: Fx4::zero(),
            mu: Fx4::zero(),
            bias: Fx4::zero(),
            acc: Fx4::zero(),
            acc_f1: Fx4::zero(),
            acc_f2: Fx4::zero(),
            va: Vec3x4::zero(),
            vb: Vec3x4::zero(),
            wa: Vec3x4::zero(),
            wb: Vec3x4::zero(),
        };

        for (l, item) in items.iter().enumerate() {
            b.fill_lane(l, *item, bodies);
        }

        // Mask inactive lanes to body 0 — a static with inv_mass 0, so all
        // deltas and writes are zero. Body-set disjointness is preserved by
        // construction (build_solver_steps).
        for l in count..4 {
            b.idx_a[l] = 0;
            b.idx_b[l] = 0;
        }
        b
    }

    /// Fill one lane from a single-point manifold (state index, manifold,
    /// state) and its two bodies. All per-lane constant factors are
    /// precomputed here so the hot iteration loop stays branch-light.
    fn fill_lane(
        &mut self,
        l: usize,
        (si, m, st): (usize, &Manifold, &ManifoldState),
        bodies: &[RigidBody],
    ) {
        debug_assert_eq!(st.count, 1, "wide batch takes single-point manifolds only");
        let (i, j) = (st.i, st.j);
        let p = m.points[0].world_point;
        let ra = p - bodies[i].position;
        let rb = p - bodies[j].position;
        let n = m.normal;

        self.state_idx[l] = si;
        self.idx_a[l] = i;
        self.idx_b[l] = j;
        self.n.set_lane(l, n);
        self.ra.set_lane(l, ra);
        self.rb.set_lane(l, rb);
        self.t1.set_lane(l, st.t1);
        self.t2.set_lane(l, st.t2);
        self.target.set_lane(l, st.target[0]);
        self.mu.set_lane(l, st.mu);
        self.bias.set_lane(l, st.bias[0]);
        self.acc.set_lane(l, st.acc[0]);

        let (a, bb) = (&bodies[i], &bodies[j]);
        self.inv_ma.set_lane(l, a.inv_mass);
        self.inv_mb.set_lane(l, bb.inv_mass);
        self.va.set_lane(l, a.velocity);
        self.vb.set_lane(l, bb.velocity);
        self.wa.set_lane(l, a.angular_velocity);
        self.wb.set_lane(l, bb.angular_velocity);

        // --- Precompute the constant factors ---
        let ra_n = ra.cross(n);
        let rb_n = rb.cross(n);
        self.ra_n.set_lane(l, ra_n);
        self.rb_n.set_lane(l, rb_n);

        let wa = Self::world_inertia(a.orientation, a.inertia);
        let wb = Self::world_inertia(bb.orientation, bb.inertia);
        self.wa_mat[l] = wa;
        self.wb_mat[l] = wb;

        let total_inv = a.inv_mass + bb.inv_mass;
        let k_n = total_inv + ra_n.dot(Self::matvec(&wa, ra_n)) + rb_n.dot(Self::matvec(&wb, rb_n));
        let ra_t1 = ra.cross(st.t1);
        let rb_t1 = rb.cross(st.t1);
        let ra_t2 = ra.cross(st.t2);
        let rb_t2 = rb.cross(st.t2);
        let k_t1 =
            total_inv + ra_t1.dot(Self::matvec(&wa, ra_t1)) + rb_t1.dot(Self::matvec(&wb, rb_t1));
        let k_t2 =
            total_inv + ra_t2.dot(Self::matvec(&wa, ra_t2)) + rb_t2.dot(Self::matvec(&wb, rb_t2));
        self.inv_k_n
            .set_lane(l, if k_n >= 1e-10 { 1.0 / k_n } else { 0.0 });
        self.inv_k_t1
            .set_lane(l, if k_t1 >= 1e-10 { 1.0 / k_t1 } else { 0.0 });
        self.inv_k_t2
            .set_lane(l, if k_t2 >= 1e-10 { 1.0 / k_t2 } else { 0.0 });

        self.apply_n_a.set_lane(l, n * a.inv_mass);
        self.apply_n_b.set_lane(l, n * bb.inv_mass);
        self.apply_w_a.set_lane(l, Self::matvec(&wa, ra_n));
        self.apply_w_b.set_lane(l, Self::matvec(&wb, rb_n));
    }

    /// Refresh velocities from the body array (called at the start of every
    /// solver step, so lanes see the updates of all earlier steps).
    pub(crate) fn gather(&mut self, bodies: &[RigidBody]) {
        for l in 0..self.count {
            let (i, j) = (self.idx_a[l], self.idx_b[l]);
            self.va.set_lane(l, bodies[i].velocity);
            self.wa.set_lane(l, bodies[i].angular_velocity);
            self.vb.set_lane(l, bodies[j].velocity);
            self.wb.set_lane(l, bodies[j].angular_velocity);
        }
    }

    /// Write velocities back to the body array. Only lanes with positive
    /// inverse mass move (statics and masked lanes are untouched).
    pub(crate) fn scatter(&self, bodies: &mut [RigidBody]) {
        for l in 0..self.count {
            if self.inv_ma.lane(l) > 0.0 {
                bodies[self.idx_a[l]].velocity = self.va.lane(l);
                bodies[self.idx_a[l]].angular_velocity = self.wa.lane(l);
            }
            if self.inv_mb.lane(l) > 0.0 {
                bodies[self.idx_b[l]].velocity = self.vb.lane(l);
                bodies[self.idx_b[l]].angular_velocity = self.wb.lane(l);
            }
        }
    }

    /// Write the accumulated normal impulses back into `states` (the warm
    /// cache persistence reads them after the iterations).
    pub(crate) fn write_back_acc(&self, states: &mut [ManifoldState]) {
        for l in 0..self.count {
            states[self.state_idx[l]].acc[0] = self.acc.lane(l);
        }
    }

    /// One Gauss-Seidel iteration over all lanes: normal impulse (projected
    /// Gauss-Seidel, scalar form) then Coulomb friction in the fixed tangent
    /// basis. Mirrors the scalar single-point code in
    /// `solve_island_velocity`, lane by lane, with the precomputed factors.
    pub(crate) fn solve_iteration(&mut self) {
        for l in 0..self.count {
            let max_friction = self.mu.lane(l) * self.acc.lane(l);
            self.solve_lane_normal(l);
            // Friction (Coulomb) along the fixed tangent basis. The relative
            // velocity is re-measured AFTER the normal impulse, exactly like
            // the scalar solver.
            let f_imp = self.solve_lane_friction(l, max_friction);
            if f_imp.length_squared() > 1e-24 {
                let inv_ma = self.inv_ma.lane(l);
                let inv_mb = self.inv_mb.lane(l);
                let ra = self.ra.lane(l);
                let rb = self.rb.lane(l);
                self.va.set_lane(l, self.va.lane(l) - f_imp * inv_ma);
                self.vb.set_lane(l, self.vb.lane(l) + f_imp * inv_mb);
                self.wa.set_lane(
                    l,
                    self.wa.lane(l) - Self::matvec(&self.wa_mat[l], ra.cross(f_imp)),
                );
                self.wb.set_lane(
                    l,
                    self.wb.lane(l) + Self::matvec(&self.wb_mat[l], rb.cross(f_imp)),
                );
            }
        }
    }

    /// Normal-direction projected Gauss-Seidel update for one lane using the
    /// precomputed effective-mass inverses and application factors.
    fn solve_lane_normal(&mut self, l: usize) {
        let n = self.n.lane(l);
        let ra = self.ra.lane(l);
        let rb = self.rb.lane(l);
        let rel = point_velocity(self.vb.lane(l), self.wb.lane(l), rb)
            - point_velocity(self.va.lane(l), self.wa.lane(l), ra);
        let vn = rel.dot(n);
        let lambda = (self.target.lane(l) - vn) * self.inv_k_n.lane(l);
        let new_acc = (self.acc.lane(l) + lambda).max(0.0);
        let delta = new_acc - self.acc.lane(l);
        self.acc.set_lane(l, new_acc);
        if delta.abs() > 1e-12 {
            // apply_impulse: linear + angular, using the precomputed
            // factors (n·inv_mass and I⁻¹_world·(ra×n)).
            self.va
                .set_lane(l, self.va.lane(l) - self.apply_n_a.lane(l) * delta);
            self.vb
                .set_lane(l, self.vb.lane(l) + self.apply_n_b.lane(l) * delta);
            self.wa
                .set_lane(l, self.wa.lane(l) - self.apply_w_a.lane(l) * delta);
            self.wb
                .set_lane(l, self.wb.lane(l) + self.apply_w_b.lane(l) * delta);
        }
    }

    /// Coulomb friction for one lane along both fixed tangents; returns the
    /// total friction impulse to apply.
    fn solve_lane_friction(&mut self, l: usize, max_friction: f32) -> Vec3 {
        let rb = self.rb.lane(l);
        let rel = point_velocity(self.vb.lane(l), self.wb.lane(l), rb)
            - point_velocity(self.va.lane(l), self.wa.lane(l), self.ra.lane(l));
        let mut f_imp = Vec3::ZERO;

        // Axis 1.
        if self.inv_k_t1.lane(l) != 0.0 {
            let t = self.t1.lane(l);
            let vt = rel.dot(t);
            let lambda_t = -vt * self.inv_k_t1.lane(l);
            let new_t = clamp_friction_impulse_wide(
                self.acc_f1.lane(l) + lambda_t,
                self.acc_f2.lane(l),
                max_friction,
            );
            f_imp += t * (new_t - self.acc_f1.lane(l));
            self.acc_f1.set_lane(l, new_t);
        }
        // Axis 2.
        if self.inv_k_t2.lane(l) != 0.0 {
            let t = self.t2.lane(l);
            let vt = rel.dot(t);
            let lambda_t = -vt * self.inv_k_t2.lane(l);
            let new_t = clamp_friction_impulse_wide(
                self.acc_f2.lane(l) + lambda_t,
                self.acc_f1.lane(l),
                max_friction,
            );
            f_imp += t * (new_t - self.acc_f2.lane(l));
            self.acc_f2.set_lane(l, new_t);
        }
        f_imp
    }

    /// One-shot restitution stage (mirrors the scalar post-iteration pass).
    /// Applied after all iterations, against the final gathered velocities.
    pub(crate) fn solve_restitution(&mut self) {
        for l in 0..self.count {
            let bias = self.bias.lane(l);
            if bias <= 0.0 {
                continue;
            }
            let n = self.n.lane(l);
            let ra = self.ra.lane(l);
            let rb = self.rb.lane(l);
            let k_eff_inv = self.inv_k_n.lane(l);
            if k_eff_inv == 0.0 {
                continue;
            }
            let rel = point_velocity(self.vb.lane(l), self.wb.lane(l), rb)
                - point_velocity(self.va.lane(l), self.wa.lane(l), ra);
            let vn = rel.dot(n);
            let lambda = (bias - vn) * k_eff_inv;
            if lambda > 0.0 {
                self.va
                    .set_lane(l, self.va.lane(l) - self.apply_n_a.lane(l) * lambda);
                self.vb
                    .set_lane(l, self.vb.lane(l) + self.apply_n_b.lane(l) * lambda);
                self.wa
                    .set_lane(l, self.wa.lane(l) - self.apply_w_a.lane(l) * lambda);
                self.wb
                    .set_lane(l, self.wb.lane(l) + self.apply_w_b.lane(l) * lambda);
            }
        }
    }
}

/// Coulomb cone projection for one wide-batch friction axis given the
/// accumulated impulse on the perpendicular axis.
fn clamp_friction_impulse_wide(new_t: f32, other: f32, max_friction: f32) -> f32 {
    let len = (new_t * new_t + other * other).sqrt();
    if len > max_friction && len > 1e-12 {
        new_t * (max_friction / len)
    } else {
        new_t
    }
}

/// Velocity of a body at a world-space point (linear + angular part).
#[inline]
fn point_velocity(v: Vec3, w: Vec3, r: Vec3) -> Vec3 {
    v + w.cross(r)
}

/// One step of the solver sequence: either a SIMD-wide batch (single-point
/// manifolds, disjoint bodies) or a scalar manifold (multi-point block LCP).
/// Steps are ordered exactly like the scalar Gauss-Seidel sequence.
// `WideBatch` is ~1.5 KB of SoA state while `Scalar` is a usize — the lint
// fires, but the batch is built once per solve and moved, so boxing it
// would only add indirection to the hot iteration loop.
#[allow(clippy::large_enum_variant)]
pub(crate) enum SolverStep {
    Wide(WideBatch),
    Scalar(usize),
}

/// Pack the manifold states into a solver-step sequence (G7):
/// consecutive single-point manifolds whose body sets are disjoint are
/// grouped into 4-lane wide batches; everything else stays scalar. The
/// global manifold order is preserved — a contact is never moved past
/// another, so the Gauss-Seidel semantics are unchanged.
pub(crate) fn build_solver_steps(
    bodies: &[RigidBody],
    manifolds: &[Manifold],
    states: &[ManifoldState],
) -> Vec<SolverStep> {
    let mut steps: Vec<SolverStep> = Vec::with_capacity(states.len());
    let mut cur: Vec<(usize, &Manifold, &ManifoldState)> = Vec::with_capacity(4);
    // Body set of the current batch (up to 8 distinct island-local indices).
    let mut cur_bodies: [usize; 8] = [usize::MAX; 8];
    let mut cur_n = 0usize;

    let flush = |steps: &mut Vec<SolverStep>,
                 cur: &mut Vec<(usize, &Manifold, &ManifoldState)>,
                 cur_bodies: &mut [usize; 8],
                 cur_n: &mut usize| {
        if cur.is_empty() {
            return;
        }
        let batch = WideBatch::build(cur, bodies);
        steps.push(SolverStep::Wide(batch));
        cur.clear();
        *cur_n = 0;
        *cur_bodies = [usize::MAX; 8];
    };

    for (si, st) in states.iter().enumerate() {
        if st.count != 1 {
            flush(&mut steps, &mut cur, &mut cur_bodies, &mut cur_n);
            steps.push(SolverStep::Scalar(si));
            continue;
        }
        // Check disjointness against the current batch.
        let (i, j) = (st.i, st.j);
        let conflicts = cur_bodies[..cur_n].contains(&i) || cur_bodies[..cur_n].contains(&j);
        if cur.len() >= 4 || conflicts {
            flush(&mut steps, &mut cur, &mut cur_bodies, &mut cur_n);
        }
        cur.push((si, &manifolds[st.mi], st));
        cur_bodies[cur_n] = i;
        cur_bodies[cur_n + 1] = j;
        cur_n += 2;
    }
    flush(&mut steps, &mut cur, &mut cur_bodies, &mut cur_n);
    steps
}

// Helper for the tests below: attach a manifold index to a state.
impl ManifoldState {
    #[cfg(test)]
    fn with_mi(mut self, mi: usize) -> Self {
        self.mi = mi;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Manifold, ManifoldPoint};

    fn body_at(pos: Vec3) -> RigidBody {
        RigidBody::new_sphere(pos, 0.5, 1.0)
    }

    fn single_state(i: usize, j: usize, _normal: Vec3, _point: Vec3, target: f32) -> ManifoldState {
        ManifoldState {
            mi: 0,
            i,
            j,
            count: 1,
            acc: [0.0; 4],
            acc_friction: [0.0; 4],
            acc_friction2: [0.0; 4],
            bias: [0.0; 4],
            target: [target; 4],
            mu: 0.3,
            t1: Vec3::X,
            t2: Vec3::Z,
            la: [Vec3::ZERO; 4],
            lb: [Vec3::ZERO; 4],
            pen0: [0.0; 4],
        }
    }

    fn manifold(i: usize, j: usize, normal: Vec3, point: Vec3) -> Manifold {
        let mut m = Manifold {
            body_a: i,
            body_b: j,
            normal,
            point_count: 1,
            points: [ManifoldPoint {
                world_point: Vec3::ZERO,
                penetration: 0.0,
            }; 4],
        };
        m.points[0] = ManifoldPoint {
            world_point: point,
            penetration: 0.01,
        };
        m
    }

    /// The wide lane solver must produce the same velocities and accumulated
    /// impulses as the scalar single-point code path for the same inputs.
    #[test]
    fn wide_batch_matches_scalar_single_point() {
        // Two independent sphere pairs with an approach velocity — the
        // simplest wide scene (2 lanes).
        let mut bodies = vec![
            body_at(Vec3::new(-1.0, 0.0, 0.0)),
            body_at(Vec3::new(1.0, 0.0, 0.0)),
            body_at(Vec3::new(2.0, 0.0, 0.0)),
            body_at(Vec3::new(4.0, 0.0, 0.0)),
        ];
        bodies[0].velocity = Vec3::new(1.0, 0.0, 0.0);
        bodies[2].velocity = Vec3::new(0.5, 0.0, 0.0);

        let n = Vec3::X;
        let manifolds = vec![
            manifold(0, 1, n, Vec3::new(0.0, 0.0, 0.0)),
            manifold(2, 3, n, Vec3::new(3.0, 0.0, 0.0)),
        ];
        let states = vec![
            single_state(0, 1, n, Vec3::ZERO, 0.0),
            single_state(2, 3, n, Vec3::new(3.0, 0.0, 0.0), 0.0),
        ];

        // Reference: run the scalar single-point solve by hand on clones.
        let mut scalar_bodies = bodies.clone();
        let mut scalar_states = states.clone();
        run_scalar(&mut scalar_bodies, &manifolds, &mut scalar_states, 8, false);

        // Wide path.
        let items: Vec<(usize, &Manifold, &ManifoldState)> = states
            .iter()
            .enumerate()
            .map(|(si, s)| (si, &manifolds[s.mi], s))
            .collect();
        let mut batch = WideBatch::build(&items, &bodies);
        let mut bodies2 = bodies.clone();
        for _ in 0..8 {
            batch.gather(&bodies2);
            batch.solve_iteration();
            batch.scatter(&mut bodies2);
        }
        let mut states2 = states.clone();
        batch.write_back_acc(&mut states2);

        for (h, (wide, scalar)) in bodies2.iter().zip(scalar_bodies.iter()).enumerate() {
            let dv = (wide.velocity - scalar.velocity).length();
            let dw = (wide.angular_velocity - scalar.angular_velocity).length();
            // The wide path uses matrix inertia vs quaternion rotations —
            // allow a small epsilon, well below any physical threshold.
            assert!(dv < 1e-4, "body {h} velocity diverged: {dv}");
            assert!(dw < 1e-4, "body {h} angular velocity diverged: {dw}");
        }
        assert!(
            bodies2[0].velocity.x < 1.0 && bodies2[3].velocity.x > 0.0,
            "impulses must slow the approach: v0.x = {}, v3.x = {}",
            bodies2[0].velocity.x,
            bodies2[3].velocity.x
        );
    }

    /// Runs the exact scalar single-point code used in solve_island_velocity.
    #[allow(clippy::needless_range_loop)]
    fn run_scalar(
        bodies: &mut [RigidBody],
        manifolds: &[Manifold],
        states: &mut [ManifoldState],
        iterations: u32,
        _allow_restitution: bool,
    ) {
        // Delegate to the PRODUCTION scalar single-point step so the parity
        // test compares the wide path against the exact code it must match
        // (single-point branch of the island velocity solver).
        for _ in 0..iterations {
            for st in states.iter_mut() {
                if st.count == 1 {
                    crate::engine::BuiltinPhysicsEngine::solve_scalar_velocity_step(
                        bodies, manifolds, st,
                    );
                }
            }
        }
    }

    #[test]
    fn batch_groups_disjoint_consecutive_contacts() {
        // (0,1), (0,2), (3,4), (5,6) — the second shares body 0 with the
        // first, so the batch must split: [0,1] + [3,4,5,6] → batches of
        // 1 and 3 lanes (the 1-lane batch preserves GS order).
        let bodies = vec![
            body_at(Vec3::ZERO),
            body_at(Vec3::new(1.0, 0.0, 0.0)),
            body_at(Vec3::new(0.0, 1.0, 0.0)),
            body_at(Vec3::new(2.0, 0.0, 0.0)),
            body_at(Vec3::new(3.0, 0.0, 0.0)),
            body_at(Vec3::new(4.0, 0.0, 0.0)),
            body_at(Vec3::new(5.0, 0.0, 0.0)),
        ];
        let n = Vec3::X;
        let manifolds = vec![
            manifold(0, 1, n, Vec3::new(0.5, 0.0, 0.0)),
            manifold(0, 2, n, Vec3::new(0.0, 0.5, 0.0)),
            manifold(3, 4, n, Vec3::new(2.5, 0.0, 0.0)),
            manifold(5, 6, n, Vec3::new(4.5, 0.0, 0.0)),
        ];
        let states: Vec<ManifoldState> = manifolds
            .iter()
            .enumerate()
            .map(|(mi, m)| {
                single_state(m.body_a, m.body_b, m.normal, m.points[0].world_point, 0.0).with_mi(mi)
            })
            .collect();

        let steps = build_solver_steps(&bodies, &manifolds, &states);
        assert_eq!(steps.len(), 2, "expected [Wide(1)] then [Wide(3)]");
        match (&steps[0], &steps[1]) {
            (SolverStep::Wide(b0), SolverStep::Wide(b1)) => {
                assert_eq!(b0.count, 1);
                assert_eq!(b1.count, 3);
            }
            _ => panic!("both steps must be wide"),
        }
    }
}
