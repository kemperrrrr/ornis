//! Per-body 6x6 assembly and dense LDL solve for the AVBD engine.
//!
//! Owns [`solve_6x6`] and the primal [`AvbdEngine::solve_body`] sweep (inertial
//! + contact/joint-row Hessian).
//!
//! Row math and discovery live in [`rows`](super::rows), joint duals in
//! [`joints_ext`](super::joints_ext). Moved verbatim from `avbd.rs` (phase 3).

use super::rows::{
    diagonalize, eff_inv_mass, friction_frame, geometric_stiffness_ball_socket, mat3_vec, outer,
    quat_diff_vec, quat_integrate, regularized_limit, row_live, warm_limit_state, world_inertia,
    wrap_pi,
};
use super::*;
use crate::constants::{DEGENERATE_LEN2, NEAR_ZERO};
use crate::engine::joints::{hinge_twist, quat_twist};
use std::f32::consts::TAU;

/// Dense LDL (no pivoting) for a 6x6 SPD system. Returns `None` on breakdown.
///
/// This is the exact per-body solve `AvbdEngine::solve_body` runs after
/// assembling the inertial + contact-row Hessian; the GPU rung-1 diagonal
/// solve (behind the `gpu` feature) cross-checks against it on diagonal
/// systems (tolerance, never bit-identical by promise).
pub fn solve_6x6(
    lhs: [[f32; SPATIAL_DOF]; SPATIAL_DOF],
    rhs: [f32; SPATIAL_DOF],
) -> Option<[f32; SPATIAL_DOF]> {
    let mut l = [[0.0f32; SPATIAL_DOF]; SPATIAL_DOF];
    let mut d = [0.0f32; SPATIAL_DOF];
    for i in 0..SPATIAL_DOF {
        for j in 0..=i {
            let mut s = lhs[i][j];
            for k in 0..j {
                s -= l[i][k] * d[k] * l[j][k];
            }
            if i == j {
                if s <= DEGENERATE_LEN2 {
                    return None;
                }
                d[i] = s;
                l[i][i] = 1.0;
            } else {
                l[i][j] = s / d[j];
            }
        }
    }
    let mut y = [0.0f32; SPATIAL_DOF];
    for i in 0..SPATIAL_DOF {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s;
    }
    let mut z = [0.0f32; SPATIAL_DOF];
    for i in 0..SPATIAL_DOF {
        z[i] = y[i] / d[i];
    }
    let mut x = [0.0f32; SPATIAL_DOF];
    for i in (0..SPATIAL_DOF).rev() {
        let mut s = z[i];
        for k in (i + 1)..SPATIAL_DOF {
            s -= l[k][i] * x[k];
        }
        x[i] = s;
    }
    Some(x)
}

impl AvbdEngine {
    /// Solve one body against all its pairs and joints (official primal).
    pub(super) fn solve_body(&mut self, h: usize) {
        let m_dt2 = 1.0 / eff_inv_mass(&self.bodies[h]) / (DT_STEP * DT_STEP);
        let iw = world_inertia(self.bodies[h].inertia, self.bodies[h].orientation);
        let mut lhs = [[0.0f32; SPATIAL_DOF]; SPATIAL_DOF];
        for (i, row) in lhs.iter_mut().enumerate().take(ANGULAR_OFFSET) {
            row[i] = m_dt2;
        }
        for a in 0..ANGULAR_OFFSET {
            for b in 0..ANGULAR_OFFSET {
                lhs[ANGULAR_OFFSET + a][ANGULAR_OFFSET + b] = iw[a][b] / (DT_STEP * DT_STEP);
            }
        }
        let mut rhs = [0.0f32; SPATIAL_DOF];
        let rl = m_dt2 * (self.bodies[h].position - self.inertial[h]);
        rhs[0] = rl.x;
        rhs[1] = rl.y;
        rhs[2] = rl.z;
        let ra = mat3_vec(
            iw,
            quat_diff_vec(self.bodies[h].orientation, self.inertial_rot[h]),
        ) / (DT_STEP * DT_STEP);
        rhs[ANGULAR_OFFSET] = ra.x;
        rhs[ANGULAR_OFFSET + 1] = ra.y;
        rhs[ANGULAR_OFFSET + 2] = ra.z;

        // Contact rows.
        for pi in 0..self.pairs.len() {
            let (is_a, is_b) = {
                let p = &self.pairs[pi];
                (p.a == h, p.b == h)
            };
            if !is_a && !is_b {
                continue;
            }
            let sign = if is_a { 1.0 } else { -1.0 };
            for qi in 0..self.pairs[pi].points.len() {
                let (n, mu, pt) = {
                    let p = &self.pairs[pi];
                    (p.n, p.mu, p.points[qi].clone())
                };
                // Separated-damper rule: an open pair (`gap > 0`) reacts
                // with a velocity-only normal row (no C0 spring term, no
                // friction, no dual memory — committed in `dual_update`
                // by the same gate). A spring strong enough to catch an
                // impact can hold weight forever, so any C0-backed support
                // while separated levitates arrivals (measured: a 5 m/s
                // drop hovered +7mm and rose) and dilutes burrowing
                // drivers 100x via (1-alpha) (measured: a 2 m/s pusher
                // ghosted). The damper kills approach velocity within the
                // step at full J strength and cannot hold a static load,
                // so arrivals touch down and drivers are tracked. Touching
                // pairs keep the full Taylor spring bit-identically.
                let touching = self.pairs[pi].gap <= 0.0;
                let c0 = if touching { self.gap_c0(pi, qi) } else { 0.0 };
                let (cn, r_up, r_lo) = self.row_c(&self.pairs[pi], &pt, n, c0);
                let r_side = if is_a { r_up } else { r_lo };
                if !touching {
                    // Imminence gate: damper rows fire only when the pair
                    // will touch down within this step (predicted gap <= 0
                    // from the center approach rate). A grazing pass has a
                    // large normal approach yet never lands — firing on it
                    // turns a 4cm clean miss into a meter-scale launch
                    // (measured: +4m lift, 8 -> 5.9 m/s). The shell stays
                    // (warm state, wake), only the rows stay quiet; slow
                    // burrows still land via the spring below once the
                    // signed gap closes.
                    let a_vel = self.bodies[self.pairs[pi].a].velocity;
                    let b_vel = self.bodies[self.pairs[pi].b].velocity;
                    let approach = (b_vel - a_vel).dot(n).max(0.0);
                    if self.pairs[pi].gap - approach * DT_STEP > 0.0 {
                        continue;
                    }
                    // One-step velocity kill, normalized per PAIR (not per
                    // point): five corner rows must not drag 5x harder than
                    // one witness row, or the catch strength depends on
                    // manifold size. pen_d totals 2*m_red/dt^2 across the
                    // pair: twice the exact one-step stopping force (margin
                    // for heavy/fast arrivals) from the REDUCED mass,
                    // identical on both sides — a per-body stiffness stamps
                    // different K into A's and B's 6x6 and pumps energy
                    // across mass ratios (measured in-engine: 1 vs 100 kg,
                    // momentum -1 -> -40.6, 0.5 -> 8.18 J with no external
                    // work). Implicit hence stable, capped for fp safety.
                    // Push-only (min): separating pairs exert nothing. No
                    // dual memory here (gated in `dual_update`): a damper
                    // that remembers holds weight and levitates arrivals.
                    if cn.abs() >= C_EPS {
                        let npts = self.pairs[pi].points.len().max(1) as f32;
                        let ka = eff_inv_mass(&self.bodies[self.pairs[pi].a]);
                        let kb = eff_inv_mass(&self.bodies[self.pairs[pi].b]);
                        let m_red = if ka + kb > NEAR_ZERO {
                            1.0 / (ka + kb)
                        } else {
                            0.0
                        };
                        if m_red > 0.0 {
                            let pen_d = (2.0 * m_red / (DT_STEP * DT_STEP) / npts).min(PENALTY_MAX);
                            let f_d = (pen_d * cn).min(0.0);
                            Self::stamp_row(&mut lhs, &mut rhs, n, pen_d, f_d, r_side, sign);
                        }
                    }
                    continue;
                }
                // Anisotropic frame (live orientations); isotropic pairs get
                // exactly `tangent_basis(n)` back.
                let t1 = {
                    let p = &self.pairs[pi];
                    friction_frame(&self.bodies[p.a], &self.bodies[p.b], p.n).0
                };
                let t2 = t1.cross(n);
                // Friction rows share coincident levers: the depth offset
                // stays in the normal row only (see `friction_levers`).
                let (ra_f, rb_f) = {
                    let p = &self.pairs[pi];
                    Self::friction_levers(&self.bodies[p.a], &self.bodies[p.b], p.n, &pt)
                };
                let r_side_f = if is_a { ra_f } else { rb_f };
                let (ct1, _, _) = self.row_c_levers(&self.pairs[pi], t1, 0.0, ra_f, rb_f);
                let (ct2, _, _) = self.row_c_levers(&self.pairs[pi], t2, 0.0, ra_f, rb_f);
                let f = Self::contact_force(cn, pt.pen, pt.lam, [ct1, ct2], mu);
                for (row, (axis, cv, fv, r)) in [
                    (n, cn, f[0], r_side),
                    (t1, ct1, f[1], r_side_f),
                    (t2, ct2, f[2], r_side_f),
                ]
                .into_iter()
                .enumerate()
                {
                    let pen = pt.pen[row];
                    // Dust guard keeps the warm holding force: only dust
                    // an exactly zero force can skip a dust violation.
                    if !row_live(cv, fv) {
                        continue;
                    }
                    Self::stamp_row(&mut lhs, &mut rhs, axis, pen, fv, r, sign);
                }
                // Rolling + torsional resistance (MuJoCo triple): pure
                // couples opposing relative spin, capped by mu x normal
                // force. Zero coefficients skip everything.
                let (mu_roll, mu_spin) = {
                    let p = &self.pairs[pi];
                    let ba = &self.bodies[p.a];
                    let bb = &self.bodies[p.b];
                    (
                        ba.rolling_friction.max(bb.rolling_friction),
                        ba.torsion_friction.max(bb.torsion_friction),
                    )
                };
                if mu_roll > 0.0 || mu_spin > 0.0 {
                    let drot_a = quat_diff_vec(
                        self.bodies[self.pairs[pi].a].orientation,
                        self.rot0[self.pairs[pi].a],
                    );
                    let drot_b = quat_diff_vec(
                        self.bodies[self.pairs[pi].b].orientation,
                        self.rot0[self.pairs[pi].b],
                    );
                    let wrel = drot_a - drot_b;
                    for (ax, li, mur) in [(t1, 0, mu_roll), (t2, 1, mu_roll), (n, 2, mu_spin)] {
                        if mur <= 0.0 {
                            continue;
                        }
                        let c = wrel.dot(ax);
                        let cap = mur * f[0].abs();
                        let fr = (ROLL_PEN * c + pt.roll_lam[li]).clamp(-cap, cap);
                        // Pure couple: angular-only gradient, opposite senses.
                        let g = if is_a { ax } else { -ax };
                        let o = outer(g, g);
                        for x in 0..ANGULAR_OFFSET {
                            for y in 0..ANGULAR_OFFSET {
                                lhs[ANGULAR_OFFSET + x][ANGULAR_OFFSET + y] += ROLL_PEN * o[x][y];
                            }
                        }
                        rhs[ANGULAR_OFFSET] += fr * g.x;
                        rhs[ANGULAR_OFFSET + 1] += fr * g.y;
                        rhs[ANGULAR_OFFSET + 2] += fr * g.z;
                    }
                }
            }
        }

        // Joint rows per kind (equalities; limits one-sided, motors servo).
        for ji in 0..self.joints.len() {
            let (is_a, is_b) = {
                let j = &self.joints[ji];
                (j.a == h, j.b == h)
            };
            if !is_a && !is_b && self.joints[ji].kind != AvbdJointKind::Gear {
                continue;
            }
            let sign = if is_a { 1.0 } else { -1.0 };
            let j = self.joints[ji].clone();
            let a = &self.bodies[j.a];
            let b = &self.bodies[j.b];
            let pa = a.position + a.orientation * j.la;
            let pb = b.position + b.orientation * j.lb;
            let pa0 = self.pos0[j.a] + self.rot0[j.a] * j.la;
            let pb0 = self.pos0[j.b] + self.rot0[j.b] * j.lb;
            let live: Vec3 = pa - pb;
            let c0v: Vec3 = pa0 - pb0;
            // Live slide/hinge axis in world space (from A's side, like the
            // official drive code).
            let wa = (a.orientation * j.ax_a).normalize_or(Vec3::Z);
            // --- linear equality rows; fl accumulates the force vector ---
            let mut fl = Vec3::ZERO;
            match j.kind {
                AvbdJointKind::Ball | AvbdJointKind::Revolute | AvbdJointKind::Fixed => {
                    for k in 0..ANGULAR_OFFSET {
                        let axis = Vec3::from_array({
                            let mut arr = [0.0f32; ANGULAR_OFFSET];
                            arr[k] = 1.0;
                            arr
                        });
                        let (live, initial) = if j.kind == AvbdJointKind::Fixed {
                            (live + a.orientation * j.dref, c0v + self.rot0[j.a] * j.dref)
                        } else {
                            (live, c0v)
                        };
                        let c = live[k] - ALPHA * initial[k];
                        let f = j.pen_l[k] * c + j.lam_l[k];
                        if !row_live(c, f) {
                            continue;
                        }
                        let r_side = if is_a {
                            a.orientation
                                * (j.la
                                    + if j.kind == AvbdJointKind::Fixed {
                                        j.dref
                                    } else {
                                        Vec3::ZERO
                                    })
                        } else {
                            b.orientation * j.lb
                        };
                        Self::stamp_row(&mut lhs, &mut rhs, axis, j.pen_l[k], f, r_side, sign);
                        fl[k] = f;
                    }
                }
                AvbdJointKind::Prismatic => {
                    let (u, v) = tangent_basis(wa);
                    for (ax, li) in [(u, 0), (v, 1)] {
                        let c = (live - ALPHA * c0v).dot(ax);
                        let f = j.pen_l[li] * c + j.lam_l[li];
                        if !row_live(c, f) {
                            continue;
                        }
                        let r_side = if is_a {
                            a.orientation * j.la
                        } else {
                            b.orientation * j.lb
                        };
                        Self::stamp_row(&mut lhs, &mut rhs, ax, j.pen_l[li], f, r_side, sign);
                        fl += ax * f;
                    }
                }
                AvbdJointKind::Distance => {
                    // Single rod row along the live anchor delta.
                    let len = live.length();
                    let dir = if len > NEAR_ZERO { live / len } else { Vec3::Y };
                    let c = (len - j.ref_val) - ALPHA * (c0v.length() - j.ref_val);
                    let f = j.pen_l[0] * c + j.lam_l[0];
                    if row_live(c, f) {
                        let r_side = if is_a {
                            a.orientation * j.la
                        } else {
                            b.orientation * j.lb
                        };
                        Self::stamp_row(&mut lhs, &mut rhs, dir, j.pen_l[0], f, r_side, sign);
                        fl = dir * f;
                    }
                }
                AvbdJointKind::Rope => {
                    // One-sided rod row: only a stretched rope stamps —
                    // slack never pushes (the dual gates the same way).
                    let len = live.length();
                    if len > j.ref_val {
                        let dir = if len > NEAR_ZERO { live / len } else { Vec3::Y };
                        let c = (len - j.ref_val) - ALPHA * (c0v.length() - j.ref_val);
                        let f = j.pen_l[0] * c + j.lam_l[0];
                        if row_live(c, f) {
                            let r_side = if is_a {
                                a.orientation * j.la
                            } else {
                                b.orientation * j.lb
                            };
                            Self::stamp_row(&mut lhs, &mut rhs, dir, j.pen_l[0], f, r_side, sign);
                            fl = dir * f;
                        }
                    }
                }
                AvbdJointKind::Spring => {
                    // Position-level spring about the rest length (the
                    // wheel-suspension discipline along the anchor delta):
                    // fixed penalty from the stiffness, velocity damping
                    // from the step delta, no dual state. Explicit/implicit
                    // is an SI-only distinction (see `JointKind::Spring`) —
                    // AVBD always solves at position level.
                    let Some(motor) = j.spec.spring_motor() else {
                        continue;
                    };
                    let len = live.length();
                    let dir = if len > NEAR_ZERO { live / len } else { Vec3::Y };
                    let rest = motor.target_position;
                    let s = len - rest;
                    let s0 = c0v.length() - rest;
                    let ka = eff_inv_mass(a);
                    let kb = eff_inv_mass(b);
                    // Reduced linear mass (wheel parity — lever terms ride
                    // the stamped gradient, not the penalty).
                    let m = if ka + kb > NEAR_ZERO {
                        1.0 / (ka + kb)
                    } else {
                        0.0
                    };
                    if m <= 0.0 {
                        continue;
                    }
                    // Absolute spring coefficients (force-based pass
                    // through, acceleration-based scale by the driven
                    // mass — the `pd_coefficients` equation, shared with
                    // the SI spring row).
                    let (stiff, dampc) = match motor.model {
                        crate::joint::MotorModel::ForceBased => (motor.stiffness, motor.damping),
                        crate::joint::MotorModel::AccelerationBased => {
                            (motor.stiffness * m, motor.damping * m)
                        }
                    };
                    if stiff <= 0.0 && dampc <= 0.0 {
                        // Neither centering nor damping (a zeroed motor):
                        // inert in every solver (SI applies zero force too).
                        continue;
                    }
                    let r_side = if is_a {
                        a.orientation * j.la
                    } else {
                        b.orientation * j.lb
                    };
                    // Violation form (wheel parity: positive `f` pushes
                    // along +dir, i.e. against +C — the physical spring
                    // force has the opposite sign).
                    if stiff > 0.0 {
                        let f = stiff * s + dampc * (s - s0) / DT_STEP;
                        Self::stamp_row(&mut lhs, &mut rhs, dir, stiff, f, r_side, sign);
                    } else {
                        // Pure damper (velocity-kind motor): viscous row
                        // only — explicit-Euler stability (`dampc * DT_STEP
                        // * inv_mass < 2`), same bound as the SI explicit
                        // spring.
                        let pen_v = dampc / DT_STEP;
                        let f = dampc * (s - s0) / DT_STEP;
                        Self::stamp_row(&mut lhs, &mut rhs, dir, pen_v, f, r_side, sign);
                    }
                }
                AvbdJointKind::Gear => {
                    // Position-level gear row (closer to Box2D than the
                    // builtin velocity-only pass): C = ca + ratio*cb - const
                    // with per-body side gradients. Prismatic-side (linear)
                    // contributions stamp here; revolute-side (angular) ones
                    // in the angular match. Shared force/penalty on
                    // lam_l[0]/pen_l[0] (distance-row discipline).
                    if let Some((sa, sb)) = self.gear_sides(&j) {
                        let c = sa.coord + j.gratio * sb.coord - j.ref_val;
                        let f = j.pen_l[0] * c + j.lam_l[0];
                        if row_live(c, f) {
                            for (side, coef) in [(&sa, 1.0), (&sb, j.gratio)] {
                                if side.kind.is_angular() {
                                    continue;
                                }
                                if h != side.a && h != side.b {
                                    continue;
                                }
                                let lsign = if h == side.a { -1.0 } else { 1.0 };
                                let r_side = if h == side.a { side.ra } else { side.rb };
                                let pen_c = j.pen_l[0] * coef * coef;
                                let f_c = f * coef;
                                Self::stamp_row(
                                    &mut lhs, &mut rhs, side.axis, pen_c, f_c, r_side, lsign,
                                );
                            }
                        }
                    }
                }
                AvbdJointKind::SixDof => {
                    // Per-axis rows in body A's live assembly frame
                    // (mirrors the builtin position pass): locked linear =
                    // drift from the assembly delta, limited linear =
                    // one-sided raw separation (no ref subtraction, like the
                    // builtin), free = nothing. Limited forces live in
                    // `sacc` (dual-owned, `lim_dual` discipline).
                    for (i, e) in SIXDOF_FRAME.iter().enumerate() {
                        let dir = (a.orientation * *e).normalize_or(*e);
                        match j.six_lin[i] {
                            AxisConfig::Free => {}
                            AxisConfig::Locked => {
                                let evec = (pb - pa) - a.orientation * j.dref;
                                let c = evec.dot(dir);
                                let f = j.pen_l[i] * c + j.lam_l[i];
                                if !row_live(c, f) {
                                    continue;
                                }
                                let r_side = if is_a {
                                    a.orientation * j.la
                                } else {
                                    b.orientation * j.lb
                                };
                                // B-minus-A measure: gradients flip vs ball rows.
                                let lsign = if is_a { -1.0 } else { 1.0 };
                                Self::stamp_row(
                                    &mut lhs, &mut rhs, dir, j.pen_l[i], f, r_side, lsign,
                                );
                            }
                            AxisConfig::Limited { min, max } => {
                                let sep = ((pb - pa) - a.orientation * j.dref).dot(dir);
                                if let Some(lower) =
                                    warm_limit_state(sep, min, max, LIMIT_SLOP_LIN, j.sacc[i])
                                {
                                    let axis0 = (self.rot0[j.a] * *e).normalize_or(*e);
                                    let initial =
                                        ((pb0 - pa0) - self.rot0[j.a] * j.dref).dot(axis0);
                                    let c = regularized_limit(
                                        sep,
                                        initial,
                                        if lower { min } else { max },
                                        lower,
                                    );
                                    let f_raw = j.pen_l[i] * c + j.sacc[i];
                                    let f = if lower {
                                        f_raw.min(0.0)
                                    } else {
                                        f_raw.max(0.0)
                                    };
                                    if !row_live(c, f) {
                                        continue;
                                    }
                                    let r_side = if is_a {
                                        a.orientation * j.la
                                    } else {
                                        b.orientation * j.lb
                                    };
                                    let lsign = if is_a { -1.0 } else { 1.0 };
                                    Self::stamp_row(
                                        &mut lhs, &mut rhs, dir, j.pen_l[i], f, r_side, lsign,
                                    );
                                }
                            }
                        }
                    }
                }
                AvbdJointKind::Wheel => {
                    // 2 perp equality rows (suspension axis `wa` is the free
                    // slide direction) + the spring row along `wa`.
                    let (u, v) = tangent_basis(wa);
                    for (ax, li) in [(u, 0), (v, 1)] {
                        let c = (live - ALPHA * c0v).dot(ax);
                        let f = j.pen_l[li] * c + j.lam_l[li];
                        if !row_live(c, f) {
                            continue;
                        }
                        let r_side = if is_a {
                            a.orientation * j.la
                        } else {
                            b.orientation * j.lb
                        };
                        Self::stamp_row(&mut lhs, &mut rhs, ax, j.pen_l[li], f, r_side, sign);
                        fl += ax * f;
                    }
                    // Suspension spring about the rest length (position
                    // level, Box2D frequency/damping semantics): fixed
                    // penalty, no dual state — static sag holds the load
                    // (F = k*s), the velocity term supplies damping.
                    // A zero/negative frequency degrades to a rigid
                    // equality row on pen_l[2]/lam_l[2] (dual below).
                    let s = (pb - pa).dot(wa) - j.ref_val;
                    let r_side = if is_a {
                        a.orientation * j.la
                    } else {
                        b.orientation * j.lb
                    };
                    // C = (B-A).wa: gradients flip vs ball rows.
                    let lsign = if is_a { -1.0 } else { 1.0 };
                    if j.susp[0] > 0.0 {
                        let ka = eff_inv_mass(a);
                        let kb = eff_inv_mass(b);
                        let m = if ka + kb > NEAR_ZERO {
                            1.0 / (ka + kb)
                        } else {
                            0.0
                        };
                        if m > 0.0 {
                            let omega = TAU * j.susp[0];
                            let stiff = m * omega * omega;
                            let dampc = 2.0 * m * j.susp[1] * omega;
                            let wa0 = (self.rot0[j.a] * j.ax_a).normalize_or(Vec3::Z);
                            let s0 = (pb0 - pa0).dot(wa0) - j.ref_val;
                            // Violation form (matches `stamp_row`: positive
                            // `f` pushes along +nn, i.e. against +C — the
                            // physical spring force has the opposite sign).
                            let f = stiff * s + dampc * (s - s0) / DT_STEP;
                            Self::stamp_row(&mut lhs, &mut rhs, wa, stiff, f, r_side, lsign);
                        }
                    } else {
                        let wa0 = (self.rot0[j.a] * j.ax_a).normalize_or(Vec3::Z);
                        let c = s - ALPHA * ((pb0 - pa0).dot(wa0) - j.ref_val);
                        let f = j.pen_l[2] * c + j.lam_l[2];
                        if row_live(c, f) {
                            Self::stamp_row(&mut lhs, &mut rhs, wa, j.pen_l[2], f, r_side, lsign);
                        }
                    }
                }
                AvbdJointKind::Motor => {
                    // No primal position rows: the free drive acts as a
                    // deadbeat velocity impulse (see `free_motor_impulse`),
                    // not as a penalty servo. A fixed-gain position servo
                    // is bang-bang at single-step dt without substeps.
                }
            }
            // Geometric stiffness (official): the lever rotates with the
            // body, and the truncated Hessian must know. Without this the
            // joint spins up through long levers (see module docs).
            {
                let r = if is_a {
                    a.orientation
                        * (j.la
                            + if j.kind == AvbdJointKind::Fixed {
                                j.dref
                            } else {
                                Vec3::ZERO
                            })
                } else {
                    -(b.orientation * j.lb)
                };
                let fa = fl.to_array();
                let mut h_mat = [[0.0f32; ANGULAR_OFFSET]; ANGULAR_OFFSET];
                for (k, fk) in fa.iter().enumerate() {
                    let g = geometric_stiffness_ball_socket(k, r);
                    for x in 0..ANGULAR_OFFSET {
                        for y in 0..ANGULAR_OFFSET {
                            h_mat[x][y] += g[x][y] * fk;
                        }
                    }
                }
                let hd = diagonalize(h_mat);
                lhs[ANGULAR_OFFSET][ANGULAR_OFFSET] += hd.x;
                lhs[4][4] += hd.y;
                lhs[5][5] += hd.z;
            }
            // --- angular equality rows ---
            match j.kind {
                AvbdJointKind::Revolute | AvbdJointKind::Prismatic => {
                    let axa = a.orientation * j.ax_a;
                    let axb = b.orientation * j.ax_b;
                    let axa0 = self.rot0[j.a] * j.ax_a;
                    let axb0 = self.rot0[j.b] * j.ax_b;
                    let (t1, t2) = tangent_basis(axa0);
                    for (t, lam, pen) in
                        [(t1, j.lam_a[0], j.pen_a[0]), (t2, j.lam_a[1], j.pen_a[1])]
                    {
                        let live_c = t.dot(axa - axb);
                        let c0_c = t.dot(axa0 - axb0);
                        let c = live_c - ALPHA * c0_c;
                        let f = pen * c + lam;
                        if !row_live(c, f) {
                            continue;
                        }
                        // Rotation-only rows: torque arms about the hinge axes.
                        let g_ang = if is_a { axa.cross(t) } else { -(axb.cross(t)) };
                        let o = outer(g_ang, g_ang);
                        for x in 0..ANGULAR_OFFSET {
                            for y in 0..ANGULAR_OFFSET {
                                lhs[ANGULAR_OFFSET + x][ANGULAR_OFFSET + y] += pen * o[x][y];
                            }
                        }
                        rhs[ANGULAR_OFFSET] += f * g_ang.x;
                        rhs[ANGULAR_OFFSET + 1] += f * g_ang.y;
                        rhs[ANGULAR_OFFSET + 2] += f * g_ang.z;
                    }
                }
                AvbdJointKind::Gear => {
                    // Revolute-side (angular) contributions of the shared
                    // gear force (prismatic sides stamp in the linear
                    // match). Gradient of a hinge twist: -axis on side A,
                    // +axis on side B.
                    if let Some((sa, sb)) = self.gear_sides(&j) {
                        let c = sa.coord + j.gratio * sb.coord - j.ref_val;
                        let f = j.pen_l[0] * c + j.lam_l[0];
                        if row_live(c, f) {
                            for (side, coef) in [(&sa, 1.0), (&sb, j.gratio)] {
                                if !side.kind.is_angular() {
                                    continue;
                                }
                                if h != side.a && h != side.b {
                                    continue;
                                }
                                let g = if h == side.a { -side.axis } else { side.axis };
                                let pen_c = j.pen_l[0] * coef * coef;
                                let f_c = f * coef;
                                let o = outer(g, g);
                                for x in 0..ANGULAR_OFFSET {
                                    for y in 0..ANGULAR_OFFSET {
                                        lhs[ANGULAR_OFFSET + x][ANGULAR_OFFSET + y] +=
                                            pen_c * o[x][y];
                                    }
                                }
                                rhs[ANGULAR_OFFSET] += f_c * g.x;
                                rhs[ANGULAR_OFFSET + 1] += f_c * g.y;
                                rhs[ANGULAR_OFFSET + 2] += f_c * g.z;
                            }
                        }
                    }
                }
                AvbdJointKind::SixDof => {
                    // Locked angular = per-axis orientation lock about the
                    // assembly-frame axis; limited angular = one-sided
                    // twist window (travel about the LOCAL frame axis minus
                    // the reference twist, like the builtin). Forces for
                    // limited axes live in `sacc[ANGULAR_OFFSET..SPATIAL_DOF]`.
                    let qrel = a.orientation.conjugate() * b.orientation;
                    let diff = quat_diff_vec(qrel, j.q_ref);
                    for (i, e) in SIXDOF_FRAME.iter().enumerate() {
                        let dir = (a.orientation * *e).normalize_or(*e);
                        match j.six_ang[i] {
                            AxisConfig::Free => {}
                            AxisConfig::Locked => {
                                let c = diff.dot(*e);
                                let f = j.pen_a[i] * c + j.lam_a[i];
                                if !row_live(c, f) {
                                    continue;
                                }
                                let g = if is_a { -dir } else { dir };
                                let o = outer(g, g);
                                for x in 0..ANGULAR_OFFSET {
                                    for y in 0..ANGULAR_OFFSET {
                                        lhs[ANGULAR_OFFSET + x][ANGULAR_OFFSET + y] +=
                                            j.pen_a[i] * o[x][y];
                                    }
                                }
                                rhs[ANGULAR_OFFSET] += f * g.x;
                                rhs[ANGULAR_OFFSET + 1] += f * g.y;
                                rhs[ANGULAR_OFFSET + 2] += f * g.z;
                            }
                            AxisConfig::Limited { min, max } => {
                                let travel = hinge_twist(a.orientation, b.orientation, *e)
                                    - quat_twist(j.q_ref, *e);
                                if let Some(lower) = warm_limit_state(
                                    travel,
                                    min,
                                    max,
                                    LIMIT_SLOP_ANG,
                                    j.sacc[ANGULAR_OFFSET + i],
                                ) {
                                    let initial = hinge_twist(self.rot0[j.a], self.rot0[j.b], *e)
                                        - quat_twist(j.q_ref, *e);
                                    let c = regularized_limit(
                                        travel,
                                        initial,
                                        if lower { min } else { max },
                                        lower,
                                    );
                                    let f_raw = j.pen_a[i] * c + j.sacc[ANGULAR_OFFSET + i];
                                    let f = if lower {
                                        f_raw.min(0.0)
                                    } else {
                                        f_raw.max(0.0)
                                    };
                                    if !row_live(c, f) {
                                        continue;
                                    }
                                    let g = if is_a { -dir } else { dir };
                                    let o = outer(g, g);
                                    for x in 0..ANGULAR_OFFSET {
                                        for y in 0..ANGULAR_OFFSET {
                                            lhs[ANGULAR_OFFSET + x][ANGULAR_OFFSET + y] +=
                                                j.pen_a[i] * o[x][y];
                                        }
                                    }
                                    rhs[ANGULAR_OFFSET] += f * g.x;
                                    rhs[ANGULAR_OFFSET + 1] += f * g.y;
                                    rhs[ANGULAR_OFFSET + 2] += f * g.z;
                                }
                            }
                        }
                    }
                }
                AvbdJointKind::Wheel => {
                    // Spin about the axle (`bx`) is free; lock the other
                    // two axes — same row pattern as the hinge, keyed on
                    // the axle instead of the suspension axis.
                    let axa = a.orientation * j.bx_a;
                    let axb = b.orientation * j.bx_b;
                    let axa0 = self.rot0[j.a] * j.bx_a;
                    let axb0 = self.rot0[j.b] * j.bx_b;
                    let (t1, t2) = tangent_basis(axa0);
                    for (t, lam, pen) in
                        [(t1, j.lam_a[0], j.pen_a[0]), (t2, j.lam_a[1], j.pen_a[1])]
                    {
                        let live_c = t.dot(axa - axb);
                        let c0_c = t.dot(axa0 - axb0);
                        let c = live_c - ALPHA * c0_c;
                        let f = pen * c + lam;
                        if !row_live(c, f) {
                            continue;
                        }
                        let g_ang = if is_a { axa.cross(t) } else { -(axb.cross(t)) };
                        let o = outer(g_ang, g_ang);
                        for x in 0..ANGULAR_OFFSET {
                            for y in 0..ANGULAR_OFFSET {
                                lhs[ANGULAR_OFFSET + x][ANGULAR_OFFSET + y] += pen * o[x][y];
                            }
                        }
                        rhs[ANGULAR_OFFSET] += f * g_ang.x;
                        rhs[ANGULAR_OFFSET + 1] += f * g_ang.y;
                        rhs[ANGULAR_OFFSET + 2] += f * g_ang.z;
                    }
                }
                AvbdJointKind::Fixed => {
                    // Relative error and multipliers are in A's frame;
                    // apply their torque and Hessian along world axes.
                    let diff = quat_diff_vec(a.orientation.conjugate() * b.orientation, j.q_ref);
                    let diff0 = quat_diff_vec(self.rot0[j.a].conjugate() * self.rot0[j.b], j.q_ref);
                    for k in 0..ANGULAR_OFFSET {
                        let c = diff[k] - ALPHA * diff0[k];
                        let f = j.pen_a[k] * c + j.lam_a[k];
                        if !row_live(c, f) {
                            continue;
                        }
                        let dir = a.orientation * SIXDOF_FRAME[k];
                        let g = if is_a { -dir } else { dir };
                        Self::stamp_angular_row(&mut lhs, &mut rhs, g, j.pen_a[k], f);
                    }
                }
                _ => {}
            }
            // --- one-sided limit row (Box2D order: limit wins over motor) ---
            // (Primal: warmstarts from `lim_dual` ONLY — dual commits in
            // `dual_update`, never here. Writing `lim_dual` from the primal
            // too lets the two chase each other into a limit cycle.)
            if let Some([lo, hi]) = j.lim {
                match j.kind {
                    AvbdJointKind::Revolute => {
                        let angle =
                            wrap_pi(hinge_twist(a.orientation, b.orientation, j.ax_a) - j.ref_val);
                        if let Some(lower) =
                            warm_limit_state(angle, lo, hi, LIMIT_SLOP_ANG, j.acc_lim)
                        {
                            let initial = wrap_pi(
                                hinge_twist(self.rot0[j.a], self.rot0[j.b], j.ax_a) - j.ref_val,
                            );
                            let c = regularized_limit(
                                angle,
                                initial,
                                if lower { lo } else { hi },
                                lower,
                            );
                            let f_raw = j.pen_a[2] * c + j.acc_lim;
                            let f = if lower {
                                f_raw.min(0.0)
                            } else {
                                f_raw.max(0.0)
                            };
                            if row_live(c, f) {
                                let g = if is_a { -wa } else { wa };
                                let o = outer(g, g);
                                for x in 0..ANGULAR_OFFSET {
                                    for y in 0..ANGULAR_OFFSET {
                                        lhs[ANGULAR_OFFSET + x][ANGULAR_OFFSET + y] +=
                                            j.pen_a[2] * o[x][y];
                                    }
                                }
                                rhs[ANGULAR_OFFSET] += f * g.x;
                                rhs[ANGULAR_OFFSET + 1] += f * g.y;
                                rhs[ANGULAR_OFFSET + 2] += f * g.z;
                            }
                        }
                    }
                    AvbdJointKind::Prismatic => {
                        let s = (pb - pa).dot(wa) - j.ref_val;
                        if let Some(lower) = warm_limit_state(s, lo, hi, LIMIT_SLOP_LIN, j.lim_dual)
                        {
                            let axis0 = (self.rot0[j.a] * j.ax_a).normalize_or(Vec3::Z);
                            let initial = (pb0 - pa0).dot(axis0) - j.ref_val;
                            let c =
                                regularized_limit(s, initial, if lower { lo } else { hi }, lower);
                            let f_raw = j.pen_l[2] * c + j.lim_dual;
                            let f = if lower {
                                f_raw.min(0.0)
                            } else {
                                f_raw.max(0.0)
                            };
                            if row_live(c, f) {
                                let r_side = if is_a {
                                    a.orientation * j.la
                                } else {
                                    b.orientation * j.lb
                                };
                                // C = (B-A).wa: gradients flip vs ball rows.
                                let lsign = if is_a { -1.0 } else { 1.0 };
                                Self::stamp_row(
                                    &mut lhs, &mut rhs, wa, j.pen_l[2], f, r_side, lsign,
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
            // Motors carry no primal rows: they act as deadbeat impulses
            // (see `motor_impulse`), not as penalty servos. A fixed-gain
            // servo is bang-bang at single-step dt without substeps.
        }

        let neg = [
            -rhs[0],
            -rhs[1],
            -rhs[2],
            -rhs[ANGULAR_OFFSET],
            -rhs[ANGULAR_OFFSET + 1],
            -rhs[ANGULAR_OFFSET + 2],
        ];
        if let Some(dx) = solve_6x6(lhs, neg) {
            let body = &mut self.bodies[h];
            body.position += Vec3::new(dx[0], dx[1], dx[2]);
            body.orientation = quat_integrate(
                body.orientation,
                Vec3::new(
                    dx[ANGULAR_OFFSET],
                    dx[ANGULAR_OFFSET + 1],
                    dx[ANGULAR_OFFSET + 2],
                ),
            );
        }
    }
}
