//! Joint rows, gear/motor bookkeeping and dual updates for the AVBD engine.
//!
//! Owns the joint-coordinate, gear-memory, motor-impulse and limit helpers plus
//! the [`AvbdEngine::dual_update`] sweep over pairs and joints. Split out of
//! `rows` so both files stay under the complexity gate; moved verbatim from
//! `avbd.rs` (phase 3).

use super::rows::{
    eff_inv_mass, friction_frame, inverse_symmetric, limit_state, mat3_vec, quat_diff_vec,
    regularized_limit, warm_limit_state, world_inertia, wrap_pi,
};
use super::*;
use crate::engine::joints::{hinge_twist, quat_twist};

/// Numerical zero for motor effective-mass guards: hinge/slide rows with `k` below this are skipped as degenerate (infinite mass ratio from static pairings or collapsed axes), so the deadbeat servo never divides by dust inertia.
/// Repeated per motor kind (revolute/prismatic/wheel) with one shared meaning.
const MIN_EFFECTIVE_MASS: f32 = 1e-9;

impl AvbdEngine {
    /// Step-start normal violation of a point (gap + margin).
    pub(super) fn gap_c0(&self, pi: usize, qi: usize) -> f32 {
        let p = &self.pairs[pi];
        let pt = &p.points[qi];
        let a = &self.bodies[p.a];
        let b = &self.bodies[p.b];
        if matches!(a.shape, Shape::Sphere { .. }) || matches!(b.shape, Shape::Sphere { .. }) {
            return p.gap + MARGIN;
        }
        let pa = self.pos0[p.a] + self.rot0[p.a] * pt.ra;
        let pb = self.pos0[p.b] + self.rot0[p.b] * pt.rb;
        p.n.dot(pa - pb) + MARGIN
    }

    /// Gear-compatible coordinate of an AVBD joint, measured from its
    /// assembly reference (mirrors the builtin `joint_coordinate`):
    /// revolute twist minus `ref_val`, prismatic slide minus `ref_val`.
    /// `None` for every other kind.
    pub(super) fn joint_coordinate(j: &AvbdJoint, a: &RigidBody, b: &RigidBody) -> Option<f32> {
        match j.kind {
            AvbdJointKind::Revolute => {
                Some(hinge_twist(a.orientation, b.orientation, j.ax_a) - j.ref_val)
            }
            AvbdJointKind::Prismatic => {
                let wa = (a.orientation * j.ax_a).normalize_or(Vec3::Z);
                let pa = a.position + a.orientation * j.la;
                let pb = b.position + b.orientation * j.lb;
                Some((pb - pa).dot(wa) - j.ref_val)
            }
            _ => None,
        }
    }

    /// Refresh gear angle memory once per step (see `gear_mem`): store the
    /// current raw side coordinates so `gear_sides` can unwrap continuity
    /// across PI crossings for the whole step.
    pub(super) fn update_gear_mem(&mut self) {
        for gi in 0..self.joints.len() {
            if !matches!(self.joints[gi].kind, AvbdJointKind::Gear) {
                continue;
            }
            let raw = {
                let g = &self.joints[gi];
                let mut out = [0.0f32; 2];
                // Unwrap is angular-only: a prismatic raw coordinate is
                // meters, and `wrap_pi` on meters teleports the ratio
                // residual by 2*PI (audit round 2: 0 -> 4 m read as -2.28).
                let mut ang = [false; 2];
                let mut ok = true;
                for (k, gb) in [g.gb[0], g.gb[1]].into_iter().enumerate() {
                    match self.joints.get(gb) {
                        Some(r) => {
                            let (ba, bb) = (&self.bodies[r.a], &self.bodies[r.b]);
                            match Self::joint_coordinate(r, ba, bb) {
                                Some(c) => {
                                    out[k] = c;
                                    ang[k] = matches!(r.kind, AvbdJointKind::Revolute);
                                }
                                None => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }
                (out, ang)
            };
            let (raw, ang) = raw;
            self.joints[gi].gear_mem = Some(match self.joints[gi].gear_mem {
                Some((prev_raw, prev_cont)) => (
                    raw,
                    [
                        if ang[0] {
                            prev_cont[0] + wrap_pi(raw[0] - prev_raw[0])
                        } else {
                            raw[0]
                        },
                        if ang[1] {
                            prev_cont[1] + wrap_pi(raw[1] - prev_raw[1])
                        } else {
                            raw[1]
                        },
                    ],
                ),
                None => (raw, raw),
            });
        }
    }

    /// Live dynamics of both gear sides for a gear joint (`None` on stale
    /// references or non-hinge/slide kinds — gears go quiet, never panic,
    /// like the builtin validation).
    pub(super) fn gear_sides(&self, j: &AvbdJoint) -> Option<(GearSide, GearSide)> {
        let rj = [self.joints.get(j.gb[0])?, self.joints.get(j.gb[1])?];
        let mut sides = Vec::with_capacity(2);
        for (k, r) in rj.into_iter().enumerate() {
            let (ba, bb) = (&self.bodies[r.a], &self.bodies[r.b]);
            let raw = Self::joint_coordinate(r, ba, bb)?;
            let (kind, axis, ra, rb) = match r.kind {
                AvbdJointKind::Revolute => {
                    let wa = (ba.orientation * r.ax_a).normalize_or(Vec3::Z);
                    (crate::flags::CoordKind::Angular, wa, Vec3::ZERO, Vec3::ZERO)
                }
                AvbdJointKind::Prismatic => {
                    let wa = (ba.orientation * r.ax_a).normalize_or(Vec3::Z);
                    let ra = ba.orientation * r.la;
                    let rb = bb.orientation * r.lb;
                    (crate::flags::CoordKind::Linear, wa, ra, rb)
                }
                _ => return None,
            };
            // Continuous coordinate: unwrap the live raw value against the
            // step-start memory (see `update_gear_mem`). Without this a
            // hinge crossing PI teleports the ratio residual by 2*PI.
            // Linear sides skip the unwrap: meters wrap to garbage.
            let coord = match j.gear_mem {
                Some((prev_raw, prev_cont)) if kind.is_angular() => {
                    prev_cont[k] + wrap_pi(raw - prev_raw[k])
                }
                _ => raw,
            };
            sides.push(GearSide {
                a: r.a,
                b: r.b,
                kind,
                axis,
                ra,
                rb,
                coord,
            });
        }
        let [sa, sb] = [sides.remove(0), sides.pop()?];
        Some((sa, sb))
    }

    /// Deadbeat motor impulses (official impulse semantics): the exact
    /// clamped velocity step through the pair effective mass, applied to
    /// the velocity fields BEFORE warmstart so the sweep integrates them
    /// jointly with the joint rows. (A post-solve position nudge was tried
    /// and reverted: the ball rows read it as anchor violation and undo
    /// half the spin every step.) Runs once per step; motors need no
    /// iteration and no dual state. Pauses while a limit is violated
    /// (Box2D order).
    pub(super) fn motor_impulse(&mut self) {
        for ji in 0..self.joints.len() {
            let (kind, mot) = {
                let j = &self.joints[ji];
                (j.kind, j.mot)
            };
            let Some([target, max]) = mot else { continue };
            if max <= 0.0 || self.joint_limit_violated(ji) {
                continue;
            }
            let (a, b) = {
                let j = &self.joints[ji];
                (j.a, j.b)
            };
            match kind {
                AvbdJointKind::Revolute => {
                    let wa =
                        (self.bodies[a].orientation * self.joints[ji].ax_a).normalize_or(Vec3::Z);
                    let w =
                        (self.bodies[b].angular_velocity - self.bodies[a].angular_velocity).dot(wa);
                    let ia = self.ang_inv_wa(a, wa);
                    let ib = self.ang_inv_wa(b, wa);
                    let k = ia.dot(wa) + ib.dot(wa);
                    if k < MIN_EFFECTIVE_MASS {
                        continue;
                    }
                    let dj = ((target - w) / k).clamp(-max * DT_STEP, max * DT_STEP);
                    if self.solvable(a) {
                        self.bodies[a].angular_velocity += -dj * ia;
                    }
                    if self.solvable(b) {
                        self.bodies[b].angular_velocity += dj * ib;
                    }
                }
                AvbdJointKind::Prismatic => {
                    let wa =
                        (self.bodies[a].orientation * self.joints[ji].ax_a).normalize_or(Vec3::Z);
                    let v = (self.bodies[b].velocity - self.bodies[a].velocity).dot(wa);
                    let ka = eff_inv_mass(&self.bodies[a]);
                    let kb = eff_inv_mass(&self.bodies[b]);
                    let k = ka + kb;
                    if k < MIN_EFFECTIVE_MASS {
                        continue;
                    }
                    let dj = ((target - v) / k).clamp(-max * DT_STEP, max * DT_STEP);
                    if self.solvable(a) {
                        self.bodies[a].velocity += -dj * ka * wa;
                    }
                    if self.solvable(b) {
                        self.bodies[b].velocity += dj * kb * wa;
                    }
                }
                AvbdJointKind::Wheel => {
                    // Deadbeat spin about the axle (same pattern as the
                    // hinge motor, keyed on `bx`).
                    let wa =
                        (self.bodies[a].orientation * self.joints[ji].bx_a).normalize_or(Vec3::Z);
                    let w =
                        (self.bodies[b].angular_velocity - self.bodies[a].angular_velocity).dot(wa);
                    let ia = self.ang_inv_wa(a, wa);
                    let ib = self.ang_inv_wa(b, wa);
                    let k = ia.dot(wa) + ib.dot(wa);
                    if k < MIN_EFFECTIVE_MASS {
                        continue;
                    }
                    let dj = ((target - w) / k).clamp(-max * DT_STEP, max * DT_STEP);
                    if self.solvable(a) {
                        self.bodies[a].angular_velocity += -dj * ia;
                    }
                    if self.solvable(b) {
                        self.bodies[b].angular_velocity += dj * ib;
                    }
                }
                _ => {}
            }
        }
    }

    /// World inverse inertia times a direction (zero unless dynamic).
    fn ang_inv_wa(&self, h: usize, wa: Vec3) -> Vec3 {
        let b = &self.bodies[h];
        if !self.solvable(h) {
            return Vec3::ZERO;
        }
        let iw = world_inertia(b.inertia, b.orientation);
        let iw_inv = inverse_symmetric(
            iw,
            Vec3::new(
                b.inertia.x.max(0.0),
                b.inertia.y.max(0.0),
                b.inertia.z.max(0.0),
            ),
        );
        mat3_vec(iw_inv, wa)
    }

    /// Current limit-violation state of a joint (for the motor pause rule).
    fn joint_limit_violated(&self, ji: usize) -> bool {
        let j = &self.joints[ji];
        let Some([lo, hi]) = j.lim else {
            return false;
        };
        let a = &self.bodies[j.a];
        let b = &self.bodies[j.b];
        match j.kind {
            AvbdJointKind::Revolute => {
                let angle = wrap_pi(hinge_twist(a.orientation, b.orientation, j.ax_a) - j.ref_val);
                limit_state(angle, lo, hi, LIMIT_SLOP_ANG).is_some()
            }
            AvbdJointKind::Prismatic => {
                let wa = (a.orientation * j.ax_a).normalize_or(Vec3::Z);
                let pa = a.position + a.orientation * j.la;
                let pb = b.position + b.orientation * j.lb;
                let s = (pb - pa).dot(wa) - j.ref_val;
                limit_state(s, lo, hi, LIMIT_SLOP_LIN).is_some()
            }
            _ => false,
        }
    }
    /// Dual update for every pair and joint (official dual core).
    pub(super) fn dual_update(&mut self) {
        for pi in 0..self.pairs.len() {
            let (n, mu) = {
                let p = &self.pairs[pi];
                (p.n, p.mu)
            };
            let (t1, _, _) = {
                let p = &self.pairs[pi];
                friction_frame(&self.bodies[p.a], &self.bodies[p.b], p.n)
            };
            let t2 = t1.cross(n);
            for qi in 0..self.pairs[pi].points.len() {
                // Separated-damper gate (mirror of the primal rule): open
                // pairs commit no dual memory — support must be earned by
                // touch, never stored from a hover. The `stuck` update
                // below still runs (slide detection for the transition).
                let touching = self.pairs[pi].gap <= 0.0;
                let c0 = if touching { self.gap_c0(pi, qi) } else { 0.0 };
                let (cn, _, _) = self.row_c(&self.pairs[pi], &self.pairs[pi].points[qi], n, c0);
                // Dual friction residuals mirror the primal rows: coincident
                // levers (see `friction_levers`), or C/J disagree.
                let (ra_f, rb_f) = {
                    let p = &self.pairs[pi];
                    Self::friction_levers(&self.bodies[p.a], &self.bodies[p.b], p.n, &p.points[qi])
                };
                let (ct1, _, _) = self.row_c_levers(&self.pairs[pi], t1, 0.0, ra_f, rb_f);
                let (ct2, _, _) = self.row_c_levers(&self.pairs[pi], t2, 0.0, ra_f, rb_f);
                let (pen, lam) = {
                    let pt = &self.pairs[pi].points[qi];
                    (pt.pen, pt.lam)
                };
                let f = Self::contact_force(cn, pen, lam, [ct1, ct2], mu);
                // Static-grip flag (official `stick`, manifold.cpp:173):
                // verbatim port — set while the UNSCALED tangential force
                // sits inside the cone (`frictionScale <= bounds`), with the
                // absolute 1e-5 position threshold. No lever scaling: the
                // whirl limit-cycle (±100m via the prismatic limit row) was
                // a DUAL-gate bug (the flag was set unconditionally), not a
                // threshold bug — fixed by gating on the cone state. The
                // 1%-of-lever scaling tried here re-broke the prismatic
                // limit test (anchors refreshed every step under gravity
                // load, support never accumulated).
                {
                    let bnd = f[0].abs() * mu[0].max(mu[1]);
                    let ft_u = [pen[1] * ct1 + lam[1], pen[2] * ct2 + lam[2]];
                    let fs = (ft_u[0] * ft_u[0] + ft_u[1] * ft_u[1]).sqrt();
                    let stuck_now = fs <= bnd && (ct1 * ct1 + ct2 * ct2).sqrt() < STICK_THRESH;
                    self.pairs[pi].points[qi].stuck = stuck_now;
                }
                let pt = &mut self.pairs[pi].points[qi];
                if touching && cn.abs() >= C_EPS {
                    pt.lam[0] = f[0];
                    if f[0] < 0.0 {
                        pt.pen[0] = (pt.pen[0] + BETA * cn.abs()).min(PENALTY_MAX);
                    }
                }
                // Elliptical within-bounds gate on the UNSCALED tangential
                // force (official `frictionScale <= bounds`).
                let b1 = f[0].abs() * mu[0];
                let b2 = f[0].abs() * mu[1];
                let ft_u = [pen[1] * ct1 + lam[1], pen[2] * ct2 + lam[2]];
                let s_ell = if b1 > 0.0 && b2 > 0.0 {
                    (ft_u[0] / b1) * (ft_u[0] / b1) + (ft_u[1] / b2) * (ft_u[1] / b2)
                } else {
                    f32::INFINITY
                };
                // The bounded force commits on every touching step, in-cone
                // or sliding: freezing `lam` outside the cone drops the
                // sliding reaction. Only the penalty ramp is gated on the
                // within-bounds ellipse (official `frictionScale <= bounds`).
                if touching {
                    pt.lam[1] = f[1];
                    pt.lam[2] = f[2];
                    if s_ell <= 1.0 {
                        if ct1.abs() >= C_EPS {
                            pt.pen[1] = (pt.pen[1] + BETA * ct1.abs()).min(PENALTY_MAX);
                        }
                        if ct2.abs() >= C_EPS {
                            pt.pen[2] = (pt.pen[2] + BETA * ct2.abs()).min(PENALTY_MAX);
                        }
                    }
                }
                // Rolling/torsion dual: same rows, torque cap from the fresh
                // normal force. No ramp, no deadband (slow rolling needs
                // small-C response).
                {
                    let p = &self.pairs[pi];
                    let ba = &self.bodies[p.a];
                    let bb = &self.bodies[p.b];
                    let mu_roll = ba.rolling_friction.max(bb.rolling_friction);
                    let mu_spin = ba.torsion_friction.max(bb.torsion_friction);
                    if touching && (mu_roll > 0.0 || mu_spin > 0.0) {
                        let drot_a = quat_diff_vec(self.bodies[p.a].orientation, self.rot0[p.a]);
                        let drot_b = quat_diff_vec(self.bodies[p.b].orientation, self.rot0[p.b]);
                        let wrel = drot_a - drot_b;
                        let lam_n = self.pairs[pi].points[qi].lam[0].abs();
                        let rl = self.pairs[pi].points[qi].roll_lam;
                        for (ax, li, mur) in [(t1, 0, mu_roll), (t2, 1, mu_roll), (n, 2, mu_spin)] {
                            if mur <= 0.0 {
                                continue;
                            }
                            let c = wrel.dot(ax);
                            let cap = mur * lam_n;
                            let fr = (ROLL_PEN * c + rl[li]).clamp(-cap, cap);
                            self.pairs[pi].points[qi].roll_lam[li] = fr;
                        }
                    }
                }
            }
        }
        for j in self.joints.iter_mut() {
            let a = &self.bodies[j.a];
            let b = &self.bodies[j.b];
            let pa = a.position + a.orientation * j.la;
            let pb = b.position + b.orientation * j.lb;
            let pa0 = self.pos0[j.a] + self.rot0[j.a] * j.la;
            let pb0 = self.pos0[j.b] + self.rot0[j.b] * j.lb;
            let live = pa - pb;
            let c0v = pa0 - pb0;
            // Linear equality rows, same C as the primal per kind.
            match j.kind {
                AvbdJointKind::Ball | AvbdJointKind::Revolute | AvbdJointKind::Fixed => {
                    for k in 0..3 {
                        let (live, initial) = if j.kind == AvbdJointKind::Fixed {
                            (live + a.orientation * j.dref, c0v + self.rot0[j.a] * j.dref)
                        } else {
                            (live, c0v)
                        };
                        let c = live[k] - ALPHA * initial[k];
                        if c.abs() >= C_EPS {
                            j.lam_l[k] += j.pen_l[k] * c;
                            j.pen_l[k] = (j.pen_l[k] + BETA * c.abs()).min(PENALTY_MAX);
                        }
                    }
                }
                AvbdJointKind::Prismatic => {
                    let wa = (a.orientation * j.ax_a).normalize_or(Vec3::Z);
                    let (u, v) = tangent_basis(wa);
                    for (ax, li) in [(u, 0), (v, 1)] {
                        let c = (live - ALPHA * c0v).dot(ax);
                        if c.abs() >= C_EPS {
                            j.lam_l[li] += j.pen_l[li] * c;
                            j.pen_l[li] = (j.pen_l[li] + BETA * c.abs()).min(PENALTY_MAX);
                        }
                    }
                }
                AvbdJointKind::Distance => {
                    let len = live.length();
                    let c = (len - j.ref_val) - ALPHA * (c0v.length() - j.ref_val);
                    if c.abs() >= C_EPS {
                        j.lam_l[0] += j.pen_l[0] * c;
                        j.pen_l[0] = (j.pen_l[0] + BETA * c.abs()).min(PENALTY_MAX);
                    }
                }
                AvbdJointKind::SixDof => {
                    // Same C as the primal per axis: locked accumulate,
                    // limited commit into the dual-owned `sacc` (zeroed
                    // when clear), penalties ramp capped.
                    for (i, e) in SIXDOF_FRAME.iter().enumerate() {
                        let dir = (a.orientation * *e).normalize_or(*e);
                        match j.six_lin[i] {
                            AxisConfig::Free => {}
                            AxisConfig::Locked => {
                                let evec = (pb - pa) - a.orientation * j.dref;
                                let c = evec.dot(dir);
                                if c.abs() >= C_EPS {
                                    j.lam_l[i] += j.pen_l[i] * c;
                                    j.pen_l[i] = (j.pen_l[i] + BETA * c.abs()).min(PENALTY_MAX);
                                }
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
                                    if c.abs() >= C_EPS {
                                        let f = j.pen_l[i] * c + j.sacc[i];
                                        j.sacc[i] = if lower { f.min(0.0) } else { f.max(0.0) };
                                        j.pen_l[i] = (j.pen_l[i] + BETA * c.abs()).min(LIM_PEN_MAX);
                                    }
                                } else {
                                    j.sacc[i] = 0.0;
                                }
                            }
                        }
                    }
                    // Angular rows mirror the primal six_ang rows (same C):
                    // locked accumulate into `lam_a`, limited commit into
                    // the dual-owned `sacc[3..6]` (zeroed when clear).
                    let qrel = a.orientation.conjugate() * b.orientation;
                    let diff = quat_diff_vec(qrel, j.q_ref);
                    for (i, e) in SIXDOF_FRAME.iter().enumerate() {
                        match j.six_ang[i] {
                            AxisConfig::Free => {}
                            AxisConfig::Locked => {
                                let c = diff.dot(*e);
                                if c.abs() >= C_EPS {
                                    j.lam_a[i] += j.pen_a[i] * c;
                                    j.pen_a[i] = (j.pen_a[i] + BETA_ANG * c.abs()).min(PENALTY_MAX);
                                }
                            }
                            AxisConfig::Limited { min, max } => {
                                let travel = hinge_twist(a.orientation, b.orientation, *e)
                                    - quat_twist(j.q_ref, *e);
                                if let Some(lower) = warm_limit_state(
                                    travel,
                                    min,
                                    max,
                                    LIMIT_SLOP_ANG,
                                    j.sacc[3 + i],
                                ) {
                                    let initial = hinge_twist(self.rot0[j.a], self.rot0[j.b], *e)
                                        - quat_twist(j.q_ref, *e);
                                    let c = regularized_limit(
                                        travel,
                                        initial,
                                        if lower { min } else { max },
                                        lower,
                                    );
                                    if c.abs() >= C_EPS {
                                        let f = j.pen_a[i] * c + j.sacc[3 + i];
                                        j.sacc[3 + i] = if lower { f.min(0.0) } else { f.max(0.0) };
                                        j.pen_a[i] =
                                            (j.pen_a[i] + BETA_ANG * c.abs()).min(LIM_PEN_MAX);
                                    }
                                } else {
                                    j.sacc[3 + i] = 0.0;
                                }
                            }
                        }
                    }
                }
                AvbdJointKind::Gear => {
                    // No per-joint dual state here: gears read other joints
                    // (borrow conflict inside this loop) — updated in the
                    // separate gear pass at the end of `dual_update`.
                }
                AvbdJointKind::Wheel => {
                    // Perp rows (same C as the primal) + the rigid-degrade
                    // equality on slot 2. The live spring carries no dual
                    // state (fixed penalty, sag holds the load).
                    let wa = (a.orientation * j.ax_a).normalize_or(Vec3::Z);
                    let (u, v) = tangent_basis(wa);
                    for (ax, li) in [(u, 0), (v, 1)] {
                        let c = (live - ALPHA * c0v).dot(ax);
                        if c.abs() >= C_EPS {
                            j.lam_l[li] += j.pen_l[li] * c;
                            j.pen_l[li] = (j.pen_l[li] + BETA * c.abs()).min(PENALTY_MAX);
                        }
                    }
                    if j.susp[0] <= 0.0 {
                        let wa0 = (self.rot0[j.a] * j.ax_a).normalize_or(Vec3::Z);
                        let pa0 = self.pos0[j.a] + self.rot0[j.a] * j.la;
                        let pb0 = self.pos0[j.b] + self.rot0[j.b] * j.lb;
                        // Same C as the primal (`live`/`c0v` are pa-pb, so
                        // the slide `s = (pb-pa).wa - ref` reads
                        // `-live.wa - ref` here).
                        let c = (-live.dot(wa) - j.ref_val)
                            - ALPHA * (-(pa0 - pb0).dot(wa0) - j.ref_val);
                        if c.abs() >= C_EPS {
                            j.lam_l[2] += j.pen_l[2] * c;
                            j.pen_l[2] = (j.pen_l[2] + BETA * c.abs()).min(LIM_PEN_MAX);
                        }
                    }
                }
            }
            // Angular equality rows.
            match j.kind {
                AvbdJointKind::Revolute | AvbdJointKind::Prismatic => {
                    let axa = a.orientation * j.ax_a;
                    let axb = b.orientation * j.ax_b;
                    let axa0 = self.rot0[j.a] * j.ax_a;
                    let axb0 = self.rot0[j.b] * j.ax_b;
                    let (t1, t2) = tangent_basis(axa0);
                    for (t, li) in [(t1, 0), (t2, 1)] {
                        let c = t.dot(axa - axb) - ALPHA * t.dot(axa0 - axb0);
                        if c.abs() < C_EPS {
                            continue;
                        }
                        let f = j.pen_a[li] * c + j.lam_a[li];
                        j.lam_a[li] = f;
                        j.pen_a[li] = (j.pen_a[li] + BETA_ANG * c.abs()).min(PENALTY_MAX);
                    }
                }
                AvbdJointKind::Wheel => {
                    // Same pattern keyed on the axle (spin about it is free).
                    let axa = a.orientation * j.bx_a;
                    let axb = b.orientation * j.bx_b;
                    let axa0 = self.rot0[j.a] * j.bx_a;
                    let axb0 = self.rot0[j.b] * j.bx_b;
                    let (t1, t2) = tangent_basis(axa0);
                    for (t, li) in [(t1, 0), (t2, 1)] {
                        let c = t.dot(axa - axb) - ALPHA * t.dot(axa0 - axb0);
                        if c.abs() < C_EPS {
                            continue;
                        }
                        let f = j.pen_a[li] * c + j.lam_a[li];
                        j.lam_a[li] = f;
                        j.pen_a[li] = (j.pen_a[li] + BETA_ANG * c.abs()).min(PENALTY_MAX);
                    }
                }
                AvbdJointKind::Fixed => {
                    let diff = quat_diff_vec(a.orientation.conjugate() * b.orientation, j.q_ref);
                    let diff0 = quat_diff_vec(self.rot0[j.a].conjugate() * self.rot0[j.b], j.q_ref);
                    for k in 0..3 {
                        let c = diff[k] - ALPHA * diff0[k];
                        if c.abs() < C_EPS {
                            continue;
                        }
                        j.lam_a[k] += j.pen_a[k] * c;
                        j.pen_a[k] = (j.pen_a[k] + BETA_ANG * c.abs()).min(PENALTY_MAX);
                    }
                }
                _ => {}
            }
            // One-sided limit accumulator (official `acc_limit` discipline:
            // clamp self-corrects on side flips; zeroed when clear).
            // DUAL-SIDE (prismatic): this is `dual_update` (mutable joints,
            // live bodies) — the ONLY writer of `lim_dual`. The primal only
            // WARMS from the slot, it never writes it (both writing lets
            // the two chase each other into a limit cycle under sustained
            // load — the prismatic slider snapped to +5.7m / ±100m after
            // ~200 steps). The revolute row keeps `acc_lim` (see field
            // docs on `lim_dual`).
            if let Some([lo, hi]) = j.lim {
                match j.kind {
                    AvbdJointKind::Revolute => {
                        let angle =
                            wrap_pi(hinge_twist(a.orientation, b.orientation, j.ax_a) - j.ref_val);
                        match warm_limit_state(angle, lo, hi, LIMIT_SLOP_ANG, j.acc_lim) {
                            Some(lower) => {
                                let initial = wrap_pi(
                                    hinge_twist(self.rot0[j.a], self.rot0[j.b], j.ax_a) - j.ref_val,
                                );
                                let c = regularized_limit(
                                    angle,
                                    initial,
                                    if lower { lo } else { hi },
                                    lower,
                                );
                                if c.abs() >= C_EPS {
                                    let f = j.pen_a[2] * c + j.acc_lim;
                                    j.acc_lim = if lower { f.min(0.0) } else { f.max(0.0) };
                                    // Capped (LIM_PEN_MAX): a limit holds a
                                    // persistent bias, an uncapped ramp
                                    // crosses explicit stability and diverges.
                                    j.pen_a[2] = (j.pen_a[2] + BETA_ANG * c.abs()).min(LIM_PEN_MAX);
                                }
                            }
                            None => {
                                j.acc_lim = 0.0;
                            }
                        }
                    }
                    AvbdJointKind::Prismatic => {
                        let wa = (a.orientation * j.ax_a).normalize_or(Vec3::Z);
                        let s = (pb - pa).dot(wa) - j.ref_val;
                        match warm_limit_state(s, lo, hi, LIMIT_SLOP_LIN, j.lim_dual) {
                            Some(lower) => {
                                let axis0 = (self.rot0[j.a] * j.ax_a).normalize_or(Vec3::Z);
                                let initial = (pb0 - pa0).dot(axis0) - j.ref_val;
                                let c = regularized_limit(
                                    s,
                                    initial,
                                    if lower { lo } else { hi },
                                    lower,
                                );
                                if c.abs() >= C_EPS {
                                    // Project the same warm-inclusive force as the
                                    // primal. At zero C the multiplier carries the
                                    // load; a signed slop residual unwinds it.
                                    let f_raw = j.pen_l[2] * c + j.lim_dual;
                                    let f = if lower {
                                        f_raw.min(0.0)
                                    } else {
                                        f_raw.max(0.0)
                                    };
                                    j.lim_dual = f;
                                    j.pen_l[2] = (j.pen_l[2] + BETA * c.abs()).min(LIM_PEN_MAX);
                                }
                            }
                            None => {
                                j.acc_lim = 0.0;
                                j.lim_dual = 0.0;
                            }
                        }
                    }
                    _ => {}
                }
            }
            // Motors carry no dual state (fresh servo solve every primal).
        }
        // Gear dual pass (separate loop — gears read other joints while
        // the per-joint loop above holds a mutable joint borrow).
        // Penalty capped at LIM_PEN_MAX (explicit-servo stability: a gear
        // holds a persistent bias like a limit, and couples four bodies).
        for gi in 0..self.joints.len() {
            if !matches!(self.joints[gi].kind, AvbdJointKind::Gear) {
                continue;
            }
            let c = {
                let g = &self.joints[gi];
                let Some((sa, sb)) = self.gear_sides(g) else {
                    continue;
                };
                sa.coord + g.gratio * sb.coord - g.ref_val
            };
            if c.abs() < C_EPS {
                continue;
            }
            let g = &mut self.joints[gi];
            g.lam_l[0] += g.pen_l[0] * c;
            g.pen_l[0] = (g.pen_l[0] + BETA * c.abs()).min(LIM_PEN_MAX);
        }
    }
}
