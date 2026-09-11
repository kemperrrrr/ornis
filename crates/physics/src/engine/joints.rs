//! Joint sub-solver for `BuiltinPhysicsEngine` (G5): velocity and position
//! stages of the ball/revolute joint constraints. Split out of `engine.rs`
//! to keep each type's method count within the structural gate's thresholds.

use std::f32::consts::{PI, TAU};

use glam::Quat;
use glam::Vec3;

use super::*;
use crate::joint::{
    AxisConfig, PrismaticLimit, PrismaticMotor, RevoluteLimit, RevoluteMotor, WheelSuspension,
};

/// Twist of B relative to A about A's hinge axis (rad, wrapped to
/// [-PI, PI]). Decomposes `qa^-1 * qb` into twist about the axis plus
/// swing; limits, motors and their tests measure travel with this.
pub(super) fn hinge_twist(qa: Quat, qb: Quat, axis_a: Vec3) -> f32 {
    quat_twist(qa.conjugate() * qb, axis_a)
}

/// Twist component of a relative rotation about a local axis (rad, wrapped
/// to [-PI, PI]): the shared core of [`hinge_twist`], also used to read the
/// assembly twist out of a stored reference orientation.
fn quat_twist(q: Quat, axis: Vec3) -> f32 {
    let t = 2.0 * q.xyz().dot(axis).atan2(q.w);
    (t + PI).rem_euclid(TAU) - PI
}

/// World hinge frame for a revolute joint: normalized world axis plus its
/// fixed tangent pair (the plane the angular correction lives in).
fn hinge_frame(orientation: Quat, axis: Vec3) -> (Vec3, Vec3, Vec3) {
    let wa = (orientation * axis).normalize_or(Vec3::Z);
    let t1 = tangent_basis(wa);
    let t2 = wa.cross(t1).normalize_or_zero();
    (wa, t1, t2)
}

impl BuiltinPhysicsEngine {
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
    /// `sub_dt` bounds the motor impulse (`max_torque * sub_dt`).
    pub(super) fn solve_joints_velocity(&mut self, sub_dt: f32) {
        if self.joints.is_empty() {
            return;
        }

        let Self {
            bodies,
            joints,
            asleep,
            velocity_iterations,
            ..
        } = self;

        for joint in joints.iter_mut() {
            let (a, b) = (joint.body_a, joint.body_b);
            // Gears resolve four bodies and check sleep in their own pass.
            if matches!(joint.kind, JointKind::Gear { .. }) {
                continue;
            }
            // A fully sleeping jointed pair is frozen; island-coherent sleep
            // guarantees both members share the sleep state.
            if asleep[a] && asleep[b] {
                continue;
            }
            if !matches!(
                joint.kind,
                JointKind::Ball { .. } | JointKind::Revolute { .. } | JointKind::Prismatic { .. }
            ) {
                solve_new_joint_velocity(bodies, joint, a, b, *velocity_iterations, sub_dt);
                continue;
            }
            let (la, lb) = joint.local_anchors();
            let align_axes = alignment_axes(&joint.kind);
            let prism_axis_a = joint.prismatic_axes().map(|(axis_a, _)| axis_a);

            // --- Warm start: re-apply the accumulated impulses (G2b pattern).
            // Anchors are computed ONCE here and reused verbatim by the
            // iterations below (original behaviour).
            let ra = bodies[a].orientation * la;
            let rb = bodies[b].orientation * lb;
            joint_warm_start(bodies, joint, a, b, ra, rb, align_axes);

            // --- Velocity iterations.
            for _ in 0..*velocity_iterations {
                if let Some(axis_a) = prism_axis_a {
                    // Box2D point-to-line: constrain the two perpendicular
                    // directions, leave the slide axis free.
                    let wa = (bodies[a].orientation * axis_a).normalize_or(Vec3::Z);
                    let t1 = tangent_basis(wa);
                    let t2 = wa.cross(t1).normalize_or_zero();
                    for t in [t1, t2] {
                        joint_prismatic_linear_iteration(bodies, joint, a, b, ra, rb, t);
                    }
                } else {
                    joint_linear_velocity_iteration(bodies, joint, a, b, ra, rb);
                }
                if let Some((axis_a, _)) = align_axes {
                    joint_angular_velocity_iteration(bodies, joint, a, b, axis_a);
                }
            }
            // --- Limit/motor drive on the free axis (revolute hinge or
            // prismatic slide). Motor first, then the limit wins past the
            // bounds (Box2D order); the motor pauses while a limit is
            // violated.
            if prism_axis_a.is_some() {
                let (limit, motor) = joint.prismatic_drive();
                if limit.is_some() || motor.is_some() {
                    joint_prismatic_drive_velocity_iteration(
                        bodies, joint, a, b, ra, rb, limit, motor, sub_dt,
                    );
                }
            } else if let Some((axis_a, _)) = align_axes {
                let (limit, motor) = joint.drive();
                if limit.is_some() || motor.is_some() {
                    joint_drive_velocity_iteration(
                        bodies, joint, a, b, axis_a, limit, motor, sub_dt,
                    );
                }
            }
        }
        // Gear pass (separate loop: gears read other joints while the
        // bodies are mutably borrowed — the two cannot mix in one loop).
        self.solve_gears_velocity(sub_dt);
    }

    /// Position stage of the joint solver (split impulse: positions only).
    /// Runs after `integrate_positions`, like Box3D's joint position pass.
    pub(super) fn solve_joints_position(&mut self) {
        if self.joints.is_empty() {
            return;
        }
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
            // Gears are velocity-only (Baumgarte-stabilized, no position
            // pass — same standing as the velocity-only hinge/slide limits).
            if matches!(joint.kind, JointKind::Gear { .. }) {
                continue;
            }
            if asleep[a] && asleep[b] {
                continue;
            }
            if !matches!(
                joint.kind,
                JointKind::Ball { .. } | JointKind::Revolute { .. } | JointKind::Prismatic { .. }
            ) {
                solve_new_joint_position(bodies, joint, a, b, *position_iterations);
                continue;
            }
            let (la, lb) = joint.local_anchors();
            let align_axes = alignment_axes(&joint.kind);
            let prism_axis_a = joint.prismatic_axes().map(|(axis_a, _)| axis_a);

            for _ in 0..*position_iterations {
                let ra = bodies[a].orientation * la;
                let rb = bodies[b].orientation * lb;
                let c = (bodies[b].position + rb) - (bodies[a].position + ra);
                if let Some(axis_a) = prism_axis_a {
                    // Prismatic: anchor coincidence only across the slide
                    // (the slide direction is free by design).
                    let wa = (bodies[a].orientation * axis_a).normalize_or(Vec3::Z);
                    let t1 = tangent_basis(wa);
                    let t2 = wa.cross(t1).normalize_or_zero();
                    for dir in [t1, t2] {
                        joint_linear_position_step(bodies, a, b, ra, rb, c.dot(dir), dir);
                    }
                } else {
                    for dir in AXES {
                        joint_linear_position_step(bodies, a, b, ra, rb, c.dot(dir), dir);
                    }
                }
                if let Some((axis_a, axis_b)) = align_axes {
                    joint_angular_position_pass(bodies, a, b, axis_a, axis_b);
                }
            }
        }
    }

    /// Gear velocity pass (separate loop — gears read the referenced joints
    /// while the bodies are mutably borrowed). One Baumgarte-stabilized
    /// velocity solve per substep, warm-started; no position pass (same
    /// standing as the velocity-only hinge/slide limits — a documented
    /// deviation from Box2D, which also solves gear positions).
    fn solve_gears_velocity(&mut self, sub_dt: f32) {
        const BETA: f32 = 0.2;
        const MAX_C: f32 = 0.5;
        for gi in 0..self.joints.len() {
            let (ja, jb, ratio, c0) = match &self.joints[gi].kind {
                JointKind::Gear {
                    joint_a,
                    joint_b,
                    ratio,
                } => (
                    *joint_a,
                    *joint_b,
                    *ratio,
                    self.joints[gi].reference_distance,
                ),
                _ => continue,
            };
            // Resolve + validate the referenced joints (removals rebuild the
            // table, but a stale index must go quiet, never panic).
            let (Some(ra), Some(rb)) = (self.joints.get(ja), self.joints.get(jb)) else {
                continue;
            };
            let (Some(sa), Some(sb)) = (
                gear_side_data(&self.bodies, ra),
                gear_side_data(&self.bodies, rb),
            ) else {
                continue;
            };
            // All four bodies asleep = frozen assembly.
            if self.asleep[sa.a] && self.asleep[sa.b] && self.asleep[sb.a] && self.asleep[sb.b] {
                continue;
            }
            // Warm start: re-apply the accumulated generalized impulse.
            let acc = self.joints[gi].acc_gear;
            if acc.abs() > 1e-12 {
                gear_apply_delta(&mut self.bodies, &sa, acc);
                gear_apply_delta(&mut self.bodies, &sb, acc * ratio);
            }
            // Re-read the sides after the warm start (axes/anchors moved).
            let (Some(sa), Some(sb)) = (
                gear_side_data(&self.bodies, &self.joints[ja]),
                gear_side_data(&self.bodies, &self.joints[jb]),
            ) else {
                continue;
            };
            let c = (sa.coord + ratio * sb.coord - c0).clamp(-MAX_C, MAX_C);
            let k = sa.eff + ratio * ratio * sb.eff;
            if k < 1e-9 || sub_dt <= 0.0 {
                continue;
            }
            let dl = -((sa.rate + ratio * sb.rate) + BETA * c / sub_dt) / k;
            self.joints[gi].acc_gear += dl;
            gear_apply_delta(&mut self.bodies, &sa, dl);
            gear_apply_delta(&mut self.bodies, &sb, dl * ratio);
        }
    }
}

/// Axis-alignment equality axes of a joint with a distinguished direction
/// (revolute hinge or prismatic slide): the two axes must stay parallel,
/// enforced by 2 angular constraints. None for every other joint (fixed,
/// wheel and six-DOF locks run their own angular passes; ball, distance
/// and gear constrain no axes).
fn alignment_axes(kind: &JointKind) -> Option<(Vec3, Vec3)> {
    match kind {
        JointKind::Revolute {
            local_axis_a,
            local_axis_b,
            ..
        }
        | JointKind::Prismatic {
            local_axis_a,
            local_axis_b,
            ..
        } => Some((*local_axis_a, *local_axis_b)),
        _ => None,
    }
}

/// Warm start for one joint: re-apply the accumulated linear and angular
/// impulses from last substep (G2b pattern).
#[allow(clippy::needless_range_loop)]
fn joint_warm_start(
    bodies: &mut [RigidBody],
    joint: &Joint,
    a: usize,
    b: usize,
    ra: Vec3,
    rb: Vec3,
    revolute_axes: Option<(Vec3, Vec3)>,
) {
    const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];
    for (k, dir) in AXES.iter().enumerate() {
        let l = joint.acc_lin[k];
        if l.abs() > 1e-12 {
            apply_impulse(bodies, a, b, dir * l, ra, rb);
        }
    }
    if let Some((axis_a, _)) = revolute_axes {
        let (_, t1, t2) = hinge_frame(bodies[a].orientation, axis_a);
        for (k, t) in [t1, t2].iter().enumerate() {
            let l = joint.acc_ang[k];
            if l.abs() > 1e-12 {
                apply_angular_impulse(bodies, a, b, t * l);
            }
        }
    }
}

/// One linear velocity iteration for one joint (3 world-axis equality
/// constraints at the anchor points). Equality constraint: no clamp, any sign
/// of impulse.
#[allow(clippy::needless_range_loop)]
fn joint_linear_velocity_iteration(
    bodies: &mut [RigidBody],
    joint: &mut Joint,
    a: usize,
    b: usize,
    ra: Vec3,
    rb: Vec3,
) {
    const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];
    for (k, dir) in AXES.iter().enumerate() {
        let k_eff = effective_mass(bodies, a, b, *dir, ra, rb);
        if k_eff < 1e-9 {
            continue;
        }
        let vrel = (point_velocity(&bodies[b], rb) - point_velocity(&bodies[a], ra)).dot(*dir);
        let dl = -vrel / k_eff;
        joint.acc_lin[k] += dl;
        apply_impulse(bodies, a, b, dir * dl, ra, rb);
    }
}

/// One angular velocity iteration for a revolute joint (2 equality
/// constraints along the tangents of the hinge axis).
#[allow(clippy::needless_range_loop)]
fn joint_angular_velocity_iteration(
    bodies: &mut [RigidBody],
    joint: &mut Joint,
    a: usize,
    b: usize,
    axis_a: Vec3,
) {
    let (_, t1, t2) = hinge_frame(bodies[a].orientation, axis_a);
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

/// One linear velocity iteration for a prismatic joint along a single
/// perpendicular direction `t` (point-to-line, Box2D formulation).
/// Equality constraint: no clamp, any sign of impulse. The impulse is
/// accumulated into the joint's world-axis totals (frame-independent —
/// a linear impulse is a linear impulse whatever basis it was solved in),
/// so the shared ball-style warm start stays exact.
#[allow(clippy::needless_range_loop)]
fn joint_prismatic_linear_iteration(
    bodies: &mut [RigidBody],
    joint: &mut Joint,
    a: usize,
    b: usize,
    ra: Vec3,
    rb: Vec3,
    t: Vec3,
) {
    let k_eff = effective_mass(bodies, a, b, t, ra, rb);
    if k_eff < 1e-9 {
        return;
    }
    let vrel = (point_velocity(&bodies[b], rb) - point_velocity(&bodies[a], ra)).dot(t);
    let dl = -vrel / k_eff;
    joint.acc_lin[0] += dl * t.x;
    joint.acc_lin[1] += dl * t.y;
    joint.acc_lin[2] += dl * t.z;
    apply_impulse(bodies, a, b, t * dl, ra, rb);
}

/// Slide-axis drive: velocity motor toward its target speed, then the
/// one-sided travel limit (Box2D order — the limit wins past the bounds).
/// Linear mirror of [`joint_drive_velocity_iteration`]: same accumulator
/// discipline (`acc_limit`), force clamp instead of torque clamp.
#[allow(clippy::too_many_arguments)]
fn joint_prismatic_drive_velocity_iteration(
    bodies: &mut [RigidBody],
    joint: &mut Joint,
    a: usize,
    b: usize,
    ra: Vec3,
    rb: Vec3,
    limit: Option<PrismaticLimit>,
    motor: Option<PrismaticMotor>,
    sub_dt: f32,
) {
    const LINEAR_SLOP: f32 = 0.002;
    let wa = (bodies[a].orientation * joint.prismatic_axes().map(|(x, _)| x).unwrap_or(Vec3::Z))
        .normalize_or(Vec3::Z);
    let s =
        ((bodies[b].position + rb) - (bodies[a].position + ra)).dot(wa) - joint.reference_length;
    let k_eff = effective_mass(bodies, a, b, wa, ra, rb);
    if k_eff < 1e-9 {
        return;
    }
    let v = (point_velocity(&bodies[b], rb) - point_velocity(&bodies[a], ra)).dot(wa);
    let violated = match limit {
        Some(lim) if s <= lim.min + LINEAR_SLOP => Some(true),
        Some(lim) if s >= lim.max - LINEAR_SLOP => Some(false),
        _ => None,
    };
    match violated {
        None => {
            joint.acc_limit = 0.0;
            if let Some(m) = motor
                && sub_dt > 0.0
                && m.max_force > 0.0
            {
                let dl = ((m.target_speed - v) / k_eff)
                    .clamp(-m.max_force * sub_dt, m.max_force * sub_dt);
                apply_impulse(bodies, a, b, wa * dl, ra, rb);
            }
        }
        // One-sided block: lower forbids v < 0 (accumulator >= 0), upper
        // forbids v > 0 (accumulator <= 0). The clamp self-corrects on side
        // flips by dumping the stale impulse in one step.
        Some(lower) => {
            let dl = -v / k_eff;
            let next = if lower {
                (joint.acc_limit + dl).max(0.0)
            } else {
                (joint.acc_limit + dl).min(0.0)
            };
            apply_impulse(bodies, a, b, wa * (next - joint.acc_limit), ra, rb);
            joint.acc_limit = next;
        }
    }
}
/// Hinge-axis drive: velocity motor toward its target speed, then the
/// one-sided travel limit (Box2D order — the limit wins past the bounds).
/// The motor pauses while a limit is violated and resumes inside the window.
///
/// Limits are velocity-only (Box2D parity): no position correction, the
/// accumulated one-sided impulse plus a small slop holds the bound. Motor
/// needs no accumulator: the torque-clamped target solve converges in one
/// iteration.
///
/// Which travel bound (if any) the hinge violates: `Some(true)` = lower,
/// `Some(false)` = upper, `None` = freely inside the window (or no limit).
/// Pure classifier: the impulse application lives in the drive iteration.
fn hinge_limit_state(angle: f32, limit: Option<RevoluteLimit>) -> Option<bool> {
    const ANGULAR_SLOP: f32 = 0.005; // ~0.3 deg of bound penetration
    match limit {
        Some(lim) if angle <= lim.min + ANGULAR_SLOP => Some(true),
        Some(lim) if angle >= lim.max - ANGULAR_SLOP => Some(false),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn joint_drive_velocity_iteration(
    bodies: &mut [RigidBody],
    joint: &mut Joint,
    a: usize,
    b: usize,
    axis_a: Vec3,
    limit: Option<RevoluteLimit>,
    motor: Option<RevoluteMotor>,
    sub_dt: f32,
) {
    let wa = (bodies[a].orientation * axis_a).normalize_or(Vec3::Z);
    let angle =
        hinge_twist(bodies[a].orientation, bodies[b].orientation, axis_a) - joint.reference_angle;
    let angle = (angle + PI).rem_euclid(TAU) - PI;
    let (ba, bb) = (&bodies[a], &bodies[b]);
    let k_eff = mul_inv_inertia(ba.inertia, ba.orientation, wa).dot(wa)
        + mul_inv_inertia(bb.inertia, bb.orientation, wa).dot(wa);
    if k_eff < 1e-9 {
        return;
    }
    let w = (bb.angular_velocity - ba.angular_velocity).dot(wa);
    match hinge_limit_state(angle, limit) {
        None => {
            joint.acc_limit = 0.0;
            if let Some(m) = motor
                && sub_dt > 0.0
                && m.max_torque > 0.0
            {
                let dl = ((m.target_speed - w) / k_eff)
                    .clamp(-m.max_torque * sub_dt, m.max_torque * sub_dt);
                apply_angular_impulse(bodies, a, b, wa * dl);
            }
        }
        // One-sided block: lower forbids w < 0 (accumulator >= 0), upper
        // forbids w > 0 (accumulator <= 0). The clamp self-corrects on side
        // flips by dumping the stale impulse in one step.
        Some(lower) => {
            let dl = -w / k_eff;
            let next = if lower {
                (joint.acc_limit + dl).max(0.0)
            } else {
                (joint.acc_limit + dl).min(0.0)
            };
            apply_angular_impulse(bodies, a, b, wa * (next - joint.acc_limit));
            joint.acc_limit = next;
        }
    }
}

/// One Baumgarte-style linear position step along `dir` for the anchor
/// separation error `e` (clamped to MAX correction).
fn joint_linear_position_step(
    bodies: &mut [RigidBody],
    a: usize,
    b: usize,
    ra: Vec3,
    rb: Vec3,
    raw_e: f32,
    dir: Vec3,
) {
    // Baumgarte-style, same β/cap policy as contacts.
    const BETA: f32 = 0.2;
    const MAX_LIN_CORRECTION: f32 = 0.25;
    let e = raw_e.clamp(-MAX_LIN_CORRECTION, MAX_LIN_CORRECTION);
    if e.abs() < 1e-6 {
        return;
    }
    let k_eff = effective_mass(bodies, a, b, dir, ra, rb);
    if k_eff < 1e-9 {
        return;
    }
    let lambda = -BETA * e / k_eff;
    apply_positional_impulse(bodies, a, b, dir * lambda, ra, rb);
}

/// Angular position correction for a revolute joint: align the two hinge
/// axes with an inertia-weighted split pseudo-rotation.
fn joint_angular_position_pass(
    bodies: &mut [RigidBody],
    a: usize,
    b: usize,
    axis_a: Vec3,
    axis_b: Vec3,
) {
    // Baumgarte-style, same β/cap policy as contacts.
    const BETA: f32 = 0.2;
    const MAX_ANG_CORRECTION: f32 = 0.5;
    let wa = (bodies[a].orientation * axis_a).normalize_or(Vec3::Z);
    let wb = (bodies[b].orientation * axis_b).normalize_or(Vec3::Z);
    // Small-angle misalignment. Rotation aligning wb with wa is δ = −(wa × wb)
    // (triple product: (wa×wb)×wb = wb·cosθ − wa, i.e. +e would PUSH wb away —
    // sign matters, a flipped sign turns the correction into an exponential
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
        // Inertia-weighted split: b rotates toward alignment, a rotates
        // against it (a static body has I⁻¹ = 0).
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

/// Gear-compatible coordinate of one joint, measured from its assembly
/// reference: hinge twist (revolute) or slide separation (prismatic).
/// `None` for every other kind — gears only coordinate these two.
pub(super) fn joint_coordinate(bodies: &[RigidBody], joint: &Joint) -> Option<f32> {
    let a = joint.body_a;
    let b = joint.body_b;
    match &joint.kind {
        JointKind::Revolute { local_axis_a, .. } => Some(
            hinge_twist(bodies[a].orientation, bodies[b].orientation, *local_axis_a)
                - joint.reference_angle,
        ),
        JointKind::Prismatic {
            local_anchor_a,
            local_anchor_b,
            local_axis_a,
            ..
        } => {
            let wa = (bodies[a].orientation * *local_axis_a).normalize_or(Vec3::Z);
            let ra = bodies[a].orientation * *local_anchor_a;
            let rb = bodies[b].orientation * *local_anchor_b;
            Some(
                ((bodies[b].position + rb) - (bodies[a].position + ra)).dot(wa)
                    - joint.reference_length,
            )
        }
        _ => None,
    }
}

/// Precomputed dynamics of one gear side: bodies, world axis, anchor levers,
/// coordinate, rate and effective mass. `angular` selects torque (revolute)
/// vs force (prismatic) application.
struct GearSideData {
    a: usize,
    b: usize,
    axis: Vec3,
    ra: Vec3,
    rb: Vec3,
    angular: bool,
    coord: f32,
    rate: f32,
    eff: f32,
}

/// Dynamics of a gear-compatible joint side; `None` for other kinds or
/// degenerate (zero-mass) axes.
fn gear_side_data(bodies: &[RigidBody], joint: &Joint) -> Option<GearSideData> {
    let (a, b) = (joint.body_a, joint.body_b);
    match &joint.kind {
        JointKind::Revolute { local_axis_a, .. } => {
            let wa = (bodies[a].orientation * *local_axis_a).normalize_or(Vec3::Z);
            let (ba, bb) = (&bodies[a], &bodies[b]);
            let eff = mul_inv_inertia(ba.inertia, ba.orientation, wa).dot(wa)
                + mul_inv_inertia(bb.inertia, bb.orientation, wa).dot(wa);
            if eff < 1e-9 {
                return None;
            }
            Some(GearSideData {
                a,
                b,
                axis: wa,
                ra: Vec3::ZERO,
                rb: Vec3::ZERO,
                angular: true,
                coord: hinge_twist(ba.orientation, bb.orientation, *local_axis_a)
                    - joint.reference_angle,
                rate: (bb.angular_velocity - ba.angular_velocity).dot(wa),
                eff,
            })
        }
        JointKind::Prismatic {
            local_anchor_a,
            local_anchor_b,
            local_axis_a,
            ..
        } => {
            let wa = (bodies[a].orientation * *local_axis_a).normalize_or(Vec3::Z);
            let ra = bodies[a].orientation * *local_anchor_a;
            let rb = bodies[b].orientation * *local_anchor_b;
            let eff = effective_mass(bodies, a, b, wa, ra, rb);
            if eff < 1e-9 {
                return None;
            }
            Some(GearSideData {
                a,
                b,
                axis: wa,
                ra,
                rb,
                angular: false,
                coord: ((bodies[b].position + rb) - (bodies[a].position + ra)).dot(wa)
                    - joint.reference_length,
                rate: (point_velocity(&bodies[b], rb) - point_velocity(&bodies[a], ra)).dot(wa),
                eff,
            })
        }
        _ => None,
    }
}

/// Apply a generalized gear impulse: torque about the hinge axis
/// (revolute) or force along the slide axis (prismatic).
fn gear_apply_delta(bodies: &mut [RigidBody], side: &GearSideData, delta: f32) {
    if side.angular {
        apply_angular_impulse(bodies, side.a, side.b, side.axis * delta);
    } else {
        apply_impulse(bodies, side.a, side.b, side.axis * delta, side.ra, side.rb);
    }
}

/// One angular lock velocity iteration about explicit world axes
/// (fixed/wheel/six-DOF): kills relative spin about each axis (equality, no
/// clamp). `acc` slots align with `axes` — the caller keeps the same order
/// at warm start.
fn joint_angular_lock_velocity_iteration(
    bodies: &mut [RigidBody],
    acc: &mut [f32],
    a: usize,
    b: usize,
    axes: &[Vec3],
) {
    for (k, t) in axes.iter().enumerate() {
        let (ba, bb) = (&bodies[a], &bodies[b]);
        let k_eff = mul_inv_inertia(ba.inertia, ba.orientation, *t).dot(*t)
            + mul_inv_inertia(bb.inertia, bb.orientation, *t).dot(*t);
        if k_eff < 1e-9 {
            continue;
        }
        let wrel = (bb.angular_velocity - ba.angular_velocity).dot(*t);
        let dl = -wrel / k_eff;
        if let Some(slot) = acc.get_mut(k) {
            *slot += dl;
        }
        apply_angular_impulse(bodies, a, b, t * dl);
    }
}

/// Warm start for an angular lock: re-apply the accumulated impulses about
/// the same axes, in the same order, as the iterations.
fn joint_angular_lock_warm_start(
    bodies: &mut [RigidBody],
    acc: &[f32],
    a: usize,
    b: usize,
    axes: &[Vec3],
) {
    for (k, t) in axes.iter().enumerate() {
        let l = acc.get(k).copied().unwrap_or(0.0);
        if l.abs() > 1e-12 {
            apply_angular_impulse(bodies, a, b, t * l);
        }
    }
}

/// Angular lock position pass: drives the relative orientation back to the
/// assembly `reference`. The error is the axis-angle of
/// `(qa^-1 * qb) * reference^-1` in the world frame; each lock axis
/// corrects its own component (Baumgarte-style, same β/cap as the hinge
/// pass). `axes` selects the locked directions (all three = fixed joint).
fn joint_angular_lock_position_pass(
    bodies: &mut [RigidBody],
    a: usize,
    b: usize,
    reference: Quat,
    axes: &[Vec3],
) {
    const BETA: f32 = 0.2;
    const MAX_ANG_CORRECTION: f32 = 0.5;
    let (qa, qb) = (bodies[a].orientation, bodies[b].orientation);
    let q_err = (qa.conjugate() * qb) * reference.conjugate();
    let angle = 2.0 * q_err.w.clamp(-1.0, 1.0).acos();
    if angle < 1e-6 {
        return;
    }
    // Vector part lives in A's frame; the error lever is needed in world.
    let e = (qa * q_err.xyz()).normalize_or(Vec3::X) * angle;
    for t in axes {
        let err = e.dot(*t).clamp(-MAX_ANG_CORRECTION, MAX_ANG_CORRECTION);
        if err.abs() < 1e-6 {
            continue;
        }
        let (ba, bb) = (&bodies[a], &bodies[b]);
        let k_eff = mul_inv_inertia(ba.inertia, ba.orientation, *t).dot(*t)
            + mul_inv_inertia(bb.inertia, bb.orientation, *t).dot(*t);
        if k_eff < 1e-9 {
            continue;
        }
        let lambda = -BETA * err / k_eff;
        let da = mul_inv_inertia(ba.inertia, ba.orientation, *t * -lambda);
        let db = mul_inv_inertia(bb.inertia, bb.orientation, *t * lambda);
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

/// Distance-rod velocity iteration: keeps the anchor separation rate at
/// zero along the anchor delta axis (rigid rod — Box2D `b2DistanceJoint`
/// with zero frequency/damping). Equality constraint, warm-started via
/// `acc_dist`.
fn joint_distance_velocity_iteration(
    bodies: &mut [RigidBody],
    joint: &mut Joint,
    a: usize,
    b: usize,
    ra: Vec3,
    rb: Vec3,
) {
    let delta = (bodies[b].position + rb) - (bodies[a].position + ra);
    let len = delta.length();
    if len < 1e-9 {
        return;
    }
    let n = delta / len;
    let k_eff = effective_mass(bodies, a, b, n, ra, rb);
    if k_eff < 1e-9 {
        return;
    }
    let vrel = (point_velocity(&bodies[b], rb) - point_velocity(&bodies[a], ra)).dot(n);
    let dl = -vrel / k_eff;
    joint.acc_dist += dl;
    apply_impulse(bodies, a, b, n * dl, ra, rb);
}

/// Wheel suspension spring (semi-implicit Euler in generalized
/// coordinates — a deliberate deviation from the Box2D gamma/bias
/// formulation, which goes unstable when the substep is small: its
/// accumulator eigenvalue is `1 - m·gamma` with
/// `m·gamma = 1/(2·zeta·omega·h + omega^2·h^2)`, so our 720 Hz substeps
/// diverge for any soft spring while Box2D's 60 Hz steps barely hold).
/// Inputs keep Box2D frequency/damping semantics
/// (`k = m·omega^2`, `c = 2·m·zeta·omega` on the reduced mass); the
/// denominator `1 + h·(k·h + c)·K` keeps the solve unconditionally stable
/// for any frequency at any substep count. Stateless (no accumulator —
/// nothing that can walk to infinity); runs once per substep with the
/// drives. Zero frequency locks the suspension rigidly (infinite
/// stiffness) instead of springing.
#[allow(clippy::too_many_arguments)]
fn joint_wheel_spring_iteration(
    bodies: &mut [RigidBody],
    joint: &mut Joint,
    a: usize,
    b: usize,
    ra: Vec3,
    rb: Vec3,
    suspension: WheelSuspension,
    sub_dt: f32,
) {
    let Some((axis_a, _, _, _)) = joint.wheel_drive() else {
        return;
    };
    let wa = (bodies[a].orientation * axis_a).normalize_or(Vec3::Z);
    let s =
        ((bodies[b].position + rb) - (bodies[a].position + ra)).dot(wa) - joint.reference_length;
    let k_eff = effective_mass(bodies, a, b, wa, ra, rb);
    if k_eff < 1e-9 {
        return;
    }
    let v = (point_velocity(&bodies[b], rb) - point_velocity(&bodies[a], ra)).dot(wa);
    if suspension.frequency_hz <= 0.0 || sub_dt <= 0.0 {
        // Rigid suspension: plain equality along the slide axis.
        let dl = -v / k_eff;
        joint.acc_limit += dl;
        apply_impulse(bodies, a, b, wa * dl, ra, rb);
        return;
    }
    // Implicit spring about the rest length: J settles velocity AND
    // position together, solved in closed form (denominator >= 1 always).
    let m = 1.0 / k_eff;
    let omega = TAU * suspension.frequency_hz;
    let stiff = m * omega * omega;
    let damp = 2.0 * m * suspension.damping_ratio * omega;
    let denom = 1.0 + sub_dt * (stiff * sub_dt + damp) * k_eff;
    let dl = sub_dt * (-stiff * s - (stiff * sub_dt + damp) * v) / denom;
    apply_impulse(bodies, a, b, wa * dl, ra, rb);
}

/// One-sided six-DOF linear limit iteration along `dir`: blocks separation
/// past `[min, max]` (meters from the assembly pose, `sep` signed along
/// `dir`) with the Box2D one-sided clamp discipline. Accumulates into
/// `acc[slot]` (slots 0..3 = X/Y/Z).
#[allow(clippy::too_many_arguments)]
fn joint_sixdof_linear_limit_iteration(
    bodies: &mut [RigidBody],
    acc: &mut [f32],
    slot: usize,
    a: usize,
    b: usize,
    ra: Vec3,
    rb: Vec3,
    dir: Vec3,
    sep: f32,
    min: f32,
    max: f32,
) {
    const LINEAR_SLOP: f32 = 0.002;
    let k_eff = effective_mass(bodies, a, b, dir, ra, rb);
    if k_eff < 1e-9 {
        return;
    }
    let v = (point_velocity(&bodies[b], rb) - point_velocity(&bodies[a], ra)).dot(dir);
    let violated = if sep <= min + LINEAR_SLOP {
        Some(true)
    } else if sep >= max - LINEAR_SLOP {
        Some(false)
    } else {
        None
    };
    // One-sided block: lower forbids v < 0 (accumulator >= 0), upper
    // forbids v > 0 (accumulator <= 0). Inside the window the accumulator
    // resets — same discipline as the hinge/slide drives.
    match violated {
        None => {
            if let Some(slot) = acc.get_mut(slot) {
                *slot = 0.0;
            }
        }
        Some(lower) => {
            let dl = -v / k_eff;
            let cur = acc.get(slot).copied().unwrap_or(0.0);
            let next = if lower {
                (cur + dl).max(0.0)
            } else {
                (cur + dl).min(0.0)
            };
            if let Some(slot) = acc.get_mut(slot) {
                *slot = next;
            }
            apply_impulse(bodies, a, b, dir * (next - cur), ra, rb);
        }
    }
}

/// One-sided six-DOF angular limit iteration about `dir` (world): blocks
/// twist past `[min, max]` (radians from the assembly twist `ref_twist`,
/// measured with [`hinge_twist`] about the matching local axis). Same
/// one-sided clamp discipline as the hinge drive; accumulates into
/// `acc[slot]` (slots 3..6 = X/Y/Z).
#[allow(clippy::too_many_arguments)]
fn joint_sixdof_angular_limit_iteration(
    bodies: &mut [RigidBody],
    acc: &mut [f32],
    slot: usize,
    a: usize,
    b: usize,
    dir: Vec3,
    angle: f32,
    ref_twist: f32,
    min: f32,
    max: f32,
) {
    const ANGULAR_SLOP: f32 = 0.005;
    let travel = angle - ref_twist;
    let (ba, bb) = (&bodies[a], &bodies[b]);
    let k_eff = mul_inv_inertia(ba.inertia, ba.orientation, dir).dot(dir)
        + mul_inv_inertia(bb.inertia, bb.orientation, dir).dot(dir);
    if k_eff < 1e-9 {
        return;
    }
    let w = (bb.angular_velocity - ba.angular_velocity).dot(dir);
    let violated = if travel <= min + ANGULAR_SLOP {
        Some(true)
    } else if travel >= max - ANGULAR_SLOP {
        Some(false)
    } else {
        None
    };
    match violated {
        None => {
            if let Some(slot) = acc.get_mut(slot) {
                *slot = 0.0;
            }
        }
        Some(lower) => {
            let dl = -w / k_eff;
            let cur = acc.get(slot).copied().unwrap_or(0.0);
            let next = if lower {
                (cur + dl).max(0.0)
            } else {
                (cur + dl).min(0.0)
            };
            if let Some(slot) = acc.get_mut(slot) {
                *slot = next;
            }
            apply_angular_impulse(bodies, a, b, dir * (next - cur));
        }
    }
}

/// Velocity stage for the fixed/distance/wheel/six-DOF joints (G5b).
/// Mirrors the legacy path discipline: warm start, equality iterations,
/// then the one-sided/spring/motor stage once per substep. Gears run in
/// [`BuiltinPhysicsEngine::solve_gears_velocity`] instead (separate loop).
#[allow(clippy::too_many_arguments)]
fn solve_new_joint_velocity(
    bodies: &mut [RigidBody],
    joint: &mut Joint,
    a: usize,
    b: usize,
    iterations: u32,
    sub_dt: f32,
) {
    const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];
    const FRAME: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];
    let (la, lb) = joint.local_anchors();
    match joint.kind {
        JointKind::Fixed { .. } => {
            let ra = bodies[a].orientation * la;
            let rb = bodies[b].orientation * lb;
            joint_warm_start(bodies, joint, a, b, ra, rb, None);
            joint_angular_lock_warm_start(bodies, &joint.acc_ang, a, b, &AXES);
            for _ in 0..iterations {
                joint_linear_velocity_iteration(bodies, joint, a, b, ra, rb);
                joint_angular_lock_velocity_iteration(bodies, &mut joint.acc_ang, a, b, &AXES);
            }
        }
        JointKind::Distance { .. } => {
            let ra = bodies[a].orientation * la;
            let rb = bodies[b].orientation * lb;
            // Warm start along the current rod axis.
            let delta = (bodies[b].position + rb) - (bodies[a].position + ra);
            if delta.length() >= 1e-9 && joint.acc_dist.abs() > 1e-12 {
                let n = delta / delta.length();
                apply_impulse(bodies, a, b, n * joint.acc_dist, ra, rb);
            }
            for _ in 0..iterations {
                joint_distance_velocity_iteration(bodies, joint, a, b, ra, rb);
            }
        }
        JointKind::Wheel { .. } => {
            let Some((susp_a, axle_a, suspension, motor)) = joint.wheel_drive() else {
                return;
            };
            let ra = bodies[a].orientation * la;
            let rb = bodies[b].orientation * lb;
            let qa = bodies[a].orientation;
            let ws = (qa * susp_a).normalize_or(Vec3::Y);
            let wx = (qa * axle_a).normalize_or(Vec3::Z);
            let third = ws.cross(wx).normalize_or(tangent_basis(ws));
            let lock = [ws, third];
            joint_warm_start(bodies, joint, a, b, ra, rb, None);
            joint_angular_lock_warm_start(bodies, &joint.acc_ang, a, b, &lock);
            // Warm-start the spring along the current suspension axis.
            if joint.acc_limit.abs() > 1e-12 {
                apply_impulse(bodies, a, b, ws * joint.acc_limit, ra, rb);
            }
            for _ in 0..iterations {
                let t1 = tangent_basis(ws);
                let t2 = ws.cross(t1).normalize_or_zero();
                for t in [t1, t2] {
                    joint_prismatic_linear_iteration(bodies, joint, a, b, ra, rb, t);
                }
                joint_angular_lock_velocity_iteration(bodies, &mut joint.acc_ang, a, b, &lock);
            }
            joint_wheel_spring_iteration(bodies, joint, a, b, ra, rb, suspension, sub_dt);
            if let Some(m) = motor {
                joint_drive_velocity_iteration(bodies, joint, a, b, axle_a, None, Some(m), sub_dt);
            }
        }
        JointKind::SixDof { .. } => {
            let Some((linear, angular)) = joint.sixdof_config() else {
                return;
            };
            let ra = bodies[a].orientation * la;
            let rb = bodies[b].orientation * lb;
            let qa = bodies[a].orientation;
            let qb = bodies[b].orientation;
            // Locked linear axes accumulate into the shared world totals
            // (frame-independent, like the prismatic perp basis).
            joint_warm_start(bodies, joint, a, b, ra, rb, None);
            // Locked angular axes, X/Y/Z order, slots aligned 1:1.
            let mut lock = [Vec3::ZERO; 3];
            let mut n_lock = 0;
            for (i, e) in FRAME.iter().enumerate() {
                if angular[i] == AxisConfig::Locked {
                    lock[n_lock] = (qa * *e).normalize_or(*e);
                    n_lock += 1;
                }
            }
            joint_angular_lock_warm_start(bodies, &joint.acc_ang, a, b, &lock[..n_lock]);
            for _ in 0..iterations {
                for (i, e) in FRAME.iter().enumerate() {
                    if linear[i] == AxisConfig::Locked {
                        let dir = (qa * *e).normalize_or(*e);
                        joint_prismatic_linear_iteration(bodies, joint, a, b, ra, rb, dir);
                    }
                }
                joint_angular_lock_velocity_iteration(
                    bodies,
                    &mut joint.acc_ang,
                    a,
                    b,
                    &lock[..n_lock],
                );
            }
            // One-sided limits, once per substep (velocity-only parity).
            let delta = (bodies[b].position + rb) - (bodies[a].position + ra);
            for (i, e) in FRAME.iter().enumerate() {
                let dir = (qa * *e).normalize_or(*e);
                if let AxisConfig::Limited { min, max } = linear[i] {
                    joint_sixdof_linear_limit_iteration(
                        bodies,
                        &mut joint.acc_6dof,
                        i,
                        a,
                        b,
                        ra,
                        rb,
                        dir,
                        delta.dot(dir),
                        min,
                        max,
                    );
                }
                if let AxisConfig::Limited { min, max } = angular[i] {
                    joint_sixdof_angular_limit_iteration(
                        bodies,
                        &mut joint.acc_6dof,
                        3 + i,
                        a,
                        b,
                        dir,
                        hinge_twist(qa, qb, *e),
                        quat_twist(joint.reference_quat, *e),
                        min,
                        max,
                    );
                }
            }
        }
        // Legacy kinds and gears never reach here (dispatched by the caller).
        JointKind::Ball { .. }
        | JointKind::Revolute { .. }
        | JointKind::Prismatic { .. }
        | JointKind::Gear { .. } => {}
    }
}

/// Position stage for the fixed/distance/wheel/six-DOF joints (split
/// impulse, positions only). Locked axes get Baumgarte steps; limited axes
/// and gears are velocity-only (same standing as the hinge/slide limits).
fn solve_new_joint_position(
    bodies: &mut [RigidBody],
    joint: &mut Joint,
    a: usize,
    b: usize,
    iterations: u32,
) {
    const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];
    const FRAME: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];
    let (la, lb) = joint.local_anchors();
    match joint.kind {
        JointKind::Fixed { .. } => {
            for _ in 0..iterations {
                let ra = bodies[a].orientation * la;
                let rb = bodies[b].orientation * lb;
                let c = (bodies[b].position + rb) - (bodies[a].position + ra);
                // Assembly-relative error: offset welds hold their offset.
                let e = c - bodies[a].orientation * joint.reference_anchor_delta;
                for dir in AXES {
                    joint_linear_position_step(bodies, a, b, ra, rb, e.dot(dir), dir);
                }
                joint_angular_lock_position_pass(bodies, a, b, joint.reference_quat, &AXES);
            }
        }
        JointKind::Distance { .. } => {
            for _ in 0..iterations {
                let ra = bodies[a].orientation * la;
                let rb = bodies[b].orientation * lb;
                let delta = (bodies[b].position + rb) - (bodies[a].position + ra);
                let len = delta.length();
                if len < 1e-9 {
                    continue;
                }
                joint_linear_position_step(
                    bodies,
                    a,
                    b,
                    ra,
                    rb,
                    len - joint.reference_distance,
                    delta / len,
                );
            }
        }
        JointKind::Wheel { .. } => {
            let Some((susp_a, axle_a, _, _)) = joint.wheel_drive() else {
                return;
            };
            for _ in 0..iterations {
                let ra = bodies[a].orientation * la;
                let rb = bodies[b].orientation * lb;
                let c = (bodies[b].position + rb) - (bodies[a].position + ra);
                let e = c - bodies[a].orientation * joint.reference_anchor_delta;
                let qa = bodies[a].orientation;
                let ws = (qa * susp_a).normalize_or(Vec3::Y);
                let wx = (qa * axle_a).normalize_or(Vec3::Z);
                let third = ws.cross(wx).normalize_or(tangent_basis(ws));
                let t1 = tangent_basis(ws);
                let t2 = ws.cross(t1).normalize_or_zero();
                for dir in [t1, t2] {
                    joint_linear_position_step(bodies, a, b, ra, rb, e.dot(dir), dir);
                }
                let lock = [ws, third];
                joint_angular_lock_position_pass(bodies, a, b, joint.reference_quat, &lock);
            }
        }
        JointKind::SixDof { .. } => {
            let Some((linear, angular)) = joint.sixdof_config() else {
                return;
            };
            // Locked angular subset, X/Y/Z order (same order as velocity).
            let mut lock = [Vec3::ZERO; 3];
            for _ in 0..iterations {
                let ra = bodies[a].orientation * la;
                let rb = bodies[b].orientation * lb;
                let c = (bodies[b].position + rb) - (bodies[a].position + ra);
                let e = c - bodies[a].orientation * joint.reference_anchor_delta;
                let qa = bodies[a].orientation;
                for (i, e_) in FRAME.iter().enumerate() {
                    if linear[i] == AxisConfig::Locked {
                        let dir = (qa * *e_).normalize_or(*e_);
                        joint_linear_position_step(bodies, a, b, ra, rb, e.dot(dir), dir);
                    }
                }
                // Locked world dirs from the current orientation (same
                // X/Y/Z order as the velocity stage).
                let mut n_lock = 0;
                for (i, e) in FRAME.iter().enumerate() {
                    if angular[i] == AxisConfig::Locked {
                        lock[n_lock] = (bodies[a].orientation * *e).normalize_or(*e);
                        n_lock += 1;
                    }
                }
                if n_lock > 0 {
                    joint_angular_lock_position_pass(
                        bodies,
                        a,
                        b,
                        joint.reference_quat,
                        &lock[..n_lock],
                    );
                }
            }
        }
        // Legacy kinds and gears never reach here (dispatched by the caller).
        JointKind::Ball { .. }
        | JointKind::Revolute { .. }
        | JointKind::Prismatic { .. }
        | JointKind::Gear { .. } => {}
    }
}
