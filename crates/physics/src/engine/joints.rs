//! Joint sub-solver for `BuiltinPhysicsEngine` (G5): velocity and position
//! stages of the ball/revolute joint constraints. Split out of `engine.rs`
//! to keep each type's method count within the structural gate's thresholds.

use std::f32::consts::{PI, TAU};

use glam::Quat;
use glam::Vec3;

use super::*;
use crate::joint::{RevoluteLimit, RevoluteMotor};

/// Twist of B relative to A about A's hinge axis (rad, wrapped to
/// [-PI, PI]). Decomposes `qa^-1 * qb` into twist about the axis plus
/// swing; limits, motors and their tests measure travel with this.
pub(super) fn hinge_twist(qa: Quat, qb: Quat, axis_a: Vec3) -> f32 {
    let q = qa.conjugate() * qb;
    let t = 2.0 * q.xyz().dot(axis_a).atan2(q.w);
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
            // A fully sleeping jointed pair is frozen; island-coherent sleep
            // guarantees both members share the sleep state.
            if asleep[a] && asleep[b] {
                continue;
            }
            let (la, lb) = joint.local_anchors();
            let revolute_axes = revolute_axes(&joint.kind);

            // --- Warm start: re-apply the accumulated impulses (G2b pattern).
            // Anchors are computed ONCE here and reused verbatim by the
            // iterations below (original behaviour).
            let ra = bodies[a].orientation * la;
            let rb = bodies[b].orientation * lb;
            joint_warm_start(bodies, joint, a, b, ra, rb, revolute_axes);

            // --- Velocity iterations.
            for _ in 0..*velocity_iterations {
                joint_linear_velocity_iteration(bodies, joint, a, b, ra, rb);
                if let Some((axis_a, _)) = revolute_axes {
                    joint_angular_velocity_iteration(bodies, joint, a, b, axis_a);
                }
            }
            // --- Limit/motor drive on the free hinge axis (revolute only).
            // Motor first, then the limit wins past the bounds (Box2D order);
            // the motor pauses while a limit is violated.
            if let Some((axis_a, _)) = revolute_axes {
                let (limit, motor) = joint.drive();
                if limit.is_some() || motor.is_some() {
                    joint_drive_velocity_iteration(
                        bodies, joint, a, b, axis_a, limit, motor, sub_dt,
                    );
                }
            }
        }
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
            if asleep[a] && asleep[b] {
                continue;
            }
            let (la, lb) = joint.local_anchors();
            let revolute_axes = revolute_axes(&joint.kind);

            for _ in 0..*position_iterations {
                let ra = bodies[a].orientation * la;
                let rb = bodies[b].orientation * lb;
                let c = (bodies[b].position + rb) - (bodies[a].position + ra);
                for dir in AXES {
                    joint_linear_position_step(bodies, a, b, ra, rb, c.dot(dir), dir);
                }
                if let Some((axis_a, axis_b)) = revolute_axes {
                    joint_angular_position_pass(bodies, a, b, axis_a, axis_b);
                }
            }
        }
    }
}

/// Hinge axes of a revolute joint (None for a ball joint).
fn revolute_axes(kind: &JointKind) -> Option<(Vec3, Vec3)> {
    match kind {
        JointKind::Revolute {
            local_axis_a,
            local_axis_b,
            ..
        } => Some((*local_axis_a, *local_axis_b)),
        JointKind::Ball { .. } => None,
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
