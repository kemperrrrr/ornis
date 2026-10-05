//! R3 gate: the free 6-DOF motor joint (`JointKind::Motor`).
//!
//! - A motor drives a box to its target linear velocity against floor
//!   friction (all three rigid solvers, force budget respected).
//! - A motor spins a free body up to its angular target.
//! - A starved budget under-delivers (documented force-limited shortfall,
//!   not a snap): weak `max_force` never breaks static friction.
//! - A static body A is the world frame (the Box3D "motor vs ground"
//!   setup); `correction == 0` is a pure velocity drive with no pose
//!   memory, `correction > 0` weakly pulls the assembly pose back.
//! - Migration routes the motor losslessly onto every rigid solver;
//!   cross-solver motors refuse explicitly (no positional row to couple);
//!   the spec drive and the `set_joint_motor` refusal are untouched by
//!   the pose-servo path (servo regression below).

use glam::Vec3;
use ornis_physics::{
    BodyHandle, CrossJointStatus, Engine, JointHandle, JointKind, JointMotor, PhysicsEngine,
    RigidBody, RoutingKind, SolverKind, WorldSnapshot,
};

const DT: f32 = 1.0 / 60.0;
const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn motor(
    linear: Vec3,
    angular: Vec3,
    max_force: f32,
    max_torque: f32,
    correction: f32,
) -> JointKind {
    JointKind::motor_checked(linear, angular, max_force, max_torque, correction)
        .expect("valid motor spec")
}

/// Static anchor far above the scene (world-frame drive: A never moves,
/// so B-minus-A targets read in the world frame).
fn anchor(e: &mut Engine, pos: Vec3) -> BodyHandle {
    e.add_body(RigidBody::new_sphere(pos, 0.1, 0.0))
}

fn floor(e: &mut Engine) -> BodyHandle {
    e.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ))
}

/// Box resting on the floor top (`y == 0`), ready to be driven along +X.
fn rest_box(e: &mut Engine) -> BodyHandle {
    e.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        1.0,
    ))
}

fn speed(e: &Engine, h: BodyHandle) -> Vec3 {
    e.get_body(h).expect("body live").velocity
}

#[test]
fn si_motor_drives_box_to_target_against_friction() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    floor(&mut e);
    let block = rest_box(&mut e);
    let anchor = anchor(&mut e, Vec3::new(0.0, 5.0, 0.0));
    e.add_joint(
        anchor,
        block,
        motor(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, 8.0, 10.0, 0.0),
    )
    .expect("valid motor joint");
    for _ in 0..300 {
        e.step(DT);
    }
    let v = speed(&e, block);
    assert!(
        (v.x - 2.0).abs() < 0.3,
        "motor must hold 2 m/s against friction, got {v}"
    );
    assert!(
        v.y.abs() < 0.3 && v.z.abs() < 0.3,
        "off-axes must stay quiet, got {v}"
    );
}

#[test]
fn avbd_motor_drives_box_to_target_against_friction() {
    let mut e = Engine::new(SolverKind::Avbd, GRAVITY);
    floor(&mut e);
    let block = rest_box(&mut e);
    let anchor = anchor(&mut e, Vec3::new(0.0, 5.0, 0.0));
    e.add_joint(
        anchor,
        block,
        motor(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, 8.0, 10.0, 0.0),
    )
    .expect("valid motor joint");
    for _ in 0..300 {
        e.step(DT);
    }
    let v = speed(&e, block);
    assert!(
        (v.x - 2.0).abs() < 0.4,
        "avbd motor must hold 2 m/s against friction, got {v}"
    );
}

#[test]
fn xpbd_motor_drives_box_to_target_against_friction() {
    let mut e = Engine::new(SolverKind::Xpbd, GRAVITY);
    floor(&mut e);
    let block = rest_box(&mut e);
    let anchor = anchor(&mut e, Vec3::new(0.0, 5.0, 0.0));
    e.add_joint(
        anchor,
        block,
        motor(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, 8.0, 10.0, 0.0),
    )
    .expect("valid motor joint");
    for _ in 0..300 {
        e.step(DT);
    }
    let v = speed(&e, block);
    assert!(
        (v.x - 2.0).abs() < 0.4,
        "xpbd motor must hold 2 m/s against friction, got {v}"
    );
}

#[test]
fn si_motor_spins_free_body_to_angular_target() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let anchor = anchor(&mut e, Vec3::ZERO);
    let ball = e.add_body(RigidBody::new_sphere(Vec3::new(0.0, 2.0, 0.0), 0.3, 1.0));
    e.add_joint(
        anchor,
        ball,
        motor(Vec3::ZERO, Vec3::new(0.0, 0.0, 3.0), 10.0, 5.0, 0.0),
    )
    .expect("valid motor joint");
    for _ in 0..240 {
        e.step(DT);
    }
    let b = e.get_body(ball).expect("ball live");
    assert!(
        (b.angular_velocity.z - 3.0).abs() < 0.3,
        "motor must spin to 3 rad/s, got {}",
        b.angular_velocity
    );
    assert!(
        b.velocity.length() < 0.3,
        "linear drive is zero: the body must spin in place, got {}",
        b.velocity
    );
}

#[test]
fn si_motor_starved_budget_under_delivers_documented() {
    // Force-limited shortfall by design: 1 N cannot break the ~4.9 N
    // static-friction grip of a 1 kg box (mu = 0.5), so the box never
    // reaches the 2 m/s target. The budget clamps the accumulated
    // impulse (limit-row discipline) — it never snaps the velocity.
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    floor(&mut e);
    let block = rest_box(&mut e);
    let anchor = anchor(&mut e, Vec3::new(0.0, 5.0, 0.0));
    e.add_joint(
        anchor,
        block,
        motor(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, 1.0, 10.0, 0.0),
    )
    .expect("valid motor joint");
    for _ in 0..300 {
        e.step(DT);
    }
    let v = speed(&e, block);
    assert!(
        v.x < 1.0,
        "a 1 N motor must fall well short of 2 m/s against friction, got {v}"
    );
}

#[test]
fn si_motor_static_anchor_is_the_world_frame() {
    // A-null analogue: A is static, so the B-minus-A target reads in the
    // world frame — A never moves while B converges to the target.
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let a = anchor(&mut e, Vec3::ZERO);
    let b = e.add_body(RigidBody::new_sphere(Vec3::new(0.0, 2.0, 0.0), 0.3, 1.0));
    e.add_joint(
        a,
        b,
        motor(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, 20.0, 5.0, 0.0),
    )
    .expect("valid motor joint");
    for _ in 0..240 {
        e.step(DT);
    }
    assert_eq!(
        e.get_body(a).expect("anchor live").position,
        Vec3::ZERO,
        "static anchor must not move"
    );
    let v = speed(&e, b);
    assert!(
        (v.x - 1.0).abs() < 0.2,
        "body must reach the world-frame target, got {v}"
    );
}

#[test]
fn si_motor_correction_pulls_assembly_pose_weakly() {
    // Zero targets + default correction: a displaced body drifts back
    // toward the assembly pose (drift control, not a servo — the pull is
    // one weak share per substep and carries no stiffness).
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let a = anchor(&mut e, Vec3::ZERO);
    let b = e.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.3, 1.0));
    e.add_joint(
        a,
        b,
        motor(
            Vec3::ZERO,
            Vec3::ZERO,
            50.0,
            0.0,
            ornis_physics::DEFAULT_MOTOR_CORRECTION,
        ),
    )
    .expect("valid motor joint");
    // Teleport B half a meter off the assembly coincidence.
    e.get_body_mut(b).expect("ball live").position = Vec3::new(0.5, 0.0, 0.0);
    for _ in 0..240 {
        e.step(DT);
    }
    let d = e.get_body(b).expect("ball live").position.length();
    assert!(
        d < 0.25,
        "correction must pull the body back toward assembly, drift {d}"
    );
}

#[test]
fn motor_migrates_losslessly_across_rigid_solvers() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    floor(&mut e);
    let block = rest_box(&mut e);
    let anchor = anchor(&mut e, Vec3::new(0.0, 5.0, 0.0));
    let spec = motor(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, 8.0, 10.0, 0.0);
    e.add_joint(anchor, block, spec).expect("valid motor");
    for _ in 0..60 {
        e.step(DT);
    }
    // SI -> AVBD keeps the joint and keeps driving.
    e.set_solver_kind(SolverKind::Avbd, GRAVITY);
    assert_eq!(e.joint_count(), 1, "motor must survive SI -> AVBD");
    for _ in 0..240 {
        e.step(DT);
    }
    let v = speed(&e, block);
    assert!(
        (v.x - 2.0).abs() < 0.5,
        "migrated motor must keep driving, got {v}"
    );
    // AVBD -> XPBD keeps the joint too (velocity drive, no position rows).
    e.set_solver_kind(SolverKind::Xpbd, GRAVITY);
    assert_eq!(e.joint_count(), 1, "motor must survive AVBD -> XPBD");
    for _ in 0..120 {
        e.step(DT);
    }
    assert!(
        e.get_body(block).expect("box live").velocity.x.is_finite(),
        "xpbd motor must step without NaN"
    );
}

#[test]
fn cross_motor_refuses_explicitly_with_cause() {
    // Cross-solver motors never half-solve: no positional row exists to
    // couple (a velocity drive cannot meet halfway), so the status names
    // the cause instead of silently dropping the joint.
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let heavy = |pos| RigidBody::new_box(pos, Vec3::splat(0.5), 100.0);
    let a = e.add_body(heavy(Vec3::ZERO));
    let b = e.add_body(heavy(Vec3::new(3.0, 0.0, 0.0)));
    let m: JointHandle = e
        .add_joint(
            a,
            b,
            motor(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, 50.0, 10.0, 0.0),
        )
        .expect("valid motor");
    e.set_routing(RoutingKind::Islands);
    e.pin_body_solver(a, Some(SolverKind::SequentialImpulse));
    e.pin_body_solver(b, Some(SolverKind::Avbd));
    for _ in 0..60 {
        e.step(DT);
    }
    match e.cross_joint_status(m) {
        Some(CrossJointStatus::Unsupported { detail }) => assert!(!detail.is_empty()),
        other => panic!("cross motor must refuse explicitly, got {other:?}"),
    }
}

#[test]
fn motor_spec_rejects_servo_override_and_pose_servo_regresses_clean() {
    // The free drive rides inline in the spec: a `set_joint_motor`
    // override has no driven axis to attach to and refuses explicitly.
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let a = anchor(&mut e, Vec3::ZERO);
    let b = e.add_body(RigidBody::new_sphere(Vec3::new(0.0, 2.0, 0.0), 0.3, 1.0));
    let m = e
        .add_joint(a, b, motor(Vec3::X, Vec3::ZERO, 20.0, 5.0, 0.0))
        .expect("valid motor");
    let servo = JointMotor::position(1.0, 60.0, 12.0, 50.0).expect("valid servo");
    assert!(e.set_joint_motor(m, Some(servo)).is_err());
    assert_eq!(e.joint_motor(m), None);

    // Pose servos are untouched: a hinge servo still converges to its
    // fixed setpoint (the motor tracks a VELOCITY, the servo a POSE).
    let hinge_a = e.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    let arm = e.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.1, 1.0, 0.1),
        1.0,
    ));
    let hinge = e
        .add_joint(
            hinge_a,
            arm,
            JointKind::Revolute {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: None,
                motor: None,
            },
        )
        .expect("valid hinge");
    e.set_joint_motor(hinge, Some(servo))
        .expect("servo attaches");
    for _ in 0..240 {
        e.step(DT);
    }
    let q = e.get_body(arm).expect("arm live").orientation;
    let angle = 2.0 * q.z.atan2(q.w);
    assert!(
        (angle - 1.0).abs() < 0.1,
        "hinge servo must still converge to its pose target, got {angle}"
    );
}

#[test]
fn motor_snapshot_round_trip_continues_bit_identical() {
    // The motor spec plus its warm accumulators (`acc_lin`/`acc_ang`)
    // ride the determinism snapshot: capture mid-drive, restore through
    // RON, continue — bit-identical bodies and snapshots.
    let mut original = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    floor(&mut original);
    let block = rest_box(&mut original);
    let anchor = anchor(&mut original, Vec3::new(0.0, 5.0, 0.0));
    original
        .add_joint(
            anchor,
            block,
            motor(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, 8.0, 10.0, 0.0),
        )
        .expect("valid motor");
    for _ in 0..90 {
        original.step(DT);
    }
    // Drain queues before capture (the `world_snapshot` pattern):
    // undrained transitions restore verbatim AND re-emit on continuation.
    let _ = original.drain_contact_events();
    let _ = original.drain_contact_force_events();
    let _ = original.drain_trigger_events();
    let snap = original.world_snapshot().expect("captures");
    let text = snap.to_ron().expect("serializes");
    let back = WorldSnapshot::from_ron(&text).expect("parses");
    let mut restored = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    restored.restore_world_snapshot(&back).expect("restores");
    for _ in 0..60 {
        original.step(DT);
        restored.step(DT);
    }
    fn bits(e: &Engine, h: BodyHandle) -> Vec<u32> {
        let b = e.get_body(h).expect("body live");
        let mut out = Vec::new();
        out.extend(b.position.to_array().map(f32::to_bits));
        out.extend(b.velocity.to_array().map(f32::to_bits));
        out.extend(b.angular_velocity.to_array().map(f32::to_bits));
        out
    }
    assert_eq!(
        bits(&original, block),
        bits(&restored, block),
        "motor drive must continue bit-identically after restore"
    );
    assert_eq!(
        original.world_snapshot().expect("captures"),
        restored.world_snapshot().expect("captures"),
        "continued snapshots must compare equal"
    );
}
