//! P6 gate: rope/spring joints plus the generalized motor model.
//!
//! - Rope holds a hanging load at its maximum (slack never pushes —
//!   a rod would hold the assembly length instead of falling).
//! - Spring oscillators settle at the rest length with damping
//!   (implicit and explicit integration, all three rigid solvers).
//! - A velocity motor spins a revolute joint to its target speed under
//!   gravity load; a position servo converges to its target angle.
//! - Migration routes or drops the new kinds explicitly (XPBD keeps
//!   rope/spring, drops wheel; cross rope couples, cross spring refuses
//!   with a reason; servo overrides survive solver switches).

use glam::Vec3;
use ornis_physics::{
    CrossJointStatus, CrossRowKind, Engine, JointHandle, JointKind, JointMotor, MotorModel,
    PhysicsEngine, RigidBody, RoutingKind, SolverKind, SpringIntegration,
};

const DT: f32 = 1.0 / 60.0;
const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn static_ball(pos: Vec3) -> RigidBody {
    RigidBody::new_sphere(pos, 0.1, 0.0)
}

fn load_ball(pos: Vec3, mass: f32) -> RigidBody {
    RigidBody::new_sphere(pos, 0.2, mass)
}

/// World separation of two joint anchors.
fn anchor_separation(
    e: &Engine,
    a: ornis_physics::BodyHandle,
    la: Vec3,
    b: ornis_physics::BodyHandle,
    lb: Vec3,
) -> f32 {
    let (ba, bb) = (e.get_body(a).unwrap(), e.get_body(b).unwrap());
    ((bb.position + bb.orientation * lb) - (ba.position + ba.orientation * la)).length()
}

/// Rope scene: coincident anchors at the origin (fully slack assembly),
/// maximum 2 m. A rod would hold length 0; a rope lets the load free-fall
/// and catches it at 2 m.
fn rope_scene(e: &mut Engine) -> (ornis_physics::BodyHandle, ornis_physics::BodyHandle) {
    let anchor = e.add_body(static_ball(Vec3::ZERO));
    let bob = e.add_body(load_ball(Vec3::new(0.0, -1.0, 0.0), 2.0));
    e.add_joint(
        anchor,
        bob,
        JointKind::Rope {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
            max_distance: 2.0,
        },
    )
    .expect("valid rope joint");
    (anchor, bob)
}

fn rope_length(
    e: &Engine,
    anchor: ornis_physics::BodyHandle,
    bob: ornis_physics::BodyHandle,
) -> f32 {
    anchor_separation(e, anchor, Vec3::ZERO, bob, Vec3::new(0.0, 1.0, 0.0))
}

#[test]
fn si_rope_holds_hanging_load_slack_is_free() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let (anchor, bob) = rope_scene(&mut e);
    // Free fall first: after 20 steps the load dropped ~0.54 m (a rod
    // would still sit at length 0 — slack pushes nothing).
    for _ in 0..20 {
        e.step(DT);
    }
    let early = rope_length(&e, anchor, bob);
    assert!(
        early > 0.3 && early < 1.9,
        "rope must let the load fall freely first, got {early}"
    );
    for _ in 0..220 {
        e.step(DT);
    }
    let held = rope_length(&e, anchor, bob);
    assert!(
        (held - 2.0).abs() < 0.15,
        "rope must catch the load at its maximum, got {held}"
    );
}

#[test]
fn avbd_rope_holds_hanging_load() {
    let mut e = Engine::new(SolverKind::Avbd, GRAVITY);
    let (anchor, bob) = rope_scene(&mut e);
    for _ in 0..240 {
        e.step(DT);
    }
    let held = rope_length(&e, anchor, bob);
    assert!(
        (held - 2.0).abs() < 0.2,
        "avbd rope must catch the load at its maximum, got {held}"
    );
}

#[test]
fn xpbd_rope_holds_hanging_load() {
    let mut e = Engine::new(SolverKind::Xpbd, GRAVITY);
    let (anchor, bob) = rope_scene(&mut e);
    for _ in 0..300 {
        e.step(DT);
    }
    let held = rope_length(&e, anchor, bob);
    assert!(
        (held - 2.0).abs() < 0.2,
        "xpbd rope must catch the load at its maximum, got {held}"
    );
}

/// Spring scene (zero gravity, pure oscillator): assembly stretched to
/// twice the rest length, released. Returns the bodies plus the anchor
/// frame used to measure the length.
fn spring_scene(
    e: &mut Engine,
    stiffness: f32,
    damping: f32,
    integration: SpringIntegration,
) -> (ornis_physics::BodyHandle, ornis_physics::BodyHandle) {
    let anchor = e.add_body(static_ball(Vec3::ZERO));
    let bob = e.add_body(load_ball(Vec3::new(0.0, -2.5, 0.0), 1.0));
    let motor = JointMotor::position(1.0, stiffness, damping, 1e6)
        .expect("valid spring motor")
        .with_model(MotorModel::ForceBased);
    e.add_joint(
        anchor,
        bob,
        JointKind::Spring {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::new(0.0, 0.5, 0.0),
            motor,
            integration,
        },
    )
    .expect("valid spring joint");
    (anchor, bob)
}

fn spring_length(
    e: &Engine,
    anchor: ornis_physics::BodyHandle,
    bob: ornis_physics::BodyHandle,
) -> f32 {
    anchor_separation(e, anchor, Vec3::ZERO, bob, Vec3::new(0.0, 0.5, 0.0))
}

fn assert_settled(
    e: &Engine,
    anchor: ornis_physics::BodyHandle,
    bob: ornis_physics::BodyHandle,
    tol: f32,
) {
    let len = spring_length(e, anchor, bob);
    assert!(
        (len - 1.0).abs() < tol,
        "spring must settle at its rest length, got {len}"
    );
    let speed = e.get_body(bob).unwrap().velocity.length();
    assert!(speed < 0.4, "settled spring must calm, speed {speed}");
}

#[test]
fn si_implicit_spring_oscillator_settles_at_rest_length() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let (anchor, bob) = spring_scene(&mut e, 30.0, 6.0, SpringIntegration::Implicit);
    for _ in 0..360 {
        e.step(DT);
    }
    assert_settled(&e, anchor, bob, 0.05);
}

#[test]
fn si_explicit_spring_oscillator_settles_at_rest_length() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let (anchor, bob) = spring_scene(&mut e, 30.0, 6.0, SpringIntegration::Explicit);
    for _ in 0..600 {
        e.step(DT);
    }
    assert_settled(&e, anchor, bob, 0.1);
}

#[test]
fn avbd_spring_oscillator_settles_at_rest_length() {
    let mut e = Engine::new(SolverKind::Avbd, Vec3::ZERO);
    let motor = JointMotor::position(1.0, 30.0, 6.0, 1e6)
        .expect("valid spring motor")
        .with_model(MotorModel::AccelerationBased);
    let anchor = e.add_body(static_ball(Vec3::ZERO));
    let bob = e.add_body(load_ball(Vec3::new(0.0, -2.5, 0.0), 1.0));
    e.add_joint(
        anchor,
        bob,
        JointKind::Spring {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::new(0.0, 0.5, 0.0),
            motor,
            integration: SpringIntegration::Implicit,
        },
    )
    .expect("valid spring joint");
    for _ in 0..600 {
        e.step(DT);
    }
    assert_settled(&e, anchor, bob, 0.1);
}

#[test]
fn xpbd_spring_oscillator_settles_at_rest_length() {
    let mut e = Engine::new(SolverKind::Xpbd, Vec3::ZERO);
    let (anchor, bob) = spring_scene(&mut e, 30.0, 6.0, SpringIntegration::Implicit);
    for _ in 0..600 {
        e.step(DT);
    }
    assert_settled(&e, anchor, bob, 0.1);
}

/// Hinge scene: static anchor at the origin, 1 kg arm hanging 1 m below,
/// hinge about Z. Under gravity the motor works against the pendulum load.
fn hinge_scene(
    e: &mut Engine,
    motor: Option<ornis_physics::RevoluteMotor>,
) -> (
    ornis_physics::BodyHandle,
    ornis_physics::BodyHandle,
    JointHandle,
) {
    let anchor = e.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    let arm = e.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.1, 1.0, 0.1),
        1.0,
    ));
    let j = e
        .add_joint(
            anchor,
            arm,
            JointKind::Revolute {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: None,
                motor,
            },
        )
        .expect("valid hinge");
    (anchor, arm, j)
}

fn hinge_speed(
    e: &Engine,
    anchor: ornis_physics::BodyHandle,
    arm: ornis_physics::BodyHandle,
) -> f32 {
    let (a, b) = (e.get_body(anchor).unwrap(), e.get_body(arm).unwrap());
    (b.angular_velocity - a.angular_velocity).dot(Vec3::Z)
}

#[test]
fn si_revolute_motor_reaches_target_speed_under_load() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let (anchor, arm, _) = hinge_scene(
        &mut e,
        Some(ornis_physics::RevoluteMotor {
            target_speed: 3.0,
            max_torque: 50.0,
        }),
    );
    for _ in 0..240 {
        e.step(DT);
    }
    // Gravity ripples the tumbling arm, so average over the tail.
    let mut mean = 0.0f32;
    for _ in 0..60 {
        e.step(DT);
        mean += hinge_speed(&e, anchor, arm);
    }
    mean /= 60.0;
    assert!(
        (mean - 3.0).abs() < 0.5,
        "motor must hold target speed under load, got {mean}"
    );
}

/// Hinge twist about Z from the assembly pose (identity assembly here).
fn hinge_angle(e: &Engine, arm: ornis_physics::BodyHandle) -> f32 {
    let q = e.get_body(arm).unwrap().orientation;
    2.0 * q.z.atan2(q.w)
}

#[test]
fn si_revolute_servo_converges_to_target_angle() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let (_, arm, j) = hinge_scene(&mut e, None);
    let servo = JointMotor::position(1.0, 60.0, 12.0, 50.0).expect("valid servo");
    e.set_joint_motor(j, Some(servo)).expect("servo attaches");
    assert_eq!(e.joint_motor(j), Some(servo));
    for _ in 0..240 {
        e.step(DT);
    }
    let angle = hinge_angle(&e, arm);
    assert!(
        (angle - 1.0).abs() < 0.1,
        "servo must converge to its target angle, got {angle}"
    );
    let speed = e.get_body(arm).unwrap().angular_velocity.length();
    assert!(speed < 0.4, "settled servo must calm, speed {speed}");
    // Clearing the override resumes the (unpowered) spec motor.
    e.set_joint_motor(j, None).expect("servo clears");
    assert_eq!(e.joint_motor(j), None);
}

#[test]
fn avbd_revolute_servo_converges_to_target_angle() {
    let mut e = Engine::new(SolverKind::Avbd, Vec3::ZERO);
    let (_, arm, j) = hinge_scene(&mut e, None);
    let servo = JointMotor::position(1.0, 60.0, 12.0, 50.0).expect("valid servo");
    e.set_joint_motor(j, Some(servo)).expect("servo attaches");
    for _ in 0..360 {
        e.step(DT);
    }
    let angle = hinge_angle(&e, arm);
    assert!(
        (angle - 1.0).abs() < 0.15,
        "avbd servo must converge to its target angle, got {angle}"
    );
}

#[test]
fn si_wheel_spin_servo_converges_about_axle() {
    use ornis_physics::sequential_impulse::joints::hinge_twist;

    // Wheel spin is the revolute special case: a position servo about the
    // axle converges to its target twist (assembly-relative, like hinges).
    // The anchors sit ON the spin axis (a free spin about an axis through
    // the center must not orbit the pinned anchors — an off-axis anchor
    // is honestly held still by the slide rows, and the spin dies there).
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let chassis = e.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.3), 0.0));
    let wheel = e.add_body(load_ball(Vec3::new(0.0, -0.6, 0.0), 1.0));
    let j = e
        .add_joint(
            chassis,
            wheel,
            JointKind::Wheel {
                local_anchor_a: Vec3::new(0.0, -0.6, 0.0),
                local_anchor_b: Vec3::ZERO,
                local_suspension_a: Vec3::Y,
                local_suspension_b: Vec3::Y,
                local_axle_a: Vec3::Z,
                local_axle_b: Vec3::Z,
                suspension: ornis_physics::WheelSuspension {
                    frequency_hz: 3.0,
                    damping_ratio: 0.7,
                },
                motor: None,
            },
        )
        .expect("valid wheel");
    let servo = JointMotor::position(0.5, 40.0, 8.0, 30.0).expect("valid servo");
    e.set_joint_motor(j, Some(servo)).expect("servo attaches");
    for _ in 0..300 {
        e.step(DT);
    }
    let (qa, qb) = (
        e.get_body(chassis).unwrap().orientation,
        e.get_body(wheel).unwrap().orientation,
    );
    let twist = hinge_twist(qa, qb, Vec3::Z);
    assert!(
        (twist - 0.5).abs() < 0.15,
        "wheel servo must converge about its axle, got {twist}"
    );
}

#[test]
fn servo_refuses_joints_without_a_driven_axis() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let anchor = e.add_body(static_ball(Vec3::ZERO));
    let bob = e.add_body(load_ball(Vec3::new(0.0, -1.0, 0.0), 1.0));
    let ball = e
        .add_joint(
            anchor,
            bob,
            JointKind::Ball {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
            },
        )
        .expect("valid ball joint");
    let servo = JointMotor::position(1.0, 60.0, 12.0, 50.0).expect("valid servo");
    assert!(e.set_joint_motor(ball, Some(servo)).is_err());
    assert!(e.joint_motor(ball).is_none());
}

#[test]
fn migration_routes_or_drops_new_kinds_explicitly() {
    // SI world with rope + spring + wheel: XPBD keeps the first two and
    // drops the wheel (no suspension rows there) — never silently.
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let anchor = e.add_body(static_ball(Vec3::ZERO));
    let bob = e.add_body(load_ball(Vec3::new(0.0, -1.0, 0.0), 1.0));
    e.add_joint(
        anchor,
        bob,
        JointKind::Rope {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            max_distance: 2.0,
        },
    )
    .expect("valid rope");
    let motor = JointMotor::position(1.0, 30.0, 6.0, 1e6).expect("valid motor");
    e.add_joint(
        anchor,
        bob,
        JointKind::Spring {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            motor,
            integration: SpringIntegration::Implicit,
        },
    )
    .expect("valid spring");
    e.add_joint(
        anchor,
        bob,
        JointKind::Wheel {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            local_suspension_a: Vec3::Y,
            local_suspension_b: Vec3::Y,
            local_axle_a: Vec3::Z,
            local_axle_b: Vec3::Z,
            suspension: ornis_physics::WheelSuspension {
                frequency_hz: 2.0,
                damping_ratio: 0.5,
            },
            motor: None,
        },
    )
    .expect("valid wheel");
    assert_eq!(e.joint_count(), 3);
    e.set_solver_kind(SolverKind::Xpbd, GRAVITY);
    assert_eq!(
        e.joint_count(),
        2,
        "xpbd migration must drop exactly the wheel joint"
    );

    // Cross-solver routing: a cross rope couples (one-sided), a cross
    // spring refuses explicitly (no compliant cross row in v1). Cross
    // ends must be dynamic (statics mirror into every solver natively),
    // so the anchors are heavy dynamics like the cross-ball precedent.
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let heavy = |pos| RigidBody::new_box(pos, Vec3::splat(0.5), 100.0);
    let a = e.add_body(heavy(Vec3::new(0.0, 1.0, 0.0)));
    let b = e.add_body(load_ball(Vec3::new(0.0, -2.0, 0.0), 1.0));
    let rope = e
        .add_joint(
            a,
            b,
            JointKind::Rope {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::new(0.0, 2.0, 0.0),
                max_distance: 2.0,
            },
        )
        .expect("valid rope");
    let c = e.add_body(load_ball(Vec3::new(2.0, -2.0, 0.0), 1.0));
    let spring = e
        .add_joint(
            a,
            c,
            JointKind::Spring {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::new(-2.0, 2.0, 0.0),
                motor,
                integration: SpringIntegration::Implicit,
            },
        )
        .expect("valid spring");
    e.set_routing(RoutingKind::Islands);
    e.pin_body_solver(a, Some(SolverKind::SequentialImpulse));
    e.pin_body_solver(b, Some(SolverKind::Avbd));
    e.pin_body_solver(c, Some(SolverKind::Avbd));
    for _ in 0..60 {
        e.step(DT);
    }
    assert_eq!(
        e.cross_joint_status(rope),
        Some(CrossJointStatus::Coupled(CrossRowKind::Rope))
    );
    match e.cross_joint_status(spring) {
        Some(CrossJointStatus::Unsupported { detail }) => assert!(!detail.is_empty()),
        other => panic!("cross spring must refuse explicitly, got {other:?}"),
    }
}

#[test]
fn servo_override_survives_solver_migration() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let (_, _, j) = hinge_scene(&mut e, None);
    let servo = JointMotor::servo(0.5, 1.0, 60.0, 12.0, 50.0).expect("valid servo");
    e.set_joint_motor(j, Some(servo)).expect("servo attaches");
    e.set_solver_kind(SolverKind::Avbd, Vec3::ZERO);
    assert_eq!(e.joint_motor(j), Some(servo));
    e.set_solver_kind(SolverKind::SequentialImpulse, Vec3::ZERO);
    assert_eq!(e.joint_motor(j), Some(servo));
}
