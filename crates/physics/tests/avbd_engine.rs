//! M1 gate: `AvbdEngine` behind the shared [`PhysicsEngine`] trait — stack
//! stability, joints, determinism, triggers, raycast parity, seam fit.

use glam::Vec3;
use ornis_physics::trigger::{ContactEventKind, TriggerEventKind};
use ornis_physics::{
    AvbdEngine, BodyHandle, BuiltinPhysicsEngine, JointKind, PhysicsEngine, PrismaticLimit,
    PrismaticMotor, Ray, RevoluteLimit, RevoluteMotor, RigidBody,
};

const DT: f32 = 1.0 / 60.0;

fn floor_stack(engine: &mut impl PhysicsEngine) -> Vec<BodyHandle> {
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let mut handles = Vec::new();
    for level in 0..4 {
        handles.push(engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.5 + level as f32 * 1.02, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            1.0,
        )));
    }
    handles
}

#[test]
fn avbd_stack_stands_like_si() {
    let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let handles = floor_stack(&mut physics);
    for _ in 0..300 {
        physics.step(DT);
    }
    for (level, &h) in handles.iter().enumerate() {
        let b = physics.get_body(h).unwrap();
        let expected_y = 0.5 + level as f32;
        assert!(
            (b.position.y - expected_y).abs() < 0.1,
            "box {level} rest height ~{expected_y}, got {}",
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
fn avbd_sphere_pile_settles() {
    // Spheres dropped in a spaced row settle on the floor (sphere-floor
    // catch + rest; M1 scope). Sphere-on-sphere stacking needs rolling
    // multi-point contact (M2 gap, see module docs) and is not exercised.
    let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let mut handles = Vec::new();
    for (i, x) in [-1.2f32, 0.0, 1.2].iter().enumerate() {
        handles.push(physics.add_body(RigidBody::new_sphere(
            Vec3::new(*x, 1.5 + i as f32 * 0.1, 0.0),
            0.5,
            1.0,
        )));
    }
    for _ in 0..300 {
        physics.step(DT);
    }
    for &h in &handles {
        let b = physics.get_body(h).unwrap();
        assert!(
            b.position.y > 0.3 && b.position.y < 0.8,
            "sphere fell through or launched: {}",
            b.position.y
        );
        assert!(
            b.velocity.length() < 0.5,
            "sphere not settled: {:?}",
            b.velocity
        );
    }
}

#[test]
fn avbd_ball_joint_pendulum_holds_anchor() {
    // Mirror the builtin G5 scene: assembled at coincidence, bob released
    // off to the side — it must swing, not fall. (Large-offset assembly
    // convergence is a known M1 limitation, see module docs.)
    let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.1, 0.0));
    let bob = physics.add_body(RigidBody::new_sphere(Vec3::new(1.0, -1.0, 0.0), 0.25, 1.0));
    let lb = Vec3::new(-1.0, 1.0, 0.0); // world anchor = origin
    let joint = physics.add_joint(
        anchor,
        bob,
        JointKind::Ball {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: lb,
        },
    );
    assert!(joint.is_some());
    for _ in 0..300 {
        physics.step(DT);
        let b = physics.get_body(bob).unwrap();
        let anchor_world = b.position + b.orientation * lb;
        // AVBD joints are penalty-based: ~cm compliance with ~10cm dynamic
        // stretch at swing bottom (M2: stiffness). Bounded is what matters.
        assert!(
            anchor_world.length() < 0.15,
            "ball anchor drifted: {anchor_world:?}"
        );
    }
    let b = physics.get_body(bob).unwrap();
    let dist = b.position.length();
    assert!(
        (dist - std::f32::consts::SQRT_2).abs() < 0.2,
        "pendulum length drifted: {dist}"
    );
}

#[test]
fn avbd_revolute_hinge_keeps_axis() {
    // Assembled at coincidence (hinge center shared), arm hanging straight
    // down with a sideways nudge: the hinge must carry the swing on its
    // axis without anchor drift.
    let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let a = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.25), 0.0));
    let b = physics.add_body(RigidBody::new_box(
        Vec3::new(0.3, -1.0, 0.0),
        Vec3::new(0.2, 1.0, 0.2),
        1.0,
    ));
    let lb = Vec3::new(-0.3, 1.0, 0.0); // world hinge center = origin
    let joint = physics.add_joint(
        a,
        b,
        JointKind::Revolute {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: lb,
            local_axis_a: Vec3::Z,
            local_axis_b: Vec3::Z,
            limit: None,
            motor: None,
        },
    );
    assert!(joint.is_some());
    for _ in 0..300 {
        physics.step(DT);
    }
    let ba = physics.get_body(a).unwrap();
    let bb = physics.get_body(b).unwrap();
    let axa = ba.orientation * Vec3::Z;
    let axb = bb.orientation * Vec3::Z;
    assert!(
        axa.dot(axb) > 0.99,
        "hinge axes diverged: {axa:?} vs {axb:?}"
    );
    let anchor_world = bb.position + bb.orientation * lb;
    assert!(
        anchor_world.length() < 0.15,
        "hinge anchor drifted: {anchor_world:?}"
    );
}

#[test]
fn avbd_unsupported_joints_return_none() {
    let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let a = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0));
    let b = physics.add_body(RigidBody::new_sphere(Vec3::X, 0.5, 1.0));
    // Wheel/Gear/SixDof row models live in M2.
    assert!(
        physics
            .add_joint(
                a,
                b,
                JointKind::Wheel {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                    local_suspension_a: Vec3::Y,
                    local_suspension_b: Vec3::Y,
                    local_axle_a: Vec3::Z,
                    local_axle_b: Vec3::Z,
                    suspension: ornis_physics::WheelSuspension {
                        frequency_hz: 2.0,
                        damping_ratio: 0.7,
                    },
                    motor: None,
                },
            )
            .is_none(),
        "wheel must be refused (M2 gap)"
    );
    assert!(
        physics
            .add_joint(
                a,
                a,
                JointKind::Ball {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                }
            )
            .is_none()
    );
}

#[test]
fn avbd_determinism_run_to_run() {
    fn run() -> Vec<[f32; 3]> {
        let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let handles = floor_stack(&mut physics);
        for _ in 0..60 {
            physics.step(DT);
        }
        handles
            .iter()
            .map(|&h| physics.get_body(h).unwrap().position.to_array())
            .collect()
    }
    let a = run();
    let b = run();
    assert_eq!(a, b, "AVBD must be bit-identical run-to-run");
}

#[test]
fn avbd_trigger_enter_exit_roundtrip() {
    let mut physics = AvbdEngine::new(Vec3::ZERO);
    let mut sensor = RigidBody::new_sphere(Vec3::ZERO, 1.0, 0.0);
    sensor.is_trigger = true;
    let s = physics.add_body(sensor);
    let m = physics.add_body(RigidBody::new_sphere(Vec3::new(5.0, 0.0, 0.0), 0.5, 1.0));
    physics.step(DT);
    assert!(physics.drain_trigger_events().is_empty());
    physics.get_body_mut(m).unwrap().position = Vec3::ZERO;
    physics.step(DT);
    let entered = physics.drain_trigger_events();
    assert_eq!(entered.len(), 1);
    assert_eq!(entered[0].kind, TriggerEventKind::Entered);
    assert_eq!((entered[0].body_a, entered[0].body_b), (s.min(m), s.max(m)));
    physics.get_body_mut(m).unwrap().position = Vec3::new(5.0, 0.0, 0.0);
    physics.step(DT);
    let exited = physics.drain_trigger_events();
    assert_eq!(exited.len(), 1);
    assert_eq!(exited[0].kind, TriggerEventKind::Exited);
}

#[test]
fn avbd_contact_begin_end_roundtrip() {
    let mut physics = AvbdEngine::new(Vec3::ZERO);
    let a = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.5, 0.0));
    let b = physics.add_body(RigidBody::new_sphere(Vec3::new(5.0, 0.0, 0.0), 0.5, 1.0));
    physics.step(DT);
    assert!(physics.drain_contact_events().is_empty());
    physics.get_body_mut(b).unwrap().position = Vec3::new(0.9, 0.0, 0.0);
    physics.step(DT);
    let begin: Vec<_> = physics.drain_contact_events();
    assert!(
        begin
            .iter()
            .any(|e| matches!(e.kind, ContactEventKind::Begin)),
        "expected a Begin, got {begin:?}"
    );
    let _ = (a, b);
    physics.get_body_mut(b).unwrap().position = Vec3::new(5.0, 0.0, 0.0);
    physics.step(DT);
    let end: Vec<_> = physics.drain_contact_events();
    assert!(
        end.iter().any(|e| matches!(e.kind, ContactEventKind::End)),
        "expected an End, got {end:?}"
    );
}

#[test]
fn avbd_raycast_matches_builtin() {
    let mut avbd = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let mut builtin = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    for engine in [
        &mut avbd as &mut dyn PhysicsEngine,
        &mut builtin as &mut dyn PhysicsEngine,
    ] {
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        engine.add_body(RigidBody::new_sphere(Vec3::new(2.0, 1.0, 0.0), 0.5, 1.0));
    }
    let ray = Ray {
        origin: Vec3::new(2.0, 5.0, 0.0),
        direction: Vec3::new(0.0, -1.0, 0.0),
    };
    let ha = avbd.raycast(ray, 10.0).expect("avbd hit");
    let hb = builtin.raycast(ray, 10.0).expect("builtin hit");
    assert_eq!(ha.handle, hb.handle);
    assert!((ha.distance - hb.distance).abs() < 1e-4);
    assert!((ha.point - hb.point).length() < 1e-4);
}

#[test]
fn avbd_seam_fits_trait_object() {
    fn drive(engine: &mut dyn PhysicsEngine) -> f32 {
        let h = engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, 5.0, 0.0), 0.5, 1.0));
        for _ in 0..60 {
            engine.step(DT);
        }
        engine.get_body(h).unwrap().position.y
    }
    let ya = drive(&mut AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0)));
    let yb = drive(&mut BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0)));
    assert!(ya < 5.0 && yb < 5.0, "both engines integrate gravity");
}

/// Hinge twist about Z read back from a body orientation (the test-side
/// mirror of the shared twist measurement; valid while the hinge rows keep
/// X/Y near zero).
fn z_twist(q: glam::Quat) -> f32 {
    2.0 * q.z.asin()
}

#[test]
fn avbd_hinge_limit_holds_bound() {
    // Pendulum released sideways with a ±0.5 rad window: the swing must
    // clamp at the bound instead of passing through.
    let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.1, 0.0));
    let bob = physics.add_body(RigidBody::new_sphere(Vec3::new(1.0, -1.0, 0.0), 0.25, 1.0));
    let lb = Vec3::new(-1.0, 1.0, 0.0);
    assert!(
        physics
            .add_joint(
                anchor,
                bob,
                JointKind::Revolute {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: lb,
                    local_axis_a: Vec3::Z,
                    local_axis_b: Vec3::Z,
                    limit: Some(RevoluteLimit {
                        min: -0.5,
                        max: 0.5
                    }),
                    motor: None,
                },
            )
            .is_some()
    );
    let mut max_abs = 0.0f32;
    for _ in 0..300 {
        physics.step(DT);
        let b = physics.get_body(bob).unwrap();
        let anchor_world = b.position + b.orientation * lb;
        assert!(
            anchor_world.length() < 0.2,
            "limited hinge anchor drifted: {anchor_world:?}"
        );
        max_abs = max_abs.max(z_twist(b.orientation).abs());
    }
    assert!(
        max_abs <= 0.62,
        "hinge blew past its limit window: {max_abs}"
    );
    assert!(max_abs > 0.2, "hinge never swung into its bound: {max_abs}");
}

#[test]
fn avbd_hinge_motor_spins_up() {
    // Spin about the bob's own center (zero lever): no orbital motion, so
    // the ball position-servo damping (documented M1 gap on long levers)
    // cannot mask the motor. The anchor is a trigger (no contact fight,
    // joints still apply) coincident with the bob center.
    let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.05, 0.0));
    physics.get_body_mut(anchor).unwrap().is_trigger = true;
    let bob = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.1, 1.0));
    assert!(
        physics
            .add_joint(
                anchor,
                bob,
                JointKind::Revolute {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                    local_axis_a: Vec3::Z,
                    local_axis_b: Vec3::Z,
                    limit: None,
                    motor: Some(RevoluteMotor {
                        target_speed: 3.0,
                        max_torque: 20.0,
                    }),
                },
            )
            .is_some()
    );
    for _ in 0..300 {
        physics.step(DT);
    }
    // The deadbeat holds 3.0 on average; gravity ripples the tumbling bob
    // ±2 rad/s instantaneously, so average over the tail, not one sample.
    let mut mean_w = 0.0f32;
    for _ in 0..60 {
        physics.step(DT);
        mean_w += physics.get_body(bob).unwrap().angular_velocity.z;
    }
    mean_w /= 60.0;
    assert!(
        (mean_w - 3.0).abs() < 0.6,
        "motor did not hold target speed: {mean_w}"
    );
}

#[test]
fn avbd_prismatic_slider_holds_line_and_limit() {
    // Vertical slide assembled just above its lower bound: gravity pulls
    // the bob 10cm down the Y axis; x/z and the axis alignment must hold,
    // and the [-2, 0] window must catch it. (Catching a multi-meter fall in
    // one position step is outside the single-step linearization envelope
    // — M2 substeps; the bound approach here is gradual, like the hinge.)
    let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.25), 0.0));
    let bob = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, -2.9, 0.0), 0.25, 1.0));
    let lb = Vec3::new(0.0, 1.0, 0.0);
    assert!(
        physics
            .add_joint(
                anchor,
                bob,
                JointKind::Prismatic {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: lb,
                    local_axis_a: Vec3::Y,
                    local_axis_b: Vec3::Y,
                    limit: Some(PrismaticLimit {
                        min: -2.0,
                        max: 0.0
                    }),
                    motor: None,
                },
            )
            .is_some()
    );
    for _ in 0..300 {
        physics.step(DT);
        let b = physics.get_body(bob).unwrap();
        assert!(
            b.position.x.abs() < 0.08 && b.position.z.abs() < 0.08,
            "slider left its line: {:?}",
            b.position
        );
    }
    let b = physics.get_body(bob).unwrap();
    assert!(
        b.position.y > -3.4 && b.position.y < -2.6,
        "slider did not rest at its lower limit: {:?}",
        b.position
    );
}

#[test]
fn avbd_prismatic_motor_drives() {
    // Horizontal slide: no gravity along the drive axis (a velocity servo
    // droops under sustained load), gravity transverse (perp rows carry it).
    let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.25), 0.0));
    let bob = physics.add_body(RigidBody::new_sphere(Vec3::new(1.0, 0.0, 0.0), 0.25, 1.0));
    assert!(
        physics
            .add_joint(
                anchor,
                bob,
                JointKind::Prismatic {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::new(-1.0, 0.0, 0.0),
                    local_axis_a: Vec3::X,
                    local_axis_b: Vec3::X,
                    limit: None,
                    motor: Some(PrismaticMotor {
                        target_speed: 2.0,
                        max_force: 50.0,
                    }),
                },
            )
            .is_some()
    );
    for _ in 0..300 {
        physics.step(DT);
    }
    let b = physics.get_body(bob).unwrap();
    assert!(
        (b.velocity.x - 2.0).abs() < 0.4,
        "motor did not reach slide speed: {:?}",
        b.velocity
    );
    assert!(
        b.position.y.abs() < 0.15 && b.position.z.abs() < 0.1,
        "motor drive left the slide line: {:?}",
        b.position
    );
}

#[test]
fn avbd_fixed_weld_holds() {
    // Two welded boxes dropped together: separation and relative rotation
    // must survive free fall and landing.
    let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let a = physics.add_body(RigidBody::new_box(
        Vec3::new(-0.5, 3.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    let b = physics.add_body(RigidBody::new_box(
        Vec3::new(0.5, 3.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    assert!(
        physics
            .add_joint(
                a,
                b,
                JointKind::Fixed {
                    local_anchor_a: Vec3::new(0.5, 0.0, 0.0),
                    local_anchor_b: Vec3::new(-0.5, 0.0, 0.0),
                },
            )
            .is_some()
    );
    for _ in 0..300 {
        physics.step(DT);
    }
    let ba = physics.get_body(a).unwrap();
    let bb = physics.get_body(b).unwrap();
    let sep = (bb.position - ba.position).length();
    assert!((sep - 1.0).abs() < 0.15, "weld separation drifted: {sep}");
    let align = (ba.orientation * Vec3::X).dot(bb.orientation * Vec3::X);
    assert!(align > 0.98, "weld rotated apart: {align}");
    assert!(
        ba.position.y > 0.2 && bb.position.y > 0.2,
        "welded pair tunneled: {:?} {:?}",
        ba.position,
        bb.position
    );
}

#[test]
fn avbd_distance_rod_holds_length() {
    // Rod pendulum: assembled 2m apart horizontally, the anchor separation
    // keeps its length while rotation stays free (the bob swings down).
    let mut physics = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.1, 0.0));
    let bob = physics.add_body(RigidBody::new_sphere(Vec3::new(2.0, 0.0, 0.0), 0.25, 1.0));
    assert!(
        physics
            .add_joint(
                anchor,
                bob,
                JointKind::Distance {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                },
            )
            .is_some()
    );
    let mut swung = false;
    for _ in 0..300 {
        physics.step(DT);
        let b = physics.get_body(bob).unwrap();
        let len = b.position.length();
        assert!((len - 2.0).abs() < 0.25, "rod length drifted: {len}");
        swung |= b.position.y < -0.5;
    }
    assert!(swung, "rod bob never swung down");
}
