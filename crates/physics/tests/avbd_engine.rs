//! M1 gate: `AvbdEngine` behind the shared [`PhysicsEngine`] trait — stack
//! stability, joints, determinism, triggers, raycast parity, seam fit.

use glam::Vec3;
use ornis_physics::trigger::{ContactEventKind, TriggerEventKind};
use ornis_physics::{
    AvbdEngine, BodyHandle, BuiltinPhysicsEngine, JointKind, PhysicsEngine, Ray, RigidBody,
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
    assert!(
        physics
            .add_joint(
                a,
                b,
                JointKind::Revolute {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                    local_axis_a: Vec3::Z,
                    local_axis_b: Vec3::Z,
                    limit: Some(ornis_physics::joint::RevoluteLimit {
                        min: -0.5,
                        max: 0.5
                    }),
                    motor: None,
                },
            )
            .is_none(),
        "limited hinge must be refused in M1"
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
