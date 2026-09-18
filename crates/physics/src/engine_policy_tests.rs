//! Assembly-state and fracture conservation checks for the orchestrator.

use super::*;
use glam::{Quat, Vec3};

fn body(position: Vec3, mass: f32) -> RigidBody {
    let mut b = RigidBody::new_sphere(position, 0.2, mass);
    b.collision_mask = 0;
    b
}

#[test]
fn references_are_not_recaptured_from_an_unsolved_pose() {
    let specs = [
        JointKind::Fixed {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
        },
        JointKind::Distance {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
        },
        JointKind::Prismatic {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            local_axis_a: Vec3::Y,
            local_axis_b: Vec3::Y,
            limit: None,
            motor: None,
        },
        JointKind::SixDof {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            linear: [AxisConfig::Locked; 3],
            angular: [AxisConfig::Locked; 3],
        },
    ];
    for kind in [SolverKind::Builtin, SolverKind::Avbd] {
        for spec in specs {
            let mut e = Engine::new(kind, Vec3::ZERO);
            let a = e.add_body(body(Vec3::ZERO, 0.0));
            let b = e.add_body(body(Vec3::new(2.0, 3.0, 4.0), 1.0));
            e.add_joint(a, b, spec).unwrap();
            let original = e.snapshot().joints[0].reference;
            e.get_body_mut(b).unwrap().position += Vec3::ONE;
            e.get_body_mut(b).unwrap().orientation = Quat::from_rotation_y(0.7);
            e.set_routing(RoutingKind::Islands);
            e.set_solver_kind(SolverKind::Builtin, Vec3::ZERO);
            e.set_solver_kind(SolverKind::Avbd, Vec3::ZERO);
            let restored = e.snapshot().joints[0].reference;
            assert_eq!(restored.distance, original.distance);
            assert_eq!(restored.length, original.length);
            assert_eq!(restored.rotation, original.rotation);
            assert_eq!(restored.anchor_delta, original.anchor_delta);
        }
    }
}

fn momentum(b: &RigidBody, origin: Vec3) -> (Vec3, Vec3, f32) {
    let linear = b.mass * b.velocity;
    let angular = b.orientation * (b.inertia * (b.orientation.conjugate() * b.angular_velocity));
    let energy = 0.5 * (b.mass * b.velocity.length_squared() + b.angular_velocity.dot(angular));
    (
        linear,
        angular + (b.position - origin).cross(linear),
        energy,
    )
}

#[test]
fn fracture_preserves_rigid_velocity_field_momentum_and_energy() {
    let mut parent = RigidBody::new_box(Vec3::new(2.0, 3.0, 4.0), Vec3::new(2.0, 1.0, 0.5), 4.0);
    parent.orientation = Quat::from_rotation_y(0.7);
    parent.velocity = Vec3::new(1.0, 2.0, 3.0);
    parent.angular_velocity = Vec3::new(2.0, -1.0, 3.0);
    let [a, b] = Engine::split_box(&parent).unwrap();
    let before = momentum(&parent, parent.position);
    let pa = momentum(&a, parent.position);
    let pb = momentum(&b, parent.position);
    assert!((pa.0 + pb.0 - before.0).length() < 1e-5);
    assert!((pa.1 + pb.1 - before.1).length() < 1e-4);
    assert!((pa.2 + pb.2 - before.2).abs() < 1e-4);
    let center_velocity =
        |b: &RigidBody| b.velocity + b.angular_velocity.cross(parent.position - b.position);
    assert!((center_velocity(&a) - parent.velocity).length() < 1e-5);
    assert!((center_velocity(&b) - parent.velocity).length() < 1e-5);
}

#[test]
fn simultaneous_fractures_report_the_final_piece_handles() {
    let mut e = Engine::new(SolverKind::Builtin, Vec3::ZERO);
    e.set_routing(RoutingKind::Islands);
    let wall = e.add_body(body(Vec3::Y * 10.0, 0.0));
    let mut victim = RigidBody::new_box(Vec3::NEG_X * 3.0, Vec3::ONE, 1.0);
    victim.fracture_impact_speed = 1.0;
    let a = e.add_body(victim.clone());
    victim.position.x = 3.0;
    let b = e.add_body(victim);
    let events: Vec<_> = [a, b]
        .into_iter()
        .map(|h| ContactEvent {
            body_a: wall,
            body_b: h,
            kind: ContactEventKind::Hit {
                point: Vec3::ZERO,
                normal: Vec3::Y,
                approach_speed: 2.0,
            },
        })
        .collect();
    e.fracture_split(&events);
    let fractures = e.drain_fracture_events();
    assert_eq!(fractures.len(), 2);
    for event in fractures {
        let center = (e.get_body(event.pieces[0]).unwrap().position
            + e.get_body(event.pieces[1]).unwrap().position)
            * 0.5;
        let expected = if event.parent == a { -3.0 } else { 3.0 };
        assert!(
            (center.x - expected).abs() < 1e-6,
            "piece handle retargeted: {event:?}"
        );
    }
}

#[test]
fn inherited_gear_phase_survives_a_multi_turn_switch() {
    let mut e = Engine::new(SolverKind::Builtin, Vec3::ZERO);
    let anchor = e.add_body(body(Vec3::ZERO, 0.0));
    let a = e.add_body(body(Vec3::NEG_X * 2.0, 1.0));
    let b = e.add_body(body(Vec3::X * 2.0, 1.0));
    let mut hinge = |h, x, motor| {
        e.add_joint(
            anchor,
            h,
            JointKind::Revolute {
                local_anchor_a: Vec3::X * x,
                local_anchor_b: Vec3::ZERO,
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: None,
                motor,
            },
        )
        .unwrap()
    };
    let ja = hinge(
        a,
        -2.0,
        Some(RevoluteMotor {
            target_speed: 2.0,
            max_torque: 50.0,
        }),
    );
    let jb = hinge(b, 2.0, None);
    e.add_joint(
        a,
        b,
        JointKind::Gear {
            joint_a: ja,
            joint_b: jb,
            ratio: 2.0,
        },
    )
    .unwrap();
    for _ in 0..240 {
        e.step(1.0 / 60.0);
    }
    let before = e.get_body(b).unwrap().orientation;
    e.set_solver_kind(SolverKind::Avbd, Vec3::ZERO);
    assert_eq!(e.joint_count(), 3);
    e.step(1.0 / 60.0);
    let after = e.get_body(b).unwrap();
    assert!(
        before.dot(after.orientation).abs() > 0.99,
        "gear phase jumped during migration"
    );
    assert!(after.angular_velocity.length() < 4.0);
}
