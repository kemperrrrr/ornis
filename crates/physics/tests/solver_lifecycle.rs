//! Public-API regressions for solver switching, routing, handles and events.
//! These exercise real solver steps; migration must not change physical recipes.

use glam::{Quat, Vec3};
use ornis_physics::{
    AxisConfig, BodyHandle, BodyType, ContactEventKind, Engine, JointHandle, JointKind,
    PhysicsEngine, Ray, RigidBody, RoutingKind, Shape, SolverKind, TriggerEventKind,
};

const DT: f32 = 1.0 / 60.0;
const KINDS: [SolverKind; 2] = [SolverKind::SequentialImpulse, SolverKind::Avbd];
const ROUTES: [RoutingKind; 2] = [RoutingKind::Single, RoutingKind::Islands];

fn sphere(pos: Vec3, mass: f32) -> RigidBody {
    RigidBody::new_sphere(pos, 0.2, mass)
}

fn isolated(pos: Vec3, mass: f32) -> RigidBody {
    let mut b = sphere(pos, mass);
    b.collision_mask = 0;
    b
}

fn slider(a: BodyHandle, b: BodyHandle, engine: &mut Engine) -> JointHandle {
    engine
        .add_joint(
            a,
            b,
            JointKind::Prismatic {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_axis_a: Vec3::X,
                local_axis_b: Vec3::X,
                limit: None,
                motor: None,
            },
        )
        .unwrap()
}

#[test]
fn removing_an_unrelated_body_preserves_the_moved_tails_joint() {
    for kind in KINDS {
        for route in ROUTES {
            let mut e = Engine::new(kind, Vec3::ZERO);
            e.set_routing(route);
            let anchor = e.add_body(isolated(Vec3::ZERO, 0.0));
            let spare = e.add_body(isolated(Vec3::X * 20.0, 1.0));
            let tail = e.add_body(isolated(Vec3::Y * 2.0, 1.0));
            e.add_joint(
                anchor,
                tail,
                JointKind::Distance {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                },
            )
            .unwrap();
            e.remove_body(spare);
            assert_eq!(e.body_count(), 2);
            assert_eq!(
                e.joint_count(),
                1,
                "{kind:?}/{route:?}: tail joint was deleted"
            );
            assert!(e.get_body(tail).is_none());
            e.get_body_mut(spare).unwrap().position.y = 3.0;
            for _ in 0..600 {
                e.step(DT);
            }
            let body = e.get_body(spare).unwrap();
            assert!(
                (body.position.y - 2.0).abs() < 0.04,
                "{kind:?}/{route:?}: {body:?}"
            );
        }
    }
}

#[test]
fn gear_references_survive_unrelated_and_dependent_joint_removal() {
    for kind in KINDS {
        for route in ROUTES {
            let mut e = Engine::new(kind, Vec3::ZERO);
            e.set_routing(route);
            let a = e.add_body(isolated(Vec3::ZERO, 0.0));
            let b = e.add_body(isolated(Vec3::X, 1.0));
            let c = e.add_body(isolated(Vec3::X * 3.0, 1.0));
            let d = e.add_body(isolated(Vec3::X * 5.0, 1.0));
            let ja = slider(a, b, &mut e);
            let jb = slider(a, c, &mut e);
            // Metadata endpoints intentionally differ from the coordinate owners.
            e.add_joint(
                a,
                d,
                JointKind::Gear {
                    joint_a: ja,
                    joint_b: jb,
                    ratio: -1.0,
                },
            )
            .unwrap();
            let jc = slider(a, d, &mut e);
            e.add_joint(
                c,
                d,
                JointKind::Gear {
                    joint_a: jb,
                    joint_b: jc,
                    ratio: -1.0,
                },
            )
            .unwrap();
            e.remove_body(b);
            assert_eq!(e.joint_count(), 3, "{kind:?}/{route:?}");
            // The second gear must still coordinate the surviving c/d sliders.
            e.get_body_mut(c).unwrap().velocity.x = 1.0;
            for _ in 0..120 {
                e.step(DT);
            }
            let sc = e.get_body(c).unwrap().position.x - 3.0;
            let sd = e.get_body(b).unwrap().position.x - 5.0;
            assert!((sc - sd).abs() < 0.1, "{kind:?}/{route:?}: {sc} vs {sd}");
            e.remove_joint(JointHandle::from_raw(0));
            assert_eq!(e.joint_count(), 1, "dependent gear must die with its side");
            e.step(DT);
        }
    }
}

#[test]
fn assembly_rest_length_survives_repeated_solver_and_routing_switches() {
    for kind in KINDS {
        let mut e = Engine::new(kind, Vec3::ZERO);
        let a = e.add_body(isolated(Vec3::ZERO, 0.0));
        let b = e.add_body(isolated(Vec3::X * 2.0, 1.0));
        e.add_joint(
            a,
            b,
            JointKind::Distance {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
            },
        )
        .unwrap();
        e.get_body_mut(b).unwrap().position.x = 3.0;
        for _ in 0..4 {
            e.set_routing(RoutingKind::Islands);
            e.set_solver_kind(SolverKind::SequentialImpulse, Vec3::ZERO);
            e.set_solver_kind(SolverKind::Avbd, Vec3::ZERO);
        }
        assert_eq!(
            e.get_body(b).unwrap().position.x,
            3.0,
            "switching must not move poses"
        );
        for _ in 0..600 {
            e.step(DT);
        }
        assert!(
            (e.get_body(b).unwrap().position.x - 2.0).abs() < 0.04,
            "rest state was recaptured: {:?}",
            e.get_body(b).unwrap()
        );
    }
}

#[test]
fn sixdof_linear_limit_is_relative_to_assembly_not_world_origin() {
    for kind in KINDS {
        for route in ROUTES {
            let mut e = Engine::new(kind, Vec3::ZERO);
            let a = e.add_body(isolated(Vec3::ZERO, 0.0));
            let mut body = isolated(Vec3::X * 5.0, 1.0);
            body.velocity.x = 1.0;
            let b = e.add_body(body);
            e.add_joint(
                a,
                b,
                JointKind::SixDof {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                    linear: [
                        AxisConfig::Limited { min: 0.0, max: 1.0 },
                        AxisConfig::Locked,
                        AxisConfig::Locked,
                    ],
                    angular: [AxisConfig::Locked; 3],
                },
            )
            .unwrap();
            e.set_routing(route);
            for _ in 0..180 {
                e.step(DT);
            }
            let x = e.get_body(b).unwrap().position.x;
            assert!((5.9..6.05).contains(&x), "{kind:?}/{route:?}: x={x}");
        }
    }
}

#[test]
fn split_queries_see_add_edit_remove_without_a_step() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    e.set_routing(RoutingKind::Islands);
    let h = e.add_body(sphere(Vec3::X * 2.0, 1.0));
    let ray = Ray {
        origin: Vec3::ZERO,
        direction: Vec3::X,
    };
    assert_eq!(e.raycast(ray, 4.0).expect("valid").expect("hit").handle, h);
    let cast = Shape::Sphere { radius: 0.1 };
    assert_eq!(
        e.shapecast(&cast, Vec3::ZERO, Vec3::X * 4.0)
            .expect("hit")
            .handle,
        h
    );
    e.get_body_mut(h).unwrap().position.x = 8.0;
    assert!(e.raycast(ray, 4.0).unwrap_or(None).is_none());
    assert!(e.shapecast(&cast, Vec3::ZERO, Vec3::X * 4.0).is_none());
    e.remove_body(h);
    assert!(e.raycast(ray, 20.0).unwrap_or(None).is_none());
}

#[test]
fn split_fixed_time_is_independent_of_host_partition() {
    let run = |dt: f32, count: usize| {
        let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
        let mut fast = isolated(Vec3::X * -10.0, 1.0);
        fast.velocity.x = 1.0;
        e.add_body(fast);
        let mut slow = isolated(Vec3::X * 10.0, 1.0);
        slow.angular_velocity.z = 0.3;
        e.add_body(slow);
        e.set_routing(RoutingKind::Islands);
        for _ in 0..count {
            e.step(dt);
        }
        assert_eq!(e.split_metrics().unwrap().simulation_steps, 60);
        (0..2usize)
            .map(|h| {
                let b = e.get_body(BodyHandle::from(h)).unwrap();
                (
                    b.position,
                    b.orientation,
                    b.velocity,
                    b.angular_velocity,
                    e.body_solver(BodyHandle::from(h)),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(DT, 60), run(DT * 0.5, 120));
    assert_eq!(run(DT, 60), run(DT * 2.0, 30));
}

#[test]
fn invalid_time_does_not_route_rebuild_or_consume_events() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    e.set_routing(RoutingKind::Islands);
    let h = e.add_body(sphere(Vec3::ZERO, 1.0));
    let rebuilds = e.split_metrics().unwrap().rebuilds;
    for dt in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        e.step(dt);
    }
    let m = e.split_metrics().unwrap();
    assert_eq!(m.simulation_steps, 0);
    assert_eq!(m.rebuilds, rebuilds);
    assert_eq!(e.get_body(h).unwrap().position, Vec3::ZERO);
}

#[test]
fn split_trigger_history_survives_rebuild_and_is_not_duplicated() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let trigger = e.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::ONE, 0.0).with_trigger(true));
    let body = e.add_body(sphere(Vec3::ZERO, 1.0));
    e.set_routing(RoutingKind::Islands);
    e.step(DT);
    // Deferred rebuilding must not consume an event waiting to be drained.
    e.add_body(sphere(Vec3::X * 20.0, 1.0));
    let events = e.drain_trigger_events();
    assert_eq!(events.len(), 1);
    assert_eq!(
        (events[0].body_a, events[0].body_b, events[0].kind),
        (trigger, body, TriggerEventKind::Entered)
    );
    e.get_body_mut(body).unwrap().velocity.x = 0.6;
    e.step(DT);
    assert_eq!(e.body_solver(body), Some(SolverKind::Avbd));
    assert!(
        e.drain_trigger_events().is_empty(),
        "migration emitted another Entered"
    );
    e.get_body_mut(body).unwrap().position.x = 4.0;
    e.step(DT);
    let events = e.drain_trigger_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, TriggerEventKind::Exited);
}

#[test]
fn split_static_copies_emit_one_trigger_transition() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    e.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::ONE, 0.0).with_trigger(true));
    let mut driver = sphere(Vec3::ZERO, 0.0);
    driver.body_type = BodyType::Kinematic;
    e.add_body(driver);
    e.set_routing(RoutingKind::Islands);
    e.step(DT);
    let events = e.drain_trigger_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, TriggerEventKind::Entered);
}

#[test]
fn split_joins_contact_owners_before_integrating() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let a = e.add_body(RigidBody::new_sphere(Vec3::X * 1.01, 0.5, 1.0));
    let b = e.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.5, 100.0));
    e.set_routing(RoutingKind::Islands);
    e.get_body_mut(a).unwrap().velocity = Vec3::NEG_X;
    e.step(DT);
    assert_eq!(e.body_solver(a), Some(SolverKind::Avbd));
    assert_eq!(e.body_solver(b), e.body_solver(a));
    let (a, b) = (e.get_body(a).unwrap(), e.get_body(b).unwrap());
    let momentum = a.mass * a.velocity + b.mass * b.velocity;
    assert!(
        (momentum - Vec3::NEG_X).length() < 0.05,
        "cross-solver impulse lost: {momentum:?}"
    );
}

#[test]
fn split_hysteresis_includes_spin_and_preserves_the_deadband_owner() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::ZERO);
    let mut moving = isolated(Vec3::ZERO, 1.0);
    moving.velocity.x = 0.3;
    let h = e.add_body(moving);
    let mut spinning = isolated(Vec3::X * 20.0, 1.0);
    spinning.angular_velocity.z = 2.0;
    let spin = e.add_body(spinning);
    e.set_routing(RoutingKind::Islands);
    for _ in 0..120 {
        e.step(DT);
    }
    assert_eq!(e.body_solver(h), Some(SolverKind::SequentialImpulse));
    assert_eq!(e.body_solver(spin), Some(SolverKind::Avbd));
    assert_eq!(e.split_metrics().unwrap().migrations, 0);
}

#[test]
fn split_role_edits_reconcile_ownership() {
    let mut e = Engine::new(SolverKind::Avbd, Vec3::NEG_Y);
    let h = e.add_body(sphere(Vec3::Y * 5.0, 1.0));
    e.set_routing(RoutingKind::Islands);
    *e.get_body_mut(h).unwrap() = sphere(Vec3::Y * 5.0, 0.0);
    e.step(DT);
    assert_eq!(e.body_solver(h), None);
    assert_eq!(e.get_body(h).unwrap().position, Vec3::Y * 5.0);
    *e.get_body_mut(h).unwrap() = sphere(Vec3::Y * 5.0, 1.0);
    e.step(DT);
    assert_eq!(e.body_solver(h), Some(SolverKind::Avbd));
    assert!(e.get_body(h).unwrap().position.y < 5.0);
}

#[test]
fn native_joint_rotation_is_quaternion_sign_invariant() {
    for kind in KINDS {
        let run = |negative: bool| {
            let mut e = Engine::new(kind, Vec3::ZERO);
            let qa = Quat::from_rotation_y(0.7);
            let mut a = isolated(Vec3::ZERO, 0.0);
            a.orientation = qa;
            let mut b = isolated(Vec3::ZERO, 1.0);
            b.orientation = qa;
            let (a, b) = (e.add_body(a), e.add_body(b));
            e.add_joint(
                a,
                b,
                JointKind::Fixed {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::ZERO,
                },
            )
            .unwrap();
            let q = qa * Quat::from_rotation_z(0.2);
            e.get_body_mut(b).unwrap().orientation = if negative { -q } else { q };
            e.step(DT);
            e.get_body(b).unwrap().orientation * Vec3::X
        };
        assert!(
            (run(false) - run(true)).length() < 1e-5,
            "{kind:?}: q/-q changed correction"
        );
    }
}

#[test]
fn contact_events_are_still_available_after_structural_changes() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, Vec3::NEG_Y * 9.81);
    e.add_body(RigidBody::new_box(
        Vec3::NEG_Y,
        Vec3::new(10.0, 1.0, 10.0),
        0.0,
    ));
    e.add_body(sphere(Vec3::Y * 0.2, 1.0));
    e.set_routing(RoutingKind::Islands);
    let mut begins = 0;
    for _ in 0..90 {
        e.step(DT);
        begins += e
            .drain_contact_events()
            .iter()
            .filter(|e| e.kind == ContactEventKind::Begin)
            .count();
    }
    assert_eq!(
        begins, 1,
        "resting contact must not re-begin when routing changes"
    );
}

#[test]
fn a_contained_sphere_is_a_trigger_overlap_in_every_solver() {
    for kind in KINDS {
        for route in ROUTES {
            let mut e = Engine::new(kind, Vec3::ZERO);
            let mut trigger = RigidBody::new_box(Vec3::ZERO, Vec3::ONE, 0.0).with_trigger(true);
            trigger.orientation = Quat::from_rotation_y(0.7);
            e.add_body(trigger);
            e.add_body(sphere(Vec3::ZERO, 1.0));
            e.set_routing(route);
            e.step(DT);
            let events = e.drain_trigger_events();
            assert_eq!(
                events.len(),
                1,
                "{kind:?}/{route:?}: containment was invisible"
            );
            assert_eq!(events[0].kind, TriggerEventKind::Entered);
        }
    }
}

#[test]
fn coincident_dynamic_spheres_do_not_remain_ghosted() {
    for kind in KINDS {
        let mut e = Engine::new(kind, Vec3::ZERO);
        let a = e.add_body(sphere(Vec3::ZERO, 1.0));
        let b = e.add_body(sphere(Vec3::ZERO, 1.0));
        for _ in 0..180 {
            e.step(DT);
        }
        let pa = e.get_body(a).unwrap().position;
        let pb = e.get_body(b).unwrap().position;
        assert!(pa.is_finite() && pb.is_finite());
        assert!(
            (pa - pb).length() > 0.3,
            "{kind:?}: concentric spheres stayed coincident"
        );
    }
}

#[test]
fn contained_boxes_and_face_crossing_capsules_are_not_missing_overlaps() {
    for kind in KINDS {
        for shape in [
            Shape::Box {
                half_extents: Vec3::splat(0.2),
            },
            Shape::Capsule {
                radius: 0.1,
                half_height: 0.2,
            },
            Shape::Capsule {
                radius: 0.1,
                half_height: 3.0,
            },
        ] {
            let mut e = Engine::new(kind, Vec3::ZERO);
            e.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::ONE, 0.0).with_trigger(true));
            let mut body = sphere(Vec3::ZERO, 1.0);
            body.shape = shape;
            body.inertia = body.shape.inertia(body.mass);
            e.add_body(body);
            e.step(DT);
            assert_eq!(
                e.drain_trigger_events().len(),
                1,
                "{kind:?}: solid overlap was missed"
            );
        }
    }
}
