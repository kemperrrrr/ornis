//! M3 release gates: assignment-invariant rest and scheduler-independent state.
//! Timing metrics are intentionally excluded from authoritative state equality.

use glam::Vec3;
use ornis_physics::{Engine, PhysicsEngine, RigidBody, RoutingKind, SolverKind};

const DT: f32 = 1.0 / 60.0;

fn floor() -> RigidBody {
    RigidBody::new_box(Vec3::NEG_Y, Vec3::new(50.0, 1.0, 50.0), 0.0)
}

#[test]
fn initial_solver_assignment_does_not_change_rest_state() {
    let run = |kind| {
        let mut e = Engine::new(kind, Vec3::NEG_Y * 9.81);
        e.add_body(floor());
        for y in [0.6, 1.7] {
            e.add_body(RigidBody::new_box(Vec3::Y * y, Vec3::splat(0.5), 1.0));
        }
        e.add_body(RigidBody::new_sphere(Vec3::new(5.0, 2.0, 0.0), 0.3, 2.0));
        e.set_routing(RoutingKind::Islands);
        for _ in 0..600 {
            e.step(DT);
        }
        (1..4)
            .map(|h| e.get_body(h).unwrap().position)
            .collect::<Vec<_>>()
    };
    let avbd = run(SolverKind::Avbd);
    let builtin = run(SolverKind::SequentialImpulse);
    for (a, b) in avbd.iter().zip(&builtin) {
        assert!(
            (*a - *b).length() < 0.1,
            "routing changed rest pose: {a:?} vs {b:?}"
        );
    }
    assert!((avbd[0].y - 0.5).abs() < 0.1);
    assert!((avbd[1].y - 1.5).abs() < 0.1);
    assert!((avbd[2].y - 0.3).abs() < 0.1);
}

#[test]
fn routed_world_is_bit_identical_with_one_or_many_workers() {
    let run = |workers| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            let mut e = Engine::new(SolverKind::Avbd, Vec3::NEG_Y * 9.81);
            e.add_body(floor());
            for x in 0..16 {
                for z in 0..16 {
                    e.add_body(RigidBody::new_box(
                        Vec3::new(x as f32 * 2.0 - 15.0, 0.4, z as f32 * 2.0 - 15.0),
                        Vec3::splat(0.4),
                        1.0,
                    ));
                }
            }
            let mut fast = RigidBody::new_sphere(Vec3::new(40.0, 10.0, 0.0), 0.3, 1.0);
            fast.velocity.x = 1.0;
            e.add_body(fast);
            e.set_routing(RoutingKind::Islands);
            let mut events = Vec::new();
            // Diet: 16 steps keep every gate — boxes land ~step 12 so the
            // contact-event history covers begins plus first contact steps,
            // and rebuilds stays 1 (initial island build, same as 40 steps).
            for _ in 0..16 {
                e.step(DT);
                // Debug's shortest float representation roundtrips finite f32
                // and preserves signed zero, unlike float PartialEq.
                events.push(format!(
                    "{:?}|{:?}",
                    e.drain_contact_events(),
                    e.drain_trigger_events()
                ));
            }
            let states = (0..e.body_count())
                .map(|h| {
                    let b = e.get_body(h).unwrap();
                    (
                        b.position.to_array().map(f32::to_bits),
                        b.orientation.to_array().map(f32::to_bits),
                        b.velocity.to_array().map(f32::to_bits),
                        b.angular_velocity.to_array().map(f32::to_bits),
                        e.body_solver(h),
                    )
                })
                .collect::<Vec<_>>();
            let metrics = e.split_metrics().unwrap();
            assert!(metrics.simulation_steps == 16 && metrics.rebuilds > 0);
            (states, events, metrics.migrations, metrics.migrated_bodies)
        })
    };
    assert_eq!(run(1), run(32));
}
