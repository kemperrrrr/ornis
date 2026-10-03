//! Contact-force events (P8): threshold opt-in, peak-force reporting and
//! deterministic drain order, in the Rapier `CONTACT_FORCE_EVENTS` spirit.
//!
//! The force bounds below are empirical (1 kg sphere, default substeps):
//! a 1 m drop peaks around 2 kN (quiet under a 5 kN threshold), a 10 m
//! drop peaks an order of magnitude higher (loud). The test pins the
//! separation, not exact solver numbers.

use glam::Vec3;

use ornis_physics::{ContactForceEvent, PhysicsEngine, RigidBody, SequentialImpulseEngine};

/// Floor plus one 1 kg sphere dropped from `height`, opted in with
/// `threshold`. Returns the engine and the sphere handle.
fn drop_scene(height: f32, threshold: f32) -> (SequentialImpulseEngine, ornis_physics::BodyHandle) {
    let mut engine = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    let ball = engine.add_body(
        RigidBody::new_sphere(Vec3::new(0.0, height, 0.0), 0.5, 1.0)
            .with_contact_force_threshold(threshold),
    );
    (engine, ball)
}

/// Steps `n` times at 60 Hz, collecting every drained force event.
fn run(engine: &mut SequentialImpulseEngine, n: u32) -> Vec<ContactForceEvent> {
    let mut out = Vec::new();
    for _ in 0..n {
        engine.step(1.0 / 60.0);
        out.extend(engine.drain_contact_force_events());
    }
    out
}

/// A 1 m drop stays under a 5 kN threshold (quiet); a 10 m drop reports
/// with a force in the impact ballpark.
#[test]
fn force_threshold_separates_gentle_and_hard_impacts() {
    const THRESHOLD: f32 = 5000.0;
    let (mut gentle, _) = drop_scene(1.0, THRESHOLD);
    let quiet = run(&mut gentle, 120);
    assert!(
        quiet.is_empty(),
        "1 m drop must stay quiet under {THRESHOLD} N, got {} events (peak {} N)",
        quiet.len(),
        quiet.iter().map(|e| e.force).fold(0.0f32, f32::max)
    );

    let (mut hard, _) = drop_scene(10.0, THRESHOLD);
    let loud = run(&mut hard, 180);
    assert!(
        !loud.is_empty(),
        "10 m drop must report at least one force event over {THRESHOLD} N"
    );
    let peak = loud.iter().map(|e| e.force).fold(0.0f32, f32::max);
    assert!(
        (THRESHOLD..=60_000.0).contains(&peak),
        "10 m impact peak {peak} N outside the [5 kN, 60 kN] ballpark"
    );
    for e in &loud {
        assert!(
            e.force.is_finite() && e.force >= THRESHOLD,
            "finite loud force"
        );
        assert!(e.point.is_finite(), "finite contact point");
        assert!(e.a < e.b, "canonical pair order");
    }
}

/// Without any opt-in threshold even hard impacts report nothing (and the
/// default threshold is infinity).
#[test]
fn force_opt_out_default_reports_nothing() {
    let mut engine = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    let ball = engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, 10.0, 0.0), 0.5, 1.0));
    assert!(
        !engine
            .get_body(ball)
            .expect("ball live")
            .contact_force_events_enabled(),
        "default body must not opt into force events"
    );
    assert_eq!(
        engine
            .get_body(ball)
            .expect("ball live")
            .contact_force_threshold,
        f32::INFINITY,
        "default threshold is infinity (disabled)"
    );
    let events = run(&mut engine, 180);
    assert!(events.is_empty(), "opt-out default must report nothing");
}

/// Drain order is canonical (sorted by pair) and reruns are identical.
#[test]
fn force_drain_order_is_deterministic() {
    let mut engine = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    // Three balls at staggered heights: several live pairs per step.
    for (i, h) in [3.0, 5.0, 8.0].into_iter().enumerate() {
        engine.add_body(
            RigidBody::new_sphere(Vec3::new(i as f32 * 1.5 - 1.5, h, 0.0), 0.4, 1.0)
                .with_contact_force_threshold(1000.0),
        );
    }
    let first = run(&mut engine, 120);
    assert!(!first.is_empty(), "staggered drops must report");
    let mut sorted = first.clone();
    sorted.sort_by_key(|e| (e.a, e.b));
    assert_eq!(
        first.iter().map(|e| (e.a, e.b)).collect::<Vec<_>>(),
        sorted.iter().map(|e| (e.a, e.b)).collect::<Vec<_>>(),
        "drain order must be canonical"
    );

    // Rebuild identically and rerun: the event stream must match exactly.
    let mut engine2 = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    engine2.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    for (i, h) in [3.0, 5.0, 8.0].into_iter().enumerate() {
        engine2.add_body(
            RigidBody::new_sphere(Vec3::new(i as f32 * 1.5 - 1.5, h, 0.0), 0.4, 1.0)
                .with_contact_force_threshold(1000.0),
        );
    }
    let second = run(&mut engine2, 120);
    assert_eq!(
        first.len(),
        second.len(),
        "rerun must report the same count"
    );
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!((a.a, a.b), (b.a, b.b), "same pairs");
        assert_eq!(
            a.force.to_bits(),
            b.force.to_bits(),
            "bit-identical forces across reruns"
        );
    }
}

/// Threshold setters sanitize: negatives clamp to zero (report
/// everything), non-finite input disables.
#[test]
fn force_threshold_setters_sanitize() {
    let mut body = RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0);
    body.set_contact_force_threshold(-3.0);
    assert_eq!(body.contact_force_threshold, 0.0);
    assert!(body.contact_force_events_enabled());
    body.set_contact_force_threshold(f32::NAN);
    assert!(!body.contact_force_events_enabled());
    assert_eq!(body.contact_force_threshold, f32::INFINITY);
    body.set_contact_force_threshold(250.0);
    assert!(body.contact_force_events_enabled());
    body.clear_contact_force_threshold();
    assert!(!body.contact_force_events_enabled());
}
