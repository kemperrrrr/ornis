//! World snapshots (P8): RON round-trip with bit-identical continuation,
//! version rejection, invalid-data errors and unsupported-path refusal.
//!
//! The continuation scene exercises contacts, a warm-started joint, sleep
//! timers and opted-in force thresholds mid-flight: snapshot at step 90
//! (bodies settled, some asleep, warm caches populated), restore into a
//! fresh engine through serialized RON, continue 60 steps on both, and
//! require exact float-bit equality.

use glam::Vec3;

use ornis_physics::{
    Engine, JointKind, PhysicsEngine, RigidBody, RoutingKind, SnapshotError, SolverKind,
    WorldSnapshot,
};

/// Floor, two stacked boxes (the lower one lands early and sleeps), a
/// ball-jointed pair and a force-tracked drop weight.
fn scene() -> Engine {
    let mut engine = Engine::new(SolverKind::SequentialImpulse, Vec3::new(0.0, -9.81, 0.0));
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    let settled = engine.add_body(RigidBody::new_box(
        Vec3::new(-2.0, 1.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    let _ = settled;
    let a = engine.add_body(RigidBody::new_box(
        Vec3::new(2.0, 3.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    let b = engine.add_body(RigidBody::new_box(
        Vec3::new(2.0, 4.5, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    engine
        .add_joint(
            a,
            b,
            JointKind::Ball {
                local_anchor_a: Vec3::new(0.0, 0.5, 0.0),
                local_anchor_b: Vec3::new(0.0, -0.5, 0.0),
            },
        )
        .expect("valid ball joint");
    engine.add_body(
        RigidBody::new_sphere(Vec3::new(-2.0, 8.0, 0.0), 0.5, 1.0)
            .with_contact_force_threshold(5000.0),
    );
    engine
}

/// Every float field of a body as raw bits, plus discrete state.
fn body_bits(engine: &Engine, steps: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for h in 0..engine.body_count() {
        let b = engine
            .get_body(ornis_physics::BodyHandle::from(h))
            .expect("body live");
        out.extend(b.position.to_array().map(f32::to_bits));
        out.extend(
            [
                b.orientation.x,
                b.orientation.y,
                b.orientation.z,
                b.orientation.w,
            ]
            .map(f32::to_bits),
        );
        out.extend(b.velocity.to_array().map(f32::to_bits));
        out.extend(b.angular_velocity.to_array().map(f32::to_bits));
        out.extend(
            [
                b.mass,
                b.inv_mass,
                b.torque.x,
                b.torque.y,
                b.torque.z,
                b.restitution,
                b.friction,
            ]
            .map(f32::to_bits),
        );
        out.extend(b.inertia.to_array().map(f32::to_bits));
        out.push(match b.body_type {
            ornis_physics::BodyType::Static => 0,
            ornis_physics::BodyType::Dynamic => 1,
            ornis_physics::BodyType::Kinematic => 2,
        });
        out.push(steps);
    }
    out
}

/// Snapshot mid-flight → RON → parse → fresh engine → identical
/// continuation, compared bit-for-bit (bodies and full snapshots).
#[test]
fn snapshot_round_trip_continues_bit_identical() {
    let mut original = scene();
    for _ in 0..90 {
        original.step(1.0 / 60.0);
    }
    let drained_contacts = original.drain_contact_events();
    let drained_forces = original.drain_contact_force_events();
    let snap = original.world_snapshot().expect("captures");
    let text = snap.to_ron().expect("serializes");
    let back = WorldSnapshot::from_ron(&text).expect("parses");

    let mut restored = Engine::new(SolverKind::SequentialImpulse, Vec3::new(0.0, -9.81, 0.0));
    restored.restore_world_snapshot(&back).expect("restores");

    for _ in 0..60 {
        original.step(1.0 / 60.0);
        restored.step(1.0 / 60.0);
    }
    assert_eq!(
        body_bits(&original, 150),
        body_bits(&restored, 150),
        "continued bodies must be bit-identical"
    );
    assert_eq!(
        original.world_snapshot().expect("captures"),
        restored.world_snapshot().expect("captures"),
        "continued snapshots must compare equal"
    );
    assert_eq!(
        original.drain_contact_events(),
        restored.drain_contact_events(),
        "continued contact streams must match"
    );
    assert_eq!(
        original.drain_contact_force_events(),
        restored.drain_contact_force_events(),
        "continued force streams must match"
    );
    // Sanity: the pre-snapshot drains above actually observed the scene
    // (impacts happened — the test is not vacuous).
    assert!(
        !drained_contacts.is_empty() || !drained_forces.is_empty(),
        "mid-flight scene must emit contact or force events"
    );
}

/// A forged version is refused with a typed mismatch naming both sides.
#[test]
fn snapshot_forged_version_is_refused() {
    let mut engine = scene();
    for _ in 0..10 {
        engine.step(1.0 / 60.0);
    }
    let text = engine
        .world_snapshot()
        .expect("captures")
        .to_ron()
        .expect("ron");
    let forged = text.replacen("version:1", "version:999", 1);
    assert_ne!(forged, text, "forgery must change the payload");
    match WorldSnapshot::from_ron(&forged) {
        Err(SnapshotError::VersionMismatch { expected, found }) => {
            assert_eq!(expected, ornis_physics::WORLD_SNAPSHOT_VERSION);
            assert_eq!(found, 999);
        }
        other => panic!("expected VersionMismatch, got {other:?}"),
    }
}

/// Dangling joint references fail loudly instead of restoring garbage.
#[test]
fn snapshot_dangling_joint_is_invalid() {
    let mut engine = scene();
    for _ in 0..10 {
        engine.step(1.0 / 60.0);
    }
    let mut snap = engine.world_snapshot().expect("captures");
    snap.joints[0].a = 9999;
    assert!(
        matches!(
            snap.instantiate_si(),
            Err(SnapshotError::InvalidData { .. })
        ),
        "dangling joint references must fail loudly"
    );
}

/// Non-SI solvers and Islands routing are refused loudly (follow-ups,
/// not silent loss).
#[test]
fn snapshot_unsupported_paths_are_refused() {
    let avbd = Engine::new(SolverKind::Avbd, Vec3::new(0.0, -9.81, 0.0));
    assert!(matches!(
        avbd.world_snapshot(),
        Err(SnapshotError::Unsupported { .. })
    ));
    let mut islands = scene();
    islands.set_routing(RoutingKind::Islands);
    assert!(matches!(
        islands.world_snapshot(),
        Err(SnapshotError::Unsupported { .. })
    ));
}
