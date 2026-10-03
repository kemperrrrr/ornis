//! P5 shapes through the public engine API: compound narrowphase against
//! an equivalent box, rounded contact ahead of the inner surface,
//! box-on-halfspace rest, and the documented compound-vs-compound skip.

use glam::{Quat, Vec3};
use ornis_physics::engine::{Manifold, NarrowShardPool, SatCache, detect_collisions_into};
use ornis_physics::{PhysicsEngine, Pose, RigidBody, SequentialImpulseEngine, Shape};

/// Unit cube as two half-boxes (the compound narrowphase must agree with
/// the plain box on this split).
fn split_cube_body(position: Vec3, mass: f32) -> RigidBody {
    RigidBody::try_new_compound(
        position,
        vec![
            (
                Shape::Box {
                    half_extents: Vec3::new(0.5, 1.0, 1.0),
                },
                Pose::new(Vec3::new(-0.5, 0.0, 0.0), Quat::IDENTITY),
            ),
            (
                Shape::Box {
                    half_extents: Vec3::new(0.5, 1.0, 1.0),
                },
                Pose::new(Vec3::new(0.5, 0.0, 0.0), Quat::IDENTITY),
            ),
        ],
        mass,
    )
    .expect("split cube builds")
}

/// One-pair narrowphase through the engine's public detector.
fn collide_pair(a: RigidBody, b: RigidBody) -> Vec<Manifold> {
    let bodies = vec![a, b];
    let asleep = vec![false, false];
    let cache = SatCache::default();
    let mut out = Vec::new();
    let mut pool = NarrowShardPool::default();
    detect_collisions_into(
        &bodies,
        &[(0, 1)],
        &asleep,
        1.0 / 240.0,
        &mut out,
        None,
        0,
        Some(&cache),
        &mut pool,
    );
    out
}

#[test]
fn compound_narrow_matches_equivalent_box() {
    // Wide thin slab: the top face (y=-0.9) overlaps the cube bottom
    // (y=-1) by 0.1 in Y only, so the minimum-penetration axis is
    // unambiguously +Y for the plain box and for each half-box child
    // (a unit floor would tie X/Y on the children and report a side).
    let floor = RigidBody::new_box(Vec3::new(0.0, -1.4, 0.0), Vec3::new(5.0, 0.5, 5.0), 0.0);
    let plain = RigidBody::new_box(Vec3::ZERO, Vec3::ONE, 1.0);
    let compound = split_cube_body(Vec3::ZERO, 8.0);
    let plain_hit = collide_pair(floor.clone(), plain);
    let compound_hit = collide_pair(floor, compound);
    assert_eq!(plain_hit.len(), 1, "plain box must touch");
    assert_eq!(compound_hit.len(), 1, "compound cube must touch");
    for manifolds in [&plain_hit, &compound_hit] {
        let n = manifolds[0].normal;
        assert!(
            (n - Vec3::Y).length() < 1e-4,
            "contact normal must point up, got {n:?}"
        );
    }
}

#[test]
fn round_touches_a_full_radius_earlier() {
    // Box face at x=1, sphere surface at x=1.15: the inner gap (0.15)
    // exceeds the speculative margin (~0.05), so the plain pair stays
    // separated while the rounded pair (gap 0.15 − 0.2 < 0) contacts.
    let sphere = RigidBody::new_sphere(Vec3::new(1.4, 0.0, 0.0), 0.25, 1.0);
    let plain = RigidBody::new_box(Vec3::ZERO, Vec3::ONE, 1.0);
    // Inner gap: 1.4 - 0.25 - 1.0 = 0.15 > margin (~0.05): no contact.
    assert!(collide_pair(plain, sphere.clone()).is_empty());
    // Rounded by 0.2: gap 0.15 - 0.2 < 0: contact with the inner normal.
    let rounded = RigidBody::try_new_round(
        Vec3::ZERO,
        Shape::Box {
            half_extents: Vec3::ONE,
        },
        0.2,
        1.0,
    )
    .expect("round builds");
    let hits = collide_pair(rounded, sphere);
    assert_eq!(hits.len(), 1, "rounded box must touch first");
    assert!(
        (hits[0].normal - Vec3::X).length() < 1e-4,
        "rounded normal is the inner normal, got {:?}",
        hits[0].normal
    );
}

#[test]
fn box_rests_on_halfspace_floor() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(
        RigidBody::try_new_halfspace(Vec3::ZERO, Vec3::Y, 0.0).expect("static plane builds"),
    );
    let h = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    for _ in 0..240 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(h).expect("box survives");
    assert!(
        (b.position.y - 0.5).abs() < 0.15,
        "box must rest on the plane, got {}",
        b.position.y
    );
    assert!(
        b.position.y.is_finite() && b.velocity.length() < 1.0,
        "rest must be quiet: {:?} {:?}",
        b.position,
        b.velocity
    );
}

#[test]
fn compound_vs_compound_reports_no_contact() {
    // Deeply overlapping yet silent: concave-concave is undefined (the loud
    // marker is `Shape::pair_support`; this pins the solver behavior).
    let a = split_cube_body(Vec3::ZERO, 1.0);
    let b = split_cube_body(Vec3::ZERO, 1.0);
    assert!(a.shape.pair_support(&b.shape) == ornis_physics::PairSupport::UnsupportedPair);
    assert!(collide_pair(a, b).is_empty());
}
