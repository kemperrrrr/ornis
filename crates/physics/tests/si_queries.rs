//! SI query/event pins: filters, contact/trigger events, ray- and shapecasts.

use glam::{Quat, Vec3};
use ornis_physics::trigger::{ContactEventKind, TriggerEventKind};
use ornis_physics::{
    BodyHandle, ContactEvent, PhysicsEngine, Ray, RigidBody, SequentialImpulseEngine, Shape,
    TriggerEvent,
};

#[test]
fn collision_filter_blocks_broadphase_and_narrowphase() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let a = physics.add_body(
        RigidBody::new_sphere(Vec3::new(-0.4, 0.0, 0.0), 0.5, 1.0)
            .with_collision_filter(0b0001, 0b0010),
    );
    let b = physics.add_body(
        RigidBody::new_sphere(Vec3::new(0.4, 0.0, 0.0), 0.5, 1.0)
            .with_collision_filter(0b0010, 0b0100),
    );

    physics.step(1.0 / 60.0);

    assert_eq!(physics.debug_contact_count(a), 0);
    assert_eq!(physics.debug_contact_count(b), 0);
    assert_eq!(physics.get_body(a).unwrap().position.x, -0.4);
    assert_eq!(physics.get_body(b).unwrap().position.x, 0.4);
}

#[test]
fn collision_filter_allows_mutual_layer_match() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let a = physics.add_body(
        RigidBody::new_sphere(Vec3::new(-0.4, 0.0, 0.0), 0.5, 1.0)
            .with_collision_filter(0b0001, 0b0010),
    );
    let b = physics.add_body(
        RigidBody::new_sphere(Vec3::new(0.4, 0.0, 0.0), 0.5, 1.0)
            .with_collision_filter(0b0010, 0b0001),
    );

    physics.step(1.0 / 60.0);

    assert!(physics.debug_contact_count(a) > 0);
    assert!(physics.debug_contact_count(b) > 0);
}

#[test]
fn collision_filter_applies_to_continuous_cast() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(
        RigidBody::new_box(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.05, 10.0), 0.0)
            .with_collision_filter(0b0010, 0b0010),
    );
    let bullet = physics.add_body(
        RigidBody::new_sphere(Vec3::new(0.0, 3.0, 0.0), 0.1, 1.0)
            .with_collision_filter(0b0001, 0b0001),
    );
    physics.get_body_mut(bullet).unwrap().velocity = Vec3::new(0.0, -80.0, 0.0);

    for _ in 0..60 {
        physics.step(1.0 / 60.0);
    }

    assert_eq!(physics.debug_contact_count(bullet), 0);
    assert!(
        physics.get_body(bullet).unwrap().position.y < -0.1,
        "filtered bullet should pass through the floor"
    );
}

/// Contact events: a dropped box begins touching the floor on impact.
#[test]
fn contact_begin_fires_on_touch() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let floor = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let klein = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    let mut begun = false;
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
        for e in physics.drain_contact_events() {
            if matches!(e.kind, ContactEventKind::Begin)
                && ((e.body_a == floor && e.body_b == klein)
                    || (e.body_a == klein && e.body_b == floor))
            {
                begun = true;
            }
        }
    }
    assert!(begun, "touchdown must emit Begin");
}

/// Contact events: a speculative near-miss (gap inside the margin, no
/// touch) emits nothing — gameplay must not see begins without contact.
#[test]
fn contact_no_begin_for_speculative_gap() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    // Hovering 2 cm above the floor: inside the 5 cm speculative margin
    // (manifolds exist), zero velocity, zero gravity — never touches.
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.52, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    for _ in 0..30 {
        physics.step(1.0 / 60.0);
        let events = physics.drain_contact_events();
        assert!(
            events.is_empty(),
            "gap contact must stay silent, got {events:?}"
        );
    }
}

/// Contact events: a fast impact records a Hit with the approach speed.
#[test]
fn contact_hit_reports_approach_speed() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let mut drop = RigidBody::new_box(Vec3::new(0.0, 6.0, 0.0), Vec3::splat(0.5), 1.0);
    drop.velocity = Vec3::new(0.0, -20.0, 0.0);
    physics.add_body(drop);
    let mut hit_speed = 0.0f32;
    for _ in 0..60 {
        physics.step(1.0 / 60.0);
        for e in physics.drain_contact_events() {
            if let ContactEventKind::Hit {
                approach_speed,
                normal,
                ..
            } = e.kind
            {
                hit_speed = hit_speed.max(approach_speed);
                assert!(
                    normal.y.abs() > 0.9,
                    "hit normal must be vertical, got {normal:?}"
                );
            }
        }
    }
    assert!(
        hit_speed > 10.0,
        "20 m/s impact must record a Hit, max approach {hit_speed}"
    );
}

/// Contact events: launching a resting box off the floor emits End.
#[test]
fn contact_end_fires_on_separation() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let floor = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let klein = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    for _ in 0..90 {
        physics.step(1.0 / 60.0);
    }
    let _ = physics.drain_contact_events();
    physics.get_body_mut(klein).unwrap().velocity = Vec3::new(0.0, 10.0, 0.0);
    // Wake it: the test sets velocity directly (a driver would too).
    physics.wake_island(klein.index());
    let mut ended = false;
    for _ in 0..60 {
        physics.step(1.0 / 60.0);
        for e in physics.drain_contact_events() {
            if matches!(e.kind, ContactEventKind::End)
                && ((e.body_a == floor && e.body_b == klein)
                    || (e.body_a == klein && e.body_b == floor))
            {
                ended = true;
            }
        }
    }
    assert!(ended, "liftoff must emit End");
}

/// Contact events: a frozen (sleeping) contact emits no churn — sleep
/// retains touch state silently.
#[test]
fn contact_frozen_pair_emits_no_churn() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let klein = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    for _ in 0..150 {
        physics.step(1.0 / 60.0);
    }
    assert!(physics.is_asleep(klein), "box must settle");
    let _ = physics.drain_contact_events();
    for _ in 0..60 {
        physics.step(1.0 / 60.0);
        let events = physics.drain_contact_events();
        assert!(
            events.is_empty(),
            "frozen contact must stay silent, got {events:?}"
        );
    }
}

/// Contact events are deterministic run-to-run on identical scenes.
#[test]
fn contact_events_deterministic_across_runs() {
    fn run() -> Vec<ContactEvent> {
        let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(5.0, 1.0, 5.0),
            0.0,
        ));
        let mut drop = RigidBody::new_box(Vec3::new(0.5, 5.0, 0.0), Vec3::splat(0.4), 1.0);
        drop.velocity = Vec3::new(-2.0, -15.0, 1.0);
        physics.add_body(drop);
        let mut all = Vec::new();
        for _ in 0..90 {
            physics.step(1.0 / 60.0);
            all.extend(physics.drain_contact_events());
        }
        all
    }
    assert_eq!(run(), run(), "contact events must be run-deterministic");
}

#[test]
fn trigger_emits_enter_and_exit_without_solving_contact() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let mut trigger_body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 0.0);
    trigger_body.set_trigger(true);
    let trigger = physics.add_body(trigger_body);
    let mover = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.8, 0.0), 0.5, 1.0));

    physics.step(1.0 / 60.0);
    assert_eq!(
        physics.drain_trigger_events(),
        vec![TriggerEvent {
            body_a: trigger.min(mover),
            body_b: trigger.max(mover),
            kind: TriggerEventKind::Entered,
        }]
    );
    assert_eq!(physics.debug_contact_count(mover), 0);
    assert_eq!(
        physics.get_body(mover).unwrap().position,
        Vec3::new(0.0, 0.8, 0.0)
    );

    physics.step(1.0 / 60.0);
    assert!(physics.drain_trigger_events().is_empty());

    physics.get_body_mut(mover).unwrap().position = Vec3::new(0.0, 3.0, 0.0);
    physics.step(1.0 / 60.0);
    assert_eq!(
        physics.drain_trigger_events(),
        vec![TriggerEvent {
            body_a: trigger.min(mover),
            body_b: trigger.max(mover),
            kind: TriggerEventKind::Exited,
        }]
    );
}

#[test]
fn removing_trigger_body_queues_exit_and_clears_pair_state() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let mut trigger_body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 0.0);
    trigger_body.set_trigger(true);
    let trigger = physics.add_body(trigger_body);
    let first = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0));
    let second = physics.add_body(RigidBody::new_sphere(Vec3::new(5.0, 0.0, 0.0), 0.5, 1.0));
    physics.step(1.0 / 60.0);
    assert_eq!(physics.drain_trigger_events().len(), 1);

    physics.remove_body(trigger);
    assert_eq!(
        physics.drain_trigger_events(),
        vec![TriggerEvent {
            body_a: trigger,
            body_b: first,
            kind: TriggerEventKind::Exited,
        }]
    );
    physics
        .get_body_mut(BodyHandle::from(second.index() - 1))
        .unwrap()
        .position = Vec3::ZERO;
    physics.step(1.0 / 60.0);
    let events = physics.drain_trigger_events();
    assert!(events.is_empty());
}

#[test]
fn raycast_hits_sphere() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0, 1.0));
    let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
    let hit = physics.raycast(ray, 10.0).expect("valid query");
    assert!(hit.is_some());
    let hit = hit.unwrap();
    assert!((hit.distance - 4.0).abs() < 0.01);
}

#[test]
fn raycast_obb_uses_exact_surface_and_normal() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
    physics.add_body(
        RigidBody::new_box(Vec3::ZERO, Vec3::new(1.0, 0.25, 0.25), 0.0).with_orientation(rotation),
    );

    let ray = Ray::new(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
    let hit = physics
        .raycast(ray, 10.0)
        .expect("valid query")
        .expect("ray must hit the rotated box");
    let expected_distance = 2.0 - 0.25 * std::f32::consts::SQRT_2;
    let expected_normal = rotation * Vec3::Y;
    assert!((hit.distance - expected_distance).abs() < 1e-4);
    assert!(hit.normal.dot(expected_normal) > 0.999);
    assert!((hit.point - ray.point_at(expected_distance)).length() < 1e-4);
}

#[test]
fn raycast_capsule_uses_spherical_cap_normal() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_capsule(Vec3::ZERO, 0.5, 1.0, 0.0));

    let ray = Ray::new(Vec3::new(0.4, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
    let hit = physics
        .raycast(ray, 10.0)
        .expect("valid query")
        .expect("ray must hit the capsule cap");
    let expected_distance = 2.0 - (1.0 + 0.3);
    let expected_normal = Vec3::new(0.8, 0.6, 0.0);
    assert!((hit.distance - expected_distance).abs() < 1e-4);
    assert!(hit.normal.dot(expected_normal) > 0.999);
    assert!((hit.point - Vec3::new(0.4, 1.3, 0.0)).length() < 1e-4);
}

#[test]
fn raycast_ignores_zero_length_rays() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 1.0, 0.0));
    // Degenerate direction is a typed error, never a hit.
    assert!(
        physics
            .raycast(Ray::new(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO), 10.0)
            .unwrap_or(None)
            .is_none()
    );
}

#[test]
fn shapecast_hits_body() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0, 0.0));
    // Cast a small sphere from origin toward the static target.
    let shape = Shape::Sphere { radius: 0.1 };
    let hit = physics.shapecast(&shape, Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0));
    assert!(hit.is_some(), "conservative shapecast should hit");
    let hit = hit.unwrap();
    assert_eq!(hit.handle, BodyHandle::from_raw(0));
    assert!(
        hit.distance > 3.0 && hit.distance < 10.0,
        "hit distance={}",
        hit.distance
    );
}

#[test]
fn shapecast_exact_hit_distance() {
    // Sphere r=0.5 cast straight down onto a half-1 box at the origin:
    // contact when the sphere center is 1.5 above the origin, so a cast
    // from y=5 must report a hit distance of exactly 3.5 (G6: the cast
    // uses analytic shape distances, not a sampled march).
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 0.0));
    let shape = Shape::Sphere { radius: 0.5 };
    let hit = physics
        .shapecast(&shape, Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO)
        .expect("cast straight down must hit the box");
    assert!(
        (hit.distance - 3.5).abs() < 1e-2,
        "hit distance={} expected 3.5",
        hit.distance
    );
    // Surface normal at the hit points up, toward the caster.
    assert!(hit.normal.y > 0.99, "normal={:?}", hit.normal);
}

#[test]
fn shapecast_thin_wall_no_tunnel() {
    // A 4 cm wall is far thinner than the cast segment: a sampled march
    // would step over it, conservative advancement must not (G6).
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_box(
        Vec3::ZERO,
        Vec3::new(2.0, 2.0, 0.02),
        0.0,
    ));
    let shape = Shape::Sphere { radius: 0.1 };
    let hit = physics.shapecast(&shape, Vec3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 2.0));
    let hit = hit.expect("cast through the thin wall must hit, not tunnel");
    // Sphere surface touches the wall face at z = -0.02 - 0.1 = -0.12,
    // i.e. 1.88 into the 4-unit cast.
    assert!(
        (hit.distance - 1.88).abs() < 1e-2,
        "hit distance={} expected 1.88",
        hit.distance
    );
}

/// Raycast hits a mesh triangle at its surface through the BVH walk.
#[test]
fn raycast_hits_trimesh() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_trimesh(
        Vec3::ZERO,
        &[
            Vec3::new(-5.0, 0.0, -5.0),
            Vec3::new(5.0, 0.0, 5.0),
            Vec3::new(5.0, 0.0, -5.0),
            Vec3::new(-5.0, 0.0, 5.0),
        ],
        &[[0, 1, 2], [0, 3, 1]],
        0.0,
    ));
    let hit = physics
        .raycast(Ray::new(Vec3::new(1.0, 4.0, 2.0), Vec3::NEG_Y), 10.0)
        .expect("valid query")
        .expect("ray must hit the mesh quad");
    assert!((hit.distance - 4.0).abs() < 1e-4, "got {}", hit.distance);
    assert!(hit.normal.dot(Vec3::Y) > 0.999);
}

/// Raycast hits the new shapes at their exact surfaces: cylinder wall,
/// cone wall, hull face, heightfield column top.
#[test]
fn raycast_hits_cylinder_cone_hull_heightfield() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_cylinder(Vec3::ZERO, 1.0, 1.0, 0.0));
    let hit = physics
        .raycast(Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z), 10.0)
        .expect("valid query")
        .expect("ray must hit the cylinder wall");
    assert!((hit.distance - 4.0).abs() < 1e-4, "got {}", hit.distance);
    assert!(hit.normal.dot(Vec3::NEG_Z) > 0.999);

    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_cone(Vec3::ZERO, 1.0, 1.0, 0.0));
    // Cone wall at y=0 has radius 0.5: ray along +X from x=-5.
    let hit = physics
        .raycast(Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X), 10.0)
        .expect("valid query")
        .expect("ray must hit the cone wall");
    assert!((hit.distance - 4.5).abs() < 1e-4, "got {}", hit.distance);

    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_convex_hull(
        Vec3::ZERO,
        vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ],
        0.0,
    ));
    let hit = physics
        .raycast(Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z), 10.0)
        .expect("valid query")
        .expect("ray must hit the hull face");
    assert!((hit.distance - 4.0).abs() < 1e-4, "got {}", hit.distance);
    assert!(hit.normal.dot(Vec3::NEG_Z) > 0.999);

    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.add_body(RigidBody::new_heightfield(
        Vec3::ZERO,
        vec![2.0f32; 9],
        3,
        3,
        1.0,
        0.0,
    ));
    let hit = physics
        .raycast(Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y), 10.0)
        .expect("valid query")
        .expect("ray must hit the heightfield top");
    assert!((hit.distance - 3.0).abs() < 1e-4, "got {}", hit.distance);
    assert!(hit.normal.dot(Vec3::Y) > 0.999);
}
