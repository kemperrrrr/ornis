//! R5 pins: tree-accelerated scene queries match the brute-force
//! [`QueryPipeline`](ornis_physics::QueryPipeline) bit-for-bit.
//!
//! The engine serves `query_*` from the live broadphase tree when it is
//! tree-backed and fresh, and from brute force otherwise; every test below
//! compares the engine answer against the brute-force oracle over the same
//! body snapshot, so the tree may only prune — never decide a hit.

use glam::{Quat, Vec3};
use ornis_physics::{
    AABB, BodyHandle, BroadPhaseKind, PhysicsEngine, QueryFilter, QueryPipeline, Ray, RigidBody,
    SequentialImpulseEngine, Shape,
};

/// Eight-shape scene (mirrors `tests/query_pipeline.rs`) plus a compound
/// body, on the tree backend with one step so the live tree is built.
struct TreeScene {
    engine: SequentialImpulseEngine,
    sphere: BodyHandle,
    cuboid: BodyHandle,
    capsule: BodyHandle,
    _cylinder: BodyHandle,
    _cone: BodyHandle,
    hull: BodyHandle,
    _terrain: BodyHandle,
    _mesh: BodyHandle,
    compound: BodyHandle,
}

fn tree_scene() -> TreeScene {
    let mut engine = SequentialImpulseEngine::new(Vec3::ZERO);
    engine.set_broadphase(BroadPhaseKind::DynamicAabbTree);
    let layer = |index: u32| (1u32 << index, u32::MAX);
    let (sphere_layer, sphere_mask) = layer(0);
    let (box_layer, box_mask) = layer(1);
    let (capsule_layer, capsule_mask) = layer(2);
    let (cylinder_layer, cylinder_mask) = layer(3);
    let (cone_layer, cone_mask) = layer(4);
    let (hull_layer, hull_mask) = layer(5);
    let (hf_layer, hf_mask) = layer(6);
    let (mesh_layer, mesh_mask) = layer(7);

    let sphere = engine.add_body(
        RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0)
            .with_collision_filter(sphere_layer, sphere_mask),
    );
    let cuboid = engine.add_body(
        RigidBody::new_box(Vec3::new(4.0, 0.0, 0.0), Vec3::splat(0.5), 0.0)
            .with_collision_filter(box_layer, box_mask),
    );
    let capsule = engine.add_body(
        RigidBody::new_capsule(Vec3::new(-4.0, 0.0, 0.0), 0.3, 0.5, 1.0)
            .with_collision_filter(capsule_layer, capsule_mask)
            .with_trigger(true),
    );
    let mut driver = RigidBody::new_cylinder(Vec3::new(0.0, 0.0, 4.0), 0.5, 0.5, 1.0);
    driver.body_type = ornis_physics::BodyType::Kinematic;
    driver.set_collision_filter(cylinder_layer, cylinder_mask);
    let cylinder = engine.add_body(driver);
    let cone = engine.add_body(
        RigidBody::new_cone(Vec3::new(0.0, 0.0, -4.0), 0.5, 1.0, 1.0)
            .with_collision_filter(cone_layer, cone_mask),
    );
    let hull = engine.add_body(
        RigidBody::try_new_convex_hull(
            Vec3::new(8.0, -0.25, -0.25),
            vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
            1.0,
        )
        .expect("tetra hull builds")
        .with_collision_filter(hull_layer, hull_mask),
    );
    let terrain = engine.add_body(
        RigidBody::try_new_heightfield(Vec3::new(-8.0, 0.0, 5.0), vec![0.0; 9], 3, 3, 1.0, 0.0)
            .expect("flat heightfield builds")
            .with_collision_filter(hf_layer, hf_mask),
    );
    let mesh = engine.add_body(
        RigidBody::try_new_trimesh(
            Vec3::new(0.0, 0.0, 8.0),
            &[
                Vec3::new(-1.0, 0.0, -1.0),
                Vec3::new(1.0, 0.0, -1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(-1.0, 0.0, 1.0),
            ],
            &[
                ornis_physics::Triangle::from_raw([0, 2, 1]),
                ornis_physics::Triangle::from_raw([0, 3, 2]),
            ],
            0.0,
        )
        .expect("quad mesh builds")
        .with_collision_filter(mesh_layer, mesh_mask),
    );
    let compound_shape = Shape::try_compound(vec![
        (
            Shape::Box {
                half_extents: Vec3::splat(0.5),
            },
            ornis_physics::Pose::new(Vec3::new(-12.0, 0.0, 0.0), Quat::IDENTITY),
        ),
        (
            Shape::Sphere { radius: 0.5 },
            ornis_physics::Pose::new(Vec3::new(-10.5, 0.0, 0.0), Quat::IDENTITY),
        ),
    ])
    .expect("compound builds");
    let compound = engine.add_body(RigidBody {
        shape: compound_shape,
        ..RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0)
    });
    // One step builds the live tree at the current poses (zero gravity, so
    // nothing moves — the snapshot the oracle reads is the tree's own).
    engine.step(1.0 / 60.0);
    TreeScene {
        engine,
        sphere,
        cuboid,
        capsule,
        _cylinder: cylinder,
        _cone: cone,
        hull,
        _terrain: terrain,
        _mesh: mesh,
        compound,
    }
}

/// Ray down +X at y = z = 0: compound sphere, capsule sensor, sphere,
/// static box, hull tetra, in that order.
fn x_ray() -> Ray {
    Ray::new(Vec3::new(-14.0, 0.0, 0.0), Vec3::X)
}

fn assert_hit_eq(
    tree: Option<ornis_physics::RaycastHit>,
    brute: Option<ornis_physics::RaycastHit>,
) {
    match (tree, brute) {
        (None, None) => {}
        (Some(t), Some(b)) => {
            assert_eq!(t.handle, b.handle);
            assert_eq!(t.distance, b.distance, "distance bits differ");
            assert_eq!(t.point, b.point, "point bits differ");
            assert_eq!(t.normal, b.normal, "normal bits differ");
        }
        (t, b) => panic!("hit mismatch: tree {t:?} vs brute {b:?}"),
    }
}

#[test]
fn tree_cast_ray_matches_brute_bitwise() {
    let scene = tree_scene();
    let brute = QueryPipeline::new();
    for filter in [
        QueryFilter::default(),
        QueryFilter {
            exclude_sensors: true,
            ..QueryFilter::default()
        },
        QueryFilter {
            exclude_dynamic: true,
            ..QueryFilter::default()
        },
        QueryFilter {
            mask: 0b0010,
            ..QueryFilter::default()
        },
    ] {
        let tree_hit = scene
            .engine
            .query_cast_ray(&x_ray(), 60.0, &filter)
            .expect("valid ray");
        let brute_hit = brute
            .cast_ray(&scene.engine.bodies, &x_ray(), 60.0, &filter)
            .expect("valid ray");
        assert_hit_eq(tree_hit, brute_hit);
    }
    // The compound sphere opens the walk: the tree must find it first.
    assert_eq!(
        scene
            .engine
            .query_cast_ray(&x_ray(), 60.0, &QueryFilter::default())
            .expect("valid ray")
            .expect("hit")
            .handle,
        scene.compound
    );
}

#[test]
fn tree_intersect_ray_matches_brute_in_order() {
    let scene = tree_scene();
    let brute = QueryPipeline::new();
    let filter = QueryFilter::default();
    let tree_hits = scene
        .engine
        .query_intersect_ray(&x_ray(), 60.0, &filter)
        .expect("valid ray");
    let brute_hits = brute
        .intersect_ray(&scene.engine.bodies, &x_ray(), 60.0, &filter)
        .expect("valid ray");
    assert_eq!(tree_hits.len(), brute_hits.len(), "hit count differs");
    for (t, b) in tree_hits.iter().zip(brute_hits.iter()) {
        assert_eq!(t.handle, b.handle, "hit order differs");
        assert_eq!(t.distance, b.distance);
        assert_eq!(t.point, b.point);
        assert_eq!(t.normal, b.normal);
    }
    // Handle order pins the walk: compound, capsule, sphere, box, hull.
    let handles: Vec<BodyHandle> = tree_hits.iter().map(|h| h.handle).collect();
    assert_eq!(
        handles,
        vec![
            scene.compound,
            scene.capsule,
            scene.sphere,
            scene.cuboid,
            scene.hull
        ]
    );
}

#[test]
fn tree_point_aabb_projection_match_brute() {
    let scene = tree_scene();
    let brute = QueryPipeline::new();
    let filter = QueryFilter::default();
    for point in [
        Vec3::ZERO,
        Vec3::new(4.0, 0.5, 0.0),
        Vec3::new(-12.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::splat(100.0),
    ] {
        assert_eq!(
            scene.engine.query_intersect_point(point, &filter),
            brute.intersect_point(&scene.engine.bodies, point, &filter),
            "point query differs at {point:?}"
        );
        let tree_projection = scene.engine.query_project_point(point, &filter);
        let brute_projection = brute.project_point(&scene.engine.bodies, point, &filter);
        match (tree_projection, brute_projection) {
            (None, None) => {}
            (Some(t), Some(b)) => {
                assert_eq!(t.handle, b.handle, "projection body differs");
                assert_eq!(t.point, b.point, "projection point bits differ");
                assert_eq!(t.is_inside, b.is_inside);
                assert_eq!(t.feature_id, b.feature_id);
            }
            (t, b) => panic!("projection mismatch at {point:?}: {t:?} vs {b:?}"),
        }
    }
    for query in [
        AABB::new(Vec3::splat(-14.0), Vec3::splat(10.0)),
        AABB::new(Vec3::splat(-1.0), Vec3::splat(1.0)),
        AABB::new(Vec3::new(-13.0, -1.0, -1.0), Vec3::new(-11.0, 1.0, 1.0)),
    ] {
        assert_eq!(
            scene.engine.query_intersect_aabb(&query, &filter),
            brute.intersect_aabb(&scene.engine.bodies, &query, &filter),
            "aabb query differs at {query:?}"
        );
    }
}

#[test]
fn tree_shape_casts_match_brute() {
    let scene = tree_scene();
    let brute = QueryPipeline::new();
    let filter = QueryFilter::default();
    let mover = Shape::Sphere { radius: 0.25 };
    for displacement in [
        Vec3::new(20.0, 0.0, 0.0),
        Vec3::new(-4.0, 0.0, 0.0),
        Vec3::new(0.0, 5.0, 0.0),
    ] {
        let tree_hit = scene.engine.query_cast_shape(
            &mover,
            Vec3::new(-14.0, 0.0, 0.0),
            Quat::IDENTITY,
            displacement,
            &filter,
        );
        let brute_hit = brute.cast_shape(
            &scene.engine.bodies,
            &mover,
            Vec3::new(-14.0, 0.0, 0.0),
            Quat::IDENTITY,
            displacement,
            &filter,
        );
        assert_hit_eq(tree_hit, brute_hit);
    }
    for (shape, position) in [
        (Shape::Sphere { radius: 0.5 }, Vec3::ZERO),
        (Shape::Sphere { radius: 0.5 }, Vec3::new(-12.0, 0.0, 0.0)),
        (
            Shape::Box {
                half_extents: Vec3::splat(0.5),
            },
            Vec3::new(4.0, 0.0, 0.0),
        ),
        (Shape::Sphere { radius: 0.5 }, Vec3::splat(100.0)),
    ] {
        assert_eq!(
            scene
                .engine
                .query_intersect_shape(&shape, position, Quat::IDENTITY, &filter),
            brute.intersect_shape(
                &scene.engine.bodies,
                &shape,
                position,
                Quat::IDENTITY,
                &filter
            ),
            "shape overlap differs at {position:?}"
        );
    }
}

/// A ray sliding past the shape but inside its fat margin must visit the
/// candidate and still miss — no false hits — while a ray just inside the
/// surface must hit at the same distance — no misses.
#[test]
fn fat_margin_grazing_ray_has_no_false_hits_or_misses() {
    let mut engine = SequentialImpulseEngine::new(Vec3::ZERO);
    engine.set_broadphase(BroadPhaseKind::DynamicAabbTree);
    let target = engine.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 0.0));
    engine.step(1.0 / 60.0);
    let brute = QueryPipeline::new();
    let filter = QueryFilter::default();
    // y = 0.55: outside the 0.5 box, inside the ~0.6 fat box.
    let graze = Ray::new(Vec3::new(-10.0, 0.55, 0.0), Vec3::X);
    assert!(
        brute
            .cast_ray(&engine.bodies, &graze, 50.0, &filter)
            .expect("valid ray")
            .is_none()
    );
    assert!(
        engine
            .query_cast_ray(&graze, 50.0, &filter)
            .expect("valid ray")
            .is_none(),
        "fat-margin graze must not become a hit"
    );
    // y = 0.45: inside the surface — the same hit both ways.
    let clip = Ray::new(Vec3::new(-10.0, 0.45, 0.0), Vec3::X);
    assert_hit_eq(
        engine
            .query_cast_ray(&clip, 50.0, &filter)
            .expect("valid ray"),
        brute
            .cast_ray(&engine.bodies, &clip, 50.0, &filter)
            .expect("valid ray"),
    );
    assert_eq!(
        engine
            .query_cast_ray(&clip, 50.0, &filter)
            .expect("valid ray")
            .expect("hit")
            .handle,
        target
    );
    // AABB covering only the fat shell (no base overlap) stays empty.
    let shell = AABB::new(Vec3::new(-1.0, 0.55, -1.0), Vec3::new(1.0, 0.7, 1.0));
    assert!(
        brute
            .intersect_aabb(&engine.bodies, &shell, &filter)
            .is_empty()
    );
    assert!(engine.query_intersect_aabb(&shell, &filter).is_empty());
}

/// Empty and stale trees take the explicit brute-force fallback and agree
/// with the oracle instead of panicking or missing.
#[test]
fn empty_and_stale_trees_fall_back_to_brute() {
    // Empty engine, tree never built: every query is trivially empty.
    let mut engine = SequentialImpulseEngine::new(Vec3::ZERO);
    engine.set_broadphase(BroadPhaseKind::DynamicAabbTree);
    let filter = QueryFilter::default();
    assert!(
        engine
            .query_cast_ray(&x_ray(), 60.0, &filter)
            .expect("valid ray")
            .is_none()
    );
    assert!(
        engine
            .query_intersect_ray(&x_ray(), 60.0, &filter)
            .expect("valid ray")
            .is_empty()
    );
    assert!(engine.query_intersect_point(Vec3::ZERO, &filter).is_empty());
    assert!(
        engine
            .query_intersect_aabb(&AABB::new(Vec3::splat(-1.0), Vec3::splat(1.0)), &filter)
            .is_empty()
    );
    assert!(engine.query_project_point(Vec3::ZERO, &filter).is_none());
    assert!(
        engine
            .query_cast_shape(
                &Shape::Sphere { radius: 0.25 },
                Vec3::ZERO,
                Quat::IDENTITY,
                Vec3::X,
                &filter
            )
            .is_none()
    );
    assert!(
        engine
            .query_intersect_shape(
                &Shape::Sphere { radius: 0.25 },
                Vec3::ZERO,
                Quat::IDENTITY,
                &filter
            )
            .is_empty()
    );
    // Stale tree: teleport past the fat without a step — the view refuses
    // to bind and the brute fallback still matches the oracle exactly.
    let mut scene = tree_scene();
    scene
        .engine
        .get_body_mut(scene.sphere)
        .expect("sphere live")
        .position = Vec3::new(500.0, 0.0, 0.0);
    let brute = QueryPipeline::new();
    assert_hit_eq(
        scene
            .engine
            .query_cast_ray(&x_ray(), 60.0, &filter)
            .expect("valid ray"),
        brute
            .cast_ray(&scene.engine.bodies, &x_ray(), 60.0, &filter)
            .expect("valid ray"),
    );
    assert_eq!(
        scene.engine.query_intersect_point(Vec3::ZERO, &filter),
        brute.intersect_point(&scene.engine.bodies, Vec3::ZERO, &filter)
    );
    assert_eq!(
        scene
            .engine
            .query_intersect_aabb(&AABB::new(Vec3::splat(-14.0), Vec3::splat(10.0)), &filter),
        brute.intersect_aabb(
            &scene.engine.bodies,
            &AABB::new(Vec3::splat(-14.0), Vec3::splat(10.0)),
            &filter
        )
    );
}

/// Queries hold only `&self`, so concurrent ray casts from several threads
/// must return bitwise-identical hits.
#[test]
fn concurrent_ray_queries_are_bitwise_deterministic() {
    let scene = tree_scene();
    let filter = QueryFilter::default();
    let expected = scene
        .engine
        .query_cast_ray(&x_ray(), 60.0, &filter)
        .expect("valid ray")
        .expect("hit");
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..4 {
            // The ray and the filter are built inside the worker: the
            // filter type carries a predicate trait object and is not
            // `Sync`, while the engine behind `&self` queries is.
            handles.push(scope.spawn(|| {
                let ray = x_ray();
                let filter = QueryFilter::default();
                for _ in 0..2 {
                    let hit = scene
                        .engine
                        .query_cast_ray(&ray, 60.0, &filter)
                        .expect("valid ray")
                        .expect("hit");
                    assert_eq!(hit.handle, expected.handle);
                    assert_eq!(hit.distance, expected.distance);
                    assert_eq!(hit.point, expected.point);
                    assert_eq!(hit.normal, expected.normal);
                }
            }));
        }
        for handle in handles {
            handle.join().expect("worker panics fail the test");
        }
    });
}
