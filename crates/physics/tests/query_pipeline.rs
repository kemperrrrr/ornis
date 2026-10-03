//! QueryPipeline pins: ray / point / AABB / projection / shape queries over
//! a scene with all eight built-in shapes, plus every filter switch.
//!
//! All queries are read-only: the suite snapshots poses before running the
//! whole pipeline and asserts they are unchanged afterwards.

use glam::{Quat, Vec3};
use ornis_physics::{
    AABB, BodyHandle, BodyType, PhysicsEngine, PointProjection, QueryError, QueryFilter,
    QueryPipeline, Ray, RigidBody, SequentialImpulseEngine, Shape, Triangle,
};

/// Eight-shape scene, spread along the axes so queries isolate cleanly.
///
/// Layers are one bit per body (`1 << index`); every mask is open.
struct Scene {
    engine: SequentialImpulseEngine,
    sphere: BodyHandle,
    cuboid: BodyHandle,
    capsule: BodyHandle,
    cylinder: BodyHandle,
    cone: BodyHandle,
    hull: BodyHandle,
    terrain: BodyHandle,
    mesh: BodyHandle,
}

fn eight_shape_scene() -> Scene {
    let mut engine = SequentialImpulseEngine::new(Vec3::ZERO);
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
    driver.body_type = BodyType::Kinematic;
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
            &[Triangle::from_raw([0, 2, 1]), Triangle::from_raw([0, 3, 2])],
            0.0,
        )
        .expect("quad mesh builds")
        .with_collision_filter(mesh_layer, mesh_mask),
    );
    Scene {
        engine,
        sphere,
        cuboid,
        capsule,
        cylinder,
        cone,
        hull,
        terrain,
        mesh,
    }
}

/// Ray down the +X axis at y = z = 0: passes the capsule sensor, the
/// sphere, the static box and the hull tetra, in that order.
fn x_ray() -> Ray {
    Ray::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::X)
}

#[test]
fn cast_ray_hits_closest_with_normal() {
    let scene = eight_shape_scene();
    let filter = QueryFilter::default();
    let hit = scene
        .engine
        .query_cast_ray(&x_ray(), 50.0, &filter)
        .expect("valid ray")
        .expect("ray hits the capsule first");
    assert_eq!(hit.handle, scene.capsule);
    assert!((hit.distance - 5.7).abs() < 1e-3, "got {}", hit.distance);
    assert!(
        (hit.normal - Vec3::NEG_X).length() < 1e-5,
        "got {:?}",
        hit.normal
    );
    assert!((hit.point.x - -4.3).abs() < 1e-3, "got {:?}", hit.point);
}

#[test]
fn cast_ray_respects_sensor_veto() {
    let scene = eight_shape_scene();
    let filter = QueryFilter {
        exclude_sensors: true,
        ..QueryFilter::default()
    };
    let hit = scene
        .engine
        .query_cast_ray(&x_ray(), 50.0, &filter)
        .expect("valid ray")
        .expect("ray hits the sphere next");
    assert_eq!(hit.handle, scene.sphere);
    assert!((hit.distance - 9.5).abs() < 1e-3, "got {}", hit.distance);
}

#[test]
fn cast_ray_rejects_invalid_input() {
    let scene = eight_shape_scene();
    let filter = QueryFilter::default();
    let bad = Ray::new(Vec3::ZERO, Vec3::ZERO);
    assert!(matches!(
        scene.engine.query_cast_ray(&bad, 10.0, &filter),
        Err(QueryError::InvalidInput { .. })
    ));
    assert!(matches!(
        scene.engine.query_cast_ray(&x_ray(), f32::NAN, &filter),
        Err(QueryError::InvalidInput { .. })
    ));
    // Clean miss is Ok(None), not an error.
    let up = Ray::new(Vec3::new(0.0, 50.0, 0.0), Vec3::Y);
    assert!(
        scene
            .engine
            .query_cast_ray(&up, 10.0, &filter)
            .expect("valid ray")
            .is_none()
    );
}

#[test]
fn intersect_ray_returns_all_hits_sorted() {
    let scene = eight_shape_scene();
    let hits = scene
        .engine
        .query_intersect_ray(&x_ray(), 50.0, &QueryFilter::default())
        .expect("valid ray");
    let handles: Vec<BodyHandle> = hits.iter().map(|h| h.handle).collect();
    assert_eq!(
        handles,
        vec![scene.capsule, scene.sphere, scene.cuboid, scene.hull]
    );
    let distances: Vec<f32> = hits.iter().map(|h| h.distance).collect();
    assert!((distances[0] - 5.7).abs() < 1e-3, "got {distances:?}");
    assert!((distances[1] - 9.5).abs() < 1e-3, "got {distances:?}");
    assert!((distances[2] - 13.5).abs() < 1e-3, "got {distances:?}");
    assert!((distances[3] - 18.0).abs() < 0.05, "got {distances:?}");
    let mut sorted = distances.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    assert_eq!(distances, sorted);
}

#[test]
fn intersect_point_contains_solids_boundary_inclusive() {
    let scene = eight_shape_scene();
    let filter = QueryFilter::default();
    // Snapshot-level call (bypasses the engine binding) for the first pin.
    assert_eq!(
        QueryPipeline::new().intersect_point(&scene.engine.bodies, Vec3::ZERO, &filter),
        vec![scene.sphere]
    );
    // Box face point counts as contained.
    assert_eq!(
        scene
            .engine
            .query_intersect_point(Vec3::new(4.0, 0.5, 0.0), &filter),
        vec![scene.cuboid]
    );
    // Sensor core is contained too (sensors are queryable by default).
    assert_eq!(
        scene
            .engine
            .query_intersect_point(Vec3::new(-4.0, 0.0, 0.0), &filter),
        vec![scene.capsule]
    );
    assert!(
        scene
            .engine
            .query_intersect_point(Vec3::splat(100.0), &filter)
            .is_empty()
    );
    assert!(
        scene
            .engine
            .query_intersect_point(Vec3::NAN, &filter)
            .is_empty()
    );
}

#[test]
fn intersect_aabb_spans_all_shapes_or_a_subset() {
    let scene = eight_shape_scene();
    let filter = QueryFilter::default();
    let all = scene
        .engine
        .query_intersect_aabb(&AABB::new(Vec3::splat(-10.0), Vec3::splat(10.0)), &filter);
    assert_eq!(all.len(), 8, "got {all:?}");
    for handle in [
        scene.sphere,
        scene.cuboid,
        scene.capsule,
        scene.cylinder,
        scene.cone,
        scene.hull,
        scene.terrain,
        scene.mesh,
    ] {
        assert!(all.contains(&handle), "span misses {handle:?}");
    }
    // Tight box around the origin: only the sphere overlaps.
    let near = scene
        .engine
        .query_intersect_aabb(&AABB::new(Vec3::splat(-1.0), Vec3::splat(1.0)), &filter);
    assert_eq!(near, vec![scene.sphere]);
}

#[test]
fn project_point_reports_surface_inside_and_feature() {
    let scene = eight_shape_scene();
    let filter = QueryFilter::default();
    // Above the sphere: surface point, outside, sphere feature 0.
    let above = scene
        .engine
        .query_project_point(Vec3::new(0.0, 2.0, 0.0), &filter)
        .expect("sphere is closest");
    assert_eq!(above.handle, scene.sphere);
    assert!((above.point - Vec3::new(0.0, 0.5, 0.0)).length() < 1e-4);
    assert!(!above.is_inside);
    assert_eq!(above.feature_id, 0);
    // Sphere center: inside, still 0.5 from the surface.
    let core = scene
        .engine
        .query_project_point(Vec3::ZERO, &filter)
        .expect("sphere contains its center");
    assert_eq!(core.handle, scene.sphere);
    assert!(core.is_inside);
    assert!(
        (core.point.length() - 0.5).abs() < 1e-4,
        "got {:?}",
        core.point
    );
    // Above the static box: top face (+Y = 2).
    let lid = scene
        .engine
        .query_project_point(Vec3::new(4.0, 2.0, 0.0), &filter)
        .expect("box is closest");
    assert_eq!(lid.handle, scene.cuboid);
    assert!((lid.point - Vec3::new(4.0, 0.5, 0.0)).length() < 1e-4);
    assert!(!lid.is_inside);
    assert_eq!(lid.feature_id, 2);
    // Non-finite query projects to nothing.
    assert!(
        scene
            .engine
            .query_project_point(Vec3::NAN, &filter)
            .is_none()
    );
}

#[test]
fn cast_shape_sweep_hits_first_touch_with_normal() {
    let scene = eight_shape_scene();
    let mover = Shape::Sphere { radius: 0.25 };
    let filter = QueryFilter::default();
    let hit = scene
        .engine
        .query_cast_shape(
            &mover,
            Vec3::new(-2.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::new(4.0, 0.0, 0.0),
            &filter,
        )
        .expect("sweep reaches the sphere");
    assert_eq!(hit.handle, scene.sphere);
    assert!((hit.distance - 1.25).abs() < 0.01, "got {}", hit.distance);
    assert!(
        (hit.normal - Vec3::NEG_X).length() < 1e-3,
        "got {:?}",
        hit.normal
    );
    // Short sweep away from everything: clean miss.
    assert!(
        scene
            .engine
            .query_cast_shape(
                &mover,
                Vec3::new(-2.0, 0.0, 0.0),
                Quat::IDENTITY,
                Vec3::new(-1.0, 0.0, 0.0),
                &filter,
            )
            .is_none()
    );
    // Degenerate sweep: no travel, no hit.
    assert!(
        scene
            .engine
            .query_cast_shape(
                &mover,
                Vec3::new(-2.0, 0.0, 0.0),
                Quat::IDENTITY,
                Vec3::ZERO,
                &filter,
            )
            .is_none()
    );
}

#[test]
fn cast_shape_filter_selects_static_box() {
    let scene = eight_shape_scene();
    // Long sweep with dynamics vetoed: the static box is the only hit.
    let mover = Shape::Sphere { radius: 0.25 };
    let filter = QueryFilter {
        exclude_dynamic: true,
        ..QueryFilter::default()
    };
    let hit = scene
        .engine
        .query_cast_shape(
            &mover,
            Vec3::new(-2.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::new(10.0, 0.0, 0.0),
            &filter,
        )
        .expect("sweep reaches the static box");
    assert_eq!(hit.handle, scene.cuboid);
}

#[test]
fn intersect_shape_lists_overlaps() {
    let scene = eight_shape_scene();
    let filter = QueryFilter::default();
    let probe = Shape::Sphere { radius: 0.5 };
    assert_eq!(
        scene
            .engine
            .query_intersect_shape(&probe, Vec3::ZERO, Quat::IDENTITY, &filter),
        vec![scene.sphere]
    );
    assert!(
        scene
            .engine
            .query_intersect_shape(&probe, Vec3::splat(100.0), Quat::IDENTITY, &filter)
            .is_empty()
    );
}

#[test]
fn filters_cover_body_types_sensors_and_solids() {
    let scene = eight_shape_scene();
    let span = AABB::new(Vec3::splat(-10.0), Vec3::splat(10.0));
    // Dynamics vetoed: statics (box, terrain, mesh) + the kinematic cylinder.
    let still = scene.engine.query_intersect_aabb(
        &span,
        &QueryFilter {
            exclude_dynamic: true,
            ..QueryFilter::default()
        },
    );
    assert_eq!(still.len(), 4, "got {still:?}");
    assert!(still.contains(&scene.cuboid));
    assert!(still.contains(&scene.cylinder));
    assert!(still.contains(&scene.terrain));
    assert!(still.contains(&scene.mesh));
    // Fixed + kinematic vetoed: the four dynamics.
    let moving = scene.engine.query_intersect_aabb(
        &span,
        &QueryFilter {
            exclude_fixed: true,
            exclude_kinematic: true,
            ..QueryFilter::default()
        },
    );
    assert_eq!(moving.len(), 4, "got {moving:?}");
    assert!(moving.contains(&scene.sphere));
    assert!(moving.contains(&scene.capsule));
    assert!(moving.contains(&scene.cone));
    assert!(moving.contains(&scene.hull));
    // Sensors vetoed: everything but the capsule.
    let solids = scene.engine.query_intersect_aabb(
        &span,
        &QueryFilter {
            exclude_sensors: true,
            ..QueryFilter::default()
        },
    );
    assert_eq!(solids.len(), 7);
    assert!(!solids.contains(&scene.capsule));
    // Solids vetoed: only the capsule.
    assert_eq!(
        scene.engine.query_intersect_aabb(
            &span,
            &QueryFilter {
                exclude_solids: true,
                ..QueryFilter::default()
            }
        ),
        vec![scene.capsule]
    );
}

#[test]
fn filter_layer_mask_is_mutual() {
    let scene = eight_shape_scene();
    let span = AABB::new(Vec3::splat(-10.0), Vec3::splat(10.0));
    // Interest mask selects the box layer only.
    let boxed = scene.engine.query_intersect_aabb(
        &span,
        &QueryFilter {
            mask: 0b0010,
            ..QueryFilter::default()
        },
    );
    assert_eq!(boxed, vec![scene.cuboid]);
}

#[test]
fn filter_query_layer_against_body_mask() {
    let mut scene = eight_shape_scene();
    // Tighten the box mask: it now collides only with layer 1 bodies.
    scene
        .engine
        .get_body_mut(scene.cuboid)
        .expect("box live")
        .set_collision_filter(0b0010, 0b0001);
    let span = AABB::new(Vec3::splat(-10.0), Vec3::splat(10.0));
    // Query membership on layer 2: the box mask (1) no longer includes it.
    let hits = scene.engine.query_intersect_aabb(
        &span,
        &QueryFilter {
            layer: 0b0010,
            ..QueryFilter::default()
        },
    );
    assert_eq!(hits.len(), 7, "got {hits:?}");
    assert!(!hits.contains(&scene.cuboid));
}

#[test]
fn filter_predicate_vetoes_single_handle() {
    let scene = eight_shape_scene();
    let span = AABB::new(Vec3::splat(-10.0), Vec3::splat(10.0));
    let skip_sphere = QueryFilter {
        predicate: Some(&|handle, _| handle != scene.sphere),
        ..QueryFilter::default()
    };
    let hits = scene.engine.query_intersect_aabb(&span, &skip_sphere);
    assert_eq!(hits.len(), 7, "got {hits:?}");
    assert!(!hits.contains(&scene.sphere));
}

#[test]
fn queries_never_mutate_the_scene() {
    let scene = eight_shape_scene();
    let engine = &scene.engine;
    let before: Vec<(Vec3, Quat)> = (0..engine.body_count())
        .map(|i| {
            let body = engine.get_body(BodyHandle::from(i)).expect("dense handles");
            (body.position, body.orientation)
        })
        .collect();
    let filter = QueryFilter::default();
    let ray = x_ray();
    let _ = engine.query_cast_ray(&ray, 50.0, &filter);
    let _ = engine.query_intersect_ray(&ray, 50.0, &filter);
    let _ = engine.query_intersect_point(Vec3::ZERO, &filter);
    let _ = engine.query_intersect_aabb(&AABB::new(Vec3::splat(-10.0), Vec3::splat(10.0)), &filter);
    let _ = engine.query_project_point(Vec3::new(4.0, 2.0, 0.0), &filter);
    let _ = engine.query_cast_shape(
        &Shape::Sphere { radius: 0.25 },
        Vec3::new(-2.0, 0.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(4.0, 0.0, 0.0),
        &filter,
    );
    let _ = engine.query_intersect_shape(
        &Shape::Sphere { radius: 0.5 },
        Vec3::ZERO,
        Quat::IDENTITY,
        &filter,
    );
    // Repeat for determinism: identical queries return identical hits.
    let first = engine
        .query_cast_ray(&ray, 50.0, &filter)
        .expect("valid ray")
        .expect("hit");
    let second = engine
        .query_cast_ray(&ray, 50.0, &filter)
        .expect("valid ray")
        .expect("hit");
    assert_eq!(first.handle, second.handle);
    assert_eq!(first.distance, second.distance);
    assert_eq!(first.point, second.point);
    for (i, (position, orientation)) in before.iter().enumerate() {
        let body = engine.get_body(BodyHandle::from(i)).expect("dense handles");
        assert_eq!(body.position, *position);
        assert_eq!(body.orientation, *orientation);
    }
    assert_eq!(engine.body_count(), before.len());
}

/// The projection type documents its degraded-shape sentinel next to the code.
#[test]
fn projection_feature_sentinel_is_max() {
    assert_eq!(PointProjection::UNKNOWN_FEATURE, u32::MAX);
}
