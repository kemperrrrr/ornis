//! Unit tests for showcase collider installation.

use super::{GameWorld, Scene, install_scene_bodies};
use glam::Vec3;
use ornis_assets::collider::ColliderDesc;
use ornis_physics::{RigidBody, Shape};

#[test]
fn box_cylinder_and_explicit_plane_get_bodies() {
    // Three supported authoring shapes; the plane needs an explicit recipe.
    let scene = Scene::from_ron(
        r#"
Scene(
    name: "collider_fixture",
    entities: [
        (
            name: "Box",
            transform: (
                translation: (1.0, 0.0, 0.0),
                rotation: (0.0, 0.0, 0.0, 1.0),
                scale: (1.0, 1.0, 1.0),
            ),
            mesh: Box(size: (2.0, 2.0, 2.0)),
            material: Dielectric(base_color: (0.8, 0.2, 0.2), roughness: 0.5),
        ),
        (
            name: "Cylinder",
            transform: (
                translation: (0.0, 0.0, 0.0),
                rotation: (0.0, 0.0, 0.0, 1.0),
                scale: (1.0, 1.0, 1.0),
            ),
            mesh: Cylinder(radius: 0.5, height: 2.0, radial_segments: 16),
            material: Dielectric(base_color: (0.2, 0.8, 0.2), roughness: 0.5),
        ),
        (
            name: "Plane",
            transform: (
                translation: (0.0, -1.0, 0.0),
                rotation: (0.7071068, 0.0, 0.0, 0.7071068),
                scale: (1.0, 1.0, 1.0),
            ),
            mesh: Plane(size: (4.0, 4.0)),
            material: Dielectric(base_color: (0.2, 0.2, 0.8), roughness: 0.5),
        ),
        (
            name: "Scaled Box",
            transform: (
                translation: (0.0, 1.0, 0.0),
                rotation: (0.0, 0.0, 0.0, 1.0),
                scale: (2.0, 1.0, 1.0),
            ),
            mesh: Box(size: (2.0, 2.0, 2.0)),
            material: Dielectric(base_color: (0.8, 0.2, 0.2), roughness: 0.5),
        ),
    ],
    lights: [],
    camera: (
        position: (0.0, 3.0, 11.0),
        target: (0.0, 0.5, 0.0),
        up: (0.0, 1.0, 0.0),
        fov: 60.0,
        near: 0.1,
        far: 100.0,
    ),
    ambient: (0.1, 0.1, 0.1),
)
"#,
    )
    .expect("fixture parses");
    let mut runtime = GameWorld::from_scene(&scene);
    let entities = runtime.entities().to_vec();
    assert_eq!(entities.len(), 4);
    {
        let store = runtime
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine carries a store");
        store.register::<ColliderDesc>();
        store.insert(
            entities[2],
            ColliderDesc::HalfSpace {
                normal: [0.0, 1.0, 0.0],
            },
        );
    }

    let without_body = install_scene_bodies(&mut runtime, &scene);
    assert_eq!(without_body, vec!["Scaled Box".to_string()]);

    let store = runtime
        .engine_mut()
        .world_mut()
        .store_mut()
        .expect("engine carries a store");
    let bodies = store.read_lane::<RigidBody>().expect("body lane exists");
    let box_body = bodies.get(entities[0]).expect("box body exists");
    assert_eq!(box_body.mass, 1.0);
    assert_eq!(box_body.position, Vec3::new(1.0, 0.0, 0.0));
    match &box_body.shape {
        Shape::Box { half_extents } => assert_eq!(*half_extents, Vec3::ONE),
        shape => panic!("expected box shape, got {shape:?}"),
    }

    let cylinder_body = bodies.get(entities[1]).expect("cylinder body exists");
    assert_eq!(cylinder_body.mass, 0.0);
    assert!(matches!(
        &cylinder_body.shape,
        Shape::Cylinder { radius, half_height } if *radius == 0.5 && *half_height == 1.0
    ));

    let plane_body = bodies.get(entities[2]).expect("plane body exists");
    assert_eq!(plane_body.mass, 0.0);
    assert_eq!(plane_body.position, Vec3::new(0.0, -1.0, 0.0));
    assert!(
        (plane_body.orientation * Vec3::Y).abs_diff_eq(Vec3::Z, 1e-6),
        "plane rotation must be preserved"
    );
    match &plane_body.shape {
        Shape::HalfSpace { normal } => assert_eq!(normal.get(), Vec3::Y),
        shape => panic!("expected half-space shape, got {shape:?}"),
    }
    assert!(
        !bodies.contains(entities[3]),
        "non-unit scale must not produce a mismatched collider"
    );
}

#[test]
fn custom_without_recipe_is_reported_not_silently_skipped() {
    let scene = Scene::from_ron(
        r#"
Scene(
    name: "custom_fixture",
    entities: [
        (
            name: "Soup",
            transform: (
                translation: (0.0, 0.0, 0.0),
                rotation: (0.0, 0.0, 0.0, 1.0),
                scale: (1.0, 1.0, 1.0),
            ),
            mesh: Custom(
                positions: [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                indices: [0, 1, 2],
            ),
            material: Dielectric(base_color: (0.5, 0.5, 0.5), roughness: 0.5),
        ),
    ],
    lights: [],
    camera: (
        position: (0.0, 3.0, 11.0),
        target: (0.0, 0.5, 0.0),
        up: (0.0, 1.0, 0.0),
        fov: 60.0,
        near: 0.1,
        far: 100.0,
    ),
    ambient: (0.1, 0.1, 0.1),
)
"#,
    )
    .expect("fixture parses");
    let mut runtime = GameWorld::from_scene(&scene);
    let entity = runtime.entities()[0];

    let without_body = install_scene_bodies(&mut runtime, &scene);
    assert_eq!(without_body, vec!["Soup".to_string()]);

    let store = runtime
        .engine_mut()
        .world_mut()
        .store_mut()
        .expect("engine carries a store");
    assert!(
        store
            .read_lane::<RigidBody>()
            .is_none_or(|lane| !lane.contains(entity)),
        "custom soup without a recipe must not get a body"
    );
}
