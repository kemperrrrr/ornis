//! Showcase-specific rigid-body construction, the hidden floor and tests.

use glam::Vec3;
use ornis_app::GameWorld;
use ornis_assets::collider::ColliderDesc;
use ornis_assets::scene::Scene;
use ornis_core::{Engine, Entity, SmartStore};
use ornis_physics::{ColliderError, RigidBody};

const SCALE_TOLERANCE: f32 = 1e-6;

/// Builds one physics body per scene entity through the shared collider
/// projection ([`ornis_physics::colliders::body_for`]).
///
/// An explicit [`ColliderDesc`] lane entry wins over the exact auto recipe;
/// otherwise sphere/box/cylinder meshes get their analytic collider. Plane,
/// quad and custom meshes without an explicit recipe stay body-less. Non-unit
/// scale is rejected until the shared projection supports it, except for a
/// uniformly scaled half-space whose plane is unchanged by that scale. Every
/// skip is logged once and its entity name is returned.
/// The first scene entity is dynamic (mass `1`) unless its recipe is static-only;
/// the rest are static.
pub(super) fn install_scene_bodies(runtime: &mut GameWorld, scene: &Scene) -> Vec<String> {
    let entities = runtime.entities().to_vec();
    let Some(store) = runtime.engine_mut().world_mut().store_mut() else {
        return Vec::new();
    };
    entities
        .into_iter()
        .enumerate()
        .filter_map(|(index, entity)| {
            let description = scene.entities.get(index)?;
            let collider = store
                .read_lane::<ColliderDesc>()
                .and_then(|lane| lane.get(entity).cloned());
            let is_half_space = matches!(collider.as_ref(), Some(ColliderDesc::HalfSpace { .. }));
            let scale = description.transform.scale;
            let unit = scale.abs_diff_eq(Vec3::ONE, SCALE_TOLERANCE);
            let uniform = scale.abs_diff_eq(Vec3::splat(scale.x), SCALE_TOLERANCE);
            match (unit, uniform, is_half_space) {
                (false, false, _) | (false, _, false) => {
                    let reason = format!("has unsupported scale {scale:?}");
                    Some(skip_entity(&description.name, &reason))
                }
                _ => {
                    let mass = match (index, is_half_space) {
                        (0, false) => 1.0,
                        _ => 0.0,
                    };
                    let result = ornis_physics::colliders::body_for(
                        &description.transform,
                        &description.mesh,
                        collider.as_ref(),
                        mass,
                    );
                    install_body_result(store, entity, &description.name, result)
                }
            }
        })
        .collect()
}

fn install_body_result(
    store: &mut SmartStore,
    entity: Entity,
    name: &str,
    result: Result<Option<RigidBody>, ColliderError>,
) -> Option<String> {
    match result {
        Ok(Some(body)) => {
            store.insert(entity, body);
            None
        }
        Ok(None) => Some(skip_entity(name, "has no collider recipe")),
        Err(error) => {
            let reason = format!("collider recipe failed: {error}");
            Some(skip_entity(name, &reason))
        }
    }
}

fn skip_entity(name: &str, reason: &str) -> String {
    eprintln!("showcase entity {name:?} {reason}; no physics body created");
    name.to_owned()
}

/// Centre of the hidden showcase static floor in world units.
const SHOWCASE_FLOOR_CENTER: [f32; 3] = [0.0, -2.0, 0.0];
/// Half-extents of the hidden showcase static floor in world units.
const SHOWCASE_FLOOR_HALF_EXTENTS: [f32; 3] = [20.0, 1.0, 20.0];

/// The showcase world has no component store, so the floor cannot be placed.
#[derive(Debug)]
pub(super) struct ShowcaseFloorError;

impl std::fmt::Display for ShowcaseFloorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("showcase world has no component store")
    }
}

impl std::error::Error for ShowcaseFloorError {}

/// Spawns the hidden showcase static floor.
///
/// The floor carries only a physics component, so it never enters the frame
/// upload; it exists so dynamic showcase bodies have ground to rest on.
///
/// # Errors
///
/// [`ShowcaseFloorError`] when the world has no component store.
pub(super) fn spawn_showcase_floor(engine: &mut Engine) -> Result<Entity, ShowcaseFloorError> {
    let Some(store) = engine.world_mut().store_mut() else {
        return Err(ShowcaseFloorError);
    };
    let floor = store.create_entity();
    store.insert(
        floor,
        RigidBody::new_box(
            Vec3::from_array(SHOWCASE_FLOOR_CENTER),
            Vec3::from_array(SHOWCASE_FLOOR_HALF_EXTENTS),
            0.0,
        ),
    );
    Ok(floor)
}

#[cfg(test)]
mod tests {
    use super::{GameWorld, Scene, Vec3, install_scene_bodies};
    use ornis_assets::collider::ColliderDesc;
    use ornis_core::Entity;
    use ornis_physics::{RigidBody, Shape};

    fn put_halfspace(runtime: &mut GameWorld, entity: Entity) {
        let store = runtime
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine carries a store");
        store.register::<ColliderDesc>();
        store.insert(
            entity,
            ColliderDesc::HalfSpace {
                normal: [0.0, 1.0, 0.0],
            },
        );
    }

    fn single_entity_runtime(ron: &str) -> (Scene, GameWorld, Entity) {
        let scene = Scene::from_ron(ron).expect("fixture parses");
        let runtime = GameWorld::from_scene(&scene);
        let entity = runtime.entities()[0];
        (scene, runtime, entity)
    }

    fn has_body(runtime: &mut GameWorld, entity: Entity) -> bool {
        runtime
            .engine_mut()
            .world_mut()
            .store_mut()
            .and_then(|store| store.read_lane::<RigidBody>())
            .is_some_and(|lane| lane.contains(entity))
    }

    fn assert_rejected(
        runtime: &mut GameWorld,
        scene: &Scene,
        entity: Entity,
        expected_name: &str,
    ) {
        let without_body = install_scene_bodies(runtime, scene);
        assert_eq!(without_body, vec![expected_name.to_string()]);
        assert!(!has_body(runtime, entity));
    }

    fn assert_cylinder(body: &RigidBody, expected_radius: f32, expected_half_height: f32) {
        match &body.shape {
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                assert_eq!(*radius, expected_radius);
                assert_eq!(*half_height, expected_half_height);
            }
            shape => panic!("expected cylinder shape, got {shape:?}"),
        }
    }

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
                rotation: (0.7071068, 0.0, 0.0, 0.7071068),
                scale: (1.0, 1.0, 1.0),
            ),
            mesh: Box(size: (2.0, 2.0, 2.0)),
            material: Dielectric(base_color: (0.8, 0.2, 0.2), roughness: 0.5),
        ),
        (
            name: "Cylinder",
            transform: (
                translation: (0.0, 2.0, 0.0),
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
        assert_eq!(entities.len(), 3);
        put_halfspace(&mut runtime, entities[2]);

        let without_body = install_scene_bodies(&mut runtime, &scene);
        assert!(without_body.is_empty());

        let store = runtime
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine carries a store");
        let bodies = store.read_lane::<RigidBody>().expect("body lane exists");
        let box_body = bodies.get(entities[0]).expect("box body exists");
        assert_eq!(box_body.mass, 1.0);
        assert_eq!(box_body.position, Vec3::new(1.0, 0.0, 0.0));
        assert!((box_body.orientation * Vec3::Y).abs_diff_eq(Vec3::Z, 1e-6));
        match &box_body.shape {
            Shape::Box { half_extents } => assert_eq!(*half_extents, Vec3::ONE),
            shape => panic!("expected box shape, got {shape:?}"),
        }

        let cylinder_body = bodies.get(entities[1]).expect("cylinder body exists");
        assert_eq!(cylinder_body.mass, 0.0);
        assert_eq!(cylinder_body.position, Vec3::new(0.0, 2.0, 0.0));
        assert_cylinder(cylinder_body, 0.5, 1.0);

        let plane_body = bodies.get(entities[2]).expect("plane body exists");
        assert_eq!(plane_body.mass, 0.0);
        assert_eq!(plane_body.position, Vec3::new(0.0, -1.0, 0.0));
        assert!((plane_body.orientation * Vec3::Y).abs_diff_eq(Vec3::Z, 1e-6));
        match &plane_body.shape {
            Shape::HalfSpace { normal } => assert_eq!(normal.get(), Vec3::Y),
            shape => panic!("expected half-space shape, got {shape:?}"),
        }
    }

    #[test]
    fn non_unit_scale_is_reported_without_body() {
        let (scene, mut runtime, entity) = single_entity_runtime(
            r#"
Scene(
    name: "scale_fixture",
    entities: [
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
        );

        assert_rejected(&mut runtime, &scene, entity, "Scaled Box");
    }

    #[test]
    fn first_uniformly_scaled_half_space_is_static() {
        let (scene, mut runtime, entity) = single_entity_runtime(
            r#"
Scene(
    name: "half_space_fixture",
    entities: [
        (
            name: "Ground",
            transform: (
                translation: (0.0, 0.0, 0.0),
                rotation: (0.0, 0.0, 0.0, 1.0),
                scale: (2.0, 2.0, 2.0),
            ),
            mesh: Plane(size: (4.0, 4.0)),
            material: Dielectric(base_color: (0.2, 0.2, 0.8), roughness: 0.5),
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
        );
        put_halfspace(&mut runtime, entity);

        let without_body = install_scene_bodies(&mut runtime, &scene);
        assert!(without_body.is_empty());
        let store = runtime
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine carries a store");
        let bodies = store.read_lane::<RigidBody>().expect("body lane exists");
        let ground = bodies.get(entity).expect("ground body exists");
        assert_eq!(ground.mass, 0.0);
        assert!(matches!(&ground.shape, Shape::HalfSpace { .. }));
    }

    #[test]
    fn demo_scene_has_exact_floor_wall_and_tower_base_bodies() {
        let scene =
            Scene::from_ron(include_str!("../assets/demo_scene.ron")).expect("demo scene parses");
        let mut runtime = GameWorld::from_scene(&scene);
        let entities = runtime.entities().to_vec();
        assert!(install_scene_bodies(&mut runtime, &scene).is_empty());

        let store = runtime
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine carries a store");
        let bodies = store.read_lane::<RigidBody>().expect("body lane exists");
        let body_for = |name: &str| -> &RigidBody {
            let index = scene
                .entities
                .iter()
                .position(|entity| entity.name == name)
                .unwrap_or_else(|| panic!("missing demo entity {name}"));
            bodies.get(entities[index]).expect("demo body exists")
        };

        let floor = body_for("Floor");
        assert_eq!(floor.mass, 0.0);
        match &floor.shape {
            Shape::Box { half_extents } => {
                assert_eq!(*half_extents, Vec3::new(12.0, 0.5, 12.0));
            }
            shape => panic!("expected floor box, got {shape:?}"),
        }
        let wall = body_for("Wall");
        assert_eq!(wall.mass, 0.0);
        match &wall.shape {
            Shape::Box { half_extents } => {
                assert_eq!(*half_extents, Vec3::new(12.0, 4.0, 0.5));
            }
            shape => panic!("expected wall box, got {shape:?}"),
        }
        let tower_base = body_for("Tower Base");
        assert_eq!(tower_base.mass, 0.0);
        assert_cylinder(tower_base, 1.5, 1.0);
    }

    #[test]
    fn custom_without_recipe_is_reported_not_silently_skipped() {
        let (scene, mut runtime, entity) = single_entity_runtime(
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
        );

        assert_rejected(&mut runtime, &scene, entity, "Soup");
    }
}
