//! `ornis` binary: native mode (winit + wgpu) and `editor-only`
//! (editor HTTP server on port 3420 without a native window).

#![warn(missing_docs)]

#[cfg(feature = "editor-only")]
use crossbeam_channel::unbounded;
#[cfg(feature = "editor-only")]
use editor_backend::RemoteEditor;
#[cfg(not(feature = "editor-only"))]
use glam::Vec3;
#[cfg(not(feature = "editor-only"))]
use ornis_app::physics_runtime::install_physics;
#[cfg(not(feature = "editor-only"))]
use ornis_app::{GameWorld, install_gameplay_physics_bridge, install_object_animation};
#[cfg(not(feature = "editor-only"))]
use ornis_assets::collider::ColliderDesc;
#[cfg(not(feature = "editor-only"))]
use ornis_assets::scene::Scene;
#[cfg(not(feature = "editor-only"))]
use ornis_audio::AudioPlugin;
#[cfg(not(feature = "editor-only"))]
use ornis_audio::bridge::install_gameplay_audio_bridge;
#[cfg(not(feature = "editor-only"))]
use ornis_core::{Engine, Entity};
#[cfg(not(feature = "editor-only"))]
use ornis_gameplay::install_gameplay;
#[cfg(not(feature = "editor-only"))]
use ornis_physics::RigidBody;
#[cfg(not(feature = "editor-only"))]
use ornis_render::OrbitCamera;

#[cfg(feature = "editor-only")]
use ornis_runner::EDITOR_HTTP_PORT;

/// Earth-surface gravity along −Y (m/s²).
#[cfg(not(feature = "editor-only"))]
const DEFAULT_GRAVITY_Y: f32 = -9.81;

// ═══════════════════════════════════════════════════════════════════════════
// "BROWSER-ONLY EDITOR" MODE (editor-only)
// ═══════════════════════════════════════════════════════════════════════════
// Run with: cargo run --features editor-only
// No native winit window is created; only the RemoteEditor HTTP server
// on port 3420 runs. The developer opens http://127.0.0.1:3420 in a
// browser and gets the full editor.
//
// Strategic pivot from native UI to the browser editor
// (July 2026, see PLAN.md). The native UI crate was removed (August 2026).
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "editor-only")]
fn main() {
    let (cmd_tx, cmd_rx) = unbounded();
    let (ev_tx, ev_rx) = unbounded();

    // Live ECS world on a dedicated thread: executes commands from
    // POST /api/command and publishes status/scene snapshots + events.
    ornis_app::session::run(cmd_rx, ev_tx);

    // The binding keeps RemoteEditor alive until main ends (Drop stops the server).
    let _editor = RemoteEditor::start(EDITOR_HTTP_PORT, cmd_tx, ev_rx);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Ornis Engine — Browser Editor Mode                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Editor:  http://127.0.0.1:3420                               ║");
    println!("║  Status:  http://127.0.0.1:3420/api/status                    ║");
    println!("║  Scene:   http://127.0.0.1:3420/api/scene                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Press Ctrl+C to stop                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Wait forever — the server runs on a separate thread.
    loop {
        std::thread::park();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FULL ENGINE WITH A NATIVE WINDOW (default mode)
// ═══════════════════════════════════════════════════════════════════════════
// Run with: cargo run
// winit window, wgpu rendering, a 3D scene of spheres (OpenPBR).
// Animation demo: `cargo run --example anim` (starter character via
// [`ornis::run`], not a binary flag).
// The browser editor server is opt-in here: `cargo run -- --remote-editor`
// serves it on port 3420 (off by default — audit §6.2, backlog #16).
// The native UI overlay was removed with the ornis-ui crate — the editor
// lives in the browser (see editor-only mode / cargo xtask editor).
// ═══════════════════════════════════════════════════════════════════════════

/// Spheres showcase world: RON scene plus physics bodies, audio and
/// object animation. `ornis::run` owns the window; content builders
/// like this one stay here.
#[cfg(not(feature = "editor-only"))]
fn showcase_engine() -> GameWorld {
    let Ok(scene) = Scene::from_ron(include_str!("../assets/demo_scene.ron")) else {
        return GameWorld::default();
    };
    let mut runtime = GameWorld::from_scene(&scene);
    runtime.spawn(OrbitCamera::from_desc(&scene.camera));
    install_physics(runtime.engine_mut(), Vec3::new(0.0, DEFAULT_GRAVITY_Y, 0.0));
    install_gameplay(runtime.engine_mut());
    install_gameplay_physics_bridge(runtime.engine_mut());
    // Audio steps in the same DAG (after motion); silently skipped when
    // no output device is available. No showcase entity carries an
    // AudioSource yet, so this is a no-op until content arrives.
    if let Some(audio) = AudioPlugin::try_default() {
        audio.install(runtime.engine_mut());
    }
    // Listener pose/gain sync; no-op until a host is installed.
    install_gameplay_audio_bridge(runtime.engine_mut());
    // Object animation in the same DAG (after body poses); no-op
    // until an entity carries animation lanes.
    install_object_animation(runtime.engine_mut());
    let _ = install_scene_bodies(&mut runtime, &scene);
    // Hidden static floor: physics only, so it stays out of the frame upload.
    if let Err(error) = spawn_showcase_floor(runtime.engine_mut()) {
        eprintln!("showcase floor was not spawned: {error}");
    }
    runtime.frame(0.0);
    runtime
}

/// Builds one physics body per scene entity through the shared collider
/// projection ([`ornis_physics::colliders::body_for`]).
///
/// An explicit [`ColliderDesc`] lane entry wins over the exact auto recipe;
/// otherwise sphere/box/cylinder meshes get their analytic collider. Plane,
/// quad and custom meshes without an explicit recipe stay body-less. Non-unit
/// scale is rejected until the shared projection supports it. Every skip is
/// logged once and its entity name is returned.
/// The first scene entity is dynamic (mass `1`) unless its recipe is static-only;
/// the rest are static.
#[cfg(not(feature = "editor-only"))]
fn install_scene_bodies(runtime: &mut GameWorld, scene: &Scene) -> Vec<String> {
    let entities = runtime.entities().to_vec();
    let Some(store) = runtime.engine_mut().world_mut().store_mut() else {
        return Vec::new();
    };
    let mut without_body = Vec::new();
    for (index, entity) in entities.into_iter().enumerate() {
        let Some(description) = scene.entities.get(index) else {
            continue;
        };
        if description.transform.scale != Vec3::ONE {
            eprintln!(
                "showcase entity {:?} has non-unit scale {:?}; no physics body created",
                description.name, description.transform.scale
            );
            without_body.push(description.name.clone());
            continue;
        }
        let collider = store
            .read_lane::<ColliderDesc>()
            .and_then(|lane| lane.get(entity).cloned());
        let mass =
            if index == 0 && !matches!(collider.as_ref(), Some(ColliderDesc::HalfSpace { .. })) {
                1.0
            } else {
                0.0
            };
        match ornis_physics::colliders::body_for(
            &description.transform,
            &description.mesh,
            collider.as_ref(),
            mass,
        ) {
            Ok(Some(body)) => {
                store.insert(entity, body);
            }
            Ok(None) => {
                eprintln!(
                    "showcase entity {:?} has no collider recipe; no physics body created",
                    description.name
                );
                without_body.push(description.name.clone());
            }
            Err(error) => {
                eprintln!(
                    "showcase entity {:?} collider recipe failed: {error}; no physics body created",
                    description.name
                );
                without_body.push(description.name.clone());
            }
        }
    }
    without_body
}

/// Centre of the hidden showcase static floor in world units.
#[cfg(not(feature = "editor-only"))]
const SHOWCASE_FLOOR_CENTER: [f32; 3] = [0.0, -2.0, 0.0];
/// Half-extents of the hidden showcase static floor in world units.
#[cfg(not(feature = "editor-only"))]
const SHOWCASE_FLOOR_HALF_EXTENTS: [f32; 3] = [20.0, 1.0, 20.0];

/// The showcase world has no component store, so the floor cannot be placed.
#[cfg(not(feature = "editor-only"))]
#[derive(Debug)]
struct ShowcaseFloorError;

#[cfg(not(feature = "editor-only"))]
impl std::fmt::Display for ShowcaseFloorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("showcase world has no component store")
    }
}

#[cfg(not(feature = "editor-only"))]
impl std::error::Error for ShowcaseFloorError {}

/// Spawns the hidden showcase static floor.
///
/// The floor carries only a physics component, so it never enters the frame
/// upload; it exists so dynamic showcase bodies have ground to rest on.
///
/// # Errors
///
/// [`ShowcaseFloorError`] when the world has no component store.
#[cfg(not(feature = "editor-only"))]
fn spawn_showcase_floor(engine: &mut Engine) -> Result<Entity, ShowcaseFloorError> {
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

#[cfg(not(feature = "editor-only"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    ornis::run(showcase_engine())
}

#[cfg(all(test, not(feature = "editor-only")))]
mod tests {
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
        assert!(
            without_body.is_empty(),
            "unexpected skips: {without_body:?}"
        );

        let store = runtime
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine carries a store");
        let bodies = store.read_lane::<RigidBody>().expect("body lane exists");
        for entity in &entities {
            assert!(bodies.contains(*entity), "entity {entity:?} has no body");
        }

        let box_body = bodies.get(entities[0]).expect("box body exists");
        assert_eq!(box_body.mass, 1.0);
        assert_eq!(box_body.position, Vec3::new(1.0, 0.0, 0.0));
        match &box_body.shape {
            Shape::Box { half_extents } => assert_eq!(*half_extents, Vec3::ONE),
            shape => panic!("expected box shape, got {shape:?}"),
        }

        let cylinder_body = bodies.get(entities[1]).expect("cylinder body exists");
        assert_eq!(cylinder_body.mass, 0.0);
        match &cylinder_body.shape {
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                assert_eq!(*radius, 0.5);
                assert_eq!(*half_height, 1.0);
            }
            shape => panic!("expected cylinder shape, got {shape:?}"),
        }

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
    }

    #[test]
    fn first_half_space_stays_static_and_scaled_box_is_rejected() {
        let scene = Scene::from_ron(
            r#"
Scene(
    name: "collider_guard_fixture",
    entities: [
        (
            name: "Ground",
            transform: (
                translation: (0.0, 0.0, 0.0),
                rotation: (0.0, 0.0, 0.0, 1.0),
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
        {
            let store = runtime
                .engine_mut()
                .world_mut()
                .store_mut()
                .expect("engine carries a store");
            store.register::<ColliderDesc>();
            store.insert(
                entities[0],
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
        let ground = bodies.get(entities[0]).expect("ground body exists");
        assert_eq!(ground.mass, 0.0);
        assert!(matches!(&ground.shape, Shape::HalfSpace { .. }));
        assert!(
            !bodies.contains(entities[1]),
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
}
