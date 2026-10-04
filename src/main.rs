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
#[cfg(not(feature = "editor-only"))]
use ornis_runner::{NativeOptions, run_native};

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
// the shared [`ornis_runner`] shell, not a binary flag).
// The browser editor server is opt-in here: `cargo run -- --remote-editor`
// serves it on port 3420 (off by default — audit §6.2, backlog #16).
// The native UI overlay was removed with the ornis-ui crate — the editor
// lives in the browser (see editor-only mode / cargo xtask editor).
// ═══════════════════════════════════════════════════════════════════════════

/// Spheres showcase world: RON scene plus physics bodies, audio and
/// object animation. The shell ([`ornis_runner::run_native`]) owns the
/// window; content builders like this one stay here.
#[cfg(not(feature = "editor-only"))]
fn showcase_engine() -> (GameWorld, u32) {
    let Ok(scene) = Scene::from_ron(include_str!("../assets/demo_scene.ron")) else {
        let runtime = GameWorld::default();
        return (runtime, 0);
    };
    let entity_count = scene.entities.len() as u32;
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
    {
        let entities = runtime.entities().to_vec();
        let Some(store) = runtime.engine_mut().world_mut().store_mut() else {
            return (runtime, entity_count);
        };
        for (index, entity) in entities.into_iter().enumerate() {
            let description = &scene.entities[index];
            let radius = match &description.mesh {
                ornis_assets::scene::MeshDesc::Sphere { radius, .. } => radius.get(),
                // Custom/Box/Plane/Cylinder need an explicit validated collider recipe.
                ornis_assets::scene::MeshDesc::Custom { .. }
                | ornis_assets::scene::MeshDesc::Box { .. }
                | ornis_assets::scene::MeshDesc::Plane { .. }
                | ornis_assets::scene::MeshDesc::Cylinder { .. } => continue,
            };
            let mass = if index == 0 { 1.0 } else { 0.0 };
            store.insert(
                entity,
                RigidBody::new_sphere(description.transform.translation, radius, mass),
            );
        }
    }
    // Hidden static floor: physics only, so it stays out of the frame upload.
    if let Err(error) = spawn_showcase_floor(runtime.engine_mut()) {
        eprintln!("showcase floor was not spawned: {error}");
    }
    runtime.frame(0.0);
    (runtime, entity_count)
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
    run_native(showcase_engine, NativeOptions::from_env("Ornis Engine"))
}
