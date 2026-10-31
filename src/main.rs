//! `ornis` binary: native mode (winit + wgpu) and `editor-only`
//! (editor HTTP server on port 3420 without a native window).

#![warn(missing_docs)]

#[cfg(feature = "editor-only")]
use crossbeam_channel::unbounded;
#[cfg(feature = "editor-only")]
use editor_backend::RemoteEditor;
#[cfg(not(feature = "editor-only"))]
mod showcase_physics;

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
use ornis_gameplay::install_gameplay;
#[cfg(not(feature = "editor-only"))]
use ornis_render::OrbitCamera;
#[cfg(not(feature = "editor-only"))]
use showcase_physics::{install_scene_bodies, spawn_showcase_floor};

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
// winit window, wgpu rendering and the OpenPBR showcase scene.
// Animation demo: `cargo run --example anim` (starter character via
// [`ornis::run`], not a binary flag).
// The browser editor server is opt-in here: `cargo run -- --remote-editor`
// serves it on port 3420 (off by default — audit §6.2, backlog #16).
// The native UI overlay was removed with the ornis-ui crate — the editor
// lives in the browser (see editor-only mode / cargo xtask editor).
// ═══════════════════════════════════════════════════════════════════════════

/// Showcase world: RON scene plus physics bodies, audio and object
/// animation. `ornis::run` owns the window; content builders
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

#[cfg(not(feature = "editor-only"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    ornis::run(showcase_engine())
}
