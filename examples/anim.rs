//! Animation demo (the engine's "fox"): the vendored starter character
//! (`assets/starter/ual1_standard.glb`, Quaternius UAL-1, CC0) with its
//! first clip autoplaying in a native window.
//!
//! Run with: `cargo run --example anim` (`--frames N` exits after N
//! frames). Shares the [`ornis_runner`] shell with the binary — no
//! duplicated winit/wgpu setup.

use ornis_app::{
    GameWorld, install_object_animation, install_skeletal_animation,
    physics_runtime::install_physics,
};
use ornis_render::{OrbitCamera, install_orbit_camera};

/// Builds the demo world: starter pack spawn + animation wiring.
fn demo_world() -> (GameWorld, u32) {
    let loaded = ornis_gltf::load_path(std::path::Path::new("assets/starter/ual1_standard.glb"))
        .expect("starter pack must load (assets/starter/ual1_standard.glb)");
    let mut runtime = GameWorld::default();
    let count = {
        let engine = runtime.engine_mut();
        let Some(store) = engine.world_mut().store_mut() else {
            return (runtime, 0);
        };
        let spawn = ornis_app::anim_wiring::spawn_gltf_world(store, &loaded);
        ornis_app::anim_wiring::wire_loaded_animation(store, &loaded, &spawn);
        spawn.entities.len() as u32
    };
    // Same DAG shape as the showcase (harmless without bodies); the
    // animation systems do the visible work.
    install_physics(runtime.engine_mut(), glam::Vec3::new(0.0, -9.81, 0.0));
    // Frame the ~1.8 m character at the origin; orbit input stays live.
    install_orbit_camera(
        runtime.engine_mut(),
        OrbitCamera::from_desc(&ornis_assets::scene::CameraDesc {
            position: [2.5, 1.8, 3.5],
            target: [0.0, 1.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 45.0,
            near: 0.1,
            far: 100.0,
        }),
    );
    install_object_animation(runtime.engine_mut());
    install_skeletal_animation(runtime.engine_mut());
    eprintln!("ornis: playing {count} starter entities");
    (runtime, count)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ornis_runner::run_native(
        demo_world,
        ornis_runner::NativeOptions::from_env("Ornis — Animation Demo"),
    )
}
