//! Animation demo: the vendored starter mannequin
//! (`assets/starter/ual1_standard.glb`, Quaternius UAL-1, CC0) playing
//! its first moving clip in a native window.
//!
//! Run with: `cargo run --example anim` (`--frames N` exits after N
//! frames). Scene-first API only: empty world, loaded asset, explicit
//! light and camera, explicit play.

use ornis_app::GameWorld;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = GameWorld::new();
    let spawned = world.spawn_gltf(std::path::Path::new("assets/starter/ual1_standard.glb"))?;
    world.set_ambient([0.1, 0.1, 0.15]);
    world.add_directional_light([1.0, 1.0, 1.0], 0.6, [1.0, 1.0, 1.0]);
    world.add_orbit_camera([2.5, 1.8, 3.5], [0.0, 1.0, 0.0]);
    eprintln!(
        "ornis: playing {} starter entities",
        spawned.mesh_entities.len()
    );
    world.play_all_animations();
    ornis_runner::run_native(world, "Ornis — Animation Demo")
}
