//! Animation demo: the vendored starter mannequin
//! (`assets/starter/ual1_standard.glb`, Quaternius UAL-1, CC0) playing
//! its first moving clip in a native window.
//!
//! Run with: `cargo run --example anim` (`--frames N` exits after N
//! frames). Scene-first API only: the model loads through the asset
//! server (`AssetServer::load::<Scene>`), then an empty world spawns it
//! with explicit light, camera and play.

use ornis_app::GameWorld;
use ornis_assets::scene::Scene;
use ornis_assets::{AssetServer, Handle};
use ornis_runner::NativeOptions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut assets = AssetServer::new();
    let mannequin: Handle<Scene> = assets.load("assets/starter/ual1_standard.glb")?;
    let mut world = GameWorld::new();
    let hero = world.spawn_scene(&assets, &mannequin)?;
    world.set_ambient([0.1, 0.1, 0.15]);
    world.add_directional_light([1.0, 1.0, 1.0], 0.6, [1.0, 1.0, 1.0]);
    world.add_orbit_camera([2.5, 1.8, 3.5], [0.0, 1.0, 0.0]);
    let entity_count = u32::try_from(hero.mesh_entities.len())?;
    eprintln!("ornis: playing {entity_count} starter entities");
    world.play_all_animations();
    ornis_runner::run_native(
        move || (world, entity_count),
        NativeOptions::from_env("Ornis — Animation Demo"),
    )
}
