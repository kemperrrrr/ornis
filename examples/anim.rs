//! Animation demo: the vendored starter mannequin
//! (`assets/starter/ual1_standard.glb`, Quaternius UAL-1, CC0) playing
//! `Walk_Loop` by name in a native window.
//!
//! Run with: `cargo run --example anim` (`--frames N` exits after N
//! frames). Scene-first API only: the model loads through the asset
//! server (`AssetServer::load::<Scene>`), then an empty world spawns it
//! with explicit light, camera and a named play. The animator sits on
//! the first mesh entity until an entity-mutation handle exists.

use glam::Vec3;
use ornis_app::{GameWorld, try_animator};
use ornis_assets::scene::Scene;
use ornis_assets::{AssetServer, Handle};
use ornis_core::{Color, Degrees, Lux, UnitVec3};
use ornis_render::{DirectionalLight, OrbitCamera};
use ornis_runner::NativeOptions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut assets = AssetServer::new();
    let mannequin: Handle<Scene> = assets.load("assets/starter/ual1_standard.glb")?;
    let mut world = GameWorld::new();
    let hero = world.spawn_scene(&assets, &mannequin)?;
    world.set_ambient(Color::linear_rgb(0.1, 0.1, 0.15));
    world.spawn(DirectionalLight {
        direction: UnitVec3::new(Vec3::new(1.0, 1.0, 1.0))?,
        illuminance: Lux(0.6),
        color: Color::WHITE,
        ..Default::default()
    });
    world.spawn(
        OrbitCamera::looking_at(Vec3::new(2.5, 1.8, 3.5), Vec3::new(0.0, 1.0, 0.0))
            .with_fov(Degrees(45.0)),
    );
    let entity_count = u32::try_from(hero.mesh_entities.len())?;
    let root = hero.mesh_entities[0];
    {
        let store = world
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine always carries a store");
        try_animator(store, root)?.play("Walk_Loop")?;
    }
    eprintln!("ornis: playing Walk_Loop on {entity_count} starter entities");
    ornis_runner::run_native(
        move || (world, entity_count),
        NativeOptions::from_env("Ornis — Animation Demo"),
    )
}
