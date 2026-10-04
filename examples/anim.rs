//! Animation demo: the vendored starter mannequin
//! (`assets/starter/ual1_standard.glb`, Quaternius UAL-1, CC0) playing
//! `Walk_Loop` by name in a native window.
//!
//! Run with: `cargo run --example anim` (`--frames N` exits after N
//! frames).

use glam::Vec3;
use ornis_app::{AnimatorAccess, GameWorld};
use ornis_assets::{AssetServer, Handle, Model};
use ornis_core::{Color, Degrees, Lux, UnitVec3};
use ornis_render::{DirectionalLight, OrbitCamera};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut assets = AssetServer::new();
    let mannequin: Handle<Model> = assets.load("assets/starter/ual1_standard.glb")?;
    let mut world = GameWorld::new();
    let hero = world.spawn_model(&assets, &mannequin)?;
    world.set_ambient(Color::hex("#1A1A26")?);
    world.spawn(DirectionalLight {
        direction: UnitVec3::new(Vec3::new(-1.0, -1.0, -1.0))?,
        illuminance: Lux(0.6),
        color: Color::WHITE,
        ..Default::default()
    });
    world.spawn(OrbitCamera::looking_at(Vec3::new(2.5, 1.8, 3.5), Vec3::Y).with_fov(Degrees(45.0)));
    world.entity_mut(hero).animator()?.play("Walk_Loop")?;
    ornis_runner::run_native(world, "Ornis — Animation Demo")
}
