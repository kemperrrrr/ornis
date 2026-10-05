//! Animation demo: the vendored starter mannequin
//! (`assets/starter/ual1_standard.glb`, Quaternius UAL-1, CC0) playing
//! `Walk_Loop` by name in a native window.
//!
//! Run with: `cargo run --example anim` (`--frames N` exits after N
//! frames).

use glam::Vec3;
use ornis_app::{AnimatorAccess, GameWorld};
use ornis_assets::Model;
use ornis_core::{Color, Degrees, Lux, UnitVec3};
use ornis_render::{DirectionalLight, OrbitCamera};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = GameWorld::new();
    world.set_title("Ornis — Animation Demo");
    let mannequin = world.load::<Model>("assets/starter/ual1_standard.glb")?;
    let hero = world.spawn(mannequin)?;
    // `direction` points toward the light. The key sits in the camera
    // octant so the front is lit; the fill and linear ambient keep the
    // far side readable. `#1A1A26` is sRGB, so it uploaded as ~0.01 linear.
    world.set_ambient(Color::linear_rgb(0.10, 0.10, 0.15));
    world.spawn(DirectionalLight {
        direction: UnitVec3::new(Vec3::new(1.0, 1.0, 1.0))?,
        illuminance: Lux(0.6),
        color: Color::WHITE,
        ..Default::default()
    });
    world.spawn(DirectionalLight {
        direction: UnitVec3::new(Vec3::new(-0.5, 0.5, -0.5))?,
        illuminance: Lux(0.3),
        color: Color::linear_rgb(0.8, 0.8, 1.0),
        ..Default::default()
    });
    world.spawn(OrbitCamera::looking_at(Vec3::new(2.5, 1.8, 3.5), Vec3::Y).with_fov(Degrees(45.0)));
    world.entity_mut(hero).animator()?.play("Walk_Loop")?;
    ornis::run(world)
}
