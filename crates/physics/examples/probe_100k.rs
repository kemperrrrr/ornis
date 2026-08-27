//! Manual timing probe for the 100k-bodies physics step: criterion cannot
//! measure it (superlinear step time at this scale), so this prints
//! per-step wall times for the baseline document.

use std::time::Instant;

use glam::Vec3;
use ornis_physics::{BuiltinPhysicsEngine, PhysicsEngine, RigidBody};

fn setup_body_grid(n: u32) -> BuiltinPhysicsEngine {
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let side = (n as f32).sqrt().ceil() as u32;
    let span = side as f32 * 2.0;
    let tile_half = 5.0f32;
    let tiles = (span / (2.0 * tile_half)).ceil() as i32;
    for tx in 0..tiles {
        for tz in 0..tiles {
            let x = (tx as f32 - tiles as f32 / 2.0 + 0.5) * 2.0 * tile_half;
            let z = (tz as f32 - tiles as f32 / 2.0 + 0.5) * 2.0 * tile_half;
            physics.add_body(RigidBody::new_box(
                Vec3::new(x, -0.5, z),
                Vec3::new(tile_half, 0.5, tile_half),
                0.0,
            ));
        }
    }
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = (gx as f32 - side as f32 / 2.0) * 2.0;
        let z = (gz as f32 - side as f32 / 2.0) * 2.0;
        physics.add_body(RigidBody::new_box(
            Vec3::new(x, 0.4, z),
            Vec3::splat(0.4),
            1.0,
        ));
    }
    physics
}

fn main() {
    let n = 100_000u32;
    let t = Instant::now();
    let mut physics = setup_body_grid(n);
    println!("setup {n}: {:?}", t.elapsed());
    for i in 0..35 {
        let t = Instant::now();
        physics.step(1.0 / 60.0);
        println!("step {i}: {:?}", t.elapsed());
    }
}
