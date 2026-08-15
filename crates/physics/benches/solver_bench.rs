use criterion::{Criterion, black_box, criterion_group, criterion_main};
use glam::Vec3;

use ornis_physics::{BuiltinPhysicsEngine, PhysicsEngine, RigidBody};

/// A GxG grid of independent 4-box stacks on one big static floor: many
/// disjoint islands — the best case for per-island parallel dispatch (G7).
fn setup_islands_grid(g: u32) -> BuiltinPhysicsEngine {
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(100.0, 0.5, 100.0),
        0.0,
    ));
    let half = Vec3::splat(0.4);
    let pitch = 2.0;
    for gx in 0..g {
        for gz in 0..g {
            let x = (gx as f32 - g as f32 / 2.0) * pitch;
            let z = (gz as f32 - g as f32 / 2.0) * pitch;
            for level in 0..4 {
                physics.add_body(RigidBody::new_box(
                    Vec3::new(x, 0.4 + level as f32 * 0.81, z),
                    half,
                    1.0,
                ));
            }
        }
    }
    physics
}

/// One tall stack: a single island — the worst case for per-island dispatch
/// (measures gather/scatter overhead against the old monolithic solve).
fn setup_big_stack(n: u32) -> BuiltinPhysicsEngine {
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    for level in 0..n {
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.4 + level as f32 * 0.81, 0.0),
            Vec3::splat(0.4),
            1.0,
        ));
    }
    physics
}

fn bench_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("physics_step");
    group.bench_function("islands_grid_16x16", |b| {
        let mut physics = setup_islands_grid(16);
        // Settle the scene so the measured step is the resting steady state.
        for _ in 0..60 {
            physics.step(1.0 / 60.0);
        }
        b.iter(|| black_box(&mut physics).step(black_box(1.0 / 60.0)));
    });
    group.bench_function("big_stack_32", |b| {
        let mut physics = setup_big_stack(32);
        for _ in 0..60 {
            physics.step(1.0 / 60.0);
        }
        b.iter(|| black_box(&mut physics).step(black_box(1.0 / 60.0)));
    });
    group.finish();
}

criterion_group!(benches, bench_step);
criterion_main!(benches);
