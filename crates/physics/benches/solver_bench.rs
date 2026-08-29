use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use glam::Vec3;

use ornis_physics::{BroadPhaseKind, BuiltinPhysicsEngine, PhysicsEngine, RigidBody};

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

/// N single dynamic boxes resting on a static floor in a sparse grid:
/// body-count scaling with minimal contact pairs (broadphase-dominated).
/// The floor is tiled (10×10 tiles) instead of one huge AABB: a single
/// floor box overlapping every body degenerates Sweep-and-Prune to O(n²)
/// (measured 2026-08-27: ~48 s/step at 100k bodies, sleep never settles).
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
    group.bench_function("deep_stack_128", |b| {
        let mut physics = setup_big_stack(128);
        for _ in 0..60 {
            physics.step(1.0 / 60.0);
        }
        b.iter(|| black_box(&mut physics).step(black_box(1.0 / 60.0)));
    });
    group.finish();
}

fn settled_body_grid(
    bodies: u32,
    backend: BroadPhaseKind,
    cell_size: Option<f32>,
) -> BuiltinPhysicsEngine {
    let mut physics = setup_body_grid(bodies);
    if let Some(cell_size) = cell_size {
        physics.set_uniform_grid_cell_size(cell_size);
    } else {
        physics.set_broadphase(backend);
    }
    for _ in 0..30 {
        physics.step(1.0 / 60.0);
    }
    physics
}

fn print_broadphase_stats(backend_name: &str, bodies: u32, physics: &BuiltinPhysicsEngine) {
    let stats = physics.broadphase_stats();
    eprintln!(
        concat!(
            "broadphase/{}/{}: bodies={} cells={} large={} pair_tests={} ",
            "filter_rejections={} static_static_skips={} aabb_rejections={} candidates={}"
        ),
        backend_name,
        bodies,
        stats.body_count,
        stats.occupied_cells,
        stats.large_bodies,
        stats.pair_tests,
        stats.filter_rejections,
        stats.static_static_skips,
        stats.aabb_rejections,
        stats.candidate_pairs,
    );
}

/// Body-count scaling: 1k / 10k dynamic bodies in one `step`.
/// 100k is intentionally not a criterion bench: the step is superlinear
/// there (2026-08-27: single huge floor AABB degenerates Sweep-and-Prune to
/// O(n²) at ~48 s/step; with a tiled floor a criterion warmup step still
/// exceeded 30 min). 100k numbers come from a manual probe — see
/// docs/quality/perf-baseline-2026-08-27.md.
fn bench_body_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("physics_bodies");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));
    // The performance workflow is manual, so keep the larger cells in the
    // matrix while we locate the knee of the cell-size curve. This is still
    // intentionally not a production default or an adaptive policy.
    let configurations = vec![
        ("sweep_and_prune", BroadPhaseKind::SweepAndPrune, None),
        (
            "uniform_grid_cell_1",
            BroadPhaseKind::UniformGrid,
            Some(1.0),
        ),
        (
            "uniform_grid_cell_2",
            BroadPhaseKind::UniformGrid,
            Some(2.0),
        ),
        (
            "uniform_grid_cell_4",
            BroadPhaseKind::UniformGrid,
            Some(4.0),
        ),
        (
            "uniform_grid_cell_8",
            BroadPhaseKind::UniformGrid,
            Some(8.0),
        ),
        (
            "uniform_grid_cell_16",
            BroadPhaseKind::UniformGrid,
            Some(16.0),
        ),
    ];
    for (backend_name, backend, cell_size) in configurations {
        for n in [1_000u32, 10_000] {
            let diagnostic = settled_body_grid(n, backend, cell_size);
            print_broadphase_stats(backend_name, n, &diagnostic);
            group.bench_function(BenchmarkId::new(backend_name, n), |b| {
                let mut physics = settled_body_grid(n, backend, cell_size);
                b.iter(|| black_box(&mut physics).step(black_box(1.0 / 60.0)));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_step, bench_body_scaling);
criterion_main!(benches);
