//! Benchmarks for the physics solver step and island configurations.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use glam::Vec3;

use ornis_physics::{BroadPhaseKind, PhysicsEngine, RigidBody, SequentialImpulseEngine};

/// World-space gravity shared by every scene.
const GRAVITY_Y: f32 = -9.81;
/// Fixed step shared by every bench.
const DT: f32 = 1.0 / 60.0;
/// Dynamic box half-extents.
const BOX_HALF: f32 = 0.4;
/// Horizontal pitch of resting boxes.
const PITCH_XZ: f32 = 2.0;
/// Vertical pitch of stacked boxes (half 0.4 + 0.01 settle gap).
const STACK_PITCH_Y: f32 = 0.81;
/// Rest height of a box sitting on a floor whose top face is y = 0.
const REST_Y: f32 = 0.4;
/// Floor half-extent on Y (top face at y = 0 when centered at -FLOOR_HALF_Y).
const FLOOR_HALF_Y: f32 = 0.5;
/// Tile-grid half-cell offset (centers tiles in their grid cells).
const TILE_CENTER: f32 = 0.5;
/// Island-field floor half-extents on XZ.
const GRID_FLOOR_HALF: f32 = 100.0;
/// Tall-stack floor half-extents on XZ.
const STACK_FLOOR_HALF: f32 = 10.0;
/// Body-grid tile half-extent on XZ.
const TILE_HALF: f32 = 5.0;
/// Boxes per island tower.
const TOWER_HEIGHT: u32 = 4;
/// Warmup/settle steps for resting scenes.
const SETTLE_STEPS: u32 = 60;
/// Shorter settle for body-scaling diagnostics.
const BODY_SETTLE_STEPS: u32 = 30;
/// Criterion sample size for body-scaling group.
const BODY_SAMPLE_SIZE: usize = 10;
/// Criterion warm-up for body-scaling group.
const BODY_WARMUP_SECS: u64 = 1;
/// Criterion measurement window for body-scaling group.
const BODY_MEASURE_SECS: u64 = 6;
/// Body counts in the broadphase scaling matrix.
const BODY_SCALE_COUNTS: [u32; 2] = [1_000, 10_000];

/// A GxG grid of independent 4-box stacks on one big static floor: many
/// disjoint islands — the best case for per-island parallel dispatch (G7).
fn setup_islands_grid(g: u32) -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -FLOOR_HALF_Y, 0.0),
        Vec3::new(GRID_FLOOR_HALF, FLOOR_HALF_Y, GRID_FLOOR_HALF),
        0.0,
    ));
    let half = Vec3::splat(BOX_HALF);
    for gx in 0..g {
        for gz in 0..g {
            let x = (gx as f32 - g as f32 / 2.0) * PITCH_XZ;
            let z = (gz as f32 - g as f32 / 2.0) * PITCH_XZ;
            for level in 0..TOWER_HEIGHT {
                physics.add_body(RigidBody::new_box(
                    Vec3::new(x, REST_Y + level as f32 * STACK_PITCH_Y, z),
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
fn setup_big_stack(n: u32) -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -FLOOR_HALF_Y, 0.0),
        Vec3::new(STACK_FLOOR_HALF, FLOOR_HALF_Y, STACK_FLOOR_HALF),
        0.0,
    ));
    for level in 0..n {
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, REST_Y + level as f32 * STACK_PITCH_Y, 0.0),
            Vec3::splat(BOX_HALF),
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
fn setup_body_grid(n: u32) -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    let side = (n as f32).sqrt().ceil() as u32;
    let span = side as f32 * PITCH_XZ;
    let tiles = (span / (2.0 * TILE_HALF)).ceil() as i32;
    for tx in 0..tiles {
        for tz in 0..tiles {
            let x = (tx as f32 - tiles as f32 / 2.0 + TILE_CENTER) * 2.0 * TILE_HALF;
            let z = (tz as f32 - tiles as f32 / 2.0 + TILE_CENTER) * 2.0 * TILE_HALF;
            physics.add_body(RigidBody::new_box(
                Vec3::new(x, -FLOOR_HALF_Y, z),
                Vec3::new(TILE_HALF, FLOOR_HALF_Y, TILE_HALF),
                0.0,
            ));
        }
    }
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = (gx as f32 - side as f32 / 2.0) * PITCH_XZ;
        let z = (gz as f32 - side as f32 / 2.0) * PITCH_XZ;
        physics.add_body(RigidBody::new_box(
            Vec3::new(x, REST_Y, z),
            Vec3::splat(BOX_HALF),
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
        for _ in 0..SETTLE_STEPS {
            physics.step(DT);
        }
        b.iter(|| std::hint::black_box(&mut physics).step(std::hint::black_box(DT)));
    });
    group.bench_function("big_stack_32", |b| {
        let mut physics = setup_big_stack(32);
        for _ in 0..SETTLE_STEPS {
            physics.step(DT);
        }
        b.iter(|| std::hint::black_box(&mut physics).step(std::hint::black_box(DT)));
    });
    group.bench_function("deep_stack_128", |b| {
        let mut physics = setup_big_stack(128);
        for _ in 0..SETTLE_STEPS {
            physics.step(DT);
        }
        b.iter(|| std::hint::black_box(&mut physics).step(std::hint::black_box(DT)));
    });
    group.finish();
}

fn settled_body_grid(
    bodies: u32,
    backend: BroadPhaseKind,
    cell_size: Option<f32>,
) -> SequentialImpulseEngine {
    let mut physics = setup_body_grid(bodies);
    if let Some(cell_size) = cell_size {
        physics.set_uniform_grid_cell_size(cell_size);
    } else {
        physics.set_broadphase(backend);
    }
    for _ in 0..BODY_SETTLE_STEPS {
        physics.step(DT);
    }
    physics
}

fn print_broadphase_stats(backend_name: &str, bodies: u32, physics: &SequentialImpulseEngine) {
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
    group.sample_size(BODY_SAMPLE_SIZE);
    group.warm_up_time(Duration::from_secs(BODY_WARMUP_SECS));
    group.measurement_time(Duration::from_secs(BODY_MEASURE_SECS));
    // The performance workflow is manual, so keep the larger cells in the
    // matrix while we locate the knee of the cell-size curve. This is still
    // intentionally not a production default or an adaptive policy.
    let configurations = vec![
        ("sweep_and_prune", BroadPhaseKind::SweepAndPrune, None),
        ("dynamic_aabb_tree", BroadPhaseKind::DynamicAabbTree, None),
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
        for n in BODY_SCALE_COUNTS {
            let diagnostic = settled_body_grid(n, backend, cell_size);
            print_broadphase_stats(backend_name, n, &diagnostic);
            group.bench_function(BenchmarkId::new(backend_name, n), |b| {
                let mut physics = settled_body_grid(n, backend, cell_size);
                b.iter(|| std::hint::black_box(&mut physics).step(std::hint::black_box(DT)));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_step, bench_body_scaling);
criterion_main!(benches);
