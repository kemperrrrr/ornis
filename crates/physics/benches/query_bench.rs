//! Query-pipeline shootout (R5): one ray through a 10k-body grid, brute
//! force vs the live dynamic-tree traversal.
//!
//! Both arms answer the same query over the same snapshot: `brute` calls
//! [`QueryPipeline::cast_ray`](ornis_physics::QueryPipeline) directly over
//! the body slice, `tree` calls
//! [`SequentialImpulseEngine::query_cast_ray`](ornis_physics::SequentialImpulseEngine)
//! on an engine whose broadphase is [`BroadPhaseKind::DynamicAabbTree`] and
//! was stepped once (so the live tree is built). Results are identical by
//! construction (the tree only prunes); the bench measures the traversal
//! saving.

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use glam::{Quat, Vec3};

use ornis_physics::{
    BroadPhaseKind, PhysicsEngine, QueryFilter, QueryPipeline, Ray, RigidBody,
    SequentialImpulseEngine, Shape,
};

/// Bodies in the grid scene.
const GRID_N: u32 = 10_000;
/// Grid pitch (m).
const PITCH: f32 = 2.0;
/// Box half-extents.
const BOX_HALF: f32 = 0.4;
/// Ray travel distance (m).
const MAX_DIST: f32 = 500.0;

/// 100×100 grid of dynamic boxes plus one static floor, tree backend,
/// stepped once so the live tree matches the poses.
fn setup_grid() -> SequentialImpulseEngine {
    let mut engine = SequentialImpulseEngine::new(Vec3::ZERO);
    engine.set_broadphase(BroadPhaseKind::DynamicAabbTree);
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(200.0, 0.5, 200.0),
        0.0,
    ));
    let side = (GRID_N as f32).sqrt().ceil() as u32;
    for i in 0..GRID_N {
        let gx = i % side;
        let gz = i / side;
        engine.add_body(RigidBody::new_box(
            Vec3::new(
                (gx as f32 - side as f32 / 2.0) * PITCH,
                0.4,
                (gz as f32 - side as f32 / 2.0) * PITCH,
            ),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    engine.step(1.0 / 60.0);
    engine
}

/// Ray down the middle row of the grid (y = box-center height).
fn grid_ray() -> Ray {
    Ray::new(Vec3::new(-250.0, 0.4, 0.0), Vec3::X)
}

/// 100×100 grid of static quad meshes (2 triangles each) as horizontal
/// tiles, tree backend, stepped once. Mesh raycasts walk a per-body BVH —
/// the expensive kernel the tree pruning saves.
fn setup_mesh_grid() -> SequentialImpulseEngine {
    let mut engine = SequentialImpulseEngine::new(Vec3::ZERO);
    engine.set_broadphase(BroadPhaseKind::DynamicAabbTree);
    let side = (GRID_N as f32).sqrt().ceil() as u32;
    let verts = [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(-1.0, 0.0, 1.0),
    ];
    let tris = [
        ornis_physics::Triangle::from_raw([0, 2, 1]),
        ornis_physics::Triangle::from_raw([0, 3, 2]),
    ];
    for i in 0..GRID_N {
        let gx = i % side;
        let gz = i / side;
        engine.add_body(
            RigidBody::try_new_trimesh(
                Vec3::new(
                    (gx as f32 - side as f32 / 2.0) * PITCH,
                    0.0,
                    (gz as f32 - side as f32 / 2.0) * PITCH,
                ),
                &verts,
                &tris,
                0.0,
            )
            .expect("quad tile builds"),
        );
    }
    engine.step(1.0 / 60.0);
    engine
}

/// Ray straight down onto the middle tile of the mesh grid.
fn mesh_ray() -> Ray {
    Ray::new(Vec3::new(0.0, 50.0, 0.0), Vec3::NEG_Y)
}

fn query_ray_10k(criterion: &mut Criterion) {
    let engine = setup_grid();
    let ray = grid_ray();
    let filter = QueryFilter::default();
    // Sanity: both arms hit the same body before timing starts.
    let tree_hit = engine
        .query_cast_ray(&ray, MAX_DIST, &filter)
        .expect("valid ray")
        .expect("ray hits the grid row");
    let brute_hit = QueryPipeline::new()
        .cast_ray(&engine.bodies, &ray, MAX_DIST, &filter)
        .expect("valid ray")
        .expect("ray hits the grid row");
    assert_eq!(tree_hit.handle, brute_hit.handle);
    assert_eq!(tree_hit.distance, brute_hit.distance);

    criterion.bench_function("query_ray_10k/brute", |bench| {
        bench.iter(|| {
            std::hint::black_box(
                QueryPipeline::new()
                    .cast_ray(
                        std::hint::black_box(&engine.bodies),
                        std::hint::black_box(&ray),
                        MAX_DIST,
                        std::hint::black_box(&filter),
                    )
                    .expect("valid ray"),
            )
        });
    });
    criterion.bench_function("query_ray_10k/tree", |bench| {
        bench.iter(|| {
            std::hint::black_box(
                engine
                    .query_cast_ray(
                        std::hint::black_box(&ray),
                        MAX_DIST,
                        std::hint::black_box(&filter),
                    )
                    .expect("valid ray"),
            )
        });
    });
}

fn query_ray_10k_mesh(criterion: &mut Criterion) {
    let engine = setup_mesh_grid();
    let ray = mesh_ray();
    let filter = QueryFilter::default();
    // Sanity: both arms hit the same tile before timing starts.
    let tree_hit = engine
        .query_cast_ray(&ray, 100.0, &filter)
        .expect("valid ray")
        .expect("ray hits the middle tile");
    let brute_hit = QueryPipeline::new()
        .cast_ray(&engine.bodies, &ray, 100.0, &filter)
        .expect("valid ray")
        .expect("ray hits the middle tile");
    assert_eq!(tree_hit.handle, brute_hit.handle);
    assert_eq!(tree_hit.distance, brute_hit.distance);

    criterion.bench_function("query_ray_10k_mesh/brute", |bench| {
        bench.iter(|| {
            std::hint::black_box(
                QueryPipeline::new()
                    .cast_ray(
                        std::hint::black_box(&engine.bodies),
                        std::hint::black_box(&ray),
                        100.0,
                        std::hint::black_box(&filter),
                    )
                    .expect("valid ray"),
            )
        });
    });
    criterion.bench_function("query_ray_10k_mesh/tree", |bench| {
        bench.iter(|| {
            std::hint::black_box(
                engine
                    .query_cast_ray(
                        std::hint::black_box(&ray),
                        100.0,
                        std::hint::black_box(&filter),
                    )
                    .expect("valid ray"),
            )
        });
    });
}

/// Sphere sweep down the middle row of the box grid: the brute arm runs
/// the conservative-advancement distance oracle over all 10k bodies, the
/// tree arm only over the swept-box candidates.
fn query_shape_10k(criterion: &mut Criterion) {
    let engine = setup_grid();
    let mover = Shape::Sphere { radius: 0.25 };
    let start = Vec3::new(-250.0, 0.4, 0.0);
    let displacement = Vec3::new(500.0, 0.0, 0.0);
    let filter = QueryFilter::default();
    // Sanity: both arms hit the same body before timing starts.
    let tree_hit = engine
        .query_cast_shape(&mover, start, Quat::IDENTITY, displacement, &filter)
        .expect("sweep hits the grid row");
    let brute_hit = QueryPipeline::new()
        .cast_shape(
            &engine.bodies,
            &mover,
            start,
            Quat::IDENTITY,
            displacement,
            &filter,
        )
        .expect("sweep hits the grid row");
    assert_eq!(tree_hit.handle, brute_hit.handle);
    assert_eq!(tree_hit.distance, brute_hit.distance);

    criterion.bench_function("query_shape_10k/brute", |bench| {
        bench.iter(|| {
            std::hint::black_box(QueryPipeline::new().cast_shape(
                std::hint::black_box(&engine.bodies),
                std::hint::black_box(&mover),
                std::hint::black_box(start),
                Quat::IDENTITY,
                std::hint::black_box(displacement),
                std::hint::black_box(&filter),
            ))
        });
    });
    criterion.bench_function("query_shape_10k/tree", |bench| {
        bench.iter(|| {
            std::hint::black_box(engine.query_cast_shape(
                std::hint::black_box(&mover),
                std::hint::black_box(start),
                Quat::IDENTITY,
                std::hint::black_box(displacement),
                std::hint::black_box(&filter),
            ))
        });
    });
}

criterion_group! {
    name = query_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3))
        .sample_size(20);
    targets = query_ray_10k, query_ray_10k_mesh, query_shape_10k
}
criterion_main!(query_benches);
