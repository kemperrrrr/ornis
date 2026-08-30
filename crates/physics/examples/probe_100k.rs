//! Manual timing probe for large physics steps. Criterion intentionally does
//! not measure this scale because the scene is too expensive on a clean
//! runner; this example prints per-step wall times and broadphase counters
//! for a chosen scene + backend combination.
//!
//! Run locally in release for realistic numbers:
//!
//! ```text
//! cargo run -p ornis-physics --release --example probe_100k -- --sweep --scene tiled --bodies 10000
//! cargo run -p ornis-physics --release --example probe_100k -- --grid --cell-size 8 --scene tiled
//! cargo run -p ornis-physics --release --example probe_100k -- --tree --scene giant_floor
//! ```
//!
//! Scenes: tiled (regular floor grid + resting dynamic), giant_floor (one
//! huge static floor), sparse (dynamic bodies far apart), islands (dense
//! stacked clusters), heterogeneous (mixed shapes/sizes).

use std::time::Instant;

use glam::Vec3;
use ornis_physics::{BroadPhaseKind, BuiltinPhysicsEngine, PhysicsEngine, RigidBody};

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

/// One huge static floor + `n` dynamic boxes resting above it. Stresses the
/// large-static-AABB path that makes Sweep-and-Prune quadratic.
fn setup_giant_floor(n: u32) -> BuiltinPhysicsEngine {
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::splat(500.0),
        0.0,
    ));
    let side = (n as f32).sqrt().ceil() as u32;
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = (gx as f32 - side as f32 / 2.0) * 2.0;
        let z = (gz as f32 - side as f32 / 2.0) * 2.0;
        let y = 1.0 + (i % 6) as f32;
        physics.add_body(RigidBody::new_box(
            Vec3::new(x, y, z),
            Vec3::splat(0.4),
            1.0,
        ));
    }
    physics
}

/// `n` dynamic bodies spread far apart so almost no pairs overlap.
fn setup_sparse(n: u32) -> BuiltinPhysicsEngine {
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let side = (n as f32).sqrt().ceil() as u32;
    let spacing = 20.0f32;
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = gx as f32 * spacing;
        let z = gz as f32 * spacing;
        physics.add_body(RigidBody::new_box(
            Vec3::new(x, 5.0, z),
            Vec3::splat(0.4),
            1.0,
        ));
    }
    physics
}

/// `n` dynamic bodies arranged in dense stacked clusters (islands), isolated
/// from each other. Stresses clustering behaviour of each backend.
fn setup_islands(n: u32) -> BuiltinPhysicsEngine {
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let per = 10u32;
    let islands = (n as f32 / per as f32).ceil() as u32;
    let cluster_spacing = 4u32;
    for c in 0..islands {
        if c * per >= n {
            break;
        }
        let cx = (c % cluster_spacing) as f32 * 4.0;
        let cz = (c / cluster_spacing) as f32 * 4.0;
        for k in 0..per {
            if c * per + k >= n {
                break;
            }
            physics.add_body(RigidBody::new_box(
                Vec3::new(cx, k as f32 + 0.5, cz),
                Vec3::splat(0.4),
                1.0,
            ));
        }
    }
    physics
}

/// `n` dynamic bodies of mixed shape and size on a regular grid.
fn setup_heterogeneous(n: u32) -> BuiltinPhysicsEngine {
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let side = (n as f32).sqrt().ceil() as u32;
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = (gx as f32 - side as f32 / 2.0) * 2.0;
        let z = (gz as f32 - side as f32 / 2.0) * 2.0;
        let s = 0.3 + (i % 7) as f32 * 0.15;
        if i % 2 == 0 {
            physics.add_body(RigidBody::new_box(
                Vec3::new(x, 1.0, z),
                Vec3::splat(s),
                1.0,
            ));
        } else {
            physics.add_body(RigidBody::new_sphere(Vec3::new(x, 1.0, z), s, 1.0));
        }
    }
    physics
}

fn parse_value<T>(flag: &str, value: Option<String>) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = value.unwrap_or_else(|| panic!("{flag} requires a value"));
    value
        .parse()
        .unwrap_or_else(|error| panic!("invalid value for {flag}: {error}"))
}

fn print_usage() {
    println!(
        "Usage: probe_100k [--sweep | --grid | --tree] [--cell-size SIZE] [--scene NAME] [--bodies N] [--steps N]"
    );
    println!("  --sweep              use the Sweep-and-Prune baseline (default)");
    println!("  --grid               use UniformGrid (default cell size: 4.0)");
    println!("  --tree               use the experimental DynamicAabbTree backend");
    println!("  --cell-size SIZE     select UniformGrid and set its cell size");
    println!(
        "  --scene NAME         tiled | giant_floor | sparse | islands | heterogeneous (default: tiled)"
    );
    println!("  --bodies N           number of dynamic bodies (default: 10000)");
    println!("  --steps N             number of measured steps (default: 20)");
}

fn run_probe(backend: BroadPhaseKind, cell_size: f32, scene: &str, bodies: u32, steps: u32) {
    let backend_name = match backend {
        BroadPhaseKind::SweepAndPrune => "sweep_and_prune",
        BroadPhaseKind::UniformGrid => "uniform_grid",
        BroadPhaseKind::DynamicAabbTree => "dynamic_aabb_tree",
    };
    let setup_started = Instant::now();
    let mut physics = match scene {
        "tiled" => setup_body_grid(bodies),
        "giant_floor" => setup_giant_floor(bodies),
        "sparse" => setup_sparse(bodies),
        "islands" => setup_islands(bodies),
        "heterogeneous" => setup_heterogeneous(bodies),
        other => panic!("unknown scene {other}; use --help for usage"),
    };
    match backend {
        BroadPhaseKind::SweepAndPrune => physics.set_broadphase(BroadPhaseKind::SweepAndPrune),
        BroadPhaseKind::UniformGrid => physics.set_uniform_grid_cell_size(cell_size),
        BroadPhaseKind::DynamicAabbTree => physics.set_broadphase(BroadPhaseKind::DynamicAabbTree),
    }
    println!(
        "probe: backend={backend_name} scene={scene} bodies={bodies} steps={steps} cell_size={cell_size}"
    );
    println!("setup {bodies}: {:?}", setup_started.elapsed());

    let mut steady = Vec::new();
    for step in 0..steps {
        let started = Instant::now();
        physics.step(1.0 / 60.0);
        let elapsed = started.elapsed();
        println!("step {step}: {elapsed:?}");
        // Skip the first step (broadphase warm-up / initial pair build).
        if step > 0 {
            steady.push(elapsed);
        }
        if step == 0 || step + 1 == steps {
            let stats = physics.broadphase_stats();
            println!(
                concat!(
                    "stats after step {}: bodies={} cells={} large={} pair_tests={} ",
                    "filter_rejections={} static_static_skips={} aabb_rejections={} candidates={}"
                ),
                step,
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
    }
    if !steady.is_empty() {
        let sum: f64 = steady.iter().map(|d| d.as_secs_f64()).sum();
        let mean = sum / steady.len() as f64;
        println!(
            "mean steady-state step ({}/{} steps): {:.3} ms/step",
            steady.len(),
            steps,
            mean * 1000.0
        );
    }
}

fn main() {
    let mut backend = BroadPhaseKind::SweepAndPrune;
    let mut cell_size = None;
    let mut scene = "tiled".to_string();
    let mut bodies = 10_000u32;
    let mut steps = 20u32;

    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--sweep" => {
                backend = BroadPhaseKind::SweepAndPrune;
                cell_size = None;
            }
            "--grid" => backend = BroadPhaseKind::UniformGrid,
            "--tree" => backend = BroadPhaseKind::DynamicAabbTree,
            "--cell-size" => {
                cell_size = Some(parse_value("--cell-size", args.next()));
                backend = BroadPhaseKind::UniformGrid;
            }
            "--scene" => scene = parse_value("--scene", args.next()),
            "--bodies" => bodies = parse_value("--bodies", args.next()),
            "--steps" => steps = parse_value("--steps", args.next()),
            "--help" | "-h" => {
                print_usage();
                return;
            }
            unknown => panic!("unknown argument {unknown}; use --help for usage"),
        }
    }
    run_probe(backend, cell_size.unwrap_or(4.0), &scene, bodies, steps);
}
