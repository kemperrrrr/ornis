//! Manual timing probe for large physics steps. Criterion intentionally does
//! not measure this scale because the scene is too expensive on a clean
//! runner; this example prints per-step wall times and broadphase counters.
//!
//! Examples:
//!
//! ```text
//! cargo run -p ornis-physics --release --example probe_100k -- --sweep
//! cargo run -p ornis-physics --release --example probe_100k -- --grid --cell-size 4
//! cargo run -p ornis-physics --release --example probe_100k -- --grid --cell-size 8
//! ```

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
        "Usage: probe_100k [--sweep | --grid] [--cell-size SIZE] [--bodies N] [--steps N]"
    );
    println!("  --sweep              use the Sweep-and-Prune baseline (default)");
    println!("  --grid               use UniformGrid (default cell size: 4.0)");
    println!("  --cell-size SIZE     select UniformGrid and set its cell size");
    println!("  --bodies N           number of dynamic bodies (default: 100000)");
    println!("  --steps N             number of measured steps (default: 35)");
}

fn main() {
    let mut backend = BroadPhaseKind::SweepAndPrune;
    let mut cell_size = None;
    let mut bodies = 100_000u32;
    let mut steps = 35u32;

    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--sweep" => {
                backend = BroadPhaseKind::SweepAndPrune;
                cell_size = None;
            }
            "--grid" => backend = BroadPhaseKind::UniformGrid,
            "--cell-size" => {
                cell_size = Some(parse_value("--cell-size", args.next()));
                backend = BroadPhaseKind::UniformGrid;
            }
            "--bodies" => bodies = parse_value("--bodies", args.next()),
            "--steps" => steps = parse_value("--steps", args.next()),
            "--help" | "-h" => {
                print_usage();
                return;
            }
            unknown => panic!("unknown argument {unknown}; use --help for usage"),
        }
    }

    let selected_cell_size = cell_size.unwrap_or(4.0);
    let backend_name = match backend {
        BroadPhaseKind::SweepAndPrune => "sweep_and_prune",
        BroadPhaseKind::UniformGrid => "uniform_grid",
    };
    let setup_started = Instant::now();
    let mut physics = setup_body_grid(bodies);
    match backend {
        BroadPhaseKind::SweepAndPrune => physics.set_broadphase(BroadPhaseKind::SweepAndPrune),
        BroadPhaseKind::UniformGrid => physics.set_uniform_grid_cell_size(selected_cell_size),
    }
    println!(
        "probe: backend={backend_name} bodies={bodies} steps={steps} cell_size={selected_cell_size}"
    );
    println!("setup {bodies}: {:?}", setup_started.elapsed());

    for step in 0..steps {
        let started = Instant::now();
        physics.step(1.0 / 60.0);
        println!("step {step}: {:?}", started.elapsed());
        if step == 0 || step + 1 == steps {
            let stats = physics.broadphase_stats();
            println!(
                "stats after step {step}: bodies={} cells={} large={} pair_tests={} "
                "filter_rejections={} static_static_skips={} aabb_rejections={} candidates={}",
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
}
