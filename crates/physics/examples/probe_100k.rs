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
use ornis_physics::{
    BodyHandle, BodyType, BroadPhaseKind, PhysicsEngine, RigidBody, SequentialImpulseEngine,
};

/// Earth-surface gravity along −Y (m/s²).
const GRAVITY_Y: f32 = -9.81;
/// Midpoint / half-extent scale.
const HALF: f32 = 0.5;
/// Floor box half-height / Y center (m).
const FLOOR_HALF_Y: f32 = HALF;
/// Tiled-floor tile half-extent (m).
const TILE_HALF: f32 = 5.0;
/// Dynamic box half-extent (m).
const BOX_HALF: f32 = 0.4;
/// Giant-floor half-extent (m).
const GIANT_FLOOR_HALF: f32 = 500.0;
/// Sparse-scene body spacing (m).
const SPARSE_SPACING: f32 = 20.0;
/// Sparse-scene spawn height (m).
const SPARSE_HEIGHT: f32 = 5.0;
/// Island cluster pitch (m).
const ISLAND_PITCH: f32 = 4.0;
/// Dynamic bodies packed into each islands-scene cluster.
const BODIES_PER_ISLAND: u32 = 10;
/// Heterogeneous size base / step (m).
const HETERO_SIZE_BASE: f32 = 0.3;
const HETERO_SIZE_STEP: f32 = 0.15;
/// Heterogeneous size period.
const HETERO_SIZE_PERIOD: u32 = 7;
/// Giant-floor Y jitter period.
const GIANT_Y_PERIOD: u32 = 6;
/// Fixed simulation timestep (s).
const DT: f32 = 1.0 / 60.0;
/// Cull dynamic bodies that fell below this Y (m).
const FALLEN_CULL_Y: f32 = -10.0;
/// Default dynamic body count.
const DEFAULT_BODIES: u32 = 10_000;
/// Default measured steps.
const DEFAULT_STEPS: u32 = 20;
/// Default uniform-grid cell size (m).
const DEFAULT_CELL_SIZE: f32 = 4.0;
/// Island clusters per XZ row when laying out the islands scene.
const ISLAND_CLUSTER_COLS: u32 = 4;

fn setup_body_grid(n: u32) -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    let side = (n as f32).sqrt().ceil() as u32;
    let span = side as f32 * 2.0;
    let tile_half = TILE_HALF;
    let tiles = (span / (2.0 * tile_half)).ceil() as i32;
    for tx in 0..tiles {
        for tz in 0..tiles {
            let x = (tx as f32 - tiles as f32 / 2.0 + HALF) * 2.0 * tile_half;
            let z = (tz as f32 - tiles as f32 / 2.0 + HALF) * 2.0 * tile_half;
            physics.add_body(RigidBody::new_box(
                Vec3::new(x, -FLOOR_HALF_Y, z),
                Vec3::new(tile_half, FLOOR_HALF_Y, tile_half),
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
            Vec3::new(x, BOX_HALF, z),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    physics
}

/// One huge static floor + `n` dynamic boxes resting above it. Stresses the
/// large-static-AABB path that makes Sweep-and-Prune quadratic.
fn setup_giant_floor(n: u32) -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -FLOOR_HALF_Y, 0.0),
        Vec3::splat(GIANT_FLOOR_HALF),
        0.0,
    ));
    let side = (n as f32).sqrt().ceil() as u32;
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = (gx as f32 - side as f32 / 2.0) * 2.0;
        let z = (gz as f32 - side as f32 / 2.0) * 2.0;
        let y = 1.0 + (i % GIANT_Y_PERIOD) as f32;
        physics.add_body(RigidBody::new_box(
            Vec3::new(x, y, z),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    physics
}

/// `n` dynamic bodies spread far apart so almost no pairs overlap.
fn setup_sparse(n: u32) -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    let side = (n as f32).sqrt().ceil() as u32;
    let spacing = SPARSE_SPACING;
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = gx as f32 * spacing;
        let z = gz as f32 * spacing;
        physics.add_body(RigidBody::new_box(
            Vec3::new(x, SPARSE_HEIGHT, z),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    physics
}

/// `n` dynamic bodies arranged in dense stacked clusters (islands), isolated
/// from each other. Stresses clustering behaviour of each backend.
fn setup_islands(n: u32) -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    let per = BODIES_PER_ISLAND;
    let islands = (n as f32 / per as f32).ceil() as u32;
    for c in 0..islands {
        if c * per >= n {
            break;
        }
        let cx = (c % ISLAND_CLUSTER_COLS) as f32 * ISLAND_PITCH;
        let cz = (c / ISLAND_CLUSTER_COLS) as f32 * ISLAND_PITCH;
        for k in 0..per {
            if c * per + k >= n {
                break;
            }
            physics.add_body(RigidBody::new_box(
                Vec3::new(cx, k as f32 + HALF, cz),
                Vec3::splat(BOX_HALF),
                1.0,
            ));
        }
    }
    physics
}

/// `n` dynamic bodies of mixed shape and size on a regular grid.
fn setup_heterogeneous(n: u32) -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    let side = (n as f32).sqrt().ceil() as u32;
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = (gx as f32 - side as f32 / 2.0) * 2.0;
        let z = (gz as f32 - side as f32 / 2.0) * 2.0;
        let s = HETERO_SIZE_BASE + (i % HETERO_SIZE_PERIOD) as f32 * HETERO_SIZE_STEP;
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
        "Usage: probe_100k [--sweep | --grid | --tree | --auto] [--cell-size SIZE] [--scene NAME] [--bodies N] [--steps N]"
    );
    println!("  --sweep              use the Sweep-and-Prune baseline (default)");
    println!("  --grid               use UniformGrid (default cell size: 4.0)");
    println!("  --tree               use the experimental DynamicAabbTree backend");
    println!("  --auto               analytic SweepAndPrune <-> UniformGrid routing");
    println!("  --cell-size SIZE     select UniformGrid and set its cell size");
    println!(
        "  --scene NAME         tiled | giant_floor | sparse | islands | heterogeneous (default: tiled)"
    );
    println!("  --bodies N           number of dynamic bodies (default: 10000)");
    println!("  --steps N             number of measured steps (default: 20)");
}

fn run_probe(
    backend: BroadPhaseKind,
    cell_size: f32,
    scene: &str,
    bodies: u32,
    steps: u32,
    kill_plane: bool,
) {
    let backend_name = match backend {
        BroadPhaseKind::SweepAndPrune => "sweep_and_prune",
        BroadPhaseKind::UniformGrid => "uniform_grid",
        BroadPhaseKind::DynamicAabbTree => "dynamic_aabb_tree",
        BroadPhaseKind::Auto => "auto",
    };
    let setup_started = Instant::now();
    let mut physics = match scene {
        "tiled" => setup_body_grid(bodies),
        "giant_floor" => setup_giant_floor(bodies),
        "sparse" => setup_sparse(bodies),
        "islands" => setup_islands(bodies),
        "heterogeneous" => setup_heterogeneous(bodies),
        other => {
            eprintln!("unknown scene {other}; use --help for usage");
            return;
        }
    };
    match backend {
        BroadPhaseKind::SweepAndPrune => physics.set_broadphase(BroadPhaseKind::SweepAndPrune),
        BroadPhaseKind::UniformGrid => physics.set_uniform_grid_cell_size(cell_size),
        BroadPhaseKind::DynamicAabbTree => physics.set_broadphase(BroadPhaseKind::DynamicAabbTree),
        BroadPhaseKind::Auto => physics.set_broadphase(BroadPhaseKind::Auto),
    }
    println!(
        "probe: backend={backend_name} scene={scene} bodies={bodies} steps={steps} cell_size={cell_size}"
    );
    println!("setup {bodies}: {:?}", setup_started.elapsed());

    let mut steady = Vec::new();
    for step in 0..steps {
        let started = Instant::now();
        physics.step(DT);
        let elapsed = started.elapsed();
        println!("step {step}: {elapsed:?}");
        // Kill-plane experiment (substep-tax probe): remove dynamics fallen
        // below y=-10 so perpetual fallers stop forcing global substeps.
        // Collect-then-remove-backwards: swap_remove shifts the tail.
        if kill_plane {
            let stats_now = physics.broadphase_stats();
            let mut fallen = Vec::new();
            for h in 0..stats_now.body_count {
                let Some(b) = physics.get_body(BodyHandle::from(h)) else {
                    continue;
                };
                if b.body_type == BodyType::Dynamic && b.position.y < FALLEN_CULL_Y {
                    fallen.push(h);
                }
            }
            for h in fallen.into_iter().rev() {
                physics.remove_body(BodyHandle::from(h));
            }
        }
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
            let timing = physics.step_timing();
            println!(
                "timing after step {}: broad={:.2}ms narrow={:.2}ms solver={:.2}ms island={:.2}ms trigger={:.2}ms substeps={}",
                step,
                timing.broad_phase_ms,
                timing.narrow_phase_ms,
                timing.solver_ms,
                timing.island_ms,
                timing.trigger_ms,
                timing.substeps,
            );
            // Sleeping-world diagnostics: chunk-sleep work starts from how
            // much of a settled scene the island sleeper actually freezes.
            let asleep = (0..stats.body_count)
                .filter(|&h| physics.is_asleep(BodyHandle::from(h)))
                .count();
            println!(
                "sleep after step {step}: asleep={asleep}/{}",
                stats.body_count
            );
            // Awake-dynamic survey: who keeps a settled scene at 12 substeps?
            let mut awake_n = 0u32;
            let mut max_v = 0.0f32;
            let mut max_w = 0.0f32;
            for h in 0..stats.body_count {
                if physics.is_asleep(BodyHandle::from(h)) {
                    continue;
                }
                let Some(b) = physics.get_body(BodyHandle::from(h)) else {
                    continue;
                };
                if b.body_type != BodyType::Dynamic {
                    continue;
                }
                awake_n += 1;
                max_v = max_v.max(b.velocity.length());
                max_w = max_w.max(b.angular_velocity.length());
            }
            println!(
                "awake dynamics after step {step}: n={awake_n} max_v={max_v:.3} max_w={max_w:.3}"
            );
            let mut min_y = f32::INFINITY;
            for h in 0..stats.body_count {
                if physics.is_asleep(BodyHandle::from(h)) {
                    continue;
                }
                let Some(b) = physics.get_body(BodyHandle::from(h)) else {
                    continue;
                };
                if b.body_type != BodyType::Dynamic {
                    continue;
                }
                min_y = min_y.min(b.position.y);
            }
            println!("awake min_y after step {step}: {min_y:.1}");
            if backend == BroadPhaseKind::Auto {
                println!(
                    "auto active backend: {:?}",
                    physics.auto_active_broadphase()
                );
            }
            let shed = physics.last_substep_shed();
            if shed > 0 {
                println!("budget shed {shed} substeps this step");
            }
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
    let mut bodies = DEFAULT_BODIES;
    let mut steps = DEFAULT_STEPS;
    let mut kill_plane = false;

    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--sweep" => {
                backend = BroadPhaseKind::SweepAndPrune;
                cell_size = None;
            }
            "--grid" => backend = BroadPhaseKind::UniformGrid,
            "--tree" => backend = BroadPhaseKind::DynamicAabbTree,
            "--auto" => backend = BroadPhaseKind::Auto,
            "--cell-size" => {
                cell_size = Some(parse_value("--cell-size", args.next()));
                backend = BroadPhaseKind::UniformGrid;
            }
            "--scene" => scene = parse_value("--scene", args.next()),
            "--bodies" => bodies = parse_value("--bodies", args.next()),
            "--steps" => steps = parse_value("--steps", args.next()),
            "--kill-plane" => kill_plane = true,
            "--help" | "-h" => {
                print_usage();
                return;
            }
            unknown => panic!("unknown argument {unknown}; use --help for usage"),
        }
    }
    run_probe(
        backend,
        cell_size.unwrap_or(DEFAULT_CELL_SIZE),
        &scene,
        bodies,
        steps,
        kill_plane,
    );
}
