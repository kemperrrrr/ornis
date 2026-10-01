//! Performance probe example for physics solver timing across scenarios.

use std::time::Instant;

use glam::Vec3;
use ornis_physics::{BodyHandle, PhysicsEngine, RigidBody, SequentialImpulseEngine, StepTiming};

/// Earth-surface gravity along −Y (m/s²).
const GRAVITY_Y: f32 = -9.81;
/// Static floor box half-height / Y center offset (m).
const FLOOR_HALF_Y: f32 = 0.5;
/// Large floor half-extent for the islands grid (m).
const GRID_FLOOR_HALF: f32 = 100.0;
/// Compact floor half-extent for stacks / drops (m).
const STACK_FLOOR_HALF: f32 = 10.0;
/// Dynamic box half-extent (m).
const BOX_HALF: f32 = 0.4;
/// Grid cell pitch between island towers (m).
const GRID_PITCH: f32 = 2.0;
/// Bodies per island tower.
const TOWER_HEIGHT: u32 = 4;
/// Vertical spacing between stacked boxes (m).
const STACK_STEP_Y: f32 = 0.82;
/// First dynamic box center Y above the floor (m).
const STACK_BASE_Y: f32 = 0.4;
/// Pitch between many-islands clusters (m).
const CLUSTER_PITCH: f32 = 12.0;
/// Cluster grid columns.
const CLUSTER_COLS: u32 = 32;
/// Contact-cluster packing spacing (m).
const CLUSTER_SPACING: f32 = 0.95;
/// Fast-drop spawn height (m).
const DROP_HEIGHT: f32 = 8.0;
/// Fast-drop impact speed (m/s).
const DROP_SPEED: f32 = 40.0;
/// Fixed simulation timestep (s).
const DT: f32 = 1.0 / 60.0;
/// Body count tracked in the grid sleep summary (floor + 16×16×4).
const GRID_BODY_COUNT: usize = 1025;
/// Frames watched for awake-set oscillation.
const AWAKE_WATCH_FRAMES: u32 = 30;
/// Log awake oscillation every N frames.
const AWAKE_LOG_STRIDE: u32 = 5;
/// Max stubborn awake bodies to print.
const STUBBORN_PRINT_CAP: usize = 6;
/// Frames to track a stubborn island.
const ISLAND_TRACK_FRAMES: u32 = 40;
/// Bodies in the big-stack probe (floor + 32).
const BIG_STACK_BODIES: usize = 33;
/// Stack diagnostic frames after settle.
const STACK_DIAG_FRAMES: u32 = 12;
/// Grid side length for the islands-grid probe.
const GRID_SIDE: u32 = 16;
/// Default measure-window length (frames).
const MEASURE_60: u32 = 60;
/// Short measure window (frames).
const MEASURE_30: u32 = 30;
/// Very short measure window (frames).
const MEASURE_20: u32 = 20;
/// Big-stack settle frames.
const STACK_SETTLE: u32 = 60;
/// Big-stack measure frames.
const STACK_MEASURE: u32 = 300;
/// Many-islands cluster count.
const MANY_ISLANDS: u32 = 256;
/// Contact cluster body counts.
const CLUSTER_2K: u32 = 2000;
const CLUSTER_5K: u32 = 5000;
/// Tall-stack body count.
const TALL_STACK_N: u32 = 50;
/// Tall-stack settle frames.
const TALL_SETTLE: u32 = 120;
/// Milliseconds scale for timing printouts.
const MS_PER_SEC: f64 = 1000.0;

fn setup_islands_grid(g: u32) -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -FLOOR_HALF_Y, 0.0),
        Vec3::new(GRID_FLOOR_HALF, FLOOR_HALF_Y, GRID_FLOOR_HALF),
        0.0,
    ));
    let half = Vec3::splat(BOX_HALF);
    let pitch = GRID_PITCH;
    for gx in 0..g {
        for gz in 0..g {
            let x = (gx as f32 - g as f32 / 2.0) * pitch;
            let z = (gz as f32 - g as f32 / 2.0) * pitch;
            for level in 0..TOWER_HEIGHT {
                physics.add_body(RigidBody::new_box(
                    Vec3::new(x, STACK_BASE_Y + level as f32 * STACK_STEP_Y, z),
                    half,
                    1.0,
                ));
            }
        }
    }
    physics
}

fn setup_big_stack(n: u32) -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -FLOOR_HALF_Y, 0.0),
        Vec3::new(STACK_FLOOR_HALF, FLOOR_HALF_Y, STACK_FLOOR_HALF),
        0.0,
    ));
    for level in 0..n {
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, STACK_BASE_Y + level as f32 * STACK_STEP_Y, 0.0),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    physics
}

fn setup_many_islands(clusters: u32) -> SequentialImpulseEngine {
    // Many small isolated stacks (4-body towers) far apart — exercises island
    // discovery/count overhead rather than large contact clusters.
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    let pitch = CLUSTER_PITCH;
    for c in 0..clusters {
        let cx = (c % CLUSTER_COLS) as f32 * pitch;
        let cz = (c / CLUSTER_COLS) as f32 * pitch;
        for level in 0..TOWER_HEIGHT {
            physics.add_body(RigidBody::new_box(
                Vec3::new(cx, STACK_BASE_Y + level as f32 * STACK_STEP_Y, cz),
                Vec3::splat(BOX_HALF),
                1.0,
            ));
        }
    }
    physics
}

fn setup_contact_cluster(n: u32) -> SequentialImpulseEngine {
    // One dense packing of n bodies in a single contact cluster — stresses the
    // solver on a large island (no island splitting helps here).
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    let side = (n as f32).sqrt().ceil() as u32;
    let spacing = CLUSTER_SPACING;
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = gx as f32 * spacing;
        let z = gz as f32 * spacing;
        physics.add_body(RigidBody::new_box(
            Vec3::new(x, (i / (side * side)) as f32 * spacing, z),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    physics
}

fn setup_tall_stack(n: u32) -> SequentialImpulseEngine {
    // Tall tower of n equal-mass bodies — stresses stability under low substeps
    // (high stacks need many substeps to settle without jitter).
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -FLOOR_HALF_Y, 0.0),
        Vec3::new(STACK_FLOOR_HALF, FLOOR_HALF_Y, STACK_FLOOR_HALF),
        0.0,
    ));
    for level in 0..n {
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, STACK_BASE_Y + level as f32 * STACK_STEP_Y, 0.0),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    physics
}

fn setup_fast_drop() -> SequentialImpulseEngine {
    // One dynamic body thrown at the floor with high speed — stress for
    // tunnelling / non-convergence at low substeps (large sub_dt).
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -FLOOR_HALF_Y, 0.0),
        Vec3::new(STACK_FLOOR_HALF, FLOOR_HALF_Y, STACK_FLOOR_HALF),
        0.0,
    ));
    let mut ball = RigidBody::new_box(Vec3::new(0.0, DROP_HEIGHT, 0.0), Vec3::splat(BOX_HALF), 1.0);
    ball.velocity = Vec3::new(0.0, -DROP_SPEED, 0.0);
    physics.add_body(ball);
    physics
}

/// Report residual motion + lowest body (tunnelling check) after a settle run.
fn log_stability(physics: &SequentialImpulseEngine, label: &str, count: usize) {
    let mut awake = 0usize;
    let mut max_v = 0.0f32;
    let mut min_y = f32::MAX;
    for h in 0..count {
        if let Some(b) = physics.get_body(BodyHandle::from(h)) {
            min_y = min_y.min(b.position.y - BOX_HALF);
            if !physics.is_asleep(BodyHandle::from(h)) {
                awake += 1;
                max_v = max_v.max(b.velocity.length());
            }
        }
    }
    println!(
        "  stability {label}: awake={awake} max_awake_v={max_v:.4} min_body_bottom_y={min_y:.3} (floor top y=-0.0)"
    );
}

fn time_steps(label: &str, physics: &mut SequentialImpulseEngine, settle: u32, measure: u32) {
    for _ in 0..settle {
        physics.step(DT);
    }
    let t0 = Instant::now();
    let mut timing_sum = StepTiming::default();
    let mut timing_peak = StepTiming::default();
    for _ in 0..measure {
        physics.step(DT);
        let t = physics.step_timing();
        timing_sum.broad_phase_ms += t.broad_phase_ms;
        timing_sum.narrow_phase_ms += t.narrow_phase_ms;
        timing_sum.solver_ms += t.solver_ms;
        timing_peak.broad_phase_ms = timing_peak.broad_phase_ms.max(t.broad_phase_ms);
        timing_peak.narrow_phase_ms = timing_peak.narrow_phase_ms.max(t.narrow_phase_ms);
        timing_peak.solver_ms = timing_peak.solver_ms.max(t.solver_ms);
    }
    let t = t0.elapsed();
    let per_frame = t.as_secs_f64() * MS_PER_SEC / measure as f64;
    let bp = timing_sum.broad_phase_ms / measure as f64;
    let np = timing_sum.narrow_phase_ms / measure as f64;
    let sl = timing_sum.solver_ms / measure as f64;
    println!(
        "{label}: {per_frame:.3} ms/frame over {measure} frames | broad {bp:.3} ms | narrow {np:.3} ms | solver {sl:.3} ms | peak-frame broad {peak_bp:.3} narrow {peak_np:.3} solver {peak_sl:.3}",
        per_frame = per_frame,
        measure = measure,
        bp = bp,
        np = np,
        sl = sl,
        peak_bp = timing_peak.broad_phase_ms,
        peak_np = timing_peak.narrow_phase_ms,
        peak_sl = timing_peak.solver_ms,
    );
}

/// Sleep diagnostics for the grid: how much of the scene went to sleep and
/// how fast the awake bodies are moving.
fn log_grid_sleep_summary(grid: &SequentialImpulseEngine) {
    let mut asleep = 0usize;
    let mut max_v = 0.0f32;
    let mut max_w = 0.0f32;
    for h in 0..GRID_BODY_COUNT {
        if grid.is_asleep(BodyHandle::from(h)) {
            asleep += 1;
        }
        if let Some(b) = grid.get_body(BodyHandle::from(h)) {
            max_v = max_v.max(b.velocity.length());
            max_w = max_w.max(b.angular_velocity.length());
        }
    }
    println!(
        "grid after settle: {asleep}/{GRID_BODY_COUNT} asleep, max |v|={max_v:.4}, max |w|={max_w:.4}"
    );
}

/// Watch the awake set for 30 more frames: does it oscillate?
fn log_awake_oscillation(grid: &mut SequentialImpulseEngine) {
    for f in 0..AWAKE_WATCH_FRAMES {
        grid.step(DT);
        let mut awake = 0usize;
        let mut mv = 0.0f32;
        for h in 0..GRID_BODY_COUNT {
            if !grid.is_asleep(BodyHandle::from(h)) {
                awake += 1;
                if let Some(b) = grid.get_body(BodyHandle::from(h)) {
                    mv = mv.max(b.velocity.length());
                }
            }
        }
        if f % AWAKE_LOG_STRIDE == 0 {
            println!("  f+{f}: awake={awake} max_awake_v={mv:.4}");
        }
    }
}

/// Who stays awake? Print the positions/velocities of a few stubborn bodies.
fn log_stubborn_bodies(grid: &mut SequentialImpulseEngine) {
    let mut stubborn = Vec::new();
    for h in 0..GRID_BODY_COUNT {
        if !grid.is_asleep(BodyHandle::from(h)) && stubborn.len() < STUBBORN_PRINT_CAP {
            let b = grid.get_body(BodyHandle::from(h)).unwrap();
            println!(
                "awake h={h} pos=({:.3},{:.3},{:.3}) v={:.4} w={:.4} island={:?}",
                b.position.x,
                b.position.y,
                b.position.z,
                b.velocity.length(),
                b.angular_velocity.length(),
                grid.debug_island_info(BodyHandle::from(h))
            );
            stubborn.push(h);
        }
    }
}

/// Track island id + timer + contact count of one stubborn stack for 40 frames.
fn log_island_tracking(grid: &mut SequentialImpulseEngine) {
    for f in 0..ISLAND_TRACK_FRAMES {
        grid.step(DT);
        println!(
            "  track f+{f}: b29=(i{:?} c{} {}) b30=(i{:?} c{} {})",
            grid.debug_island_info(BodyHandle::from_raw(29)),
            grid.debug_contact_count(BodyHandle::from_raw(29)),
            if grid.is_asleep(BodyHandle::from_raw(29)) {
                "ZZ"
            } else {
                "  "
            },
            grid.debug_island_info(BodyHandle::from_raw(30)),
            grid.debug_contact_count(BodyHandle::from_raw(30)),
            if grid.is_asleep(BodyHandle::from_raw(30)) {
                "ZZ"
            } else {
                "  "
            },
        );
    }
}

/// Sleep summary plus per-frame manifold diagnostics for the big stack.
fn log_stack_diagnostics(stack: &mut SequentialImpulseEngine) {
    let mut stack_asleep = 0;
    for h in 0..BIG_STACK_BODIES {
        if stack.is_asleep(BodyHandle::from(h)) {
            stack_asleep += 1;
        }
    }
    println!("stack after settle: {stack_asleep}/{BIG_STACK_BODIES} asleep");

    // Which pair of the big stack lacks a manifold, per frame?
    for f in 0..STACK_DIAG_FRAMES {
        stack.step(DT);
        let mut line = format!("stack f+{f}:");
        for h in 1..BIG_STACK_BODIES {
            if !stack.is_asleep(BodyHandle::from(h)) {
                let b = stack.get_body(BodyHandle::from(h)).unwrap();
                line += &format!(
                    " {h}(i{},t{:.2},v{:.3},w{:.3})",
                    stack
                        .debug_island_info(BodyHandle::from(h))
                        .map(|(r, _)| r)
                        .unwrap_or(0),
                    stack
                        .debug_island_info(BodyHandle::from(h))
                        .map(|(_, t)| t)
                        .unwrap_or(0.0),
                    b.velocity.length(),
                    b.angular_velocity.length()
                );
            }
        }
        println!("{line}");
    }
}

fn main() {
    let mut grid = setup_islands_grid(GRID_SIDE);
    time_steps("islands_grid_16x16 (1025 bodies)", &mut grid, 0, MEASURE_60);

    log_grid_sleep_summary(&grid);
    log_awake_oscillation(&mut grid);
    log_stubborn_bodies(&mut grid);
    log_island_tracking(&mut grid);

    let mut stack = setup_big_stack((BIG_STACK_BODIES - 1) as u32);
    time_steps(
        "big_stack_32 (33 bodies)",
        &mut stack,
        STACK_SETTLE,
        STACK_MEASURE,
    );
    log_stack_diagnostics(&mut stack);

    let mut islands = setup_many_islands(MANY_ISLANDS); // 1024 bodies, many 4-towers
    time_steps(
        "many_islands_256 (1024 bodies)",
        &mut islands,
        0,
        MEASURE_60,
    );

    // Solver tuning sweep on islands_grid: how much solver_ms moves with
    // substeps / velocity+position iterations. Isolates the dominant cost.
    let mut sweep = setup_islands_grid(GRID_SIDE);
    for (sub, vel, pos) in [(12, 8, 4), (4, 8, 4), (4, 4, 2), (2, 4, 2)] {
        sweep.set_substeps(sub);
        sweep.set_velocity_iterations(vel);
        sweep.set_position_iterations(pos);
        time_steps(
            &format!("islands_grid sub={sub} vel={vel} pos={pos}"),
            &mut sweep,
            0,
            MEASURE_30,
        );
    }

    let mut cluster2k = setup_contact_cluster(CLUSTER_2K);
    time_steps(
        "contact_cluster_2k (2000 bodies)",
        &mut cluster2k,
        0,
        MEASURE_30,
    );

    let mut cluster5k = setup_contact_cluster(CLUSTER_5K);
    time_steps(
        "contact_cluster_5k (5000 bodies)",
        &mut cluster5k,
        0,
        MEASURE_20,
    );

    // per-island demo: many slow islands + one fast tower (heterogeneous
    // velocities). Global adaptive would push all islands to 12 substeps;
    // per-island keeps slow islands at 2-3 iters.
    {
        let mut hetero = setup_many_islands(MANY_ISLANDS);
        // kick the top of the first tower (handle 3) — one fast island
        if let Some(b) = hetero.get_body_mut(BodyHandle::from_raw(3)) {
            b.velocity = Vec3::new(0.0, -DROP_SPEED, 0.0);
        }
        time_steps(
            "hetero_many_islands 255 slow +1 fast (1024 bodies)",
            &mut hetero,
            0,
            MEASURE_60,
        );
    }

    // Stability vs substeps: does lowering substeps (4) cause jitter or
    // tunnelling on stiff scenes where the default (12) is conservative?
    for sub in [12u32, 4u32] {
        let mut tall = setup_tall_stack(TALL_STACK_N);
        tall.set_substeps(sub);
        time_steps(
            &format!("tall_stack_50 sub={sub}"),
            &mut tall,
            TALL_SETTLE,
            MEASURE_60,
        );
        log_stability(
            &tall,
            &format!("tall_stack_50 sub={sub}"),
            TALL_STACK_N as usize + 1,
        );

        let mut fast = setup_fast_drop();
        fast.set_substeps(sub);
        time_steps(
            &format!("fast_drop sub={sub}"),
            &mut fast,
            STACK_SETTLE,
            MEASURE_30,
        );
        log_stability(&fast, &format!("fast_drop sub={sub}"), 2);
    }
}
