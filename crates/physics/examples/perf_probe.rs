use std::time::Instant;

use glam::Vec3;
use ornis_physics::{BuiltinPhysicsEngine, PhysicsEngine, RigidBody, StepTiming};

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
                    Vec3::new(x, 0.4 + level as f32 * 0.82, z),
                    half,
                    1.0,
                ));
            }
        }
    }
    physics
}

fn setup_big_stack(n: u32) -> BuiltinPhysicsEngine {
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    for level in 0..n {
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.4 + level as f32 * 0.82, 0.0),
            Vec3::splat(0.4),
            1.0,
        ));
    }
    physics
}

fn setup_many_islands(clusters: u32) -> BuiltinPhysicsEngine {
    // Many small isolated stacks (4-body towers) far apart — exercises island
    // discovery/count overhead rather than large contact clusters.
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let pitch = 12.0f32;
    for c in 0..clusters {
        let cx = (c % 32) as f32 * pitch;
        let cz = (c / 32) as f32 * pitch;
        for level in 0..4 {
            physics.add_body(RigidBody::new_box(
                Vec3::new(cx, 0.4 + level as f32 * 0.82, cz),
                Vec3::splat(0.4),
                1.0,
            ));
        }
    }
    physics
}

fn setup_contact_cluster(n: u32) -> BuiltinPhysicsEngine {
    // One dense packing of n bodies in a single contact cluster — stresses the
    // solver on a large island (no island splitting helps here).
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let side = (n as f32).sqrt().ceil() as u32;
    let spacing = 0.95f32;
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = gx as f32 * spacing;
        let z = gz as f32 * spacing;
        physics.add_body(RigidBody::new_box(
            Vec3::new(x, (i / (side * side)) as f32 * spacing, z),
            Vec3::splat(0.4),
            1.0,
        ));
    }
    physics
}

fn setup_tall_stack(n: u32) -> BuiltinPhysicsEngine {
    // Tall tower of n equal-mass bodies — stresses stability under low substeps
    // (high stacks need many substeps to settle without jitter).
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    for level in 0..n {
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.4 + level as f32 * 0.82, 0.0),
            Vec3::splat(0.4),
            1.0,
        ));
    }
    physics
}

fn setup_fast_drop() -> BuiltinPhysicsEngine {
    // One dynamic body thrown at the floor with high speed — stress for
    // tunnelling / non-convergence at low substeps (large sub_dt).
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    let mut ball = RigidBody::new_box(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
    ball.velocity = Vec3::new(0.0, -40.0, 0.0);
    physics.add_body(ball);
    physics
}

/// Report residual motion + lowest body (tunnelling check) after a settle run.
fn log_stability(physics: &BuiltinPhysicsEngine, label: &str, count: usize) {
    let mut awake = 0usize;
    let mut max_v = 0.0f32;
    let mut min_y = f32::MAX;
    for h in 0..count {
        if let Some(b) = physics.get_body(h) {
            min_y = min_y.min(b.position.y - 0.4);
            if !physics.is_asleep(h) {
                awake += 1;
                max_v = max_v.max(b.velocity.length());
            }
        }
    }
    println!(
        "  stability {label}: awake={awake} max_awake_v={max_v:.4} min_body_bottom_y={min_y:.3} (floor top y=-0.0)"
    );
}

fn time_steps(label: &str, physics: &mut BuiltinPhysicsEngine, settle: u32, measure: u32) {
    for _ in 0..settle {
        physics.step(1.0 / 60.0);
    }
    let t0 = Instant::now();
    let mut timing_sum = StepTiming::default();
    let mut timing_peak = StepTiming::default();
    for _ in 0..measure {
        physics.step(1.0 / 60.0);
        let t = physics.step_timing();
        timing_sum.broad_phase_ms += t.broad_phase_ms;
        timing_sum.narrow_phase_ms += t.narrow_phase_ms;
        timing_sum.solver_ms += t.solver_ms;
        timing_peak.broad_phase_ms = timing_peak.broad_phase_ms.max(t.broad_phase_ms);
        timing_peak.narrow_phase_ms = timing_peak.narrow_phase_ms.max(t.narrow_phase_ms);
        timing_peak.solver_ms = timing_peak.solver_ms.max(t.solver_ms);
    }
    let t = t0.elapsed();
    let per_frame = t.as_secs_f64() * 1000.0 / measure as f64;
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
fn log_grid_sleep_summary(grid: &BuiltinPhysicsEngine) {
    let mut asleep = 0usize;
    let mut max_v = 0.0f32;
    let mut max_w = 0.0f32;
    for h in 0..1025 {
        if grid.is_asleep(h) {
            asleep += 1;
        }
        if let Some(b) = grid.get_body(h) {
            max_v = max_v.max(b.velocity.length());
            max_w = max_w.max(b.angular_velocity.length());
        }
    }
    println!("grid after settle: {asleep}/1025 asleep, max |v|={max_v:.4}, max |w|={max_w:.4}");
}

/// Watch the awake set for 30 more frames: does it oscillate?
fn log_awake_oscillation(grid: &mut BuiltinPhysicsEngine) {
    for f in 0..30 {
        grid.step(1.0 / 60.0);
        let mut awake = 0usize;
        let mut mv = 0.0f32;
        for h in 0..1025 {
            if !grid.is_asleep(h) {
                awake += 1;
                if let Some(b) = grid.get_body(h) {
                    mv = mv.max(b.velocity.length());
                }
            }
        }
        if f % 5 == 0 {
            println!("  f+{f}: awake={awake} max_awake_v={mv:.4}");
        }
    }
}

/// Who stays awake? Print the positions/velocities of a few stubborn bodies.
fn log_stubborn_bodies(grid: &mut BuiltinPhysicsEngine) {
    let mut stubborn = Vec::new();
    for h in 0..1025 {
        if !grid.is_asleep(h) && stubborn.len() < 6 {
            let b = grid.get_body(h).unwrap();
            println!(
                "awake h={h} pos=({:.3},{:.3},{:.3}) v={:.4} w={:.4} island={:?}",
                b.position.x,
                b.position.y,
                b.position.z,
                b.velocity.length(),
                b.angular_velocity.length(),
                grid.debug_island_info(h)
            );
            stubborn.push(h);
        }
    }
}

/// Track island id + timer + contact count of one stubborn stack for 40 frames.
fn log_island_tracking(grid: &mut BuiltinPhysicsEngine) {
    for f in 0..40 {
        grid.step(1.0 / 60.0);
        println!(
            "  track f+{f}: b29=(i{:?} c{} {}) b30=(i{:?} c{} {})",
            grid.debug_island_info(29),
            grid.debug_contact_count(29),
            if grid.is_asleep(29) { "ZZ" } else { "  " },
            grid.debug_island_info(30),
            grid.debug_contact_count(30),
            if grid.is_asleep(30) { "ZZ" } else { "  " },
        );
    }
}

/// Sleep summary plus per-frame manifold diagnostics for the big stack.
fn log_stack_diagnostics(stack: &mut BuiltinPhysicsEngine) {
    let mut stack_asleep = 0;
    for h in 0..33 {
        if stack.is_asleep(h) {
            stack_asleep += 1;
        }
    }
    println!("stack after settle: {stack_asleep}/33 asleep");

    // Which pair of the big stack lacks a manifold, per frame?
    for f in 0..12 {
        stack.step(1.0 / 60.0);
        let mut line = format!("stack f+{f}:");
        for h in 1..33usize {
            if !stack.is_asleep(h) {
                let b = stack.get_body(h).unwrap();
                line += &format!(
                    " {h}(i{},t{:.2},v{:.3},w{:.3})",
                    stack.debug_island_info(h).map(|(r, _)| r).unwrap_or(0),
                    stack.debug_island_info(h).map(|(_, t)| t).unwrap_or(0.0),
                    b.velocity.length(),
                    b.angular_velocity.length()
                );
            }
        }
        println!("{line}");
    }
}

fn main() {
    let mut grid = setup_islands_grid(16);
    time_steps("islands_grid_16x16 (1025 bodies)", &mut grid, 0, 60);

    log_grid_sleep_summary(&grid);
    log_awake_oscillation(&mut grid);
    log_stubborn_bodies(&mut grid);
    log_island_tracking(&mut grid);

    let mut stack = setup_big_stack(32);
    time_steps("big_stack_32 (33 bodies)", &mut stack, 60, 300);
    log_stack_diagnostics(&mut stack);

    let mut islands = setup_many_islands(256); // 1024 bodies, many 4-towers
    time_steps("many_islands_256 (1024 bodies)", &mut islands, 0, 60);

    // Solver tuning sweep on islands_grid: how much solver_ms moves with
    // substeps / velocity+position iterations. Isolates the dominant cost.
    let mut sweep = setup_islands_grid(16);
    for (sub, vel, pos) in [(12, 8, 4), (4, 8, 4), (4, 4, 2), (2, 4, 2)] {
        sweep.set_substeps(sub);
        sweep.set_velocity_iterations(vel);
        sweep.set_position_iterations(pos);
        time_steps(
            &format!("islands_grid sub={sub} vel={vel} pos={pos}"),
            &mut sweep,
            0,
            30,
        );
    }

    let mut cluster2k = setup_contact_cluster(2000);
    time_steps("contact_cluster_2k (2000 bodies)", &mut cluster2k, 0, 30);

    let mut cluster5k = setup_contact_cluster(5000);
    time_steps("contact_cluster_5k (5000 bodies)", &mut cluster5k, 0, 20);

    // Stability vs substeps: does lowering substeps (4) cause jitter or
    // tunnelling on stiff scenes where the default (12) is conservative?
    for sub in [12u32, 4u32] {
        let mut tall = setup_tall_stack(50);
        tall.set_substeps(sub);
        time_steps(&format!("tall_stack_50 sub={sub}"), &mut tall, 120, 60);
        log_stability(&tall, &format!("tall_stack_50 sub={sub}"), 51);

        let mut fast = setup_fast_drop();
        fast.set_substeps(sub);
        time_steps(&format!("fast_drop sub={sub}"), &mut fast, 60, 30);
        log_stability(&fast, &format!("fast_drop sub={sub}"), 2);
    }
}
