//! GPU SI trajectories: one scene stepped by the CPU and the GPU solver side
//! by side, reporting the trajectory next to the per-step cost.
//!
//! An A/B of two branches compares per-step wall time, which is only meaningful
//! when both sides move the same bodies the same way: a solver that drops a
//! stack through the floor is faster at being wrong. This bench therefore
//! prints the trajectory (mean height per sample, worst per-body drift against
//! the CPU reference, deepest floor penetration) beside the timings, so a
//! timing delta can be read against the motion it bought.
//!
//! Reproduce with:
//!
//! ```text
//! cargo bench -p ornis-physics --features gpu --bench gpu_si_trajectory
//! ORNIS_SI_STACKS=1 ORNIS_SI_PER_STACK=20 cargo bench ...   # narrow tower
//! ```
//!
//! The default scene is `8 x 4` spheres: that is the tallest tower the GPU
//! path still holds. `ORNIS_SI_PER_STACK=20` (or any taller stack) makes the
//! GPU side sink through the floor while the CPU keeps the tower standing —
//! the trajectory columns show the divergence, and the timings of that run are
//! the cost of the wrong answer rather than of the solver.
//!
//! Without an adapter the bench prints a note and exits zero; set
//! `ORNIS_REQUIRE_GPU=1` to make a missing adapter a failure instead.

use glam::Vec3;
use ornis_physics::engine::SequentialImpulseEngine;
use ornis_physics::gpu::GpuSequentialImpulse;
use ornis_physics::{PhysicsEngine, RigidBody};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default stacks in the scene (override with `ORNIS_SI_STACKS`).
const DEFAULT_STACKS: usize = 8;
/// Default spheres per stack (override with `ORNIS_SI_PER_STACK`).
///
/// Four is what the GPU path still holds: on taller stacks it lets the tower
/// sink and then fall through the floor while the CPU keeps it standing, so a
/// taller scene measures that divergence rather than the solver's cost.
const DEFAULT_PER_STACK: usize = 4;
/// Sphere radius (m).
const RADIUS: f32 = 0.5;
/// Ground box half extents (m).
const GROUND: Vec3 = Vec3::new(4.0, 0.5, 4.0);
/// Fixed step (s).
const DT: f32 = 1.0 / 60.0;
/// Steps per run.
const STEPS: usize = 600;
/// Sample the trajectory every this many steps.
const SAMPLE: usize = 60;
/// Default cap on staged bodies when `ORNIS_SI_MAX_BODIES` is unset.
const DEFAULT_BODY_CAP: usize = 64;
/// Default cap on staged batches when `ORNIS_SI_MAX_BATCHES` is unset.
const DEFAULT_BATCH_CAP: usize = 512;
/// Warmup runs.
const WARMUP: usize = 1;
/// Timed runs.
const RUNS: usize = 3;

fn main() {
    let stacks = env_count("ORNIS_SI_STACKS", DEFAULT_STACKS);
    let per_stack = env_count("ORNIS_SI_PER_STACK", DEFAULT_PER_STACK);
    let bodies = stacks * per_stack;
    let body_cap = env_count("ORNIS_SI_MAX_BODIES", DEFAULT_BODY_CAP);
    let batch_cap = env_count("ORNIS_SI_MAX_BATCHES", DEFAULT_BATCH_CAP);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        backend_options: Default::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
    });
    let adapter =
        match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        {
            Ok(adapter) => adapter,
            Err(err) => return no_adapter(&err.to_string()),
        };
    let info = adapter.get_info();
    let (device, queue) =
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpu si trajectory"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })) {
            Ok(pair) => pair,
            Err(err) => return no_adapter(&err.to_string()),
        };
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    println!("gpu si trajectory");
    println!("adapter: {} ({:?})", info.name, info.backend);
    println!("scene: {stacks} stacks x {per_stack} spheres, r={RADIUS}, dt={DT:.5}, {STEPS} steps",);
    println!();
    println!("step\tcpu mean y\tgpu mean y\tgpu-cpu max\tcpu ms/gpu ms");

    let mut cpu = scene(stacks, per_stack);
    let mut gpu = scene(stacks, per_stack);
    gpu.set_gpu_solver(GpuSequentialImpulse::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        body_cap.max(bodies + 1),
        batch_cap.max(bodies * 2),
    ));

    let mut cpu_total = Duration::ZERO;
    let mut gpu_total = Duration::ZERO;
    for step in 1..=STEPS {
        let started = Instant::now();
        cpu.step(DT);
        cpu_total += started.elapsed();
        let started = Instant::now();
        gpu.step(DT);
        gpu_total += started.elapsed();
        if step % SAMPLE == 0 {
            report(&cpu, &gpu, bodies, step, cpu_total, gpu_total);
        }
    }

    let mut cpu_runs = Vec::with_capacity(RUNS);
    let mut gpu_runs = Vec::with_capacity(RUNS);
    for run in 0..WARMUP + RUNS {
        let mut engine = scene(stacks, per_stack);
        let started = Instant::now();
        for _ in 0..STEPS {
            engine.step(DT);
        }
        let cpu = started.elapsed();
        let mut engine = scene(stacks, per_stack);
        engine.set_gpu_solver(GpuSequentialImpulse::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            body_cap.max(bodies + 1),
            batch_cap.max(bodies * 2),
        ));
        let started = Instant::now();
        for _ in 0..STEPS {
            engine.step(DT);
        }
        let gpu = started.elapsed();
        if run >= WARMUP {
            cpu_runs.push(cpu);
            gpu_runs.push(gpu);
        }
    }
    println!();
    println!(
        "per-step median: cpu {:.3} ms, gpu {:.3} ms, ratio {:.2}",
        median_ms(&cpu_runs),
        median_ms(&gpu_runs),
        median_ms(&cpu_runs) / median_ms(&gpu_runs),
    );
}

fn env_count(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(fallback)
}

fn no_adapter(reason: &str) {
    if std::env::var_os("ORNIS_REQUIRE_GPU").is_some() {
        eprintln!("ORNIS_REQUIRE_GPU is set but the trajectory bench has no adapter: {reason}");
        std::process::exit(1);
    }
    eprintln!("gpu si trajectory: no wgpu adapter — skipped ({reason})");
}

/// The scene both engines step: a ground box and resting stacks of spheres.
///
/// Bodies start at their resting heights: the two solvers order their contacts
/// differently, so a stack that starts by falling measures that order rather
/// than the solver.
fn scene(stacks: usize, per_stack: usize) -> SequentialImpulseEngine {
    let mut engine = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let ground_top = -GROUND.y;
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, ground_top - GROUND.y, 0.0),
        GROUND,
        0.0,
    ));
    for stack in 0..stacks {
        let origin = stack_origin(stack, stacks);
        for level in 0..per_stack {
            let y = ground_top + RADIUS + level as f32 * (2.0 * RADIUS);
            engine.add_body(RigidBody::new_sphere(
                Vec3::new(origin.x, y, origin.z),
                RADIUS,
                1.0,
            ));
        }
    }
    engine
}

/// Ground position of one stack: a centred square grid.
fn stack_origin(index: usize, stacks: usize) -> Vec3 {
    let columns = (stacks as f32).sqrt().ceil() as usize;
    let rows = stacks.div_ceil(columns.max(1));
    let (column, row) = (index % columns, index / columns);
    let x = (column as f32 - (columns as f32 - 1.0) / 2.0) * 2.0;
    let z = (row as f32 - (rows as f32 - 1.0) / 2.0) * 2.0;
    Vec3::new(x, 0.0, z)
}

fn mean_y(engine: &SequentialImpulseEngine, bodies: usize) -> f32 {
    let (mut sum, mut count) = (0.0f32, 0usize);
    for index in 1..=bodies {
        if let Some(body) = engine.get_body(index.into()) {
            sum += body.position.y;
            count += 1;
        }
    }
    sum / count.max(1) as f32
}

fn worst_drift(cpu: &SequentialImpulseEngine, gpu: &SequentialImpulseEngine, bodies: usize) -> f32 {
    let mut worst = 0.0f32;
    for index in 1..=bodies {
        let (Some(a), Some(b)) = (cpu.get_body(index.into()), gpu.get_body(index.into())) else {
            continue;
        };
        worst = worst.max((a.position.y - b.position.y).abs());
    }
    worst
}

/// How far the lowest body sits below the ground top: positive means the
/// stack fell through the floor, which no timing delta can excuse.
fn floor_penetration(engine: &SequentialImpulseEngine, bodies: usize) -> f32 {
    let ground_top = -GROUND.y;
    let mut lowest = f32::INFINITY;
    for index in 1..=bodies {
        if let Some(body) = engine.get_body(index.into()) {
            lowest = lowest.min(body.position.y - RADIUS);
        }
    }
    ground_top - lowest
}

fn report(
    cpu: &SequentialImpulseEngine,
    gpu: &SequentialImpulseEngine,
    bodies: usize,
    step: usize,
    cpu_total: Duration,
    gpu_total: Duration,
) {
    println!(
        "{step}\t{:.4}\t\t{:.4}\t\t{:.4}\t\t{:.3} / {:.3}",
        mean_y(cpu, bodies),
        mean_y(gpu, bodies),
        worst_drift(cpu, gpu, bodies),
        cpu_total.as_secs_f64() * 1e3 / step as f64,
        gpu_total.as_secs_f64() * 1e3 / step as f64,
    );
    let penetration = floor_penetration(gpu, bodies).max(floor_penetration(cpu, bodies));
    if penetration > RADIUS {
        println!("  floor penetration {penetration:.3} m (bodies below the ground top)");
    }
}

fn median_ms(samples: &[Duration]) -> f64 {
    let mut values: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e3).collect();
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    values[values.len() / 2] / STEPS as f64
}
