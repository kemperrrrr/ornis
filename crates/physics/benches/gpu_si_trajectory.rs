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
//! the cost of the wrong answer rather than of the solver. That sink is
//! pre-existing (#286); this bench reports it and does not change the solver.
//!
//! Without an adapter the bench prints a note and exits zero; set
//! `ORNIS_REQUIRE_GPU=1` to make a missing adapter a failure instead.

fn main() {
    work::entry();
}

mod work {
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
    /// Gravity (m/s²), the same downward acceleration the demos use.
    const GRAVITY_Y: f32 = -9.81;
    /// Fixed step (s).
    const DT: f32 = 1.0 / 60.0;
    /// Steps per run.
    const STEPS: usize = 600;
    /// Sample the trajectory every this many steps.
    const SAMPLE: usize = 60;
    /// Milliseconds in one second, for the per-step timing columns.
    const MS_PER_SEC: f64 = 1e3;
    /// Default cap on staged bodies when `ORNIS_SI_MAX_BODIES` is unset.
    const DEFAULT_BODY_CAP: usize = 64;
    /// Default cap on staged batches when `ORNIS_SI_MAX_BATCHES` is unset.
    const DEFAULT_BATCH_CAP: usize = 512;
    /// Warmup runs.
    const WARMUP: usize = 1;
    /// Timed runs.
    const RUNS: usize = 3;

    /// Device the GPU side steps on.
    struct Gpu {
        name: String,
        backend: wgpu::Backend,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    }

    /// Scene size and the caps the GPU solver is staged with.
    struct Layout {
        stacks: usize,
        per_stack: usize,
        bodies: usize,
        body_cap: usize,
        batch_cap: usize,
    }

    /// One printed trajectory sample.
    struct Sample {
        step: usize,
        cpu_y: f32,
        gpu_y: f32,
        drift: f32,
        cpu_ms: f64,
        gpu_ms: f64,
        penetration: f32,
    }

    pub(super) fn entry() {
        let stacks = env_count("ORNIS_SI_STACKS", DEFAULT_STACKS);
        let per_stack = env_count("ORNIS_SI_PER_STACK", DEFAULT_PER_STACK);
        let body_cap = env_count("ORNIS_SI_MAX_BODIES", DEFAULT_BODY_CAP);
        let batch_cap = env_count("ORNIS_SI_MAX_BATCHES", DEFAULT_BATCH_CAP);
        let Some(gpu) = open_gpu() else {
            return;
        };
        let layout = Layout {
            stacks,
            per_stack,
            bodies: stacks * per_stack,
            body_cap,
            batch_cap,
        };
        announce(&gpu, &layout);
        trace(&gpu, &layout);
        time(&gpu, &layout);
    }

    fn env_count(name: &str, fallback: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|value| *value > 0)
            .unwrap_or(fallback)
    }

    fn open_gpu() -> Option<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            backend_options: Default::default(),
            memory_budget_thresholds: Default::default(),
            display: None,
        });
        let adapter = match pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        ) {
            Ok(adapter) => adapter,
            Err(err) => return miss(&err.to_string()),
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
                Err(err) => return miss(&err.to_string()),
            };
        Some(Gpu {
            name: info.name,
            backend: info.backend,
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    fn miss(reason: &str) -> Option<Gpu> {
        if std::env::var_os("ORNIS_REQUIRE_GPU").is_some() {
            eprintln!("ORNIS_REQUIRE_GPU is set but the trajectory bench has no adapter: {reason}");
            std::process::exit(1);
        }
        eprintln!("gpu si trajectory: no wgpu adapter — skipped ({reason})");
        None
    }

    fn announce(gpu: &Gpu, layout: &Layout) {
        println!("gpu si trajectory");
        println!("adapter: {} ({:?})", gpu.name, gpu.backend);
        println!(
            "scene: {} stacks x {} spheres, r={RADIUS}, dt={DT:.5}, {STEPS} steps",
            layout.stacks, layout.per_stack,
        );
        println!();
        println!("step\tcpu mean y\tgpu mean y\tgpu-cpu max\tcpu ms/gpu ms");
    }

    fn trace(gpu: &Gpu, layout: &Layout) {
        let mut cpu = scene(layout.stacks, layout.per_stack);
        let mut gpu_engine = solved(gpu, layout);
        let mut cpu_total = Duration::ZERO;
        let mut gpu_total = Duration::ZERO;
        for step in 1..=STEPS {
            cpu_total += step_for(&mut cpu);
            gpu_total += step_for(&mut gpu_engine);
            if step % SAMPLE == 0 {
                report(&Sample {
                    step,
                    cpu_y: mean_y(&cpu, layout.bodies),
                    gpu_y: mean_y(&gpu_engine, layout.bodies),
                    drift: worst_drift(&cpu, &gpu_engine, layout.bodies),
                    cpu_ms: per_step_ms(cpu_total, step),
                    gpu_ms: per_step_ms(gpu_total, step),
                    penetration: floor_penetration(&gpu_engine, layout.bodies)
                        .max(floor_penetration(&cpu, layout.bodies)),
                });
            }
        }
    }

    fn time(gpu: &Gpu, layout: &Layout) {
        let mut cpu_runs = Vec::with_capacity(RUNS);
        let mut gpu_runs = Vec::with_capacity(RUNS);
        for run in 0..WARMUP + RUNS {
            let cpu = time_engine(&mut scene(layout.stacks, layout.per_stack));
            let gpu_time = time_engine(&mut solved(gpu, layout));
            if run >= WARMUP {
                cpu_runs.push(cpu);
                gpu_runs.push(gpu_time);
            }
        }
        let cpu_ms = median_ms(&cpu_runs);
        let gpu_ms = median_ms(&gpu_runs);
        println!();
        println!(
            "per-step median: cpu {cpu_ms:.3} ms, gpu {gpu_ms:.3} ms, ratio {:.2}",
            cpu_ms / gpu_ms,
        );
    }

    fn solved(gpu: &Gpu, layout: &Layout) -> SequentialImpulseEngine {
        let mut engine = scene(layout.stacks, layout.per_stack);
        engine.set_gpu_solver(GpuSequentialImpulse::new(
            Arc::clone(&gpu.device),
            Arc::clone(&gpu.queue),
            layout.body_cap.max(layout.bodies + 1),
            layout.batch_cap.max(layout.bodies * 2),
        ));
        engine
    }

    fn step_for(engine: &mut SequentialImpulseEngine) -> Duration {
        let started = Instant::now();
        engine.step(DT);
        started.elapsed()
    }

    fn time_engine(engine: &mut SequentialImpulseEngine) -> Duration {
        let started = Instant::now();
        for _ in 0..STEPS {
            engine.step(DT);
        }
        started.elapsed()
    }

    /// The scene both engines step: a ground box and resting stacks of spheres.
    ///
    /// Bodies start at their resting heights: the two solvers order their contacts
    /// differently, so a stack that starts by falling measures that order rather
    /// than the solver.
    fn scene(stacks: usize, per_stack: usize) -> SequentialImpulseEngine {
        let mut engine = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
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

    fn body_heights(engine: &SequentialImpulseEngine, bodies: usize) -> Vec<f32> {
        let mut heights = Vec::with_capacity(bodies);
        for index in 1..=bodies {
            if let Some(body) = engine.get_body(index.into()) {
                heights.push(body.position.y);
            }
        }
        heights
    }

    fn mean_y(engine: &SequentialImpulseEngine, bodies: usize) -> f32 {
        let heights = body_heights(engine, bodies);
        let sum: f32 = heights.iter().copied().sum();
        sum / heights.len().max(1) as f32
    }

    fn worst_drift(
        cpu: &SequentialImpulseEngine,
        gpu: &SequentialImpulseEngine,
        bodies: usize,
    ) -> f32 {
        let cpu_y = body_heights(cpu, bodies);
        let gpu_y = body_heights(gpu, bodies);
        cpu_y
            .iter()
            .zip(gpu_y.iter())
            .map(|(cpu, gpu)| (cpu - gpu).abs())
            .fold(0.0, f32::max)
    }

    /// How far the lowest body sits below the ground top: positive means the
    /// stack fell through the floor, which no timing delta can excuse.
    fn floor_penetration(engine: &SequentialImpulseEngine, bodies: usize) -> f32 {
        let ground_top = -GROUND.y;
        let lowest = body_heights(engine, bodies)
            .into_iter()
            .map(|y| y - RADIUS)
            .fold(f32::INFINITY, f32::min);
        ground_top - lowest
    }

    fn per_step_ms(total: Duration, step: usize) -> f64 {
        total.as_secs_f64() * MS_PER_SEC / step as f64
    }

    fn report(sample: &Sample) {
        println!(
            "{}\t{:.4}\t\t{:.4}\t\t{:.4}\t\t{:.3} / {:.3}",
            sample.step, sample.cpu_y, sample.gpu_y, sample.drift, sample.cpu_ms, sample.gpu_ms,
        );
        if sample.penetration > RADIUS {
            println!(
                "  floor penetration {:.3} m (bodies below the ground top)",
                sample.penetration
            );
        }
    }

    fn median_ms(samples: &[Duration]) -> f64 {
        let mut values: Vec<f64> = samples
            .iter()
            .map(|d| d.as_secs_f64() * MS_PER_SEC)
            .collect();
        values.sort_by(f64::total_cmp);
        match values.get(values.len() / 2) {
            Some(median) => median / STEPS as f64,
            None => 0.0,
        }
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn median_is_the_middle_run_per_step() {
            let samples = [
                std::time::Duration::from_millis(600),
                std::time::Duration::from_millis(1_800),
                std::time::Duration::from_millis(1_200),
            ];
            let got = super::median_ms(&samples);
            assert!((got - 2.0).abs() < 1e-9, "median per step {got}");
        }

        #[test]
        fn an_empty_timing_list_has_no_median() {
            assert_eq!(super::median_ms(&[]), 0.0);
        }

        #[test]
        fn an_unset_count_keeps_the_fallback() {
            let name = "ORNIS_SI_TRAJECTORY_TEST_UNSET";
            std::env::remove_var(name);
            assert_eq!(super::env_count(name, 4), 4);
        }

        #[test]
        fn four_stacks_sit_on_a_centred_square() {
            let origin = super::stack_origin(0, 4);
            assert!(origin.x < 0.0 && origin.z < 0.0, "{origin:?}");
            assert_eq!(super::stack_origin(0, 1), glam::Vec3::ZERO);
        }

        #[test]
        fn a_resting_sphere_starts_on_the_ground() {
            let engine = super::scene(1, 1);
            let heights = super::body_heights(&engine, 1);
            assert_eq!(heights.len(), 1);
            // Ground top is `-GROUND.y`; the first sphere rests one radius above it.
            let resting = -super::GROUND.y + super::RADIUS;
            assert!((heights[0] - resting).abs() < 1e-5, "height {}", heights[0]);
            assert!((super::mean_y(&engine, 1) - heights[0]).abs() < 1e-5);
            assert_eq!(super::worst_drift(&engine, &engine, 1), 0.0);
            assert!(super::floor_penetration(&engine, 1) <= 0.0);
            assert!(super::GRAVITY_Y < 0.0);
        }

        #[test]
        fn a_missing_adapter_skips_unless_gpu_is_required() {
            if std::env::var_os("ORNIS_REQUIRE_GPU").is_none() {
                assert!(super::miss("bench test").is_none());
            }
        }

        #[test]
        fn the_adapter_can_be_announced_when_present() {
            if let Some(gpu) = super::open_gpu() {
                super::announce(
                    &gpu,
                    &super::Layout {
                        stacks: 1,
                        per_stack: 1,
                        bodies: 1,
                        body_cap: 1,
                        batch_cap: 1,
                    },
                );
            }
        }
    }
}
