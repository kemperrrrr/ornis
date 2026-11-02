//! Warmed A/B timings of the CPU skin and the wasm2gpu guest (issue #228).

use std::time::Duration;
use std::time::Instant;

use super::SkinGpu;
use super::SkinInputs;
use super::SyntheticSkin;
use super::lower_guest;
use super::synthetic_skin;

const WARMUP: usize = 5;
const REPEATS: usize = 30;
/// Scene sizes `(vertices, joints)`: the `anim` mannequin is ~17k vertices.
const SCENES: [(usize, usize); 2] = [(17_000, 24), (60_000, 24)];
const NANOS_PER_MICROSECOND: f64 = 1_000.0;

/// Runs the release A/B bench; setup and WGSL lowering are reported separately.
pub fn run_skin_ab() -> String {
    let timing = Timing {
        warmup: if cfg!(test) { 0 } else { WARMUP },
        repeats: if cfg!(test) { 1 } else { REPEATS },
    };
    run_skin_ab_with(timing)
}

/// Warmup/repeat shape shared by every timed scene.
#[derive(Clone, Copy)]
struct Timing {
    warmup: usize,
    repeats: usize,
}

fn run_skin_ab_with(timing: Timing) -> String {
    let lowered = lower_guest();
    let mut report = format!(
        concat!(
            "wasm2gpu skinning A/B (issue #228)\n",
            "guest build + lift: {:?}\n",
            "CPU is `ornis_animation::skin_vertices` on this host; guest is one ",
            "compute dispatch over persistent buffers (prepare is untimed), ",
            "wall time includes submission and blocking poll.\n",
            "guest WGSL bytes: {}\n",
        ),
        lowered.elapsed,
        lowered.wgsl.len(),
    );
    let Some(session) = Session::open(&mut report) else {
        return report;
    };
    report.push_str("vertices\tjoints\tcpu median us\twasm2gpu median us\tcpu/guest\n");
    for (vertices, joints) in SCENES {
        let scene = BenchScene {
            vertices,
            joints,
            timing,
        };
        report.push_str(&bench_scene(&session, &lowered.wgsl, scene));
    }
    report
}

struct Session {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Session {
    fn open(report: &mut String) -> Option<Self> {
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
            Err(err) => return no_adapter(&err.to_string()),
        };
        let info = adapter.get_info();
        report.push_str(&format!(
            "adapter: {}\nbackend: {:?}\ndevice_type: {:?}\n",
            info.name, info.backend, info.device_type,
        ));
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wasm2gpu skinning A/B"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })) {
            Ok((device, queue)) => Some(Self { device, queue }),
            Err(err) => no_adapter(&err.to_string()),
        }
    }
}

fn no_adapter(reason: &str) -> Option<Session> {
    if crate::gpu_required() {
        eprintln!("ORNIS_REQUIRE_GPU is set but skinning A/B has no wgpu device: {reason}");
        std::process::exit(1);
    }
    eprintln!("wasm2gpu skinning A/B: no wgpu adapter — timings skipped ({reason})");
    None
}

/// One bench scene: synthetic mesh shape plus the timing shape.
#[derive(Clone, Copy)]
struct BenchScene {
    vertices: usize,
    joints: usize,
    timing: Timing,
}

fn bench_scene(session: &Session, wgsl: &str, scene: BenchScene) -> String {
    let synthetic = synthetic_skin(scene.vertices, scene.joints);
    let Some(inputs) = synthetic.guest_inputs() else {
        return format!("{}\t{}\tpack failed\n", scene.vertices, scene.joints);
    };
    let cpu = median_us(&time_cpu(&synthetic, scene.timing));
    let guest_gpu = SkinGpu::new(
        &session.device,
        wgsl,
        inputs.vertex_count,
        inputs.joint_count,
    );
    let guest = median_us(&time_guest(session, &guest_gpu, &inputs, scene.timing));
    format!(
        "{}\t{}\t{cpu:.2}\t{guest:.2}\t{:.2}\n",
        scene.vertices,
        scene.joints,
        cpu / guest,
    )
}

fn time_cpu(scene: &SyntheticSkin, timing: Timing) -> Vec<Duration> {
    let mut samples = Vec::with_capacity(timing.repeats);
    for run in 0..timing.warmup + timing.repeats {
        let started = Instant::now();
        let skinned = scene.cpu_skin();
        let elapsed = started.elapsed();
        std::hint::black_box(&skinned.positions);
        std::hint::black_box(&skinned.normals);
        if run >= timing.warmup {
            samples.push(elapsed);
        }
    }
    samples
}

fn time_guest(
    session: &Session,
    gpu: &SkinGpu,
    inputs: &SkinInputs,
    timing: Timing,
) -> Vec<Duration> {
    gpu.prepare(&session.queue, inputs);
    let mut samples = Vec::with_capacity(timing.repeats);
    for run in 0..timing.warmup + timing.repeats {
        let started = Instant::now();
        let mut encoder = session
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wasm2gpu skin A/B"),
            });
        gpu.encode(&mut encoder);
        session.queue.submit(std::iter::once(encoder.finish()));
        let _ = session.device.poll(wgpu::PollType::wait_indefinitely());
        if run >= timing.warmup {
            samples.push(started.elapsed());
        }
    }
    samples
}

fn median_us(samples: &[Duration]) -> f64 {
    let mut values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.as_nanos() as f64 / NANOS_PER_MICROSECOND)
        .collect();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::run_skin_ab;

    #[test]
    fn benchmark_smoke_reports_guest_build_even_without_an_adapter() {
        let report = run_skin_ab();
        assert!(report.contains("guest build + lift"));
    }
}
