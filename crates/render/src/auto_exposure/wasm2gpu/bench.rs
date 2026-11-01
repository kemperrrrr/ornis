//! Warmed A/B timings of production auto-exposure and its three-pass guest.

mod scene;

use std::time::Duration;
use std::time::Instant;

use crate::auto_exposure::AutoExposure;
use crate::auto_exposure::AutoExposureSettings;
use crate::auto_exposure::DEFAULT_FRAME_DT_SECONDS;
use crate::auto_exposure::ExposureFrame;
use crate::auto_exposure::adapt_wgsl;
use crate::auto_exposure::histogram_wgsl;
use crate::auto_exposure::wasm2gpu::GuestGpu;
use crate::auto_exposure::wasm2gpu::GuestInputs;
use crate::auto_exposure::wasm2gpu::lower_guest;
use scene::{TextureUpload, bench_texels, texture};

const WARMUP: usize = 5;
const REPEATS: usize = 30;
const BENCH_MODE: u32 = 2;
const SCENE_SIZES: [u32; 2] = [16, 64];
const NANOS_PER_MICROSECOND: f64 = 1_000.0;
const FIRST_QUARTILE: f64 = 0.25;
const MEDIAN: f64 = 0.5;
const THIRD_QUARTILE: f64 = 0.75;

/// Runs the release A/B bench; setup and WGSL lowering are reported separately.
pub fn run_auto_exposure_ab() -> String {
    let warmup = if cfg!(test) { 0 } else { WARMUP };
    let repeats = if cfg!(test) { 1 } else { REPEATS };
    run_auto_exposure_ab_with(warmup, repeats)
}

fn run_auto_exposure_ab_with(warmup: usize, repeats: usize) -> String {
    let lowered = lower_guest();
    let mut report = format!(
        concat!(
            "wasm2gpu auto-exposure A/B (issue #211)\n",
            "guest build + lift: {:?}\n",
            "production dispatches: histogram + adapt (2); guest dispatches: ",
            "log-luminance + per-bin scan + adapt (3).\n",
            "HDR textures and the guest storage mirror are prepared before timing; ",
            "wall time includes submission and blocking poll.\n",
            "WGSL bytes: production={} guest log={} histogram={} adapt={} total={}\n",
        ),
        lowered.elapsed,
        histogram_wgsl().len() + adapt_wgsl().len(),
        lowered.log_luma.len(),
        lowered.histogram.len(),
        lowered.adapt.len(),
        lowered.log_luma.len() + lowered.histogram.len() + lowered.adapt.len(),
    );
    let Some(session) = open_session(warmup, repeats) else {
        return report;
    };
    report.push_str(&session.header);
    report.push_str("scene\tpixels\tproduction median [q1–q3] us\twasm2gpu median [q1–q3] us\n");
    for edge in SCENE_SIZES {
        report.push_str(&bench_size(&session, &lowered, edge, warmup, repeats));
    }
    report
}

struct Session {
    device: wgpu::Device,
    queue: wgpu::Queue,
    header: String,
}

struct ProductionFrame<'a> {
    session: &'a Session,
    pass: &'a AutoExposure,
    hdr: &'a wgpu::TextureView,
    hdr_fwd: &'a wgpu::TextureView,
    width: u32,
    height: u32,
    settings: AutoExposureSettings,
}

fn open_session(warmup: usize, repeats: usize) -> Option<Session> {
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
            Err(err) => return no_adapter(err.to_string()),
        };
    let info = adapter.get_info();
    let (device, queue) =
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wasm2gpu auto-exposure A/B"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })) {
            Ok(pair) => pair,
            Err(err) => return no_adapter(err.to_string()),
        };
    Some(Session {
        header: adapter_header(&info, warmup, repeats),
        device,
        queue,
    })
}

fn no_adapter(reason: String) -> Option<Session> {
    if crate::gpu_required() {
        eprintln!("ORNIS_REQUIRE_GPU is set but auto-exposure A/B has no wgpu device: {reason}");
        std::process::exit(1);
    }
    eprintln!("wasm2gpu auto-exposure A/B: no wgpu adapter — timings skipped ({reason})");
    None
}

fn adapter_header(info: &wgpu::AdapterInfo, warmup: usize, repeats: usize) -> String {
    let lower_name = info.name.to_ascii_lowercase();
    let software = info.device_type == wgpu::DeviceType::Cpu
        || ["lavapipe", "llvmpipe", "swiftshader"]
            .iter()
            .any(|word| lower_name.contains(*word));
    format!(
        concat!(
            "adapter: {}\nbackend: {:?}\ndevice_type: {:?}\ndriver: {} {}\n",
            "software: {}\nwarmup: {warmup} repeats: {repeats}\n",
            "Times are wall-clock microseconds (encode, submit, and blocking poll); ",
            "software-adapter times are CPU software-rendering costs, not discrete-GPU costs.\n",
        ),
        info.name,
        info.backend,
        info.device_type,
        info.driver,
        info.driver_info,
        if software { "yes" } else { "no" },
    )
}

fn bench_size(
    session: &Session,
    lowered: &super::Lowered,
    edge: u32,
    warmup: usize,
    repeats: usize,
) -> String {
    let width = edge;
    let height = edge;
    let (deferred, forward) = bench_texels(width, height);
    let inputs = GuestInputs {
        width,
        height,
        mode: BENCH_MODE,
        deferred: &deferred,
        forward: &forward,
    };
    let settings = AutoExposureSettings::default().sanitized();
    let guest = GuestGpu::new(&session.device, lowered, width, height);
    guest.prepare(
        &session.queue,
        inputs,
        settings,
        DEFAULT_FRAME_DT_SECONDS,
        1.0,
    );
    let deferred_tex = texture(
        &session.device,
        &session.queue,
        TextureUpload {
            label: "ae bench deferred",
            width,
            height,
            texels: &deferred,
        },
    );
    let forward_tex = texture(
        &session.device,
        &session.queue,
        TextureUpload {
            label: "ae bench forward",
            width,
            height,
            texels: &forward,
        },
    );
    let production = AutoExposure::new(&session.device);
    let production_frame = ProductionFrame {
        session,
        pass: &production,
        hdr: &deferred_tex.1,
        hdr_fwd: &forward_tex.1,
        width,
        height,
        settings,
    };
    for _ in 0..warmup {
        run_production(&production_frame);
        run_guest(session, &guest);
    }
    let mut production_samples = Vec::with_capacity(repeats);
    let mut guest_samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        production_samples.push(run_production(&production_frame));
        guest_samples.push(run_guest(session, &guest));
    }
    format!(
        "{width}x{height}\t{}\t{}\t{}\n",
        width * height,
        quartiles(&production_samples),
        quartiles(&guest_samples),
    )
}

fn run_production(frame: &ProductionFrame<'_>) -> Duration {
    let started = Instant::now();
    let mut encoder = frame
        .session
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("auto-exposure production A/B"),
        });
    frame.pass.encode(
        &frame.session.device,
        &frame.session.queue,
        &mut encoder,
        ExposureFrame {
            hdr: frame.hdr,
            hdr_fwd: frame.hdr_fwd,
            mode: BENCH_MODE,
            width: frame.width,
            height: frame.height,
            settings: frame.settings,
            dt_seconds: DEFAULT_FRAME_DT_SECONDS,
        },
    );
    frame
        .session
        .queue
        .submit(std::iter::once(encoder.finish()));
    let _ = frame
        .session
        .device
        .poll(wgpu::PollType::wait_indefinitely());
    started.elapsed()
}

fn run_guest(session: &Session, guest: &GuestGpu) -> Duration {
    let started = Instant::now();
    let mut encoder = session
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("auto-exposure wasm2gpu A/B"),
        });
    guest.encode(&mut encoder);
    session.queue.submit(std::iter::once(encoder.finish()));
    let _ = session.device.poll(wgpu::PollType::wait_indefinitely());
    started.elapsed()
}

fn quartiles(samples: &[Duration]) -> String {
    let mut values: Vec<f64> = samples
        .iter()
        .map(Duration::as_nanos)
        .map(|n| n as f64 / NANOS_PER_MICROSECOND)
        .collect();
    values.sort_by(f64::total_cmp);
    format!(
        "{:.2} [{:.2}–{:.2}]",
        quantile(&values, MEDIAN),
        quantile(&values, FIRST_QUARTILE),
        quantile(&values, THIRD_QUARTILE),
    )
}

fn quantile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::run_auto_exposure_ab;

    #[test]
    fn benchmark_smoke_reports_guest_build_even_without_an_adapter() {
        let report = run_auto_exposure_ab();
        assert!(report.contains("guest build + lift"));
    }
}
