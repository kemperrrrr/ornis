//! Manual M1 benchmark for the editor outline pass (issue #215 review).
//!
//! Compares a full hybrid + FXAA frame with the outline idle, one selected
//! and all instances selected. The GPU column uses pass/encoder timestamp
//! queries when the adapter exposes them; otherwise it falls back to wall
//! time. The scene is 512 unit spheres on a grid, 1280×720, Fxaa.
//!
//! Run (ignored by default):
//! `cargo test -p ornis-render --test outline_bench -- --ignored --nocapture`

use glam::{Mat4, Quat, Vec3};
use ornis_render::{
    AaWiring, AntiAliasMode, Bloom, CameraView, InstanceData, MaterialIdx, Mesh, RenderContext,
    RenderFrame3D, Renderer3D, Technique, camera_view_projection, create_sphere,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const COUNT: u32 = 512;
const ROWS: u32 = 16;
const COLS: u32 = 32;
const SPACING: f32 = 1.5;
const RADIUS: f32 = 0.5;
const CAMERA: [f32; 3] = [0.0, 0.0, 30.0];
const WARMUP: u32 = 10;
const SAMPLES: u32 = 30;
const QUERY_BYTES: u64 = 2 * std::mem::size_of::<u64>() as u64;

struct Bench<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    renderer: &'a Renderer3D,
    mesh: &'a Mesh,
    target: &'a wgpu::TextureView,
    queries: Option<&'a wgpu::QuerySet>,
    resolve: Option<&'a wgpu::Buffer>,
    readback: Option<&'a wgpu::Buffer>,
    period_ns: f32,
}

impl Bench<'_> {
    /// One full frame; returns GPU nanoseconds when timestamps are on.
    fn sample(&self, plan: &mut RenderFrame3D, flags: &[u32]) -> Option<f64> {
        plan.set_outline_flags(flags);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        if let Some(queries) = self.queries {
            encoder.write_timestamp(queries, 0);
        }
        plan.render(
            RenderContext {
                device: self.device,
                queue: self.queue,
                encoder: &mut encoder,
                target: self.target,
            },
            self.renderer,
            self.mesh,
            COUNT,
        );
        if let (Some(queries), Some(resolve), Some(readback)) =
            (self.queries, self.resolve, self.readback)
        {
            encoder.write_timestamp(queries, 1);
            encoder.resolve_query_set(queries, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, QUERY_BYTES);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let readback = self.readback?;
        let slice = readback.slice(..QUERY_BYTES);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok()?.ok()?;
        let view = slice.get_mapped_range().ok()?;
        let stamps: &[u64] = bytemuck::cast_slice(&view);
        let delta = stamps.get(1)?.saturating_sub(*stamps.first()?);
        drop(view);
        readback.unmap();
        Some(delta as f64 * f64::from(self.period_ns))
    }
}

fn open() -> Option<(wgpu::Device, wgpu::Queue, bool)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let wanted = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    let features = wanted & adapter.features();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("outline bench"),
        required_features: features,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .ok()?;
    let timed = features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
    Some((device, queue, timed))
}

fn instances() -> Vec<InstanceData> {
    let mut out = Vec::with_capacity(COUNT as usize);
    for row in 0..ROWS {
        for col in 0..COLS {
            let x = (col as f32 - (COLS as f32 - 1.0) / 2.0) * SPACING;
            let y = (row as f32 - (ROWS as f32 - 1.0) / 2.0) * SPACING;
            out.push(InstanceData {
                model_matrix: Mat4::from_scale_rotation_translation(
                    Vec3::splat(RADIUS),
                    Quat::IDENTITY,
                    Vec3::new(x, y, 0.0),
                ),
                normal_matrix: Mat4::IDENTITY,
                material_index: MaterialIdx::from_raw(0),
            });
        }
    }
    out
}

fn target(device: &wgpu::Device) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("outline bench target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn summary(mut values: Vec<f64>) -> (f64, f64) {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite timing"));
    let min = values.first().copied().unwrap_or_default();
    let median = values.get(values.len() / 2).copied().unwrap_or_default();
    (min, median)
}

// qual:allow(complexity, max_function_lines=122) reason: "one benchmark entry wires adapter, scene, three flag modes and the report"
// qual:allow(dry, boilerplate) reason: "the benchmark frame context is one struct literal"
#[test]
#[ignore = "manual M1 benchmark; run with --ignored --nocapture"]
fn outline_bench_full_frame() {
    let Some((device, queue, timed)) = open() else {
        eprintln!("SKIPPED (outline bench): no wgpu adapter");
        return;
    };
    let renderer = Renderer3D::new(
        &device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: FORMAT,
            width: WIDTH,
            height: HEIGHT,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        },
        1,
    );
    renderer.set_anti_alias(AntiAliasMode::Fxaa);
    let view = CameraView {
        eye: Vec3::from_array(CAMERA),
        target: Vec3::ZERO,
        up: Vec3::Y,
        projection: ornis_assets::scene::CameraProjection::Perspective,
        fov: ornis_core::units::Degrees::new(60.0),
        near: ornis_core::units::Meters::new(0.1),
        far: ornis_core::units::Meters::new(100.0),
    };
    let (view_proj, eye) = camera_view_projection(&view, (WIDTH, HEIGHT));
    renderer.set_camera(&queue, &view_proj.to_cols_array_2d(), eye.to_array());
    renderer.upload_instances(&device, &queue, &instances());
    let mesh = create_sphere(&device, 1.0, 32, 24);
    let target = target(&device);
    let mut plan = RenderFrame3D::new_with_aa_outline(
        FORMAT,
        (WIDTH, HEIGHT),
        Technique::Hybrid,
        Bloom::Off,
        AaWiring::new(AntiAliasMode::Fxaa),
    );
    let queries = timed.then(|| {
        device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("outline bench stamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        })
    });
    let resolve = timed.then(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outline bench resolve"),
            size: QUERY_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });
    let readback = timed.then(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outline bench readback"),
            size: QUERY_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    let bench = Bench {
        device: &device,
        queue: &queue,
        renderer: &renderer,
        mesh: &mesh,
        target: &target,
        queries: queries.as_ref(),
        resolve: resolve.as_ref(),
        readback: readback.as_ref(),
        period_ns: queue.get_timestamp_period(),
    };
    let idle = vec![0u32; COUNT as usize];
    let mut one = idle.clone();
    one[(8 * COLS + COLS / 2) as usize] = 2;
    let all = vec![2u32; COUNT as usize];
    let modes = [
        ("idle", idle.as_slice()),
        ("one selected", one.as_slice()),
        ("all selected", all.as_slice()),
    ];
    let mut gpu: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut wall: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (_, flags) in modes {
        for _ in 0..WARMUP {
            let _ = bench.sample(&mut plan, flags);
        }
    }
    for _ in 0..SAMPLES {
        for (index, (_, flags)) in modes.iter().enumerate() {
            let started = std::time::Instant::now();
            let gpu_ns = bench.sample(&mut plan, flags);
            wall[index].push(started.elapsed().as_secs_f64() * 1e3);
            if let Some(ns) = gpu_ns {
                gpu[index].push(ns / 1e6);
            }
        }
    }
    println!(
        "outline bench: {WIDTH}x{HEIGHT}, {COUNT} spheres, hybrid+FXAA, {} samples",
        SAMPLES
    );
    println!(
        "timestamps: {} (INSIDE_ENCODERS {})",
        bench.queries.is_some(),
        timed
    );
    for (index, (label, _)) in modes.iter().enumerate() {
        let (wall_min, wall_med) = summary(wall[index].clone());
        let gpu_text = if gpu[index].is_empty() {
            "unmeasured".to_string()
        } else {
            let (min, med) = summary(gpu[index].clone());
            format!("gpu min {min:.4} ms, median {med:.4} ms")
        };
        println!("{label}: {gpu_text}; wall min {wall_min:.3} ms, median {wall_med:.3} ms");
    }
}
