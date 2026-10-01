//! S5b bench: sequential vs parallel command recording, CPU side.
//!
//! Parallel recording optimizes the CPU side of the frame (command recording
//! into encoders) — that is what we measure: both paths pay the same
//! submit; render() without poll. Headless adapter (lavapipe on CI): numbers
//! show the relative difference between recording paths on one machine, not
//! the absolute frame cost on a discrete GPU.
//!
//! Compile-checked by the gate; manual run:
//!   cargo bench -p ornis-render --bench recording_bench

use criterion::{Criterion, criterion_group, criterion_main};
use glam::{Mat4, Vec3};
use ornis_assets::scene::ShadowCast;
use ornis_render::render_backend::RenderContext;
use ornis_render::{
    Bloom, InstanceData, MaterialIdx, OpenPBRMaterial, RenderFrame3D, Renderer3D, Technique,
};

const SIZE: u32 = 256;
/// Sphere mesh longitude segments.
const SPHERE_LONGITUDES: u32 = 16;
/// Sphere mesh latitude segments.
const SPHERE_LATITUDES: u32 = 12;
/// Dielectric base color for the bench material.
const BASE_COLOR: [f32; 3] = [0.8, 0.4, 0.2];
/// Scene ambient RGB (flat grey fill).
const AMBIENT: [f32; 3] = [0.1, 0.1, 0.1];
/// Directional light direction (world space).
const LIGHT_DIR: [f32; 3] = [0.3, -1.0, 0.5];
/// Camera eye distance along +Z.
const CAMERA_Z: f32 = 3.0;
/// Vertical FOV in degrees.
const FOV_DEG: f32 = 60.0;
/// Perspective near plane.
const NEAR: f32 = 0.1;
/// Perspective far plane.
const FAR: f32 = 10.0;
/// Default dielectric specular roughness for the bench material.
const DEFAULT_ROUGHNESS: f32 = 0.5;

async fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok()?;
    adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .ok()
}

fn target_view(device: &wgpu::Device) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("recording bench target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn bench_recording(c: &mut Criterion) {
    let Some((device, queue)) = pollster::block_on(device()) else {
        eprintln!("SKIP: no wgpu adapter (run on any machine with lavapipe/GPU)");
        return;
    };

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: SIZE,
        height: SIZE,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    let renderer = Renderer3D::new(&device, &surface_config, 1);
    let mesh = ornis_render::create_sphere(&device, 1.0, SPHERE_LONGITUDES, SPHERE_LATITUDES);

    let material = {
        let mut mat = OpenPBRMaterial::dielectric();
        mat.base.color_rgb(BASE_COLOR);
        mat.specular.roughness(DEFAULT_ROUGHNESS);
        mat
    };
    renderer.upload_materials(&device, &queue, &[material]);
    let model = Mat4::from_translation(Vec3::ZERO);
    let instance = InstanceData {
        model_matrix: model,
        normal_matrix: model.inverse().transpose(),
        material_index: MaterialIdx::from_raw(0),
    };
    renderer.upload_instances(&device, &queue, &[instance]);
    renderer.set_lights(
        &queue,
        AMBIENT,
        &[ornis_assets::scene::LightDesc::Directional {
            direction: LIGHT_DIR,
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            shadow: ShadowCast::Disabled,
        }],
    );
    let view =
        glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, CAMERA_Z), Vec3::ZERO, Vec3::Y);
    let proj =
        glam::camera::rh::proj::directx::perspective(FOV_DEG.to_radians(), 1.0, NEAR, FAR);
    let view_proj = proj * view;
    renderer.set_camera(
        &queue,
        &view_proj.to_cols_array_2d(),
        [0.0, 0.0, CAMERA_Z],
    );

    let view_seq = target_view(&device);
    let view_par = target_view(&device);
    let mut seq = RenderFrame3D::new_with(
        wgpu::TextureFormat::Rgba8Unorm,
        (SIZE, SIZE),
        Technique::Hybrid,
        Bloom::On,
    );
    let mut par = RenderFrame3D::new_with(
        wgpu::TextureFormat::Rgba8Unorm,
        (SIZE, SIZE),
        Technique::Hybrid,
        Bloom::On,
    );
    par.set_parallel_recording(true);

    // Warm both pools so texture allocation stays out of the measurement.
    let warm = |frame: &mut RenderFrame3D, target: &wgpu::TextureView| {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("warm"),
        });
        frame.render(
            RenderContext {
                device: &device,
                queue: &queue,
                encoder: &mut encoder,
                target,
            },
            &renderer,
            &mesh,
            1,
        );
        queue.submit(std::iter::once(encoder.finish()));
    };
    warm(&mut seq, &view_seq);
    warm(&mut par, &view_par);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let mut group = c.benchmark_group("recording");
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let desc = wgpu::CommandEncoderDescriptor { label: None };
            let mut encoder = device.create_command_encoder(&desc);
            std::hint::black_box(&mut seq).render(
                RenderContext {
                    device: &device,
                    queue: &queue,
                    encoder: &mut encoder,
                    target: &view_seq,
                },
                &renderer,
                &mesh,
                1,
            );
            queue.submit(std::iter::once(encoder.finish()));
        })
    });
    group.bench_function("parallel", |b| {
        b.iter(|| {
            let desc = wgpu::CommandEncoderDescriptor { label: None };
            let mut encoder = device.create_command_encoder(&desc);
            std::hint::black_box(&mut par).render(
                RenderContext {
                    device: &device,
                    queue: &queue,
                    encoder: &mut encoder,
                    target: &view_par,
                },
                &renderer,
                &mesh,
                1,
            );
            queue.submit(std::iter::once(encoder.finish()));
        })
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    group.finish();
}

criterion_group!(benches, bench_recording);
criterion_main!(benches);
