#![allow(deprecated)]
//! S5b bench: sequential vs parallel command recording, CPU side.
//!
//! Параллельная запись оптимизирует CPU-сторону кадра (запись команд в
//! encoder'ы) — именно её и меряем: оба варианта платят одинаковые
//! submit; render() без poll. Headless-адаптер (в CI — lavapipe): числа
//! показывают относительную разницу путей записи на одном железе, а не
//! абсолютный кадр на дискретном GPU.
//!
//! Compile-checked гейтом; ручной запуск:
//!   cargo bench -p ornis-render --bench recording_bench

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use glam::{Mat4, Vec3};
use ornis_render::render_backend::RenderContext;
use ornis_render::{InstanceData, OpenPBRMaterial, RenderFrame3D, Renderer3D, Technique};

const SIZE: u32 = 256;

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
    let mesh = ornis_render::create_sphere(&device, 1.0, 16, 12);

    let material = {
        let mut mat = OpenPBRMaterial::dielectric();
        mat.base.color_rgb([0.8, 0.4, 0.2]);
        mat.specular.roughness(0.5);
        mat
    };
    renderer.upload_materials(&queue, &[material]);
    let model = Mat4::from_translation(Vec3::ZERO);
    let instance = InstanceData {
        model_matrix: model,
        normal_matrix: model.inverse().transpose(),
        material_index: 0,
    };
    renderer.upload_instances(&queue, &[instance]);
    renderer.set_lights(
        &queue,
        [0.1, 0.1, 0.1],
        &[([0.3, -1.0, 0.5], 1.0, [1.0, 1.0, 1.0])],
    );
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(60f32.to_radians(), 1.0, 0.1, 10.0);
    let view_proj = proj * view;
    renderer.set_camera(&queue, &view_proj.to_cols_array_2d(), [0.0, 0.0, 3.0]);

    let view_seq = target_view(&device);
    let view_par = target_view(&device);
    let mut seq = RenderFrame3D::new_with(
        wgpu::TextureFormat::Rgba8Unorm,
        (SIZE, SIZE),
        Technique::Hybrid,
        true,
    );
    let mut par = RenderFrame3D::new_with(
        wgpu::TextureFormat::Rgba8Unorm,
        (SIZE, SIZE),
        Technique::Hybrid,
        true,
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
            black_box(&mut seq).render(
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
            black_box(&mut par).render(
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
