//! S5b gate (PLAN Приложение C): parallel command recording must be
//! pixel-identical to the sequential path. Runs headless — on CI via
//! lavapipe, locally on any adapter; skipped when no adapter is found.
//! The scene is one lit sphere (minimal setup from render_graph_probe).

use glam::{Mat4, Quat, Vec3};
use ornis_render::render_backend::RenderContext;
use ornis_render::{InstanceData, OpenPBRMaterial, RenderGraph3D, Renderer3D, Technique};

const SIZE: u32 = 128;
const BPP: u32 = 4;

async fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .ok()
}

fn target(device: &wgpu::Device, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let unpadded = SIZE * BPP;
    let padded = unpadded.div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parallel readback"),
        size: (padded * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("parallel readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).is_ok());
    device.poll(wgpu::PollType::Wait).expect("poll readback");
    receiver
        .recv()
        .expect("map callback")
        .expect("map readback");
    let data = slice.get_mapped_range().to_vec();
    buffer.unmap();
    data
}

#[test]
fn parallel_recording_matches_sequential_pixels() {
    let Some((device, queue)) = pollster::block_on(device()) else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
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
    };
    let renderer = Renderer3D::new(&device, &surface_config, 1);
    let mesh = ornis_render::create_sphere(&device, 1.0, 16, 12);

    let material = OpenPBRMaterial::dielectric()
        .base_color_rgb([0.8, 0.4, 0.2])
        .specular_roughness(0.5);
    renderer.upload_materials(&queue, &[material]);
    let model = Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, Vec3::ZERO);
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

    let (seq_tex, seq_view) = target(&device, "sequential target");
    let (par_tex, par_view) = target(&device, "parallel target");

    // Sequential: the default single-encoder path.
    let mut seq = RenderGraph3D::new_with(
        wgpu::TextureFormat::Rgba8Unorm,
        (SIZE, SIZE),
        Technique::Hybrid,
        true,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sequential encoder"),
    });
    seq.render(
        RenderContext {
            device: &device,
            queue: &queue,
            encoder: &mut encoder,
            target: &seq_view,
        },
        &renderer,
        &mesh,
        1,
    );
    queue.submit(std::iter::once(encoder.finish()));

    // Parallel: a level at a time (lighting ∥ forward inside), own
    // encoders per pass, single ordered submit.
    let mut par = RenderGraph3D::new_with(
        wgpu::TextureFormat::Rgba8Unorm,
        (SIZE, SIZE),
        Technique::Hybrid,
        true,
    );
    assert!(!par.parallel_recording(), "sequential by default");
    par.set_parallel_recording(true);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("parallel caller encoder"),
    });
    par.render(
        RenderContext {
            device: &device,
            queue: &queue,
            encoder: &mut encoder,
            target: &par_view,
        },
        &renderer,
        &mesh,
        1,
    );
    queue.submit(std::iter::once(encoder.finish()));

    device.poll(wgpu::PollType::Wait).expect("poll render");

    let seq_pixels = read_back(&device, &queue, &seq_tex);
    let par_pixels = read_back(&device, &queue, &par_tex);
    assert_eq!(
        seq_pixels,
        par_pixels,
        "parallel recording must be pixel-identical"
    );
}
