//! Shared headless-GPU harness for the render pixel-parity gates
//! (`parallel_render.rs` — S5b, `schedule_render.rs` — E1): adapter and
//! device acquisition, offscreen targets, texel readback and the
//! lit-sphere scene. One copy of the setup — the gate binaries keep
//! zero duplicated harness code (the rustqual ratchet would flag an
//! exact `DUPLICATE` pair otherwise).
//!
//! Skip contract: [`HeadlessScene::new`] returns `None` when no wgpu
//! adapter is available; the caller prints a SKIP note and returns, so
//! the gate stays green on machines without GPU drivers (CI provides
//! lavapipe).

use glam::{Mat4, Quat, Vec3};
use ornis_render::render_backend::RenderContext;
use ornis_render::{InstanceData, OpenPBRMaterial, RenderFrame3D, Renderer3D, Technique};

/// Offscreen target edge in pixels (square).
pub const SIZE: u32 = 128;
/// Bytes per pixel of the read-back format.
pub const BPP: u32 = 4;

/// The lit-sphere scene shared by the parity gates: device pair plus a
/// renderer with one uploaded material/instance, lights and camera set.
pub struct HeadlessScene {
    /// Logical device (software adapter on CI — lavapipe).
    pub device: wgpu::Device,
    /// Upload/submit queue.
    pub queue: wgpu::Queue,
    /// Deferred renderer with pipelines and buffers.
    pub renderer: Renderer3D,
    /// Unit sphere mesh (16×12 tessellation).
    pub mesh: ornis_render::Mesh,
}

async fn request_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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

fn surface_config() -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: SIZE,
        height: SIZE,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    }
}

impl HeadlessScene {
    /// Creates the scene, or `None` when no adapter exists (skip).
    pub fn new() -> Option<Self> {
        let (device, queue) = pollster::block_on(request_device())?;
        let renderer = Renderer3D::new(&device, &surface_config(), 1);
        let mesh = ornis_render::create_sphere(&device, 1.0, 16, 12);

        let material = {
            let mut mat = OpenPBRMaterial::dielectric();
            mat.base.color_rgb([0.8, 0.4, 0.2]);
            mat.specular.roughness(0.5);
            mat
        };
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

        let view =
            glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(60f32.to_radians(), 1.0, 0.1, 10.0);
        let view_proj = proj * view;
        renderer.set_camera(&queue, &view_proj.to_cols_array_2d(), [0.0, 0.0, 3.0]);

        Some(Self {
            device,
            queue,
            renderer,
            mesh,
        })
    }
}

/// Runs `gate` with a headless scene, or prints a skip note and returns
/// when no adapter exists — the standard entry for the parity gates.
pub fn with_headless_scene<F>(gate: F)
where
    F: FnOnce(&HeadlessScene),
{
    let Some(scene) = HeadlessScene::new() else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return;
    };
    gate(&scene);
}

/// The sequential graph path over the shared scene — the reference
/// rendering every gate compares its variant against.
pub fn sequential_reference_pixels(scene: &HeadlessScene) -> Vec<u8> {
    render_frame_pixels(scene, |plan, context| {
        plan.render(context, &scene.renderer, &scene.mesh, 1);
    })
}

/// Fresh offscreen render target (`COPY_SRC` for readback).
pub fn target(device: &wgpu::Device, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
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

/// Renders one Hybrid+bloom frame into a fresh target: builds the plan,
/// hands it to `drive` (the gate-specific recording call), submits and
/// reads the pixels back. The gates differ only in the closure.
pub fn render_frame_pixels<F>(scene: &HeadlessScene, drive: F) -> Vec<u8>
where
    F: FnOnce(&mut RenderFrame3D, RenderContext<'_>),
{
    let device = &scene.device;
    let queue = &scene.queue;
    let (texture, view) = target(device, "pixel-parity target");
    let mut plan = RenderFrame3D::new_with(
        wgpu::TextureFormat::Rgba8Unorm,
        (SIZE, SIZE),
        Technique::Hybrid,
        true,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pixel-parity encoder"),
    });
    drive(
        &mut plan,
        RenderContext {
            device,
            queue,
            encoder: &mut encoder,
            target: &view,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    read_back(device, queue, &texture)
}

/// Reads a `COPY_SRC` texture back to CPU bytes (blocking).
pub fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let padded = SIZE * BPP;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixel-parity readback buffer"),
        size: (padded * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pixel-parity readback encoder"),
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
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll readback");
    receiver
        .recv()
        .expect("map callback")
        .expect("map readback");
    let data = slice.get_mapped_range().unwrap().to_vec();
    buffer.unmap();
    data
}
