//! Headless gate: indirect shared-batch draws (`draw_indexed_indirect`
//! over the culler's compacted batch) render the same frame as the
//! direct `draw_indexed` fallback — gbuffer and shadow pre-pass included.

use super::{GbufferTargets, InstanceData, MaterialIdx, Renderer3D};
use crate::gpu_culling::Culler;
use crate::mesh::{Mesh, create_sphere};

const FRAME_W: u32 = 320;
const FRAME_H: u32 = 180;
const BPP: u32 = 4;
const SCENE_INSTANCES: u32 = 3;

fn open_adapter() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
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
    })
}

fn surface_config() -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        width: FRAME_W,
        height: FRAME_H,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    }
}

fn target_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("indirect parity target"),
        size: wgpu::Extent3d {
            width: FRAME_W,
            height: FRAME_H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn readback(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let unpadded = FRAME_W * BPP;
    let padded = unpadded.div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indirect parity readback"),
        size: u64::from(padded * FRAME_H),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("indirect parity copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: FRAME_W,
            height: FRAME_H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("readback callback")
        .expect("readback maps");
    let view = slice.get_mapped_range().expect("mapped range");
    let mut pixels = Vec::with_capacity((unpadded * FRAME_H) as usize);
    for row in 0..FRAME_H {
        let start = (row * padded) as usize;
        pixels.extend_from_slice(&view[start..start + unpadded as usize]);
    }
    drop(view);
    buffer.unmap();
    pixels
}

/// Scene + camera + lights uploaded into `renderer`; returns the mesh
/// and the three instances (two on-screen, one far offscreen).
fn setup_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer3D,
    shadowed: bool,
) -> (Mesh, Vec<InstanceData>, glam::Mat4) {
    let mesh = create_sphere(device, 1.0, 24, 16);
    let mut gray = ornis_core::OpenPBRMaterial::dielectric();
    gray.base.color_rgb([0.8, 0.8, 0.8]);
    gray.specular.roughness(0.5);
    renderer.upload_materials(device, queue, &[gray]);
    let place = |translation: [f32; 3], scale: f32| InstanceData {
        model_matrix: glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(scale),
            glam::Quat::IDENTITY,
            glam::Vec3::from(translation),
        ),
        normal_matrix: glam::Mat4::IDENTITY,
        material_index: MaterialIdx::from_raw(0),
    };
    let instances = vec![
        place([0.0, 0.0, 0.0], 1.5),
        place([2.5, 0.5, -1.0], 1.0),
        place([100.0, 0.0, 0.0], 1.0),
    ];
    renderer.upload_instances(device, queue, &instances);
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::new(0.0, 2.5, 9.0),
        glam::Vec3::new(0.5, 0.0, 0.0),
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        55.0f32.to_radians(),
        FRAME_W as f32 / FRAME_H as f32,
        0.1,
        100.0,
    );
    let view_proj = proj * view;
    renderer.set_camera(queue, &view_proj.to_cols_array_2d(), [0.0, 2.5, 9.0]);
    renderer.set_lights(
        queue,
        [0.05, 0.05, 0.08],
        &[ornis_assets::scene::LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(0.42, 0.84, 0.3))
                .expect("axis"),
            intensity: 1.2,
            color: [1.0, 1.0, 1.0],
            shadow: ornis_assets::scene::ShadowCast::from(shadowed),
        }],
    );
    (mesh, instances, view_proj)
}

/// One full frame rendered either through the direct fallback draws or
/// through the culler's compacted batch (`draw_indexed_indirect` in
/// gbuffer + shadows); returns the composite pixels.
fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    indirect: bool,
    shadowed: bool,
) -> Vec<u8> {
    let renderer = Renderer3D::new(device, &surface_config(), 1);
    let (mesh, instances, view_proj) = setup_scene(device, queue, &renderer, shadowed);
    let (target, target_view) = target_texture(device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("indirect parity frame"),
    });
    let g = GbufferTargets {
        albedo: &renderer.gbuffer.albedo_view,
        normal: &renderer.gbuffer.normal_view,
        material_id: &renderer.gbuffer.material_id_view,
        world_position: &renderer.gbuffer.world_position_view,
        material_params: &renderer.gbuffer.material_params_view,
        depth: &renderer.gbuffer.depth_view,
    };
    if indirect {
        let culler = Culler::try_new(device, SCENE_INSTANCES).expect("capacity above zero");
        culler
            .compact(device, queue, &view_proj, mesh.num_indices, &instances)
            .expect("scene fits the capacity");
        renderer.render_gbuffer_indirect(device, &mut encoder, &g, &mesh, &culler);
        renderer.render_shadows_indirect(device, &mut encoder, &mesh, &culler);
    } else {
        renderer.render_gbuffer(device, &mut encoder, &g, &mesh, SCENE_INSTANCES);
        renderer.render_shadows(device, &mut encoder, &mesh, SCENE_INSTANCES);
    }
    renderer.render_lighting(device, &mut encoder, &g, &renderer.pbr_texture_view);
    renderer.render_composite(
        device,
        queue,
        &mut encoder,
        crate::renderer::CompositeInputs {
            target: &target_view,
            hdr: &renderer.pbr_texture_view,
            hdr_fwd: &renderer.pbr_texture_view,
            bloom: &renderer.pbr_texture_view,
            bloom_intensity: 0.0,
            mode: 0,
        },
    );
    queue.submit([encoder.finish()]);
    readback(device, queue, &target)
}

/// The compacted-batch submits must render the same pixels as the
/// direct fallback, with and without a shadowed light (the shadow
/// pre-pass draws indirectly too).
#[test]
fn indirect_batch_draws_match_direct() {
    let Some((device, queue)) = open_adapter() else {
        eprintln!("SKIPPED (indirect_batch_draws_match_direct): no wgpu adapter");
        return;
    };
    for shadowed in [false, true] {
        let direct = render_frame(&device, &queue, false, shadowed);
        let indirect = render_frame(&device, &queue, true, shadowed);
        assert!(
            direct.iter().any(|channel| *channel > 0),
            "scene must render something (shadowed={shadowed})"
        );
        assert_eq!(
            direct, indirect,
            "indirect batch draws must match the direct fallback (shadowed={shadowed})"
        );
    }
}
