//! Headless gate: the indirect gbuffer draw (`draw_indexed_indirect`
//! over the culler's compacted batch) renders the same frame as the
//! direct `draw_indexed` fallback. Shadows stay on the direct path:
//! the depth pre-pass must cover off-frame casters, so it never reuses
//! the camera-compacted batch (a per-light-frustum compaction is future
//! work).

use super::{GbufferTargets, InstanceData, MaterialIdx, Renderer3D};
use crate::camera::Frustum;
use crate::gpu_culling::{Culler, INDIRECT_ARGS_WORD_COUNT};
use crate::mesh::{Mesh, create_sphere};

const FRAME_W: u32 = 320;
const FRAME_H: u32 = 180;
const BPP: u32 = 4;
/// To-light axis of the probe directional light (matches `set_lights`
/// below): the off-frame caster sits on this axis so its shadow lands
/// on the in-frame receiver.
const TO_LIGHT: [f32; 3] = [0.42, 0.84, 0.3];
/// Caster distance from the origin along [`TO_LIGHT`]: far enough to
/// leave the camera frustum, close enough for the ±12 m shadow ortho
/// box (depth 21 of the 10..50 range).
const CASTER_DIST: f32 = 9.0;
/// Pixels that must change between the with/without-caster frames
/// (probe-calibrated order: the r=0.9 occluder moves ~340 px).
const MIN_SHADOW_PIXELS: usize = 200;
/// Per-pixel RGB swing counting as "changed" (same as the shadow
/// probe in `render_backend/tests/shadows_dir.rs`).
const PIXEL_SWING: i16 = 12;

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
/// and the instances (two on-screen plus the off-frame caster on the
/// light axis — culled from the camera image, but its shadow reaches
/// the frame, so the shadow pre-pass must keep it).
fn setup_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer3D,
    shadowed: bool,
    with_caster: bool,
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
    let mut instances = vec![place([0.0, 0.0, 0.0], 1.5), place([2.5, 0.5, -1.0], 1.0)];
    if with_caster {
        let axis = glam::Vec3::from_array(TO_LIGHT).normalize();
        let center = axis * CASTER_DIST;
        instances.push(place(center.to_array(), 1.2));
    }
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
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::from_array(TO_LIGHT))
                .expect("axis"),
            intensity: 1.2,
            color: [1.0, 1.0, 1.0],
            shadow: ornis_assets::scene::ShadowCast::from(shadowed),
        }],
    );
    (mesh, instances, view_proj)
}

/// One full frame rendered either through the direct fallback draws or
/// through the culler's compacted gbuffer batch (`draw_indexed_indirect`;
/// shadows always take the direct pre-pass so off-frame casters keep
/// their shadows); returns the composite pixels.
fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    indirect: bool,
    shadowed: bool,
    with_caster: bool,
) -> Vec<u8> {
    let renderer = Renderer3D::new(device, &surface_config(), 1);
    let (mesh, instances, view_proj) = setup_scene(device, queue, &renderer, shadowed, with_caster);
    let staged = instances.len() as u32;
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
        let culler = Culler::try_new(device, staged).expect("scene is non-empty");
        culler
            .compact(device, queue, &view_proj, mesh.num_indices, &instances)
            .expect("scene fits the capacity");
        renderer.render_gbuffer_indirect(device, &mut encoder, &g, &mesh, &culler);
        renderer.render_shadows(device, &mut encoder, &mesh, staged);
    } else {
        renderer.render_gbuffer(device, &mut encoder, &g, &mesh, staged);
        renderer.render_shadows(device, &mut encoder, &mesh, staged);
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

/// The compacted gbuffer submit must render the same pixels as the
/// direct fallback, with and without a shadowed light (shadows draw
/// directly in both arms, including the off-frame caster's shadow).
#[test]
fn indirect_batch_draws_match_direct() {
    let Some((device, queue)) = open_adapter() else {
        eprintln!("SKIPPED (indirect_batch_draws_match_direct): no wgpu adapter");
        return;
    };
    for shadowed in [false, true] {
        let direct = render_frame(&device, &queue, false, shadowed, true);
        let indirect = render_frame(&device, &queue, true, shadowed, true);
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

/// Words of one `DrawIndexedIndirectArgs` read back from a GPU buffer.
fn readback_args(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
) -> [u32; INDIRECT_ARGS_WORD_COUNT] {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indirect args readback"),
        size: (INDIRECT_ARGS_WORD_COUNT * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("indirect args copy"),
    });
    encoder.copy_buffer_to_buffer(
        source,
        0,
        &staging,
        0,
        (INDIRECT_ARGS_WORD_COUNT * 4) as u64,
    );
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("readback callback")
        .expect("readback maps");
    let view = slice.get_mapped_range().expect("mapped range");
    let words: [u32; INDIRECT_ARGS_WORD_COUNT] = bytemuck::cast_slice(&view)
        [..INDIRECT_ARGS_WORD_COUNT]
        .try_into()
        .expect("five words");
    drop(view);
    staging.unmap();
    words
}

/// The args buffer a `compact` prepares must carry the full shared-mesh
/// index range with exactly the visible count: `index_count` survives
/// the dispatch untouched, `instance_count` counts the two on-screen
/// spheres (the off-frame caster is compacted away, never drawn).
#[test]
fn indirect_args_match_compacted_batch() {
    let Some((device, queue)) = open_adapter() else {
        eprintln!("SKIPPED (indirect_args_match_compacted_batch): no wgpu adapter");
        return;
    };
    let renderer = Renderer3D::new(&device, &surface_config(), 1);
    let (mesh, instances, view_proj) = setup_scene(&device, &queue, &renderer, true, true);
    let culler = Culler::try_new(&device, instances.len() as u32).expect("scene is non-empty");
    culler
        .compact(&device, &queue, &view_proj, mesh.num_indices, &instances)
        .expect("scene fits the capacity");
    assert_eq!(
        readback_args(&device, &queue, &culler.args_buffer),
        [mesh.num_indices, 2, 0, 0, 0],
    );
}

/// Shadow-map regression (review #95): the depth pre-pass draws the
/// staged range directly, so an off-frame caster on the light axis
/// still darkens the in-frame receiver. Fails if shadows ever reuse
/// the camera-compacted batch — the caster would vanish from the
/// shadow map and both frames would match.
#[test]
fn offscreen_caster_shadow_reaches_frame() {
    let Some((device, queue)) = open_adapter() else {
        eprintln!("SKIPPED (offscreen_caster_shadow_reaches_frame): no wgpu adapter");
        return;
    };
    let with = render_frame(&device, &queue, false, true, true);
    let without = render_frame(&device, &queue, false, true, false);
    // Premise pin: the caster sphere is truly outside the camera
    // frustum, so only an unculled shadow pass can darken the frame.
    let probe = Renderer3D::new(&device, &surface_config(), 1);
    let (_, instances, view_proj) = setup_scene(&device, &queue, &probe, true, true);
    let caster = instances.last().expect("caster staged");
    let (center, radius) = crate::extraction::instance_sphere(caster);
    assert!(
        !Frustum::from_view_proj(&view_proj).sphere_visible(center, radius),
        "caster must sit outside the camera frustum (center={center:?}, radius={radius})"
    );
    assert_eq!(with.len(), without.len());
    let mut diff_px = 0usize;
    for (a, b) in with.chunks_exact(4).zip(without.chunks_exact(4)) {
        let swing = (a[0] as i16 - b[0] as i16).abs()
            + (a[1] as i16 - b[1] as i16).abs()
            + (a[2] as i16 - b[2] as i16).abs();
        if swing > PIXEL_SWING {
            diff_px += 1;
        }
    }
    let lum = |px: &[u8]| {
        px.chunks_exact(4)
            .map(|c| f64::from(c[0]) + f64::from(c[1]) + f64::from(c[2]))
            .sum::<f64>()
            / (px.len() / 4) as f64
    };
    let (mean_with, mean_without) = (lum(&with), lum(&without));
    eprintln!(
        "offscreen caster: diff_px={diff_px} mean_with={mean_with:.2} mean_without={mean_without:.2}"
    );
    assert!(
        diff_px > MIN_SHADOW_PIXELS,
        "off-frame caster cast no visible shadow: {diff_px} pixels differ"
    );
    assert!(
        mean_with + 0.5 < mean_without,
        "caster frame is not darker: with={mean_with:.2} without={mean_without:.2}"
    );
}
