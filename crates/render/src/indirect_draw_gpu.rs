//! Headless gate: the indirect gbuffer draw (`draw_indexed_indirect`
//! over the culler's compacted batch) renders the same frame as the
//! direct `draw_indexed` fallback. Shadows stay on the direct path:
//! the depth pre-pass must cover off-frame casters, so it never reuses
//! the camera-compacted batch (a per-light-frustum compaction is future
//! work).

use super::test_util;
use super::{
    BatchDraw, GbufferTargets, InstanceData, MaterialIdx, Renderer3D, select_shared_batch,
};
use crate::camera::Frustum;
use crate::gpu_culling::{Culler, CullerError, INDIRECT_ARGS_WORD_COUNT};
use crate::mesh::{Mesh, create_sphere};
use ornis_core::System;

const FRAME_W: u32 = 320;
const FRAME_H: u32 = 180;
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
    let (target, target_view) = test_util::offscreen_target(device, FRAME_W, FRAME_H);
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
        motion: None,
        previous_depth: None,
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
    renderer.render_lighting(device, &mut encoder, &g, &renderer.pbr_texture_view, None);
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
        crate::auto_exposure::DEFAULT_FRAME_DT_SECONDS,
    );
    queue.submit([encoder.finish()]);
    test_util::readback_rgba8(device, queue, &target, FRAME_W, FRAME_H)
}

/// The compacted gbuffer submit must render the same pixels as the
/// direct fallback, with and without a shadowed light (shadows draw
/// directly in both arms, including the off-frame caster's shadow).
#[test]
fn indirect_batch_draws_match_direct() {
    let Some((device, queue)) = test_util::try_device() else {
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
    let Some((device, queue)) = test_util::try_device() else {
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
/// still darkens the in-frame receiver — through the direct and the
/// indirect gbuffer path alike. Fails if shadows ever reuse the
/// camera-compacted batch: the caster would vanish from the shadow
/// map and the frames would match the caster-free reference.
#[test]
fn offscreen_caster_shadow_reaches_frame() {
    let Some((device, queue)) = test_util::try_device() else {
        eprintln!("SKIPPED (offscreen_caster_shadow_reaches_frame): no wgpu adapter");
        return;
    };
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
    let without = render_frame(&device, &queue, false, true, false);
    for indirect in [false, true] {
        let with = render_frame(&device, &queue, indirect, true, true);
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
            "offscreen caster (indirect={indirect}): diff_px={diff_px} mean_with={mean_with:.2} mean_without={mean_without:.2}"
        );
        assert!(
            diff_px > MIN_SHADOW_PIXELS,
            "off-frame caster cast no visible shadow (indirect={indirect}): {diff_px} pixels differ"
        );
        assert!(
            mean_with + 0.5 < mean_without,
            "caster frame is not darker (indirect={indirect}): with={mean_with:.2} without={mean_without:.2}"
        );
    }
}

/// Entities for the submit-selection test: two spheres plus one
/// camera-facing custom quad (XY plane, +Z normal).
fn submit_test_scene(store: &mut ornis_core::SmartStore) {
    use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
    use ornis_core::units::{Clamped01, PositiveF32, UnitQuat};
    let material = || MaterialDesc::Dielectric {
        base_color: [0.8, 0.8, 0.8],
        roughness: Clamped01::new(0.5),
        emission: [0.0, 0.0, 0.0],
        metallic: ornis_core::Metallic::new(0.0),
    };
    let sphere = || MeshDesc::Sphere {
        radius: PositiveF32::expect_valid(1.0),
        segments: 16,
        rings: 12,
    };
    let place =
        |store: &mut ornis_core::SmartStore, translation: [f32; 3], scale: f32, mesh: MeshDesc| {
            let handle = store.create_entity();
            store.insert(
                handle,
                TransformDesc {
                    translation: glam::Vec3::from(translation),
                    rotation: UnitQuat::IDENTITY,
                    scale: glam::Vec3::splat(scale),
                },
            );
            store.insert(handle, mesh);
            store.insert(handle, material());
        };
    place(store, [0.0, 0.0, 0.0], 1.5, sphere());
    place(store, [2.5, 0.5, -1.0], 1.0, sphere());
    place(
        store,
        [-1.5, 0.8, 2.0],
        1.0,
        MeshDesc::Custom {
            positions: vec![
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        },
    );
}

/// Slice 2c regression (review #98): `RenderSubmit` stages the shared
/// sphere batch ONLY for indirect compaction — customs draw their own
/// direct slots. Drives the real submit with a customs-bearing world,
/// then asserts the staged batch holds exactly the two spheres (never
/// the merged vector) and the prepared args count exactly them.
/// Fails while submit stages spheres-plus-customs (the compacted set
/// would carry the custom record).
#[test]
fn submit_stages_spheres_only_for_indirect() {
    const W: u32 = 320;
    const H: u32 = 180;
    let Some((device, queue)) = test_util::try_device() else {
        eprintln!("SKIPPED (submit_stages_spheres_only_for_indirect): no wgpu adapter");
        return;
    };
    let mut engine = ornis_core::Engine::new();
    submit_test_scene(engine.world_mut().store_mut().expect("store"));
    let orbit = crate::camera::OrbitCamera::looking_at(
        glam::Vec3::new(0.0, 2.5, 9.0),
        glam::Vec3::new(0.5, 0.0, 0.0),
    );
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        width: W,
        height: H,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    let renderer = Renderer3D::new(&device, &surface_config, 1);
    renderer.set_indirect_draw(true);
    let frame3d =
        crate::frame_exec::RenderFrame3D::new(wgpu::TextureFormat::Rgba8UnormSrgb, (W, H));
    engine
        .world_mut()
        .insert(std::sync::Mutex::new(orbit.clone()));
    engine
        .world_mut()
        .insert(crate::extraction::RenderLights::default());
    engine
        .world_mut()
        .insert(crate::gpu_resources::GpuDevice(device.clone()));
    engine
        .world_mut()
        .insert(crate::gpu_resources::GpuQueue(queue.clone()));
    engine
        .world_mut()
        .insert(crate::gpu_resources::GpuSurfaceState {
            size: (W, H),
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        });
    engine
        .world_mut()
        .insert(std::sync::Mutex::new(crate::gpu_resources::GpuFrameState {
            renderer,
            frame3d,
        }));
    engine
        .world_mut()
        .insert(crate::gpu_resources::GpuCustomMeshes::default());
    crate::gpu_resources::RenderSubmit.run(engine.world().resources());
    let resources = engine.world().resources();
    let customs = resources
        .get::<crate::gpu_resources::GpuCustomMeshes>()
        .expect("customs resource");
    let staged_customs = customs
        .0
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(staged_customs.len(), 1, "one custom entity staged");
    let fs = resources
        .get::<std::sync::Mutex<crate::gpu_resources::GpuFrameState>>()
        .expect("frame state")
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // The staged batch is the shared sphere batch ONLY — two records,
    // never the merged spheres-plus-customs vector. Scoped: the read
    // guard must drop before `prepare_indirect_batch` takes the write
    // lock below (same-thread RwLock read+write deadlocks).
    {
        let staged = super::read_lock(&fs.renderer.staged_batch);
        let (_, staged_instances) = staged.as_ref().expect("batch staged while armed");
        assert_eq!(staged_instances.len(), 2);
    }
    // Premise pin: the custom center is truly visible, so staging it
    // would wrongly put it in the camera-compacted set (vacuity guard).
    let (center, radius) = crate::extraction::instance_sphere(&staged_customs[0].instance);
    let (view_proj, _) = crate::camera::camera_view_projection(&orbit.view_parameters(), (W, H));
    assert!(
        crate::camera::Frustum::from_view_proj(&view_proj).sphere_visible(center, radius),
        "custom center must sit inside the camera frustum (center={center:?})"
    );
    // The prepared args count exactly the two spheres.
    let sphere_mesh = create_sphere(&device, 1.0, 24, 16);
    fs.renderer
        .prepare_indirect_batch(&device, &queue, sphere_mesh.num_indices);
    assert!(fs.renderer.indirect_batch_ready());
    let culler = super::read_lock(&fs.renderer.culler);
    let args = readback_args(
        &device,
        &queue,
        &culler.as_ref().expect("culler built").args_buffer,
    );
    assert_eq!(args, [sphere_mesh.num_indices, 2, 0, 0, 0]);
}

/// Slice 2c selection through the frame-plan node path: with the
/// opt-in flag armed and a prepared batch, `render_gbuffer_with_custom`
/// submits the compacted batch — pixel-identical to both the direct
/// fallback and the explicit indirect entry point. The selection seam
/// itself is pinned first: only armed-with-culler routes indirectly.
///
/// Over-capacity frames fall back to the direct path: [`Culler::compact`]
/// reports [`CullerError::CapacityExceeded`], and the frame selection
/// maps every non-armed combination (opt-out, missed prepare, exhausted
/// capacity) to `draw_indexed` — never to a partial indirect draw.
#[test]
fn capacity_exceeded_falls_back_to_direct() {
    let Some((device, queue)) = test_util::try_device() else {
        eprintln!("SKIPPED (capacity_exceeded_falls_back_to_direct): no wgpu adapter");
        return;
    };
    let culler = Culler::try_new(&device, 2).expect("non-zero capacity");
    let instances = (0..3)
        .map(|_| InstanceData {
            model_matrix: glam::Mat4::IDENTITY,
            normal_matrix: glam::Mat4::IDENTITY,
            material_index: MaterialIdx::from_raw(0),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        culler.compact(&device, &queue, &glam::Mat4::IDENTITY, 42, &instances),
        Err(CullerError::CapacityExceeded {
            count: 3,
            capacity: 2,
        })
    );
    assert!(matches!(
        select_shared_batch(true, None, 3),
        BatchDraw::Direct { .. }
    ));
    assert!(matches!(
        select_shared_batch(false, Some(&culler), 3),
        BatchDraw::Direct { .. }
    ));
    let renderer = Renderer3D::new(&device, &surface_config(), 1);
    renderer.prepare_indirect_batch(&device, &queue, 42);
    assert!(
        !renderer.indirect_batch_ready(),
        "an unarmed frame must stay on the direct fallback"
    );
}

#[test]
fn armed_with_custom_matches_direct_and_indirect() {
    let Some((device, queue)) = test_util::try_device() else {
        eprintln!("SKIPPED (armed_with_custom_matches_direct_and_indirect): no wgpu adapter");
        return;
    };
    let staged_probe = Culler::try_new(&device, 4).expect("non-zero capacity");
    assert!(matches!(
        select_shared_batch(true, Some(&staged_probe), 3),
        BatchDraw::Indirect { .. }
    ));
    assert!(matches!(
        select_shared_batch(true, None, 3),
        BatchDraw::Direct { .. }
    ));
    assert!(matches!(
        select_shared_batch(false, Some(&staged_probe), 3),
        BatchDraw::Direct { .. }
    ));
    let renderer = Renderer3D::new(&device, &surface_config(), 1);
    let (mesh, instances, view_proj) = setup_scene(&device, &queue, &renderer, true, true);
    let staged = instances.len() as u32;
    renderer.set_indirect_draw(true);
    renderer.stage_batch_inputs(view_proj, instances);
    renderer.prepare_indirect_batch(&device, &queue, mesh.num_indices);
    assert!(renderer.indirect_batch_ready());
    let (target, target_view) = test_util::offscreen_target(&device, FRAME_W, FRAME_H);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("armed selection frame"),
    });
    let g = GbufferTargets {
        albedo: &renderer.gbuffer.albedo_view,
        normal: &renderer.gbuffer.normal_view,
        material_id: &renderer.gbuffer.material_id_view,
        world_position: &renderer.gbuffer.world_position_view,
        material_params: &renderer.gbuffer.material_params_view,
        depth: &renderer.gbuffer.depth_view,
        motion: None,
        previous_depth: None,
    };
    renderer.render_gbuffer_with_custom(&device, &mut encoder, &g, &mesh, staged, &[]);
    renderer.render_shadows(&device, &mut encoder, &mesh, staged);
    renderer.render_lighting(&device, &mut encoder, &g, &renderer.pbr_texture_view, None);
    renderer.render_composite(
        &device,
        &queue,
        &mut encoder,
        crate::renderer::CompositeInputs {
            target: &target_view,
            hdr: &renderer.pbr_texture_view,
            hdr_fwd: &renderer.pbr_texture_view,
            bloom: &renderer.pbr_texture_view,
            bloom_intensity: 0.0,
            mode: 0,
        },
        crate::auto_exposure::DEFAULT_FRAME_DT_SECONDS,
    );
    queue.submit([encoder.finish()]);
    let routed = test_util::readback_rgba8(&device, &queue, &target, FRAME_W, FRAME_H);
    let direct = render_frame(&device, &queue, false, true, true);
    let indirect = render_frame(&device, &queue, true, true, true);
    assert!(
        routed.iter().any(|channel| *channel > 0),
        "armed frame must render something"
    );
    assert_eq!(
        routed, direct,
        "armed with_custom must match the direct fallback"
    );
    assert_eq!(
        routed, indirect,
        "armed with_custom must match the explicit indirect entry point"
    );
}
