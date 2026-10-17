//! CSM directional probes: the near scene stays in parity with the legacy
//! shadow box, and a far receiver stays dark without shimmering when the
//! camera moves by less than a texel.

use super::super::*;
use super::*;
use crate::flags::ShadowCast;
use crate::mesh::Mesh;

const PROBE_W: u32 = 320;
const PROBE_H: u32 = 180;
/// Channel sum above which two pixels count as different.
const PIXEL_DIFF: i16 = 12;

struct CsmRig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    backend: Box<dyn RenderBackend>,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    mesh: Mesh,
    to_light: glam::Vec3,
}

fn open_csm_rig(label: &'static str, skip: &'static str) -> Option<CsmRig> {
    {
        let Some((device, queue)) = try_device() else {
            eprintln!("{skip}");
            return None;
        };
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: PROBE_W,
                height: PROBE_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: PROBE_W,
            height: PROBE_H,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        let mut backend = create_render_backend(
            &device,
            &RenderBackendConfig {
                surface_config,
                sample_count: 1,
                exposure: 1.0,
                max_objects: 256,
                max_materials: 64,
                indirect_draw: false,
                ..RenderBackendConfig::default()
            },
        );
        let mesh = crate::mesh::create_sphere(&device, 1.0, 24, 16);
        let mut gray = ornis_core::OpenPBRMaterial::dielectric();
        gray.base.color_rgb([0.8, 0.8, 0.8]);
        gray.specular.roughness(0.5);
        backend.upload_materials(&device, &queue, &[gray, gray]);
        Some(CsmRig {
            device,
            queue,
            backend,
            target,
            view,
            mesh,
            to_light: glam::Vec3::new(0.42, 0.84, 0.3).normalize(),
        })
    }
}

fn csm_instance(translation: [f32; 3], scale: f32) -> crate::renderer::InstanceData {
    let model = glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::splat(scale),
        glam::Quat::IDENTITY,
        glam::Vec3::from(translation),
    );
    crate::renderer::InstanceData {
        model_matrix: model,
        normal_matrix: model.inverse().transpose(),
        material_index: crate::renderer::MaterialIdx::from_raw(0),
    }
}

/// Aim the probe camera and return the projection so a later nudge can
/// reuse it with a new view.
fn aim_csm(
    rig: &mut CsmRig,
    eye: glam::Vec3,
    target: glam::Vec3,
    near: f32,
    far: f32,
) -> glam::Mat4 {
    {
        let fwd = (target - eye).normalize();
        let view = glam::camera::rh::view::look_at_mat4(eye, target, glam::Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(
            55.0f32.to_radians(),
            PROBE_W as f32 / PROBE_H as f32,
            near,
            far,
        );
        rig.backend.set_camera(
            &rig.queue,
            &(proj * view).to_cols_array_2d(),
            eye.to_array(),
        );
        rig.backend.set_csm_camera(near, far, fwd.to_array());
        proj
    }
}

fn pixel_diff(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(x, y)| {
            (x[0] as i16 - y[0] as i16).abs()
                + (x[1] as i16 - y[1] as i16).abs()
                + (x[2] as i16 - y[2] as i16).abs()
                > PIXEL_DIFF
        })
        .count()
}

fn mean_luma(px: &[u8]) -> f64 {
    px.chunks_exact(4)
        .map(|c| (c[0] as f64 + c[1] as f64 + c[2] as f64) / 3.0)
        .sum::<f64>()
        / (px.len() / 4) as f64
}

fn csm_probe_frame(rig: &mut CsmRig, shadow: ShadowCast) -> Vec<u8> {
    {
        rig.backend.set_lights(
            &rig.queue,
            [0.05, 0.05, 0.08],
            &[ornis_assets::scene::LightDesc::Directional {
                direction: ornis_core::units::UnitVec3::normalize(rig.to_light)
                    .expect("non-zero direction"),
                intensity: 1.2,
                color: [1.0, 1.0, 1.0],
                shadow: ornis_assets::scene::ShadowCast::from(shadow.is_enabled()),
            }],
        );
        let mut encoder = rig
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("csm probe encoder"),
            });
        rig.backend.render_scene(
            RenderContext {
                device: &rig.device,
                queue: &rig.queue,
                encoder: &mut encoder,
                target: &rig.view,
            },
            &rig.mesh,
            2,
        );
        let bpp = 4u32;
        let unpadded = PROBE_W * bpp;
        let padded = unpadded.div_ceil(256) * 256;
        let readback = rig.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("csm probe readback"),
            size: (padded * PROBE_H) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &rig.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(PROBE_H),
                },
            },
            wgpu::Extent3d {
                width: PROBE_W,
                height: PROBE_H,
                depth_or_array_layers: 1,
            },
        );
        rig.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        rig.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = slice.get_mapped_range().unwrap();
        let mut pixels = vec![0u8; (unpadded * PROBE_H) as usize];
        for y in 0..PROBE_H as usize {
            pixels[y * unpadded as usize..][..unpadded as usize]
                .copy_from_slice(&data[y * padded as usize..][..unpadded as usize]);
        }
        drop(data);
        readback.unmap();
        pixels
    }
}

/// CSM near-parity probe (#121, diagnostic): the legacy near scene with
/// CSM opted in must still darken the receiver — the first cascade covers
/// the same geometry the legacy box did.
#[test]
fn csm_directional_shadows_near_receiver_like_legacy() {
    {
        let Some(mut rig) = open_csm_rig(
            "csm parity probe target",
            "no GPU adapter; skipping CSM parity probe",
        ) else {
            return;
        };
        rig.backend.upload_instances(
            &rig.device,
            &rig.queue,
            &[
                csm_instance([0.0, 0.0, 0.0], 2.0),
                csm_instance([1.5, 2.9, 1.05], 0.9),
            ],
        );
        aim_csm(
            &mut rig,
            glam::Vec3::new(0.0, 3.0, 10.0),
            glam::Vec3::new(0.0, 1.0, 0.0),
            0.1,
            100.0,
        );
        let on = csm_probe_frame(&mut rig, ShadowCast::Enabled);
        let off = csm_probe_frame(&mut rig, ShadowCast::Disabled);
        let effect = pixel_diff(&on, &off);
        eprintln!("csm parity probe: effect={effect}");
        assert!(effect > 200, "CSM on the near scene casts no shadow");
    }
}

/// CSM probe (#121): on a scene deeper than 100 m the cascaded directional
/// shadow must still darken a far receiver, and a sub-texel camera nudge
/// must not shimmer the frame. Skipped when no adapter is available.
#[test]
fn csm_directional_shadows_far_receiver_without_shimmer() {
    {
        let Some(mut rig) = open_csm_rig("csm probe target", "no GPU adapter; skipping CSM probe")
        else {
            return;
        };
        const NEAR: f32 = 0.1;
        const FAR: f32 = 200.0;
        // Far receiver (r=10) with a clear-flying occluder (r=5): the
        // occluder sits on the light ray through the camera-facing disc
        // center, so its shadow lands mid-disc instead of grazing the rim.
        let receiver = glam::Vec3::new(0.0, 0.0, -70.0);
        let disc = receiver + glam::Vec3::new(0.0, 1.1, 10.0);
        let occluder = disc + rig.to_light * 18.0;
        rig.backend.upload_instances(
            &rig.device,
            &rig.queue,
            &[
                csm_instance(receiver.to_array(), 10.0),
                csm_instance(occluder.to_array(), 5.0),
            ],
        );
        let eye = glam::Vec3::new(0.0, 10.0, 20.0);
        let target = glam::Vec3::new(0.0, 2.0, -60.0);
        let proj = aim_csm(&mut rig, eye, target, NEAR, FAR);
        let on = csm_probe_frame(&mut rig, ShadowCast::Enabled);
        let off = csm_probe_frame(&mut rig, ShadowCast::Disabled);
        let effect = pixel_diff(&on, &off);
        let (mean_on, mean_off) = (mean_luma(&on), mean_luma(&off));
        eprintln!("csm probe: effect={effect} mean_on={mean_on:.2} mean_off={mean_off:.2}");
        assert!(effect > 100, "far cascade casts no visible shadow");
        assert!(
            mean_on + 0.4 < mean_off,
            "shadowed frame is not darker: on={mean_on:.2} off={mean_off:.2}"
        );
        // No shimmer: a 5 mm camera nudge (below the far-cascade texel of
        // several centimeters) must leave the frame (nearly) untouched.
        let nudged_eye = eye + glam::Vec3::new(0.005, 0.0, 0.0);
        let nudged_view = glam::camera::rh::view::look_at_mat4(nudged_eye, target, glam::Vec3::Y);
        rig.backend.set_camera(
            &rig.queue,
            &(proj * nudged_view).to_cols_array_2d(),
            nudged_eye.to_array(),
        );
        let on_nudged = csm_probe_frame(&mut rig, ShadowCast::Enabled);
        let shimmer = pixel_diff(&on, &on_nudged);
        eprintln!("csm probe: shimmer={shimmer}");
        assert!(
            shimmer < 100,
            "sub-texel camera motion shimmers the CSM frame: {shimmer} pixels differ"
        );
    }
}
