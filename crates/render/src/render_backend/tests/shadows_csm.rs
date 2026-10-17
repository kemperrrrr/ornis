//! CSM directional probes: the near scene stays in parity with the legacy
//! shadow box, and a far receiver stays dark without shimmering when the
//! camera moves by less than a texel.

use super::super::*;
use super::probe::pixel_diff;
use super::*;
use crate::flags::ShadowCast;
use crate::mesh::Mesh;

const PROBE_W: u32 = 320;
const PROBE_H: u32 = 180;
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
        let (target, view, surface_config) =
            super::probe::probe_color_target(&device, label, PROBE_W, PROBE_H);
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
        super::probe::read_color_target(
            &rig.device,
            &rig.queue,
            encoder,
            &rig.target,
            (PROBE_W, PROBE_H),
        )
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

/// Practical-split boundaries, same blend the shader uploads.
fn practical_splits(near: f32, far: f32) -> [f32; 5] {
    let count = crate::renderer::CSM_CASCADES;
    let lambda = crate::renderer::CSM_SPLIT_LAMBDA;
    let mut splits = [far; 5];
    splits[0] = near;
    for (i, slot) in splits.iter_mut().enumerate().take(count).skip(1) {
        let t = i as f32 / count as f32;
        let uniform = near + (far - near) * t;
        let log = near * (far / near).powf(t);
        *slot = lambda * log + (1.0 - lambda) * uniform;
    }
    splits
}

/// Center and radius of a sphere that stays inside cascade `index`,
/// below the blend band at its far edge.
fn interior_sphere(splits: &[f32], index: usize) -> (f32, f32) {
    let lower = splits[index];
    let upper = splits[index + 1];
    let band = (upper - lower) * crate::shaders::helpers::CSM_BLEND_FRACTION;
    let span = upper - lower - band;
    let radius = (span * 0.2).clamp(1.0, 12.0);
    (lower + span * 0.5, radius)
}

fn bright_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|c| {
            let luma = (u16::from(c[0]) + u16::from(c[1]) + u16::from(c[2])) / 3;
            luma > 180
        })
        .count()
}

struct CascadeProbe<'a> {
    renderer: &'a crate::renderer::Renderer3D,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    mesh: &'a Mesh,
    texture: &'a wgpu::Texture,
    view: &'a wgpu::TextureView,
}

fn marked_cascade_frame(probe: &CascadeProbe<'_>, depths: [f32; 4]) -> Vec<u8> {
    let mut encoder = probe
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csm cascade select"),
        });
    probe.renderer.render_deferred_with_csm_depths(
        RenderContext {
            device: probe.device,
            queue: probe.queue,
            encoder: &mut encoder,
            target: probe.view,
        },
        probe.mesh,
        1,
        depths,
    );
    super::probe::read_color_target(
        probe.device,
        probe.queue,
        encoder,
        probe.texture,
        (PROBE_W, PROBE_H),
    )
}

/// Each cascade is a constant depth marker. A receiver between `split[i]`
/// and `split[i + 1]`, outside the blend band, must sample layer `i`.
/// Sampling the next cascade (`+ 1`, clamped on the last) leaves that
/// receiver lit for `i` in 0..2.
#[test]
fn csm_receiver_samples_the_cascade_of_its_view_depth() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping cascade select probe");
        return;
    };
    let (texture, view, surface_config) =
        super::probe::probe_color_target(&device, "csm cascade select", PROBE_W, PROBE_H);
    let renderer = crate::renderer::Renderer3D::new(&device, &surface_config, 1);
    let mesh = crate::mesh::create_sphere(&device, 1.0, 24, 16);
    let mut gray = ornis_core::OpenPBRMaterial::dielectric();
    gray.base.color_rgb([0.8, 0.8, 0.8]);
    gray.specular.roughness(0.5);
    renderer.upload_materials(&device, &queue, &[gray]);
    renderer.set_shading_debug(crate::renderer::ShadingDebug::Shadow);

    const NEAR: f32 = 0.1;
    const FAR: f32 = 200.0;
    let eye = glam::Vec3::new(0.0, 1.0, 0.0);
    let forward = glam::Vec3::new(0.0, 0.0, -1.0);
    let splits = practical_splits(NEAR, FAR);
    let look = glam::camera::rh::view::look_at_mat4(eye, eye + forward, glam::Vec3::Y);
    let proj = glam::camera::rh::proj::directx::perspective(
        35.0f32.to_radians(),
        PROBE_W as f32 / PROBE_H as f32,
        NEAR,
        FAR,
    );
    renderer.set_camera(&queue, &(proj * look).to_cols_array_2d(), eye.to_array());
    renderer.set_csm_camera(NEAR, FAR, forward.to_array());
    let to_light = ornis_core::units::UnitVec3::normalize(glam::Vec3::Y).expect("up light");
    renderer.set_lights(
        &queue,
        [0.05, 0.05, 0.08],
        &[ornis_assets::scene::LightDesc::Directional {
            direction: to_light,
            intensity: 1.2,
            color: [1.0, 1.0, 1.0],
            shadow: ornis_assets::scene::ShadowCast::from(true),
        }],
    );

    for cascade in 0..crate::renderer::CSM_CASCADES {
        let (center, radius) = interior_sphere(&splits, cascade);
        let blend = crate::shaders::helpers::CSM_BLEND_FRACTION;
        let upper = splits[cascade + 1];
        let lower = splits[cascade];
        let band_start = upper - (upper - lower) * blend;
        assert!(
            center - radius > lower && center < band_start,
            "cascade {cascade} sphere crosses a split or the blend band"
        );
        let pos = eye + forward * center;
        renderer.upload_instances(&device, &queue, &[csm_instance(pos.to_array(), radius)]);
        let probe = CascadeProbe {
            renderer: &renderer,
            device: &device,
            queue: &queue,
            mesh: &mesh,
            texture: &texture,
            view: &view,
        };
        let lit = marked_cascade_frame(&probe, [1.0; 4]);
        let mut depths = [1.0; 4];
        depths[cascade] = 0.0;
        let marked = marked_cascade_frame(&probe, depths);
        let lit_n = bright_pixels(&lit);
        let marked_n = bright_pixels(&marked);
        eprintln!(
            "cascade {cascade}: lit={lit_n} marked={marked_n} depth={center:.2} r={radius:.2}"
        );
        assert!(
            lit_n > 40,
            "cascade {cascade} receiver is not visible ({lit_n} bright pixels)"
        );
        assert!(
            marked_n * 4 < lit_n,
            "cascade {cascade} did not sample layer {cascade} ({marked_n} bright vs {lit_n} lit)"
        );
    }
}
