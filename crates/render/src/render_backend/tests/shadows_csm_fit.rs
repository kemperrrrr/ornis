//! GPU gates for the CSM review (#121): a camera yaw moves the shadow
//! map by whole texels, a caster behind the camera still darkens a
//! visible receiver, and a receiver past the last split is shadowed by
//! an occluder that sits in an earlier slice.

use super::super::*;
use super::probe::{pixel_diff, yawed_eye};
use super::*;
use crate::flags::ShadowCast;
use crate::mesh::Mesh;
use crate::renderer::{InstanceData, MaterialIdx, Renderer3D, SHADOW_SIZE};

const PROBE_W: u32 = 320;
const PROBE_H: u32 = 180;
/// Depth below this belongs to a rendered caster, not the cleared far plane.
const OCCUPIED_DEPTH: f32 = 0.98;

struct FitRig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer3D,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    mesh: Mesh,
    instances: u32,
}

fn open_fit(label: &'static str) -> Option<FitRig> {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping {label}");
        return None;
    };
    let (target, view, surface_config) =
        super::probe::probe_color_target(&device, label, PROBE_W, PROBE_H);
    let renderer = Renderer3D::new(&device, &surface_config, 1);
    let mesh = crate::mesh::create_sphere(&device, 1.0, 24, 16);
    let mut gray = ornis_core::OpenPBRMaterial::dielectric();
    gray.base.color_rgb([0.8, 0.8, 0.8]);
    gray.specular.roughness(0.5);
    renderer.upload_materials(&device, &queue, &[gray]);
    Some(FitRig {
        device,
        queue,
        renderer,
        target,
        view,
        mesh,
        instances: 0,
    })
}

fn sphere(translation: [f32; 3], scale: f32) -> InstanceData {
    let model = glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::splat(scale),
        glam::Quat::IDENTITY,
        glam::Vec3::from(translation),
    );
    InstanceData {
        model_matrix: model,
        normal_matrix: model.inverse().transpose(),
        material_index: MaterialIdx::from_raw(0),
    }
}

fn upload(rig: &mut FitRig, instances: &[InstanceData]) {
    rig.instances = instances.len() as u32;
    rig.renderer
        .upload_instances(&rig.device, &rig.queue, instances);
}

fn aim(rig: &FitRig, eye: glam::Vec3, target: glam::Vec3, near: f32, far: f32, fov_deg: f32) {
    let fwd = (target - eye).normalize();
    let view = glam::camera::rh::view::look_at_mat4(eye, target, glam::Vec3::Y);
    let proj = glam::camera::rh::proj::directx::perspective(
        fov_deg.to_radians(),
        PROBE_W as f32 / PROBE_H as f32,
        near,
        far,
    );
    rig.renderer.set_camera(
        &rig.queue,
        &(proj * view).to_cols_array_2d(),
        eye.to_array(),
    );
    rig.renderer.set_csm_camera(near, far, fwd.to_array());
}

fn paint(rig: &FitRig, to_light: glam::Vec3, shadow: ShadowCast) -> Vec<u8> {
    rig.renderer.set_lights(
        &rig.queue,
        [0.05, 0.05, 0.08],
        &[ornis_assets::scene::LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(to_light).expect("non-zero"),
            intensity: 1.2,
            color: [1.0, 1.0, 1.0],
            shadow: ornis_assets::scene::ShadowCast::from(shadow.is_enabled()),
        }],
    );
    let mut encoder = rig
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csm fit encoder"),
        });
    rig.renderer.render_scene(
        &rig.device,
        &rig.queue,
        &mut encoder,
        &rig.view,
        &rig.mesh,
        rig.instances,
    );
    super::probe::read_color_target(
        &rig.device,
        &rig.queue,
        encoder,
        &rig.target,
        (PROBE_W, PROBE_H),
    )
}

fn shadow_effect(rig: &FitRig, to_light: glam::Vec3) -> usize {
    let on = paint(rig, to_light, ShadowCast::Enabled);
    let off = paint(rig, to_light, ShadowCast::Disabled);
    pixel_diff(&on, &off)
}

/// Centroid of casters in one cascade, in texel units.
fn occupied_centroid(depth: &[f32]) -> Option<(f64, f64, u64)> {
    let size = SHADOW_SIZE as usize;
    let mut count = 0u64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for (i, texel) in depth.iter().enumerate() {
        if *texel < OCCUPIED_DEPTH {
            sx += (i % size) as f64;
            sy += (i / size) as f64;
            count += 1;
        }
    }
    (count > 0).then_some((sx / count as f64, sy / count as f64, count))
}

fn dist_to_integer(delta: f64) -> f64 {
    (delta - delta.round()).abs()
}

/// A camera yaw slides the cascade by whole texels. Anchoring the light
/// view on the moving slice (or skipping the snap) leaves a fractional
/// centroid, and this fails.
#[test]
fn csm_yaw_moves_the_shadow_map_by_whole_texels() {
    let Some(mut rig) = open_fit("csm yaw") else {
        return;
    };
    let eye = glam::Vec3::new(0.0, 10.0, 20.0);
    let target = glam::Vec3::new(0.0, 2.0, -60.0);
    let to_light = glam::Vec3::new(0.42, 0.84, 0.3).normalize();
    upload(&mut rig, &[sphere([0.0, 2.0, 8.0], 2.5)]);
    aim(&rig, eye, target, 0.1, 200.0, 55.0);
    paint(&rig, to_light, ShadowCast::Enabled);
    let base = rig
        .renderer
        .read_csm_layer_for_tests(&rig.device, &rig.queue, 0);
    let (bx, by, bn) = occupied_centroid(&base).expect("cascade 0 holds the caster");
    assert!(
        bn > 400,
        "cascade 0 occupancy {bn} is too small to judge the snap"
    );
    for degrees in [1.0_f32, 5.0] {
        let yawed = yawed_eye(eye, target, degrees.to_radians());
        aim(&rig, yawed, target, 0.1, 200.0, 55.0);
        paint(&rig, to_light, ShadowCast::Enabled);
        let layer = rig
            .renderer
            .read_csm_layer_for_tests(&rig.device, &rig.queue, 0);
        let (x, y, n) = occupied_centroid(&layer).expect("yawed cascade 0 holds the caster");
        let (dx, dy) = (x - bx, y - by);
        eprintln!("csm yaw {degrees}°: centroid delta ({dx:.3}, {dy:.3}) occupancy {bn}->{n}");
        assert!(
            dist_to_integer(dx) < 0.05 && dist_to_integer(dy) < 0.05,
            "yaw {degrees}° left a fractional texel drift ({dx}, {dy})"
        );
        assert!(
            dx.hypot(dy) > 0.5,
            "yaw {degrees}° did not move the map ({dx}, {dy})"
        );
    }
}

/// An occluder behind the camera darkens a visible receiver only when
/// its AABB widens the cascade depth. Slice-only depth clips it.
#[test]
fn csm_occluder_behind_the_camera_still_shadows() {
    let Some(mut rig) = open_fit("csm behind camera") else {
        return;
    };
    let eye = glam::Vec3::new(0.0, 4.0, 14.0);
    let target = glam::Vec3::ZERO;
    let fwd = (target - eye).normalize();
    let occluder = glam::Vec3::new(0.0, 80.0, 0.0);
    let depth = (occluder - eye).dot(fwd);
    assert!(
        depth < -1.0,
        "premise: the occluder is behind the camera, view depth {depth}"
    );
    upload(
        &mut rig,
        &[
            sphere([0.0, 0.0, 0.0], 3.0),
            sphere(occluder.to_array(), 5.0),
        ],
    );
    aim(&rig, eye, target, 0.1, 80.0, 55.0);
    let to_light = glam::Vec3::Y;
    rig.renderer.clear_csm_casters();
    let bare = shadow_effect(&rig, to_light);
    rig.renderer
        .set_csm_casters([-6.0, -4.0, -6.0], [6.0, 86.0, 6.0]);
    let covered = shadow_effect(&rig, to_light);
    eprintln!("csm behind camera: bare={bare} covered={covered} depth={depth:.2}");
    assert!(
        bare < 40,
        "without the caster AABB the occluder already shadows ({bare}); it is inside the slice"
    );
    assert!(
        covered > 100,
        "caster AABB did not bring the occluder into the cascade ({covered})"
    );
}

/// A receiver past the last split is darkened by an occluder that lives
/// in an earlier slice. Slice-only depth clips that occluder out of the
/// last cascade; skipping cascade 3 samples a map that does not cover
/// this receiver's light-space texel.
#[test]
fn csm_last_cascade_shadows_a_receiver_past_the_split() {
    let Some(mut rig) = open_fit("csm last cascade") else {
        return;
    };
    let eye = glam::Vec3::new(0.0, 12.0, 25.0);
    let target = glam::Vec3::new(0.0, 0.0, -100.0);
    let fwd = (target - eye).normalize();
    let right = fwd.cross(glam::Vec3::Y).normalize();
    // Far enough off the view axis that cascade 2's map does not cover
    // this texel. Sampling cascade 2 (the "never use cascade 3" mutation)
    // clamps to an empty border and leaves the receiver lit.
    let lateral = 112.0;
    let receiver = eye + fwd * 130.0 + right * lateral;
    let occluder = eye + fwd * 18.0 + right * lateral;
    assert!(
        (receiver - eye).dot(fwd) > 100.0,
        "premise: receiver is past the last practical split"
    );
    assert!(
        (occluder - eye).dot(fwd) < 25.0,
        "premise: occluder sits in the first slice"
    );
    let to_light = (occluder - receiver).normalize();
    upload(
        &mut rig,
        &[
            sphere(receiver.to_array(), 8.0),
            sphere(occluder.to_array(), 4.0),
        ],
    );
    aim(&rig, eye, target, 0.1, 200.0, 55.0);
    let effect = shadow_effect(&rig, to_light);
    eprintln!("csm last cascade: effect={effect}");
    assert!(
        effect > 40,
        "receiver past the last split is not shadowed by the earlier occluder ({effect})"
    );
}

/// Pixels whose shadow-debug luma is neither fully lit nor fully dark.
fn partial_shadow_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|c| {
            let luma = (u16::from(c[0]) + u16::from(c[1]) + u16::from(c[2])) / 3;
            (30..220).contains(&luma)
        })
        .count()
}

/// The cascade seam blends. A receiver that crosses a split, with a
/// shadow edge on that seam, has a band of partial visibility. Removing
/// `mix(vis1, vis, t)` collapses the band to the PCF step.
#[test]
fn csm_blend_softens_the_cascade_seam() {
    let Some(mut rig) = open_fit("csm blend") else {
        return;
    };
    rig.renderer
        .set_shading_debug(crate::renderer::ShadingDebug::Shadow);
    // The cap faces the camera at view depth ~9.9, inside the top 8% of
    // cascade 0 (split ≈ 10.3 for near 0.1 / far 80). Measured on lavapipe:
    // ~1629 partial pixels with the mix, ~1302 when it is removed.
    upload(
        &mut rig,
        &[sphere([0.0, 0.0, 0.0], 4.0), sphere([1.2, 6.0, 0.4], 1.0)],
    );
    aim(
        &rig,
        glam::Vec3::new(0.0, 1.5, 13.8),
        glam::Vec3::ZERO,
        0.1,
        80.0,
        25.0,
    );
    let px = paint(&rig, glam::Vec3::Y, ShadowCast::Enabled);
    let partial = partial_shadow_pixels(&px);
    eprintln!("csm blend partial={partial}");
    assert!(
        partial > 1500,
        "cascade seam is not blended ({partial} partial pixels; no-mix stays near 1300)"
    );
}
