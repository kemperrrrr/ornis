//! GTAO regressions the corner gate does not catch (#124).
//!
//! A flat floor and a pedestal compare the GPU factor to
//! [`gtao_sample`](ornis_render::gtao_sample) on horizons clamped with
//! [`gtao_clamp_horizon`](ornis_render::gtao_clamp_horizon). An overhang
//! past the effect radius must stay bright: forcing the horizon weight
//! to 1 closes it. `ORNIS_REQUIRE_GPU=1` panics when no adapter exists.

#[allow(dead_code)]
mod common;

use common::sized::{FrameSpec, render_and_readback};
use ornis_render::render_backend::RenderContext;
use ornis_render::{
    AoMethod, AoSettings, AoWiring, Bloom, InstanceData, MaterialIdx, OpenPBRMaterial,
    RenderFrame3D, Renderer3D, Technique,
};

/// Offscreen frame width (same extent as the corner gate).
const W: u32 = 320;
/// Offscreen frame height.
const H: u32 = 180;
/// Read-back format: composite output, so the factor is recovered through
/// the sRGB encode and the ACES fit.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// How far a recovered factor may sit from the CPU slice. Half-resolution
/// and the bilateral blur move a flat region by less than this; the sign
/// flip and the missing horizon clamp move it by more.
const MATCH_TOL: f32 = 0.12;

/// One instanced box: translation, then full side length of the unit mesh.
type BoxDesc = ([f32; 3], [f32; 3]);

/// Camera plus clip planes for one AO probe.
struct Shot {
    eye: [f32; 3],
    look: [f32; 3],
    fov_y: f32,
    near: f32,
    far: f32,
}

fn frame_spec() -> FrameSpec {
    FrameSpec {
        size: (W, H),
        format: FORMAT,
    }
}

fn view_proj(shot: &Shot) -> glam::Mat4 {
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::from_array(shot.eye),
        glam::Vec3::from_array(shot.look),
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        shot.fov_y.to_radians(),
        W as f32 / H as f32,
        shot.near,
        shot.far,
    );
    proj * view
}

/// Pixel of a world point. Callers assert the pixel is inside the frame.
fn project(shot: &Shot, world: [f32; 3]) -> (i32, i32) {
    let clip = view_proj(shot) * glam::Vec3::from_array(world).extend(1.0);
    let (x, y) = common::sized::clip_to_pixel(clip, W, H);
    (x.round() as i32, y.round() as i32)
}

/// Device and queue of one headless shot. One argument, so the draw
/// below stays inside the parameter budget.
struct Gpu<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
}

fn draw(
    gpu: &Gpu<'_>,
    renderer: &Renderer3D,
    mesh: &ornis_render::Mesh,
    instances: u32,
    ao: Option<AoWiring>,
) -> Vec<u8> {
    let spec = frame_spec();
    render_and_readback(gpu.device, gpu.queue, &spec, |encoder, view| {
        let mut plan = match ao {
            Some(wiring) => {
                RenderFrame3D::new_with_ao(FORMAT, (W, H), Technique::Deferred, Bloom::Off, wiring)
            }
            None => RenderFrame3D::new_with(FORMAT, (W, H), Technique::Deferred, Bloom::Off),
        };
        plan.render(
            RenderContext {
                device: gpu.device,
                queue: gpu.queue,
                encoder,
                target: view,
            },
            renderer,
            mesh,
            instances,
        );
    })
}

/// AO-free frame and the GTAO frame of `boxes` (translation, full side length).
fn render_pair(gpu: &Gpu<'_>, shot: &Shot, radius: f32, boxes: &[BoxDesc]) -> (Vec<u8>, Vec<u8>) {
    let renderer = Renderer3D::new(gpu.device, &frame_spec().surface_config(), 1);
    let mesh = ornis_render::create_box(gpu.device, [1.0, 1.0, 1.0]);
    let mut material = OpenPBRMaterial::dielectric();
    material.base.color_rgb([0.9, 0.9, 0.9]);
    material.specular.roughness(1.0);
    renderer.upload_materials(gpu.device, gpu.queue, &[material]);
    let instances: Vec<InstanceData> = boxes
        .iter()
        .map(|(translation, scale)| {
            let model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::from_array(*scale),
                glam::Quat::IDENTITY,
                glam::Vec3::from_array(*translation),
            );
            InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: MaterialIdx::from_raw(0),
            }
        })
        .collect();
    renderer.upload_instances(gpu.device, gpu.queue, &instances);
    renderer.set_camera(gpu.queue, &view_proj(shot).to_cols_array_2d(), shot.eye);
    renderer.set_lights(gpu.queue, [0.55, 0.55, 0.6], &[]);
    let count = u32::try_from(boxes.len()).expect("box count");
    let reference = draw(gpu, &renderer, &mesh, count, None);
    let wiring = AoWiring::new(
        AoSettings::new(AoMethod::Gtao, radius, 1.0, 0.02).expect("valid AO settings"),
    );
    let ao = draw(gpu, &renderer, &mesh, count, Some(wiring));
    (reference, ao)
}

fn channel_linear(byte: u8) -> f32 {
    let encoded = f32::from(byte) / 255.0;
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse of the Narkowicz ACES fit. The positive root is the pre-tonemap
/// channel; the composite writes that fit into an sRGB target.
fn aces_invert(y: f32) -> f32 {
    let aq = y * 2.43 - 2.51;
    let bq = y * 0.59 - 0.03;
    let cq = y * 0.14;
    let disc = (bq * bq - 4.0 * aq * cq).max(0.0);
    if aq.abs() <= 1.0e-6 {
        return 0.0;
    }
    ((-bq - disc.sqrt()) / (aq + aq)).max(0.0)
}

/// Mean AO factor in a window: sRGB-decode, invert ACES, divide by the
/// AO-free frame. `n > 0` so a probe past the frame cannot pass as 0.
fn mean_factor(reference: &[u8], pixels: &[u8], x: i32, y: i32) -> f32 {
    let mut acc = 0.0;
    let mut n = 0.0;
    for dy in -2..=2 {
        for dx in -2..=2 {
            acc += pixel_factor(reference, pixels, x + dx, y + dy, &mut n);
        }
    }
    assert!(n > 0.0, "AO factor window ({x}, {y}) is outside the frame");
    acc / n
}

fn pixel_factor(reference: &[u8], pixels: &[u8], x: i32, y: i32, n: &mut f32) -> f32 {
    if x < 0 || y < 0 || x >= W as i32 || y >= H as i32 {
        return 0.0;
    }
    let i = ((y as u32 * W + x as u32) * 4) as usize;
    let mut acc = 0.0;
    for channel in 0..3 {
        let denom = aces_invert(channel_linear(reference[i + channel]));
        if denom <= 0.02 {
            continue;
        }
        acc += aces_invert(channel_linear(pixels[i + channel])) / denom;
        *n += 1.0;
    }
    acc
}

/// Open hemisphere after the GPU horizon clamp, at a normal angle where a
/// flipped arc sign is a different number. `n = 0` hides that flip.
fn clamped_open_visibility() -> f32 {
    let n = std::f32::consts::FRAC_PI_6;
    let pi = std::f32::consts::PI;
    ornis_render::gtao_sample(
        ornis_render::gtao_clamp_horizon(n + pi, n),
        ornis_render::gtao_clamp_horizon(n - pi, n),
        n,
        1.0,
    )
}

/// Floor plus a thin pedestal, oblique camera. Both surfaces are open
/// hemispheres, so the GPU factor is the clamped XeGTAO slice. The normal
/// is off the view axis: a flipped arc sign is not the same number.
fn pedestal_shot() -> (Shot, Vec<BoxDesc>) {
    let shot = Shot {
        eye: [0.0, 0.7, 0.5],
        look: [0.0, 0.2, 0.0],
        fov_y: 50.0,
        near: 0.05,
        far: 30.0,
    };
    let boxes = vec![
        ([0.0, -0.05, 0.0], [8.0, 0.1, 8.0]),
        ([0.0, 0.17, 0.0], [0.14, 0.06, 0.14]),
    ];
    (shot, boxes)
}

#[test]
fn gtao_gpu_matches_the_clamped_cpu_slice() {
    let Some((_adapter, device, queue)) = common::adapter_or_skip() else {
        return;
    };
    let gpu = Gpu {
        device: &device,
        queue: &queue,
    };
    let (shot, boxes) = pedestal_shot();
    let (reference, ao) = render_pair(&gpu, &shot, 0.55, &boxes);
    let expected = clamped_open_visibility();
    let top = project(&shot, [0.0, 0.2, 0.0]);
    let floor = project(&shot, [0.0, 0.02, -1.1]);
    let top_ao = mean_factor(&reference, &ao, top.0, top.1);
    let floor_ao = mean_factor(&reference, &ao, floor.0, floor.1);
    eprintln!("gtao cpu {expected:.3} pedestal {top:?} {top_ao:.3} floor {floor:?} {floor_ao:.3}");
    assert!(
        (floor_ao - expected).abs() < MATCH_TOL,
        "open floor {floor_ao} != cpu slice {expected} at {floor:?}"
    );
    assert!(
        (top_ao - expected).abs() < MATCH_TOL,
        "pedestal {top_ao} != clamped cpu slice {expected} at {top:?}"
    );
}

/// A card facing the camera, and a second card closer to the camera off to
/// the side. From the probe the second card is about half a metre away, so
/// a 0.38 m radius must ignore it while a 1 m radius still darkens. The
/// second card is nearer than the probe, which keeps it inside the screen
/// march of the smaller radius.
fn cards_shot() -> (Shot, Vec<BoxDesc>) {
    let shot = Shot {
        eye: [0.0, 0.0, 3.0],
        look: [0.0, 0.0, 0.0],
        fov_y: 45.0,
        near: 0.05,
        far: 20.0,
    };
    // Card front at z = 1.52, right edge at x = 0.2. The second card is
    // closer to the camera (z ≈ 2.04) and starts at x = 0.30, in front of
    // the card plane. From (0.12, 0, 1.52) the nearest point is ~0.51 m:
    // past a 0.38 m radius, inside a 1 m radius.
    let boxes = vec![
        ([-0.15, 0.0, 1.50], [0.70, 1.2, 0.04]),
        ([0.55, 0.0, 2.04], [0.50, 1.5, 0.08]),
    ];
    (shot, boxes)
}

#[test]
fn gtao_falloff_ignores_an_occluder_past_the_radius() {
    let Some((_adapter, device, queue)) = common::adapter_or_skip() else {
        return;
    };
    let gpu = Gpu {
        device: &device,
        queue: &queue,
    };
    let (shot, boxes) = cards_shot();
    // Right side of the card. The nearer card is ~0.51 m away: outside
    // 0.38 m, inside 1 m, and inside the screen march of the smaller radius.
    let probe = project(&shot, [0.12, 0.0, 1.52]);
    let (reference, covered) = render_pair(&gpu, &shot, 1.0, &boxes);
    let covered_ao = mean_factor(&reference, &covered, probe.0, probe.1);
    let (_, past) = render_pair(&gpu, &shot, 0.38, &boxes);
    let past_ao = mean_factor(&reference, &past, probe.0, probe.1);
    eprintln!("gtao falloff probe {probe:?} radius 1 m {covered_ao:.3} radius 0.38 m {past_ao:.3}");
    assert!(
        covered_ao < 0.985,
        "occluder inside 1 m did not darken the card ({covered_ao} at {probe:?})"
    );
    assert!(
        past_ao > 0.97,
        "occluder past 0.38 m darkened the card ({past_ao} at {probe:?})"
    );
}
