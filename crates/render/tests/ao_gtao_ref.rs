//! GTAO regressions the corner gate does not catch (#124).
//!
//! A floor corner against two walls, with the floor ending inside the
//! first screen step, is compared to [`gtao_sample`](ornis_render::gtao_sample)
//! after [`gtao_clamp_horizon`](ornis_render::gtao_clamp_horizon). The walls
//! hold the factor in `0.5..0.8`; the drop is past the hemisphere, so a
//! march that starts horizons at -1 darkens it, and so does a flipped arc
//! sign. An occluder near the effect radius stays bright because of the
//! falloff. `ORNIS_REQUIRE_GPU=1` panics when no adapter exists.

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
/// Recovered factor vs the clamped CPU march. Half-resolution, the bilateral
/// blur and the 8-bit target sit inside this. Starting horizons at -1
/// moves the ledge by about 0.12; flipping the arc sign moves it by about
/// 0.30.
const LEDGE_TOL: f32 = 0.05;
/// Radius of the ledge march, in meters. The wall is inside it and the
/// drop off the floor is inside the first step.
const LEDGE_RADIUS: f32 = 0.50;

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
    let mut materials = vec![material];
    if boxes.len() > 1 {
        // A second albedo makes the walls readable in the readback. The
        // recovered factor divides it out.
        let mut wall = OpenPBRMaterial::dielectric();
        wall.base.color_rgb([0.85, 0.2, 0.2]);
        wall.specular.roughness(1.0);
        materials.push(wall);
    }
    renderer.upload_materials(gpu.device, gpu.queue, &materials);
    let instances: Vec<InstanceData> = boxes
        .iter()
        .enumerate()
        .map(|(index, (translation, scale))| {
            let model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::from_array(*scale),
                glam::Quat::IDENTITY,
                glam::Vec3::from_array(*translation),
            );
            InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: MaterialIdx::from_raw(
                    u32::try_from(index.min(materials.len() - 1)).unwrap(),
                ),
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

/// Mean channel ratio of one pixel.
fn factor_at(reference: &[u8], pixels: &[u8], x: i32, y: i32) -> f32 {
    let mut n = 0.0;
    let acc = pixel_factor(reference, pixels, x, y, &mut n);
    assert!(n > 0.0, "AO probe ({x}, {y}) is outside the frame");
    acc / n
}

/// Floor corner. The probe sits a couple of centimeters in front of both
/// walls, and the floor ends inside the first step, so one horizon is past
/// the hemisphere while the walls keep the factor in `0.5..0.8`.
fn ledge_shot() -> (Shot, Vec<BoxDesc>) {
    let shot = Shot {
        eye: [0.15, 0.75, 0.85],
        look: [-0.45, 0.5, 0.0],
        fov_y: 50.0,
        near: 0.05,
        far: 40.0,
    };
    // Floor ends at x = -0.33, so a probe near the wall is inside the
    // first screen step of the drop. One wall faces the probe at x = -0.40;
    // the other faces it at z = -0.08, so both slice directions occlude.
    let boxes = vec![
        ([-0.765, 0.45, 0.0], [0.87, 0.1, 2.0]),
        ([-0.55, 0.9, 0.0], [0.3, 1.0, 2.0]),
        ([-0.45, 0.9, -0.23], [0.5, 1.0, 0.3]),
    ];
    (shot, boxes)
}

#[test]
fn gtao_ledge_matches_the_clamped_cpu_horizon() {
    let Some((_adapter, device, queue)) = common::adapter_or_skip() else {
        return;
    };
    let gpu = Gpu {
        device: &device,
        queue: &queue,
    };
    let (shot, boxes) = ledge_shot();
    let aabbs: Vec<common::gtao_cpu::Aabb> = boxes
        .iter()
        .map(|(translation, scale)| common::gtao_cpu::aabb(*translation, *scale))
        .collect();
    let probe = [-0.38_f32, 0.5, 0.0];
    let (reference, ao) = render_pair(&gpu, &shot, LEDGE_RADIUS, &boxes);
    let pixel = project(&shot, probe);
    let gpu_ao = factor_at(&reference, &ao, pixel.0, pixel.1);
    let uv = glam::Vec2::new(
        (pixel.0 as f32 + 0.5) / W as f32,
        (pixel.1 as f32 + 0.5) / H as f32,
    );
    let expected =
        common::gtao_cpu::expected_ao(shot.eye, view_proj(&shot), &aabbs, uv, LEDGE_RADIUS);
    eprintln!("gtao ledge pixel {pixel:?} gpu {gpu_ao:.3} cpu {expected:.3}");
    assert!(
        (0.50..0.80).contains(&expected) && (0.50..0.80).contains(&gpu_ao),
        "ledge factor gpu {gpu_ao} cpu {expected} is not in 0.50..0.80"
    );
    assert!(
        (gpu_ao - expected).abs() < LEDGE_TOL,
        "ledge gpu {gpu_ao} != clamped cpu {expected} at {pixel:?}"
    );
}

/// A card facing the camera, and a second card closer to the camera off to
/// the side. From the probe the second card is under 0.3 m away: inside a
/// 1 m radius, where it darkens the card, and inside a 0.50 m radius, where
/// the falloff leaves the card bright. The second card is nearer than the
/// probe, which keeps it inside the screen march.
fn cards_shot() -> (Shot, Vec<BoxDesc>) {
    let shot = Shot {
        eye: [0.0, 0.0, 3.0],
        look: [0.0, 0.0, 0.0],
        fov_y: 45.0,
        near: 0.05,
        far: 20.0,
    };
    // Card front at z = 1.52, right edge at x = 0.2. The second card is
    // closer to the camera (near face z = 1.80) and starts at x = 0.17.
    // From (0.12, 0, 1.52) the nearest point is ~0.28 m: a 0.50 m radius
    // reaches it with a small falloff, a 1 m radius still darkens.
    let boxes = vec![
        ([-0.15, 0.0, 1.50], [0.70, 1.2, 0.04]),
        ([0.43, 0.0, 1.85], [0.52, 1.6, 0.10]),
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
    // Right side of the card. The nearer card is ~0.28 m away: a weak
    // contribution at 0.50 m, a clear one at 1 m.
    let probe = project(&shot, [0.12, 0.0, 1.52]);
    let (reference, covered) = render_pair(&gpu, &shot, 1.0, &boxes);
    let covered_ao = mean_factor(&reference, &covered, probe.0, probe.1);
    let (_, past) = render_pair(&gpu, &shot, 0.50, &boxes);
    let past_ao = mean_factor(&reference, &past, probe.0, probe.1);
    eprintln!("gtao falloff probe {probe:?} radius 1 m {covered_ao:.3} radius 0.50 m {past_ao:.3}");
    assert!(
        covered_ao < 0.90,
        "occluder inside 1 m did not darken the card enough ({covered_ao} at {probe:?})"
    );
    assert!(
        past_ao > 0.97,
        "occluder near the 0.50 m radius darkened the card ({past_ao} at {probe:?})"
    );
    assert!(
        past_ao - covered_ao > 0.08,
        "falloff gap {} is inside the overlap (1 m {covered_ao}, 0.50 m {past_ao})",
        past_ao - covered_ao
    );
}
