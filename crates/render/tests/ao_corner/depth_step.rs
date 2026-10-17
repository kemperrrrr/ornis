//! Depth-step blur probe for the AO corner gate: a red fin at ~1 m in
//! front of a blue wall at ~5 m. The crease is the silhouette one
//! bilateral tap can reach; the two albedos name the sides.

use super::common;
use super::{
    AoMethod, AoSettings, AoWiring, H, InstanceData, MaterialIdx, OpenPBRMaterial, Renderer3D, TOL,
    W, frame_spec, render_count,
};

/// Red fin at ~1 m in front of a blue wall at ~5 m. The crease is the
/// silhouette the bilateral tap can reach; the two albedos name the sides.
fn build_depth_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (Renderer3D, ornis_render::Mesh) {
    let renderer = Renderer3D::new(device, &frame_spec().surface_config(), 1);
    let mesh = ornis_render::create_box(device, [1.0, 1.0, 1.0]);
    let mut near_mat = OpenPBRMaterial::dielectric();
    near_mat.base.color_rgb([0.9, 0.25, 0.25]);
    near_mat.specular.roughness(1.0);
    let mut far_mat = OpenPBRMaterial::dielectric();
    far_mat.base.color_rgb([0.25, 0.25, 0.9]);
    far_mat.specular.roughness(1.0);
    renderer.upload_materials(device, queue, &[near_mat, far_mat]);
    // A facing wall plus a thin fin, ~1 m away, with the blue wall ~5 m
    // behind the fin. The crease sits on the silhouette the blur can reach.
    let boxes = [
        ([0.0, -0.02, 0.0], [2.0, 0.04, 2.0], 0u32),
        ([-0.15, 0.28, 0.0], [0.55, 0.56, 0.06], 0),
        ([0.38, 0.28, 0.18], [0.03, 0.56, 0.4], 0),
        ([0.0, 1.0, -4.0], [12.0, 5.0, 0.1], 1),
    ];
    let instances: Vec<InstanceData> = boxes
        .iter()
        .map(|(t, s, material)| {
            let model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::from_array(*s),
                glam::Quat::IDENTITY,
                glam::Vec3::from_array(*t),
            );
            InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: MaterialIdx::from_raw(*material),
            }
        })
        .collect();
    renderer.upload_instances(device, queue, &instances);
    let eye = glam::Vec3::new(0.15, 0.4, 1.15);
    let view =
        glam::camera::rh::view::look_at_mat4(eye, glam::Vec3::new(0.05, 0.28, 0.0), glam::Vec3::Y);
    let proj = glam::camera::rh::proj::directx::perspective(
        50.0f32.to_radians(),
        W as f32 / H as f32,
        0.1,
        30.0,
    );
    renderer.set_camera(
        queue,
        &(proj * view).to_cols_array_2d(),
        [eye.x, eye.y, eye.z],
    );
    renderer.set_lights(queue, [0.55, 0.55, 0.6], &[]);
    (renderer, mesh)
}

/// Full-res pixels covered by one bilateral tap: 1.5 half-res texels times
/// the AO divisor. A weights=1 blur mixes exactly this far.
const BLUR_TAP_PX: i32 = 3;
/// Sum-of-RGB drop on the red side. One weights=1 tap out of five samples
/// would move the blue side by at least this divided by five.
const STEP_NEAR_DROP: i32 = 36;
/// Sum-of-RGB drop still accepted on the blue side of the tap.
const STEP_FAR_DROP: i32 = 6;
/// Silhouette pairs a depth-aware blur keeps. Weight 1 leaves at most a
/// couple on lavapipe (the fin bleeds onto the backdrop, or the near drop
/// itself falls under [`STEP_NEAR_DROP`]); the bilateral kernel keeps
/// more than thirty. Eight was inside the noise of a box blur.
const STEP_PAIR_MIN: usize = 16;

#[test]
fn ao_blur_does_not_cross_a_depth_step() {
    let Some((_adapter, device, queue)) = common::adapter_or_skip() else {
        return;
    };
    let (renderer, mesh) = build_depth_step(&device, &queue);
    let reference = render_count(&device, &queue, &renderer, &mesh, 4, None);
    for method in [AoMethod::Ssao, AoMethod::Gtao] {
        let wiring = AoWiring::new(AoSettings::new(method, 0.6, 1.0, 0.02).expect("valid"));
        let pixels = render_count(&device, &queue, &renderer, &mesh, 4, Some(wiring));
        let pairs = depth_step_pairs(&reference, &pixels);
        eprintln!("{method:?} depth-step pairs {pairs}");
        assert!(
            pairs >= STEP_PAIR_MIN,
            "{method:?} kept only {pairs} bilateral pairs (need {STEP_PAIR_MIN})"
        );
    }
}

/// Red/blue pairs one bilateral tap apart: the red pixel is occluded and the
/// blue pixel is not. Returns how many such pairs the frame contains.
fn depth_step_pairs(reference: &[u8], pixels: &[u8]) -> usize {
    let mut pairs = 0usize;
    for y in 0..H as i32 {
        pairs += depth_step_row(reference, pixels, y);
    }
    pairs
}

fn depth_step_row(reference: &[u8], pixels: &[u8], y: i32) -> usize {
    let mut pairs = 0usize;
    for x in 0..W as i32 {
        pairs += depth_step_at(reference, pixels, x, y);
    }
    pairs
}

fn depth_step_at(reference: &[u8], pixels: &[u8], x: i32, y: i32) -> usize {
    if !is_red(reference, x, y) {
        return 0;
    }
    let near = rgb_sum(reference, x, y) - rgb_sum(pixels, x, y);
    if near < STEP_NEAR_DROP {
        return 0;
    }
    blur_taps_onto_blue(reference, pixels, x, y, near)
}

/// Cardinal taps at [`BLUR_TAP_PX`]. A weights=1 blur (center plus four
/// taps, one of them this red pixel) would push `near / 5` onto the blue
/// side; that has to clear [`STEP_FAR_DROP`] while the real tap does not.
fn blur_taps_onto_blue(reference: &[u8], pixels: &[u8], x: i32, y: i32, near: i32) -> usize {
    let mut pairs = 0usize;
    for (dx, dy) in [
        (BLUR_TAP_PX, 0),
        (-BLUR_TAP_PX, 0),
        (0, BLUR_TAP_PX),
        (0, -BLUR_TAP_PX),
    ] {
        if tap_stays_on_blue(reference, pixels, x + dx, y + dy, near) {
            pairs += 1;
        }
    }
    pairs
}

fn tap_stays_on_blue(reference: &[u8], pixels: &[u8], x: i32, y: i32, near: i32) -> bool {
    if x < 0 || y < 0 || x >= W as i32 || y >= H as i32 || !is_blue(reference, x, y) {
        return false;
    }
    let far = rgb_sum(reference, x, y) - rgb_sum(pixels, x, y);
    far <= STEP_FAR_DROP && near / 5 > STEP_FAR_DROP
}

fn rgb_sum(pixels: &[u8], x: i32, y: i32) -> i32 {
    let i = ((y as u32 * W + x as u32) * 4) as usize;
    i32::from(pixels[i]) + i32::from(pixels[i + 1]) + i32::from(pixels[i + 2])
}

/// Near fin: red albedo, clearly above the blue backdrop.
fn is_red(pixels: &[u8], x: i32, y: i32) -> bool {
    let i = ((y as u32 * W + x as u32) * 4) as usize;
    i32::from(pixels[i]) > i32::from(pixels[i + 2]) + i32::from(TOL) * 5
}

/// Backdrop: blue albedo, clearly above the red fin.
fn is_blue(pixels: &[u8], x: i32, y: i32) -> bool {
    let i = ((y as u32 * W + x as u32) * 4) as usize;
    i32::from(pixels[i + 2]) > i32::from(pixels[i]) + i32::from(TOL) * 5
}
