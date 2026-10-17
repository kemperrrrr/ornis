//! AO corner gate (#124): a three-plane corner (floor + two walls) under
//! ambient-only light, rendered through the plan path with AO off and with
//! each kernel on.
//!
//! Acceptance shape from the issue: the disabled mode is bit-identical to
//! the AO-free frame, and the corner visibly darkens while no pixel ever
//! brightens (the factor only modulates ambient/IBL, never direct light).
//! Skip contract: returns early with a note when no adapter exists (same
//! as the other headless gates; CI provides lavapipe).
//!
//! Adapter acquisition and the render-and-read-back flow live in the shared
//! `common` harness (no per-gate copies).

#[allow(dead_code)]
mod common;

use common::sized::{FrameSpec, render_and_readback};
use ornis_render::render_backend::RenderContext;
use ornis_render::{
    AoMethod, AoSettings, AoWiring, Bloom, InstanceData, MaterialIdx, OpenPBRMaterial,
    RenderFrame3D, Renderer3D, Technique,
};

/// Offscreen frame extent (MSAA-gate precedent).
const W: u32 = 320;
/// Offscreen frame height.
const H: u32 = 180;
/// Read-back format: matches the composite output.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Per-channel drift tolerated before a pixel counts as changed (same
/// window as the golden probe: driver rounding, not engine regressions).
const TOL: u8 = 4;
/// Channel delta that counts as real occlusion (clearly above noise).
const OCCLUSION_DELTA: u8 = 6;
/// Minimum fraction of darkened pixels: the corner junctions must cover
/// this share of the frame at radius 0.75 m.
const MIN_DARK_FRACTION: f64 = 0.005;
/// Minimum fraction of non-black pixels: the corner must be visible.
const MIN_LIT_FRACTION: f64 = 0.2;

/// Offscreen frame spec (shared harness builds targets from it).
fn frame_spec() -> FrameSpec {
    FrameSpec {
        size: (W, H),
        format: FORMAT,
    }
}

/// One renderer over the corner scene: unit-box mesh instanced into a
/// floor plus two walls meeting at (-2, 0, -2), rough white dielectric,
/// ambient-only light (no direct term for AO to touch by mistake).
fn build_corner(device: &wgpu::Device, queue: &wgpu::Queue) -> (Renderer3D, ornis_render::Mesh) {
    let renderer = Renderer3D::new(device, &frame_spec().surface_config(), 1);
    let mesh = ornis_render::create_box(device, [1.0, 1.0, 1.0]);
    let mut material = OpenPBRMaterial::dielectric();
    material.base.color_rgb([0.9, 0.9, 0.9]);
    material.specular.roughness(1.0);
    renderer.upload_materials(device, queue, &[material]);
    // (translation, scale) per box: floor, wall x=-2, wall z=-2.
    let boxes = [
        ([0.0, -0.1, 0.0], [10.0, 0.2, 10.0]),
        ([-2.1, 3.0, 0.0], [0.2, 6.0, 10.0]),
        ([0.0, 3.0, -2.1], [10.0, 6.0, 0.2]),
    ];
    let instances: Vec<InstanceData> = boxes
        .iter()
        .map(|(t, s)| {
            let model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::from_array(*s),
                glam::Quat::IDENTITY,
                glam::Vec3::from_array(*t),
            );
            InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: MaterialIdx::from_raw(0),
            }
        })
        .collect();
    renderer.upload_instances(device, queue, &instances);
    let eye = glam::Vec3::new(3.0, 2.2, 3.0);
    let center = glam::Vec3::new(-0.8, 0.8, -0.8);
    let view = glam::camera::rh::view::look_at_mat4(eye, center, glam::Vec3::Y);
    let proj = glam::camera::rh::proj::directx::perspective(
        55.0f32.to_radians(),
        W as f32 / H as f32,
        0.1,
        100.0,
    );
    renderer.set_camera(
        queue,
        &(proj * view).to_cols_array_2d(),
        [eye.x, eye.y, eye.z],
    );
    renderer.set_lights(queue, [0.55, 0.55, 0.6], &[]);
    (renderer, mesh)
}

/// Renders one plan-path frame (Deferred, no bloom) with the given AO
/// wiring (`None` = AO-free) into a fresh target and reads it back.
fn render_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer3D,
    mesh: &ornis_render::Mesh,
    ao: Option<AoWiring>,
) -> Vec<u8> {
    render_and_readback(device, queue, &frame_spec(), |encoder, view| {
        let mut plan = match ao {
            Some(wiring) => {
                RenderFrame3D::new_with_ao(FORMAT, (W, H), Technique::Deferred, Bloom::Off, wiring)
            }
            None => RenderFrame3D::new_with(FORMAT, (W, H), Technique::Deferred, Bloom::Off),
        };
        plan.render(
            RenderContext {
                device,
                queue,
                encoder,
                target: view,
            },
            renderer,
            mesh,
            3,
        );
    })
}

fn wiring(method: AoMethod, intensity: f32) -> AoWiring {
    AoWiring::new(AoSettings::new(method, 0.75, intensity, 0.025).expect("valid AO settings"))
}

/// Adapter, corner scene, and the AO-free reference frame. `None` when no
/// adapter exists (the gate skips, same contract as the other headless tests).
fn prepared_corner() -> Option<(
    wgpu::Device,
    wgpu::Queue,
    Renderer3D,
    ornis_render::Mesh,
    Vec<u8>,
)> {
    let Some((_adapter, device, queue)) = pollster::block_on(common::request_device()) else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return None;
    };
    let (renderer, mesh) = build_corner(&device, &queue);
    let reference = render_pixels(&device, &queue, &renderer, &mesh, None);
    Some((device, queue, renderer, mesh, reference))
}

/// Fraction of texels whose RGB sum clears the near-black cutoff.
fn lit_fraction(pixels: &[u8]) -> f64 {
    let lit = pixels
        .chunks_exact(4)
        .filter(|c| c[0] as u16 + c[1] as u16 + c[2] as u16 > 36)
        .count() as f64;
    lit / (pixels.len() / 4) as f64
}

#[test]
fn ao_disabled_is_bit_identical_to_no_ao() {
    let Some((device, queue, renderer, mesh, reference)) = prepared_corner() else {
        return;
    };
    // Zero intensity collapses both kernels to the white factor on the
    // GPU, exactly like the unbound fallback.
    for method in [AoMethod::Ssao, AoMethod::Gtao] {
        let pixels = render_pixels(&device, &queue, &renderer, &mesh, Some(wiring(method, 0.0)));
        assert_eq!(
            pixels, reference,
            "{method:?} at zero intensity must be bit-identical to AO-free"
        );
    }
}

#[test]
fn ao_darkens_the_corner_and_never_brightens() {
    let Some((device, queue, renderer, mesh, reference)) = prepared_corner() else {
        return;
    };
    let lit = lit_fraction(&reference);
    eprintln!("ao corner probe: lit fraction = {lit:.3}");
    assert!(
        lit > MIN_LIT_FRACTION,
        "corner scene rendered dark: {lit:.3}"
    );

    for method in [AoMethod::Ssao, AoMethod::Gtao] {
        let pixels = render_pixels(&device, &queue, &renderer, &mesh, Some(wiring(method, 1.0)));
        // Determinism: the hash is per-pixel, with no temporal component.
        let again = render_pixels(&device, &queue, &renderer, &mesh, Some(wiring(method, 1.0)));
        assert_eq!(pixels, again, "{method:?} AO frame is not deterministic");
        let mut dark = 0usize;
        let mut bright = 0usize;
        for (a, b) in reference.chunks_exact(4).zip(pixels.chunks_exact(4)) {
            let darker = [0, 1, 2]
                .iter()
                .any(|&i| a[i].saturating_sub(b[i]) > OCCLUSION_DELTA);
            let brighter = [0, 1, 2].iter().any(|&i| b[i].saturating_sub(a[i]) > TOL);
            dark += darker as usize;
            bright += brighter as usize;
        }
        let total = (W * H) as f64;
        let fraction = dark as f64 / total;
        eprintln!(
            "ao corner probe: {method:?} darkened pixels = {dark} ({fraction:.4}), brightened = {bright}"
        );
        assert_eq!(
            bright, 0,
            "{method:?} brightened {bright} pixels (AO must never add light)"
        );
        assert!(
            fraction >= MIN_DARK_FRACTION,
            "{method:?} darkened too little: {fraction:.4} < {MIN_DARK_FRACTION}"
        );
    }
}
