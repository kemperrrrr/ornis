//! Orthographic pixel-parity gate (K0: 2D as degenerate 3D): a box of
//! known size under an `Orthographic { half_height: 2 m }` camera renders
//! a painted bbox of `size / (2 * half_height) * H` pixels in both
//! dimensions (±1 px), and the same bbox from two eye distances (±1 px) —
//! screen size is distance-independent in orthographic mode.
//!
//! Skip contract: returns early with a note when no adapter exists (same
//! as the other headless gates; CI provides lavapipe).
//!
//! Adapter acquisition and the render-and-read-back flow live in the shared
//! `common` harness (no per-gate copies — the rustqual ratchet flags exact
//! `DUPLICATE` pairs).

// The harness is shared across the gate binaries; this gate uses the
// sized (non-square) half of it.
#[allow(dead_code)]
mod common;

use common::sized::{FrameSpec, render_and_readback};
use ornis_assets::scene::CameraDesc;
use ornis_core::units::Meters;
use ornis_render::{
    InstanceData, MaterialIdx, OpenPBRMaterial, OrbitCamera, Renderer3D, camera_view_projection,
};

/// Offscreen frame extent (matches the MSAA gate).
const W: u32 = 320;
/// Offscreen frame height.
const H: u32 = 180;
/// Read-back format: matches the composite output.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Orthographic half-height in meters (the authored 2D view height).
const HALF_HEIGHT: f32 = 2.0;
/// Box full extents in meters (a 2 m cube at the origin).
const BOX_SIZE: f32 = 2.0;
/// Expected painted bbox edge: `BOX_SIZE / (2 * HALF_HEIGHT) * H` = 90 px.
const EXPECTED_BBOX: f32 = BOX_SIZE / (2.0 * HALF_HEIGHT) * H as f32;
/// Bbox tolerance in pixels (rasterizer edge rounding).
const BBOX_TOL_PX: i32 = 1;
/// Per-channel lit threshold (same window as the MSAA gate).
const LIT_SUM: u16 = 36;
/// Minimum fraction of lit pixels: the box must render, not a clear color.
const MIN_LIT_FRACTION: f64 = 0.05;

/// Offscreen frame spec (shared harness builds targets from it).
fn frame_spec() -> FrameSpec {
    FrameSpec {
        size: (W, H),
        format: FORMAT,
    }
}

/// One renderer over the parity scene: a 2 m dielectric box at the origin,
/// one frontal directional light, no shadows, and the K0 orthographic
/// camera (`half_height` 2 m) at `eye_z` on the +Z axis.
fn build_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    eye_z: f32,
) -> (Renderer3D, ornis_render::Mesh) {
    let renderer = Renderer3D::new(device, &frame_spec().surface_config(), 1);
    let mesh = ornis_render::mesh::create_box(device, [BOX_SIZE, BOX_SIZE, BOX_SIZE]);
    let mut material = OpenPBRMaterial::dielectric();
    material.base.color_rgb([0.8, 0.8, 0.8]);
    material.specular.roughness(0.5);
    renderer.upload_materials(device, queue, &[material]);
    let model = glam::Mat4::IDENTITY;
    renderer.upload_instances(
        device,
        queue,
        &[InstanceData {
            model_matrix: model,
            normal_matrix: model.inverse().transpose(),
            material_index: MaterialIdx::from_raw(0),
        }],
    );
    let desc = CameraDesc::try_orthographic_units(
        [Meters::new(0.0), Meters::new(0.0), Meters::new(eye_z)],
        [Meters::new(0.0), Meters::new(0.0), Meters::new(0.0)],
        [0.0, 1.0, 0.0],
        Meters::new(HALF_HEIGHT),
        Meters::new(0.1),
        Meters::new(100.0),
    )
    .expect("valid orthographic test camera");
    let orbit = OrbitCamera::from_desc(&desc);
    let (view_proj, eye) = camera_view_projection(&orbit.view_parameters(), (W, H));
    renderer.set_camera(queue, &view_proj.to_cols_array_2d(), eye.to_array());
    renderer.set_lights(
        queue,
        [0.05, 0.05, 0.08],
        &[ornis_assets::scene::LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(0.25, 0.5, 1.0))
                .expect("non-zero direction"),
            intensity: 1.2,
            color: [1.0, 1.0, 1.0],
            shadow: ornis_assets::scene::ShadowCast::Disabled,
        }],
    );
    (renderer, mesh)
}

/// Renders one legacy-path frame into a fresh target and reads it back.
fn render_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer3D,
    mesh: &ornis_render::Mesh,
) -> Vec<u8> {
    render_and_readback(device, queue, &frame_spec(), |encoder, view| {
        renderer.render_scene(device, queue, encoder, view, mesh, 1);
    })
}

/// Painted bbox `(width, height)` over lit pixels plus the lit fraction.
fn painted_bbox(pixels: &[u8]) -> (i32, i32, f64) {
    let (mut min_x, mut min_y) = (W as i32, H as i32);
    let (mut max_x, mut max_y) = (-1i32, -1i32);
    let mut lit = 0usize;
    for (i, px) in pixels.chunks_exact(4).enumerate() {
        if px[0] as u16 + px[1] as u16 + px[2] as u16 > LIT_SUM {
            lit += 1;
            let x = (i % W as usize) as i32;
            let y = (i / W as usize) as i32;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    let total = (W * H) as f64;
    (max_x - min_x + 1, max_y - min_y + 1, lit as f64 / total)
}

#[test]
fn ortho_box_bbox_matches_view_height_and_holds_across_distances() {
    let Some((_, device, queue)) = pollster::block_on(common::request_device()) else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return;
    };

    let (near, mesh_near) = build_scene(&device, &queue, 9.0);
    let (far, mesh_far) = build_scene(&device, &queue, 15.0);
    let px_near = render_pixels(&device, &queue, &near, &mesh_near);
    let px_far = render_pixels(&device, &queue, &far, &mesh_far);

    let (w_near, h_near, lit_near) = painted_bbox(&px_near);
    let (w_far, h_far, lit_far) = painted_bbox(&px_far);
    eprintln!(
        "ortho probe: bbox near={w_near}x{h_near} lit={lit_near:.3}, \
         far={w_far}x{h_far} lit={lit_far:.3}, expected={EXPECTED_BBOX:.0}px"
    );
    assert!(
        lit_near > MIN_LIT_FRACTION,
        "near frame rendered nothing lit: {lit_near:.3}"
    );
    assert!(
        lit_far > MIN_LIT_FRACTION,
        "far frame rendered nothing lit: {lit_far:.3}"
    );

    // The 2 m box under a 2 m half-height fills 90 px in both dimensions.
    for (label, got) in [("near width", w_near), ("near height", h_near)] {
        assert!(
            (got as f32 - EXPECTED_BBOX).abs() <= BBOX_TOL_PX as f32,
            "{label} = {got}px, expected {EXPECTED_BBOX:.0}px ±{BBOX_TOL_PX}px"
        );
    }
    // Screen size does not depend on the eye distance.
    assert!(
        (w_near - w_far).abs() <= BBOX_TOL_PX,
        "width drifted with distance: {w_near}px vs {w_far}px"
    );
    assert!(
        (h_near - h_far).abs() <= BBOX_TOL_PX,
        "height drifted with distance: {h_near}px vs {h_far}px"
    );
}
