//! Unlit sprite headless pixel gate: a `Quad` with an `Unlit` material in
//! front of the camera must render identically under wildly different
//! lighting (one directional vs eight colored point lights, shadows on and
//! off), while the same quad with a `Dielectric` material must shade
//! noticeably differently (proving the gate is light-sensitive), and the
//! quad must cover the expected central frame area.
//!
//! A fourth case pins the shader flag itself: an `OpenPBRMaterial` with
//! nonzero lobes (gray 0.5 base, roughness 0.5), the sprite color as
//! emission and `ShadingMode::Unlit` must also render light-independently —
//! with the lobes active a disabled flag would shade light-dependently, so
//! the case fails when the flag is ignored (mutation-checked by raising
//! the shader threshold to 2.0). It runs through both the legacy
//! `render_scene` (deferred) path and `RenderFrame3D` with
//! `Technique::Forward`.
//!
//! Skip contract: the shared `with_device` entry returns early with a note
//! when no adapter exists (same as the other headless gates; CI provides
//! lavapipe).
//!
//! The extraction case goes through the real pipeline (`Engine` lanes +
//! [`ornis_render::extract_render_data`]), so the gate also pins the
//! `Quad` size scale and the `Unlit` GPU mapping end to end.
//!
//! Adapter acquisition and the render-and-read-back flow live in the shared
//! `common` harness (no per-gate copies — the rustqual ratchet flags exact
//! `DUPLICATE` pairs).

// The harness is shared across the gate binaries; this gate uses the
// sized (non-square) half of it.
#[allow(dead_code)]
mod common;

use common::sized::{FrameSpec, render_and_readback};
use ornis_assets::scene::{LightDesc, MaterialDesc, MeshDesc, ShadowCast, TransformDesc};
use ornis_core::Engine;
use ornis_core::units::{Clamped01, LinearRgb, Meters, UnitVec3};
use ornis_render::render_backend::RenderContext;
use ornis_render::{
    Bloom, InstanceData, MaterialIdx, OpenPBRMaterial, RenderFrame3D, Renderer3D, Technique,
    extract_render_data,
};

/// Offscreen frame extent (same as the MSAA probe).
const W: u32 = 320;
/// Offscreen frame height.
const H: u32 = 180;
/// Read-back format: matches the composite output.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Per-channel drift tolerated for the unlit-equality gates.
const TOL: u8 = 1;
/// Brightness gate isolating quad pixels from the near-black background
/// (same window as the MSAA probe's lit fraction).
const LIT_SUM: u16 = 36;
/// Minimum fraction of differing pixels proving the dielectric control is
/// light-sensitive (per-channel swing above [`CTRL_TOL`]).
const CTRL_MIN_DIFF_FRACTION: f64 = 0.5;
/// Per-channel swing counting as "noticeably different" for the control.
const CTRL_TOL: u8 = 10;
/// Minimum stable-lit fraction: the quad must render, not a clear color
/// (the upper band is pinned by the bbox case below).
const MIN_COVERAGE: f64 = 0.08;
/// Expected painted bbox side: a 2 m quad at distance 4 under a 55-degree
/// perspective spans `2 / (2*4*tan(27.5°)) * 180 ≈ 86.4` px.
const EXPECTED_SIDE: f32 = 86.4;
/// Bbox width/height/center tolerance in pixels (rasterizer edge rounding).
const BBOX_TOL_PX: f32 = 1.0;

/// Offscreen frame spec (shared harness builds targets from it).
fn frame_spec() -> FrameSpec {
    FrameSpec {
        size: (W, H),
        format: FORMAT,
    }
}

/// Runs `gate` with a device pair, or prints the single SKIP note and
/// returns when no adapter exists — the only SKIP branch in this file.
fn with_device(gate: impl FnOnce(&wgpu::Device, &wgpu::Queue)) {
    let Some((_, device, queue)) = pollster::block_on(common::request_device()) else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return;
    };
    gate(&device, &queue);
}

/// Sprite under test: 2x2 unlit steel-blue.
fn unlit_material() -> MaterialDesc {
    MaterialDesc::unlit_units(LinearRgb::new([0.2, 0.4, 0.8]))
}

/// Control material: mid-gray dielectric, roughness 0.5.
fn dielectric_material() -> MaterialDesc {
    MaterialDesc::dielectric_units(LinearRgb::new([0.5, 0.5, 0.5]), Clamped01::new(0.5))
}

/// Flag-sensitive unlit material: nonzero lobes (gray 0.5 base, roughness
/// 0.5) so an ignored flag would shade light-dependently, the sprite
/// color as emission, unlit flag set. Built on the shared
/// [`ornis_render::unlit_material`] mapping with the lobes restored.
fn unlit_flag_material() -> OpenPBRMaterial {
    let mut material = ornis_render::unlit_material(LinearRgb::new([0.2, 0.4, 0.8]));
    material.base.color_rgb([0.5, 0.5, 0.5]);
    material.base.weight(1.0);
    material.specular.weight(1.0);
    material.specular.roughness(0.5);
    material
}

/// One directional light, optionally shadowed.
fn directional_lights(shadow: ShadowCast) -> Vec<LightDesc> {
    vec![LightDesc::Directional {
        direction: UnitVec3::normalize(glam::Vec3::new(0.42, 0.84, 0.3))
            .expect("non-zero direction"),
        intensity: 1.5,
        color: [1.0, 1.0, 1.0],
        shadow,
    }]
}

/// Eight point lights of distinct colors on a ring facing the quad.
fn point_lights(shadow: ShadowCast) -> Vec<LightDesc> {
    const COLORS: [[f32; 3]; 8] = [
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.1, 1.0],
        [1.0, 1.0, 0.1],
        [0.1, 1.0, 1.0],
        [1.0, 0.1, 1.0],
        [1.0, 0.5, 0.1],
        [0.9, 0.9, 0.8],
    ];
    COLORS
        .iter()
        .enumerate()
        .map(|(i, color)| {
            let angle = i as f32 / 8.0 * std::f32::consts::TAU;
            LightDesc::Point {
                position: glam::Vec3::new(3.0 * angle.cos(), 3.0 * angle.sin(), 2.5),
                intensity: 3.0,
                color: *color,
                range: Meters::new(12.0),
                shadow,
            }
        })
        .collect()
}

/// The shared quad camera: 55-degree perspective at +Z 4 looking at the
/// origin (the bbox case derives [`EXPECTED_SIDE`] from this setup).
fn set_quad_camera(renderer: &Renderer3D, queue: &wgpu::Queue) {
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::new(0.0, 0.0, 4.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        55.0f32.to_radians(),
        W as f32 / H as f32,
        0.1,
        100.0,
    );
    renderer.set_camera(queue, &(proj * view).to_cols_array_2d(), [0.0, 0.0, 4.0]);
}

/// One renderer over the 2x2 quad with `material` uploaded directly (no
/// extraction) plus the shared camera and `lights`.
fn upload_direct(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material: &OpenPBRMaterial,
    lights: &[LightDesc],
) -> (Renderer3D, ornis_render::Mesh) {
    let renderer = Renderer3D::new(device, &frame_spec().surface_config(), 1);
    let mesh = ornis_render::mesh::create_quad(device, [1.0, 1.0]);
    renderer.upload_materials(device, queue, std::slice::from_ref(material));
    // Unit mesh with the 2x2 sprite scale baked into the model matrix
    // (shared-unit-mesh convention, like the sphere radius).
    let model = glam::Mat4::from_scale(glam::Vec3::new(2.0, 2.0, 1.0));
    renderer.upload_instances(
        device,
        queue,
        &[InstanceData {
            model_matrix: model,
            normal_matrix: model.inverse().transpose(),
            material_index: MaterialIdx::from_raw(0),
        }],
    );
    set_quad_camera(&renderer, queue);
    renderer.set_lights(queue, [0.05, 0.05, 0.08], lights);
    (renderer, mesh)
}

/// Renders one frame of the `Quad` + `material` scene under `lights` and
/// reads it back (legacy hybrid path: gbuffer, deferred lighting, forward,
/// composite).
fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material: MaterialDesc,
    lights: &[LightDesc],
) -> Vec<u8> {
    let renderer = Renderer3D::new(device, &frame_spec().surface_config(), 1);
    // Unit mesh: the primitive's size is baked into the extraction model
    // matrix (shared-unit-mesh convention, like the sphere radius), so a
    // 2x2 `Quad` desc pairs with a 1x1 mesh for a 2x2 effective sprite.
    let mesh = ornis_render::mesh::create_quad(device, [1.0, 1.0]);
    let mut engine = Engine::new();
    let store = engine.world_mut().store_mut().expect("store");
    let entity = store.create_entity();
    store.insert(entity, TransformDesc::IDENTITY);
    store.insert(
        entity,
        MeshDesc::try_quad_units([Meters::new(2.0), Meters::new(2.0)]).expect("positive size"),
    );
    store.insert(entity, material);
    let upload = extract_render_data(engine.world().store().expect("store"));
    assert_eq!(upload.instances.len(), 1, "quad entity must extract");
    renderer.upload_materials(device, queue, &upload.materials);
    renderer.upload_instances(device, queue, &upload.instances);
    set_quad_camera(&renderer, queue);
    renderer.set_lights(queue, [0.05, 0.05, 0.08], lights);
    render_and_readback(device, queue, &frame_spec(), |encoder, view| {
        renderer.render_scene(device, queue, encoder, view, &mesh, 1);
    })
}

/// Renders `material` under `lights` through both the legacy
/// `render_scene` (deferred) path and `RenderFrame3D` with
/// `Technique::Forward`, returning the `(legacy, forward)` pair.
fn render_pair(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material: &OpenPBRMaterial,
    lights: &[LightDesc],
) -> (Vec<u8>, Vec<u8>) {
    let (renderer, mesh) = upload_direct(device, queue, material, lights);
    let legacy = render_and_readback(device, queue, &frame_spec(), |encoder, view| {
        renderer.render_scene(device, queue, encoder, view, &mesh, 1);
    });
    let mut plan = RenderFrame3D::new_with(FORMAT, (W, H), Technique::Forward, Bloom::Off);
    let forward = render_and_readback(device, queue, &frame_spec(), |encoder, view| {
        plan.render(
            RenderContext {
                device,
                queue,
                encoder,
                target: view,
            },
            &renderer,
            &mesh,
            1,
        );
    });
    (legacy, forward)
}

/// Per-pixel max channel spread across frames.
fn spread(frames: &[Vec<u8>], pixel: usize) -> u8 {
    let at = |f: &[u8]| [f[4 * pixel], f[4 * pixel + 1], f[4 * pixel + 2]];
    let first = at(&frames[0]);
    frames[1..]
        .iter()
        .flat_map(|f| {
            let c = at(f);
            (0..3).map(move |ch| first[ch].abs_diff(c[ch]))
        })
        .max()
        .unwrap_or(0)
}

fn is_lit(px: &[u8], pixel: usize) -> bool {
    px[4 * pixel] as u16 + px[4 * pixel + 1] as u16 + px[4 * pixel + 2] as u16 > LIT_SUM
}

/// Painted bbox over lit pixels: `(width, height, center_x, center_y)`
/// plus the lit pixel count.
fn painted_bbox(px: &[u8]) -> (i32, i32, f32, f32, usize) {
    let (mut min_x, mut min_y) = (W as i32, H as i32);
    let (mut max_x, mut max_y) = (-1i32, -1i32);
    let mut lit = 0usize;
    for (i, c) in px.chunks_exact(4).enumerate() {
        if c[0] as u16 + c[1] as u16 + c[2] as u16 > LIT_SUM {
            lit += 1;
            let x = (i % W as usize) as i32;
            let y = (i / W as usize) as i32;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    let w = max_x - min_x + 1;
    let h = max_y - min_y + 1;
    let cx = (f64::from(min_x) + f64::from(max_x)) / 2.0;
    let cy = (f64::from(min_y) + f64::from(max_y)) / 2.0;
    (w, h, cx as f32, cy as f32, lit)
}

#[test]
fn unlit_quad_is_light_and_shadow_independent() {
    with_device(|device, queue| {
        let frames = [
            render_frame(
                device,
                queue,
                unlit_material(),
                &directional_lights(ShadowCast::Disabled),
            ),
            render_frame(
                device,
                queue,
                unlit_material(),
                &directional_lights(ShadowCast::Enabled),
            ),
            render_frame(
                device,
                queue,
                unlit_material(),
                &point_lights(ShadowCast::Disabled),
            ),
            render_frame(
                device,
                queue,
                unlit_material(),
                &point_lights(ShadowCast::Enabled),
            ),
        ];
        // Quad mask: lit in the first frame and stable across all four.
        let mask: Vec<usize> = (0..(W * H) as usize)
            .filter(|p| is_lit(&frames[0], *p) && spread(&frames, *p) <= TOL)
            .collect();
        let coverage = mask.len() as f64 / (W as f64 * H as f64);
        assert!(
            coverage > MIN_COVERAGE,
            "quad rendered too small to gate: {coverage:.4}"
        );
        // Every stable lit pixel must match across all four frames within TOL
        // (the mask construction already enforces this; re-assert the worst
        // spread explicitly for the report).
        let worst = (0..(W * H) as usize)
            .filter(|p| is_lit(&frames[0], *p))
            .map(|p| spread(&frames, p))
            .max()
            .unwrap_or(0);
        assert!(
            worst <= TOL,
            "unlit quad changed under different light/shadow: spread {worst} > {TOL}"
        );
    });
}

#[test]
fn dielectric_control_is_light_sensitive() {
    with_device(|device, queue| {
        let dir = render_frame(
            device,
            queue,
            dielectric_material(),
            &directional_lights(ShadowCast::Disabled),
        );
        let pts = render_frame(
            device,
            queue,
            dielectric_material(),
            &point_lights(ShadowCast::Disabled),
        );
        // Same geometric mask as the unlit gate: pixels the quad covers.
        let mask: Vec<usize> = (0..(W * H) as usize).filter(|p| is_lit(&dir, *p)).collect();
        assert!(!mask.is_empty(), "control rendered nothing lit");
        let changed = mask
            .iter()
            .filter(|p| {
                let a = [dir[4 * *p], dir[4 * *p + 1], dir[4 * *p + 2]];
                let b = [pts[4 * *p], pts[4 * *p + 1], pts[4 * *p + 2]];
                (0..3).any(|ch| a[ch].abs_diff(b[ch]) > CTRL_TOL)
            })
            .count();
        let fraction = changed as f64 / mask.len() as f64;
        assert!(
            fraction > CTRL_MIN_DIFF_FRACTION,
            "control is not light-sensitive: only {fraction:.4} of quad pixels changed"
        );
    });
}

#[test]
fn unlit_quad_covers_the_expected_frame_area() {
    with_device(|device, queue| {
        let px = render_frame(
            device,
            queue,
            unlit_material(),
            &directional_lights(ShadowCast::Disabled),
        );
        let (w, h, cx, cy, lit) = painted_bbox(&px);
        assert!(lit > 0, "quad rendered nothing lit");
        // A 2 m quad at distance 4 under the 55-degree camera paints an
        // ~86 px square centered on the frame (±1 px raster rounding).
        assert!(
            ((w as f32) - EXPECTED_SIDE).abs() <= BBOX_TOL_PX,
            "bbox width = {w}px, expected {EXPECTED_SIDE:.1}px ±{BBOX_TOL_PX:.0}px"
        );
        assert!(
            ((h as f32) - EXPECTED_SIDE).abs() <= BBOX_TOL_PX,
            "bbox height = {h}px, expected {EXPECTED_SIDE:.1}px ±{BBOX_TOL_PX:.0}px"
        );
        assert!(
            (cx - (W as f32 - 1.0) / 2.0).abs() <= BBOX_TOL_PX,
            "bbox center x = {cx:.1}px, expected frame center ±{BBOX_TOL_PX:.0}px"
        );
        assert!(
            (cy - (H as f32 - 1.0) / 2.0).abs() <= BBOX_TOL_PX,
            "bbox center y = {cy:.1}px, expected frame center ±{BBOX_TOL_PX:.0}px"
        );
    });
}

#[test]
fn unlit_flag_material_is_light_independent_on_both_paths() {
    with_device(|device, queue| {
        let material = unlit_flag_material();
        let light_sets = [
            directional_lights(ShadowCast::Disabled),
            directional_lights(ShadowCast::Enabled),
            point_lights(ShadowCast::Disabled),
            point_lights(ShadowCast::Enabled),
        ];
        let mut frames = Vec::with_capacity(8);
        for lights in &light_sets {
            let (legacy, forward) = render_pair(device, queue, &material, lights);
            frames.push(legacy);
            frames.push(forward);
        }
        // The quad must render (nonempty lit mask in the first frame).
        let lit_count = (0..(W * H) as usize)
            .filter(|p| is_lit(&frames[0], *p))
            .count();
        assert!(lit_count > 0, "flag material rendered nothing lit");
        // Every pixel of all eight frames (legacy + forward × four light
        // sets) matches the first frame within ±1 per channel. With live
        // lobes an ignored flag would shade light-dependently and fail.
        let worst = (0..(W * H) as usize)
            .map(|p| spread(&frames, p))
            .max()
            .unwrap_or(0);
        assert!(
            worst <= TOL,
            "flagged material changed across lights/paths/shadows: spread {worst} > {TOL}"
        );
    });
}
