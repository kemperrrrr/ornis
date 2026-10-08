//! MSAA 4x headless pixel gate: the same smooth-shaded scene (dielectric,
//! roughness 0.5 — the shimmer report's material) rendered at 1x and at the
//! negotiated sample count must both produce a real image, the negotiated
//! count must be deterministic across two runs (resolve determinism), and a
//! true 4x frame must stay within a small edge-only tolerance of the 1x
//! frame. Exact 1x==4x equality is never asserted: edge coverage
//! legitimately differs.
//!
//! Skip contract: returns early with a note when no adapter exists (same as
//! the other headless gates; CI provides lavapipe). When the adapter cannot
//! do 4x, [`negotiate_sample_count`](ornis_render::negotiate_sample_count)
//! falls back to 1x and the gate pins that fallback (negotiated output is
//! then byte-identical to the 1x frame) instead of failing.
//!
//! Adapter acquisition and the render-and-read-back flow live in the shared
//! `common` harness (no per-gate copies — the rustqual ratchet flags exact
//! `DUPLICATE` pairs).

// The harness is shared across the gate binaries; this gate uses the
// sized (non-square) half of it.
#[allow(dead_code)]
mod common;

use common::sized::{FrameSpec, render_and_readback};
use ornis_render::render_backend::RenderContext;
use ornis_render::{
    InstanceData, MaterialIdx, OpenPBRMaterial, RenderFrame3D, Renderer3D, negotiate_sample_count,
};

/// Offscreen frame extent (shadow-probe precedent: fast, large enough for
/// silhouette edges).
const W: u32 = 320;
/// Offscreen frame height.
const H: u32 = 180;
/// Read-back format: matches the composite output.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Per-channel drift tolerated before a pixel counts as "different" (same
/// window as the golden probe: driver rounding, not engine regressions).
const TOL: u8 = 4;
/// Maximum fraction of differing pixels between the 1x and true-4x frames.
/// Edges only: interior pixels shade identically after resolve. Calibrated
/// on Metal (observed 125/57600 ≈ 0.002 on the smooth-sphere scene, with a
/// large per-pixel swing on isolated sparkle pixels — the shimmer this
/// fixes); lavapipe rasterizes edges up to a pixel apart (see the golden
/// probe's 1px-shift allowance), so the bound keeps wide headroom on top.
const MAX_DIFF_FRACTION: f64 = 0.05;
/// Minimum fraction of non-black pixels: both frames must render the sphere,
/// not a clear color.
const MIN_LIT_FRACTION: f64 = 0.05;

/// Offscreen frame spec (shared harness builds targets from it).
fn frame_spec() -> FrameSpec {
    FrameSpec {
        size: (W, H),
        format: FORMAT,
    }
}

/// One renderer over the shimmer-report scene: smooth dielectric sphere,
/// roughness 0.5, single directional light, no shadows (the gate targets
/// edge resolve, not the shadow pre-pass).
fn build_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sample_count: u32,
) -> (Renderer3D, ornis_render::Mesh) {
    let renderer = Renderer3D::new(device, &frame_spec().surface_config(), sample_count);
    let mesh = ornis_render::create_sphere(device, 1.0, 24, 16);
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
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::new(0.0, 0.5, 4.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        55.0f32.to_radians(),
        W as f32 / H as f32,
        0.1,
        100.0,
    );
    renderer.set_camera(queue, &(proj * view).to_cols_array_2d(), [0.0, 0.5, 4.0]);
    renderer.set_lights(
        queue,
        [0.05, 0.05, 0.08],
        &[ornis_assets::scene::LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(0.42, 0.84, 0.3))
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

/// Renders one plan-path frame (the native shell combo: Hybrid plan +
/// renderer at the same negotiated count) into a fresh target and reads
/// it back. The plan pool holds the multisampled targets at 4x; the
/// single-sample resolves stay renderer-owned.
fn render_plan_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer3D,
    mesh: &ornis_render::Mesh,
    sample_count: u32,
) -> Vec<u8> {
    let mut plan = RenderFrame3D::new_with_samples(
        FORMAT,
        (W, H),
        ornis_render::Technique::Hybrid,
        ornis_render::Bloom::Off,
        sample_count,
    );
    assert_eq!(plan.sample_count(), sample_count);
    render_and_readback(device, queue, &frame_spec(), |encoder, view| {
        plan.render(
            RenderContext {
                device,
                queue,
                encoder,
                target: view,
            },
            renderer,
            mesh,
            1,
        );
    })
}

#[test]
fn msaa_4x_resolve_is_deterministic_and_close_to_1x() {
    let Some((adapter, device, queue)) = pollster::block_on(common::request_device()) else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return;
    };
    let negotiated = negotiate_sample_count(&adapter, ornis_render::MSAA_SAMPLE_COUNT);
    assert!(
        negotiated == 1 || negotiated == ornis_render::MSAA_SAMPLE_COUNT,
        "negotiated count must be 1x or 4x, got {negotiated}"
    );
    eprintln!("msaa probe: negotiated sample count = {negotiated}");

    let (one, mesh_one) = build_scene(&device, &queue, 1);
    let (msaa, mesh_msaa) = build_scene(&device, &queue, negotiated);
    assert_eq!(one.sample_count(), 1);
    assert_eq!(msaa.sample_count(), negotiated);
    // Resolve targets exist exactly in MSAA mode (and grow the budget).
    if negotiated > 1 {
        assert!(
            msaa.texture_budget() > one.texture_budget(),
            "4x must allocate MSAA + resolve storage over 1x"
        );
    } else {
        assert_eq!(msaa.texture_budget(), one.texture_budget());
    }

    let px_1x = render_pixels(&device, &queue, &one, &mesh_one);
    let px_ms_a = render_pixels(&device, &queue, &msaa, &mesh_msaa);
    let px_ms_b = render_pixels(&device, &queue, &msaa, &mesh_msaa);

    let lit_fraction = |px: &[u8]| -> f64 {
        let lit = px
            .chunks_exact(4)
            .filter(|c| c[0] as u16 + c[1] as u16 + c[2] as u16 > 36)
            .count();
        lit as f64 / (px.len() / 4) as f64
    };
    let (lit_1x, lit_ms) = (lit_fraction(&px_1x), lit_fraction(&px_ms_a));
    eprintln!("msaa probe: lit fraction 1x={lit_1x:.3} negotiated={lit_ms:.3}");
    assert!(
        lit_1x > MIN_LIT_FRACTION,
        "1x frame rendered nothing lit: {lit_1x:.3}"
    );
    assert!(
        lit_ms > MIN_LIT_FRACTION,
        "negotiated frame rendered nothing lit: {lit_ms:.3}"
    );

    // Resolve determinism: two runs at the negotiated count agree exactly.
    assert_eq!(
        px_ms_a.len(),
        px_ms_b.len(),
        "readback size drifted between runs"
    );
    let nondet = px_ms_a
        .iter()
        .zip(px_ms_b.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!("msaa probe: nondeterministic bytes across two {negotiated}x runs = {nondet}");
    assert_eq!(nondet, 0, "MSAA resolve is not deterministic run-to-run");

    if negotiated == 1 {
        // Documented 1x fallback: same code path, so byte-identical to 1x.
        let drift = px_1x
            .iter()
            .zip(px_ms_a.iter())
            .filter(|(a, b)| a != b)
            .count();
        eprintln!("msaa probe: fallback 1x drift vs 1x = {drift} bytes");
        assert_eq!(drift, 0, "1x fallback must render exactly like 1x");
        return;
    }

    // True 4x: edge-only tolerance against 1x (never exact equality).
    let total = (W * H) as usize;
    let mut diff_px = 0usize;
    let mut max_diff: u8 = 0;
    for (a, b) in px_1x.chunks_exact(4).zip(px_ms_a.chunks_exact(4)) {
        let d = (a[0].abs_diff(b[0]))
            .max(a[1].abs_diff(b[1]))
            .max(a[2].abs_diff(b[2]));
        max_diff = max_diff.max(d);
        if d > TOL {
            diff_px += 1;
        }
    }
    let fraction = diff_px as f64 / total as f64;
    eprintln!(
        "msaa probe: 1x vs 4x differing pixels = {diff_px}/{total} ({fraction:.4}), max channel diff = {max_diff}"
    );
    assert!(
        fraction <= MAX_DIFF_FRACTION,
        "4x frame drifted beyond edges: {fraction:.4} > {MAX_DIFF_FRACTION}"
    );
}

#[test]
fn msaa_plan_renders_nonblack_at_negotiated_count() {
    // Plan-pool MSAA proof: the native shell combo (Hybrid plan + renderer
    // at the same negotiated count) must render the sphere, not a clear
    // color. On the documented 1x fallback this exercises the 1x plan
    // path; on true 4x it proves the pool MSAA + renderer-resolve wiring.
    let Some((adapter, device, queue)) = pollster::block_on(common::request_device()) else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return;
    };
    let negotiated = negotiate_sample_count(&adapter, ornis_render::MSAA_SAMPLE_COUNT);
    assert!(
        negotiated == 1 || negotiated == ornis_render::MSAA_SAMPLE_COUNT,
        "negotiated count must be 1x or 4x, got {negotiated}"
    );
    let (renderer, mesh) = build_scene(&device, &queue, negotiated);
    assert_eq!(renderer.sample_count(), negotiated);
    let px = render_plan_pixels(&device, &queue, &renderer, &mesh, negotiated);
    let lit = px
        .chunks_exact(4)
        .filter(|c| c[0] as u16 + c[1] as u16 + c[2] as u16 > 36)
        .count() as f64
        / (px.len() / 4) as f64;
    eprintln!("msaa plan probe: lit fraction at {negotiated}x = {lit:.3}");
    assert!(
        lit > MIN_LIT_FRACTION,
        "plan frame at {negotiated}x rendered nothing lit: {lit:.3}"
    );
}
