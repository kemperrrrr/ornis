//! Light-budget, scene-load, shadow-fit, and GPU upload-reporting gates.

use super::super::*;
use super::*;

/// Light-limit gate: ten scene lights upload eight and report two
/// dropped — the silent-truncate path is now explicit. Pure CPU
/// (no adapter needed).
#[test]
fn ten_scene_lights_preview_two_dropped() {
    let lights: Vec<ornis_assets::scene::LightDesc> = (0..10)
        .map(|_| ornis_assets::scene::LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(1.0, 1.0, 1.0))
                .expect("non-zero direction"),
            intensity: 0.6,
            color: [1.0, 1.0, 1.0],
            shadow: ornis_assets::scene::ShadowCast::Disabled,
        })
        .collect();
    let rig = crate::extraction::RenderLights {
        ambient: ornis_core::Color::linear_rgb(0.10, 0.10, 0.15),
        lights,
        ambient_intensity: ornis_core::Lux::new(1.0),
        exposure: ornis_core::Lux::new(1.0),
        environment_weight: None,
    };
    let stats = rig.light_upload_stats();
    assert_eq!(stats.uploaded, 8, "{stats:?}");
    assert_eq!(stats.dropped_lights, 2, "{stats:?}");
    assert_eq!(stats.dropped_shadows, 0, "{stats:?}");
    assert_eq!(stats, crate::renderer::count_light_drops(&rig.lights));
}

/// Back-compat gate: the shipped demo RON (no IBL fields) loads and
/// the resource defaults both multipliers to the exact no-op `1.0`.
/// Pure CPU (no adapter needed).
#[test]
fn old_scene_ron_loads_with_ibl_defaults() {
    let ron_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/scene.ron");
    let ron = std::fs::read_to_string(&ron_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", ron_path.display()));
    let scene = ornis_assets::scene::Scene::from_ron(&ron).expect("parse assets/scene.ron");
    let rig = crate::extraction::RenderLights::from_scene(&scene);
    assert_eq!(rig.ambient_intensity, ornis_core::Lux::new(1.0));
    assert_eq!(rig.exposure, ornis_core::Lux::new(1.0));
    // The demo scene fits the limits — the scene-load log stays quiet.
    let stats = rig.light_upload_stats();
    assert_eq!(stats.dropped_lights, 0, "{stats:?}");
    assert_eq!(stats.dropped_shadows, 0, "{stats:?}");
}

/// Shadow-fit gate: a radius-50 scene AABB grows the directional
/// ortho box to cover it (default stays ±12). Pure CPU.
#[test]
fn shadow_bounds_fit_covers_radius_50_scene() {
    let (center, half) = crate::renderer::shadow_fit_for_bounds([-50.0; 3], [50.0; 3]);
    assert_eq!(center, [0.0, 0.0, 0.0]);
    assert!(half >= 50.0, "half={half}");
    let (_, small) = crate::renderer::shadow_fit_for_bounds([-1.0; 3], [1.0; 3]);
    assert_eq!(small, crate::renderer::SHADOW_ORTHO_HALF);
}

/// GPU path: `set_lights_full` uploads ten lights, reports two
/// dropped, publishes the report, and the scene fit switches the
/// ortho half from ±12 to the fitted value and back. Skipped when no
/// adapter is available.
#[test]
fn gpu_upload_reports_drops_and_applies_scene_fit() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: 320,
        height: 180,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    let renderer = crate::renderer::Renderer3D::new(&device, &surface_config, 1);
    assert_eq!(
        renderer.shadow_half_extent(),
        crate::renderer::SHADOW_ORTHO_HALF
    );
    renderer.set_shadow_bounds([-50.0; 3], [50.0; 3]);
    assert!(
        renderer.shadow_half_extent() >= 50.0,
        "half={}",
        renderer.shadow_half_extent()
    );
    let lights: Vec<ornis_assets::scene::LightDesc> = (0..10)
        .map(|_| ornis_assets::scene::LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(0.0, 1.0, 1.0))
                .expect("non-zero direction"),
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            shadow: ornis_assets::scene::ShadowCast::Disabled,
        })
        .collect();
    let stats = renderer.set_lights_full(&queue, [0.1, 0.1, 0.15], 1.0, 1.0, &lights);
    assert_eq!(stats.uploaded, 8, "{stats:?}");
    assert_eq!(stats.dropped_lights, 2, "{stats:?}");
    assert_eq!(renderer.light_upload_stats(), stats);
    renderer.clear_shadow_bounds();
    assert_eq!(
        renderer.shadow_half_extent(),
        crate::renderer::SHADOW_ORTHO_HALF
    );
}
