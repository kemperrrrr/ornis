//! Frame-submit gate for CSM (#121): a perspective camera turns cascades
//! on and publishes the extracted caster bounds; an orthographic camera
//! turns both off again.

use super::*;
use ornis_assets::scene::{LightDesc, MaterialDesc, MeshDesc, ShadowCast, TransformDesc};
use ornis_core::System;
use ornis_core::units::{Clamped01, Meters, PositiveF32, UnitQuat, UnitVec3};

fn place_sphere(store: &mut ornis_core::SmartStore) {
    let handle = store.create_entity();
    store.insert(
        handle,
        TransformDesc {
            translation: glam::Vec3::ZERO,
            rotation: UnitQuat::IDENTITY,
            scale: glam::Vec3::splat(2.0),
        },
    );
    store.insert(
        handle,
        MeshDesc::Sphere {
            radius: PositiveF32::expect_valid(1.0),
            segments: 8,
            rings: 6,
        },
    );
    store.insert(
        handle,
        MaterialDesc::Dielectric {
            base_color: [0.8, 0.8, 0.8],
            roughness: Clamped01::new(0.5),
            emission: [0.0, 0.0, 0.0],
            metallic: ornis_core::Metallic::new(0.0),
        },
    );
}

fn shadowed_sun() -> crate::extraction::RenderLights {
    crate::extraction::RenderLights {
        lights: vec![LightDesc::Directional {
            direction: UnitVec3::Y,
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            shadow: ShadowCast::Enabled,
        }],
        ..crate::extraction::RenderLights::default()
    }
}

fn ortho_camera() -> crate::camera::OrbitCamera {
    let desc = ornis_assets::scene::CameraDesc::try_orthographic_units(
        [Meters::new(0.0), Meters::new(4.0), Meters::new(12.0)],
        [Meters::new(0.0), Meters::new(0.0), Meters::new(0.0)],
        [0.0, 1.0, 0.0],
        Meters::new(4.0),
        Meters::new(0.1),
        Meters::new(200.0),
    )
    .expect("orthographic camera");
    crate::camera::OrbitCamera::from_desc(&desc)
}

/// Perspective submit publishes four cascades and the sphere's bounds.
/// The same world with an orthographic camera clears both. Skipping
/// `set_csm_camera` or `set_csm_casters` in the submit fails this.
#[test]
fn frame_submit_enables_csm_for_perspective() {
    const W: u32 = 320;
    const H: u32 = 180;
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping frame CSM submit");
        return;
    };
    let mut engine = ornis_core::Engine::new();
    place_sphere(engine.world_mut().store_mut().expect("store"));
    let orbit =
        crate::camera::OrbitCamera::looking_at(glam::Vec3::new(0.0, 4.0, 12.0), glam::Vec3::ZERO)
            .with_clip(Meters::new(0.1), Meters::new(200.0));
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        width: W,
        height: H,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    let renderer = crate::renderer::Renderer3D::new(&device, &surface_config, 1);
    let frame3d =
        crate::frame_exec::RenderFrame3D::new(wgpu::TextureFormat::Rgba8UnormSrgb, (W, H));
    engine.world_mut().insert(std::sync::Mutex::new(orbit));
    engine.world_mut().insert(shadowed_sun());
    engine
        .world_mut()
        .insert(crate::gpu_resources::GpuDevice(device));
    engine
        .world_mut()
        .insert(crate::gpu_resources::GpuQueue(queue));
    engine
        .world_mut()
        .insert(crate::gpu_resources::GpuSurfaceState {
            size: (W, H),
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        });
    engine
        .world_mut()
        .insert(std::sync::Mutex::new(crate::gpu_resources::GpuFrameState {
            renderer,
            frame3d,
        }));
    engine
        .world_mut()
        .insert(crate::gpu_resources::GpuCustomMeshes::default());
    crate::gpu_resources::RenderSubmit.run(engine.world().resources());
    {
        let resources = engine.world().resources();
        let fs = resources
            .get::<std::sync::Mutex<crate::gpu_resources::GpuFrameState>>()
            .expect("frame state")
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(
            fs.renderer.csm_cascade_count(),
            crate::renderer::CSM_CASCADES as u32,
            "perspective submit did not enable CSM"
        );
        let (min, max) = fs
            .renderer
            .csm_casters_for_tests()
            .expect("perspective submit did not publish caster bounds");
        assert!(
            min.iter().all(|v| *v < -1.0) && max.iter().all(|v| *v > 1.0),
            "caster AABB {min:?}..{max:?} does not cover the sphere"
        );
    }
    {
        let resources = engine.world().resources();
        let camera = resources
            .get::<std::sync::Mutex<crate::camera::OrbitCamera>>()
            .expect("camera");
        *camera
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = ortho_camera();
    }
    crate::gpu_resources::RenderSubmit.run(engine.world().resources());
    let resources = engine.world().resources();
    let fs = resources
        .get::<std::sync::Mutex<crate::gpu_resources::GpuFrameState>>()
        .expect("frame state")
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(
        fs.renderer.csm_cascade_count(),
        0,
        "orthographic submit left CSM enabled"
    );
    assert!(
        fs.renderer.csm_casters_for_tests().is_none(),
        "orthographic submit left caster bounds in place"
    );
}
