//! X3 (S5e) gate: lighting as a world resource — `RenderSubmit` no
//! longer hardcodes `set_lights`; ambient and directionals come from the
//! `RenderLights` resource (written by the scene loader, defaulted to
//! the legacy rig). The probe renders the harness scene twice — with
//! the exact legacy hardcoded arguments vs the resource converted
//! through the same `set_lights_args` canon the system uses — and
//! requires identical pixels. Headless — lavapipe on CI, any adapter
//! locally; skipped when no adapter is found. Harness in `common`.

// The harness is shared across the gate binaries; this gate uses only
// the device and render path, not the sequential reference.
#[allow(dead_code)]
mod common;

use ornis_render::RenderLights;
use ornis_render::scene::{
    CameraDesc, EntityDesc, LightDesc, MaterialDesc, MeshDesc, Scene, TransformDesc,
};

/// The legacy rig as scene data — what `RenderLights::default`
/// reproduces and what `RenderSubmit` hardcoded before X3.
fn legacy_scene() -> Scene {
    Scene {
        name: "light probe".into(),
        entities: vec![EntityDesc {
            name: "sphere".into(),
            transform: TransformDesc {
                translation: [1.0, 2.0, 3.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
            mesh: MeshDesc::Sphere {
                radius: 2.0,
                segments: 48,
                rings: 32,
            },
            material: MaterialDesc::Metal {
                base_color: [0.9, 0.8, 0.2],
                roughness: 0.2,
            },
        }],
        lights: vec![
            LightDesc::Directional {
                direction: [1.0, 1.0, 1.0],
                intensity: 0.6,
                color: [1.0, 1.0, 1.0],
            },
            LightDesc::Directional {
                direction: [-0.5, 0.5, -0.5],
                intensity: 0.3,
                color: [0.8, 0.8, 1.0],
            },
        ],
        camera: CameraDesc {
            position: [0.0, 2.5, 9.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 60.0,
            near: 0.1,
            far: 100.0,
        },
        ambient: [0.10, 0.10, 0.15],
    }
}

#[test]
fn world_lights_reproduce_legacy_rig_pixels() {
    common::with_headless_scene(|scene| {
        // Legacy path: the exact arguments `RenderSubmit` inlined.
        let legacy = common::render_frame_pixels(scene, |plan, context| {
            scene.renderer.set_lights(
                context.queue,
                [0.10, 0.10, 0.15],
                &[
                    ([1.0, 1.0, 1.0], 0.6, [1.0, 1.0, 1.0]),
                    ([-0.5, 0.5, -0.5], 0.3, [0.8, 0.8, 1.0]),
                ],
            );
            plan.render(context, &scene.renderer, &scene.mesh, 1);
        });
        // X3 path: the rig flows Scene -> `RenderLights` resource ->
        // the same `set_lights_args` conversion the system uses.
        let world = ornis_render::RenderWorld::from_scene(&legacy_scene());
        let lights = world
            .engine()
            .world()
            .resources()
            .get::<RenderLights>()
            .expect("scene loader publishes RenderLights");
        let driven = common::render_frame_pixels(scene, |plan, context| {
            scene
                .renderer
                .set_lights(context.queue, lights.ambient, &lights.set_lights_args());
            plan.render(context, &scene.renderer, &scene.mesh, 1);
        });
        assert_eq!(
            legacy, driven,
            "X3: world-driven lights must be pixel-identical"
        );
    });
}
