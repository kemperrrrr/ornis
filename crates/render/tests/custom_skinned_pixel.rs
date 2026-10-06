//! Pixel gate for the production custom-mesh path (`RenderSubmit` staging
//! through `RenderFrame3D`): one skinned entry — GPU-blended and CPU
//! fallback — must render non-black pixels, and the unified `*_with_custom`
//! passes must keep spheres pixel-identical. Runs headless — on CI via
//! lavapipe, locally on any adapter; skipped when no adapter is found.
//! The scene and harness live in `common` (shared with the S5b/E1/E2
//! parity gates). Geometry is synthesized inline (no binary fixtures),
//! mirroring the extraction `custom_quad_*` / `push_skinned_triangle`
//! fixtures.

mod common;

use glam::Mat4;
use ornis_animation::{JointId, JointPose, Skeleton, SkinnedMesh, SkinningMode};
use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
use ornis_core::Engine;
use ornis_core::units::Clamped01;

/// Orange dielectric shared by the gate entries (matches the diagnosed
/// all-Custom character materials closely enough to prove the draw).
fn orange_dielectric() -> MaterialDesc {
    MaterialDesc::Dielectric {
        base_color: [1.0, 0.45, 0.1],
        roughness: Clamped01::new(0.5),
        emission: [0.0, 0.0, 0.0],
        metallic: ornis_core::Metallic::new(0.0),
    }
}

/// Inline two-joint scene: a root (`root` + `child`, identity pose) plus one
/// skinned triangle entity filling the headless camera frustum (bind
/// positions around the origin in the `z = 0` plane, `+Z` normals).
/// Identity skin, so GPU blend and CPU rows coincide — the gate proves the
/// draw path, not the blend math (pinned unit-side by the parity tests).
fn push_two_joint_triangle(engine: &mut Engine) {
    {
        let store = engine.world_mut().store_mut().expect("store");
        store.register::<Skeleton>();
        store.register::<JointPose>();
        store.register::<SkinnedMesh>();
    }
    let root = {
        let store = engine.world_mut().store_mut().expect("store");
        let root = store.create_entity();
        store.insert(
            root,
            Skeleton::new(
                vec![None, Some(JointId::from_raw(0))],
                vec![Mat4::IDENTITY; 2],
                vec!["root".to_string(), "child".to_string()],
            ),
        );
        store.insert(root, JointPose::identity(2));
        root
    };
    let positions = vec![[-1.5, -1.5, 0.0], [1.5, -1.5, 0.0], [0.0, 1.5, 0.0]];
    {
        let store = engine.world_mut().store_mut().expect("store");
        let mesh = store.create_entity();
        store.insert(mesh, TransformDesc::IDENTITY);
        store.insert(
            mesh,
            MeshDesc::Custom {
                positions: positions.clone(),
                indices: vec![0, 1, 2],
            },
        );
        store.insert(mesh, orange_dielectric());
        store.insert(
            mesh,
            SkinnedMesh::new(
                root,
                vec![[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
                vec![
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                ],
                positions,
                vec![[0.0, 0.0, 1.0]; 3],
                vec![[0.0, 0.0]; 3],
                vec![0, 1, 2],
            ),
        );
    }
}

/// Pixels with any lit channel (alpha ignored — the composite clears to
/// opaque black, so lit geometry is exactly the non-black set).
fn nonblack_count(pixels: &[u8]) -> usize {
    pixels
        .chunks_exact(4)
        .filter(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
        .count()
}

#[test]
fn skinned_entry_renders_nonblack_through_production_path() {
    common::with_headless_scene(|scene| {
        let mut engine = Engine::new();
        push_two_joint_triangle(&mut engine);
        let upload = ornis_render::extract_render_data(engine.world().store().expect("store"));
        assert_eq!(upload.custom_meshes.len(), 1, "one skinned entry");
        assert_eq!(
            upload.custom_meshes[0].skinning,
            SkinningMode::Gpu,
            "identity two-joint skin stages a palette"
        );
        // Production staging: the same call `RenderSubmit` makes.
        let staged =
            scene
                .renderer
                .stage_custom_meshes(&scene.device, &scene.queue, &upload.custom_meshes);
        assert_eq!(staged.len(), 1, "valid entry stages, never skipped");
        scene
            .renderer
            .upload_materials(&scene.device, &scene.queue, &upload.materials);
        let instances: Vec<ornis_render::InstanceData> =
            staged.iter().map(|entry| entry.instance).collect();
        scene
            .renderer
            .upload_instances(&scene.device, &scene.queue, &instances);
        let items = ornis_render::custom_draw_items(&staged, 0);
        let pixels = common::render_frame_pixels(scene, |plan, context| {
            plan.render_with_custom(context, &scene.renderer, &scene.mesh, 0, &items);
        });
        let empty = common::render_frame_pixels(scene, |plan, context| {
            plan.render(context, &scene.renderer, &scene.mesh, 0);
        });
        assert_eq!(nonblack_count(&empty), 0, "empty frame stays black");
        assert_ne!(pixels, empty, "skinned entry must draw something");
        assert!(
            nonblack_count(&pixels) > 100,
            "skinned entry covers real pixels, got {}",
            nonblack_count(&pixels)
        );
    });
}

#[test]
fn cpu_fallback_entry_renders_nonblack_through_production_path() {
    common::with_headless_scene(|scene| {
        let mut engine = Engine::new();
        push_two_joint_triangle(&mut engine);
        // Destroy every skeleton root: the mesh lane outlives its
        // skeleton, so extraction falls back to CPU rows with no palette
        // (same honesty as the `missing_skeleton_*` extraction test).
        let roots: Vec<ornis_core::Entity> = {
            let store = engine.world().store().expect("store");
            store
                .read_lane::<Skeleton>()
                .map(|lane| lane.entities.clone())
                .unwrap_or_default()
        };
        for root in roots {
            let store = engine.world_mut().store_mut().expect("store");
            if store.is_alive(root) {
                store.destroy_entity(root);
            }
        }
        let upload = ornis_render::extract_render_data(engine.world().store().expect("store"));
        assert_eq!(upload.custom_meshes.len(), 1, "one skinned entry");
        assert_eq!(
            upload.custom_meshes[0].skinning,
            SkinningMode::Cpu,
            "skeletonless skin falls back to CPU rows"
        );
        assert!(upload.custom_meshes[0].joint_palette.is_none());
        let staged =
            scene
                .renderer
                .stage_custom_meshes(&scene.device, &scene.queue, &upload.custom_meshes);
        assert_eq!(staged.len(), 1, "CPU entry stages its pre-skinned rows");
        scene
            .renderer
            .upload_materials(&scene.device, &scene.queue, &upload.materials);
        let instances: Vec<ornis_render::InstanceData> =
            staged.iter().map(|entry| entry.instance).collect();
        scene
            .renderer
            .upload_instances(&scene.device, &scene.queue, &instances);
        let items = ornis_render::custom_draw_items(&staged, 0);
        let pixels = common::render_frame_pixels(scene, |plan, context| {
            plan.render_with_custom(context, &scene.renderer, &scene.mesh, 0, &items);
        });
        assert_ne!(
            pixels,
            common::render_frame_pixels(scene, |plan, context| {
                plan.render(context, &scene.renderer, &scene.mesh, 0);
            }),
            "CPU fallback entry must draw something"
        );
        assert!(
            nonblack_count(&pixels) > 100,
            "CPU fallback covers real pixels, got {}",
            nonblack_count(&pixels)
        );
    });
}

#[test]
fn unified_custom_path_keeps_spheres_pixel_identical() {
    common::with_headless_scene(|scene| {
        // Empty customs through the unified passes must match the legacy
        // reference exactly — spheres-unchanged by construction, pinned.
        let unified = common::render_frame_pixels(scene, |plan, context| {
            plan.render_with_custom(context, &scene.renderer, &scene.mesh, 1, &[]);
        });
        assert_eq!(
            common::sequential_reference_pixels(scene),
            unified,
            "empty customs must render exactly like the legacy path"
        );
    });
}
