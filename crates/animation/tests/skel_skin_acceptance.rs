//! Phase C acceptance (design `docs/animation-design.md` §4): imported
//! skin data → [`Skeleton`] + [`SkinnedMesh`] (via the no-dependency bridge
//! builders) → `skel_sample` + `skel_skin_cpu` → the extraction publishes
//! one GPU-skinning entry with world-space vertices and `IDENTITY`
//! matrices.
//!
//! The [`SkinImport`]/[`SkinnedMeshImport`] values below play the importer's
//! role: hand-filled in the multi-format contract shape (today's producer
//! is `ornis-gltf`; this crate must not depend on loader crates, so the
//! values mirror what any compliant importer emits). Real decoding is
//! covered by loader fixture tests, the builders by `import_tests`; this
//! file pins the end-to-end contract only.

use std::f32::consts::FRAC_PI_2;

use glam::{Mat4, Quat, Vec3};
use ornis_animation::{
    ClipId, JointId, JointPose, JointTrack, Key, KeyTrack, SkelClip, SkelPlayer, SkelSampleSystem,
    SkelSkinSystem, SkinImport, SkinnedMesh, SkinnedMeshImport, SkinningMode, skeleton_from_import,
    skinned_mesh_from_import,
};
use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
use ornis_core::units::Clamped01;
use ornis_core::{Engine, Entity, Stage};

/// Registers the skeletal lanes (the engine runs both phase B systems on
/// PostFrame behind the explicit `skel_sample → skel_skin_cpu` edge).
fn skel_engine() -> Engine {
    let mut engine = Engine::new();
    let store = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore");
    store.register::<SkelPlayer>();
    store.register::<ornis_animation::Skeleton>();
    store.register::<JointPose>();
    store.register::<TransformDesc>();
    store.register::<SkinnedMesh>();
    store.register_cold::<SkelClip>();
    engine.add_stage_system(Stage::PostFrame, SkelSampleSystem::new());
    engine.add_stage_system(Stage::PostFrame, SkelSkinSystem::new());
    engine
        .stage_schedule_mut(Stage::PostFrame)
        .try_order_before("skel_sample", "skel_skin_cpu")
        .expect("explicit edge skel_sample -> skel_skin_cpu");
    engine
}

fn identity_transform() -> TransformDesc {
    TransformDesc {
        translation: Vec3::ZERO.to_array(),
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: Vec3::ONE.to_array(),
    }
}

/// What the importer emits for a two-joint primitive: root plus one child,
/// identity binds, node names kept.
fn two_bone_skin_import() -> SkinImport {
    SkinImport {
        parents: vec![-1, 0],
        inverse_bind: vec![Mat4::IDENTITY; 2],
        joint_names: vec!["root".to_string(), "tip".to_string()],
    }
}

/// What the importer emits for the skinned triangle: top-4 `u16` joints,
/// normalized weights, verbatim bind soup.
fn triangle_mesh_import() -> SkinnedMeshImport {
    SkinnedMeshImport {
        joints: vec![[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
        weights: vec![
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        positions: vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 3],
        uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        indices: vec![0, 1, 2],
    }
}

/// Single-key channels hold forever: joint 1 sits at `+X` and spins 90°
/// about Z (`M1 = T·R`), so the pose is time-independent and exact.
fn two_bone_clip() -> SkelClip {
    SkelClip {
        duration: 1.0,
        tracks: vec![JointTrack {
            joint: JointId::from_raw(1),
            translation: KeyTrack {
                keys: vec![Key {
                    time: 0.0,
                    value: Vec3::new(1.0, 0.0, 0.0),
                }],
            },
            rotation: KeyTrack {
                keys: vec![Key {
                    time: 0.0,
                    value: Quat::from_rotation_z(FRAC_PI_2),
                }],
            },
            scale: KeyTrack { keys: Vec::new() },
        }],
    }
}

fn test_material() -> MaterialDesc {
    MaterialDesc::Dielectric {
        base_color: [0.8, 0.2, 0.2],
        roughness: Clamped01::new(0.4),
        emission: [0.0, 0.0, 0.0],
    }
}

/// Skeleton root plus one skinned mesh entity, wired exactly as the loader
/// contract prescribes (mesh `TransformDesc` at identity, `MeshDesc::Custom`
/// bind soup matching the import positions).
fn spawn_skinned_pair(engine: &mut Engine) -> Entity {
    let skeleton = skeleton_from_import(&two_bone_skin_import()).expect("two-bone topology builds");
    let store = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore");
    let root = store.create_entity();
    store.insert(root, identity_transform());
    store.insert(root, skeleton);
    store.insert(root, JointPose::identity(2));
    store.insert_cold(root, two_bone_clip());
    store.insert(
        root,
        SkelPlayer {
            clip: ClipId(root),
            time: 0.0,
            speed: 1.0,
            weight: 1.0,
            playing: true,
        },
    );
    let import = triangle_mesh_import();
    let mesh_entity = store.create_entity();
    store.insert(mesh_entity, identity_transform());
    store.insert(
        mesh_entity,
        MeshDesc::Custom {
            positions: import.positions.clone(),
            indices: import.indices.clone(),
        },
    );
    store.insert(mesh_entity, test_material());
    store.insert(
        mesh_entity,
        skinned_mesh_from_import(root, 2, &import).expect("triangle bind builds"),
    );
    mesh_entity
}

fn close(actual: [f32; 3], expected: Vec3) -> bool {
    (Vec3::from_array(actual) - expected).length() < 1e-4
}

#[test]
fn skinned_primitive_extracts_world_vertices_with_identity() {
    // End-to-end: import-shaped data → builders → sample + CPU skin →
    // one GPU-skinning entry (world vertices, `IDENTITY` matrices, staged
    // two-joint palette).
    let mut engine = skel_engine();
    spawn_skinned_pair(&mut engine);

    engine.run_frame(0.5);

    let (upload, stats) =
        ornis_render::extract_render_data_with_stats(engine.world().store().expect("store"));
    assert_eq!(upload.instances.len(), 0, "skinned never rides the batch");
    assert_eq!(upload.custom_meshes.len(), 1);
    assert_eq!(stats.skipped_bad_skin, 0);
    assert_eq!(stats.skipped_bad_custom, 0);
    // The skeleton root carries a placement but no mesh/material: it is
    // not renderable by design (joints are never ECS entities), so the
    // single incomplete skip is the root itself.
    assert_eq!(stats.skipped_incomplete, 1);

    let entry = &upload.custom_meshes[0];
    assert_eq!(
        entry.skinning,
        SkinningMode::Gpu,
        "valid skin stages the palette"
    );
    let palette = entry
        .joint_palette
        .as_ref()
        .expect("GPU entry carries the palette");
    assert_eq!(palette.len(), 2, "two-bone skeleton stages two joints");
    assert_eq!(entry.instance.model_matrix, Mat4::IDENTITY);
    assert_eq!(entry.instance.normal_matrix, Mat4::IDENTITY);
    assert_eq!(
        entry.instance.material_index,
        ornis_render::MaterialIdx::from_raw(0)
    );
    assert_eq!(upload.materials.len(), 1);
    // Hand-computed skinning (`M1 = T(1,0,0)·Rz(90°)`, identity binds):
    // v0 rides joint 1 fully, v1 stays on the root, v2 blends half.
    let vertices = &entry.vertices;
    assert_eq!(vertices.len(), 3);
    assert!(
        close(vertices[0].position, Vec3::new(1.0, 1.0, 0.0)),
        "v0 skinned to {:?}",
        vertices[0].position
    );
    assert!(
        close(vertices[1].position, Vec3::ZERO),
        "v1 stays on the root"
    );
    assert!(
        close(vertices[2].position, Vec3::new(1.0, 0.5, 0.0)),
        "v2 blends half"
    );
    // Normals survive the Z spin; uvs/indices pass through untouched.
    for vertex in vertices {
        assert!(close(vertex.normal, Vec3::Z));
    }
    assert_eq!(
        vertices.iter().map(|vertex| vertex.uv).collect::<Vec<_>>(),
        vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    );
    assert_eq!(entry.indices, vec![0, 1, 2]);
}

#[test]
fn corrupt_skin_lane_skips_with_counter() {
    // Same pair, but the mesh lane arrays disagree (weights too long):
    // the entity skips with `skipped_bad_skin` — no stub, no classic
    // bind-pose fallback for a claimed skin.
    let mut engine = skel_engine();
    let mesh_entity = spawn_skinned_pair(&mut engine);
    {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        let mut lane = store.write_lane::<SkinnedMesh>().expect("lane set");
        let mesh = lane.get_mut(mesh_entity).expect("mesh placed");
        mesh.weights.push([1.0, 0.0, 0.0, 0.0]);
    }

    engine.run_frame(0.5);

    let (upload, stats) =
        ornis_render::extract_render_data_with_stats(engine.world().store().expect("store"));
    assert!(upload.custom_meshes.is_empty());
    assert!(upload.instances.is_empty());
    assert_eq!(stats.skipped_bad_skin, 1);
}
