//! Phase B acceptance (design `docs/animation-design.md` §2 and §5):
//! `skel_sample` resolves joint-local matrices to model space along
//! `parents`, `skel_skin_cpu` blends a two-bone chain with known matrices,
//! bad skin skips the entity with a counter, and `MAX_JOINTS` caps at 128.
//!
//! Mirrors `anim_object.rs`: the engine runs both systems on PostFrame
//! behind the explicit `skel_sample → skel_skin_cpu` edge, while counter
//! assertions drive the public `run_*` entry points over hand-built
//! resources (purer verdicts, no frame clock involved).

use std::any::TypeId;
use std::f32::consts::FRAC_PI_2;

use glam::{Mat4, Quat, Vec3};
use ornis_animation::{
    ClipId, JointId, JointPose, JointTrack, Key, KeyTrack, MAX_JOINTS, SkelClip, SkelError,
    SkelPlayer, SkelSampleStats, SkelSampleSystem, SkelSkinStats, SkelSkinSystem, Skeleton,
    SkinnedMesh, run_skel_sample, run_skel_skin,
};
use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
use ornis_core::units::{Clamped01, Seconds};
use ornis_core::{Engine, Entity, Resources, SmartStore, Stage, System, Time};
use ornis_render::extract_render_data;

/// Registers the skeletal lanes (edge tests control system order).
fn register_skel_lanes(engine: &mut Engine) {
    let store = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore");
    store.register::<SkelPlayer>();
    store.register::<Skeleton>();
    store.register::<JointPose>();
    store.register::<TransformDesc>();
    store.register::<SkinnedMesh>();
    store.register_cold::<SkelClip>();
}

/// Engine with both phase B systems on PostFrame behind the explicit edge.
fn skel_engine() -> Engine {
    let mut engine = Engine::new();
    register_skel_lanes(&mut engine);
    engine.add_stage_system(Stage::PostFrame, SkelSampleSystem::new());
    engine.add_stage_system(Stage::PostFrame, SkelSkinSystem::new());
    engine
        .stage_schedule_mut(Stage::PostFrame)
        .try_order_before("skel_sample", "skel_skin_cpu")
        .expect("explicit edge skel_sample -> skel_skin_cpu");
    engine
}

fn identity_transform() -> TransformDesc {
    TransformDesc::IDENTITY
}

fn vec_keys(value: Vec3) -> KeyTrack<Vec3> {
    KeyTrack::linear(vec![Key { time: 0.0, value }])
}

fn quat_keys(value: Quat) -> KeyTrack<Quat> {
    KeyTrack::linear(vec![Key { time: 0.0, value }])
}

fn no_translation() -> KeyTrack<Vec3> {
    KeyTrack::linear(Vec::new())
}

fn no_rotation() -> KeyTrack<Quat> {
    KeyTrack::linear(Vec::new())
}

fn no_scale() -> KeyTrack<Vec3> {
    KeyTrack::linear(Vec::new())
}

/// Two-joint skeleton: root plus one child (bind matrices identity).
fn two_bone_skeleton() -> Skeleton {
    Skeleton::new(
        vec![None, Some(JointId::from_raw(0))],
        vec![Mat4::IDENTITY, Mat4::IDENTITY],
        vec!["root".to_string(), "tip".to_string()],
    )
}

/// Skeleton root with identity placement, identity pose and a playing cursor.
fn spawn_root(engine: &mut Engine, clip: SkelClip) -> Entity {
    let store = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore");
    let root = store.create_entity();
    store.insert(root, identity_transform());
    store.insert(root, two_bone_skeleton());
    store.insert(root, JointPose::identity(2));
    store.insert_cold(root, clip);
    store.insert(
        root,
        SkelPlayer {
            clip: ClipId(root),
            time: Seconds::ZERO,
            speed: Seconds::new(1.0),
            weight: Clamped01::ONE,
            playing: true,
            looping: true,
        },
    );
    root
}

/// Mesh entity skinned against `root` (no placement of its own: the skin
/// system reads bind data plus the root pose only).
#[allow(clippy::too_many_arguments)]
fn spawn_mesh(
    engine: &mut Engine,
    root: Entity,
    joints: Vec<[u16; 4]>,
    weights: Vec<[f32; 4]>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
) -> Entity {
    let store = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore");
    let mesh = store.create_entity();
    store.insert(
        mesh,
        SkinnedMesh::new(
            root,
            joints,
            weights,
            positions,
            normals,
            vec![[0.0, 0.0]; 3],
            vec![0, 1, 2],
        ),
    );
    mesh
}

fn pose_of(engine: &Engine, root: Entity) -> Vec<Mat4> {
    engine
        .world()
        .store()
        .expect("world owns SmartStore")
        .read_lane::<JointPose>()
        .expect("lane registered")
        .get(root)
        .expect("pose kept")
        .matrices
        .clone()
}

fn buffers_of(engine: &Engine, mesh: Entity) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let store = engine.world().store().expect("world owns SmartStore");
    let lane = store.read_lane::<SkinnedMesh>().expect("lane registered");
    let mesh = lane.get(mesh).expect("mesh kept");
    (mesh.skinned_positions.clone(), mesh.skinned_normals.clone())
}

fn close(actual: Vec3, expected: Vec3) -> bool {
    (actual - expected).length() < 1e-4
}

#[test]
fn sample_resolves_parent_chain() {
    // Joint 0 spins 90° about Z, joint 1 offsets +X: the child model must
    // carry the parent rotation (local +X offset lands on +Y).
    let mut engine = skel_engine();
    let clip = SkelClip {
        name: String::new(),
        duration: 1.0,
        tracks: vec![
            JointTrack {
                joint: JointId::from_raw(0),
                node: None,
                translation: no_translation(),
                rotation: quat_keys(Quat::from_rotation_z(FRAC_PI_2)),
                scale: no_scale(),
            },
            JointTrack {
                joint: JointId::from_raw(1),
                node: None,
                translation: vec_keys(Vec3::new(1.0, 0.0, 0.0)),
                rotation: no_rotation(),
                scale: no_scale(),
            },
        ],
    };
    let root = spawn_root(&mut engine, clip);

    engine.run_frame(0.5);

    let pose = pose_of(&engine, root);
    assert_eq!(pose.len(), 2);
    assert!(
        close(pose[0].transform_point3(Vec3::X), Vec3::Y),
        "joint 0 must spin +X to +Y"
    );
    assert!(
        close(pose[1].transform_point3(Vec3::ZERO), Vec3::Y),
        "child origin must inherit the parent spin, got {:?}",
        pose[1].transform_point3(Vec3::ZERO)
    );
    assert!(
        close(pose[1].transform_point3(Vec3::X), Vec3::new(0.0, 2.0, 0.0)),
        "child +X rides its offset through the parent spin"
    );
}

#[test]
fn skin_two_bone_chain_with_known_matrices() {
    // Joint 1 sits at +X and spins 90° about Z (`M1 = T·R`); bind equals
    // model space (identity inverse bind), so hand-computed targets apply.
    let mut engine = skel_engine();
    let clip = SkelClip {
        name: String::new(),
        duration: 1.0,
        tracks: vec![JointTrack {
            joint: JointId::from_raw(1),
            node: None,
            translation: vec_keys(Vec3::new(1.0, 0.0, 0.0)),
            rotation: quat_keys(Quat::from_rotation_z(FRAC_PI_2)),
            scale: no_scale(),
        }],
    };
    let root = spawn_root(&mut engine, clip);
    let mesh = spawn_mesh(
        &mut engine,
        root,
        vec![[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
        vec![
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    );

    engine.run_frame(0.5);

    // Cursor advances on the wrapped clip; pose lands exactly on M1.
    let store = engine.world().store().expect("world owns SmartStore");
    let players = store.read_lane::<SkelPlayer>().expect("lane registered");
    let player = *players.get(root).expect("player kept");
    assert!((player.time.get() - 0.5).abs() < 1e-6);
    let pose = pose_of(&engine, root);
    assert!(close(
        pose[1].transform_point3(Vec3::ZERO),
        Vec3::new(1.0, 0.0, 0.0)
    ));
    assert!(close(
        pose[1].transform_point3(Vec3::X),
        Vec3::new(1.0, 1.0, 0.0)
    ));

    // v0 rides joint 1 fully, v1 stays on the identity root, v2 blends half.
    let (positions, normals) = buffers_of(&engine, mesh);
    assert!(close(
        Vec3::from_array(positions[0]),
        Vec3::new(1.0, 1.0, 0.0)
    ));
    assert!(close(Vec3::from_array(positions[1]), Vec3::ZERO));
    assert!(close(
        Vec3::from_array(positions[2]),
        Vec3::new(1.0, 0.5, 0.0)
    ));
    // Z normals survive the Z spin untouched.
    for normal in &normals {
        assert!(close(Vec3::from_array(*normal), Vec3::Z));
    }
}

/// One mesh component with single-joint bind data (identity skin).
fn simple_mesh(skeleton: Entity) -> SkinnedMesh {
    SkinnedMesh::new(
        skeleton,
        vec![[0, 0, 0, 0]],
        vec![[1.0, 0.0, 0.0, 0.0]],
        vec![[2.0, 0.0, 0.0]],
        vec![[0.0, 1.0, 0.0]],
        vec![[0.0, 0.0]],
        vec![0, 0, 0],
    )
}

#[test]
fn bad_skin_skips_entity_and_counts() {
    // One good mesh, one pointing at a skeleton-less entity, one with
    // mismatched bind arrays: only the good mesh skins, both bad ones
    // count — and their buffers keep the pre-run sentinel (no partial
    // writes, no stubs).
    let mut store = SmartStore::new();
    store.register::<Skeleton>();
    store.register::<JointPose>();
    store.register::<SkinnedMesh>();
    let root = store.create_entity();
    store.insert(
        root,
        Skeleton::new(vec![None], vec![Mat4::IDENTITY], vec!["root".to_string()]),
    );
    store.insert(root, JointPose::identity(1));
    let ghost = store.create_entity();

    let good = store.create_entity();
    store.insert(good, simple_mesh(root));
    let missing = store.create_entity();
    store.insert(missing, simple_mesh(ghost));
    let broken = store.create_entity();
    let mut broken_mesh = simple_mesh(root);
    broken_mesh.weights.push([1.0, 0.0, 0.0, 0.0]);
    store.insert(broken, broken_mesh);

    // Sentinels prove the skip writes nothing.
    {
        let mut meshes = store.write_lane::<SkinnedMesh>().expect("lane set");
        for entity in [missing, broken] {
            let mesh = meshes.get_mut(entity).expect("mesh placed");
            mesh.skinned_positions = vec![[9.0, 9.0, 9.0]];
            mesh.skinned_normals = vec![[9.0, 9.0, 9.0]];
        }
    }

    let mut resources = Resources::new();
    resources.insert(store);
    resources.insert(Time::new());
    let stats = run_skel_skin(&resources);
    assert_eq!(
        stats,
        SkelSkinStats {
            skinned: 1,
            skinned_vertices: 1,
            skipped_bad_skin: 2,
        }
    );

    let store = resources.get::<SmartStore>().expect("store resource kept");
    let meshes = store.read_lane::<SkinnedMesh>().expect("lane kept");
    let good = meshes.get(good).expect("good mesh kept");
    assert_eq!(good.skinned_positions, vec![[2.0, 0.0, 0.0]]);
    for entity in [missing, broken] {
        let mesh = meshes.get(entity).expect("bad mesh kept");
        assert_eq!(mesh.skinned_positions, vec![[9.0, 9.0, 9.0]]);
        assert_eq!(mesh.skinned_normals, vec![[9.0, 9.0, 9.0]]);
    }
}

#[test]
fn joint_cap_is_128() {
    assert_eq!(MAX_JOINTS, 128, "cap is a design constant");
    let names = |count: usize| vec!["joint".to_string(); count];
    let bones = |count: usize| vec![Mat4::IDENTITY; count];
    let roots = |count: usize| vec![None; count];
    assert_eq!(
        Skeleton::new(roots(128), bones(128), names(128)).validate(),
        Ok(128),
        "exactly 128 joints sample fine"
    );
    assert_eq!(
        Skeleton::new(roots(129), bones(129), names(129)).validate(),
        Err(SkelError::TooManyJoints),
        "129 joints reject, never silently truncate"
    );

    // System level: the over-cap skeleton skips with a counter, the pose
    // lane keeps its previous content (no stub write).
    let mut store = SmartStore::new();
    store.register::<SkelPlayer>();
    store.register::<Skeleton>();
    store.register::<JointPose>();
    store.register_cold::<SkelClip>();
    let root = store.create_entity();
    store.insert(root, Skeleton::new(roots(129), bones(129), names(129)));
    store.insert(
        root,
        JointPose {
            matrices: Vec::new(),
        },
    );
    store.insert_cold(
        root,
        SkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: Vec::new(),
        },
    );
    store.insert(
        root,
        SkelPlayer {
            clip: ClipId(root),
            time: Seconds::ZERO,
            speed: Seconds::new(1.0),
            weight: Clamped01::ONE,
            playing: true,
            looping: true,
        },
    );
    let mut resources = Resources::new();
    resources.insert(store);
    resources.insert(Time::new());
    let stats = run_skel_sample(&resources);
    assert_eq!(
        stats,
        SkelSampleStats {
            sampled: 0,
            skipped_bad_skin: 1,
        }
    );

    // A cursor without a clip is the same honesty case: skip and count.
    {
        let store = resources
            .get_mut::<SmartStore>()
            .expect("store resource kept");
        let ghost = Entity::new(999);
        let orphan = store.create_entity();
        store.insert(
            orphan,
            SkelPlayer {
                clip: ClipId(ghost),
                time: Seconds::ZERO,
                speed: Seconds::new(1.0),
                weight: Clamped01::ONE,
                playing: true,
                looping: true,
            },
        );
    }
    let stats = run_skel_sample(&resources);
    assert_eq!(
        stats,
        SkelSampleStats {
            sampled: 0,
            skipped_bad_skin: 2,
        },
        "over-cap root plus clip-less cursor both skip"
    );
    let store = resources.get::<SmartStore>().expect("store resource kept");
    let poses = store.read_lane::<JointPose>().expect("lane kept");
    let pose = poses.get(root).expect("pose kept");
    assert!(pose.matrices.is_empty(), "no stub pose is ever written");
}

#[test]
fn classic_custom_entries_report_unskinned() {
    // The only extraction path touched by phase B: classic bind-pose soups
    // are never pre-skinned, so the new mode reads CPU with no palette.
    let mut store = SmartStore::new();
    let entity = store.create_entity();
    store.insert(entity, TransformDesc::IDENTITY);
    store.insert(
        entity,
        MeshDesc::Custom {
            positions: vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        },
    );
    store.insert(
        entity,
        MaterialDesc::Dielectric {
            base_color: [0.8, 0.2, 0.2],
            roughness: Clamped01::new(0.4),
            emission: [0.0, 0.0, 0.0],
            metallic: ornis_core::Metallic::new(0.0),
        },
    );

    let upload = extract_render_data(&store);
    assert_eq!(upload.custom_meshes.len(), 1);
    assert_eq!(
        upload.custom_meshes[0].skinning,
        ornis_animation::SkinningMode::Cpu,
        "classic soup path is never pre-skinned"
    );
    assert!(
        upload.custom_meshes[0].joint_palette.is_none(),
        "classic soup stages no palette"
    );
}

#[test]
fn systems_advertise_names_accesses_and_edge() {
    assert_eq!(SkelSampleSystem::new().name(), "skel_sample");
    assert_eq!(SkelSkinSystem::new().name(), "skel_skin_cpu");

    let sample = SkelSampleSystem::new().access();
    for lane in [
        TypeId::of::<SkelPlayer>(),
        TypeId::of::<SkelClip>(),
        TypeId::of::<Skeleton>(),
        TypeId::of::<TransformDesc>(),
        TypeId::of::<JointPose>(),
        TypeId::of::<ornis_core::Transform>(),
        TypeId::of::<ornis_core::GlobalTransform>(),
    ] {
        let declared = sample.reads_lanes.contains(&lane) || sample.writes_lanes.contains(&lane);
        assert!(declared, "sample lane {lane:?} must be declared");
    }
    // Cursor advance writes the player lane; poses are the output.
    assert!(sample.writes_lanes.contains(&TypeId::of::<SkelPlayer>()));
    assert!(sample.writes_lanes.contains(&TypeId::of::<JointPose>()));

    let skin = SkelSkinSystem::new().access();
    for lane in [
        TypeId::of::<Skeleton>(),
        TypeId::of::<JointPose>(),
        TypeId::of::<SkinnedMesh>(),
    ] {
        let declared = skin.reads_lanes.contains(&lane) || skin.writes_lanes.contains(&lane);
        assert!(declared, "skin lane {lane:?} must be declared");
    }
    assert!(skin.writes_lanes.contains(&TypeId::of::<SkinnedMesh>()));

    // The edge the session wiring pins runs forward.
    let mut engine = Engine::new();
    register_skel_lanes(&mut engine);
    engine.add_stage_system(Stage::PostFrame, SkelSampleSystem::new());
    engine.add_stage_system(Stage::PostFrame, SkelSkinSystem::new());
    engine
        .stage_schedule_mut(Stage::PostFrame)
        .try_order_before("skel_sample", "skel_skin_cpu")
        .expect("explicit edge skel_sample -> skel_skin_cpu");
    let mermaid = engine.stage_schedule(Stage::PostFrame).mermaid();
    assert!(mermaid.contains("skel_sample"));
    assert!(mermaid.contains("skel_skin_cpu"));
}

/// A non-looping cursor clamps at the clip duration instead of wrapping.
#[test]
fn non_looping_cursor_clamps_at_duration() {
    let mut engine = skel_engine();
    let root = spawn_root(
        &mut engine,
        SkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: Vec::new(),
        },
    );
    {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        let mut players = store.write_lane::<SkelPlayer>().expect("players");
        players.get_mut(root).expect("player").looping = false;
    }
    engine.run_frame(2.0);
    let time = engine
        .world()
        .store()
        .expect("world owns SmartStore")
        .read_lane::<SkelPlayer>()
        .expect("players")
        .get(root)
        .expect("player")
        .time;
    assert!((time.get() - 1.0).abs() < 1e-5);
}
