//! Hierarchy playback: a track on a mesh-less node writes that node's local
//! [`Transform`](ornis_core::Transform), and [`propagate_registered`](ornis_core::propagate_registered)
//! moves the child mesh's [`GlobalTransform`](ornis_core::GlobalTransform).
//!
//! Object tracks and skeletal joints share the rule. The child keeps its
//! identity local pose; the motion comes from the parent.

use glam::{Mat4, Vec3};
use ornis_animation::{
    AnimClip, AnimPlayer, AnimSampleSystem, AnimTrack, ClipId, JointId, JointMask, JointPose,
    JointTrack, JointTrs, Key, KeyTrack, MaskId, SkelClip, SkelLayer, SkelPlayer, SkelSampleSystem,
    Skeleton,
};
use ornis_assets::scene::MeshDesc;
use ornis_core::units::{Clamped01, PositiveF32, Seconds};
use ornis_core::{Engine, Entity, GlobalTransform, SmartStore, Stage, Transform, set_parent};

fn engine_with(sample: impl ornis_core::System + 'static) -> Engine {
    let mut engine = Engine::new();
    engine.add_stage_system(Stage::PostFrame, sample);
    engine
}

fn store_mut(engine: &mut Engine) -> &mut SmartStore {
    engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
}

/// Mesh-less parent plus a child mesh at local identity.
fn parent_and_mesh(store: &mut SmartStore) -> (Entity, Entity) {
    let parent = store.create_entity();
    store.insert(
        parent,
        Transform {
            translation: Vec3::ZERO,
            rotation: ornis_core::UnitQuat::IDENTITY,
            scale: Vec3::new(2.0, 2.0, 2.0),
        },
    );
    store.insert(parent, GlobalTransform::IDENTITY);
    let mesh = store.create_entity();
    store.insert(mesh, Transform::IDENTITY);
    store.insert(mesh, GlobalTransform::IDENTITY);
    store.insert(
        mesh,
        MeshDesc::Sphere {
            radius: PositiveF32::expect_valid(1.0),
            segments: 8,
            rings: 4,
        },
    );
    set_parent(store, mesh, parent).expect("mesh under the node");
    (parent, mesh)
}

fn global_of(engine: &Engine, entity: Entity) -> GlobalTransform {
    *engine
        .world()
        .store()
        .expect("store")
        .read_lane::<GlobalTransform>()
        .expect("globals")
        .get(entity)
        .expect("global")
}

fn local_of(engine: &Engine, entity: Entity) -> Transform {
    *engine
        .world()
        .store()
        .expect("store")
        .read_lane::<Transform>()
        .expect("locals")
        .get(entity)
        .expect("local")
}

#[test]
fn object_track_on_meshless_parent_moves_child_mesh() {
    let mut engine = engine_with(AnimSampleSystem::<ornis_animation::NoPhysics>::new());
    let (parent, mesh) = {
        let store = store_mut(&mut engine);
        let (parent, mesh) = parent_and_mesh(store);
        store.register::<AnimPlayer>();
        store.register_cold::<AnimClip>();
        store.insert_cold(
            parent,
            AnimClip {
                name: "slide".to_string(),
                duration: 1.0,
                looping: false,
                tracks: vec![AnimTrack {
                    entity: parent,
                    rest: JointTrs::IDENTITY,
                    translation: KeyTrack::linear(vec![
                        Key {
                            time: 0.0,
                            value: Vec3::ZERO,
                        },
                        Key {
                            time: 1.0,
                            value: Vec3::new(10.0, 0.0, 0.0),
                        },
                    ]),
                    rotation: KeyTrack::linear(Vec::new()),
                    scale: KeyTrack::linear(Vec::new()),
                }],
            },
        );
        store.insert(
            parent,
            AnimPlayer {
                clip: ClipId(parent),
                time: 0.0,
                speed: 1.0,
                weight: 1.0,
                playing: true,
            },
        );
        assert!(
            store
                .read_lane::<MeshDesc>()
                .is_none_or(|lane| lane.get(parent).is_none()),
            "the animated node has no mesh"
        );
        (parent, mesh)
    };

    engine.run_frame(0.5);

    let parent_local = local_of(&engine, parent);
    assert!(
        (parent_local.translation.x - 5.0).abs() < 1e-4,
        "parent local x is halfway, got {}",
        parent_local.translation.x
    );
    assert!(
        (parent_local.scale.x - 2.0).abs() < 1e-4,
        "absent scale channel keeps the node scale"
    );
    let child_local = local_of(&engine, mesh);
    assert_eq!(
        child_local.translation,
        Vec3::ZERO,
        "child local stays identity"
    );
    let child_global = global_of(&engine, mesh);
    assert!(
        (child_global.translation.x - 5.0).abs() < 1e-4,
        "child world x follows the parent, got {}",
        child_global.translation.x
    );
    assert!(
        (child_global.scale.x - 2.0).abs() < 1e-4,
        "child world scale inherits the parent, got {}",
        child_global.scale
    );
}

#[test]
fn joint_track_on_meshless_node_moves_child_mesh() {
    let mut engine = engine_with(SkelSampleSystem::new());
    let (node, mesh, root) = {
        let store = store_mut(&mut engine);
        let (node, mesh) = parent_and_mesh(store);
        let root = store.create_entity();
        store.register::<SkelPlayer>();
        store.register::<Skeleton>();
        store.register::<JointPose>();
        store.register_cold::<SkelClip>();
        store.insert(
            root,
            Skeleton::new(vec![None], vec![Mat4::IDENTITY], vec!["root".to_string()]),
        );
        store.insert(root, JointPose::identity(1));
        store.insert_cold(
            root,
            SkelClip {
                name: "joint".to_string(),
                duration: 1.0,
                tracks: vec![JointTrack {
                    joint: JointId::from_raw(0),
                    node: Some(node),
                    translation: KeyTrack::linear(vec![
                        Key {
                            time: 0.0,
                            value: Vec3::ZERO,
                        },
                        Key {
                            time: 1.0,
                            value: Vec3::new(8.0, 0.0, 0.0),
                        },
                    ]),
                    rotation: KeyTrack::linear(Vec::new()),
                    scale: KeyTrack::linear(Vec::new()),
                }],
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
                looping: false,
            },
        );
        (node, mesh, root)
    };

    engine.run_frame(0.5);

    let node_local = local_of(&engine, node);
    assert!(
        (node_local.translation.x - 4.0).abs() < 1e-4,
        "joint node local x is halfway, got {}",
        node_local.translation.x
    );
    assert!(
        (node_local.scale.x - 2.0).abs() < 1e-4,
        "empty joint scale channel leaves the node scale"
    );
    assert_eq!(local_of(&engine, mesh).translation, Vec3::ZERO);
    let child_global = global_of(&engine, mesh);
    assert!(
        (child_global.translation.x - 4.0).abs() < 1e-4,
        "child world x follows the joint node, got {}",
        child_global.translation.x
    );
    let pose = engine
        .world()
        .store()
        .expect("store")
        .read_lane::<JointPose>()
        .expect("poses")
        .get(root)
        .expect("pose")
        .matrices[0];
    assert!(
        (pose.w_axis.x - 4.0).abs() < 1e-4,
        "joint pose still samples, got {}",
        pose.w_axis.x
    );
}

/// Joint track holding one constant translation, bound to `node` when given.
fn held_joint(joint: u32, node: Option<Entity>, value: Vec3) -> JointTrack {
    JointTrack {
        joint: JointId::from_raw(joint),
        node,
        translation: KeyTrack::linear(vec![Key { time: 0.0, value }]),
        rotation: KeyTrack::linear(Vec::new()),
        scale: KeyTrack::linear(Vec::new()),
    }
}

/// One-second skeletal clip over `tracks`.
fn skel_clip(name: &str, tracks: Vec<JointTrack>) -> SkelClip {
    SkelClip {
        name: name.to_string(),
        duration: 1.0,
        tracks,
    }
}

/// Looping, playing cursor on `clip` at full weight.
fn skel_player(clip: ClipId) -> SkelPlayer {
    SkelPlayer {
        clip,
        time: Seconds::ZERO,
        speed: Seconds::new(1.0),
        weight: Clamped01::ONE,
        playing: true,
        looping: true,
    }
}

/// A layer alone drives joint 1, which the base clip never touches. Its
/// bound node must follow the blended joint, not keep its bind pose.
#[test]
fn layer_only_joint_moves_its_bound_node() {
    let mut engine = engine_with(SkelSampleSystem::new());
    let (node, root) = {
        let store = store_mut(&mut engine);
        let (node, _mesh) = parent_and_mesh(store);
        store.register::<SkelPlayer>();
        store.register::<SkelLayer>();
        store.register::<Skeleton>();
        store.register::<JointPose>();
        store.register_cold::<SkelClip>();
        store.register_cold::<JointMask>();
        let root = store.create_entity();
        let skeleton = Skeleton::new(
            vec![None, Some(JointId::from_raw(0))],
            vec![Mat4::IDENTITY; 2],
            vec!["hip".to_string(), "leg".to_string()],
        );
        store.insert(root, skeleton);
        store.insert(root, JointPose::identity(2));

        let base = store.create_entity();
        let idle = skel_clip("idle", vec![held_joint(0, None, Vec3::ZERO)]);
        store.insert_cold(base, idle);

        let overlay = store.create_entity();
        let wave_joint = held_joint(1, Some(node), Vec3::new(4.0, 0.0, 0.0));
        store.insert_cold(overlay, skel_clip("wave", vec![wave_joint]));

        let mask = store.create_entity();
        let mask_weights = JointMask {
            name: "all".to_string(),
            weights: vec![Clamped01::ONE, Clamped01::ONE],
        };
        store.insert_cold(mask, mask_weights);

        store.insert(root, skel_player(ClipId(base)));
        let layer = SkelLayer {
            player: skel_player(ClipId(overlay)),
            mask: MaskId(mask),
        };
        store.insert(root, layer);
        (node, root)
    };

    engine.run_frame(0.5);

    let node_local = local_of(&engine, node);
    assert!(
        (node_local.translation.x - 4.0).abs() < 1e-4,
        "layer-only joint moves its node, got {}",
        node_local.translation.x
    );
    let pose = engine
        .world()
        .store()
        .expect("store")
        .read_lane::<JointPose>()
        .expect("poses")
        .get(root)
        .expect("pose")
        .matrices[1];
    assert!(
        (pose.w_axis.x - 4.0).abs() < 1e-4,
        "joint 1 pose follows the layer, got {}",
        pose.w_axis.x
    );
}
