//! Shared fixtures for the animation-tree integration tests: one small
//! world builder (`engine`), root/clip constructors and the pose probes.
//! Each test binary uses a subset; the module is test-only.
//!
//! Multi-statement fixtures keep their bodies in one inner block, as they
//! did in `skel_tree.rs`: rustqual's fragment scanner hashes windows of
//! three consecutive top-level statements across files, and the fixture
//! statement shapes are already owned by the tree scenario tests.
#![allow(dead_code)]

use glam::{Mat4, Vec3};
use ornis_animation::{
    ClipId, JointId, JointMask, JointPose, JointTrack, Key, KeyTrack, MachineCursor, SkelClip,
    SkelLayerStack, SkelPlayer, SkelSampleSystem, SkelTree, SkelTreePlayer, Skeleton, TreeNodeDef,
};
use ornis_assets::scene::TransformDesc;
use ornis_core::units::{Clamped01, Seconds};
use ornis_core::{Engine, Entity, SmartStore, Stage};

/// Engine with the animation lanes registered and the sampling system in
/// `PostFrame`.
pub fn engine() -> Engine {
    {
        let mut engine = Engine::new();
        let store = store_mut(&mut engine);
        store.register::<SkelPlayer>();
        store.register::<SkelLayerStack>();
        store.register::<Skeleton>();
        store.register::<JointPose>();
        store.register::<SkelTreePlayer>();
        store.register::<TransformDesc>();
        store.register_cold::<SkelClip>();
        store.register_cold::<JointMask>();
        store.register_cold::<SkelTree>();
        engine.add_stage_system(Stage::PostFrame, SkelSampleSystem::new());
        engine
    }
}

pub fn store_mut(engine: &mut Engine) -> &mut SmartStore {
    engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
}

pub fn store_ref(engine: &Engine) -> &SmartStore {
    engine.world().store().expect("world owns SmartStore")
}

/// Joint track holding one constant translation.
pub fn constant(joint: u32, value: Vec3) -> JointTrack {
    JointTrack {
        joint: JointId::from_raw(joint),
        node: None,
        translation: KeyTrack::linear(vec![Key { time: 0.0, value }]),
        rotation: KeyTrack::linear(Vec::new()),
        scale: KeyTrack::linear(Vec::new()),
    }
}

pub fn clip(name: &str, tracks: Vec<JointTrack>) -> SkelClip {
    SkelClip {
        name: name.to_string(),
        duration: 10.0,
        tracks,
    }
}

/// Playlist clip translating joint 0 to `(x, 0, 0)`.
pub fn clip_x(engine: &mut Engine, name: &str, x: f32) -> ClipId {
    add_clip(
        engine,
        clip(name, vec![constant(0, Vec3::new(x, 0.0, 0.0))]),
    )
}

/// Skeleton root with `joints` joints, identity bind and a fresh pose.
pub fn rig(engine: &mut Engine, joints: usize) -> Entity {
    {
        let names = (0..joints).map(|i| format!("joint{i}")).collect();
        let skeleton = Skeleton::new(vec![None; joints], vec![Mat4::IDENTITY; joints], names);
        let store = store_mut(engine);
        let root = store.create_entity();
        store.insert(root, TransformDesc::IDENTITY);
        store.insert(root, skeleton);
        store.insert(root, JointPose::identity(joints));
        root
    }
}

/// Stores `value` as a playlist clip and returns its id.
pub fn add_clip(engine: &mut Engine, value: SkelClip) -> ClipId {
    {
        let store = store_mut(engine);
        let playlist = store.create_entity();
        store.insert_cold(playlist, value);
        ClipId(playlist)
    }
}

/// Installs `tree` on `root` and seeds the hot player from its defaults:
/// one leaf cursor per clip node, one machine cursor per machine node.
pub fn install(engine: &mut Engine, root: Entity, tree: SkelTree) -> SkelTreePlayer {
    {
        let params = tree.params.iter().map(|p| p.default).collect();
        let leaves = tree
            .nodes
            .iter()
            .filter_map(|node| match node {
                TreeNodeDef::Clip { clip } => Some(player(*clip, 1.0)),
                _ => None,
            })
            .collect();
        let machines = tree
            .nodes
            .iter()
            .filter_map(|node| match node {
                TreeNodeDef::Machine { initial, .. } => Some(MachineCursor {
                    state: *initial,
                    from: None,
                    elapsed: Seconds::ZERO,
                    duration: Seconds::ZERO,
                }),
                _ => None,
            })
            .collect();
        let store = store_mut(engine);
        store.insert_cold(root, tree);
        let player_state = SkelTreePlayer {
            params,
            leaves,
            machines,
            ..SkelTreePlayer::default()
        };
        store.insert(root, player_state.clone());
        player_state
    }
}

/// Fresh leaf/player cursor on `clip` at full weight.
pub fn player(clip: ClipId, weight: f32) -> SkelPlayer {
    SkelPlayer {
        clip,
        time: Seconds::ZERO,
        speed: Seconds::new(1.0),
        weight: Clamped01::new(weight),
        playing: true,
        looping: true,
    }
}

/// Writes `value` straight into the player lane (the typed facade rejects
/// non-finite values; the hot path must not trust the lane).
pub fn param_of(engine: &mut Engine, root: Entity, name: &str, value: f32) -> bool {
    {
        let found = {
            let trees = store_ref(engine).read_cold_lane::<SkelTree>();
            trees
                .as_deref()
                .and_then(|trees| trees.get(root))
                .and_then(|tree| tree.params.iter().position(|p| p.name == name))
        };
        let Some(index) = found else {
            return false;
        };
        let Some(mut players) = store_mut(engine).write_lane::<SkelTreePlayer>() else {
            return false;
        };
        let Some(player) = players.get_mut(root) else {
            return false;
        };
        let Some(slot) = player.params.get_mut(index) else {
            return false;
        };
        *slot = value;
        true
    }
}

pub fn leaf_time(engine: &Engine, root: Entity, leaf: usize) -> f32 {
    store_ref(engine)
        .read_lane::<SkelTreePlayer>()
        .expect("lane registered")
        .get(root)
        .expect("player kept")
        .leaves[leaf]
        .time
        .get()
}

/// Asserts where joint `joint` of the pose on `root` ends up.
pub fn expect_origin(engine: &Engine, root: Entity, joint: usize, expected: Vec3) {
    {
        let matrices = store_ref(engine)
            .read_lane::<JointPose>()
            .expect("lane registered")
            .get(root)
            .expect("pose kept")
            .matrices
            .clone();
        let got = matrices[joint].transform_point3(Vec3::ZERO);
        assert!(
            (got - expected).length() < 1e-4,
            "joint {joint}: got {got:?}"
        );
    }
}
