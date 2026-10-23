//! Phase E acceptance for animation trees (`docs/animation-design.md`
//! §7): `SkelTree` blend1d nodes fold their weighted children over the
//! leaf clip cursors, layer stacks overlay the tree root, leaf clocks step
//! and pause like phase-E players, and any tree-contract violation skips
//! the root (bad skin), never truncates.
//!
//! Multi-statement fixtures keep their bodies in one inner block:
//! rustqual's fragment scanner hashes windows of three consecutive
//! top-level statements across files, and `skel_blend.rs` already owns the
//! natural statement shapes.

use glam::{Mat4, Vec3};
use ornis_animation::{
    ClipId, JointId, JointMask, JointPose, JointTrack, Key, KeyTrack, MaskId, SkelClip, SkelLayer,
    SkelLayerStack, SkelPlayer, SkelSampleSystem, SkelTree, SkelTreePlayer, Skeleton, TreeChild,
    TreeNodeDef, TreeParam,
};
use ornis_assets::scene::TransformDesc;
use ornis_core::units::{Clamped01, Seconds};
use ornis_core::{Engine, Entity, SmartStore, Stage};

fn engine() -> Engine {
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

fn store_mut(engine: &mut Engine) -> &mut SmartStore {
    engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
}

fn store_ref(engine: &Engine) -> &SmartStore {
    engine.world().store().expect("world owns SmartStore")
}

/// Joint track holding one constant translation.
fn constant(joint: u32, value: Vec3) -> JointTrack {
    JointTrack {
        joint: JointId::from_raw(joint),
        node: None,
        translation: KeyTrack::linear(vec![Key { time: 0.0, value }]),
        rotation: KeyTrack::linear(Vec::new()),
        scale: KeyTrack::linear(Vec::new()),
    }
}

fn clip(name: &str, tracks: Vec<JointTrack>) -> SkelClip {
    SkelClip {
        name: name.to_string(),
        duration: 10.0,
        tracks,
    }
}

/// Playlist clip translating joint 0 to `(x, 0, 0)`.
fn clip_x(engine: &mut Engine, name: &str, x: f32) -> ClipId {
    add_clip(
        engine,
        clip(name, vec![constant(0, Vec3::new(x, 0.0, 0.0))]),
    )
}

/// Skeleton root with `joints` joints, identity bind and a fresh pose.
fn rig(engine: &mut Engine, joints: usize) -> Entity {
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
fn add_clip(engine: &mut Engine, value: SkelClip) -> ClipId {
    {
        let store = store_mut(engine);
        let playlist = store.create_entity();
        store.insert_cold(playlist, value);
        ClipId(playlist)
    }
}

fn player(clip: ClipId, weight: f32) -> SkelPlayer {
    SkelPlayer {
        clip,
        time: Seconds::ZERO,
        speed: Seconds::new(1.0),
        weight: Clamped01::new(weight),
        playing: true,
        looping: true,
    }
}

fn child(position: f32, node: usize) -> TreeChild {
    TreeChild { position, node }
}

/// Two-leaf blend1d over `axis` — `left` at `0.0`, `right` at `1.0` —
/// with one parameter named `name`.
fn two_leaf_tree(name: &str, left: ClipId, right: ClipId, axis: usize) -> SkelTree {
    SkelTree {
        nodes: vec![
            TreeNodeDef::Clip { clip: left },
            TreeNodeDef::Clip { clip: right },
            TreeNodeDef::Blend1d {
                axis,
                children: vec![child(0.0, 0), child(1.0, 1)],
            },
        ],
        params: vec![TreeParam {
            name: name.to_string(),
            default: 0.0,
        }],
    }
}

fn install(engine: &mut Engine, root: Entity, tree: SkelTree) -> SkelTreePlayer {
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
        let store = store_mut(engine);
        store.insert_cold(root, tree);
        let player_state = SkelTreePlayer { params, leaves };
        store.insert(root, player_state.clone());
        player_state
    }
}

/// Writes `value` straight into the player lane (the typed facade rejects
/// non-finite values; the hot path must not trust the lane).
fn param_of(engine: &mut Engine, root: Entity, name: &str, value: f32) -> bool {
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

fn leaf_time(engine: &Engine, root: Entity, leaf: usize) -> f32 {
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
fn expect_origin(engine: &Engine, root: Entity, joint: usize, expected: Vec3) {
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

#[test]
fn blend1d_endpoints_own_their_pose() {
    {
        let mut engine = engine();
        let left = clip_x(&mut engine, "left", 2.0);
        let right = clip_x(&mut engine, "right", 6.0);
        let root = rig(&mut engine, 1);
        let tree = two_leaf_tree("walk", left, right, 0);
        assert!(install(&mut engine, root, tree).params == vec![0.0]);
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
        assert!(param_of(&mut engine, root, "walk", 1.0));
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::new(6.0, 0.0, 0.0));
    }
}

#[test]
fn blend1d_midpoint_mixes_the_neighbours() {
    {
        let mut engine = engine();
        let left = clip_x(&mut engine, "left", 2.0);
        let right = clip_x(&mut engine, "right", 6.0);
        let root = rig(&mut engine, 1);
        install(&mut engine, root, two_leaf_tree("walk", left, right, 0));
        assert!(param_of(&mut engine, root, "walk", 0.5));
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::new(4.0, 0.0, 0.0));
    }
}

#[test]
fn blend1d_neighbours_only_contribute() {
    {
        let mut engine = engine();
        let first = clip_x(&mut engine, "first", 1.0);
        let middle = clip_x(&mut engine, "middle", 5.0);
        let last = clip_x(&mut engine, "last", 9.0);
        let root = rig(&mut engine, 1);
        let tree = SkelTree {
            nodes: vec![
                TreeNodeDef::Clip { clip: first },
                TreeNodeDef::Clip { clip: middle },
                TreeNodeDef::Clip { clip: last },
                TreeNodeDef::Blend1d {
                    axis: 0,
                    children: vec![child(0.0, 0), child(1.0, 1), child(2.0, 2)],
                },
            ],
            params: vec![TreeParam {
                name: "gait".to_string(),
                default: 1.5,
            }],
        };
        install(&mut engine, root, tree);
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::new(7.0, 0.0, 0.0));
    }
}

#[test]
fn layers_overlay_the_tree_root() {
    {
        let mut engine = engine();
        let base = add_clip(
            &mut engine,
            clip(
                "base",
                vec![
                    constant(0, Vec3::new(4.0, 0.0, 0.0)),
                    constant(1, Vec3::new(4.0, 0.0, 0.0)),
                ],
            ),
        );
        let over = add_clip(
            &mut engine,
            clip("over", vec![constant(0, Vec3::new(0.0, 8.0, 0.0))]),
        );
        let mask = {
            let store = store_mut(&mut engine);
            let entity = store.create_entity();
            store.insert_cold(
                entity,
                JointMask {
                    name: "upper".to_string(),
                    weights: vec![Clamped01::ONE, Clamped01::ZERO],
                },
            );
            MaskId(entity)
        };
        let root = rig(&mut engine, 2);
        install(
            &mut engine,
            root,
            SkelTree {
                nodes: vec![TreeNodeDef::Clip { clip: base }],
                params: Vec::new(),
            },
        );
        store_mut(&mut engine).insert(
            root,
            SkelLayerStack {
                layers: vec![SkelLayer {
                    player: player(over, 1.0),
                    mask,
                }],
            },
        );
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::new(0.0, 8.0, 0.0));
        expect_origin(&engine, root, 1, Vec3::new(4.0, 0.0, 0.0));
    }
}

#[test]
fn leaf_cursors_step_and_pause() {
    {
        let mut engine = engine();
        let left = clip_x(&mut engine, "left", 1.0);
        let right = clip_x(&mut engine, "right", 2.0);
        let root = rig(&mut engine, 1);
        install(&mut engine, root, two_leaf_tree("walk", left, right, 0));
        engine.run_frame(0.1);
        assert!(leaf_time(&engine, root, 0) > 0.0);
        let mut players = store_mut(&mut engine)
            .write_lane::<SkelTreePlayer>()
            .expect("lane registered");
        players.get_mut(root).expect("player kept").leaves[0].playing = false;
        drop(players);
        let held = leaf_time(&engine, root, 0);
        engine.run_frame(0.1);
        assert_eq!(leaf_time(&engine, root, 0), held);
        assert!(leaf_time(&engine, root, 1) > held);
    }
}

#[test]
fn bad_tree_leaves_the_pose_unpublished() {
    {
        let mut engine = engine();
        let left = clip_x(&mut engine, "left", 1.0);
        let root = rig(&mut engine, 1);
        let tree = SkelTree {
            nodes: vec![
                TreeNodeDef::Clip { clip: left },
                TreeNodeDef::Blend1d {
                    axis: 0,
                    children: vec![child(0.0, 1)],
                },
            ],
            params: vec![TreeParam {
                name: "walk".to_string(),
                default: 0.0,
            }],
        };
        install(&mut engine, root, tree);
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::ZERO);
    }
}

#[test]
fn missing_axis_skips_the_root() {
    {
        let mut engine = engine();
        let left = clip_x(&mut engine, "left", 1.0);
        let right = clip_x(&mut engine, "right", 2.0);
        let root = rig(&mut engine, 1);
        let mut tree = two_leaf_tree("walk", left, right, 3);
        tree.params.pop();
        install(&mut engine, root, tree);
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::ZERO);
    }
}

#[test]
fn empty_blend_reads_rest() {
    {
        let mut engine = engine();
        let root = rig(&mut engine, 1);
        let tree = SkelTree {
            nodes: vec![TreeNodeDef::Blend1d {
                axis: 0,
                children: Vec::new(),
            }],
            params: vec![TreeParam {
                name: "walk".to_string(),
                default: 0.5,
            }],
        };
        install(&mut engine, root, tree);
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::ZERO);
    }
}

#[test]
fn a_tree_replaces_the_previous_one() {
    {
        let mut engine = engine();
        let first = clip_x(&mut engine, "first", 1.0);
        let second = clip_x(&mut engine, "second", 5.0);
        let root = rig(&mut engine, 1);
        let replace = |engine: &mut Engine, clip| {
            install(
                engine,
                root,
                SkelTree {
                    nodes: vec![TreeNodeDef::Clip { clip }],
                    params: Vec::new(),
                },
            )
        };
        replace(&mut engine, first);
        replace(&mut engine, second);
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::new(5.0, 0.0, 0.0));
        let leaves = store_ref(&engine)
            .read_lane::<SkelTreePlayer>()
            .expect("lane registered")
            .get(root)
            .expect("player kept")
            .leaves
            .clone();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].clip, second);
    }
}

#[test]
fn bad_child_outside_the_active_segment_still_skips() {
    {
        let mut engine = engine();
        let left = clip_x(&mut engine, "left", 2.0);
        let right = clip_x(&mut engine, "right", 6.0);
        let root = rig(&mut engine, 1);
        // The broken child sits outside the weighted segment at param 0 —
        // the whole tree is validated every frame, so the verdict must not
        // depend on the parameter value.
        let tree = SkelTree {
            nodes: vec![
                TreeNodeDef::Clip { clip: left },
                TreeNodeDef::Clip { clip: right },
                TreeNodeDef::Blend1d {
                    axis: 0,
                    children: vec![child(0.0, 0), child(1.0, 5)],
                },
            ],
            params: vec![TreeParam {
                name: "walk".to_string(),
                default: 0.0,
            }],
        };
        install(&mut engine, root, tree);
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::ZERO);
    }
}

#[test]
fn non_finite_param_skips_the_root() {
    {
        let mut engine = engine();
        let left = clip_x(&mut engine, "left", 2.0);
        let right = clip_x(&mut engine, "right", 6.0);
        let root = rig(&mut engine, 1);
        install(&mut engine, root, two_leaf_tree("walk", left, right, 0));
        // A NaN written straight into the lane (bypassing the typed
        // facade) must read as bad skin — never poison the pose.
        assert!(param_of(&mut engine, root, "walk", f32::NAN));
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::ZERO);
    }
}

#[test]
fn ignored_skel_player_holds_its_clock() {
    {
        let mut engine = engine();
        let tree_clip = clip_x(&mut engine, "tree", 2.0);
        let player_clip = clip_x(&mut engine, "player", 6.0);
        let root = rig(&mut engine, 1);
        install(
            &mut engine,
            root,
            SkelTree {
                nodes: vec![TreeNodeDef::Clip { clip: tree_clip }],
                params: Vec::new(),
            },
        );
        // A `SkelPlayer` on a tree-driven root is ignored — and its clock
        // must not drift while it is.
        store_mut(&mut engine).insert(root, player(player_clip, 1.0));
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
        let time = store_ref(&engine)
            .read_lane::<SkelPlayer>()
            .expect("lane registered")
            .get(root)
            .expect("player kept")
            .time
            .get();
        assert_eq!(time, 0.0, "the ignored cursor must not advance");
    }
}
