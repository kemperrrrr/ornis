//! Phase E acceptance for animation trees (`docs/animation-design.md`
//! §7): `SkelTree` blend1d nodes fold their weighted children over the
//! leaf clip cursors, layer stacks overlay the tree root, leaf clocks step
//! and pause like phase-E players, and any tree-contract violation skips
//! the root (bad skin), never truncates.
//!
//! Shared world fixtures live in `common` (same bodies as the
//! `skel_machine` tests — extracted once, not duplicated).

mod common;

use common::*;
use glam::Vec3;
use ornis_animation::{
    ClipId, JointId, JointMask, MaskId, RetargetMap, SkelLayer, SkelLayerStack, SkelPlayer,
    SkelTree, SkelTreePlayer, TreeChild, TreeNodeDef, TreeParam,
};
use ornis_core::Engine;
use ornis_core::units::Clamped01;

fn child(position: f32, node: usize) -> TreeChild {
    TreeChild { position, node }
}

/// 2×2 blend2d over parameters `x`/`y` — the four `clips` at the lattice
/// corners (0,0), (1,0), (0,1), (1,1) — with `defaults` on the axes.
fn quad_tree(clips: [ClipId; 4], defaults: [f32; 2]) -> SkelTree {
    SkelTree {
        nodes: vec![
            TreeNodeDef::Clip { clip: clips[0] },
            TreeNodeDef::Clip { clip: clips[1] },
            TreeNodeDef::Clip { clip: clips[2] },
            TreeNodeDef::Clip { clip: clips[3] },
            TreeNodeDef::Blend2d {
                axes: [0, 1],
                children: vec![
                    child2d(0.0, 0.0, 0),
                    child2d(1.0, 0.0, 1),
                    child2d(0.0, 1.0, 2),
                    child2d(1.0, 1.0, 3),
                ],
            },
        ],
        params: vec![
            TreeParam {
                name: "x".to_string(),
                default: defaults[0],
            },
            TreeParam {
                name: "y".to_string(),
                default: defaults[1],
            },
        ],
    }
}

/// The four corner clips of a [`quad_tree`] with x poses 2, 6, 4, 8.
fn quad_clips(engine: &mut Engine) -> [ClipId; 4] {
    [
        clip_x(engine, "a", 2.0),
        clip_x(engine, "b", 6.0),
        clip_x(engine, "c", 4.0),
        clip_x(engine, "d", 8.0),
    ]
}

/// Runs one frame of a [`quad_tree`] at `defaults` and expects the root
/// joint origin at `want`.
fn blend2d_at(defaults: [f32; 2], want: Vec3) {
    let mut engine = engine();
    let clips = quad_clips(&mut engine);
    let root = rig(&mut engine, 1);
    install(&mut engine, root, quad_tree(clips, defaults));
    engine.run_frame(0.1);
    expect_origin(&engine, root, 0, want);
}

#[test]
fn blend2d_cell_midpoint_mixes_all_four() {
    blend2d_at([0.5, 0.5], Vec3::new(5.0, 0.0, 0.0));
}

#[test]
fn blend2d_lattice_lines_mix_two() {
    blend2d_at([0.5, 0.0], Vec3::new(4.0, 0.0, 0.0));
}

#[test]
fn blend2d_outside_the_lattice_clamps_to_the_corner() {
    blend2d_at([2.0, -1.0], Vec3::new(6.0, 0.0, 0.0));
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

/// A tree leaf authored on the source skeleton plays through the root's
/// [`RetargetMap`]: joint 0 of the clip lands on joint 1, translation
/// scaled by the baked rest-length ratio (§8.3).
#[test]
fn a_tree_leaf_samples_through_the_retarget_map() {
    let mut engine = engine();
    let clip = add_clip(
        &mut engine,
        clip("source", vec![constant(0, Vec3::new(4.0, 0.0, 0.0))]),
    );
    let root = rig(&mut engine, 2);
    install(
        &mut engine,
        root,
        SkelTree {
            nodes: vec![TreeNodeDef::Clip { clip }],
            params: Vec::new(),
        },
    );
    store_mut(&mut engine).insert_cold(
        root,
        RetargetMap {
            joints: vec![Some(JointId::from_raw(1))],
            scales: vec![2.0],
        },
    );
    engine.run_frame(0.1);
    expect_origin(&engine, root, 0, Vec3::ZERO);
    expect_origin(&engine, root, 1, Vec3::new(8.0, 0.0, 0.0));
}
