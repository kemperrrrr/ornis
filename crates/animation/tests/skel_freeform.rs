//! Freeform2d blend-tree acceptance (`docs/animation-design.md` §7.1):
//! pose fold through the tree (vertex / triangle midpoint), hot-path bad
//! skin from a malformed freeform node, skipped root on a non-finite
//! parameter, and `AnimEventQueue` effective weights equal to the fold.

mod common;

use common::*;
use glam::Vec3;
use ornis_animation::{ClipId, SkelTree, TreeNodeDef, TreeParam};
use ornis_core::Engine;

/// 3-point freeform over parameters `x`/`y` — the `clips` at the
/// triangle corners (0,0), (2,0), (0,2) — with `defaults` on the axes.
fn triangle_tree(clips: [ClipId; 3], defaults: [f32; 2]) -> SkelTree {
    SkelTree {
        nodes: vec![
            TreeNodeDef::Clip { clip: clips[0] },
            TreeNodeDef::Clip { clip: clips[1] },
            TreeNodeDef::Clip { clip: clips[2] },
            TreeNodeDef::Freeform2d {
                axes: [0, 1],
                children: vec![
                    child2d(0.0, 0.0, 0),
                    child2d(2.0, 0.0, 1),
                    child2d(0.0, 2.0, 2),
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

/// The three corner clips of a [`triangle_tree`] with x poses 2, 6, 4.
fn triangle_clips(engine: &mut Engine) -> [ClipId; 3] {
    [
        clip_x(engine, "a", 2.0),
        clip_x(engine, "b", 6.0),
        clip_x(engine, "c", 4.0),
    ]
}

/// Runs one frame of a [`triangle_tree`] at `defaults` and expects the
/// root joint origin at `want`.
fn freeform2d_at(defaults: [f32; 2], want: f32) {
    let mut engine = engine();
    let root = rig(&mut engine, 1);
    let clips = triangle_clips(&mut engine);
    install(&mut engine, root, triangle_tree(clips, defaults));
    engine.run_frame(0.1);
    expect_origin(&engine, root, 0, Vec3::new(want, 0.0, 0.0));
}

#[test]
fn freeform2d_vertex_owns_the_pose() {
    freeform2d_at([2.0, 0.0], 6.0);
}

#[test]
fn freeform2d_triangle_midpoint_mixes_three() {
    // (0.5, 0.5) is the barycentric (1/2, 1/4, 1/4) of the corner
    // clips' x poses 2, 6, 4.
    freeform2d_at([0.5, 0.5], 3.5);
}

#[test]
fn bad_freeform_tree_leaves_the_pose_unpublished() {
    let mut engine = engine();
    let clips = triangle_clips(&mut engine);
    let root = rig(&mut engine, 1);
    // Repeated points — installed straight into the lane (bypassing
    // the typed facade) — must read as bad skin on the hot path.
    let tree = SkelTree {
        nodes: vec![
            TreeNodeDef::Clip { clip: clips[0] },
            TreeNodeDef::Clip { clip: clips[1] },
            TreeNodeDef::Freeform2d {
                axes: [0, 1],
                children: vec![child2d(0.0, 0.0, 0), child2d(0.0, 0.0, 1)],
            },
        ],
        params: vec![
            TreeParam {
                name: "x".to_string(),
                default: 0.5,
            },
            TreeParam {
                name: "y".to_string(),
                default: 0.5,
            },
        ],
    };
    install(&mut engine, root, tree);
    engine.run_frame(0.1);
    expect_origin(&engine, root, 0, Vec3::ZERO);
}

#[test]
fn non_finite_freeform_param_skips_the_root() {
    let mut engine = engine();
    let clips = triangle_clips(&mut engine);
    let root = rig(&mut engine, 1);
    install(&mut engine, root, triangle_tree(clips, [0.5, 0.5]));
    assert!(param_of(&mut engine, root, "x", f32::NAN));
    engine.run_frame(0.1);
    expect_origin(&engine, root, 0, Vec3::ZERO);
}

#[test]
fn freeform2d_events_carry_the_fold_weights() {
    use ornis_animation::{AnimEvent, AnimEventQueue, ClipEvents};
    let mut engine = engine();
    let clips = triangle_clips(&mut engine);
    for clip in clips {
        store_mut(&mut engine).insert_cold(
            clip.0,
            ClipEvents {
                events: vec![AnimEvent {
                    name: "step".to_string(),
                    at: 0.05,
                }],
            },
        );
    }
    let root = rig(&mut engine, 1);
    install(&mut engine, root, triangle_tree(clips, [0.5, 0.5]));
    store_mut(&mut engine).insert(root, AnimEventQueue::default());
    engine.run_frame(1.0);
    let store = store_ref(&engine);
    let fired = store
        .read_lane::<AnimEventQueue>()
        .expect("queue lane kept")
        .get(root)
        .expect("root queue kept")
        .fired
        .clone();
    assert_eq!(fired.len(), 3, "each leaf fires once");
    // Leaf order = clip-node order; the weights are the freeform fold
    // at (0.5, 0.5) — barycentric (1/2, 1/4, 1/4) of the corners.
    let want = [0.5, 0.25, 0.25];
    for (fired, want) in fired.iter().zip(want) {
        assert_eq!(fired.event, "step");
        assert!(
            (fired.weight - want).abs() < 1e-4,
            "weight {} vs {want}",
            fired.weight
        );
    }
}
