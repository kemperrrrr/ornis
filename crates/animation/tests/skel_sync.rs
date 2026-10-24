//! Phase-E acceptance for clip sync marks
//! (`docs/animation-design.md` §7.5): a machine transition re-phases the
//! incoming state's looping leaf cursors so that same-named marks of the
//! leaving and arriving clips land on the same wall-clock hit. Leaves
//! without a partner (no shared name, no marks, not looping) keep their
//! clocks.
//!
//! Shared world fixtures live in `common`; the machine scene here is its
//! own fixture, like in `skel_machine.rs`.

mod common;

use common::*;
use glam::Vec3;
use ornis_animation::{
    ClipId, ClipSyncMarks, SkelClip, SkelTree, SkelTreePlayer, SyncMark, TreeCmp, TreeCond,
    TreeNodeDef, TreeParam, TreeTransition,
};
use ornis_core::units::Seconds;
use ornis_core::{Engine, Entity};

/// `from` crossfades to `to` while the tree's single parameter compares
/// `op` against `at`, over `fade` seconds.
fn transition(from: usize, to: usize, op: TreeCmp, at: f32, fade: f32) -> TreeTransition {
    TreeTransition {
        from,
        to,
        when: TreeCond {
            param: 0,
            op,
            value: at,
        },
        fade: Seconds::new(fade),
    }
}

/// One sync mark.
fn marks(name: &str, at: f32) -> Vec<SyncMark> {
    vec![SyncMark {
        name: name.to_string(),
        at,
    }]
}

/// Stores clip `name` (`duration` long, pose `x`) with `marks` on it.
fn marked_clip(
    engine: &mut Engine,
    name: &str,
    duration: f32,
    x: f32,
    marks: Vec<SyncMark>,
) -> ClipId {
    let id = add_clip(
        engine,
        SkelClip {
            name: name.to_string(),
            duration,
            tracks: vec![constant(0, Vec3::new(x, 0.0, 0.0))],
        },
    );
    store_mut(engine).insert_cold(id.0, ClipSyncMarks { marks });
    id
}

/// Machine over the `walk` (1.0 s, x=2) and `run` (0.5 s, x=4) leaves
/// with one `speed` parameter.
fn machine(walk: ClipId, run: ClipId, transitions: Vec<TreeTransition>) -> SkelTree {
    SkelTree {
        nodes: vec![
            TreeNodeDef::Clip { clip: walk },
            TreeNodeDef::Clip { clip: run },
            TreeNodeDef::Machine {
                states: vec![0, 1],
                initial: 0,
                transitions,
            },
        ],
        params: vec![TreeParam {
            name: "speed".to_string(),
            default: 0.0,
        }],
    }
}

/// The marked two-clip machine scene, stepped one 0.25 s frame (the
/// `walk` cursor sits at 0.25) before any handoff can fire.
fn scene(
    walk_marks: Vec<SyncMark>,
    run_marks: Vec<SyncMark>,
    transitions: Vec<TreeTransition>,
) -> (Engine, Entity) {
    let mut engine = engine();
    let walk = marked_clip(&mut engine, "walk", 1.0, 2.0, walk_marks);
    let run = marked_clip(&mut engine, "run", 0.5, 4.0, run_marks);
    let root = rig(&mut engine, 1);
    install(&mut engine, root, machine(walk, run, transitions));
    engine.run_frame(0.25);
    (engine, root)
}

/// Sets `speed = 1.0` and steps the 0.25 s frame the handoff fires on;
/// returns the arriving leaf's cursor afterwards.
fn fired(engine: &mut Engine, root: Entity) -> f32 {
    assert!(param_of(engine, root, "speed", 1.0));
    engine.run_frame(0.25);
    leaf_time(engine, root, 1)
}

/// Cursor time of leaf `slot` on `root`.
fn leaf_time(engine: &Engine, root: Entity, slot: usize) -> f32 {
    store_ref(engine)
        .read_lane::<SkelTreePlayer>()
        .expect("lane registered")
        .get(root)
        .expect("player kept")
        .leaves[slot]
        .time
        .get()
}

/// Forces `looping` on leaf `slot` (tree leaves have no facade input
/// yet).
fn set_looping(engine: &mut Engine, root: Entity, slot: usize, looping: bool) {
    store_mut(engine)
        .write_lane::<SkelTreePlayer>()
        .expect("lane registered")
        .get_mut(root)
        .expect("player kept")
        .leaves[slot]
        .looping = looping;
}

#[test]
fn shared_marks_rephase_the_incoming_leaf() {
    let (mut engine, root) = scene(
        marks("foot", 0.75),
        marks("foot", 0.25),
        vec![transition(0, 1, TreeCmp::Greater, 0.5, 1.0)],
    );
    assert_eq!(fired(&mut engine, root), 0.375);
}

#[test]
fn disjoint_names_leave_the_cursors_alone() {
    let (mut engine, root) = scene(
        marks("a", 0.75),
        marks("b", 0.25),
        vec![transition(0, 1, TreeCmp::Greater, 0.5, 1.0)],
    );
    assert_eq!(fired(&mut engine, root), 0.0);
}

#[test]
fn unmarked_clips_leave_the_cursors_alone() {
    let (mut engine, root) = scene(
        Vec::new(),
        Vec::new(),
        vec![transition(0, 1, TreeCmp::Greater, 0.5, 1.0)],
    );
    assert_eq!(fired(&mut engine, root), 0.0);
}

#[test]
fn a_cut_rephases_too() {
    let (mut engine, root) = scene(
        marks("foot", 0.75),
        marks("foot", 0.25),
        vec![transition(0, 1, TreeCmp::Greater, 0.5, 0.0)],
    );
    assert_eq!(fired(&mut engine, root), 0.375);
}

#[test]
fn non_looping_leaves_keep_their_clock() {
    let (mut engine, root) = scene(
        marks("foot", 0.75),
        marks("foot", 0.25),
        vec![transition(0, 1, TreeCmp::Greater, 0.5, 1.0)],
    );
    set_looping(&mut engine, root, 1, false);
    assert_eq!(fired(&mut engine, root), 0.5);
}
