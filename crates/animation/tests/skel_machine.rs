//! Phase E acceptance for animation-tree state machines
//! (`docs/animation-design.md` §7.4): `TreeNodeDef::Machine` stands on
//! its initial state, crossfades between states as `TreeTransition`s fire
//! on the live parameters (`SkelFade` mechanics — the outgoing state's
//! cursors keep stepping, non-positive fade cuts), and any machine-
//! contract violation skips the root (bad skin), never truncates.
//!
//! Shared world fixtures live in `common` (same bodies as the
//! `skel_tree` tests — extracted once, not duplicated).

mod common;

use common::*;
use glam::Vec3;
use ornis_animation::{
    ClipId, SkelTree, SkelTreePlayer, TreeCmp, TreeCond, TreeNodeDef, TreeParam, TreeTransition,
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

/// Machine over two clip states (`left`/`right` at nodes 0 and 1, the
/// machine itself at node 2) with one parameter named `name`.
fn machine_tree(
    name: &str,
    left: ClipId,
    right: ClipId,
    initial: usize,
    transitions: Vec<TreeTransition>,
) -> SkelTree {
    SkelTree {
        nodes: vec![
            TreeNodeDef::Clip { clip: left },
            TreeNodeDef::Clip { clip: right },
            TreeNodeDef::Machine {
                states: vec![0, 1],
                initial,
                transitions,
            },
        ],
        params: vec![TreeParam {
            name: name.to_string(),
            default: 0.0,
        }],
    }
}

/// World with the machine installed on a fresh single-joint root and two
/// X-clips at `lx`/`rx`; returns the clip ids for reinstall scenarios.
fn scene(
    initial: usize,
    transitions: Vec<TreeTransition>,
    lx: f32,
    rx: f32,
) -> (Engine, Entity, ClipId, ClipId) {
    let mut engine = engine();
    let left = clip_x(&mut engine, "left", lx);
    let right = clip_x(&mut engine, "right", rx);
    let root = rig(&mut engine, 1);
    install(
        &mut engine,
        root,
        machine_tree("speed", left, right, initial, transitions),
    );
    (engine, root, left, right)
}

/// Sets the `speed` parameter and steps one frame — the frame a fired
/// transition starts (or completes, for a non-positive fade) on.
fn drive(mut engine: Engine, root: Entity, speed: f32) -> Engine {
    assert!(param_of(&mut engine, root, "speed", speed));
    engine.run_frame(0.1);
    engine
}

/// Steps `frames` frames of `dt` on.
fn settled(mut engine: Engine, dt: f32, frames: u32) -> Engine {
    for _ in 0..frames {
        engine.run_frame(dt);
    }
    engine
}

/// `speed` goes to 1.0: the `Greater 0.5` handoff has just started with
/// `fade` seconds on its clock (or cut, for a non-positive fade).
fn fired(lx: f32, rx: f32, fade: f32, start: f32) -> (Engine, Entity) {
    let handoff = vec![transition(0, 1, TreeCmp::Greater, 0.5, fade)];
    let (engine, root, _, _) = scene(0, handoff, lx, rx);
    let engine = drive(engine, root, start);
    (engine, root)
}

/// Fresh machine that has stepped one frame on its `initial` state.
fn stand(initial: usize, lx: f32, rx: f32) -> (Engine, Entity) {
    let (mut engine, root, _, _) = scene(initial, Vec::new(), lx, rx);
    engine.run_frame(0.1);
    (engine, root)
}

/// The handoff's guard never matches: `speed` stays at its default.
fn waiting(lx: f32, rx: f32) -> (Engine, Entity) {
    let handoff = vec![transition(0, 1, TreeCmp::Greater, 0.5, 0.4)];
    let (mut engine, root, _, _) = scene(0, handoff, lx, rx);
    engine.run_frame(0.1);
    engine.run_frame(0.1);
    (engine, root)
}

/// The `Less -0.5` handoff fires the moment `speed` reaches -1.0.
fn fell(lx: f32, rx: f32) -> (Engine, Entity) {
    let (engine, root, _, _) = scene(0, vec![transition(0, 1, TreeCmp::Less, -0.5, 0.0)], lx, rx);
    let engine = drive(engine, root, -1.0);
    (engine, root)
}

/// Two handoffs match at once from state 0 — only the first listed may
/// fire (to 1); the second is a self-loop decoy.
fn raced(lx: f32, rx: f32) -> (Engine, Entity) {
    let (engine, root, _, _) = scene(
        0,
        vec![
            transition(0, 1, TreeCmp::Greater, 0.5, 0.0),
            transition(0, 0, TreeCmp::Greater, -99.0, 0.0),
        ],
        lx,
        rx,
    );
    let engine = drive(engine, root, 1.0);
    (engine, root)
}

/// A 2s handoff starts, the way-back guard matches mid-fade, and 4 more
/// half-seconds pass: the way back may only start once the first fade
/// completes — halfway back by the end.
fn tempted(lx: f32, rx: f32) -> (Engine, Entity) {
    let pair = vec![
        transition(0, 1, TreeCmp::Greater, 0.5, 2.0),
        transition(1, 0, TreeCmp::Less, 0.25, 1.0),
    ];
    let (engine, root, _, _) = scene(0, pair, lx, rx);
    let mut engine = drive(engine, root, 1.0);
    engine = settled(engine, 0.5, 1);
    expect_origin(&engine, root, 0, Vec3::new(3.0, 0.0, 0.0));
    assert!(
        param_of(&mut engine, root, "speed", -1.0),
        "the way back matches mid-fade"
    );
    engine = settled(engine, 0.5, 4);
    (engine, root)
}

/// After a cut to the right state, a fresh tree seeds the machine back
/// on `initial`.
fn reinstalled() -> (Engine, Entity) {
    let handoff = vec![transition(0, 1, TreeCmp::Greater, 0.5, 0.0)];
    let (engine, root, left, right) = scene(0, handoff, 2.0, 6.0);
    let mut engine = drive(engine, root, 1.0);
    install(
        &mut engine,
        root,
        machine_tree("speed", left, right, 0, Vec::new()),
    );
    engine.run_frame(0.1);
    (engine, root)
}

/// Live machine cursor state of the single machine on `root`.
fn machine_state(engine: &Engine, root: Entity) -> usize {
    store_ref(engine)
        .read_lane::<SkelTreePlayer>()
        .expect("lane registered")
        .get(root)
        .expect("player kept")
        .machines[0]
        .state
}

#[test]
fn machine_stands_on_its_initial_state() {
    let (engine, root) = stand(1, 1.0, 3.0);
    expect_origin(&engine, root, 0, Vec3::new(3.0, 0.0, 0.0));
}

#[test]
fn crossfade_starts_from_the_outgoing_state() {
    let (engine, root) = fired(2.0, 6.0, 1.0, 1.0);
    expect_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
}

#[test]
fn transition_crossfades_between_states() {
    let (engine, root) = fired(2.0, 6.0, 1.0, 1.0);
    expect_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
    let engine = settled(engine, 0.5, 1);
    expect_origin(&engine, root, 0, Vec3::new(4.0, 0.0, 0.0));
    let engine = settled(engine, 0.5, 1);
    expect_origin(&engine, root, 0, Vec3::new(6.0, 0.0, 0.0));
}

#[test]
fn transition_waits_for_its_condition() {
    let (engine, root) = waiting(2.0, 8.0);
    assert!(
        leaf_time(&engine, root, 0) > 0.0 && leaf_time(&engine, root, 1) > 0.0,
        "the standing state's cursors step"
    );
    expect_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
}

#[test]
fn zero_fade_cuts_at_once() {
    let (engine, root) = fired(1.0, 5.0, 0.0, 1.0);
    expect_origin(&engine, root, 0, Vec3::new(5.0, 0.0, 0.0));
}

#[test]
fn less_condition_fires_below_the_threshold() {
    let (engine, root) = fell(1.0, 4.0);
    expect_origin(&engine, root, 0, Vec3::new(4.0, 0.0, 0.0));
}

#[test]
fn first_matching_transition_wins() {
    let (engine, root) = raced(2.0, 6.0);
    let state = machine_state(&engine, root);
    assert_eq!(state, 1);
}

#[test]
fn inflight_crossfade_is_never_interrupted() {
    let (engine, root) = tempted(2.0, 6.0);
    expect_origin(&engine, root, 0, Vec3::new(4.0, 0.0, 0.0));
}

#[test]
fn a_new_tree_resets_the_machine() {
    let (engine, root) = reinstalled();
    expect_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
}

#[test]
fn broken_machine_leaves_the_pose_unpublished() {
    let (mut engine, root, _, _) = scene(1, Vec::new(), 2.0, 6.0);
    let Some(mut players) = store_mut(&mut engine).write_lane::<SkelTreePlayer>() else {
        panic!("lane registered");
    };
    players.get_mut(root).expect("player kept").machines[0].state = 5;
    drop(players);
    engine.run_frame(0.1);
    expect_origin(&engine, root, 0, Vec3::ZERO);
}
