//! Phase E acceptance for animation-tree switch nodes
//! (`docs/animation-design.md` §7.7): `TreeNodeDef::Switch` stands on the
//! child its rounded parameter selects (clamped to the covered range),
//! crossfades between selections with `SkelFade` mechanics (an in-flight
//! crossfade is never interrupted, non-positive fade cuts), re-phases the
//! leaving and arriving leaves through `ClipSyncMarks`, plays through the
//! root's `RetargetMap` (§8.3), and any switch-contract violation skips
//! the root (bad skin), never truncates.
//!
//! Shared world fixtures live in `common` (same bodies as the
//! `skel_tree` tests — extracted once, not duplicated).

mod common;

use common::*;
use glam::Vec3;
use ornis_animation::{
    ClipId, JointId, RetargetMap, SkelTree, SkelTreePlayer, SwitchCursor, TreeNodeDef, TreeParam,
    switch_selection,
};
use ornis_core::units::Seconds;
use ornis_core::{Engine, Entity};

/// Switch over two clip children (`left`/`right` at nodes 0 and 1, the
/// switch itself at node 2) with one parameter named `name` standing on
/// `default`, crossfading over `fade` seconds.
fn switch_tree(name: &str, default: f32, left: ClipId, right: ClipId, fade: f32) -> SkelTree {
    SkelTree {
        nodes: vec![
            TreeNodeDef::Clip { clip: left },
            TreeNodeDef::Clip { clip: right },
            TreeNodeDef::Switch {
                param: 0,
                children: vec![0, 1],
                fade: Seconds::new(fade),
            },
        ],
        params: vec![TreeParam {
            name: name.to_string(),
            default,
        }],
    }
}

/// World with the switch installed on a fresh single-joint root and two
/// X-clips at `lx`/`rx`.
fn scene(default: f32, fade: f32, lx: f32, rx: f32) -> (Engine, Entity) {
    {
        let mut engine = engine();
        let left = clip_x(&mut engine, "left", lx);
        let right = clip_x(&mut engine, "right", rx);
        let root = rig(&mut engine, 1);
        install(
            &mut engine,
            root,
            switch_tree("pick", default, left, right, fade),
        );
        (engine, root)
    }
}

/// One `dt` frame with `pick` set to `value` beforehand.
fn drive(mut engine: Engine, root: Entity, value: f32) -> Engine {
    assert!(param_of(&mut engine, root, "pick", value));
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

#[test]
fn switch_stands_on_its_default_selection() {
    let (mut engine, root) = scene(1.0, 0.4, 2.0, 6.0);
    engine.run_frame(0.1);
    expect_origin(&engine, root, 0, Vec3::new(6.0, 0.0, 0.0));
}

#[test]
fn switch_selection_rounds_and_clamps() {
    assert_eq!(switch_selection(0.4, 2), 0);
    assert_eq!(switch_selection(0.6, 2), 1);
    assert_eq!(switch_selection(2.5, 2), 1, "beyond the range clamps");
    assert_eq!(switch_selection(-3.0, 2), 0, "below the range clamps");
    assert_eq!(switch_selection(2.5, 0), 0, "empty children read as 0");
}

#[test]
fn switch_crossfades_between_selections() {
    let (engine, root) = scene(0.0, 1.0, 2.0, 6.0);
    let engine = drive(engine, root, 1.0);
    expect_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
    let engine = settled(engine, 0.5, 1);
    expect_origin(&engine, root, 0, Vec3::new(4.0, 0.0, 0.0));
    let engine = settled(engine, 0.5, 1);
    expect_origin(&engine, root, 0, Vec3::new(6.0, 0.0, 0.0));
}

#[test]
fn switch_fade_is_never_interrupted() {
    let (engine, root) = scene(0.0, 1.0, 2.0, 6.0);
    let engine = drive(engine, root, 1.0);
    // The selection flips back mid-flight: the fade still heads for 1.
    let engine = drive(engine, root, 0.0);
    let engine = settled(engine, 0.4, 1);
    expect_origin(&engine, root, 0, Vec3::new(4.0, 0.0, 0.0));
    // Once it completes, the fresh selection starts its own fade back.
    let engine = settled(engine, 0.6, 1);
    expect_origin(&engine, root, 0, Vec3::new(6.0, 0.0, 0.0));
    let engine = settled(engine, 0.5, 1);
    expect_origin(&engine, root, 0, Vec3::new(4.0, 0.0, 0.0));
}

#[test]
fn non_positive_fade_cuts_at_once() {
    let (engine, root) = scene(0.0, 0.0, 2.0, 6.0);
    let engine = drive(engine, root, 1.0);
    expect_origin(&engine, root, 0, Vec3::new(6.0, 0.0, 0.0));
}

#[test]
fn out_of_range_selection_clamps_to_the_edge() {
    let (engine, root) = scene(0.0, 0.0, 2.0, 6.0);
    let engine = drive(engine, root, 5.0);
    expect_origin(&engine, root, 0, Vec3::new(6.0, 0.0, 0.0));
    let engine = drive(engine, root, -3.0);
    expect_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
}

#[test]
fn switch_contract_violation_is_bad_skin() {
    {
        let mut engine = engine();
        let clip = clip_x(&mut engine, "only", 2.0);
        let root = rig(&mut engine, 1);
        // A hand-installed switch with no children (the typed facade
        // rejects it) must read as bad skin — never poison the pose.
        let tree = SkelTree {
            nodes: vec![
                TreeNodeDef::Clip { clip },
                TreeNodeDef::Switch {
                    param: 0,
                    children: Vec::new(),
                    fade: Seconds::ZERO,
                },
            ],
            params: Vec::new(),
        };
        let mut player_state = SkelTreePlayer::default();
        player_state.leaves = vec![player(clip, 1.0)];
        player_state.switches = vec![SwitchCursor {
            child: 0,
            from: None,
            elapsed: Seconds::ZERO,
            duration: Seconds::ZERO,
        }];
        let store = store_mut(&mut engine);
        store.insert_cold(root, tree);
        store.insert(root, player_state);
        engine.run_frame(0.1);
        expect_origin(&engine, root, 0, Vec3::ZERO);
    }
}

/// Switch leaves authored on the source skeleton play through the root's
/// [`RetargetMap`] on the standing selection: joint 0 of the clip lands
/// on joint 1, translation scaled by the baked rest-length ratio
/// (§8.3, "дерево на ретаргете").
#[test]
fn switch_leaves_sample_through_the_retarget_map() {
    let mut engine = engine();
    let forward = add_clip(
        &mut engine,
        clip("forward", vec![constant(0, Vec3::new(4.0, 0.0, 0.0))]),
    );
    let backward = add_clip(
        &mut engine,
        clip("backward", vec![constant(0, Vec3::new(2.0, 0.0, 0.0))]),
    );
    let root = rig(&mut engine, 2);
    install(
        &mut engine,
        root,
        switch_tree("pick", 1.0, forward, backward, 0.0),
    );
    let map = RetargetMap {
        joints: vec![Some(JointId::from_raw(1))],
        scales: vec![2.0],
    };
    store_mut(&mut engine).insert_cold(root, map);
    engine.run_frame(0.1);
    expect_origin(&engine, root, 1, Vec3::new(4.0, 0.0, 0.0));
}
