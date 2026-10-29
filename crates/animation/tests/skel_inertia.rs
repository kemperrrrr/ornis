//! Phase E acceptance for inertialized match transitions
//! (`docs/animation-design.md` §8.5): the outgoing base freezes into
//! per-joint offsets that decay over `SkelInertia::duration`, so a matched
//! frame takes over without the double silhouette of a crossfade.

mod common;

use common::*;
use glam::Vec3;
use ornis_animation::{ClipId, SkelInertia, SkelPlayer};
use ornis_core::units::Seconds;

#[test]
fn an_inertialized_switch_freezes_the_outgoing_pose() {
    let mut engine = engine();
    let out = clip_x(&mut engine, "out", 2.0);
    let into = clip_x(&mut engine, "into", 6.0);
    let root = rig(&mut engine, 1);
    store_mut(&mut engine).insert(root, player(into, 1.0));
    store_mut(&mut engine).insert(root, SkelInertia::new(player(out, 1.0), Seconds::new(1.0)));
    engine.run_frame(0.5);
    expect_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
}

#[test]
fn an_inertialized_switch_decays_into_the_matched_frame() {
    let mut engine = engine();
    let out = clip_x(&mut engine, "out", 3.0);
    let into = clip_x(&mut engine, "into", 7.0);
    let root = rig(&mut engine, 1);
    store_mut(&mut engine).insert(root, SkelInertia::new(player(out, 1.0), Seconds::new(1.0)));
    store_mut(&mut engine).insert(root, player(into, 1.0));
    engine.run_frame(0.25);
    expect_origin(&engine, root, 0, Vec3::new(3.0, 0.0, 0.0));
    engine.run_frame(0.25);
    expect_origin(&engine, root, 0, Vec3::new(4.0, 0.0, 0.0));
    engine.run_frame(0.25);
    expect_origin(&engine, root, 0, Vec3::new(5.0, 0.0, 0.0));
    engine.run_frame(0.25);
    expect_origin(&engine, root, 0, Vec3::new(6.0, 0.0, 0.0));
}

#[test]
fn a_vanished_outgoing_clip_captures_against_rest() {
    let mut engine = engine();
    let into = clip_x(&mut engine, "into", 4.0);
    let root = rig(&mut engine, 1);
    let ghost = SkelPlayer {
        clip: ClipId(store_mut(&mut engine).create_entity()),
        ..player(into, 1.0)
    };
    store_mut(&mut engine).insert(root, SkelInertia::new(ghost, Seconds::new(0.5)));
    store_mut(&mut engine).insert(root, player(into, 1.0));
    engine.run_frame(0.25);
    expect_origin(&engine, root, 0, Vec3::new(0.0, 0.0, 0.0));
    engine.run_frame(0.25);
    expect_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
}

#[test]
fn an_inertialized_switch_retires_itself_with_the_falloff() {
    let mut engine = engine();
    let out = clip_x(&mut engine, "out", 2.5);
    let into = clip_x(&mut engine, "into", 6.5);
    let root = rig(&mut engine, 1);
    let inertia = SkelInertia::new(player(out, 1.0), Seconds::new(0.5));
    store_mut(&mut engine).insert(root, player(into, 1.0));
    store_mut(&mut engine).insert(root, inertia);
    engine.run_frame(0.25);
    expect_origin(&engine, root, 0, Vec3::new(2.5, 0.0, 0.0));
    engine.run_frame(0.25);
    expect_origin(&engine, root, 0, Vec3::new(4.5, 0.0, 0.0));
    engine.run_frame(0.25);
    expect_origin(&engine, root, 0, Vec3::new(6.5, 0.0, 0.0));
    let inertias = store_ref(&engine)
        .read_lane::<SkelInertia>()
        .expect("lane registered");
    assert!(inertias.get(root).is_none());
}

#[test]
fn a_non_positive_falloff_cuts_the_switch_at_once() {
    let mut engine = engine();
    let out = clip_x(&mut engine, "out", 1.5);
    let into = clip_x(&mut engine, "into", 5.5);
    let root = rig(&mut engine, 1);
    store_mut(&mut engine).insert(root, SkelInertia::new(player(out, 1.0), Seconds::ZERO));
    store_mut(&mut engine).insert(root, player(into, 1.0));
    engine.run_frame(0.1);
    expect_origin(&engine, root, 0, Vec3::new(5.5, 0.0, 0.0));
    engine.run_frame(0.1);
    let inertias = store_ref(&engine)
        .read_lane::<SkelInertia>()
        .expect("lane registered");
    assert!(inertias.get(root).is_none());
}
