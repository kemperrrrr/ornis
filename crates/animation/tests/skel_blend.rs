//! Phase E acceptance for skeletal blending (`docs/animation-design.md` §5 E):
//! player weight pulls toward the rest pose, crossfades mix two clips over
//! time and retire, and overlay layers follow their per-joint mask. Bad
//! masks skip the entity (bad skin), never truncate.

use glam::{Mat4, Vec3};
use ornis_animation::{
    Animator, ClipId, JointId, JointMask, JointPose, JointTrack, Key, KeyTrack, MaskId,
    SkelClip, SkelFade, SkelLayer, SkelPlayer, SkelSampleSystem, Skeleton, try_animator,
};
use ornis_assets::scene::TransformDesc;
use ornis_core::units::{Clamped01, Seconds};
use ornis_core::{Engine, Entity, Stage};

fn engine() -> Engine {
    let mut engine = Engine::new();
    {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        store.register::<SkelPlayer>();
        store.register::<SkelLayer>();
        store.register::<SkelFade>();
        store.register::<Skeleton>();
        store.register::<JointPose>();
        store.register::<TransformDesc>();
        store.register_cold::<SkelClip>();
        store.register_cold::<JointMask>();
    }
    engine.add_stage_system(Stage::PostFrame, SkelSampleSystem::new());
    engine
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

/// Skeleton root with `parents`, identity bind and a fresh pose lane.
fn rig(engine: &mut Engine, parents: Vec<Option<JointId>>) -> Entity {
    let store = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore");
    let joints = parents.len();
    let root = store.create_entity();
    store.insert(root, TransformDesc::IDENTITY);
    store.insert(
        root,
        Skeleton::new(
            parents,
            vec![Mat4::IDENTITY; joints],
            (0..joints).map(|i| format!("joint{i}")).collect(),
        ),
    );
    store.insert(root, JointPose::identity(joints));
    root
}

/// Stores `value` as a playlist clip and returns its id.
fn add_clip(engine: &mut Engine, value: SkelClip) -> ClipId {
    let store = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore");
    let playlist = store.create_entity();
    store.insert_cold(playlist, value);
    ClipId(playlist)
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

fn set_player(engine: &mut Engine, root: Entity, value: SkelPlayer) {
    engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
        .insert(root, value);
}

fn pose(engine: &Engine, root: Entity) -> Vec<Mat4> {
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

fn origin(matrix: Mat4) -> Vec3 {
    matrix.transform_point3(Vec3::ZERO)
}

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < 1e-4
}

#[test]
fn half_weight_pulls_the_pose_halfway_to_rest() {
    let mut engine = engine();
    let root = rig(&mut engine, vec![None]);
    let walk = add_clip(
        &mut engine,
        clip("walk", vec![constant(0, Vec3::new(4.0, 0.0, 0.0))]),
    );
    set_player(&mut engine, root, player(walk, 0.5));

    engine.run_frame(0.1);

    assert!(close(origin(pose(&engine, root)[0]), Vec3::new(2.0, 0.0, 0.0)));
}

#[test]
fn crossfade_mixes_both_clips_then_retires() {
    let mut engine = engine();
    let root = rig(&mut engine, vec![None]);
    let idle = add_clip(
        &mut engine,
        clip("idle", vec![constant(0, Vec3::ZERO)]),
    );
    let walk = add_clip(
        &mut engine,
        clip("walk", vec![constant(0, Vec3::new(4.0, 0.0, 0.0))]),
    );
    set_player(&mut engine, root, player(walk, 1.0));
    engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
        .insert(
            root,
            SkelFade {
                from: player(idle, 1.0),
                duration: Seconds::new(2.0),
                elapsed: Seconds::ZERO,
            },
        );

    // Halfway through a 2 s fade: the outgoing and incoming poses average.
    engine.run_frame(1.0);
    assert!(close(origin(pose(&engine, root)[0]), Vec3::new(2.0, 0.0, 0.0)));
    let fading = engine
        .world()
        .store()
        .expect("world owns SmartStore")
        .read_lane::<SkelFade>()
        .expect("lane registered")
        .get(root)
        .is_some();
    assert!(fading, "fade stays active until its duration elapses");

    // Past the end: the incoming clip alone, and the fade component is gone.
    engine.run_frame(1.0);
    assert!(close(origin(pose(&engine, root)[0]), Vec3::new(4.0, 0.0, 0.0)));
    let retired = engine
        .world()
        .store()
        .expect("world owns SmartStore")
        .read_lane::<SkelFade>()
        .expect("lane registered")
        .get(root)
        .is_none();
    assert!(retired, "finished fade is removed");
}

#[test]
fn layer_lerps_only_the_masked_joints() {
    let mut engine = engine();
    // Joint 1 is a child of joint 0, so its origin is joint 0's origin plus
    // its own local translation.
    let root = rig(
        &mut engine,
        vec![None, Some(JointId::from_raw(0))],
    );
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
    let wave = add_clip(
        &mut engine,
        clip(
            "wave",
            vec![
                constant(0, Vec3::new(0.0, 8.0, 0.0)),
                constant(1, Vec3::new(0.0, 8.0, 0.0)),
            ],
        ),
    );
    let mask = {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        let entity = store.create_entity();
        store.insert_cold(
            entity,
            JointMask {
                name: "upper".to_string(),
                weights: vec![Clamped01::ONE, Clamped01::ZERO],
            },
        );
        entity
    };
    set_player(&mut engine, root, player(base, 1.0));
    engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
        .insert(
            root,
            SkelLayer {
                clip: wave,
                mask: MaskId(mask),
                time: Seconds::ZERO,
                speed: Seconds::new(1.0),
                weight: Clamped01::ONE,
                playing: true,
                looping: true,
            },
        );

    engine.run_frame(0.1);

    let poses = pose(&engine, root);
    // Masked joint 0 takes the layer; joint 1 (mask 0) keeps the base pose.
    assert!(close(origin(poses[0]), Vec3::new(0.0, 8.0, 0.0)));
    assert!(close(origin(poses[1]), Vec3::new(4.0, 8.0, 0.0)));
}

#[test]
fn mask_length_mismatch_skips_the_entity() {
    let mut engine = engine();
    let root = rig(&mut engine, vec![None]);
    let base = add_clip(
        &mut engine,
        clip("base", vec![constant(0, Vec3::new(4.0, 0.0, 0.0))]),
    );
    let layer = add_clip(
        &mut engine,
        clip("layer", vec![constant(0, Vec3::new(0.0, 8.0, 0.0))]),
    );
    let mask = {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        let entity = store.create_entity();
        store.insert_cold(
            entity,
            JointMask {
                name: "wrong".to_string(),
                weights: vec![Clamped01::ONE, Clamped01::ONE],
            },
        );
        entity
    };
    set_player(&mut engine, root, player(base, 1.0));
    engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
        .insert(
            root,
            SkelLayer {
                clip: layer,
                mask: MaskId(mask),
                time: Seconds::ZERO,
                speed: Seconds::new(1.0),
                weight: Clamped01::ONE,
                playing: true,
                looping: true,
            },
        );

    engine.run_frame(0.1);

    // The pose is never published: the lane keeps its identity placeholder.
    assert_eq!(pose(&engine, root)[0], Mat4::IDENTITY);
}

#[test]
fn animator_crossfade_starts_a_fade_and_play_cuts_it() {
    let mut engine = engine();
    let root = rig(&mut engine, vec![None]);
    let idle = add_clip(&mut engine, clip("idle", vec![constant(0, Vec3::ZERO)]));
    let walk = add_clip(
        &mut engine,
        clip("walk", vec![constant(0, Vec3::new(4.0, 0.0, 0.0))]),
    );
    {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        let clips = [("idle".to_string(), idle), ("walk".to_string(), walk)]
            .into_iter()
            .collect();
        store.insert(root, Animator::new(clips, vec![root]));
    }

    {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        try_animator(store, root)
            .expect("animator present")
            .play("idle")
            .expect("idle known");
        try_animator(store, root)
            .expect("animator present")
            .play_crossfade("walk", Seconds::new(1.0))
            .expect("walk known");
    }

    let fade = engine
        .world()
        .store()
        .expect("world owns SmartStore")
        .read_lane::<SkelFade>()
        .expect("lane registered")
        .get(root)
        .copied()
        .expect("crossfade started");
    assert_eq!(fade.from.clip, idle);
    assert_eq!(fade.duration, Seconds::new(1.0));

    {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        try_animator(store, root)
            .expect("animator present")
            .play("walk")
            .expect("walk known");
    }
    let cut = engine
        .world()
        .store()
        .expect("world owns SmartStore")
        .read_lane::<SkelFade>()
        .expect("lane registered")
        .get(root)
        .is_none();
    assert!(cut, "a hard play cancels the crossfade");
}
