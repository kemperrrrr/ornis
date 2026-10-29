//! Phase E acceptance for skeletal blending (`docs/animation-design.md` §5 E):
//! player weight pulls toward the rest pose, crossfades mix two clips over
//! time and retire, and overlay layer stacks follow their per-joint masks
//! bottom-up. Bad masks skip the entity (bad skin), never truncate.

use glam::{Mat4, Quat, Vec3};
use ornis_animation::{
    Animator, AnimatorMut, ClipId, JointId, JointMask, JointPose, JointTrack, JointTrs, Key,
    KeyTrack, MaskId, SkelClip, SkelFade, SkelLayer, SkelLayerStack, SkelPlayer, SkelSampleSystem,
    Skeleton, try_animator,
};
use ornis_assets::scene::TransformDesc;
use ornis_core::units::{Clamped01, Seconds};
use ornis_core::{Engine, Entity, SmartStore, Stage, Transform};

fn engine() -> Engine {
    let mut engine = Engine::new();
    {
        let store = store_mut(&mut engine);
        store.register::<SkelPlayer>();
        store.register::<SkelLayerStack>();
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

fn store_mut(engine: &mut Engine) -> &mut SmartStore {
    engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
}

fn store_ref(engine: &Engine) -> &SmartStore {
    engine.world().store().expect("world owns SmartStore")
}

fn put<T: Clone + Send + Sync + 'static>(engine: &mut Engine, entity: Entity, value: T) {
    store_mut(engine).insert(entity, value);
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
    let joints = parents.len();
    let names = (0..joints).map(|i| format!("joint{i}")).collect();
    let skeleton = Skeleton::new(parents, vec![Mat4::IDENTITY; joints], names);
    let store = store_mut(engine);
    let root = store.create_entity();
    store.insert(root, TransformDesc::IDENTITY);
    store.insert(root, skeleton);
    store.insert(root, JointPose::identity(joints));
    root
}

/// Stores `value` as a playlist clip and returns its id.
fn add_clip(engine: &mut Engine, value: SkelClip) -> ClipId {
    let store = store_mut(engine);
    let playlist = store.create_entity();
    store.insert_cold(playlist, value);
    ClipId(playlist)
}

/// Stores a per-joint mask as a playlist entity and returns its id.
fn add_mask(engine: &mut Engine, name: &str, weights: Vec<Clamped01>) -> MaskId {
    let store = store_mut(engine);
    let entity = store.create_entity();
    store.insert_cold(
        entity,
        JointMask {
            name: name.to_string(),
            weights,
        },
    );
    MaskId(entity)
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

fn layer(clip: ClipId, mask: MaskId) -> SkelLayer {
    SkelLayer {
        player: player(clip, 1.0),
        mask,
    }
}

/// Installs `layers` as a root stack, bottom-up (see [`SkelLayerStack`]).
fn stack(layers: Vec<SkelLayer>) -> SkelLayerStack {
    SkelLayerStack { layers }
}

/// Skeleton root with one joint whose authored rest pose is `rest`.
fn rig_with_rest(engine: &mut Engine, rest: JointTrs) -> Entity {
    let names = vec!["root".to_string()];
    let mut skeleton = Skeleton::new(vec![None], vec![Mat4::IDENTITY], names);
    skeleton = skeleton.with_rest(vec![rest]);
    let store = store_mut(engine);
    let root = store.create_entity();
    store.insert(root, TransformDesc::IDENTITY);
    store.insert(root, skeleton);
    store.insert(root, JointPose::identity(1));
    root
}

fn pose(engine: &Engine, root: Entity) -> Vec<Mat4> {
    store_ref(engine)
        .read_lane::<JointPose>()
        .expect("lane registered")
        .get(root)
        .expect("pose kept")
        .matrices
        .clone()
}

fn fade_of(engine: &Engine, root: Entity) -> Option<SkelFade> {
    store_ref(engine)
        .read_lane::<SkelFade>()
        .expect("lane registered")
        .get(root)
        .copied()
}

fn animator(engine: &mut Engine, root: Entity) -> AnimatorMut<'_> {
    try_animator(store_mut(engine), root).expect("animator present")
}

fn origin(matrix: Mat4) -> Vec3 {
    matrix.transform_point3(Vec3::ZERO)
}

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < 1e-4
}

/// Asserts where joint `joint` of the pose on `root` ends up.
fn assert_origin(engine: &Engine, root: Entity, joint: usize, expected: Vec3) {
    let got = origin(pose(engine, root)[joint]);
    assert!(close(got, expected), "joint {joint}: got {got:?}");
}

fn has_fade(engine: &Engine, root: Entity) -> bool {
    fade_of(engine, root).is_some()
}

/// Playlist clip: joint 0 rests at the origin.
fn idle_clip(engine: &mut Engine) -> ClipId {
    add_clip(engine, clip("idle", vec![constant(0, Vec3::ZERO)]))
}

/// Playlist clip: joint 0 sits at x = 4.
fn walk_clip(engine: &mut Engine) -> ClipId {
    add_clip(
        engine,
        clip("walk", vec![constant(0, Vec3::new(4.0, 0.0, 0.0))]),
    )
}

#[test]
fn half_weight_pulls_the_pose_halfway_to_rest() {
    let mut engine = engine();
    let root = rig(&mut engine, vec![None]);
    let walk = walk_clip(&mut engine);
    put(&mut engine, root, player(walk, 0.5));

    engine.run_frame(0.1);

    assert_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
}

#[test]
fn untracked_joint_reads_its_authored_rest_pose() {
    let mut engine = engine();
    let rest = JointTrs {
        translation: Vec3::new(2.0, 0.0, 0.0),
        ..JointTrs::IDENTITY
    };
    let root = rig_with_rest(&mut engine, rest);
    let empty = add_clip(&mut engine, clip("empty", Vec::new()));
    put(&mut engine, root, player(empty, 1.0));

    engine.run_frame(0.1);

    // No track drives the joint: at full weight it sits at its node pose,
    // not at identity (which would deform a bind-posed mesh).
    assert_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
}

#[test]
fn half_weight_blends_toward_the_authored_rest_not_identity() {
    let mut engine = engine();
    let rest = JointTrs {
        translation: Vec3::new(2.0, 0.0, 0.0),
        ..JointTrs::IDENTITY
    };
    let root = rig_with_rest(&mut engine, rest);
    let walk = walk_clip(&mut engine);
    put(&mut engine, root, player(walk, 0.5));

    engine.run_frame(0.1);

    // rest + 0.5 · (4 − 2) = 3; blending toward identity would give 2.
    assert_origin(&engine, root, 0, Vec3::new(3.0, 0.0, 0.0));
}

#[test]
fn half_weight_rotation_slerps_halfway_to_the_authored_rest() {
    let mut engine = engine();
    let rest = JointTrs {
        rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_3),
        ..JointTrs::IDENTITY
    };
    let root = rig_with_rest(&mut engine, rest);
    let upright_key = Key {
        time: 0.0,
        value: Quat::IDENTITY,
    };
    let track = JointTrack {
        joint: JointId::from_raw(0),
        node: None,
        translation: KeyTrack::linear(Vec::new()),
        rotation: KeyTrack::linear(vec![upright_key]),
        scale: KeyTrack::linear(Vec::new()),
    };
    let upright = add_clip(&mut engine, clip("upright", vec![track]));
    put(&mut engine, root, player(upright, 0.5));

    engine.run_frame(0.1);

    // Halfway from the 60° rest back to identity: 30°, not identity.
    let (_, rotation, _) = pose(&engine, root)[0].to_scale_rotation_translation();
    let halfway = Quat::from_rotation_y(std::f32::consts::FRAC_PI_6);
    assert!(rotation.angle_between(halfway) < 1e-4, "got {rotation:?}");
}

#[test]
fn crossfade_mixes_both_clips_then_retires() {
    let mut engine = engine();
    let root = rig(&mut engine, vec![None]);
    let idle = idle_clip(&mut engine);
    let walk = walk_clip(&mut engine);
    put(&mut engine, root, player(walk, 1.0));
    let fade = SkelFade {
        from: player(idle, 1.0),
        duration: Seconds::new(2.0),
        elapsed: Seconds::ZERO,
    };
    put(&mut engine, root, fade);

    // Halfway through a 2 s fade: the outgoing and incoming poses average.
    engine.run_frame(1.0);
    assert_origin(&engine, root, 0, Vec3::new(2.0, 0.0, 0.0));
    assert!(has_fade(&engine, root), "fade still running");

    // Past the end: the incoming clip alone, and the fade component is gone.
    engine.run_frame(1.0);
    assert_origin(&engine, root, 0, Vec3::new(4.0, 0.0, 0.0));
    assert!(!has_fade(&engine, root), "finished fade is removed");
}

#[test]
fn layer_lerps_only_the_masked_joints() {
    let mut engine = engine();
    // Joint 1 is a child of joint 0, so its origin is joint 0's origin plus
    // its own local translation.
    let root = rig(&mut engine, vec![None, Some(JointId::from_raw(0))]);
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
    let upper = vec![Clamped01::ONE, Clamped01::ZERO];
    let mask = add_mask(&mut engine, "upper", upper);
    put(&mut engine, root, player(base, 1.0));
    put(&mut engine, root, stack(vec![layer(wave, mask)]));

    engine.run_frame(0.1);

    // Masked joint 0 takes the layer; joint 1 (mask 0) keeps the base pose.
    assert_origin(&engine, root, 0, Vec3::new(0.0, 8.0, 0.0));
    assert_origin(&engine, root, 1, Vec3::new(4.0, 8.0, 0.0));
}

#[test]
fn mask_length_mismatch_skips_the_entity() {
    let mut engine = engine();
    let root = rig(&mut engine, vec![None]);
    let base = walk_clip(&mut engine);
    let wave = add_clip(
        &mut engine,
        clip("wave", vec![constant(0, Vec3::new(0.0, 8.0, 0.0))]),
    );
    let wrong = vec![Clamped01::ONE, Clamped01::ONE];
    let mask = add_mask(&mut engine, "wrong", wrong);
    put(&mut engine, root, player(base, 1.0));
    put(&mut engine, root, stack(vec![layer(wave, mask)]));

    engine.run_frame(0.1);

    // The pose is never published: the lane keeps its identity placeholder.
    assert_eq!(pose(&engine, root)[0], Mat4::IDENTITY);
}

/// Single-joint rig whose animator knows "idle" and "walk" and plays "idle".
fn idle_walk_animator(engine: &mut Engine) -> (Entity, ClipId) {
    let root = rig(engine, vec![None]);
    let idle = idle_clip(engine);
    let walk = walk_clip(engine);
    let clips = [("idle".to_string(), idle), ("walk".to_string(), walk)]
        .into_iter()
        .collect();
    put(engine, root, Animator::new(clips, vec![root]));
    animator(engine, root).play("idle").unwrap();
    (root, idle)
}

#[test]
fn animator_crossfade_is_cut_by_play_and_stop() {
    let mut engine = engine();
    let (root, idle) = idle_walk_animator(&mut engine);
    animator(&mut engine, root)
        .play_crossfade("walk", Seconds::new(1.0))
        .unwrap();

    let fade = fade_of(&engine, root).expect("crossfade started");
    assert_eq!(fade.from.clip, idle);
    assert_eq!(fade.duration, Seconds::new(1.0));

    animator(&mut engine, root).play("walk").unwrap();
    assert!(!has_fade(&engine, root), "play cuts the fade");

    animator(&mut engine, root)
        .play_crossfade("idle", Seconds::new(1.0))
        .unwrap();
    assert!(has_fade(&engine, root), "crossfade restarted");
    animator(&mut engine, root).stop().unwrap();
    assert!(!has_fade(&engine, root), "stop cancels the fade");
}

/// A layer alone drives joint 1, which the base clip never touches. The node
/// bound to that joint must follow the blended pose, not keep its bind pose.
#[test]
fn layer_only_joint_moves_its_bound_node() {
    let mut engine = engine();
    let node = store_mut(&mut engine).create_entity();
    put(&mut engine, node, Transform::IDENTITY);
    let root = rig(&mut engine, vec![None, Some(JointId::from_raw(0))]);

    let base = add_clip(&mut engine, clip("idle", vec![constant(0, Vec3::ZERO)]));
    let wave_track = JointTrack {
        node: Some(node),
        ..constant(1, Vec3::new(4.0, 0.0, 0.0))
    };
    let wave = add_clip(&mut engine, clip("wave", vec![wave_track]));
    let mask = add_mask(&mut engine, "all", vec![Clamped01::ONE, Clamped01::ONE]);
    put(&mut engine, root, player(base, 1.0));
    put(&mut engine, root, stack(vec![layer(wave, mask)]));

    engine.run_frame(0.5);

    let moved = store_ref(&engine)
        .read_lane::<Transform>()
        .expect("lane registered")
        .get(node)
        .expect("node keeps its transform")
        .translation;
    assert!(close(moved, Vec3::new(4.0, 0.0, 0.0)), "got {moved:?}");
    assert_origin(&engine, root, 1, Vec3::new(4.0, 0.0, 0.0));
}

/// Two full-mask layers over one base: the stack blends bottom-up, so the
/// top layer owns the joint where both claim it (a top-down application
/// would leave the bottom value).
#[test]
fn layers_stack_bottom_up() {
    {
        let mut engine = engine();
        let root = rig(&mut engine, vec![None]);
        let base = walk_clip(&mut engine);
        let lower = add_clip(
            &mut engine,
            clip("lower", vec![constant(0, Vec3::new(0.0, 8.0, 0.0))]),
        );
        let upper = add_clip(
            &mut engine,
            clip("upper", vec![constant(0, Vec3::new(0.0, 2.0, 0.0))]),
        );
        let full = add_mask(&mut engine, "full", vec![Clamped01::ONE]);
        put(&mut engine, root, player(base, 1.0));
        put(
            &mut engine,
            root,
            stack(vec![layer(lower, full), layer(upper, full)]),
        );

        engine.run_frame(0.1);

        assert_origin(&engine, root, 0, Vec3::new(0.0, 2.0, 0.0));
    }
}

/// Upper-body over locomotion: the base walk drives every joint, a bottom
/// layer retargets the legs, the top layer raises the arms — all three at
/// once, each where its mask says.
#[test]
fn upper_body_layer_rides_over_locomotion_stack() {
    {
        let mut engine = engine();
        let root = rig(
            &mut engine,
            vec![None, Some(JointId::from_raw(0)), Some(JointId::from_raw(0))],
        );
        let walk = add_clip(
            &mut engine,
            clip(
                "walk",
                vec![
                    constant(0, Vec3::new(4.0, 0.0, 0.0)),
                    constant(1, Vec3::new(4.0, 0.0, 0.0)),
                    constant(2, Vec3::new(4.0, 0.0, 0.0)),
                ],
            ),
        );
        let stride = add_clip(
            &mut engine,
            clip("stride", vec![constant(1, Vec3::new(0.0, 6.0, 0.0))]),
        );
        let raise = add_clip(
            &mut engine,
            clip("raise", vec![constant(2, Vec3::new(0.0, 0.0, 7.0))]),
        );
        let legs = add_mask(
            &mut engine,
            "legs",
            vec![Clamped01::ZERO, Clamped01::ONE, Clamped01::ZERO],
        );
        let arms = add_mask(
            &mut engine,
            "arms",
            vec![Clamped01::ZERO, Clamped01::ZERO, Clamped01::ONE],
        );
        put(&mut engine, root, player(walk, 1.0));
        put(
            &mut engine,
            root,
            stack(vec![layer(stride, legs), layer(raise, arms)]),
        );

        engine.run_frame(0.1);

        assert_origin(&engine, root, 0, Vec3::new(4.0, 0.0, 0.0));
        assert_origin(&engine, root, 1, Vec3::new(4.0, 6.0, 0.0));
        assert_origin(&engine, root, 2, Vec3::new(4.0, 0.0, 7.0));
    }
}

/// Stack cursors are advanced like base players: each element keeps its
/// own clock, stepped at its own `speed`.
#[test]
fn stack_layer_cursors_advance_at_their_own_speeds() {
    {
        let mut engine = engine();
        let root = rig(&mut engine, vec![None]);
        let base = walk_clip(&mut engine);
        let shared = add_clip(&mut engine, clip("shared", Vec::new()));
        let mask = add_mask(&mut engine, "full", vec![Clamped01::ONE]);
        let mut slow = layer(shared, mask);
        slow.player.speed = Seconds::new(1.0);
        let mut fast = layer(shared, mask);
        fast.player.speed = Seconds::new(2.0);
        put(&mut engine, root, player(base, 1.0));
        put(&mut engine, root, stack(vec![slow, fast]));

        engine.run_frame(0.1);

        let lane = store_ref(&engine)
            .read_lane::<SkelLayerStack>()
            .expect("lane registered");
        let times: Vec<f32> = lane
            .get(root)
            .expect("stack kept")
            .layers
            .iter()
            .map(|layer| layer.player.time.get())
            .collect();
        assert_eq!(times.len(), 2);
        assert!((times[0] - 0.1).abs() < 1e-5, "slow got {times:?}");
        assert!((times[1] - 0.2).abs() < 1e-5, "fast got {times:?}");
    }
}

/// A bad mask anywhere in the stack rejects the whole root (bad skin):
/// the pose is never half-published from the layers below it.
#[test]
fn bad_mask_on_a_stacked_layer_skips_the_root() {
    {
        let mut engine = engine();
        let root = rig(&mut engine, vec![None]);
        let base = walk_clip(&mut engine);
        let lower = add_clip(
            &mut engine,
            clip("lower", vec![constant(0, Vec3::new(0.0, 8.0, 0.0))]),
        );
        let upper = add_clip(
            &mut engine,
            clip("upper", vec![constant(0, Vec3::new(0.0, 2.0, 0.0))]),
        );
        let full = add_mask(&mut engine, "full", vec![Clamped01::ONE]);
        let wrong = add_mask(&mut engine, "wrong", vec![Clamped01::ONE, Clamped01::ONE]);
        put(&mut engine, root, player(base, 1.0));
        put(
            &mut engine,
            root,
            stack(vec![layer(lower, full), layer(upper, wrong)]),
        );

        engine.run_frame(0.1);

        assert_eq!(pose(&engine, root)[0], Mat4::IDENTITY);
    }
}
