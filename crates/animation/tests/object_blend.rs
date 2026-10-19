//! Phase E object-track blending (`docs/animation-design.md` §5 E): object
//! players blend toward the authored [`AnimTrack::rest`] at weight below
//! `1`, leave channels no track provides untouched, and crossfade through an
//! [`AnimFade`] component on the entity.
//!
//! Same engine shape as `anim_object.rs`: the sampler runs on PostFrame over
//! a local stand-in physics lane, and the target is a renderable sphere whose
//! [`TransformDesc`] receives the pose.

use glam::{Quat, Vec3};
use ornis_animation::{
    AnimClip, AnimFade, AnimPlayer, AnimSampleSystem, AnimTrack, ClipId, JointTrs, Key, KeyTrack,
};
use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
use ornis_core::units::{Clamped01, PositiveF32};
use ornis_core::{Engine, Entity, Metallic, Seconds, Stage};
use ornis_gameplay::Position;

/// Stand-in for the physics-authoritative lane (see `anim_object.rs`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct TestBody;

fn engine() -> Engine {
    let mut engine = Engine::new();
    {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        store.register::<AnimPlayer>();
        store.register::<AnimFade>();
        store.register::<TransformDesc>();
        store.register::<MeshDesc>();
        store.register::<MaterialDesc>();
        store.register::<Position>();
        store.register::<TestBody>();
        store.register_cold::<AnimClip>();
    }
    engine.add_stage_system(Stage::PostFrame, AnimSampleSystem::<TestBody>::new());
    engine
}

/// Engine plus one renderable target placed at `placement`.
fn scene_at(placement: TransformDesc) -> (Engine, Entity) {
    let mut engine = engine();
    let target = spawn(&mut engine, placement);
    (engine, target)
}

/// Engine plus one target at the origin with unit scale.
fn scene() -> (Engine, Entity) {
    let placement = TransformDesc::from_arrays([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]);
    scene_at(placement)
}

/// Inserts `value` on `entity` in the world's store.
fn insert<T: Clone + Send + Sync + 'static>(engine: &mut Engine, entity: Entity, value: T) {
    let store = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore");
    store.insert(entity, value);
}

/// Renderable sphere placed at `placement`; the sampler writes its
/// [`TransformDesc`] when a track drives it.
fn spawn(engine: &mut Engine, placement: TransformDesc) -> Entity {
    let entity = {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        store.create_entity()
    };
    insert(engine, entity, placement);
    insert(
        engine,
        entity,
        MeshDesc::Sphere {
            radius: PositiveF32::expect_valid(1.0),
            segments: 16,
            rings: 8,
        },
    );
    insert(
        engine,
        entity,
        MaterialDesc::Dielectric {
            base_color: [0.5, 0.5, 0.5],
            roughness: Clamped01::new(0.9),
            emission: [0.0, 0.0, 0.0],
            metallic: Metallic::new(0.0),
        },
    );
    entity
}

/// Starts a one-second fade on `target` from `outgoing` at full weight.
fn fade_from(engine: &mut Engine, target: Entity, outgoing: ClipId) {
    let fade = AnimFade {
        from: player(outgoing, 1.0),
        duration: Seconds::new(1.0),
        elapsed: Seconds::ZERO,
    };
    insert(engine, target, fade);
}

/// Constant track: two keys holding `value` over `0..1`, or no keys.
fn held<T: Copy>(value: Option<T>) -> KeyTrack<T> {
    let Some(value) = value else {
        return KeyTrack::linear(Vec::new());
    };
    let keys = vec![Key { time: 0.0, value }, Key { time: 1.0, value }];
    KeyTrack::linear(keys)
}

/// Track on `target` with constant translation and rotation channels.
fn track(
    target: Entity,
    rest: JointTrs,
    translation: Option<Vec3>,
    rotation: Option<Quat>,
) -> AnimTrack {
    AnimTrack {
        entity: target,
        rest,
        translation: held(translation),
        rotation: held(rotation),
        scale: KeyTrack::linear(Vec::new()),
    }
}

/// Playlist entity holding a one-track looping clip (cold lane).
fn add_clip(engine: &mut Engine, name: &str, track: AnimTrack) -> ClipId {
    let entity = {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        store.create_entity()
    };
    let clip = AnimClip {
        name: name.to_string(),
        duration: 1.0,
        looping: true,
        tracks: vec![track],
    };
    let store = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore");
    store.insert_cold(entity, clip);
    ClipId(entity)
}

/// Playing cursor on `clip` at time zero with the given weight.
fn player(clip: ClipId, weight: f32) -> AnimPlayer {
    AnimPlayer {
        clip,
        time: 0.0,
        speed: 1.0,
        weight,
        playing: true,
    }
}

fn play(engine: &mut Engine, target: Entity, clip: ClipId, weight: f32) {
    insert(engine, target, player(clip, weight));
}

fn transform_of(engine: &Engine, entity: Entity) -> TransformDesc {
    engine
        .world()
        .store()
        .expect("world owns SmartStore")
        .read_lane::<TransformDesc>()
        .expect("lane registered")
        .get(entity)
        .expect("entity placed")
        .clone()
}

fn fade_active(engine: &Engine, entity: Entity) -> bool {
    engine
        .world()
        .store()
        .expect("world owns SmartStore")
        .read_lane::<AnimFade>()
        .expect("lane registered")
        .get(entity)
        .is_some()
}

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < 1e-4
}

fn rest_at(translation: Vec3, rotation: Quat) -> JointTrs {
    JointTrs {
        translation,
        rotation,
        scale: Vec3::ONE,
    }
}

#[test]
fn weight_below_one_pulls_translation_toward_rest() {
    let (mut engine, target) = scene();
    let rest = rest_at(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY);
    let walk_track = track(target, rest, Some(Vec3::new(4.0, 0.0, 0.0)), None);
    let walk = add_clip(&mut engine, "walk", walk_track);
    play(&mut engine, target, walk, 0.5);
    engine.run_frame(0.1);
    let placed = transform_of(&engine, target).translation;
    assert!(close(placed, Vec3::new(3.0, 0.0, 0.0)), "got {placed:?}");
}

#[test]
fn zero_weight_holds_the_authored_rest_pose() {
    let placement = TransformDesc::from_arrays([9.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]);
    let (mut engine, target) = scene_at(placement);
    let tilt = Quat::from_rotation_y(std::f32::consts::FRAC_PI_3);
    let rest = rest_at(Vec3::new(2.0, 0.0, 0.0), tilt);
    let walk_track = track(
        target,
        rest,
        Some(Vec3::new(4.0, 0.0, 0.0)),
        Some(Quat::IDENTITY),
    );
    let walk = add_clip(&mut engine, "walk", walk_track);
    play(&mut engine, target, walk, 0.0);
    engine.run_frame(0.1);
    let desc = transform_of(&engine, target);
    assert!(close(desc.translation, Vec3::new(2.0, 0.0, 0.0)));
    assert!(desc.rotation.get().angle_between(tilt) < 1e-4);
}

#[test]
fn half_weight_rotation_slerps_halfway_to_rest() {
    let (mut engine, target) = scene();
    let tilt = Quat::from_rotation_y(std::f32::consts::FRAC_PI_3);
    let rest = rest_at(Vec3::ZERO, tilt);
    let idle_track = track(target, rest, None, Some(Quat::IDENTITY));
    let idle = add_clip(&mut engine, "idle", idle_track);
    play(&mut engine, target, idle, 0.5);
    engine.run_frame(0.1);
    let half = Quat::from_rotation_y(std::f32::consts::FRAC_PI_6);
    let placed = transform_of(&engine, target).rotation.get();
    assert!(placed.angle_between(half) < 1e-4, "got {placed:?}");
}

#[test]
fn absent_channels_stay_untouched_even_below_full_weight() {
    let placement = TransformDesc::from_arrays([0.0; 3], [0.0, 0.0, 0.0, 1.0], [2.0; 3]);
    let (mut engine, target) = scene_at(placement);
    // Rest scale is 1 but the track has no scale channel: placement survives.
    let rest = rest_at(Vec3::ZERO, Quat::IDENTITY);
    let slide_track = track(target, rest, Some(Vec3::new(1.0, 0.0, 0.0)), None);
    let slide = add_clip(&mut engine, "slide", slide_track);
    play(&mut engine, target, slide, 0.5);
    engine.run_frame(0.1);
    let scale = transform_of(&engine, target).scale;
    assert!(close(scale, Vec3::splat(2.0)), "got {scale:?}");
}

#[test]
fn crossfade_mixes_retires_and_falls_back_to_rest_without_a_track() {
    let (mut engine, target) = scene();
    let rest = rest_at(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY);
    let idle_track = track(target, rest, Some(Vec3::ZERO), None);
    let idle = add_clip(&mut engine, "idle", idle_track);
    let walk_track = track(target, rest, Some(Vec3::new(10.0, 0.0, 0.0)), None);
    let walk = add_clip(&mut engine, "walk", walk_track);
    play(&mut engine, target, walk, 1.0);
    fade_from(&mut engine, target, idle);

    engine.run_frame(0.5);
    let mid = transform_of(&engine, target).translation;
    assert!(close(mid, Vec3::new(5.0, 0.0, 0.0)), "midway got {mid:?}");
    assert!(fade_active(&engine, target));

    engine.run_frame(0.5);
    let end = transform_of(&engine, target).translation;
    assert!(close(end, Vec3::new(10.0, 0.0, 0.0)), "end got {end:?}");
    assert!(
        !fade_active(&engine, target),
        "finished fade must be dropped"
    );

    // Incoming clip without a track for `target`: the outgoing idle pose (0)
    // falls back toward its rest (2), so halfway reads 1.
    let placement = TransformDesc::from_arrays([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]);
    let other = spawn(&mut engine, placement);
    let elsewhere = track(other, rest, Some(Vec3::new(9.0, 0.0, 0.0)), None);
    let elsewhere_clip = add_clip(&mut engine, "elsewhere", elsewhere);
    play(&mut engine, target, elsewhere_clip, 1.0);
    fade_from(&mut engine, target, idle);
    engine.run_frame(0.5);
    let back = transform_of(&engine, target).translation;
    assert!(
        close(back, Vec3::new(1.0, 0.0, 0.0)),
        "fallback got {back:?}"
    );
}
