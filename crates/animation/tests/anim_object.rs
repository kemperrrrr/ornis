//! Phase A acceptance (design `docs/animation-design.md` §5): the object
//! animation sampler drives a sphere along its keys, ignores
//! physics-driven entities and needs no new extraction counters.
//!
//! The engine is built directly over [`ornis_core::Stage::PostFrame`] (the
//! `GameStage` dictionary is gone); `RigidBody` itself is not nameable here
//! (`ornis-render` must not depend on `ornis-physics`), so a local
//! stand-in lane plays the physics-authority role — the sampler is generic
//! over that lane and the owner wires the real body type at registration.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

use glam::{Quat, Vec3};
use ornis_animation::{AnimClip, AnimPlayer, AnimSampleSystem, AnimTrack, ClipId, Key, KeyTrack};
use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
use ornis_core::units::{Clamped01, PositiveF32};
use ornis_core::{Engine, Entity, Resources, Stage, System, SystemAccess};
use ornis_gameplay::Position;
use ornis_render::{ExtractionStats, extract_render_data_with_stats};

/// Stand-in for the physics-authoritative lane: `ornis-render` cannot name
/// `ornis_physics::RigidBody` (no such dependency by design), so tests use
/// this local marker while production wires the real body type.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TestBody;

/// Registers the animation lanes without adding any system (lets edge
/// tests control registration order: explicit edges only run forward).
fn register_anim_lanes(engine: &mut Engine) {
    let store = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore");
    store.register::<AnimPlayer>();
    store.register::<TransformDesc>();
    store.register::<MeshDesc>();
    store.register::<MaterialDesc>();
    store.register::<Position>();
    store.register::<TestBody>();
    store.register_cold::<AnimClip>();
}

/// Engine with animation lanes registered and `anim_sample` on PostFrame.
fn anim_engine() -> Engine {
    let mut engine = Engine::new();
    register_anim_lanes(&mut engine);
    engine.add_stage_system(Stage::PostFrame, AnimSampleSystem::<TestBody>::new());
    engine
}

/// Clip riding `target` from x=0 to x=10 over one second, rotating 0→90° (Y).
/// Non-looping; scale track empty (placement scale must survive).
fn ride_clip(target: Entity) -> AnimClip {
    AnimClip {
        name: "ride".to_string(),
        duration: 1.0,
        looping: false,
        tracks: vec![AnimTrack {
            entity: target,
            translation: KeyTrack::linear(vec![
                Key {
                    time: 0.0,
                    value: Vec3::ZERO,
                },
                Key {
                    time: 1.0,
                    value: Vec3::new(10.0, 0.0, 0.0),
                },
            ]),
            rotation: KeyTrack::linear(vec![
                Key {
                    time: 0.0,
                    value: Quat::IDENTITY,
                },
                Key {
                    time: 1.0,
                    value: Quat::from_rotation_y(FRAC_PI_2),
                },
            ]),
            scale: KeyTrack::linear(Vec::new()),
        }],
    }
}

/// Renderable sphere with a playing cursor over its own playlist entity.
fn spawn_sphere(engine: &mut Engine) -> Entity {
    let store = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore");
    let entity = store.create_entity();
    store.insert(
        entity,
        TransformDesc {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [2.0, 2.0, 2.0],
        },
    );
    store.insert(
        entity,
        MeshDesc::Sphere {
            radius: PositiveF32::expect_valid(1.0),
            segments: 16,
            rings: 8,
        },
    );
    store.insert(
        entity,
        MaterialDesc::Dielectric {
            base_color: [0.5, 0.5, 0.5],
            roughness: Clamped01::new(0.9),
            emission: [0.0, 0.0, 0.0],
        },
    );
    store.insert_cold(entity, ride_clip(entity));
    store.insert(
        entity,
        AnimPlayer {
            clip: ClipId(entity),
            time: 0.0,
            speed: 1.0,
            weight: 1.0,
            playing: true,
        },
    );
    entity
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

fn rotation_of(desc: &TransformDesc) -> Quat {
    Quat::from_xyzw(
        desc.rotation[0],
        desc.rotation[1],
        desc.rotation[2],
        desc.rotation[3],
    )
}

#[test]
fn sphere_rides_keys_and_track() {
    let mut engine = anim_engine();
    let sphere = spawn_sphere(&mut engine);

    engine.run_frame(0.5);
    let mid = transform_of(&engine, sphere);
    assert!(
        (mid.translation[0] - 5.0).abs() < 1e-4,
        "halfway x must be 5, got {:?}",
        mid.translation
    );
    assert!(
        rotation_of(&mid).angle_between(Quat::from_rotation_y(FRAC_PI_4)) < 1e-4,
        "halfway rotation must be 45°, got {:?}",
        mid.rotation
    );
    // Empty scale track leaves the entity size alone.
    assert_eq!(mid.scale, [2.0, 2.0, 2.0]);

    engine.run_frame(0.5);
    let end = transform_of(&engine, sphere);
    assert!((end.translation[0] - 10.0).abs() < 1e-4);
    assert!(rotation_of(&end).angle_between(Quat::from_rotation_y(FRAC_PI_2)) < 1e-4);

    // Non-looping clip clamps at the end, never wraps.
    engine.run_frame(1.0);
    let held = transform_of(&engine, sphere);
    assert!((held.translation[0] - 10.0).abs() < 1e-4);
}

#[test]
fn physics_entity_is_ignored() {
    let mut engine = anim_engine();
    let sphere = spawn_sphere(&mut engine);
    {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        store.insert(sphere, TestBody);
        store.insert(sphere, Position(Vec3::new(7.0, 7.0, 7.0)));
    }

    engine.run_frame(0.5);

    // Physics is authoritative: neither placement lane moves...
    let desc = transform_of(&engine, sphere);
    assert_eq!(desc.translation, [0.0, 0.0, 0.0]);
    assert_eq!(desc.rotation, [0.0, 0.0, 0.0, 1.0]);
    let position = engine
        .world()
        .store()
        .expect("world owns SmartStore")
        .read_lane::<Position>()
        .expect("lane registered")
        .get(sphere)
        .expect("position kept")
        .0;
    assert_eq!(position, Vec3::new(7.0, 7.0, 7.0));
    // ...and not even the playback cursor advances.
    let time = engine
        .world()
        .store()
        .expect("world owns SmartStore")
        .read_lane::<AnimPlayer>()
        .expect("lane registered")
        .get(sphere)
        .expect("player kept")
        .time;
    assert_eq!(time, 0.0);
}

#[test]
fn paused_player_holds_pose() {
    let mut engine = anim_engine();
    let sphere = spawn_sphere(&mut engine);
    engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
        .write_lane::<AnimPlayer>()
        .expect("lane registered")
        .get_mut(sphere)
        .expect("player placed")
        .playing = false;

    engine.run_frame(0.5);

    let desc = transform_of(&engine, sphere);
    assert_eq!(desc.translation, [0.0, 0.0, 0.0]);
}

#[test]
fn position_mirror_only_when_lane_exists() {
    let mut engine = anim_engine();
    let plain = spawn_sphere(&mut engine);
    let mirrored = spawn_sphere(&mut engine);
    engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
        .insert(mirrored, Position(Vec3::ZERO));

    engine.run_frame(0.5);

    let store = engine.world().store().expect("world owns SmartStore");
    let positions = store.read_lane::<Position>().expect("lane registered");
    assert_eq!(
        positions.get(mirrored).expect("mirror kept").0,
        Vec3::new(5.0, 0.0, 0.0)
    );
    // Animation never grows the lane: no Position invented for `plain`.
    assert_eq!(positions.len(), 1);
    assert!(positions.get(plain).is_none());
}

#[test]
fn marker_without_mesh_writes_position() {
    let mut engine = anim_engine();
    let marker = engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
        .create_entity();
    {
        let store = engine
            .world_mut()
            .store_mut()
            .expect("world owns SmartStore");
        store.insert(marker, Position(Vec3::ZERO));
        store.insert_cold(marker, ride_clip(marker));
        store.insert(
            marker,
            AnimPlayer {
                clip: ClipId(marker),
                time: 0.0,
                speed: 1.0,
                weight: 1.0,
                playing: true,
            },
        );
    }

    engine.run_frame(0.5);

    let position = engine
        .world()
        .store()
        .expect("world owns SmartStore")
        .read_lane::<Position>()
        .expect("lane registered")
        .get(marker)
        .expect("marker placed")
        .0;
    assert!((position - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-4);
}

#[test]
fn extraction_sees_final_pose_without_new_counters() {
    let mut engine = anim_engine();
    let _sphere = spawn_sphere(&mut engine);

    engine.run_frame(0.5);

    let store = engine.world().store().expect("world owns SmartStore");
    let (upload, stats) = extract_render_data_with_stats(store);
    // Object animation rides the existing instancing path: no skips, no stubs,
    // and no new honesty counters (every counter stays at zero).
    assert_eq!(stats, ExtractionStats::default());
    assert_eq!(stats.skipped_incomplete, 0);
    assert_eq!(stats.skipped_bad_custom, 0);
    assert_eq!(stats.skipped_unknown_mesh, 0);
    assert_eq!(stats.materials_deduped, 0);
    assert_eq!(upload.instances.len(), 1);
    let x = upload.instances[0].model_matrix.w_axis.x;
    assert!(
        (x - 5.0).abs() < 1e-4,
        "extracted instance must carry the animated pose, got x={x}"
    );
}

/// Declaration twin of the app-side `body_to_transform`: same name and lane
/// shape, no physics type (unnameable from `ornis-render`). Exists only to
/// prove the explicit WaW edge the design demands.
struct BodyStub;

impl System for BodyStub {
    fn name(&self) -> &'static str {
        "body_to_transform"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<ornis_core::SmartStore>()
            .reads_lane::<TestBody>()
            .writes_lane::<Position>()
            .writes_lane::<TransformDesc>()
    }

    fn run(&self, _resources: &Resources) {}
}

#[test]
fn explicit_edge_orders_body_before_anim() {
    let mut engine = Engine::new();
    register_anim_lanes(&mut engine);
    // Registration order is the conflict tie-break: stub first, then anim,
    // so the explicit edge runs forward.
    engine
        .stage_schedule_mut(Stage::PostFrame)
        .add_system(BodyStub);
    engine.add_stage_system(Stage::PostFrame, AnimSampleSystem::<TestBody>::new());
    engine
        .stage_schedule_mut(Stage::PostFrame)
        .try_order_before("body_to_transform", "anim_sample")
        .expect("explicit WaW edge body_to_transform -> anim_sample");

    let sphere = spawn_sphere(&mut engine);
    engine.run_frame(0.5);

    let desc = transform_of(&engine, sphere);
    assert!((desc.translation[0] - 5.0).abs() < 1e-4);
    let mermaid = engine.stage_schedule(Stage::PostFrame).mermaid();
    assert!(mermaid.contains("body_to_transform"));
    assert!(mermaid.contains("anim_sample"));
}
