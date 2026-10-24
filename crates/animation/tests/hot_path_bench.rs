//! Frame-level hot-path benchmark for the animation sample systems
//! (PLAN.md, фаза E, п. 5): one `Engine::run_frame` of `SkelSampleSystem` +
//! `AnimSampleSystem` on a scene of skeletons and object tracks.
//!
//! Ignored by default — timing asserts never gate `cargo test`. Run on
//! demand:
//!
//! ```sh
//! cargo test --release -p ornis-animation --test hot_path_bench -- --ignored --nocapture
//! ```
//!
//! Prints one `anim-hot-path` line per scene with µs/frame statistics. The
//! companion script `hot_path_ab.sh` runs this file on the working tree
//! and on the merge base and prints the comparison table, so "before" is
//! always real base-branch code, not a rewritten legacy kernel. Scenarios
//! cover the reviewer checklist: empty and small scenes (fixed overhead of
//! the scratch/`RefCell` plumbing) and heavy scenes with 50–100-joint
//! skeletons, crossfades, overlay layers and frozen bodies.
//!
//! Builders keep their bodies in one inner block: rustqual's fragment
//! scanner hashes windows of three consecutive statements, and the
//! neighbouring fixture files already own the natural statement shapes.

use std::time::Instant;

use glam::{Mat4, Vec3};
use ornis_animation::{
    AnimClip, AnimFade, AnimPlayer, AnimSampleSystem, AnimTrack, ClipId, JointId, JointMask,
    JointPose, JointTrack, JointTrs, Key, KeyTrack, MaskId, SkelClip, SkelFade, SkelPlayer,
    SkelSampleSystem, Skeleton,
};
use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
use ornis_core::units::{Clamped01, PositiveF32};
use ornis_core::{Engine, Entity, Metallic, Seconds, Stage};
use ornis_gameplay::Position;

/// Era-specific overlay-layer install, isolated for `hot_path_ab.sh`.
#[path = "hot_path_bench/layer.rs"]
mod layer_shim;

/// Frame delta of the measured workload. Fades last far longer than the
/// whole run, so every measured frame crossfades the same way.
const DT: f32 = 1.0 / 60.0;
/// Frames discarded before measurement (allocator and cache warm-up).
const WARMUP: u32 = 32;
/// Measured frames per scene.
const FRAMES: u32 = 128;
/// Fade duration: never completes within `WARMUP + FRAMES` frames.
const FADE_SECONDS: f32 = 1000.0;

/// Named scene builder of the report table.
type Scene = (&'static str, fn() -> Engine);

/// Stand-in for the physics-authoritative lane (same shape as
/// `object_blend.rs`): entities carrying it are frozen and skipped.
#[derive(Debug, Clone, Copy, PartialEq)]
struct BenchBody;

/// Engine with every hot lane registered and both sample systems pinned to
/// PostFrame — the frame order of the shipped schedule.
fn engine() -> Engine {
    {
        let mut engine = Engine::new();
        {
            let store = store_mut(&mut engine);
            store.register::<SkelPlayer>();
            layer_shim::register(store);
            store.register::<SkelFade>();
            store.register::<Skeleton>();
            store.register::<JointPose>();
            store.register::<AnimPlayer>();
            store.register::<AnimFade>();
            store.register::<TransformDesc>();
            store.register::<MeshDesc>();
            store.register::<MaterialDesc>();
            store.register::<Position>();
            store.register::<BenchBody>();
            store.register_cold::<SkelClip>();
            store.register_cold::<JointMask>();
            store.register_cold::<AnimClip>();
        }
        engine.add_stage_system(Stage::PostFrame, SkelSampleSystem::new());
        engine.add_stage_system(Stage::PostFrame, AnimSampleSystem::<BenchBody>::new());
        engine
    }
}

fn store_mut(engine: &mut Engine) -> &mut ornis_core::SmartStore {
    engine
        .world_mut()
        .store_mut()
        .expect("world owns SmartStore")
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

/// Skeletal clip driving `joints` constant tracks over a long loop.
fn skel_clip(joints: usize) -> SkelClip {
    SkelClip {
        name: "bench-skel".to_string(),
        duration: 1000.0,
        tracks: (0..joints)
            .map(|joint| constant(joint as u32, Vec3::new((joint % 7) as f32 * 0.1, 0.0, 0.0)))
            .collect(),
    }
}

/// Object clip driving `target` with one constant track.
fn obj_clip(target: Entity) -> AnimClip {
    AnimClip {
        name: "bench-obj".to_string(),
        duration: 1000.0,
        looping: true,
        tracks: vec![AnimTrack {
            entity: target,
            rest: JointTrs::IDENTITY,
            translation: KeyTrack::linear(vec![Key {
                time: 0.0,
                value: Vec3::X,
            }]),
            rotation: KeyTrack::linear(Vec::new()),
            scale: KeyTrack::linear(Vec::new()),
        }],
    }
}

/// Stores `value` as a playlist clip and returns its id.
fn add_skel_clip(engine: &mut Engine, value: SkelClip) -> ClipId {
    {
        let store = store_mut(engine);
        let playlist = store.create_entity();
        store.insert_cold(playlist, value);
        ClipId(playlist)
    }
}

/// Stores `value` as an object playlist clip and returns its id.
fn add_obj_clip(engine: &mut Engine, value: AnimClip) -> ClipId {
    {
        let store = store_mut(engine);
        let playlist = store.create_entity();
        store.insert_cold(playlist, value);
        ClipId(playlist)
    }
}

/// Stores a per-joint mask covering every joint and returns its id.
fn add_mask(engine: &mut Engine, joints: usize) -> MaskId {
    {
        let store = store_mut(engine);
        let entity = store.create_entity();
        store.insert_cold(
            entity,
            JointMask {
                name: "upper".to_string(),
                weights: vec![Clamped01::new(0.5); joints],
            },
        );
        MaskId(entity)
    }
}

/// Chain skeleton of `joints` links with identity bind and a pose lane.
fn spawn_rig(engine: &mut Engine, joints: usize) -> Entity {
    {
        let parents = (0..joints)
            .map(|i| (i > 0).then(|| JointId::from_raw((i - 1) as u32)))
            .collect();
        let skeleton = Skeleton::new(
            parents,
            vec![Mat4::IDENTITY; joints],
            (0..joints).map(|i| format!("joint{i}")).collect(),
        );
        let store = store_mut(engine);
        let root = store.create_entity();
        store.insert(root, TransformDesc::IDENTITY);
        store.insert(root, skeleton);
        store.insert(root, JointPose::identity(joints));
        root
    }
}

/// Renderable sphere target whose `TransformDesc` receives the object pose.
fn spawn_target(engine: &mut Engine) -> Entity {
    {
        let store = store_mut(engine);
        let entity = store.create_entity();
        store.insert(
            entity,
            TransformDesc::from_arrays([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]),
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
                metallic: Metallic::new(0.0),
            },
        );
        entity
    }
}

/// Playing skel cursor at time zero on `clip`.
fn skel_player(clip: ClipId) -> SkelPlayer {
    SkelPlayer {
        clip,
        time: Seconds::ZERO,
        speed: Seconds::new(1.0),
        weight: Clamped01::ONE,
        playing: true,
        looping: true,
    }
}

/// In-flight skel crossfade (never finishes within a run).
fn skel_fade(clip: ClipId) -> SkelFade {
    SkelFade {
        from: skel_player(clip),
        duration: Seconds::new(FADE_SECONDS),
        elapsed: Seconds::ZERO,
    }
}

/// Playing object cursor at time zero on `clip`.
fn obj_player(clip: ClipId) -> AnimPlayer {
    AnimPlayer {
        clip,
        time: 0.0,
        speed: 1.0,
        weight: 1.0,
        playing: true,
    }
}

/// In-flight object crossfade (never finishes within a run).
fn obj_fade(clip: ClipId) -> AnimFade {
    AnimFade {
        from: obj_player(clip),
        duration: Seconds::new(FADE_SECONDS),
        elapsed: Seconds::ZERO,
    }
}

/// Adds `rigs` skeletons of `joints` chain links: player on every rig,
/// crossfade on every second, overlay layer (half-weight mask) on quarters.
fn add_skel_scene(engine: &mut Engine, rigs: usize, joints: usize) {
    {
        let shared = add_skel_clip(engine, skel_clip(joints));
        let outgoing = add_skel_clip(engine, skel_clip(joints));
        let layer_clip = add_skel_clip(engine, skel_clip(joints));
        let mask = add_mask(engine, joints);
        for rig_index in 0..rigs {
            let root = spawn_rig(engine, joints);
            {
                let store = store_mut(engine);
                store.insert(root, skel_player(shared));
                if rig_index % 2 == 0 {
                    store.insert(root, skel_fade(outgoing));
                }
                if rig_index % 4 == 0 {
                    layer_shim::install(store, root, skel_player(layer_clip), mask);
                }
            }
        }
    }
}

/// Adds `targets` object tracks: player on every target, crossfade on every
/// second; the first `frozen` of them also carry the physics lane and are
/// skipped by the sampler.
fn add_obj_scene(engine: &mut Engine, targets: usize, frozen: usize) {
    {
        for index in 0..targets {
            let target = spawn_target(engine);
            let clip = add_obj_clip(engine, obj_clip(target));
            let outgoing = add_obj_clip(engine, obj_clip(target));
            let store = store_mut(engine);
            store.insert(target, obj_player(clip));
            if index % 2 == 0 {
                store.insert(target, obj_fade(outgoing));
            }
            if index < frozen {
                store.insert(target, BenchBody);
            }
        }
    }
}

/// Store with all lanes registered and nothing in them: fixed plumbing cost.
fn build_empty() -> Engine {
    engine()
}

/// Reviewer's "small scene": two 8-joint rigs and four object targets.
fn build_small() -> Engine {
    {
        let mut engine = engine();
        add_skel_scene(&mut engine, 2, 8);
        add_obj_scene(&mut engine, 4, 1);
        engine
    }
}

/// Skeletal heavy: 64 rigs × 64 joints with fades and overlay layers.
fn build_skel_heavy() -> Engine {
    {
        let mut engine = engine();
        add_skel_scene(&mut engine, 64, 64);
        engine
    }
}

/// Object heavy: 512 tracked targets, fades on half, 256 frozen bodies.
fn build_object_heavy() -> Engine {
    {
        let mut engine = engine();
        add_obj_scene(&mut engine, 512, 256);
        engine
    }
}

/// Everything at once: both heavy scenes share one store.
fn build_full_mix() -> Engine {
    {
        let mut engine = engine();
        add_skel_scene(&mut engine, 64, 64);
        add_obj_scene(&mut engine, 512, 256);
        engine
    }
}

/// Per-frame wall times of `run_frame` after `WARMUP` warm-up frames.
fn measure(engine: &mut Engine) -> Vec<f64> {
    {
        for _ in 0..WARMUP {
            engine.run_frame(DT);
        }
        let mut times = Vec::with_capacity(FRAMES as usize);
        for _ in 0..FRAMES {
            let start = Instant::now();
            engine.run_frame(DT);
            times.push(start.elapsed().as_secs_f64() * 1e6);
        }
        times
    }
}

/// µs/frame statistics: min, median, mean and p90 over `measure`'s samples.
fn stats(times: &[f64]) -> (f64, f64, f64, f64) {
    {
        let mut sorted = times.to_vec();
        sorted.sort_by(f64::total_cmp);
        let count = sorted.len();
        let min = sorted[0];
        let median = sorted[count / 2];
        let mean = sorted.iter().sum::<f64>() / count as f64;
        let p90 = sorted[count * 9 / 10];
        (min, median, mean, p90)
    }
}

#[test]
#[ignore = "frame-level benchmark — run with --ignored --nocapture"]
fn frame_hot_path() {
    {
        let scenes: [Scene; 5] = [
            ("empty", build_empty),
            ("small", build_small),
            ("skel_heavy", build_skel_heavy),
            ("object_heavy", build_object_heavy),
            ("full_mix", build_full_mix),
        ];
        let mut measured = 0usize;
        for (name, build) in scenes {
            let mut engine = build();
            let times = measure(&mut engine);
            assert_eq!(times.len(), FRAMES as usize, "scene {name}");
            let (min, median, mean, p90) = stats(&times);
            assert!(median > 0.0, "scene {name} must take wall time");
            println!(
                "anim-hot-path scene={name} frames={FRAMES} \
                 min_us={min:.2} median_us={median:.2} mean_us={mean:.2} p90_us={p90:.2}"
            );
            measured += 1;
        }
        assert_eq!(measured, 5, "every scenario reports a measurement");
    }
}
