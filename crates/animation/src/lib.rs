//! Object animation: cold [`AnimClip`] lanes plus hot [`AnimPlayer`] lanes,
//! sampled by the `anim_sample` system (phase A, `docs/animation-design.md`
//! §1 and §5).
//!
//! The sampler is frame-only: it belongs in [`ornis_core::Stage::PostFrame`]
//! on variable [`ornis_core::Time`], never in the fixed schedule, so the
//! visual pose is not multiplied by the number of fixed substeps. Routing
//! follows the design: renderable entities (those with a [`MeshDesc`] lane
//! entry) are written into [`TransformDesc`], with the translation mirrored
//! into the gameplay `Position` lane only when that lane already exists on the
//! entity; entities without [`MeshDesc`] (pure gameplay markers) are written
//! into [`Position`]. Lanes are never created for the sake of animation.
//! Entities with a physics-authoritative lane are skipped (physics wins).
//! Root motion is out of scope: gameplay-affecting motion travels the fixed
//! `Velocity → RigidBody` path, never this sampler.

use std::marker::PhantomData;

use glam::{Quat, Vec3};
use ornis_core::{Entity, Resources, SmartStore, System, SystemAccess, Time};
use ornis_gameplay::Position;

use ornis_assets::scene::{MeshDesc, TransformDesc};

/// One animation key: the channel value at `time` seconds from clip start.
///
/// Keys inside a [`KeyTrack`] must be sorted by ascending `time`; the sampler
/// clamps out-of-range lookups to the nearest end key.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Key<T> {
    /// Seconds from the clip start.
    pub time: f32,
    /// Channel value at `time`.
    pub value: T,
}

/// One sorted channel of keys (translation, rotation or scale).
///
/// An empty track means "channel absent": sampling returns [`None`] and the
/// sampler leaves the placement component untouched instead of zeroing it
/// (so entity size survives when no scale track exists).
#[derive(Debug, Clone, PartialEq)]
pub struct KeyTrack<T> {
    /// Keys sorted by ascending `time`.
    pub keys: Vec<Key<T>>,
}

impl KeyTrack<Vec3> {
    /// Samples the translation/scale channel at clip time `t` seconds.
    ///
    /// Linear interpolation between the surrounding keys, clamped to the
    /// end keys outside the key range. Returns [`None`] when the track is
    /// empty (channel absent).
    pub fn sample(&self, t: f32) -> Option<Vec3> {
        let (before, after, alpha) = segment(&self.keys, t)?;
        Some(before.value.lerp(after.value, alpha))
    }
}

impl KeyTrack<Quat> {
    /// Samples the rotation channel at clip time `t` seconds.
    ///
    /// Spherical interpolation ([`Quat::slerp`]) between the surrounding
    /// keys, clamped to the end keys outside the key range. Returns [`None`]
    /// when the track is empty (channel absent).
    pub fn sample(&self, t: f32) -> Option<Quat> {
        let (before, after, alpha) = segment(&self.keys, t)?;
        Some(before.value.slerp(after.value, alpha))
    }
}

/// Locates the key segment surrounding `t`: the two keys to blend plus the
/// blend factor in `0.0..=1.0`. Returns [`None`] for an empty track.
fn segment<T: Copy>(keys: &[Key<T>], t: f32) -> Option<(&Key<T>, &Key<T>, f32)> {
    if keys.is_empty() {
        return None;
    }
    if keys.len() == 1 {
        return Some((&keys[0], &keys[0], 0.0));
    }
    let upper = keys.partition_point(|key| key.time <= t);
    if upper == 0 {
        return Some((&keys[0], &keys[0], 0.0));
    }
    if upper >= keys.len() {
        let last = keys.len() - 1;
        return Some((&keys[last], &keys[last], 0.0));
    }
    let (before, after) = (&keys[upper - 1], &keys[upper]);
    let span = after.time - before.time;
    let alpha = if span > 1e-6 {
        ((t - before.time) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some((before, after, alpha))
}

/// Object-space tracks for one animated entity inside an [`AnimClip`].
#[derive(Debug, Clone, PartialEq)]
pub struct AnimTrack {
    /// Animated entity (playlist model: the clip lives on a playlist entity,
    /// each track points at its target).
    pub entity: Entity,
    /// Translation keys; empty means "leave translation untouched".
    pub translation: KeyTrack<Vec3>,
    /// Rotation keys (unit quaternions); empty means "leave rotation untouched".
    pub rotation: KeyTrack<Quat>,
    /// Scale keys; empty means "leave scale untouched".
    pub scale: KeyTrack<Vec3>,
}

/// Shareable object-animation clip: cold data, sampled often, changed rarely.
///
/// Lives in the cold lane (`SmartStore::register_cold` / `insert_cold`) on a
/// playlist entity; [`AnimPlayer`]s reference it through [`ClipId`].
#[derive(Debug, Clone, PartialEq)]
pub struct AnimClip {
    /// Human-readable label (diagnostics only; identity is the lane entity).
    pub name: String,
    /// Clip length in seconds; player time wraps or clamps against it.
    pub duration: f32,
    /// Whether player time wraps (`time % duration`) instead of clamping at
    /// the ends (design §1.1 `loop`; renamed — `loop` is a Rust keyword).
    pub looping: bool,
    /// Per-entity tracks; at most one track per entity is sampled (first match).
    pub tracks: Vec<AnimTrack>,
}

/// Handle of the playlist entity holding the [`AnimClip`] cold component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClipId(pub Entity);

/// Hot per-entity playback cursor into an [`AnimClip`].
///
/// `Clone + Send + Sync` per the hot-lane contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimPlayer {
    /// Playlist entity holding the [`AnimClip`] in the cold lane.
    pub clip: ClipId,
    /// Current clip time in seconds; advanced by `dt * speed` each PostFrame.
    pub time: f32,
    /// Playback rate multiplier (`1.0` = real time; negative scrubs backwards).
    pub speed: f32,
    /// Blend weight, reserved for phase E (crossfade/masks); ignored here.
    pub weight: f32,
    /// Paused players hold their pose: time does not advance and placement
    /// is not rewritten.
    pub playing: bool,
}

/// Default physics-authority lane for [`AnimSampleSystem`]: a lane that is
/// never populated, so nothing is ever skipped.
///
/// `ornis-render` must not depend on `ornis-physics` (dependency direction:
/// the app crate owns the cross-domain bridge), so the authoritative lane
/// arrives as the generic parameter `P`. The owner wires the real body type
/// at registration — `AnimSampleSystem::<RigidBody>` in the app crate —
/// while tests use a local stand-in lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoPhysics;

impl NoPhysics {
    /// Creates the never-populated marker (only useful as a type argument).
    pub fn value() -> Self {
        Self
    }
}

/// Object-animation sampler (system name `anim_sample`).
///
/// PostFrame-only visual pose: advances playing [`AnimPlayer`] cursors on
/// variable [`Time`], samples their [`AnimClip`] tracks and publishes the
/// pose as described in the module docs. Entities carrying the
/// physics-authoritative lane `P` are ignored entirely (not even their
/// cursor advances).
///
/// Ordering: register after `body_to_transform` and pin
/// `try_order_before("body_to_transform", "anim_sample")` — the WaW conflict
/// on [`TransformDesc`] / [`Position`] already separates the levels, the edge
/// makes the determinism explicit.
#[derive(Debug, Clone, Copy)]
pub struct AnimSampleSystem<P = NoPhysics> {
    physics: PhantomData<P>,
}

impl<P> AnimSampleSystem<P> {
    /// Creates the sampler; `P` is the physics-authoritative lane to skip
    /// (the real body type at the app wiring site, a stand-in in tests).
    pub fn new() -> Self {
        Self {
            physics: PhantomData,
        }
    }
}

impl<P> Default for AnimSampleSystem<P> {
    /// Creates the sampler (see [`AnimSampleSystem::new`]).
    fn default() -> Self {
        Self::new()
    }
}

impl<P: 'static + Send + Sync> System for AnimSampleSystem<P> {
    fn name(&self) -> &'static str {
        "anim_sample"
    }

    fn access(&self) -> SystemAccess {
        // Honest per schedule.rs:415-439 — every lane touched below is
        // declared. Two deliberate deviations from design §1.3, both forced
        // by enforcement, not taste: `AnimPlayer` is *written* (cursor
        // advance), so it must be `writes_lane`, not `reads_lane`; `MeshDesc`
        // is read for the renderable/marker routing. `AnimClip` is declared
        // although it travels the cold lane: the declaration only feeds level
        // planning (cold reads need no enforcement grant, schedule.rs:412-414).
        SystemAccess::new()
            .reads::<Time>()
            .reads::<SmartStore>()
            .reads_lane::<AnimClip>()
            .reads_lane::<MeshDesc>()
            .reads_lane::<P>()
            .writes_lane::<AnimPlayer>()
            .writes_lane::<TransformDesc>()
            .writes_lane::<Position>()
    }

    fn run(&self, resources: &Resources) {
        run_anim_sample::<P>(resources);
    }
}

/// Sampled pose for one entity: one entry per channel, [`None`] = absent.
struct SampledPose {
    translation: Option<Vec3>,
    rotation: Option<Quat>,
    scale: Option<Vec3>,
}

/// Advances a playback cursor; pure function of `(clip, time)` for determinism.
fn advance_player_time(current: f32, dt: f32, speed: f32, duration: f32, looping: bool) -> f32 {
    if duration <= 0.0 {
        return current;
    }
    let next = current + dt * speed;
    if looping {
        next.rem_euclid(duration)
    } else {
        next.clamp(0.0, duration)
    }
}

fn run_anim_sample<P: 'static + Send + Sync>(resources: &Resources) {
    let Some(time) = resources.get::<Time>() else {
        return;
    };
    let dt = time.delta_seconds();
    let Some(store) = resources.get::<SmartStore>() else {
        return;
    };

    // Snapshot of physics-driven entities (empty when the authority lane is
    // not registered): both passes below skip them, cursor included.
    let driven: Vec<Entity> = store
        .read_lane::<P>()
        .map(|lane| lane.entities.clone())
        .unwrap_or_default();

    // Pass 1 — advance playing cursors against their clip duration.
    {
        let Some(clips) = store.read_cold_lane::<AnimClip>() else {
            return;
        };
        let Some(mut players) = store.write_lane::<AnimPlayer>() else {
            return;
        };
        for index in 0..players.data.len() {
            let entity = players.entities[index];
            let player = &mut players.data[index];
            if !player.playing || driven.contains(&entity) {
                continue;
            }
            let Some(clip) = clips.get(player.clip.0) else {
                continue;
            };
            player.time =
                advance_player_time(player.time, dt, player.speed, clip.duration, clip.looping);
        }
    }
    sample_and_publish(resources, driven);
}

/// Samples playing cursors and publishes the poses.
///
/// `driven` is the snapshot of physics-driven entities to skip (empty when
/// no authority lane is registered).
fn sample_and_publish(resources: &Resources, driven: Vec<Entity>) {
    let Some(store) = resources.get::<SmartStore>() else {
        return;
    };
    // Route first, then publish: collect the work so no two lane guards overlap.
    struct Work {
        entity: Entity,
        pose: SampledPose,
        renderable: bool,
    }
    let work: Vec<Work> = {
        let Some(clips) = store.read_cold_lane::<AnimClip>() else {
            return;
        };
        let Some(players) = store.read_lane::<AnimPlayer>() else {
            return;
        };
        let meshes = store.read_lane::<MeshDesc>();
        let mut out = Vec::new();
        for (&entity, player) in players.entities.iter().zip(&players.data) {
            if !player.playing || driven.contains(&entity) {
                continue;
            }
            let Some(clip) = clips.get(player.clip.0) else {
                continue;
            };
            let Some(track) = clip.tracks.iter().find(|track| track.entity == entity) else {
                continue;
            };
            out.push(Work {
                entity,
                pose: SampledPose {
                    translation: track.translation.sample(player.time),
                    rotation: track.rotation.sample(player.time),
                    scale: track.scale.sample(player.time),
                },
                renderable: meshes
                    .as_ref()
                    .is_some_and(|lane| lane.get(entity).is_some()),
            });
        }
        out
    };
    if work.is_empty() {
        return;
    }
    let mut descs = store.write_lane::<TransformDesc>();
    let mut positions = store.write_lane::<Position>();
    if descs.is_none() && positions.is_none() {
        return;
    }
    for item in &work {
        if item.renderable
            && let Some(desc) = descs.as_mut().and_then(|lane| lane.get_mut(item.entity))
        {
            if let Some(translation) = item.pose.translation {
                desc.translation = translation.to_array();
            }
            if let Some(rotation) = item.pose.rotation {
                desc.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
            }
            if let Some(scale) = item.pose.scale {
                desc.scale = scale.to_array();
            }
        }
        // Translation mirror (renderables) or marker placement (no MeshDesc):
        // update only, never insert — animation must not grow the lane.
        if let Some(translation) = item.pose.translation
            && let Some(position) = positions
                .as_mut()
                .and_then(|lane| lane.get_mut(item.entity))
        {
            position.0 = translation;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::TypeId;

    fn vec_track(keys: &[(f32, Vec3)]) -> KeyTrack<Vec3> {
        KeyTrack {
            keys: keys
                .iter()
                .map(|&(time, value)| Key { time, value })
                .collect(),
        }
    }

    fn quat_track(keys: &[(f32, Quat)]) -> KeyTrack<Quat> {
        KeyTrack {
            keys: keys
                .iter()
                .map(|&(time, value)| Key { time, value })
                .collect(),
        }
    }

    #[test]
    fn system_name_is_anim_sample() {
        assert_eq!(AnimSampleSystem::<NoPhysics>::new().name(), "anim_sample");
    }

    #[test]
    fn access_declares_honest_lanes() {
        let access = AnimSampleSystem::<NoPhysics>::new().access();
        for lane in [
            TypeId::of::<AnimPlayer>(),
            TypeId::of::<AnimClip>(),
            TypeId::of::<MeshDesc>(),
            TypeId::of::<NoPhysics>(),
            TypeId::of::<TransformDesc>(),
            TypeId::of::<Position>(),
        ] {
            let declared =
                access.reads_lanes.contains(&lane) || access.writes_lanes.contains(&lane);
            assert!(declared, "lane {lane:?} must be declared");
        }
        // Cursor advance writes the player lane; TransformDesc/Position are outputs.
        assert!(access.writes_lanes.contains(&TypeId::of::<AnimPlayer>()));
        assert!(access.writes_lanes.contains(&TypeId::of::<TransformDesc>()));
        assert!(access.writes_lanes.contains(&TypeId::of::<Position>()));
    }

    #[test]
    fn translation_lerps_between_keys() {
        let track = vec_track(&[(0.0, Vec3::ZERO), (1.0, Vec3::new(10.0, 0.0, 0.0))]);
        assert_eq!(track.sample(0.5), Some(Vec3::new(5.0, 0.0, 0.0)));
    }

    #[test]
    fn sampling_clamps_outside_key_range() {
        let track = vec_track(&[(1.0, Vec3::ONE), (2.0, Vec3::new(3.0, 3.0, 3.0))]);
        assert_eq!(track.sample(0.0), Some(Vec3::ONE));
        assert_eq!(track.sample(99.0), Some(Vec3::new(3.0, 3.0, 3.0)));
    }

    #[test]
    fn empty_track_samples_nothing() {
        let track: KeyTrack<Vec3> = KeyTrack { keys: Vec::new() };
        assert_eq!(track.sample(0.5), None);
    }

    #[test]
    fn single_key_track_holds_forever() {
        let track = vec_track(&[(0.0, Vec3::new(1.0, 2.0, 3.0))]);
        assert_eq!(track.sample(0.0), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(track.sample(57.0), Some(Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn rotation_slerps_halfway() {
        let from = Quat::IDENTITY;
        let to = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let track = quat_track(&[(0.0, from), (1.0, to)]);
        let mid = track.sample(0.5).expect("mid key");
        let expected = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        assert!(
            mid.angle_between(expected) < 1e-5,
            "halfway slerp must be 45°, got {mid:?}"
        );
    }

    #[test]
    fn advance_clamps_without_loop() {
        assert_eq!(advance_player_time(0.9, 0.5, 1.0, 1.0, false), 1.0);
        assert_eq!(advance_player_time(0.0, 0.25, 2.0, 1.0, false), 0.5);
    }

    #[test]
    fn advance_wraps_with_loop() {
        assert!((advance_player_time(0.9, 0.5, 1.0, 1.0, true) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn advance_supports_negative_speed() {
        assert!((advance_player_time(0.2, 0.5, -1.0, 1.0, true) - 0.7).abs() < 1e-6);
        assert_eq!(advance_player_time(0.2, 0.5, -1.0, 1.0, false), 0.0);
    }

    #[test]
    fn advance_freezes_on_zero_duration() {
        assert_eq!(advance_player_time(3.0, 1.0, 1.0, 0.0, true), 3.0);
    }
}
