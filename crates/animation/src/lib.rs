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
//!
//! Phase B (`docs/animation-design.md` §2 and §5) adds the skeletal side in
//! the same file: [`Skeleton`]/[`JointPose`]/[`SkelPlayer`] hot lanes plus
//! the cold [`SkelClip`] lane and [`SkinnedMesh`] bind data, sampled by
//! `skel_sample` ([`SkelSampleSystem`]) and skinned on the CPU by
//! `skel_skin_cpu` ([`SkelSkinSystem`]). [`Animator`] on the character root
//! selects a skeletal clip by name; a player stays empty until that call.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use glam::{Mat3, Mat4, Quat, Vec3};
use ornis_core::{
    Clamped01, ColdComponentStore, ComponentStore, Entity, Resources, Seconds, SmartStore, System,
    SystemAccess, Time,
};
use ornis_gameplay::Position;

use ornis_assets::scene::{MeshDesc, TransformDesc};
use ornis_core::units::UnitQuat;

/// Squared length / determinant floor for degenerate skinning joints.
const DEGENERATE_LEN2: f32 = 1e-12;
/// Keyframe span / weight sum floor before divide.
const NEAR_ZERO: f32 = 1e-6;
/// Max joint influences per skinned vertex (glTF / engine contract).
const MAX_INFLUENCES: usize = 4;
/// Indices per triangle (flat soup alignment).
const TRIANGLE_VERTS: usize = 3;

/// Phase D GPU-skinning contract: [`SkinningMode`], [`JointCount`] /
/// [`JointLimit`], [`SkinningResources`], [`SkinError`] and the shader-mirror
/// reference blend.
mod animator;
pub use animator::{Animator, AnimatorAccess, AnimatorError, AnimatorMut, try_animator};

pub mod skinning;

pub use skinning::{
    CPU_GPU_TOLERANCE, JointCount, JointLimit, SkinError, SkinningMode, SkinningResources,
    blend_vertex_reference, canonical_staged_weights,
};

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
pub enum Interpolation<T> {
    /// Blend between surrounding keys (`lerp`/`slerp`).
    Linear,
    /// Hold the earlier key (glTF `STEP`).
    Step,
    /// Cubic Hermite blend (glTF `CUBICSPLINE`): per-second derivatives, one
    /// per key in key order (the loader contract; `keys.len()` entries each).
    Cubic {
        /// Derivative arriving at each key.
        in_tangents: Vec<T>,
        /// Derivative leaving each key.
        out_tangents: Vec<T>,
    },
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
    /// How to blend between keys.
    pub interpolation: Interpolation<T>,
}

impl<T> KeyTrack<T> {
    /// Builds a linearly blended track.
    pub fn linear(keys: Vec<Key<T>>) -> Self {
        Self {
            keys,
            interpolation: Interpolation::Linear,
        }
    }

    /// Builds a hold-previous track (glTF `STEP`).
    pub fn stepped(keys: Vec<Key<T>>) -> Self {
        Self {
            keys,
            interpolation: Interpolation::Step,
        }
    }

    /// Builds a cubic Hermite track (glTF `CUBICSPLINE`).
    ///
    /// `in_tangents`/`out_tangents` hold per-second derivatives, one per key
    /// in key order. Lengths must match `keys.len()` (checked with
    /// `debug_assert` in test builds; the sampler never panics on a
    /// mismatch and reads missing tangents as zero instead).
    pub fn cubic(keys: Vec<Key<T>>, in_tangents: Vec<T>, out_tangents: Vec<T>) -> Self {
        debug_assert_eq!(
            in_tangents.len(),
            keys.len(),
            "cubic in-tangents must match keys one-to-one"
        );
        debug_assert_eq!(
            out_tangents.len(),
            keys.len(),
            "cubic out-tangents must match keys one-to-one"
        );
        Self {
            keys,
            interpolation: Interpolation::Cubic {
                in_tangents,
                out_tangents,
            },
        }
    }
}

impl KeyTrack<Vec3> {
    /// Samples the translation/scale channel at clip time `t` seconds.
    ///
    /// Linear blends between the surrounding keys, Step holds the earlier
    /// key, Cubic blends with the Hermite basis over the segment tangents;
    /// all clamp to the end keys outside the key range. Returns [`None`]
    /// when the track is empty (channel absent).
    pub fn sample(&self, t: f32) -> Option<Vec3> {
        let (before_idx, after_idx, s) = segment_index(&self.keys, t)?;
        let (before, after) = (&self.keys[before_idx], &self.keys[after_idx]);
        match &self.interpolation {
            Interpolation::Linear => Some(before.value.lerp(after.value, s)),
            Interpolation::Step => Some(before.value),
            // Missing tangents mean a loader bug: read them as zero vectors
            // (documented fallback — the hot path never panics), which
            // degrades that segment to smoothstep between the keys.
            Interpolation::Cubic {
                in_tangents,
                out_tangents,
            } => {
                if before_idx == after_idx {
                    return Some(before.value);
                }
                let dt = after.time - before.time;
                if dt <= NEAR_ZERO {
                    return Some(before.value);
                }
                let m0 = out_tangents.get(before_idx).copied().unwrap_or(Vec3::ZERO);
                let m1 = in_tangents.get(after_idx).copied().unwrap_or(Vec3::ZERO);
                Some(hermite_vec3(before.value, m0, after.value, m1, dt, s))
            }
        }
    }
}

impl KeyTrack<Quat> {
    /// Samples the rotation channel at clip time `t` seconds.
    ///
    /// Linear blends ([`Quat::slerp`]) between the surrounding keys, Step
    /// holds the earlier key, Cubic blends per-component with the Hermite
    /// basis and renormalizes; all clamp to the end keys outside the key
    /// range. Returns [`None`] when the track is empty (channel absent).
    pub fn sample(&self, t: f32) -> Option<Quat> {
        let (before_idx, after_idx, s) = segment_index(&self.keys, t)?;
        let (before, after) = (&self.keys[before_idx], &self.keys[after_idx]);
        match &self.interpolation {
            Interpolation::Linear => Some(before.value.slerp(after.value, s)),
            Interpolation::Step => Some(before.value),
            // Missing tangents mean a loader bug: read them as zero quats
            // (the `(0,0,0,0)` derivative, not identity — the hot path never
            // panics), which degrades that segment to smoothstep.
            Interpolation::Cubic {
                in_tangents,
                out_tangents,
            } => {
                if before_idx == after_idx {
                    return Some(before.value);
                }
                let dt = after.time - before.time;
                if dt <= NEAR_ZERO {
                    return Some(before.value);
                }
                let zero = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
                let m0 = out_tangents.get(before_idx).copied().unwrap_or(zero);
                let m1 = in_tangents.get(after_idx).copied().unwrap_or(zero);
                Some(hermite_quat(before.value, m0, after.value, m1, dt, s))
            }
        }
    }
}

/// Cubic Hermite basis over one segment (glTF 2.0 `CUBICSPLINE` model):
/// `m0` is the out-tangent at `p0`, `m1` the in-tangent at `p1` (both
/// per-second derivatives), `dt` the key span, `s` the clamped factor.
fn hermite_scalar(p0: f32, m0: f32, p1: f32, m1: f32, dt: f32, s: f32) -> f32 {
    let s2 = s * s;
    let s3 = s2 * s;
    (2.0 * s3 - 3.0 * s2 + 1.0) * p0
        + (s3 - 2.0 * s2 + s) * m0 * dt
        + (-2.0 * s3 + 3.0 * s2) * p1
        + (s3 - s2) * m1 * dt
}

/// Vector Hermite: the scalar basis applied per component.
fn hermite_vec3(p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, dt: f32, s: f32) -> Vec3 {
    Vec3::new(
        hermite_scalar(p0.x, m0.x, p1.x, m1.x, dt, s),
        hermite_scalar(p0.y, m0.y, p1.y, m1.y, dt, s),
        hermite_scalar(p0.z, m0.z, p1.z, m1.z, dt, s),
    )
}

/// Quaternion Hermite: the scalar basis applied per `(x, y, z, w)`
/// component, then renormalized. A degenerate blend (antipodal keys at
/// `s = 0.5` with cancelling tangents) falls back to [`Quat::slerp`] so the
/// result stays a unit quaternion, never NaN.
fn hermite_quat(p0: Quat, m0: Quat, p1: Quat, m1: Quat, dt: f32, s: f32) -> Quat {
    let blended = Quat::from_xyzw(
        hermite_scalar(p0.x, m0.x, p1.x, m1.x, dt, s),
        hermite_scalar(p0.y, m0.y, p1.y, m1.y, dt, s),
        hermite_scalar(p0.z, m0.z, p1.z, m1.z, dt, s),
        hermite_scalar(p0.w, m0.w, p1.w, m1.w, dt, s),
    );
    let length_squared = blended.length_squared();
    if length_squared.is_finite() && length_squared > DEGENERATE_LEN2 {
        blended.normalize()
    } else {
        p0.slerp(p1, s)
    }
}

/// Locates the key segment surrounding `t`: the indices of the two keys to
/// blend plus the blend factor in `0.0..=1.0`. Returns [`None`] for an empty
/// track; clamped ends report the same index twice with factor `0.0`.
fn segment_index<T>(keys: &[Key<T>], t: f32) -> Option<(usize, usize, f32)> {
    if keys.is_empty() {
        return None;
    }
    if keys.len() == 1 {
        return Some((0, 0, 0.0));
    }
    let upper = keys.partition_point(|key| key.time <= t);
    if upper == 0 {
        return Some((0, 0, 0.0));
    }
    if upper >= keys.len() {
        let last = keys.len() - 1;
        return Some((last, last, 0.0));
    }
    let (before, after) = (&keys[upper - 1], &keys[upper]);
    let span = after.time - before.time;
    let alpha = if span > NEAR_ZERO {
        ((t - before.time) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some((upper - 1, upper, alpha))
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
                desc.translation = translation;
            }
            // A degenerate sampled rotation keeps the previous orientation.
            if let Some(rotation) = item.pose.rotation.and_then(UnitQuat::normalize) {
                desc.rotation = rotation;
            }
            if let Some(scale) = item.pose.scale {
                desc.scale = scale;
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
        KeyTrack::linear(
            keys.iter()
                .map(|&(time, value)| Key { time, value })
                .collect(),
        )
    }

    fn quat_track(keys: &[(f32, Quat)]) -> KeyTrack<Quat> {
        KeyTrack::linear(
            keys.iter()
                .map(|&(time, value)| Key { time, value })
                .collect(),
        )
    }

    fn stepped_vec_track(keys: &[(f32, Vec3)]) -> KeyTrack<Vec3> {
        KeyTrack::stepped(
            keys.iter()
                .map(|&(time, value)| Key { time, value })
                .collect(),
        )
    }

    fn stepped_quat_track(keys: &[(f32, Quat)]) -> KeyTrack<Quat> {
        KeyTrack::stepped(
            keys.iter()
                .map(|&(time, value)| Key { time, value })
                .collect(),
        )
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
        let track: KeyTrack<Vec3> = KeyTrack::linear(Vec::new());
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
    fn step_translation_holds_before_value() {
        let track = stepped_vec_track(&[(0.0, Vec3::ZERO), (1.0, Vec3::new(10.0, 0.0, 0.0))]);
        assert_eq!(track.sample(0.5), Some(Vec3::ZERO));
        assert_eq!(track.sample(0.99), Some(Vec3::ZERO));
    }

    #[test]
    fn step_rotation_holds_before_value() {
        let from = Quat::IDENTITY;
        let to = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let track = stepped_quat_track(&[(0.0, from), (1.0, to)]);
        let mid = track.sample(0.5).expect("mid key");
        assert!(
            mid.angle_between(from) < 1e-5,
            "step must hold the earlier key, got {mid:?}"
        );
    }

    #[test]
    fn step_clamps_outside_key_range() {
        let track = stepped_vec_track(&[(1.0, Vec3::ONE), (2.0, Vec3::new(3.0, 3.0, 3.0))]);
        assert_eq!(track.sample(0.0), Some(Vec3::ONE));
        assert_eq!(track.sample(99.0), Some(Vec3::new(3.0, 3.0, 3.0)));
        let quat_track =
            stepped_quat_track(&[(1.0, Quat::IDENTITY), (2.0, Quat::from_rotation_y(1.0))]);
        assert_eq!(quat_track.sample(0.0), Some(Quat::IDENTITY));
    }

    #[test]
    fn step_returns_key_value_on_exact_time() {
        let track = stepped_vec_track(&[
            (0.0, Vec3::ZERO),
            (1.0, Vec3::ONE),
            (2.0, Vec3::new(3.0, 3.0, 3.0)),
        ]);
        assert_eq!(track.sample(1.0), Some(Vec3::ONE));
        assert_eq!(track.sample(2.0), Some(Vec3::new(3.0, 3.0, 3.0)));
    }

    #[test]
    fn step_empty_track_samples_nothing() {
        let track: KeyTrack<Vec3> = KeyTrack::stepped(Vec::new());
        assert_eq!(track.sample(0.5), None);
        let quat_track: KeyTrack<Quat> = KeyTrack::stepped(Vec::new());
        assert_eq!(quat_track.sample(0.5), None);
    }

    #[test]
    fn step_single_key_track_holds_forever() {
        let track = stepped_vec_track(&[(0.0, Vec3::new(1.0, 2.0, 3.0))]);
        assert_eq!(track.sample(0.0), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(track.sample(57.0), Some(Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn cubic_zero_tangents_midpoint_is_smoothstep() {
        let track = KeyTrack::cubic(
            vec![
                Key {
                    time: 0.0,
                    value: Vec3::ZERO,
                },
                Key {
                    time: 1.0,
                    value: Vec3::ONE,
                },
            ],
            vec![Vec3::ZERO, Vec3::ZERO],
            vec![Vec3::ZERO, Vec3::ZERO],
        );
        assert_eq!(track.sample(0.0), Some(Vec3::ZERO));
        assert_eq!(track.sample(1.0), Some(Vec3::ONE));
        let mid = track.sample(0.5).expect("mid key");
        assert!(
            (mid - Vec3::splat(0.5)).length() < 1e-6,
            "zero-tangent cubic must smoothstep to 0.5, got {mid:?}"
        );
    }

    #[test]
    fn cubic_nonzero_tangent_shifts_midpoint() {
        // p(0.5) = 0.5·(p0+p1) + 0.125·(m0-m1)·dt: m0·dt = (4,0,0) lifts the
        // midpoint from 1.0 to 1.5.
        let track = KeyTrack::cubic(
            vec![
                Key {
                    time: 0.0,
                    value: Vec3::ZERO,
                },
                Key {
                    time: 2.0,
                    value: Vec3::new(2.0, 0.0, 0.0),
                },
            ],
            vec![Vec3::ZERO, Vec3::ZERO],
            vec![Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO],
        );
        let mid = track.sample(1.0).expect("mid key");
        assert!(
            (mid.x - 1.5).abs() < 1e-6 && mid.y.abs() < 1e-6 && mid.z.abs() < 1e-6,
            "tangent must shape the midpoint, got {mid:?}"
        );
    }

    #[test]
    fn cubic_quat_output_is_unit() {
        let from = Quat::IDENTITY;
        let to = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let track = KeyTrack::cubic(
            vec![
                Key {
                    time: 0.0,
                    value: from,
                },
                Key {
                    time: 1.0,
                    value: to,
                },
            ],
            vec![
                Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
                Quat::from_xyzw(-1.0, 0.0, 0.0, 0.0),
            ],
            vec![
                Quat::from_xyzw(1.0, 0.0, 0.0, 0.0),
                Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
            ],
        );
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let got = track.sample(t).expect("sample");
            assert!(
                (got.length() - 1.0).abs() < 1e-5,
                "cubic quat must stay unit at t={t}, got {got:?}"
            );
        }
        assert!(track.sample(0.0).expect("start").angle_between(from) < 1e-5);
        // Clamped ends return the stored key bit-exact (`to` itself is only
        // unit to 1e-7 under `from_rotation_y`, so no angle check here).
        assert_eq!(track.sample(1.0), Some(to));
    }

    #[test]
    fn cubic_missing_tangents_fall_back_to_zero() {
        // Loader-bug path: built literally to bypass `KeyTrack::cubic`
        // (whose `debug_assert`s pin the one-to-one lens) — the sampler
        // reads missing entries as zero, degrading to smoothstep.
        let track: KeyTrack<Vec3> = KeyTrack {
            keys: vec![
                Key {
                    time: 0.0,
                    value: Vec3::ZERO,
                },
                Key {
                    time: 1.0,
                    value: Vec3::ONE,
                },
            ],
            interpolation: Interpolation::Cubic {
                in_tangents: Vec::new(),
                out_tangents: Vec::new(),
            },
        };
        let mid = track.sample(0.5).expect("mid key");
        assert!(
            (mid - Vec3::splat(0.5)).length() < 1e-6,
            "missing tangents must degrade to smoothstep, got {mid:?}"
        );
        let quat_track: KeyTrack<Quat> = KeyTrack {
            keys: vec![
                Key {
                    time: 0.0,
                    value: Quat::IDENTITY,
                },
                Key {
                    time: 1.0,
                    value: Quat::from_rotation_y(1.0),
                },
            ],
            interpolation: Interpolation::Cubic {
                in_tangents: Vec::new(),
                out_tangents: Vec::new(),
            },
        };
        let got = quat_track.sample(0.5).expect("mid key");
        assert!(
            (got.length() - 1.0).abs() < 1e-5,
            "missing quat tangents must stay unit, got {got:?}"
        );
    }

    #[test]
    fn cubic_empty_single_and_clamp_match_linear() {
        let empty: KeyTrack<Vec3> = KeyTrack::cubic(Vec::new(), Vec::new(), Vec::new());
        assert_eq!(empty.sample(0.5), None);
        let empty_quat: KeyTrack<Quat> = KeyTrack::cubic(Vec::new(), Vec::new(), Vec::new());
        assert_eq!(empty_quat.sample(0.5), None);
        let single = KeyTrack::cubic(
            vec![Key {
                time: 0.0,
                value: Vec3::new(1.0, 2.0, 3.0),
            }],
            vec![Vec3::ZERO],
            vec![Vec3::ZERO],
        );
        assert_eq!(single.sample(0.0), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(single.sample(57.0), Some(Vec3::new(1.0, 2.0, 3.0)));
        let track = KeyTrack::cubic(
            vec![
                Key {
                    time: 1.0,
                    value: Vec3::ONE,
                },
                Key {
                    time: 2.0,
                    value: Vec3::new(3.0, 3.0, 3.0),
                },
            ],
            vec![Vec3::ZERO, Vec3::ZERO],
            vec![Vec3::ZERO, Vec3::ZERO],
        );
        assert_eq!(track.sample(0.0), Some(Vec3::ONE));
        assert_eq!(track.sample(99.0), Some(Vec3::new(3.0, 3.0, 3.0)));
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

// ── Phase B: skeletal animation (design `docs/animation-design.md` §2 and §5) ──
//
// Hot lanes on the skeleton root (`Skeleton`, `JointPose`, `SkelPlayer`),
// the cold clip lane (`SkelClip`) and the per-mesh bind data (`SkinnedMesh`)
// are sampled by `skel_sample` (local → model matrices along `parents`) and
// skinned on the CPU by `skel_skin_cpu` (linear blend skinning into the
// mesh's own output buffers). Both are PostFrame-only, like `anim_sample`.
// Bad skin (missing/invalid skeleton, pose or bind data) skips the entity
// and bumps a counter — never a stub pose. Joints are never ECS entities:
// the flat `JointPose` vector is the only pose representation.

/// Maximum joints per skeleton: the uniform/storage limit of the GPU path
/// (design §2.1), spelled through [`JointLimit::GPU`] — the type is the
/// single source of truth, this constant is its `usize` projection.
///
/// Skeletons beyond the cap are rejected, never silently truncated:
/// [`Skeleton::validate`] fails, loaders must fail, and the systems skip
/// the entity while counting it (`skipped_bad_skin`).
pub const MAX_JOINTS: usize = JointLimit::GPU.get() as usize;

/// Why a [`Skeleton`] is unusable (and its entities must be skipped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkelError {
    /// No joints: nothing to sample or skin.
    Empty,
    /// More than [`MAX_JOINTS`] joints.
    TooManyJoints,
    /// `parents`/`inverse_bind` length mismatch.
    LengthMismatch,
    /// Parent index outside `[-1, joint_count)`.
    BadParent,
    /// Parent cycle (including self-parenting).
    Cycle,
}

/// Index of one joint inside its owning [`Skeleton`].
///
/// Newtype over `u32` so joint indices never mix with entity ids or
/// particle indices at the type level. Parents use `None` for roots
/// (replacing the legacy `-1` sentinel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JointId(u32);

impl JointId {
    /// Wraps a raw `u32` joint index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Maps a raw `i32` parent link: negative means root (`None`).
    pub const fn from_raw_opt(raw: i32) -> Option<Self> {
        if raw < 0 {
            None
        } else {
            Some(Self(raw as u32))
        }
    }

    /// Raw `u32` joint index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Joint index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for JointId {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for JointId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<JointId> for u32 {
    fn from(h: JointId) -> Self {
        h.0
    }
}

impl From<JointId> for usize {
    fn from(h: JointId) -> Self {
        h.0 as usize
    }
}

/// Topology of one skeleton: joint parents plus per-joint inverse bind matrices.
///
/// Hot lane on the skeleton root. `parents`/`inverse_bind` (and names) ride
/// an [`Arc`] so cloning the lane never copies the arrays. Lengths of
/// `parents` and `inverse_bind` must match; `parents[joint]` is the parent
/// joint (`None` = root). Parent order is unrestricted — chains
/// are resolved per joint, cycles are rejected.
#[derive(Debug, Clone, PartialEq)]
pub struct Skeleton {
    /// Parent joint per joint (`None` = root); length equals the joint count.
    pub parents: Arc<[Option<JointId>]>,
    /// Bind-pose inverses; skinning uses `model[joint] * inverse_bind[joint]`.
    pub inverse_bind: Arc<[Mat4]>,
    /// Human-readable joint labels, diagnostics only (length unchecked).
    pub joint_names: Arc<[String]>,
}

impl Skeleton {
    /// Builds a skeleton; topology is checked by [`Skeleton::validate`],
    /// not here, so loaders can assemble first and fail with a reason.
    pub fn new(
        parents: Vec<Option<JointId>>,
        inverse_bind: Vec<Mat4>,
        joint_names: Vec<String>,
    ) -> Self {
        Self {
            parents: Arc::from(parents),
            inverse_bind: Arc::from(inverse_bind),
            joint_names: Arc::from(joint_names),
        }
    }

    /// Number of joints (`parents.len()`).
    pub fn joint_count(&self) -> usize {
        self.parents.len()
    }

    /// Checks joint count, cap, array lengths, parent range and cycles.
    ///
    /// Returns the joint count on success.
    ///
    /// # Errors
    ///
    /// Returns [`SkelError`] describing the first defect found.
    pub fn validate(&self) -> Result<usize, SkelError> {
        let count = self.parents.len();
        if count == 0 {
            return Err(SkelError::Empty);
        }
        if count > MAX_JOINTS {
            return Err(SkelError::TooManyJoints);
        }
        if self.inverse_bind.len() != count {
            return Err(SkelError::LengthMismatch);
        }
        for joint in 0..count {
            validate_chain(&self.parents, joint, count)?;
        }
        Ok(count)
    }
}

/// Validates one joint's ancestor chain: every link must index a joint,
/// `None` terminates at a root, anything else is [`SkelError::BadParent`];
/// more links than joints means a cycle ([`SkelError::Cycle`]).
///
/// `joint` ranges over `0..count` (caller-checked), so the first lookup
/// always hits.
fn validate_chain(
    parents: &[Option<JointId>],
    joint: usize,
    count: usize,
) -> Result<(), SkelError> {
    let mut cursor = joint;
    let mut steps = 0;
    while cursor < parents.len() {
        let Some(parent) = parents[cursor] else {
            break;
        };
        let parent = parent.index();
        if parent >= count {
            return Err(SkelError::BadParent);
        }
        cursor = parent;
        steps += 1;
        if steps > count {
            return Err(SkelError::Cycle);
        }
    }
    if cursor < parents.len() {
        Ok(())
    } else {
        Err(SkelError::BadParent)
    }
}

/// Sampled model-space pose of one skeleton: one matrix per joint.
///
/// Hot lane on the skeleton root, written whole by `skel_sample` (vector
/// swap, never element-wise). Length must equal the [`Skeleton`] joint
/// count; stale lengths make the entity bad skin.
#[derive(Debug, Clone, PartialEq)]
pub struct JointPose {
    /// Local-to-model matrices, one per joint (`root * chain * local`).
    pub matrices: Vec<Mat4>,
}

impl JointPose {
    /// Bind/rest pose placeholder: every joint at identity.
    ///
    /// The loader seeds roots with this (design §4.5); the sampler
    /// overwrites it. Over-cap counts are kept (systems reject them).
    pub fn identity(joint_count: usize) -> Self {
        Self {
            matrices: vec![Mat4::IDENTITY; joint_count],
        }
    }
}

/// One joint channel of a [`SkelClip`]: sorted keys per transform channel.
///
/// Field names follow the design (`t`/`r`/`s`); empty tracks mean
/// "identity channel" (unlike object tracks, joints cannot leave a
/// channel untouched or the chain would collapse).
#[derive(Debug, Clone, PartialEq)]
pub struct JointTrack {
    /// Animated joint index into [`Skeleton`]/[`JointPose`].
    pub joint: JointId,
    /// Translation keys; empty means zero translation.
    pub translation: KeyTrack<Vec3>,
    /// Rotation keys (unit quaternions); empty means identity.
    pub rotation: KeyTrack<Quat>,
    /// Scale keys; empty means unit scale.
    pub scale: KeyTrack<Vec3>,
}

/// Shareable skeletal clip: cold data, sampled often, changed rarely.
///
/// Lives in the cold lane on a playlist entity; [`SkelPlayer`]s reference
/// it through [`ClipId`]. Whether time wraps past `duration` is the
/// player's `looping` flag (a non-positive `duration` holds the pose
/// either way). [`Self::name`] is the glTF animation name; the
/// character-root [`Animator`] indexes these playlist entities by that name.
#[derive(Debug, Clone, PartialEq)]
pub struct SkelClip {
    /// Clip name (glTF animation name; diagnostics and clip selection).
    pub name: String,
    /// Clip length in seconds. Wrapping is the player's `looping` flag.
    pub duration: f32,
    /// Per-joint tracks; first track per joint wins on duplicates.
    pub tracks: Vec<JointTrack>,
}

/// Hot per-root playback cursor into a [`SkelClip`].
///
/// Same playlist model as [`AnimPlayer`]. `weight` is reserved for phase E
/// blending and ignored by the sampler. `time` and `speed` are [`Seconds`]
/// so the hot lane does not carry a bare `f32`: `time` is the cursor in
/// seconds, `speed` is the playback rate (`Seconds::new(1.0)` is real time;
/// a negative raw value scrubs backwards). Wiring leaves this component
/// absent until [`AnimatorMut::play`].
///
/// `Clone + Send + Sync` per the hot-lane contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkelPlayer {
    /// Playlist entity holding the [`SkelClip`] in the cold lane.
    pub clip: ClipId,
    /// Current clip time; advanced by `dt * speed` each PostFrame.
    pub time: Seconds,
    /// Playback rate (`Seconds::new(1.0)` = real time; negative scrubs backwards).
    pub speed: Seconds,
    /// Blend weight in `[0, 1]`, reserved for phase E (crossfade/masks);
    /// ignored by the sampler.
    pub weight: Clamped01,
    /// Paused players hold their pose: time does not advance and the pose
    /// is not rewritten.
    pub playing: bool,
    /// `true` wraps time past the clip duration; `false` clamps at the ends.
    pub looping: bool,
}

/// One skinned mesh: bind geometry plus joint influences and the CPU
/// skinning output buffers.
///
/// Hot lane on the mesh entity (design §4.5); `skeleton` points at the
/// root entity carrying [`Skeleton`] + [`JointPose`]. Bind arrays ride an
/// [`Arc`] (shared with the loader/fallback soup); the `skinned_*` output
/// buffers are owned world-space results, refreshed whole by
/// `skel_skin_cpu`. `uvs`/`indices` are passthrough bind data for the
/// future upload path (skinning never touches them).
#[derive(Debug, Clone, PartialEq)]
pub struct SkinnedMesh {
    /// Skeleton root entity ([`Skeleton`] + [`JointPose`] live there).
    pub skeleton: Entity,
    /// Influencing joints per vertex (top-4, `u16` — the seam contract
    /// every loader truncates to).
    pub joints: Vec<[u16; 4]>,
    /// Influence weights per vertex (canonicalized at skin time: finite
    /// positive sum normalizes, otherwise `(1,0,0,0)`).
    pub weights: Vec<[f32; 4]>,
    /// Bind-pose positions (engine units).
    pub positions: Arc<[[f32; 3]]>,
    /// Bind-pose shading normals (unit length).
    pub normals: Arc<[[f32; 3]]>,
    /// Bind-pose texture coordinates (passthrough).
    pub uvs: Arc<[[f32; 2]]>,
    /// Triangle index list (passthrough).
    pub indices: Arc<[u32]>,
    /// Skinned world-space positions, refreshed whole by `skel_skin_cpu`.
    pub skinned_positions: Vec<[f32; 3]>,
    /// Skinned world-space normals, refreshed whole by `skel_skin_cpu`.
    pub skinned_normals: Vec<[f32; 3]>,
}

impl SkinnedMesh {
    /// Builds a mesh; output buffers start as the bind pose (identity
    /// skin) until `skel_skin_cpu` refreshes them.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        skeleton: Entity,
        joints: Vec<[u16; 4]>,
        weights: Vec<[f32; 4]>,
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<u32>,
    ) -> Self {
        Self {
            skeleton,
            joints,
            weights,
            skinned_positions: positions.clone(),
            skinned_normals: normals.clone(),
            positions: Arc::from(positions),
            normals: Arc::from(normals),
            uvs: Arc::from(uvs),
            indices: Arc::from(indices),
        }
    }

    /// Vertex count of the bind pose (`positions.len()`).
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }
}

/// Per-run honesty counters of `skel_sample`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SkelSampleStats {
    /// Poses written into [`JointPose`] lanes.
    pub sampled: u32,
    /// Entities skipped for missing/invalid skeleton, clip or pose lane.
    pub skipped_bad_skin: u32,
}

/// Per-run honesty counters of `skel_skin_cpu`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SkelSkinStats {
    /// Meshes skinned into their output buffers.
    pub skinned: u32,
    /// Skinned vertices across [`SkelSkinStats::skinned`] meshes.
    pub skinned_vertices: u32,
    /// Entities skipped for missing/invalid skeleton, pose or bind data.
    pub skipped_bad_skin: u32,
}

/// Composes one joint-local matrix (`T·R·S`, the render convention).
///
/// Absent channels read as identity (zero translation, unit rotation,
/// unit scale) so untracked joints hold the bind offset instead of
/// collapsing the chain.
///
/// # Examples
///
/// ```
/// # use glam::{Quat, Vec3};
/// # use ornis_animation::joint_local_matrix;
/// let local = joint_local_matrix(Some(Vec3::ONE), None, None);
/// assert_eq!(local, glam::Mat4::from_translation(Vec3::ONE));
/// assert_eq!(joint_local_matrix(None, None, None), glam::Mat4::IDENTITY);
/// # let _ = Quat::IDENTITY;
/// ```
pub fn joint_local_matrix(
    translation: Option<Vec3>,
    rotation: Option<Quat>,
    scale: Option<Vec3>,
) -> Mat4 {
    Mat4::from_scale_rotation_translation(
        scale.unwrap_or(Vec3::ONE),
        rotation.unwrap_or(Quat::IDENTITY),
        translation.unwrap_or(Vec3::ZERO),
    )
}

/// Resolves local matrices to model space along `parents`.
///
/// `models[joint] = root * ... * local[joint]` over each parent chain, so
/// parent order is unrestricted; out-of-range parents and cycles resolve
/// to [`None`]. Cost is `O(joints × depth)` (joints ≤ [`MAX_JOINTS`]).
///
/// Returns [`None`] when the topology is bad (empty, over-cap, length or
/// parent defect) — the caller skips the entity and counts it.
pub fn compose_model_matrices(
    skeleton: &Skeleton,
    locals: &[Mat4],
    root: &Mat4,
) -> Option<Vec<Mat4>> {
    let count = checked_count(skeleton, locals.len())?;
    (0..count)
        .map(|joint| model_via_chain(&skeleton.parents, locals, root, joint))
        .collect()
}

/// Joint count after the structural checks shared by sampling and skinning.
///
/// [`None`] on empty, over-cap or length defects (reasons live in
/// [`Skeleton::validate`]; the hot path only needs the verdict).
fn checked_count(skeleton: &Skeleton, locals_len: usize) -> Option<usize> {
    let count = skeleton.parents.len();
    if count == 0 || count > MAX_JOINTS {
        return None;
    }
    if skeleton.inverse_bind.len() != count || locals_len != count {
        return None;
    }
    Some(count)
}

/// Resolves one joint: multiplies `root * local` down its parent chain.
///
/// [`None`] on out-of-range parents or cycles. Terminates: every pushed
/// link is new (the `contains` guard), drawn from a finite joint set.
fn model_via_chain(
    parents: &[Option<JointId>],
    locals: &[Mat4],
    root: &Mat4,
    joint: usize,
) -> Option<Mat4> {
    let mut chain = vec![joint];
    while let Some(&cursor) = chain.last() {
        let Some(parent) = *parents.get(cursor)? else {
            break;
        };
        let parent = parent.index();
        if parent >= parents.len() || chain.contains(&parent) {
            return None;
        }
        chain.push(parent);
    }
    let mut model = *root;
    for &link in chain.iter().rev() {
        model *= locals[link];
    }
    Some(model)
}

/// Builds final skinning matrices: `model[joint] * inverse_bind[joint]`.
///
/// Lengths are caller-checked (systems validate first); extra entries of
/// the longer slice are ignored.
pub fn skinning_matrices(models: &[Mat4], inverse_bind: &[Mat4]) -> Vec<Mat4> {
    models
        .iter()
        .zip(inverse_bind.iter())
        .map(|(model, bind)| *model * *bind)
        .collect()
}

/// Linear blend skinning over bind vertices (positions and normals).
///
/// `vertex' = Σ weight·M·vertex`; normals use the per-joint
/// inverse-transpose 3×3 (degenerate joints contribute unrotated) and are
/// renormalized unless they collapse to zero. Weights are canonicalized
/// per vertex (finite positive sum normalizes, otherwise `(1,0,0,0)` on
/// joint 0). Out-of-range joint indices read as identity; length defects
/// truncate to the shortest input — callers (the systems) reject such
/// meshes as bad skin before calling.
pub fn skin_vertices(
    joint_matrices: &[Mat4],
    joints: &[[u16; 4]],
    weights: &[[f32; 4]],
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let normal_matrices: Vec<Mat3> = joint_matrices.iter().map(normal_part).collect();
    let count = joints
        .len()
        .min(weights.len())
        .min(positions.len())
        .min(normals.len());
    let mut out_positions = Vec::with_capacity(count);
    let mut out_normals = Vec::with_capacity(count);
    for index in 0..count {
        let weights = canonical_weights(weights[index]);
        out_positions.push(blend_position(
            joint_matrices,
            joints[index],
            weights,
            positions[index],
        ));
        out_normals.push(blend_normal(
            &normal_matrices,
            joints[index],
            weights,
            normals[index],
        ));
    }
    (out_positions, out_normals)
}

/// 3×3 inverse-transpose of a joint matrix for normal blending.
///
/// Degenerate (singular/non-finite) joints contribute [`Mat3::IDENTITY`]:
/// positions still skin honestly, normals stay unrotated (documented
/// fallback, not a silent entity-level stub).
fn normal_part(joint: &Mat4) -> Mat3 {
    let determinant = joint.determinant();
    if !determinant.is_finite() || determinant.abs() < DEGENERATE_LEN2 {
        return Mat3::IDENTITY;
    }
    Mat3::from_mat4(joint.inverse().transpose())
}

/// Canonical per-vertex weights: normalize a finite positive sum,
/// otherwise fall back to full weight on joint 0 (design §2.1 rule).
fn canonical_weights(weights: [f32; MAX_INFLUENCES]) -> [f32; MAX_INFLUENCES] {
    let finite = weights.iter().all(|slot| slot.is_finite());
    let sum: f32 = weights.iter().sum();
    if finite && sum > NEAR_ZERO {
        [
            weights[0] / sum,
            weights[1] / sum,
            weights[2] / sum,
            weights[3] / sum,
        ]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// Blends one bind position over its four influences.
fn blend_position(
    joint_matrices: &[Mat4],
    joint_index: [u16; 4],
    weights: [f32; 4],
    position: [f32; 3],
) -> [f32; 3] {
    let vertex = Vec3::from_array(position);
    let mut blended = Vec3::ZERO;
    for slot in 0..MAX_INFLUENCES {
        let joint = joint_matrices
            .get(joint_index[slot] as usize)
            .copied()
            .unwrap_or(Mat4::IDENTITY);
        blended += joint.transform_point3(vertex) * weights[slot];
    }
    blended.to_array()
}

/// Blends one bind normal over its four influences, renormalized unless
/// the blend collapses to zero (then the zero vector is kept, not NaN).
fn blend_normal(
    normal_matrices: &[Mat3],
    joint_index: [u16; 4],
    weights: [f32; 4],
    normal: [f32; 3],
) -> [f32; 3] {
    let direction = Vec3::from_array(normal);
    let mut blended = Vec3::ZERO;
    for slot in 0..MAX_INFLUENCES {
        let joint = normal_matrices
            .get(joint_index[slot] as usize)
            .copied()
            .unwrap_or(Mat3::IDENTITY);
        blended += joint * direction * weights[slot];
    }
    if blended.length_squared() > DEGENERATE_LEN2 {
        blended.normalize().to_array()
    } else {
        blended.to_array()
    }
}

/// Skeleton pose sampler (system name `skel_sample`).
///
/// PostFrame-only visual pose on variable [`Time`]: advances playing
/// [`SkelPlayer`] cursors (wrap or clamp per `looping`), samples their [`SkelClip`] tracks
/// to joint-local matrices and resolves them to model space along
/// [`Skeleton::parents`], rooted at the entity's [`TransformDesc`]
/// (identity when the lane is absent). Poses publish whole into
/// [`JointPose`]; lanes are never created for animation.
///
/// Ordering: register after `anim_sample`, pin
/// `try_order_before("anim_sample", "skel_sample")` and
/// `try_order_before("skel_sample", "skel_skin_cpu")` — the skel lanes
/// are disjoint from `anim_sample`, the edges make the determinism
/// explicit for the owner wiring the session.
#[derive(Debug, Clone, Copy)]
pub struct SkelSampleSystem;

impl SkelSampleSystem {
    /// Creates the sampler.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SkelSampleSystem {
    /// Creates the sampler (see [`SkelSampleSystem::new`]).
    fn default() -> Self {
        Self::new()
    }
}

impl System for SkelSampleSystem {
    fn name(&self) -> &'static str {
        "skel_sample"
    }

    fn access(&self) -> SystemAccess {
        // Honest per schedule.rs:415-439 — every lane touched below is
        // declared. One deliberate deviation from the design table (same
        // as `anim_sample`): `SkelPlayer` is *written* (cursor advance),
        // so it must be `writes_lane`, not `reads_lane`. `SkelClip`
        // travels the cold lane: declared for level planning although
        // cold reads need no enforcement grant.
        SystemAccess::new()
            .reads::<Time>()
            .reads::<SmartStore>()
            .reads_lane::<SkelClip>()
            .reads_lane::<Skeleton>()
            .reads_lane::<TransformDesc>()
            .writes_lane::<SkelPlayer>()
            .writes_lane::<JointPose>()
    }

    fn run(&self, resources: &Resources) {
        // The scheduled wrapper keeps the `System` contract; hosts and
        // tests needing the per-run honesty counters call
        // `run_skel_sample` directly (persistent counters land with the
        // extraction wiring in phase C).
        let _ = run_skel_sample(resources);
    }
}

/// Advances playing cursors and publishes model-space poses.
///
/// Pure of schedule position: a deterministic function of lanes + `dt`.
/// Returns per-run counters (sampled poses, bad-skin skips).
pub fn run_skel_sample(resources: &Resources) -> SkelSampleStats {
    let Some(time) = resources.get::<Time>() else {
        return SkelSampleStats::default();
    };
    let Some(store) = resources.get::<SmartStore>() else {
        return SkelSampleStats::default();
    };
    advance_skel_players(store, time.delta_seconds());
    publish_poses(store)
}

/// Advances playing [`SkelPlayer`] cursors against their clip duration.
///
/// Missing clips hold their cursor (counted as bad skin at sample time,
/// not here).
fn advance_skel_players(store: &SmartStore, dt: f32) {
    let Some(clips) = store.read_cold_lane::<SkelClip>() else {
        return;
    };
    let Some(mut players) = store.write_lane::<SkelPlayer>() else {
        return;
    };
    for index in 0..players.data.len() {
        let player = &mut players.data[index];
        if !player.playing {
            continue;
        }
        let Some(clip) = clips.get(player.clip.0) else {
            continue;
        };
        player.time = Seconds::new(advance_player_time(
            player.time.get(),
            dt,
            player.speed.get(),
            clip.duration,
            player.looping,
        ));
    }
}

/// Samples playing cursors and publishes whole [`JointPose`] vectors.
fn publish_poses(store: &SmartStore) -> SkelSampleStats {
    let mut stats = SkelSampleStats::default();
    let work = collect_pose_work(store, &mut stats);
    if work.is_empty() {
        return stats;
    }
    let Some(mut poses) = store.write_lane::<JointPose>() else {
        return stats;
    };
    for (entity, matrices) in work {
        if let Some(pose) = poses.get_mut(entity) {
            pose.matrices = matrices;
            stats.sampled += 1;
        }
    }
    stats
}

/// Collects `(entity, model matrices)` without holding lane guards across
/// the later write: every imperfect entity bumps `skipped_bad_skin`.
fn collect_pose_work(store: &SmartStore, stats: &mut SkelSampleStats) -> Vec<(Entity, Vec<Mat4>)> {
    let Some(clips) = store.read_cold_lane::<SkelClip>() else {
        return Vec::new();
    };
    let Some(players) = store.read_lane::<SkelPlayer>() else {
        return Vec::new();
    };
    let Some(skeletons) = store.read_lane::<Skeleton>() else {
        return Vec::new();
    };
    let transforms = store.read_lane::<TransformDesc>();
    let poses = store.read_lane::<JointPose>();
    let mut work = Vec::new();
    for (entity, player) in players.entities.iter().zip(players.data.iter()) {
        if !player.playing {
            continue;
        }
        match sample_entity(
            *entity,
            player,
            &clips,
            &skeletons,
            transforms.as_deref(),
            poses.as_deref(),
        ) {
            Some(matrices) => work.push((*entity, matrices)),
            None => stats.skipped_bad_skin += 1,
        }
    }
    work
}

/// Samples one playing cursor: tracks → locals → model matrices.
///
/// [`None`] (bad skin) on missing clip/skeleton/pose lane or on invalid
/// topology; the caller counts it. Absent [`TransformDesc`] roots at
/// identity (lenient); everything else is strict.
fn sample_entity(
    entity: Entity,
    player: &SkelPlayer,
    clips: &ColdComponentStore<SkelClip>,
    skeletons: &ComponentStore<Skeleton>,
    transforms: Option<&ComponentStore<TransformDesc>>,
    poses: Option<&ComponentStore<JointPose>>,
) -> Option<Vec<Mat4>> {
    let clip = clips.get(player.clip.0)?;
    let skeleton = skeletons.get(entity)?;
    if poses.is_none_or(|lane| lane.get(entity).is_none()) {
        return None;
    }
    let locals = sample_locals(clip, skeleton.joint_count(), player.time.get());
    compose_model_matrices(skeleton, &locals, &root_matrix(transforms, entity))
}

/// Samples every [`JointTrack`] at `time`; untracked joints stay identity.
///
/// First track per joint wins on duplicates (same rule as `AnimClip`).
fn sample_locals(clip: &SkelClip, joint_count: usize, time: f32) -> Vec<Mat4> {
    let mut locals = vec![Mat4::IDENTITY; joint_count];
    let mut painted = vec![false; joint_count];
    for track in &clip.tracks {
        let joint = track.joint.index();
        if joint >= joint_count || painted[joint] {
            continue;
        }
        painted[joint] = true;
        locals[joint] = joint_local_matrix(
            track.translation.sample(time),
            track.rotation.sample(time),
            track.scale.sample(time),
        );
    }
    locals
}

/// Root matrix from the entity's [`TransformDesc`], identity when absent.
fn root_matrix(transforms: Option<&ComponentStore<TransformDesc>>, entity: Entity) -> Mat4 {
    let Some(desc) = transforms.and_then(|lane| lane.get(entity)) else {
        return Mat4::IDENTITY;
    };
    // `TransformDesc::rotation` is a `UnitQuat`: normalized by type, so
    // no degenerate-quaternion guard is needed before the matrix.
    Mat4::from_scale_rotation_translation(desc.scale, desc.rotation.get(), desc.translation)
}

/// CPU skinning pass (system name `skel_skin_cpu`).
///
/// PostFrame, after `skel_sample`: builds final joint matrices
/// (`model * inverse_bind`) per valid skeleton and blends every
/// [`SkinnedMesh`] into its own output buffers (world-space positions
/// and normals — the future `CustomMeshEntry` payload with
/// `model_matrix = IDENTITY`). Buffers publish whole; imperfect meshes
/// keep their previous buffers and count as bad skin — never stubs.
///
/// Ordering: register after `skel_sample`, pin
/// `try_order_before("skel_sample", "skel_skin_cpu")`.
#[derive(Debug, Clone, Copy)]
pub struct SkelSkinSystem;

impl SkelSkinSystem {
    /// Creates the CPU skinning pass.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SkelSkinSystem {
    /// Creates the CPU skinning pass (see [`SkelSkinSystem::new`]).
    fn default() -> Self {
        Self::new()
    }
}

impl System for SkelSkinSystem {
    fn name(&self) -> &'static str {
        "skel_skin_cpu"
    }

    fn access(&self) -> SystemAccess {
        // `SkinnedMesh` is both read (bind data) and written (output
        // buffers): declared `writes_lane` (which covers reads), and the
        // implementation sequences read and write guards so they never
        // overlap on the lane `RwLock`.
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<Skeleton>()
            .reads_lane::<JointPose>()
            .writes_lane::<SkinnedMesh>()
    }

    fn run(&self, resources: &Resources) {
        // Same contract as `SkelSampleSystem::run`: per-run counters via
        // `run_skel_skin`; persistent counters land with phase C wiring.
        let _ = run_skel_skin(resources);
    }
}

/// Skins every valid [`SkinnedMesh`] into its output buffers.
///
/// Returns per-run counters (skinned meshes/vertices, bad-skin skips).
pub fn run_skel_skin(resources: &Resources) -> SkelSkinStats {
    let mut stats = SkelSkinStats::default();
    let Some(store) = resources.get::<SmartStore>() else {
        return stats;
    };
    let palette = joint_palette(store);
    let jobs = collect_skin_jobs(store, &palette, &mut stats);
    if jobs.is_empty() {
        return stats;
    }
    let Some(mut meshes) = store.write_lane::<SkinnedMesh>() else {
        return stats;
    };
    for job in jobs {
        if let Some(mesh) = meshes.get_mut(job.entity) {
            mesh.skinned_positions = job.positions;
            mesh.skinned_normals = job.normals;
            stats.skinned += 1;
            stats.skinned_vertices += job.vertex_count;
        }
    }
    stats
}

/// Final joint matrices per valid skeleton root.
///
/// Roots fail the palette (stale pose length, invalid topology) without
/// counting here: meshes pointing at them count the bad skin instead.
fn joint_palette(store: &SmartStore) -> HashMap<Entity, Vec<Mat4>> {
    let Some(skeletons) = store.read_lane::<Skeleton>() else {
        return HashMap::new();
    };
    let Some(poses) = store.read_lane::<JointPose>() else {
        return HashMap::new();
    };
    let mut palette = HashMap::new();
    for (entity, skeleton) in skeletons.entities.iter().zip(skeletons.data.iter()) {
        let Ok(count) = skeleton.validate() else {
            continue;
        };
        let Some(pose) = poses.get(*entity) else {
            continue;
        };
        if pose.matrices.len() != count {
            continue;
        }
        palette.insert(
            *entity,
            skinning_matrices(&pose.matrices, &skeleton.inverse_bind),
        );
    }
    palette
}

/// One committed skinning result: owned buffers plus their owner.
struct SkinJob {
    /// Mesh entity receiving the buffers.
    entity: Entity,
    /// World-space skinned positions.
    positions: Vec<[f32; 3]>,
    /// World-space skinned normals.
    normals: Vec<[f32; 3]>,
    /// `positions.len()` (saturates at `u32::MAX`).
    vertex_count: u32,
}

/// Computes every mesh without holding the write guard: imperfect meshes
/// bump `skipped_bad_skin`, nothing is published partially.
fn collect_skin_jobs(
    store: &SmartStore,
    palette: &HashMap<Entity, Vec<Mat4>>,
    stats: &mut SkelSkinStats,
) -> Vec<SkinJob> {
    let Some(meshes) = store.read_lane::<SkinnedMesh>() else {
        return Vec::new();
    };
    let mut jobs = Vec::new();
    for (entity, mesh) in meshes.entities.iter().zip(meshes.data.iter()) {
        match skin_job(*entity, mesh, palette) {
            Some(job) => jobs.push(job),
            None => stats.skipped_bad_skin += 1,
        }
    }
    jobs
}

/// Skins one mesh against the palette; [`None`] (bad skin) on missing
/// skeleton/pose, array length defects, empty binds or out-of-range
/// joint indices.
fn skin_job(
    entity: Entity,
    mesh: &SkinnedMesh,
    palette: &HashMap<Entity, Vec<Mat4>>,
) -> Option<SkinJob> {
    let matrices = palette.get(&mesh.skeleton)?;
    if mesh.joints.len() != mesh.weights.len() {
        return None;
    }
    if mesh.positions.len() != mesh.joints.len() || mesh.normals.len() != mesh.joints.len() {
        return None;
    }
    if mesh.positions.is_empty() {
        return None;
    }
    let known = matrices.len();
    if mesh
        .joints
        .iter()
        .flatten()
        .any(|index| (*index as usize) >= known)
    {
        return None;
    }
    let (positions, normals) = skin_vertices(
        matrices,
        &mesh.joints,
        &mesh.weights,
        &mesh.positions,
        &mesh.normals,
    );
    Some(SkinJob {
        entity,
        vertex_count: positions.len().min(u32::MAX as usize) as u32,
        positions,
        normals,
    })
}

// ── Import bridge: format-neutral skin input (design
// `docs/animation-design.md` §4) ──
//
// Loader crates are foreign leaves: this crate must not depend on them, so
// the mapping travels through the plain input structs below. Their shape
// is the multi-format contract, not one format's echo: any loader (glTF
// today, FBX/Collada tomorrow) parses its own skin dialect and emits
// these values; per-format code stays in loader crates. Top-4 `u16`
// joints with normalized weights, parent links and inverse bind poses
// are universal to skeletal animation (and to GPU skinning), not glTF
// inventions — glTF names (`JOINTS_0`, `skins[]`) appear below only as
// the first producer's vocabulary. The host fills the structs from the
// import output and calls the builders, which validate. Local adapters in
// tests play the importer's role — no loader types cross this boundary,
// not even in `dev-dependencies`.

/// Imported skin topology in plain values: the no-dependency bridge into
/// [`skeleton_from_import`].
///
/// Multi-format contract: parent links, layout-explicit bind poses and
/// labels — whatever loader (glTF `skins[]` today) fills them.
#[derive(Debug, Clone, PartialEq)]
pub struct SkinImport {
    /// Parent joint per joint (`-1` = root); length equals the joint count.
    pub parents: Vec<i32>,
    /// Bind-pose inverses as layout-explicit matrices: each loader
    /// converts its own convention once (glTF column-major via
    /// `Mat4::from_cols_array_2d`, row-major sources via
    /// `Mat4::from_rows_array`), so this seam never speaks a format
    /// dialect.
    pub inverse_bind: Vec<Mat4>,
    /// Human-readable joint labels, diagnostics only.
    pub joint_names: Vec<String>,
}

/// Imported skinned primitive in plain values: the no-dependency bridge
/// into [`skinned_mesh_from_import`].
///
/// Multi-format contract: top-4 joints (`u16`) with normalized weights
/// plus resolved bind data — truncation and canonicalization happen in
/// the loader, idempotent re-normalization here. Positions stay verbatim
/// primitive-local: when the mesh node carries a world offset against
/// the skeleton root, the host bakes the relative transform with
/// [`bake_bind_transform`] first.
#[derive(Debug, Clone, PartialEq)]
pub struct SkinnedMeshImport {
    /// Influencing joints per vertex (top-4, `u16`).
    pub joints: Vec<[u16; 4]>,
    /// Influence weights per vertex (normalized, `sum == 1`).
    pub weights: Vec<[f32; 4]>,
    /// Bind-pose positions (engine units).
    pub positions: Vec<[f32; 3]>,
    /// Bind-pose shading normals (unit length).
    pub normals: Vec<[f32; 3]>,
    /// Bind-pose texture coordinates (passthrough to upload).
    pub uvs: Vec<[f32; 2]>,
    /// Triangle index list (passthrough to upload).
    pub indices: Vec<u32>,
}

/// Why a [`SkinnedMeshImport`] cannot become a [`SkinnedMesh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkinBuildError {
    /// No vertices: nothing to skin.
    Empty,
    /// `joints`/`weights`/`positions`/`normals`/`uvs` length mismatch, or a
    /// malformed index list (empty, not a multiple of 3, out of range).
    LengthMismatch,
    /// A joint index reaches past the skeleton joint count.
    JointOutOfRange,
}

/// Builds a [`Skeleton`] from imported skin topology.
///
/// Bind poses arrive layout-explicit (see [`SkinImport`]), so this only
/// runs the full [`Skeleton::validate`] — topology defects report as
/// [`SkelError`], never as truncated skeletons.
///
/// # Errors
///
/// Returns [`SkelError`] on empty, over-cap, length-mismatched or cyclic
/// topology (see [`Skeleton::validate`]).
///
/// # Examples
///
/// ```
/// # use glam::Mat4;
/// # use ornis_animation::{SkinImport, skeleton_from_import};
/// let skeleton = skeleton_from_import(&SkinImport {
///     parents: vec![-1, 0],
///     inverse_bind: vec![Mat4::IDENTITY; 2],
///     joint_names: vec!["root".to_string(), "tip".to_string()],
/// })
/// .expect("two-bone topology validates");
/// assert_eq!(skeleton.joint_count(), 2);
/// ```
pub fn skeleton_from_import(import: &SkinImport) -> Result<Skeleton, SkelError> {
    let skeleton = Skeleton::new(
        import
            .parents
            .iter()
            .map(|&raw| JointId::from_raw_opt(raw))
            .collect(),
        import.inverse_bind.clone(),
        import.joint_names.clone(),
    );
    skeleton.validate().map(|_| skeleton)
}

/// Builds a [`SkinnedMesh`] from an imported primitive.
///
/// Weights are canonicalized idempotently (finite positive sums normalize,
/// otherwise `(1,0,0,0)` on the first slot — same rule as the importer, so
/// double normalization is a no-op). Output buffers start as the bind pose
/// until `skel_skin_cpu` refreshes them (see [`SkinnedMesh::new`]).
///
/// # Errors
///
/// Returns [`SkinBuildError::Empty`] on zero vertices,
/// [`SkinBuildError::LengthMismatch`] on array or index-list defects, and
/// [`SkinBuildError::JointOutOfRange`] when a joint index reaches past
/// `joint_count`.
///
/// # Examples
///
/// ```
/// # use ornis_animation::{SkinnedMeshImport, skinned_mesh_from_import};
/// # use ornis_core::SmartStore;
/// let mut store = SmartStore::new();
/// let root = store.create_entity();
/// let mesh = skinned_mesh_from_import(
///     root,
///     1,
///     &SkinnedMeshImport {
///         joints: vec![[0, 0, 0, 0]],
///         weights: vec![[2.0, 2.0, 0.0, 0.0]],
///         positions: vec![[1.0, 0.0, 0.0]],
///         normals: vec![[0.0, 0.0, 1.0]],
///         uvs: vec![[0.0, 0.0]],
///         indices: vec![0, 0, 0],
///     },
/// )
/// .expect("single-joint bind builds");
/// assert_eq!(mesh.weights, vec![[0.5, 0.5, 0.0, 0.0]]);
/// ```
pub fn skinned_mesh_from_import(
    skeleton: Entity,
    joint_count: usize,
    import: &SkinnedMeshImport,
) -> Result<SkinnedMesh, SkinBuildError> {
    if import.positions.is_empty() {
        return Err(SkinBuildError::Empty);
    }
    if import.joints.len() != import.weights.len()
        || import.positions.len() != import.joints.len()
        || import.normals.len() != import.joints.len()
        || import.uvs.len() != import.joints.len()
    {
        return Err(SkinBuildError::LengthMismatch);
    }
    if import.indices.is_empty()
        || !import.indices.len().is_multiple_of(TRIANGLE_VERTS)
        || import
            .indices
            .iter()
            .any(|index| (*index as usize) >= import.positions.len())
    {
        return Err(SkinBuildError::LengthMismatch);
    }
    if import
        .joints
        .iter()
        .flatten()
        .any(|index| (*index as usize) >= joint_count)
    {
        return Err(SkinBuildError::JointOutOfRange);
    }
    let weights = import
        .weights
        .iter()
        .map(|weight| canonical_weights(*weight))
        .collect();
    Ok(SkinnedMesh::new(
        skeleton,
        import.joints.clone(),
        weights,
        import.positions.clone(),
        import.normals.clone(),
        import.uvs.clone(),
        import.indices.clone(),
    ))
}

/// Bakes a mesh-node offset into bind arrays before
/// [`skinned_mesh_from_import`].
///
/// The skinning contract emits world-space vertices, so bind data must be
/// expressed in the skeleton-root frame: the host passes
/// `inverse(root_world) * mesh_world` as a layout-explicit [`Mat4`] here
/// when the skinned mesh node is offset from the root (each loader builds
/// it from its own matrix convention). Positions map as points, normals
/// through the inverse-transpose 3×3 and renormalized unless they
/// collapse (then kept as-is, never NaN).
pub fn bake_bind_transform(positions: &mut [[f32; 3]], normals: &mut [[f32; 3]], matrix: Mat4) {
    for position in positions.iter_mut() {
        *position = matrix
            .transform_point3(Vec3::from_array(*position))
            .to_array();
    }
    let normals_part = normal_part(&matrix);
    for normal in normals.iter_mut() {
        let blended = normals_part * Vec3::from_array(*normal);
        if blended.length_squared() > DEGENERATE_LEN2 {
            *normal = blended.normalize().to_array();
        } else {
            *normal = blended.to_array();
        }
    }
}

#[cfg(test)]
mod import_tests {
    use super::*;

    fn two_bone_import() -> SkinImport {
        SkinImport {
            parents: vec![-1, 0],
            inverse_bind: vec![Mat4::IDENTITY; 2],
            joint_names: vec!["root".to_string(), "tip".to_string()],
        }
    }

    fn single_vertex_import() -> SkinnedMeshImport {
        SkinnedMeshImport {
            joints: vec![[0, 0, 0, 0]],
            weights: vec![[1.0, 0.0, 0.0, 0.0]],
            positions: vec![[1.0, 0.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]],
            uvs: vec![[0.0, 0.0]],
            indices: vec![0, 0, 0],
        }
    }

    #[test]
    fn skeleton_build_converts_binds_and_validates() {
        let skeleton = skeleton_from_import(&two_bone_import()).expect("two bones build");
        assert_eq!(skeleton.joint_count(), 2);
        assert_eq!(skeleton.inverse_bind.as_ref(), &[Mat4::IDENTITY; 2]);
        assert_eq!(
            skeleton.joint_names.as_ref(),
            &["root".to_string(), "tip".to_string()]
        );
    }

    #[test]
    fn skeleton_build_rejects_bad_topology() {
        // Length mismatch.
        let mut bad = two_bone_import();
        bad.inverse_bind.pop();
        assert_eq!(skeleton_from_import(&bad), Err(SkelError::LengthMismatch));
        // Cycle.
        let cyclic = SkinImport {
            parents: vec![1, 0],
            inverse_bind: vec![Mat4::IDENTITY; 2],
            joint_names: vec!["a".to_string(), "b".to_string()],
        };
        assert_eq!(skeleton_from_import(&cyclic), Err(SkelError::Cycle));
        // Empty.
        let empty = SkinImport {
            parents: Vec::new(),
            inverse_bind: Vec::new(),
            joint_names: Vec::new(),
        };
        assert_eq!(skeleton_from_import(&empty), Err(SkelError::Empty));
    }

    #[test]
    fn skeleton_build_keeps_translation_bind() {
        // Bind matrices arrive layout-explicit: a translated bind
        // survives the build verbatim.
        let mut import = two_bone_import();
        import.inverse_bind[0].w_axis = glam::Vec4::new(5.0, 0.0, 0.0, 1.0);
        let skeleton = skeleton_from_import(&import).expect("bind builds");
        assert_eq!(
            skeleton.inverse_bind[0].transform_point3(Vec3::ZERO),
            Vec3::new(5.0, 0.0, 0.0)
        );
    }

    #[test]
    fn mesh_build_normalizes_weights_idempotently() {
        let store_skeleton = SmartStore::new();
        let root = store_skeleton.create_entity();
        let mut import = single_vertex_import();
        import.weights = vec![[2.0, 2.0, 0.0, 0.0]];
        let mesh = skinned_mesh_from_import(root, 1, &import).expect("bind builds");
        assert_eq!(mesh.weights, vec![[0.5, 0.5, 0.0, 0.0]]);
        assert_eq!(mesh.vertex_count(), 1);
        // Zero-sum falls back to the first slot, joints untouched.
        import.weights = vec![[0.0, 0.0, 0.0, 0.0]];
        import.joints = vec![[3, 0, 0, 0]];
        let mesh = skinned_mesh_from_import(root, 4, &import).expect("fallback builds");
        assert_eq!(mesh.weights, vec![[1.0, 0.0, 0.0, 0.0]]);
        assert_eq!(mesh.joints, vec![[3, 0, 0, 0]]);
    }

    #[test]
    fn mesh_build_rejects_defects() {
        let store = SmartStore::new();
        let root = store.create_entity();
        // Empty.
        let mut import = single_vertex_import();
        import.positions.clear();
        assert_eq!(
            skinned_mesh_from_import(root, 1, &import),
            Err(SkinBuildError::Empty)
        );
        // Length mismatch (uvs short).
        let mut import = single_vertex_import();
        import.uvs.clear();
        assert_eq!(
            skinned_mesh_from_import(root, 1, &import),
            Err(SkinBuildError::LengthMismatch)
        );
        // Bad index list (out of range).
        let mut import = single_vertex_import();
        import.indices = vec![0, 0, 7];
        assert_eq!(
            skinned_mesh_from_import(root, 1, &import),
            Err(SkinBuildError::LengthMismatch)
        );
        // Joint out of range.
        let import = single_vertex_import();
        assert_eq!(
            skinned_mesh_from_import(root, 0, &import),
            Err(SkinBuildError::JointOutOfRange)
        );
    }

    #[test]
    fn bake_applies_offset_to_positions_and_normals() {
        // Translation +X plus a 90° Z spin: points ride fully, normals spin.
        let rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let matrix = Mat4::from_rotation_translation(rotation, Vec3::new(1.0, 0.0, 0.0));
        let mut positions = vec![[1.0, 0.0, 0.0]];
        let mut normals = vec![[1.0, 0.0, 0.0]];
        bake_bind_transform(&mut positions, &mut normals, matrix);
        assert!((Vec3::from_array(positions[0]) - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-6);
        assert!((Vec3::from_array(normals[0]) - Vec3::Y).length() < 1e-6);
    }
}
