//! Object animation: cold [`AnimClip`] lanes plus hot [`AnimPlayer`] lanes,
//! sampled by the `anim_sample` system (phase A, `docs/animation-design.md`
//! §1 and §5).
//!
//! The sampler is frame-only: it belongs in [`ornis_core::Stage::PostFrame`]
//! on variable [`ornis_core::Time`], never in the fixed schedule, so the
//! visual pose is not multiplied by the number of fixed substeps. Routing
//! follows the design: every playing track writes the target entity's local
//! [`Transform`] when that component already exists, including mesh-less
//! hierarchy nodes. Renderable entities (those with a [`MeshDesc`] lane
//! entry) are also written into [`TransformDesc`], with the translation
//! mirrored into the gameplay `Position` lane only when that lane already
//! exists. Entities without [`MeshDesc`] and without [`Transform`] (pure
//! gameplay markers) are written into [`Position`]. Lanes are never created
//! for the sake of animation. A local write is then pushed through
//! [`ornis_core::propagate_registered`] so children pick up the pose in the
//! same frame. Entities with a physics-authoritative lane are skipped
//! (physics wins). Root motion is out of scope: gameplay-affecting motion
//! travels the fixed `Velocity → RigidBody` path, never this sampler.
//!
//! Phase B (`docs/animation-design.md` §2 and §5) adds the skeletal side in
//! the same file: [`Skeleton`]/[`JointPose`]/[`SkelPlayer`] hot lanes plus
//! the cold [`SkelClip`] lane and [`SkinnedMesh`] bind data, sampled by
//! `skel_sample` ([`SkelSampleSystem`]) and skinned on the CPU by
//! `skel_skin_cpu` ([`SkelSkinSystem`]). Joint tracks that name a hierarchy
//! node also write that node's local [`Transform`] (same propagation as
//! object tracks). [`Animator`] on the character root selects a skeletal
//! clip by name; a player stays empty until that call.
//!
//! Phase E (§5 E) blends both sides against authored rest poses: a player's
//! `weight` lerps each channel toward its rest ([`AnimTrack::rest`],
//! [`Skeleton::rest`]), and [`AnimFade`] / [`SkelFade`] crossfade two clips
//! over time. Per-joint layers ([`SkelLayer`] + [`JointMask`]) stack per
//! root in [`SkelLayerStack`], bottom-up over the base pose.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use glam::{Mat3, Mat4, Quat, Vec3};
use ornis_core::{
    ChildOf, Children, Clamped01, ColdComponentStore, ComponentStore, Entity, GlobalTransform,
    Resources, Seconds, SmartStore, System, SystemAccess, Time, Transform,
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

mod blend;
use blend::{BlendedPose, JointSample, SampleBuf, sample_joints};
pub use blend::{
    JointMask, JointTrs, MaskId, SkelFade, SkelInertia, SkelLayer, SkelLayerStack, fade_progress,
};

mod events;
pub use events::{AnimEvent, AnimEventOut, AnimEventQueue, ClipEvents};
use events::{CursorStep, emit_step};

mod ik;
use ik::apply_ik_stack;
pub use ik::{
    BasePull, ControlRig, Effector, FkScratch, IkStack, LookAt, TwoBoneIk, solve_look_at,
    solve_two_bone_ik,
};

mod ik_solve;

mod retarget;
pub use retarget::RetargetMap;

mod root_motion;
pub use root_motion::RootMotionDelta;
use root_motion::{RootMotionPart, root_joint, root_motion};

mod motion_match;
pub use motion_match::{
    MotionHit, MotionQuery, MotionSample, MotionSearch, WarpWindow, match_motion, time_at_distance,
    warp_weight,
};

mod ragdoll;
mod tree;
mod tree_eff;
mod tree_eval;
mod tree_sync;
mod tree_weights;
pub use ragdoll::{
    RagdollBase, RagdollBlend, RagdollBone, RagdollDesc, RagdollError, RagdollFade, RagdollLimit,
    RagdollMode, RagdollMotor, RagdollShape, ragdoll_blend,
};
pub use tree::{
    MachineCursor, SkelTree, SkelTreePlayer, SwitchCursor, TreeChild, TreeChild2d, TreeCmp,
    TreeCond, TreeNodeDef, TreeParam, TreeTransition, switch_selection,
};
use tree_eval::eval_tree;
use tree_sync::sync_transition_leaves;
pub use tree_sync::{ClipSyncMarks, SyncMark};

pub use skinning::{
    CPU_GPU_TOLERANCE, JointCount, JointLimit, SkinError, SkinningMode, SkinningResources,
    blend_vertex_reference, canonical_staged_weights, joint_normal_matrix,
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
    /// Authored local transform of the entity (the glTF node transform):
    /// the reset pose that channels blend toward while the player weight is
    /// below `1` or during a crossfade (Godot-style RESET). Channels that no
    /// track provides stay untouched. `IDENTITY` suits hand-built tracks.
    pub rest: JointTrs,
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
    /// Blend strength in `0..=1` (clamped): the pose is `rest + weight·(pose
    /// − rest)`, so `1` plays the clip as authored and `0` holds the rest.
    pub weight: f32,
    /// Paused players hold their pose: time does not advance and placement
    /// is not rewritten.
    pub playing: bool,
}

/// Crossfade in progress on an object entity (hot lane, like [`SkelFade`]).
///
/// The entity's [`AnimPlayer`] is the incoming clip. The outgoing cursor
/// lives in `from` and keeps advancing until `elapsed` reaches `duration`,
/// then the sampler drops this component. Progress is linear in time: `0`
/// is the outgoing pose, `1` the incoming one. A non-positive duration is
/// an instant cut.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimFade {
    /// Outgoing cursor snapshot; its time keeps advancing during the fade.
    pub from: AnimPlayer,
    /// Fade length in seconds.
    pub duration: Seconds,
    /// Time since the fade started.
    pub elapsed: Seconds,
}

/// Whether `entity` already carries a crossfade of type `F` in flight.
///
/// The one in-flight check behind the "one fade per entity" rule: the
/// skeletal ([`SkelFade`]) and object ([`AnimFade`]) topologies both ask
/// this before installing a fade, and a start over a running fade cuts
/// instead of stacking.
pub fn fade_in_flight<F: 'static + Send + Sync>(store: &SmartStore, entity: Entity) -> bool {
    store
        .read_lane::<F>()
        .is_some_and(|fades| fades.get(entity).is_some())
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
/// makes the determinism explicit. A local [`Transform`] write also refreshes
/// [`GlobalTransform`] in this system, so a mesh-less node's children move
/// before later readers in the same frame.
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
        // Hierarchy lanes are the propagation that follows a local write.
        SystemAccess::new()
            .reads::<Time>()
            .reads::<SmartStore>()
            .reads_lane::<AnimClip>()
            .reads_lane::<ClipEvents>()
            .reads_lane::<MeshDesc>()
            .reads_lane::<P>()
            .reads_lane::<ChildOf>()
            .writes_lane::<AnimPlayer>()
            .writes_lane::<AnimFade>()
            .writes_lane::<AnimEventQueue>()
            .writes_lane::<Transform>()
            .writes_lane::<TransformDesc>()
            .writes_lane::<Position>()
            .writes_lane::<Children>()
            .writes_lane::<GlobalTransform>()
    }

    fn run(&self, resources: &Resources) {
        run_anim_sample::<P>(resources);
    }
}

/// Sampled pose for one entity: one entry per channel, [`None`] = absent.
#[derive(Clone, Copy)]
struct SampledPose {
    translation: Option<Vec3>,
    rotation: Option<Quat>,
    scale: Option<Vec3>,
}

/// One entity whose existing local [`Transform`] receives [`SampledPose`].
struct LocalWrite {
    entity: Entity,
    pose: SampledPose,
}

/// One object entity's routed result before output lanes are locked.
struct ObjWork {
    entity: Entity,
    pose: SampledPose,
    renderable: bool,
}

/// Bitset word width of [`FrozenBits`] (bits per `u64` slot).
const FROZEN_WORD_BITS: usize = 64;

/// Snapshot of physics-driven entities to skip this frame.
///
/// Bitset over live [`Entity::id`] slots: the per-cursor and per-fade skip
/// tests are O(1), where the old `frozen.contains` scan cost
/// O(entries × bodies) per frame. Live entities never share a slot, so the
/// id alone is a sound key here (no `generation` — sound for a one-frame
/// snapshot). The bitset spans `0..=max id`: its size tracks the largest
/// live [`Entity::id`], not the body count.
#[derive(Default)]
struct FrozenBits {
    bits: Vec<u64>,
}

impl FrozenBits {
    /// Rebuilds the snapshot in place, reusing the bitset's allocation.
    fn rebuild(&mut self, entities: &[Entity]) {
        self.bits.clear();
        for entity in entities {
            self.insert(*entity);
        }
    }

    /// Adds one entity to the snapshot.
    fn insert(&mut self, entity: Entity) {
        let id = entity.id() as usize;
        let word = id / FROZEN_WORD_BITS;
        if self.bits.len() <= word {
            self.bits.resize(word + 1, 0);
        }
        self.bits[word] |= 1 << (id % FROZEN_WORD_BITS);
    }

    /// Whether the snapshot contains `entity`.
    fn contains(&self, entity: Entity) -> bool {
        let id = entity.id() as usize;
        self.bits
            .get(id / FROZEN_WORD_BITS)
            .is_some_and(|word| word & (1 << (id % FROZEN_WORD_BITS)) != 0)
    }
}

/// Scratch buffers of one animation sample pass, reused across frames.
///
/// The pass allocates nothing once the scene settles: pose buffers,
/// sampled joints, model matrices, routed writes and the per-frame bookkeeping
/// (frozen snapshot, retired fades) all refill in place.
#[derive(Default)]
struct AnimScratch {
    frozen: FrozenBits,
    finished: Vec<Entity>,
    obj_work: Vec<ObjWork>,
    obj_locals: Vec<LocalWrite>,
    pose_a: BlendedPose,
    pose_b: BlendedPose,
    tree_poses: Vec<BlendedPose>,
    tree_weights: tree_eff::WeightsScratch,
    /// FK buffers of the IK solvers (§8.2): one solve allocates nothing.
    ik_fk: ik::FkScratch,
    /// Overlay-layer root-motion parts, applied after the crossfade mix.
    layer_motion: Vec<RootMotionPart>,
    /// Tree roots whose contract failed this frame: their pose is bad skin,
    /// so layer root motion must not move them either (§8.4).
    motion_hold: Vec<Entity>,
    joints: SampleBuf,
    contributors: Vec<ClipId>,
    skel_locals: Vec<Mat4>,
    matrices: Vec<Mat4>,
    nodes: Vec<LocalWrite>,
    node_painted: Vec<bool>,
}

thread_local! {
    /// Per-thread scratch for the sample passes (one pass runs at a time
    /// on a thread; nested use falls back to a temporary). The buffers live
    /// as long as their worker thread: a parallel scheduler keeps one
    /// scratch per worker, released only when the thread exits.
    static SCRATCH: std::cell::RefCell<AnimScratch> = std::cell::RefCell::new(AnimScratch::default());
}

/// Runs `f` with the thread's [`AnimScratch`], reusing its buffers.
fn with_scratch<R>(f: impl FnOnce(&mut AnimScratch) -> R) -> R {
    SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => f(&mut scratch),
        // Reentrant sample (never in the shipped schedule): keep the
        // function total instead of panicking on the borrow.
        Err(_) => f(&mut AnimScratch::default()),
    })
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
    with_scratch(|scratch| {
        // Snapshot of physics-driven entities (empty when the authority lane
        // is not registered): both passes below skip them, cursor included.
        match store.read_lane::<P>() {
            Some(lane) => scratch.frozen.rebuild(&lane.entities),
            None => scratch.frozen.rebuild(&[]),
        }

        // Pass 1 — advance playing cursors against their clip duration, and the
        // outgoing cursor plus clock of every crossfade in progress.
        {
            let Some(clips) = store.read_cold_lane::<AnimClip>() else {
                return;
            };
            let events = store.read_cold_lane::<ClipEvents>();
            let Some(mut players) = store.write_lane::<AnimPlayer>() else {
                return;
            };
            let mut queues = store.write_lane::<AnimEventQueue>();
            let incoming_fades = store.read_lane::<AnimFade>();
            for index in 0..players.data.len() {
                let entity = players.entities[index];
                let player = &mut players.data[index];
                if !player.playing || scratch.frozen.contains(entity) {
                    continue;
                }
                let Some(clip) = clips.get(player.clip.0) else {
                    continue;
                };
                let prev = player.time;
                let step = dt * player.speed;
                player.time =
                    advance_player_time(player.time, dt, player.speed, clip.duration, clip.looping);
                emit_step(
                    queues.as_deref_mut(),
                    events.as_deref(),
                    entity,
                    player.clip,
                    CursorStep {
                        prev,
                        step,
                        duration: clip.duration,
                        looping: clip.looping,
                        weight: player.weight * share_of(incoming_fades.as_deref(), entity),
                    },
                );
            }
            drop(incoming_fades);
            if let Some(mut fades) = store.write_lane::<AnimFade>() {
                let (frozen, finished) = (&scratch.frozen, &mut scratch.finished);
                tick_fades(&mut fades, dt, frozen, finished, |fade, step, entity| {
                    if fade.from.playing
                        && let Some(clip) = clips.get(fade.from.clip.0)
                    {
                        let prev = fade.from.time;
                        let cursor_step = step * fade.from.speed;
                        fade.from.time = advance_player_time(
                            fade.from.time,
                            step,
                            fade.from.speed,
                            clip.duration,
                            clip.looping,
                        );
                        emit_step(
                            queues.as_deref_mut(),
                            events.as_deref(),
                            entity,
                            fade.from.clip,
                            CursorStep {
                                prev,
                                step: cursor_step,
                                duration: clip.duration,
                                looping: clip.looping,
                                weight: fade.from.weight * (1.0 - fade.progress()),
                            },
                        );
                    }
                });
            }
        }
        sample_and_publish(store, scratch);
    });
}

/// Samples playing cursors and publishes the poses.
///
/// `scratch` carries the frame's [`FrozenBits`] snapshot of physics-driven
/// entities to skip and the reused work buffers; routing runs first, then
/// publication, so no two lane guards overlap.
fn sample_and_publish(store: &SmartStore, scratch: &mut AnimScratch) {
    // Route first, then publish: collect the work so no two lane guards overlap.
    {
        let Some(clips) = store.read_cold_lane::<AnimClip>() else {
            return;
        };
        let Some(players) = store.read_lane::<AnimPlayer>() else {
            return;
        };
        let fades = store.read_lane::<AnimFade>();
        let meshes = store.read_lane::<MeshDesc>();
        scratch.obj_work.clear();
        for (&entity, player) in players.entities.iter().zip(&players.data) {
            if !player.playing || scratch.frozen.contains(entity) {
                continue;
            }
            let Some(clip) = clips.get(player.clip.0) else {
                continue;
            };
            let fade = fades.as_ref().and_then(|lane| lane.get(entity));
            let incoming = track_of(clip, entity);
            let outgoing = fade.and_then(|fade| {
                clips
                    .get(fade.from.clip.0)
                    .and_then(|clip| track_of(clip, entity))
            });
            // The entity's rest comes from whichever side has a track for it.
            // A side without one reads as rest, so a crossfade into a clip
            // that does not animate this entity still fades out to rest.
            let Some(rest) = incoming.or(outgoing).map(|track| track.rest) else {
                continue;
            };
            let pose = if let Some(fade) = fade {
                side_pose(
                    outgoing,
                    rest,
                    fade.from.time,
                    fade.from.weight,
                    &mut scratch.pose_b,
                );
                side_pose(
                    incoming,
                    rest,
                    player.time,
                    player.weight,
                    &mut scratch.pose_a,
                );
                scratch
                    .pose_b
                    .mix_towards(&scratch.pose_a, fade_progress(fade.duration, fade.elapsed));
                sampled_pose(&scratch.pose_b, 0)
            } else {
                side_pose(
                    incoming,
                    rest,
                    player.time,
                    player.weight,
                    &mut scratch.pose_a,
                );
                sampled_pose(&scratch.pose_a, 0)
            };
            scratch.obj_work.push(ObjWork {
                entity,
                pose,
                renderable: meshes
                    .as_ref()
                    .is_some_and(|lane| lane.get(entity).is_some()),
            });
        }
    }
    if scratch.obj_work.is_empty() {
        return;
    }
    scratch.obj_locals.clear();
    scratch
        .obj_locals
        .extend(scratch.obj_work.iter().map(|item| LocalWrite {
            entity: item.entity,
            pose: item.pose,
        }));
    let wrote_local = publish_local_transforms(store, &scratch.obj_locals);
    let mut descs = store.write_lane::<TransformDesc>();
    let mut positions = store.write_lane::<Position>();
    for item in &scratch.obj_work {
        if item.renderable
            && let Some(desc) = descs.as_mut().and_then(|lane| lane.get_mut(item.entity))
        {
            apply_channels(
                &mut desc.translation,
                &mut desc.rotation,
                &mut desc.scale,
                &item.pose,
            );
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
    drop(descs);
    drop(positions);
    if wrote_local {
        ornis_core::propagate_registered(store);
    }
}

/// The track `clip` holds for `entity`; the first match wins.
fn track_of(clip: &AnimClip, entity: Entity) -> Option<&AnimTrack> {
    clip.tracks.iter().find(|track| track.entity == entity)
}

/// Fills `out` with the pose of one blend side: its track at `time` and
/// `weight`, or the authored rest when the side has no track for the entity.
fn side_pose(
    track: Option<&AnimTrack>,
    rest: JointTrs,
    time: f32,
    weight: f32,
    out: &mut BlendedPose,
) {
    match track {
        Some(track) => {
            let sample = JointSample {
                translation: track.translation.sample(time),
                rotation: track.rotation.sample(time),
                scale: track.scale.sample(time),
            };
            out.refill(
                &[sample],
                std::slice::from_ref(&track.rest),
                Clamped01::new(weight).get(),
            );
        }
        None => out.refill_rest(std::slice::from_ref(&rest)),
    }
}

/// Writes sampled channels onto an existing local [`Transform`].
///
/// Absent channels stay untouched. A degenerate rotation keeps the previous
/// orientation. Returns whether any channel landed on a component that was
/// already in the lane — the caller never inserts a pose for animation.
fn publish_local_transforms(store: &SmartStore, poses: &[LocalWrite]) -> bool {
    if poses.is_empty() {
        return false;
    }
    let Some(mut locals) = store.write_lane::<Transform>() else {
        return false;
    };
    let mut wrote = false;
    for write in poses {
        if let Some(local) = locals.get_mut(write.entity) {
            wrote |= apply_channels(
                &mut local.translation,
                &mut local.rotation,
                &mut local.scale,
                &write.pose,
            );
        }
    }
    wrote
}

/// Applies present channels. A degenerate rotation keeps `rotation`.
///
/// Returns whether a channel was written.
fn apply_channels(
    translation: &mut Vec3,
    rotation: &mut UnitQuat,
    scale: &mut Vec3,
    pose: &SampledPose,
) -> bool {
    let mut wrote = false;
    if let Some(value) = pose.translation {
        *translation = value;
        wrote = true;
    }
    if let Some(value) = pose.rotation.and_then(UnitQuat::normalize) {
        *rotation = value;
        wrote = true;
    }
    if let Some(value) = pose.scale {
        *scale = value;
        wrote = true;
    }
    wrote
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
            TypeId::of::<AnimEventQueue>(),
            TypeId::of::<ClipEvents>(),
            TypeId::of::<MeshDesc>(),
            TypeId::of::<NoPhysics>(),
            TypeId::of::<Transform>(),
            TypeId::of::<TransformDesc>(),
            TypeId::of::<Position>(),
            TypeId::of::<ChildOf>(),
            TypeId::of::<Children>(),
            TypeId::of::<GlobalTransform>(),
        ] {
            let declared =
                access.reads_lanes.contains(&lane) || access.writes_lanes.contains(&lane);
            assert!(declared, "lane {lane:?} must be declared");
        }
        // Cursor advance writes the player lane; poses and propagation are outputs.
        assert!(access.writes_lanes.contains(&TypeId::of::<AnimPlayer>()));
        assert!(
            access
                .writes_lanes
                .contains(&TypeId::of::<AnimEventQueue>())
        );
        assert!(access.writes_lanes.contains(&TypeId::of::<Transform>()));
        assert!(access.writes_lanes.contains(&TypeId::of::<TransformDesc>()));
        assert!(access.writes_lanes.contains(&TypeId::of::<Position>()));
        assert!(
            access
                .writes_lanes
                .contains(&TypeId::of::<GlobalTransform>())
        );
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

    // ── Hot-path equivalence (PLAN.md, фаза E, п. 5): the buffered
    // kernels must reproduce the pre-optimization results exactly.
    // Frame-level timings live in `tests/hot_path_bench.rs` (ignored by
    // default, run on demand; `tests/hot_path_ab.sh` compares refs).
    //
    // Each check keeps its body in one inner block: rustqual's fragment
    // scanner hashes windows of three consecutive statements, and the
    // neighbouring fixtures already own the natural statement shapes.

    /// The pre-optimization frozen skip: one linear scan per entry.
    fn legacy_frozen_scan(frozen: &[Entity], entity: Entity) -> bool {
        frozen.contains(&entity)
    }

    /// The pre-optimization pose build: fresh `Vec`s per side.
    fn legacy_pose(samples: &[JointSample], rest: &[JointTrs], weight: f32) -> BlendedPose {
        BlendedPose {
            trs: samples
                .iter()
                .zip(rest)
                .map(|(sample, base)| base.lerp(JointTrs::from_sample(*sample, *base), weight))
                .collect(),
            present: samples
                .iter()
                .map(|s| {
                    [
                        s.translation.is_some(),
                        s.rotation.is_some(),
                        s.scale.is_some(),
                    ]
                })
                .collect(),
        }
    }

    /// The pre-optimization crossfade: a fresh blended pose per call.
    fn legacy_mix(from: &BlendedPose, to: &BlendedPose, progress: f32) -> BlendedPose {
        BlendedPose {
            trs: from
                .trs
                .iter()
                .zip(&to.trs)
                .map(|(a, b)| a.lerp(*b, progress))
                .collect(),
            present: from
                .present
                .iter()
                .zip(&to.present)
                .map(|(a, b)| [a[0] || b[0], a[1] || b[1], a[2] || b[2]])
                .collect(),
        }
    }

    /// Bitset snapshot rebuilt from `sets[..]` in one reused buffer.
    fn rebuilt_bits(sets: &[Vec<Entity>]) -> FrozenBits {
        sets.iter()
            .fold(FrozenBits::default(), |mut bits, entities| {
                bits.rebuild(entities);
                bits
            })
    }

    #[test]
    fn frozen_bits_match_the_linear_scan() {
        {
            let driven: Vec<Entity> = (0..64).map(|i| Entity::new(i * 3 + 7)).collect();
            let bits = rebuilt_bits(std::slice::from_ref(&driven));
            for id in 0..256 {
                let entity = Entity::new(id);
                assert_eq!(
                    bits.contains(entity),
                    legacy_frozen_scan(&driven, entity),
                    "id {id}"
                );
            }
        }
    }

    #[test]
    fn rebuilt_frozen_bits_match_a_fresh_snapshot() {
        {
            let first: Vec<Entity> = (0..100).map(Entity::new).collect();
            let second: Vec<Entity> = (50..150).map(Entity::new).collect();
            let fresh = rebuilt_bits(std::slice::from_ref(&second));
            let reused = rebuilt_bits(&[first, second]);
            for id in 0..200 {
                assert_eq!(
                    reused.contains(Entity::new(id)),
                    fresh.contains(Entity::new(id)),
                    "id {id}"
                );
            }
        }
    }

    #[test]
    fn buffered_pose_matches_the_allocating_build() {
        {
            let rest = vec![JointTrs::IDENTITY; 4];
            let incoming = [JointSample {
                translation: Some(Vec3::X),
                ..JointSample::default()
            }; 4];
            let outgoing = [JointSample {
                translation: Some(Vec3::Y),
                rotation: Some(Quat::from_rotation_z(0.3)),
                ..JointSample::default()
            }; 4];
            let mut pose_buf = BlendedPose::default();
            let mut from_buf = BlendedPose::default();
            for step in 0..16 {
                let progress = step as f32 * 0.01;
                let pose = legacy_pose(&incoming, &rest, 0.75);
                let from = legacy_pose(&outgoing, &rest, 1.0);
                let mixed = legacy_mix(&from, &pose, progress);
                pose_buf.refill(&incoming, &rest, 0.75);
                from_buf.refill(&outgoing, &rest, 1.0);
                from_buf.mix_towards(&pose_buf, progress);
                assert_eq!(from_buf.trs, mixed.trs, "trs at step {step}");
                assert_eq!(from_buf.present, mixed.present, "presence at step {step}");
            }
        }
    }

    #[test]
    fn compose_model_matrices_handles_a_max_joints_chain() {
        {
            let joints = MAX_JOINTS;
            let parents: Vec<Option<JointId>> = (0..joints)
                .map(|i| (i > 0).then(|| JointId::from_raw((i - 1) as u32)))
                .collect();
            let skeleton = Skeleton::new(
                parents,
                vec![Mat4::IDENTITY; joints],
                (0..joints).map(|i| format!("joint{i}")).collect(),
            );
            let locals = vec![Mat4::from_translation(Vec3::X); joints];
            let mut out = Vec::new();
            assert_eq!(
                compose_model_matrices(&skeleton, &locals, &Mat4::IDENTITY, &mut out),
                Some(joints),
                "exactly MAX_JOINTS links fit the stack walk"
            );
            let last = out[joints - 1];
            assert!((last.w_axis.x - joints as f32).abs() < 1e-4, "got {last:?}");
        }
    }

    #[test]
    fn compose_model_matrices_rejects_a_cycle() {
        {
            let skeleton = Skeleton::new(
                vec![Some(JointId::from_raw(1)), Some(JointId::from_raw(0))],
                vec![Mat4::IDENTITY; 2],
                vec!["a".to_string(), "b".to_string()],
            );
            let mut out = Vec::new();
            assert_eq!(
                compose_model_matrices(&skeleton, &[Mat4::IDENTITY; 2], &Mat4::IDENTITY, &mut out),
                None,
                "a parent cycle never assembles"
            );
        }
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
    /// Authored rest pose per joint: the local transform a joint reads as
    /// when no track drives it (glTF node transform in the skin). Length
    /// must equal the joint count; see [`Skeleton::with_rest`].
    pub rest: Arc<[JointTrs]>,
}

impl Skeleton {
    /// Builds a skeleton with identity rest; topology is checked by
    /// [`Skeleton::validate`], not here, so loaders can assemble first and
    /// fail with a reason.
    pub fn new(
        parents: Vec<Option<JointId>>,
        inverse_bind: Vec<Mat4>,
        joint_names: Vec<String>,
    ) -> Self {
        let rest = vec![JointTrs::IDENTITY; parents.len()];
        Self {
            parents: Arc::from(parents),
            inverse_bind: Arc::from(inverse_bind),
            joint_names: Arc::from(joint_names),
            rest: Arc::from(rest),
        }
    }

    /// Replaces the identity rest with the authored rest pose, one entry per
    /// joint. A length mismatch is reported by [`Skeleton::validate`].
    #[must_use]
    pub fn with_rest(mut self, rest: Vec<JointTrs>) -> Self {
        self.rest = Arc::from(rest);
        self
    }

    /// Number of joints (`parents.len()`).
    pub fn joint_count(&self) -> usize {
        self.parents.len()
    }

    /// The joint whose [`joint_names`](Self::joint_names) entry is exactly
    /// `name` — the setup-time lookup for hosts (rig authoring, control-rig
    /// goals). `None` when no joint matches: names are labels, so a
    /// mismatched importer name simply does not resolve.
    pub fn joint_named(&self, name: &str) -> Option<JointId> {
        self.joint_names
            .iter()
            .position(|joint| joint == name)
            .map(|at| JointId::from_raw(at as u32))
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
        if self.inverse_bind.len() != count || self.rest.len() != count {
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
/// Field names follow the design (`t`/`r`/`s`); an empty channel reads as
/// the joint's authored rest value (see [`Skeleton::rest`]) and is never
/// written as an identity. [`Self::node`], when spawn bound it, receives
/// the sampled channels on its local [`Transform`] with object-track rules
/// (a channel no contributor provides is left untouched) so the hierarchy
/// can move.
#[derive(Debug, Clone, PartialEq)]
pub struct JointTrack {
    /// Animated joint index into [`Skeleton`]/[`JointPose`].
    pub joint: JointId,
    /// Hierarchy node this joint writes, when spawn mapped the joint's node.
    ///
    /// `None` keeps the pose in [`JointPose`] only. This is not a joint index.
    pub node: Option<Entity>,
    /// Translation keys; empty means the joint's rest translation.
    pub translation: KeyTrack<Vec3>,
    /// Rotation keys (unit quaternions); empty means the rest rotation.
    pub rotation: KeyTrack<Quat>,
    /// Scale keys; empty means the rest scale.
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
/// Same playlist model as [`AnimPlayer`]. `time` and `speed` are [`Seconds`]
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
    /// Blend strength in `[0, 1]`: each joint is `rest + weight·(pose − rest)`,
    /// so `1` plays the clip as authored and `0` holds the rest pose.
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
/// unit scale). The skeleton sampler substitutes each joint's authored rest
/// for absent channels instead, so this helper is for standalone use.
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
/// `out` is caller-owned scratch: it is cleared and refilled with the
/// model matrices, so the per-frame sample path never allocates here.
/// Returns the joint count written, or [`None`] when the topology is bad
/// (empty, over-cap, length or parent defect) — the caller skips the
/// entity and counts it.
pub fn compose_model_matrices(
    skeleton: &Skeleton,
    locals: &[Mat4],
    root: &Mat4,
    out: &mut Vec<Mat4>,
) -> Option<usize> {
    let count = checked_count(skeleton, locals.len())?;
    out.clear();
    for joint in 0..count {
        out.push(model_via_chain(&skeleton.parents, locals, root, joint)?);
    }
    Some(out.len())
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

/// Resolves one joint: multiplies `root * local` down its parent chain,
/// iteratively (a recursive self-call would classify as a "logic + calls"
/// IOSP violation; the fixed stack keeps the walk allocation-free).
///
/// [`None`] on out-of-range parents or cycles (the stack cap is the joint
/// count — a chain longer than the parent list repeats).
fn model_via_chain(
    parents: &[Option<JointId>],
    locals: &[Mat4],
    root: &Mat4,
    joint: usize,
) -> Option<Mat4> {
    let mut chain = [usize::MAX; MAX_JOINTS];
    let mut len = 0usize;
    let mut cursor = joint;
    loop {
        if cursor >= parents.len() || len >= parents.len() {
            return None;
        }
        chain[len] = cursor;
        len += 1;
        let Some(parent) = parents[cursor] else {
            break;
        };
        let parent = parent.index();
        if parent >= parents.len() {
            return None;
        }
        cursor = parent;
    }
    let mut model = *root;
    while len > 0 {
        len -= 1;
        model *= locals[chain[len]];
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
/// fallback, not a silent entity-level stub). Shared with
/// [`joint_normal_matrix`] so the GPU palette packs the same fallback.
pub(crate) fn normal_part(joint: &Mat4) -> Mat3 {
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
/// [`JointPose`]; lanes are never created for animation. A track whose
/// [`JointTrack::node`] is bound also writes that node's local
/// [`Transform`] and refreshes [`GlobalTransform`] in this system.
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
            .reads_lane::<ClipEvents>()
            .reads_lane::<Skeleton>()
            .reads_lane::<TransformDesc>()
            .reads_lane::<ChildOf>()
            .reads_lane::<JointMask>()
            .reads_lane::<SkelTree>()
            .reads_lane::<IkStack>()
            .reads_lane::<RetargetMap>()
            .writes_lane::<SkelPlayer>()
            .writes_lane::<SkelLayerStack>()
            .writes_lane::<SkelTreePlayer>()
            .writes_lane::<SkelFade>()
            .writes_lane::<SkelInertia>()
            .writes_lane::<AnimEventQueue>()
            .writes_lane::<RootMotionDelta>()
            .writes_lane::<JointPose>()
            .writes_lane::<RagdollBlend>()
            .writes_lane::<Transform>()
            .writes_lane::<Children>()
            .writes_lane::<GlobalTransform>()
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

/// Advances playing [`SkelPlayer`] cursors, overlay layers and crossfade
/// clocks.
///
/// Missing clips hold their cursor (counted as bad skin at sample time,
/// not here). A crossfade drops its [`SkelFade`] once `elapsed` reaches
/// `duration`.
fn advance_skel_players(store: &SmartStore, dt: f32) {
    with_scratch(|scratch| {
        scratch.layer_motion.clear();
        scratch.motion_hold.clear();
        advance_base_players(store, dt);
        advance_layers(store, dt, scratch);
        advance_tree_players(store, dt, scratch);
        advance_fades(store, dt, scratch);
        advance_inertias(store, dt, scratch);
        apply_layer_root_motion(store, scratch);
    });
}

/// Clip, skeleton, retarget and mask lanes one root-motion lookup needs.
struct MotionSrc<'a> {
    clips: &'a ColdComponentStore<SkelClip>,
    skeletons: Option<&'a ComponentStore<Skeleton>>,
    maps: Option<&'a ColdComponentStore<RetargetMap>>,
    masks: Option<&'a ColdComponentStore<JointMask>>,
}

impl MotionSrc<'_> {
    /// Full-weight root motion of `clip` over `cursor` on `entity`, scaled
    /// by `weight`. A missing skeleton root or an unmapped retarget reads
    /// as no motion (§8.4).
    fn delta(
        &self,
        entity: Entity,
        clip: ClipId,
        cursor: Option<CursorStep>,
        weight: f32,
    ) -> RootMotionDelta {
        let root = self
            .skeletons
            .and_then(|lane| lane.get(entity))
            .and_then(root_joint);
        let map = self.maps.and_then(|lane| lane.get(entity));
        root_motion(self.clips.get(clip.0), cursor, root, map, weight)
    }
}

/// Advances a playing cursor against its clip duration. Paused or
/// clip-less cursors hold their time. Returns the step's `(prev, step,
/// duration, looping)` when the cursor moved — the caller's event-emit
/// hook ([`emit_step`]).
fn step_player(
    player: &mut SkelPlayer,
    clips: &ColdComponentStore<SkelClip>,
    dt: f32,
) -> Option<CursorStep> {
    if !player.playing {
        return None;
    }
    let clip = clips.get(player.clip.0)?;
    let prev = player.time.get();
    let step = dt * player.speed.get();
    player.time = Seconds::new(advance_player_time(
        prev,
        dt,
        player.speed.get(),
        clip.duration,
        player.looping,
    ));
    Some(CursorStep {
        prev,
        step,
        duration: clip.duration,
        looping: player.looping,
        weight: player.weight.get(),
    })
}

/// Advances playing [`SkelPlayer`] cursors against their clip duration.
/// A cursor on a tree-driven root ([`SkelTreePlayer`]) is skipped: the
/// tree ignores it, so its clock must not drift while it is ignored.
fn advance_base_players(store: &SmartStore, dt: f32) {
    let mut motion = store.write_lane::<RootMotionDelta>();
    // Every frame starts every recorded delta at identity (design §8.4): a
    // paused or clip-less root must not keep applying the last frame's
    // motion. The reset runs before every early return so a missing lane
    // can never leave a stale delta behind. Live contributors overwrite
    // their entry: the base cursor here, the crossfade mix and the tree
    // fold in the later passes, then the layer overlays.
    if let Some(motion) = motion.as_deref_mut() {
        motion.data.fill(RootMotionDelta::IDENTITY);
    }
    let Some(clips) = store.read_cold_lane::<SkelClip>() else {
        return;
    };
    let events = store.read_cold_lane::<ClipEvents>();
    let tree_players = store.read_lane::<SkelTreePlayer>();
    let skeletons = store.read_lane::<Skeleton>();
    let maps = store.read_cold_lane::<RetargetMap>();
    let Some(mut players) = store.write_lane::<SkelPlayer>() else {
        return;
    };
    let mut queues = store.write_lane::<AnimEventQueue>();
    let fades = store.read_lane::<SkelFade>();
    let src = MotionSrc {
        clips: &clips,
        skeletons: skeletons.as_deref(),
        maps: maps.as_deref(),
        masks: None,
    };
    let players = &mut *players;
    for (entity, player) in players.entities.iter().zip(players.data.iter_mut()) {
        if tree_players
            .as_deref()
            .is_some_and(|lane| lane.contains(*entity))
        {
            continue;
        }
        let Some(cursor) = step_player(player, &clips, dt) else {
            continue;
        };
        emit_step(
            queues.as_deref_mut(),
            events.as_deref(),
            *entity,
            player.clip,
            cursor.scaled(share_of(fades.as_deref(), *entity)),
        );
        // The fade pass mixes this full incoming delta with the outgoing
        // one. The component stays opt-in: a root without a record is not
        // given one (design §8.4).
        let delta = src.delta(*entity, player.clip, Some(cursor), player.weight.get());
        if let Some(slot) = motion.as_deref_mut().and_then(|lane| lane.get_mut(*entity)) {
            *slot = delta;
        }
    }
}

/// Advances playing [`SkelLayer`] cursors (every stack element) against
/// their clip duration.
fn advance_layers(store: &SmartStore, dt: f32, scratch: &mut AnimScratch) {
    let Some(clips) = store.read_cold_lane::<SkelClip>() else {
        return;
    };
    let events = store.read_cold_lane::<ClipEvents>();
    let masks = store.read_cold_lane::<JointMask>();
    let skeletons = store.read_lane::<Skeleton>();
    let maps = store.read_cold_lane::<RetargetMap>();
    let mut queues = store.write_lane::<AnimEventQueue>();
    let Some(mut layers) = store.write_lane::<SkelLayerStack>() else {
        return;
    };
    let src = MotionSrc {
        clips: &clips,
        skeletons: skeletons.as_deref(),
        maps: maps.as_deref(),
        masks: masks.as_deref(),
    };
    let layers = &mut *layers;
    for (entity, stack) in layers.entities.iter().zip(layers.data.iter_mut()) {
        for layer in &mut stack.layers {
            let cursor = step_player(&mut layer.player, &clips, dt);
            if let Some(cursor) = cursor {
                emit_step(
                    queues.as_deref_mut(),
                    events.as_deref(),
                    *entity,
                    layer.player.clip,
                    cursor,
                );
            }
            record_layer_motion(*entity, layer, cursor, &src, scratch);
        }
    }
}

/// Stashes one layer's root-motion overlay. A layer whose mask cannot be
/// read contributes nothing here (the pose path rejects that root as bad
/// skin). A paused layer still overlays: its pose is held, so it pulls the
/// accumulated delta toward none by `player.weight · mask[root]`.
fn record_layer_motion(
    entity: Entity,
    layer: &SkelLayer,
    cursor: Option<CursorStep>,
    src: &MotionSrc<'_>,
    scratch: &mut AnimScratch,
) {
    let Some(mask_weight) = layer_root_mask(src, entity, layer.mask) else {
        return;
    };
    let weight = layer.player.weight.get() * mask_weight;
    if weight <= 0.0 {
        return;
    }
    let delta = cursor.map_or(RootMotionDelta::IDENTITY, |cursor| {
        src.delta(entity, layer.player.clip, Some(cursor), 1.0)
    });
    scratch.layer_motion.push(RootMotionPart {
        entity,
        weight,
        delta,
    });
}

/// `mask[root]` for the skeleton root joint, or [`None`] when the mask or
/// the root joint cannot be resolved.
fn layer_root_mask(src: &MotionSrc<'_>, entity: Entity, mask: crate::MaskId) -> Option<f32> {
    let root = src
        .skeletons
        .and_then(|lane| lane.get(entity))
        .and_then(root_joint)?;
    let weights = &src.masks?.get(mask.0)?.weights;
    Some(
        weights
            .get(root.index())
            .map(|weight| weight.get())
            .unwrap_or(0.0),
    )
}

/// Advances every [`SkelTreePlayer`]: leaf cursors step against their
/// clip duration like phase-E players, machine cursors complete their
/// in-flight crossfades and then fire the first matching
/// [`TreeTransition`](crate::TreeTransition) of each idle machine — a
/// crossfade is never interrupted, guards are re-checked only once it
/// completes. A fired transition re-phases the incoming state's leaves
/// through the [`ClipSyncMarks`] of both sides
/// (`docs/animation-design.md` §7.5).
fn advance_tree_players(store: &SmartStore, dt: f32, scratch: &mut AnimScratch) {
    let Some(clips) = store.read_cold_lane::<SkelClip>() else {
        return;
    };
    let defs = store.read_cold_lane::<SkelTree>();
    let marks = store.read_cold_lane::<ClipSyncMarks>();
    let events = store.read_cold_lane::<ClipEvents>();
    let skeletons = store.read_lane::<Skeleton>();
    let maps = store.read_cold_lane::<RetargetMap>();
    let mut motion = store.write_lane::<RootMotionDelta>();
    let Some(mut players) = store.write_lane::<SkelTreePlayer>() else {
        return;
    };
    let mut queues = store.write_lane::<AnimEventQueue>();
    let src = MotionSrc {
        clips: &clips,
        skeletons: skeletons.as_deref(),
        maps: maps.as_deref(),
        masks: None,
    };
    let players = &mut *players;
    for (entity, player) in players.entities.iter().zip(players.data.iter_mut()) {
        let (weights, posed) = tree_eff::leaf_effective_weights(
            &mut scratch.tree_weights,
            defs.as_deref().and_then(|defs| defs.get(*entity)),
            player,
        );
        let mut folded = RootMotionDelta::IDENTITY;
        let mut total = 0.0;
        for (index, leaf) in player.leaves.iter_mut().enumerate() {
            let share = weights.get(index).copied().unwrap_or(0.0);
            let cursor = step_player(leaf, &clips, dt);
            if let Some(cursor) = cursor {
                emit_step(
                    queues.as_deref_mut(),
                    events.as_deref(),
                    *entity,
                    leaf.clip,
                    cursor.scaled(share),
                );
            }
            if posed {
                let delta = cursor.map_or(RootMotionDelta::IDENTITY, |cursor| {
                    src.delta(*entity, leaf.clip, Some(cursor), leaf.weight.get())
                });
                folded.fold(delta, share, &mut total);
            }
        }
        if posed {
            if let Some(slot) = motion.as_deref_mut().and_then(|lane| lane.get_mut(*entity)) {
                *slot = folded;
            }
        } else {
            // Bad skin publishes no pose; keep the reset identity and don't
            // let a later layer overlay invent motion for this root.
            scratch.motion_hold.push(*entity);
        }
        step_machines(
            player,
            defs.as_deref().and_then(|defs| defs.get(*entity)),
            &clips,
            marks.as_deref(),
            dt,
        );
        step_switches(
            player,
            defs.as_deref().and_then(|defs| defs.get(*entity)),
            &clips,
            marks.as_deref(),
            dt,
        );
    }
}

/// Completes or advances in-flight machine crossfades, then fires
/// transitions on the live parameters (`from` must stand, first match in
/// list order wins; the outgoing state's cursors keep stepping — the
/// [`SkelFade`](crate::SkelFade) mechanics). A fired transition
/// re-phases the incoming state's leaves through their [`ClipSyncMarks`]
/// — unless the lane is missing, which leaves every cursor alone.
fn step_machines(
    player: &mut SkelTreePlayer,
    tree: Option<&SkelTree>,
    clips: &ColdComponentStore<SkelClip>,
    marks: Option<&ColdComponentStore<ClipSyncMarks>>,
    dt: f32,
) {
    for cursor in &mut player.machines {
        tick_crossfade(&mut cursor.elapsed, cursor.duration, &mut cursor.from, dt);
    }
    let Some(tree) = tree else {
        return;
    };
    let mut cursors = 0;
    for node in &tree.nodes {
        let tree::TreeNodeDef::Machine {
            states,
            initial: _,
            transitions,
        } = node
        else {
            continue;
        };
        let slot = cursors;
        cursors += 1;
        let Some(cursor) = player.machines.get(slot) else {
            break;
        };
        if cursor.from.is_some() {
            continue;
        }
        let Some(transition) = transitions.iter().find(|transition| {
            transition.from == cursor.state && transition.when.holds(&player.params)
        }) else {
            continue;
        };
        let (Some(&from_node), Some(&to_node)) =
            (states.get(transition.from), states.get(transition.to))
        else {
            continue;
        };
        if let Some(marks) = marks {
            sync_transition_leaves(&mut player.leaves, tree, from_node, to_node, clips, marks);
        }
        let Some(cursor) = player.machines.get_mut(slot) else {
            break;
        };
        cursor.from = Some(transition.from);
        cursor.state = transition.to;
        cursor.elapsed = Seconds::ZERO;
        cursor.duration = transition.fade;
    }
}

/// Completes or advances in-flight switch crossfades, then starts a new
/// one whenever the live parameter's rounded selection ([`tree::switch_selection`])
/// no longer matches the standing child (an in-flight crossfade is never
/// interrupted — a new selection re-checks once it completes, the
/// [`SkelFade`](crate::SkelFade) mechanics). A started crossfade
/// re-phases the leaving and arriving children's leaves through their
/// [`ClipSyncMarks`] — unless the lane is missing, which leaves every
/// cursor alone.
fn step_switches(
    player: &mut SkelTreePlayer,
    tree: Option<&SkelTree>,
    clips: &ColdComponentStore<SkelClip>,
    marks: Option<&ColdComponentStore<ClipSyncMarks>>,
    dt: f32,
) {
    for cursor in &mut player.switches {
        tick_crossfade(&mut cursor.elapsed, cursor.duration, &mut cursor.from, dt);
    }
    let Some(tree) = tree else {
        return;
    };
    let mut cursors = 0;
    for node in &tree.nodes {
        let tree::TreeNodeDef::Switch {
            param,
            children,
            fade,
        } = node
        else {
            continue;
        };
        let slot = cursors;
        cursors += 1;
        let Some(cursor) = player.switches.get(slot) else {
            break;
        };
        if cursor.from.is_some() {
            continue;
        }
        let Some(&value) = player.params.get(*param) else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        let selection = tree::switch_selection(value, children.len());
        if selection == cursor.child {
            continue;
        }
        let (Some(&from_node), Some(&to_node)) =
            (children.get(cursor.child), children.get(selection))
        else {
            continue;
        };
        if let Some(marks) = marks {
            sync_transition_leaves(&mut player.leaves, tree, from_node, to_node, clips, marks);
        }
        let Some(cursor) = player.switches.get_mut(slot) else {
            break;
        };
        cursor.from = Some(cursor.child);
        cursor.child = selection;
        cursor.elapsed = Seconds::ZERO;
        cursor.duration = *fade;
    }
}

/// Advances one in-flight crossfade clock by `dt`, clearing `from` once
/// the fade completes ([`SkelFade`](crate::SkelFade) mechanics: a
/// non-positive duration completes at once). Shared by the machine and
/// switch steppers.
fn tick_crossfade(elapsed: &mut Seconds, duration: Seconds, from: &mut Option<usize>, dt: f32) {
    if from.is_none() {
        return;
    }
    *elapsed = Seconds::new(elapsed.get() + dt);
    if fade_progress(duration, *elapsed) >= 1.0 {
        *from = None;
    }
}

/// Clock of a crossfade component, shared by [`AnimFade`] and [`SkelFade`].
trait FadeClock: 'static + Send + Sync {
    /// Adds `dt` seconds to the clock; `true` once the fade is over.
    fn tick(&mut self, dt: f32) -> bool;

    /// Linear crossfade progress in `0..=1`
    /// ([`crate::blend::fade_progress`] of the component's clock).
    fn progress(&self) -> f32;
}

impl FadeClock for AnimFade {
    fn tick(&mut self, dt: f32) -> bool {
        self.elapsed = Seconds::new(self.elapsed.get() + dt);
        self.elapsed.get() >= self.duration.get()
    }

    fn progress(&self) -> f32 {
        fade_progress(self.duration, self.elapsed)
    }
}

impl FadeClock for SkelFade {
    fn tick(&mut self, dt: f32) -> bool {
        self.elapsed = Seconds::new(self.elapsed.get() + dt);
        self.elapsed.get() >= self.duration.get()
    }

    fn progress(&self) -> f32 {
        fade_progress(self.duration, self.elapsed)
    }
}

impl FadeClock for SkelInertia {
    fn tick(&mut self, dt: f32) -> bool {
        // A pending transition waits for its capture — its clock starts
        // only with the captured offsets (§8.5).
        if self.pending() {
            return false;
        }
        self.elapsed = Seconds::new(self.elapsed.get() + dt);
        self.elapsed.get() >= self.duration.get()
    }

    fn progress(&self) -> f32 {
        fade_progress(self.duration, self.elapsed)
    }
}

/// The crossfade share of `entity` read straight from `fades` — the
/// [`FadeClock::progress`] of its incoming cursor, `1.0` when no fade
/// flies on it. Direct lane access: no per-frame collection or scan on
/// the hot path (design §8.1).
fn share_of<F: FadeClock>(fades: Option<&ComponentStore<F>>, entity: Entity) -> f32 {
    fades
        .and_then(|fades| fades.get(entity))
        .map_or(1.0, |fade| fade.progress())
}

/// Steps every crossfade by `dt`: `advance_from` moves the outgoing cursor,
/// the clock moves on, and finished fades are removed. Entities in `frozen`
/// keep their fade untouched this frame.
fn tick_fades<F: FadeClock>(
    fades: &mut ComponentStore<F>,
    dt: f32,
    frozen: &FrozenBits,
    finished: &mut Vec<Entity>,
    mut advance_from: impl FnMut(&mut F, f32, Entity),
) {
    finished.clear();
    for index in 0..fades.data.len() {
        let entity = fades.entities[index];
        if frozen.contains(entity) {
            continue;
        }
        let fade = &mut fades.data[index];
        advance_from(fade, dt, entity);
        if fade.tick(dt) {
            finished.push(entity);
        }
    }
    for &entity in finished.iter() {
        fades.remove(entity);
    }
}

/// Advances [`SkelInertia`] falloff clocks and retires finished ones
/// (design §8.5). A pending transition is left untouched — its clock
/// starts with the capture in the sample pass.
fn advance_inertias(store: &SmartStore, dt: f32, scratch: &mut AnimScratch) {
    let Some(mut inertias) = store.write_lane::<SkelInertia>() else {
        return;
    };
    tick_fades(
        &mut inertias,
        dt,
        &FrozenBits::default(),
        &mut scratch.finished,
        |_, _, _| {},
    );
}

/// Advances outgoing crossfade clocks and retires finished fades.
fn advance_fades(store: &SmartStore, dt: f32, scratch: &mut AnimScratch) {
    let Some(clips) = store.read_cold_lane::<SkelClip>() else {
        return;
    };
    let events = store.read_cold_lane::<ClipEvents>();
    let trees = store.read_lane::<SkelTreePlayer>();
    let skeletons = store.read_lane::<Skeleton>();
    let maps = store.read_cold_lane::<RetargetMap>();
    let mut queues = store.write_lane::<AnimEventQueue>();
    let mut motion = store.write_lane::<RootMotionDelta>();
    let Some(mut fades) = store.write_lane::<SkelFade>() else {
        return;
    };
    let src = MotionSrc {
        clips: &clips,
        skeletons: skeletons.as_deref(),
        maps: maps.as_deref(),
        masks: None,
    };
    tick_fades(
        &mut fades,
        dt,
        &FrozenBits::default(),
        &mut scratch.finished,
        |fade, step, entity| {
            let cursor = step_player(&mut fade.from, &clips, step);
            if let Some(cursor) = cursor {
                emit_step(
                    queues.as_deref_mut(),
                    events.as_deref(),
                    entity,
                    fade.from.clip,
                    cursor.scaled(1.0 - fade.progress()),
                );
            }
            // A tree-driven root ignores `SkelFade` in the pose; the delta
            // follows. Shares are the pre-tick progress (design §8.1/§8.4).
            let tree_driven = trees.as_deref().is_some_and(|lane| lane.contains(entity));
            if tree_driven {
                return;
            }
            let Some(slot) = motion.as_deref_mut().and_then(|lane| lane.get_mut(entity)) else {
                return;
            };
            let outgoing = cursor.map_or(RootMotionDelta::IDENTITY, |cursor| {
                src.delta(entity, fade.from.clip, Some(cursor), fade.from.weight.get())
            });
            *slot = outgoing.mix(*slot, fade.progress());
        },
    );
}

/// Applies the stashed layer overlays onto the mixed base delta, bottom-up
/// (§8.4). Roots held for a broken tree are skipped: bad skin publishes
/// no pose, and a layer must not move the host on its own.
fn apply_layer_root_motion(store: &SmartStore, scratch: &AnimScratch) {
    let Some(mut motion) = store.write_lane::<RootMotionDelta>() else {
        return;
    };
    for part in &scratch.layer_motion {
        if scratch.motion_hold.contains(&part.entity) {
            continue;
        }
        let Some(slot) = motion.get_mut(part.entity) else {
            continue;
        };
        *slot = slot.mix(part.delta, part.weight);
    }
}

/// Samples playing cursors and publishes whole [`JointPose`] vectors.
///
/// Bound joint nodes also receive a local [`Transform`] write, then one
/// propagation so their children move in this frame.
fn publish_poses(store: &SmartStore) -> SkelSampleStats {
    with_scratch(|scratch| {
        let mut stats = SkelSampleStats::default();
        collect_pose_work(store, &mut stats, scratch);
        if publish_local_transforms(store, &scratch.nodes) {
            ornis_core::propagate_registered(store);
        }
        stats
    })
}

/// Read-only view of every lane one pose evaluation needs.
struct SampleLanes<'a> {
    clips: &'a ColdComponentStore<SkelClip>,
    skeletons: &'a ComponentStore<Skeleton>,
    transforms: Option<&'a ComponentStore<TransformDesc>>,
    fades: Option<&'a ComponentStore<SkelFade>>,
    layers: Option<&'a ComponentStore<SkelLayerStack>>,
    masks: Option<&'a ColdComponentStore<JointMask>>,
    trees: Option<&'a ColdComponentStore<SkelTree>>,
    tree_players: Option<&'a ComponentStore<SkelTreePlayer>>,
    ik: Option<&'a ComponentStore<IkStack>>,
    retarget: Option<&'a ColdComponentStore<RetargetMap>>,
}

/// Samples every playing cursor into the scratch buffers: model matrices
/// land in each root's [`JointPose`] (reusing its buffer), node writes in
/// `scratch.nodes`. Every imperfect entity bumps `skipped_bad_skin`.
fn collect_pose_work(store: &SmartStore, stats: &mut SkelSampleStats, scratch: &mut AnimScratch) {
    scratch.nodes.clear();
    let Some(clips) = store.read_cold_lane::<SkelClip>() else {
        return;
    };
    let Some(players) = store.read_lane::<SkelPlayer>() else {
        return;
    };
    let Some(skeletons) = store.read_lane::<Skeleton>() else {
        return;
    };
    let transforms = store.read_lane::<TransformDesc>();
    let fades = store.read_lane::<SkelFade>();
    let layers = store.read_lane::<SkelLayerStack>();
    let masks = store.read_cold_lane::<JointMask>();
    let trees = store.read_cold_lane::<SkelTree>();
    let tree_players = store.read_lane::<SkelTreePlayer>();
    let ik = store.read_lane::<IkStack>();
    let retarget = store.read_cold_lane::<RetargetMap>();
    // Blend lane: per-entity writes (drive targets of §4.2); a second lane guard never nests on the same lock.
    let mut ragdolls = store.write_lane::<RagdollBlend>();
    let mut inertias = store.write_lane::<SkelInertia>();
    let mut poses = store.write_lane::<JointPose>();
    let lanes = SampleLanes {
        clips: &clips,
        skeletons: &skeletons,
        transforms: transforms.as_deref(),
        fades: fades.as_deref(),
        layers: layers.as_deref(),
        masks: masks.as_deref(),
        trees: trees.as_deref(),
        tree_players: tree_players.as_deref(),
        ik: ik.as_deref(),
        retarget: retarget.as_deref(),
    };
    for (entity, player) in players.entities.iter().zip(players.data.iter()) {
        if !player.playing {
            continue;
        }
        if lanes
            .tree_players
            .is_some_and(|lane| lane.contains(*entity))
        {
            continue;
        }
        let blend = ragdolls
            .as_deref_mut()
            .and_then(|lane| lane.get_mut(*entity));
        let inertia = inertias
            .as_deref_mut()
            .and_then(|lane| lane.get_mut(*entity));
        match sample_entity(
            *entity,
            player,
            &lanes,
            poses.as_deref_mut(),
            scratch,
            blend,
            inertia,
        ) {
            Some(()) => stats.sampled += 1,
            None => stats.skipped_bad_skin += 1,
        }
    }
    // Tree-driven roots evaluate their tree, then run the same layer and
    // node-write tail as player-driven ones.
    if let Some(tree_players) = lanes.tree_players {
        let (sampled, skipped) = sample_tree_roots(
            tree_players,
            &lanes,
            poses.as_deref_mut(),
            scratch,
            ragdolls.as_deref_mut(),
        );
        stats.sampled += sampled;
        stats.skipped_bad_skin += skipped;
    }
}

/// Tree-driven roots of one sample pass: evaluates each tree, then runs
/// the same layer and node-write tail as player-driven ones. Returns the
/// `(sampled, skipped_bad_skin)` counts for the caller's stats — the
/// tree roots take no [`SkelInertia`](crate::SkelInertia) (§8.5).
fn sample_tree_roots(
    tree_players: &ComponentStore<SkelTreePlayer>,
    lanes: &SampleLanes<'_>,
    mut poses: Option<&mut ComponentStore<JointPose>>,
    scratch: &mut AnimScratch,
    mut ragdolls: Option<&mut ComponentStore<RagdollBlend>>,
) -> (u32, u32) {
    let mut sampled = 0;
    let mut skipped = 0;
    for (entity, player) in tree_players.entities.iter().zip(tree_players.data.iter()) {
        let blend = ragdolls
            .as_deref_mut()
            .and_then(|lane| lane.get_mut(*entity));
        match sample_tree_entity(*entity, player, lanes, poses.as_deref_mut(), scratch, blend) {
            Some(()) => sampled += 1,
            None => skipped += 1,
        }
    }
    (sampled, skipped)
}

/// Skeleton and joint count of `entity`, strict: the rest table must match
/// the joint count. Shared start of [`sample_entity`] and
/// [`sample_tree_entity`].
fn strict_skeleton<'a>(
    lanes: &'a SampleLanes<'_>,
    entity: Entity,
) -> Option<(&'a Skeleton, usize)> {
    let skeleton = lanes.skeletons.get(entity)?;
    let count = skeleton.joint_count();
    if skeleton.rest.len() != count {
        return None;
    }
    Some((skeleton, count))
}

/// Fits the per-node pose scratch to `nodes` without dropping live
/// [`BlendedPose`] elements: the steady state (unchanged node count) is a
/// no-op, so every `refill`/`refill_rest` reuses the inner joint buffers
/// and the sample pass stays allocation-free. `resize_with` truncates only
/// when a tree shrinks.
/// Grows `poses` to `nodes`, keeping every already-allocated [`BlendedPose`]
/// buffer: the scratch is shared across roots, so a smaller tree must not
/// truncate and drop buffers a larger one just filled. Callers evaluate
/// into `&mut poses[..nodes]` and take the root from `poses[nodes - 1]`.
fn fit_tree_poses(poses: &mut Vec<BlendedPose>, nodes: usize) {
    if poses.len() < nodes {
        poses.resize_with(nodes, BlendedPose::default);
    }
}

/// Applies the root's ragdoll blend over the finished animated pose
/// (`docs/ragdoll-design.md` §2.3: after the layer overlays, before the
/// model-matrix compose). A non-finite physical frame or a length
/// mismatch skips the blend for the frame — the animated pose wins (the
/// fill counts it in [`RagdollBlend::skipped_frames`]). Before any of
/// that the animated local rotations land in
/// [`RagdollBlend::anim_rotations`] — the drive targets of §4.2, kept
/// fresh even when the frame below is skipped. Leaf.
fn apply_ragdoll(blend: Option<&mut RagdollBlend>, pose: &mut BlendedPose) {
    let Some(blend) = blend else {
        return;
    };
    let count = pose.trs.len();
    blend.anim_rotations.resize(count, Quat::IDENTITY);
    for (recorded, trs) in blend.anim_rotations.iter_mut().zip(pose.trs.iter()) {
        *recorded = trs.rotation;
    }
    if blend.physical.len() != count || blend.weights.len() != count {
        return;
    }
    if !blend
        .physical
        .iter()
        .all(|trs| trs.translation.is_finite() && trs.rotation.is_finite() && trs.scale.is_finite())
    {
        return;
    }
    ragdoll_blend(&mut pose.trs, &blend.physical, &blend.weights);
}

/// Composes the entity's model matrices from the blended pose and routes
/// node writes, committing the pose. Shared tail of [`sample_entity`] and
/// [`sample_tree_entity`].
fn finish_pose(
    entity: Entity,
    lanes: &SampleLanes<'_>,
    skeleton: &Skeleton,
    scratch: &mut AnimScratch,
    pose: &mut JointPose,
    blend: Option<&mut RagdollBlend>,
) -> Option<()> {
    // Design §8.2: IK corrects the animated pose after the tree/layer
    // stack and before the ragdoll blend — where physics weights are up,
    // physics wins.
    if let Some(stack) = lanes.ik.and_then(|lane| lane.get(entity)) {
        apply_ik_stack(
            &mut scratch.pose_a.trs,
            &skeleton.parents,
            stack,
            &mut scratch.ik_fk,
        );
    }
    apply_ragdoll(blend, &mut scratch.pose_a);
    scratch.skel_locals.clear();
    scratch
        .skel_locals
        .extend(scratch.pose_a.trs.iter().map(|trs| trs.matrix()));
    let root = root_matrix(lanes.transforms, entity);
    compose_model_matrices(skeleton, &scratch.skel_locals, &root, &mut scratch.matrices)?;
    // §8.3: a retargeted root drives its `JointPose` only — the hierarchy
    // stays still (no node re-binds through the map), trees included.
    if !lanes
        .retarget
        .is_some_and(|lane| lane.get(entity).is_some())
    {
        node_poses(
            lanes.clips,
            &scratch.contributors,
            &scratch.pose_a,
            &mut scratch.nodes,
            &mut scratch.node_painted,
        );
    }
    pose.matrices.clear();
    pose.matrices.extend_from_slice(&scratch.matrices);
    Some(())
}

/// Samples one playing cursor: base clip, optional crossfade, the overlay
/// layer stack (bottom-up), then model matrices into `poses` and node
/// writes into `scratch`.
///
/// [`None`] (bad skin) on missing clip/skeleton/pose lane, invalid
/// topology, or a layer whose mask or clip is missing or whose mask does
/// not match the joint count. The caller counts it. A missing outgoing
/// clip reads as rest. Absent [`TransformDesc`] roots at identity
/// (lenient); everything else is strict.
fn sample_entity(
    entity: Entity,
    player: &SkelPlayer,
    lanes: &SampleLanes<'_>,
    poses: Option<&mut ComponentStore<JointPose>>,
    scratch: &mut AnimScratch,
    blend: Option<&mut RagdollBlend>,
    inertia: Option<&mut SkelInertia>,
) -> Option<()> {
    let clip = lanes.clips.get(player.clip.0)?;
    let (skeleton, _) = strict_skeleton(lanes, entity)?;
    // Absent `JointPose` is bad skin: verdict before any blending work.
    let pose = poses?.get_mut(entity)?;
    blend_skel_pose(
        entity,
        player,
        lanes,
        clip,
        &skeleton.rest,
        scratch,
        inertia,
    )?;
    finish_pose(entity, lanes, skeleton, scratch, pose, blend)
}

/// Blends one root's base clip, optional crossfade and the overlay layer
/// stack into `scratch.pose_a`, recording the clips that shaped the pose
/// (base, outgoing fade, each layer) in `scratch.contributors`.
///
/// [`None`] when a layer's mask does not match the joint count or its
/// mask/clip is missing — the same bad-skin verdict as a missing clip. A
/// vanished outgoing clip reads as rest, as on object tracks. Layers blend
/// bottom-up ([`SkelLayerStack`]): each overlays the pose below it.
///
/// `contributors` carries every clip that shapes the pose, so a joint
/// driven only by a layer or by the outgoing side of a fade still moves
/// its bound node. It is the scratch id buffer, reused across entities.
fn blend_skel_pose<'a>(
    entity: Entity,
    player: &SkelPlayer,
    lanes: &SampleLanes<'a>,
    clip: &'a SkelClip,
    rest: &[JointTrs],
    scratch: &mut AnimScratch,
    inertia: Option<&mut SkelInertia>,
) -> Option<()> {
    // `rest.len()` is the joint count: `strict_skeleton` guarantees the
    // rest table matches it before any caller reaches this far.
    let count = rest.len();
    scratch.contributors.clear();
    scratch.contributors.push(player.clip);
    let map = lanes.retarget.and_then(|lane| lane.get(entity));
    sample_joints(clip, count, player.time.get(), &mut scratch.joints, map);
    scratch
        .pose_a
        .refill(&scratch.joints.samples, rest, player.weight.get());
    let mut inertia = inertia;
    // §8.5: capture a pending inertialized switch against this base
    // before the crossfade/layer mix. The outgoing snapshot is frozen —
    // its clock never moves; a vanished clip reads as rest, as on object
    // tracks.
    if let Some(inertia) = &mut inertia
        && inertia.pending()
    {
        match lanes.clips.get(inertia.from.clip.0) {
            Some(from_clip) => {
                sample_joints(
                    from_clip,
                    count,
                    inertia.from.time.get(),
                    &mut scratch.joints,
                    map,
                );
                scratch
                    .pose_b
                    .refill(&scratch.joints.samples, rest, inertia.from.weight.get());
            }
            None => scratch.pose_b.refill_rest(rest),
        }
        inertia.capture(&scratch.pose_b.trs, &scratch.pose_a.trs);
    }
    if let Some(fade) = lanes.fades.and_then(|lane| lane.get(entity)) {
        match lanes.clips.get(fade.from.clip.0) {
            Some(from_clip) => {
                scratch.contributors.push(fade.from.clip);
                sample_joints(
                    from_clip,
                    count,
                    fade.from.time.get(),
                    &mut scratch.joints,
                    map,
                );
                scratch
                    .pose_b
                    .refill(&scratch.joints.samples, rest, fade.from.weight.get());
            }
            // A vanished outgoing clip reads as rest, as on object tracks.
            None => scratch.pose_b.refill_rest(rest),
        }
        scratch
            .pose_b
            .mix_towards(&scratch.pose_a, fade_progress(fade.duration, fade.elapsed));
        std::mem::swap(&mut scratch.pose_a, &mut scratch.pose_b);
    }
    // §8.5: the inertialized base keeps its decaying captured offsets;
    // the layers overlay it afterwards, live over the transition.
    if let Some(inertia) = &inertia {
        inertia.apply(&mut scratch.pose_a.trs);
    }
    overlay_layers(entity, lanes, count, rest, scratch)
}

/// Overlays the entity's [`SkelLayerStack`] bottom-up onto the base pose
/// already blended in `scratch.pose_a`, recording each layer's clip in
/// `scratch.contributors`. Shared by player-driven and tree-driven roots.
///
/// [`None`] when a layer's mask does not match the joint count or its
/// mask/clip is missing — the same bad-skin verdict as a missing clip.
fn overlay_layers(
    entity: Entity,
    lanes: &SampleLanes<'_>,
    count: usize,
    rest: &[JointTrs],
    scratch: &mut AnimScratch,
) -> Option<()> {
    let map = lanes.retarget.and_then(|lane| lane.get(entity));
    if let Some(stack) = lanes.layers.and_then(|lane| lane.get(entity)) {
        for layer in &stack.layers {
            let mask = lanes.masks?.get(layer.mask.0)?;
            if mask.weights.len() != count {
                return None;
            }
            let layer_clip = lanes.clips.get(layer.player.clip.0)?;
            scratch.contributors.push(layer.player.clip);
            sample_joints(
                layer_clip,
                count,
                layer.player.time.get(),
                &mut scratch.joints,
                map,
            );
            scratch.pose_a.overlay(
                &scratch.joints.samples,
                rest,
                &mask.weights,
                layer.player.weight,
            );
        }
    }
    Some(())
}

/// Samples one tree-driven root: evaluates its [`SkelTree`] into
/// `scratch.tree_poses`, takes the root (last) pose as the base, overlays
/// the layer stack and finishes with the same model matrices and node
/// writes as [`sample_entity`]. A `SkelPlayer` on the same entity is
/// ignored — the tree drives the root.
///
/// [`None`] (bad skin) on the same strict conditions as [`sample_entity`],
/// plus any tree-contract violation (see [`eval_tree`]).
fn sample_tree_entity(
    entity: Entity,
    player: &SkelTreePlayer,
    lanes: &SampleLanes<'_>,
    poses: Option<&mut ComponentStore<JointPose>>,
    scratch: &mut AnimScratch,
    blend: Option<&mut RagdollBlend>,
) -> Option<()> {
    let tree = lanes.trees?.get(entity)?;
    let (skeleton, count) = strict_skeleton(lanes, entity)?;
    // Absent `JointPose` is bad skin: verdict before any blending work.
    let pose = poses?.get_mut(entity)?;
    scratch.contributors.clear();
    let nodes = tree.nodes.len();
    fit_tree_poses(&mut scratch.tree_poses, nodes);
    let map = lanes.retarget.and_then(|lane| lane.get(entity));
    eval_tree(
        tree,
        player,
        lanes.clips,
        count,
        &skeleton.rest,
        tree_eval::EvalBuf {
            poses: &mut scratch.tree_poses[..nodes],
            joints: &mut scratch.joints,
            contributors: &mut scratch.contributors,
            map,
        },
    )?;
    scratch
        .pose_a
        .clone_from(scratch.tree_poses.get(nodes.checked_sub(1)?)?);
    overlay_layers(entity, lanes, count, &skeleton.rest, scratch)?;
    finish_pose(entity, lanes, skeleton, scratch, pose, blend)
}

/// Local channels for joint tracks bound to a hierarchy node.
///
/// `contributors` lists every clip that shapes `pose` (base, outgoing
/// fade, layers) as playlist ids; a vanished one reads as nothing. First
/// track per joint wins across them, matching [`sample_joints`]. Channels
/// no contributor provided stay [`None`] so the node's bind pose survives;
/// the blended values of the others come from the final pose. `out`
/// accumulates across entities (deduplicated per call) and `painted` is
/// reusable scratch.
fn node_poses(
    clips: &ColdComponentStore<SkelClip>,
    contributors: &[ClipId],
    pose: &BlendedPose,
    out: &mut Vec<LocalWrite>,
    painted: &mut Vec<bool>,
) {
    let start = out.len();
    painted.clear();
    painted.resize(pose.trs.len(), false);
    let tracks = contributors
        .iter()
        .filter_map(|id| clips.get(id.0))
        .flat_map(|clip| clip.tracks.iter());
    for track in tracks {
        let joint = track.joint.index();
        if joint >= painted.len() || painted[joint] {
            continue;
        }
        painted[joint] = true;
        let Some(entity) = track.node else {
            continue;
        };
        if out[start..].iter().any(|write| write.entity == entity) {
            continue;
        }
        out.push(LocalWrite {
            entity,
            pose: sampled_pose(pose, joint),
        });
    }
}

/// Final channels of one joint, with absent ones as [`None`].
fn sampled_pose(pose: &BlendedPose, joint: usize) -> SampledPose {
    let trs = pose.trs[joint];
    let [translation, rotation, scale] = pose.present[joint];
    SampledPose {
        translation: translation.then_some(trs.translation),
        rotation: rotation.then_some(trs.rotation),
        scale: scale.then_some(trs.scale),
    }
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
    /// Authored rest transform per joint (the node transform in the source
    /// skin); length equals the joint count. Use [`JointTrs::IDENTITY`] when
    /// the source carries none.
    pub rest: Vec<JointTrs>,
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
/// # use ornis_animation::{JointTrs, SkinImport, skeleton_from_import};
/// let skeleton = skeleton_from_import(&SkinImport {
///     parents: vec![-1, 0],
///     inverse_bind: vec![Mat4::IDENTITY; 2],
///     joint_names: vec!["root".to_string(), "tip".to_string()],
///     rest: vec![JointTrs::IDENTITY; 2],
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
    )
    .with_rest(import.rest.clone());
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
            rest: vec![JointTrs::IDENTITY; 2],
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
    fn joint_named_resolves_exact_names() {
        let skeleton = skeleton_from_import(&two_bone_import()).expect("two bones build");
        assert_eq!(skeleton.joint_named("root"), Some(JointId::from_raw(0)));
        assert_eq!(skeleton.joint_named("tip"), Some(JointId::from_raw(1)));
        assert_eq!(skeleton.joint_named("Tip"), None);
        assert_eq!(skeleton.joint_named("missing"), None);
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
            rest: vec![JointTrs::IDENTITY; 2],
        };
        assert_eq!(skeleton_from_import(&cyclic), Err(SkelError::Cycle));
        // Empty.
        let empty = SkinImport {
            parents: Vec::new(),
            inverse_bind: Vec::new(),
            joint_names: Vec::new(),
            rest: Vec::new(),
        };
        assert_eq!(skeleton_from_import(&empty), Err(SkelError::Empty));
    }

    #[test]
    fn skeleton_build_carries_authored_rest() {
        let mut import = two_bone_import();
        let tilt = Quat::from_rotation_z(0.5);
        import.rest[1].rotation = tilt;
        let skeleton = skeleton_from_import(&import).expect("rest builds");
        assert_eq!(skeleton.rest[0], JointTrs::IDENTITY);
        assert_eq!(skeleton.rest[1].rotation, tilt);
        // A rest list shorter than the joints is a defect, not a default.
        import.rest.pop();
        assert_eq!(
            skeleton_from_import(&import),
            Err(SkelError::LengthMismatch)
        );
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

    #[test]
    fn tree_pose_scratch_keeps_buffers_across_frames() {
        let mut poses: Vec<BlendedPose> = Vec::new();
        fit_tree_poses(&mut poses, 5);
        for pose in &mut poses {
            pose.refill_rest(&[JointTrs::IDENTITY; 3]);
        }
        let before: Vec<usize> = poses.iter().map(|pose| pose.trs.capacity()).collect();
        fit_tree_poses(&mut poses, 3);
        assert_eq!(
            poses.len(),
            5,
            "a smaller root keeps the allocation — the tail buffers stay for the next larger tree"
        );
        let after_shrink: Vec<usize> = poses.iter().map(|pose| pose.trs.capacity()).collect();
        assert_eq!(before, after_shrink, "shrink must not drop buffers");
        fit_tree_poses(&mut poses, 5);
        let after_regrow: Vec<usize> = poses.iter().map(|pose| pose.trs.capacity()).collect();
        assert_eq!(
            before, after_regrow,
            "an N -> M < N -> N cycle must not reallocate the pose buffers"
        );
    }
}
