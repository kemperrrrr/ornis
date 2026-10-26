//! Motion matching, distance matching and warp weights
//! (`docs/animation-design.md` §8.5).
//!
//! A query, not a sampler stage: the host asks which frame of which clip
//! is closest to a feature pose, which clip time has covered a distance,
//! and how strongly a two-bone IK constraint should pull inside a warp
//! window. Nothing here runs per frame inside `skel_sample`, and nothing
//! here allocates per candidate — the search reuses one pose scratch.

use glam::{Quat, Vec3};
use ornis_core::ColdComponentStore;

use crate::blend::{SampleBuf, sample_joints};
use crate::events::CursorStep;
use crate::root_motion::{root_joint, root_motion};
use crate::{ClipId, JointId, JointTrs, SkelClip, Skeleton};

/// Steps at or below this read as no velocity window.
const MIN_STEP: f32 = 1.0e-4;

/// Segment lengths at or below this do not subdivide a distance match.
const MIN_SEGMENT: f32 = 1.0e-6;

/// One authored frame the matcher may choose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionSample {
    /// Playlist clip to sample.
    pub clip: ClipId,
    /// Time in seconds from the clip start.
    pub time: f32,
    /// Whether the velocity window wraps at the clip ends (a locomotion
    /// loop). A one-shot sample stays clamped.
    pub looping: bool,
}

/// The pose the host wants to match: model-space positions of the feature
/// joints (same order as the joint list passed to [`match_motion`]) and
/// the root's linear velocity in model space, units per second.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionQuery {
    /// One model-space position per feature joint.
    pub positions: Vec<Vec3>,
    /// Root linear velocity, model space.
    pub velocity: Vec3,
}

/// The winning sample of a [`match_motion`] search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionHit {
    /// Clip of the winning sample.
    pub clip: ClipId,
    /// Time of the winning sample, seconds.
    pub time: f32,
    /// Squared feature cost (positions plus weighted velocity). Lower
    /// is closer. Ties keep the earlier sample.
    pub cost: f32,
}

/// Inputs of one [`match_motion`] search, grouped so the query stays one
/// argument.
pub struct MotionSearch<'a> {
    /// Cold clip lane the samples name.
    pub clips: &'a ColdComponentStore<SkelClip>,
    /// Skeleton the feature joints and the root velocity are read on.
    pub skeleton: &'a Skeleton,
    /// Candidate frames, in the order ties fall back to (earlier wins).
    pub samples: &'a [MotionSample],
    /// Feature joints, model space. Same length as
    /// [`MotionQuery::positions`].
    pub joints: &'a [JointId],
    /// Pose to match.
    pub query: &'a MotionQuery,
    /// Seconds of root motion the velocity is measured over. A non-finite
    /// or tiny step contributes no velocity.
    pub step: f32,
    /// Scale on the squared velocity error. A non-finite weight
    /// contributes nothing.
    pub velocity_weight: f32,
}

/// The warp window of one contact, in seconds of clip time.
///
/// Weight rises from `0` at `start` to `1` at `contact` and falls back
/// to `0` at `end` ([`warp_weight`]). The host multiplies a
/// [`crate::TwoBoneIk`] weight by it — warping is that IK pull, not a
/// second solver (§8.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarpWindow {
    /// Clip time where the pull starts.
    pub start: f32,
    /// Clip time of the contact (full pull).
    pub contact: f32,
    /// Clip time where the pull releases.
    pub end: f32,
}

/// Pose buffers of one search. Reused across candidates.
struct PoseScratch {
    samples: SampleBuf,
    locals: Vec<JointTrs>,
    model_pos: Vec<Vec3>,
    model_rot: Vec<Quat>,
}

/// Nearest candidate frame to `search.query`.
///
/// Cost is the sum of squared model-space position errors of the feature
/// joints plus `velocity_weight` times the squared root-velocity error.
/// The root velocity is the clip's root motion over `step` seconds
/// (design §8.4), divided by `step`. A missing clip is skipped. Equal
/// finite costs keep the earlier sample. [`None`] when nothing qualifies:
/// the feature list and the query disagree in length, a feature joint is
/// outside the skeleton, or every sample is missing or non-finite.
#[must_use]
pub fn match_motion(search: &MotionSearch<'_>) -> Option<MotionHit> {
    if search.query.positions.len() != search.joints.len() {
        return None;
    }
    let count = search.skeleton.joint_count();
    if search.skeleton.rest.len() != count {
        return None;
    }
    if search.joints.iter().any(|joint| joint.index() >= count) {
        return None;
    }
    let mut scratch = PoseScratch {
        samples: SampleBuf::default(),
        locals: Vec::new(),
        model_pos: Vec::new(),
        model_rot: Vec::new(),
    };
    let mut best: Option<MotionHit> = None;
    for sample in search.samples {
        let Some(cost) = sample_cost(search, sample, &mut scratch) else {
            continue;
        };
        let take = best.as_ref().is_none_or(|hit| cost < hit.cost);
        if take {
            best = Some(MotionHit {
                clip: sample.clip,
                time: sample.time,
                cost,
            });
        }
    }
    best
}

/// Clip time at which the root joint's translation has travelled
/// `distance` (model units), measured as the chord length of the
/// translation channel at its keys.
///
/// Exact for linear tracks (the chord is the motion). Step and cubic
/// tracks use the same key polyline, not the curve between keys.
/// `distance <= 0` is the first key; past the end is the last key.
/// [`None`] for a non-finite or negative distance, or a clip with no
/// translation keys on `root`. The distance is in the clip's authored
/// space — a retargeted root scales it by the map's rest-length ratio
/// before the call (§8.3).
#[must_use]
pub fn time_at_distance(clip: &SkelClip, root: JointId, distance: f32) -> Option<f32> {
    if !distance.is_finite() || distance < 0.0 {
        return None;
    }
    let track = clip.tracks.iter().find(|track| track.joint == root)?;
    let keys = &track.translation.keys;
    let first = keys.first()?;
    if distance == 0.0 || keys.len() == 1 {
        return Some(first.time);
    }
    let mut traveled = 0.0;
    let mut prev_time = first.time;
    let mut prev = track.translation.sample(prev_time)?;
    for key in keys.iter().skip(1) {
        let next = track.translation.sample(key.time)?;
        let segment = (next - prev).length();
        if traveled + segment >= distance && segment > MIN_SEGMENT {
            let along = (distance - traveled) / segment;
            return Some(prev_time + (key.time - prev_time) * along);
        }
        traveled += segment;
        prev_time = key.time;
        prev = next;
    }
    keys.last().map(|key| key.time)
}

/// IK pull strength in `0.0..=1.0` at clip `time` inside `window`.
///
/// `0` outside the window and on a non-finite or unordered window
/// (`start <= contact <= end` is required, and `end` must be strictly
/// after `start`). The host writes `TwoBoneIk::weight` from this value
/// (§8.5): the two-bone solver is the warp.
#[must_use]
pub fn warp_weight(time: f32, window: WarpWindow) -> f32 {
    let WarpWindow {
        start,
        contact,
        end,
    } = window;
    if !time.is_finite() || !start.is_finite() || !contact.is_finite() || !end.is_finite() {
        return 0.0;
    }
    if start > contact || contact > end || end <= start {
        return 0.0;
    }
    if time <= start || time >= end {
        return 0.0;
    }
    if time <= contact {
        let span = contact - start;
        return if span > MIN_SEGMENT {
            (time - start) / span
        } else {
            1.0
        };
    }
    let span = end - contact;
    if span > MIN_SEGMENT {
        (end - time) / span
    } else {
        1.0
    }
}

/// Feature cost of one sample, or [`None`] when it cannot be read.
fn sample_cost(
    search: &MotionSearch<'_>,
    sample: &MotionSample,
    scratch: &mut PoseScratch,
) -> Option<f32> {
    if !sample.time.is_finite() {
        return None;
    }
    let clip = search.clips.get(sample.clip.0)?;
    pose_at(clip, search.skeleton, sample.time, scratch)?;
    let mut cost = 0.0;
    for (joint, query) in search.joints.iter().zip(&search.query.positions) {
        let delta = scratch.model_pos[joint.index()] - *query;
        cost += delta.length_squared();
    }
    let velocity = root_velocity(clip, search.skeleton, sample, search.step);
    let weight = if search.velocity_weight.is_finite() {
        search.velocity_weight
    } else {
        0.0
    };
    let vel = velocity - search.query.velocity;
    let total = cost + weight * vel.length_squared();
    // A non-finite cost (NaN/inf query or clip channels) never qualifies:
    // the first such candidate would stick as `best` — `cost < NaN` never
    // displaces it (§8.5, the `match_motion` contract).
    total.is_finite().then_some(total)
}

/// Root linear velocity of `sample` over `step` seconds, from the clip's
/// root motion (§8.4). A tiny step is no velocity.
fn root_velocity(clip: &SkelClip, skeleton: &Skeleton, sample: &MotionSample, step: f32) -> Vec3 {
    if !step.is_finite() || step <= MIN_STEP {
        return Vec3::ZERO;
    }
    let cursor = CursorStep {
        prev: sample.time,
        step,
        duration: clip.duration,
        looping: sample.looping,
        weight: 1.0,
    };
    let delta = root_motion(Some(clip), Some(cursor), root_joint(skeleton), None, 1.0);
    delta.translation / step
}

/// Samples `clip` at `time` into model-space positions in `scratch`.
/// [`None`] when the skeleton has no joints.
fn pose_at(
    clip: &SkelClip,
    skeleton: &Skeleton,
    time: f32,
    scratch: &mut PoseScratch,
) -> Option<()> {
    let count = skeleton.joint_count();
    if count == 0 {
        return None;
    }
    sample_joints(clip, count, time, &mut scratch.samples, None);
    scratch.locals.clear();
    scratch.locals.extend(
        scratch
            .samples
            .samples
            .iter()
            .zip(skeleton.rest.iter())
            .map(|(sample, rest)| JointTrs::from_sample(*sample, *rest)),
    );
    forward_kinematics(
        &scratch.locals,
        &skeleton.parents,
        &mut scratch.model_pos,
        &mut scratch.model_rot,
    );
    Some(())
}

/// Model-space positions and rotations, parents before children. The same
/// walk as the two-bone solver, writing into the caller buffers.
fn forward_kinematics(
    locals: &[JointTrs],
    parents: &[Option<JointId>],
    model_pos: &mut Vec<Vec3>,
    model_rot: &mut Vec<Quat>,
) {
    reset_model_buffers(locals.len(), model_pos, model_rot);
    for (index, local) in locals.iter().enumerate() {
        let (position, rotation) = match parents.get(index).copied().flatten() {
            Some(parent) if parent.index() < index => {
                let parent = parent.index();
                (
                    model_pos[parent] + model_rot[parent] * local.translation,
                    model_rot[parent] * local.rotation,
                )
            }
            _ => (local.translation, local.rotation),
        };
        model_pos[index] = position;
        model_rot[index] = rotation;
    }
}

/// Clears and resizes the model-space buffers to `count` joints.
fn reset_model_buffers(count: usize, model_pos: &mut Vec<Vec3>, model_rot: &mut Vec<Quat>) {
    model_pos.clear();
    model_rot.clear();
    model_pos.resize(count, Vec3::ZERO);
    model_rot.resize(count, Quat::IDENTITY);
}

#[cfg(test)]
mod tests {
    use glam::Mat4;
    use ornis_core::{Entity, SmartStore};

    use super::*;
    use crate::{JointTrack, Key, KeyTrack};

    fn clip_at(x0: f32, x1: f32) -> SkelClip {
        SkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: vec![JointTrack {
                joint: JointId::from_raw(0),
                node: None,
                translation: KeyTrack::linear(vec![
                    Key {
                        time: 0.0,
                        value: Vec3::new(x0, 0.0, 0.0),
                    },
                    Key {
                        time: 1.0,
                        value: Vec3::new(x1, 0.0, 0.0),
                    },
                ]),
                rotation: KeyTrack::linear(vec![]),
                scale: KeyTrack::linear(vec![]),
            }],
        }
    }

    fn skeleton() -> Skeleton {
        Skeleton::new(vec![None], vec![Mat4::IDENTITY], vec![String::new()])
    }

    fn lane(clips: Vec<SkelClip>) -> (SmartStore, Vec<Entity>) {
        let mut store = SmartStore::new();
        let ids = clips
            .into_iter()
            .map(|clip| {
                let entity = store.create_entity();
                store.insert_cold(entity, clip);
                entity
            })
            .collect();
        (store, ids)
    }

    /// The shared [`match_motion`] stage of the matcher tests (rustqual
    /// DRY-001 keeps the fixture in one place): a lane of two clips, one
    /// feature joint and the candidate list built from `(clip, time)`
    /// pairs. Answers the hit with the lane ids.
    fn search_two(
        clips: [(f32, f32); 2],
        candidates: [(usize, f32); 2],
        positions: Vec<Vec3>,
        velocity: Vec3,
        velocity_weight: f32,
    ) -> (Vec<Entity>, Option<MotionHit>) {
        let (store, ids) = lane(vec![
            clip_at(clips[0].0, clips[0].1),
            clip_at(clips[1].0, clips[1].1),
        ]);
        let clips = store.read_cold_lane::<SkelClip>().expect("clips");
        let skeleton = skeleton();
        let joints = [JointId::from_raw(0)];
        let samples: Vec<MotionSample> = candidates
            .into_iter()
            .map(|(clip, time)| MotionSample {
                clip: ClipId(ids[clip]),
                time,
                looping: false,
            })
            .collect();
        let query = MotionQuery {
            positions,
            velocity,
        };
        let hit = match_motion(&MotionSearch {
            clips: &clips,
            skeleton: &skeleton,
            samples: &samples,
            joints: &joints,
            query: &query,
            step: 0.0,
            velocity_weight,
        });
        (ids, hit)
    }

    #[test]
    fn the_closer_pose_wins_and_ties_keep_the_earlier_sample() {
        let (ids, hit) = search_two(
            [(0.0, 2.0), (10.0, 10.0)],
            [(0, 0.0), (1, 0.0)],
            vec![Vec3::new(10.0, 0.0, 0.0)],
            Vec3::ZERO,
            1.0,
        );
        let hit = hit.expect("a candidate");
        assert_eq!(hit.clip, ClipId(ids[1]));
        assert!(hit.cost < 1e-6, "cost {}", hit.cost);

        let (_, tied) = search_two(
            [(0.0, 2.0), (10.0, 10.0)],
            [(1, 0.0), (1, 0.5)],
            vec![Vec3::new(10.0, 0.0, 0.0)],
            Vec3::ZERO,
            0.0,
        );
        assert_eq!(tied.expect("tie").time, 0.0);
    }

    #[test]
    fn velocity_breaks_a_pose_tie() {
        let (store, ids) = lane(vec![clip_at(0.0, 0.0), clip_at(0.0, 2.0)]);
        let clips = store.read_cold_lane::<SkelClip>().expect("clips");
        let skeleton = skeleton();
        let samples = [
            MotionSample {
                clip: ClipId(ids[0]),
                time: 0.0,
                looping: false,
            },
            MotionSample {
                clip: ClipId(ids[1]),
                time: 0.0,
                looping: false,
            },
        ];
        let query = MotionQuery {
            positions: vec![Vec3::ZERO],
            velocity: Vec3::X * 2.0,
        };
        let hit = match_motion(&MotionSearch {
            clips: &clips,
            skeleton: &skeleton,
            samples: &samples,
            joints: &[JointId::from_raw(0)],
            query: &query,
            step: 1.0,
            velocity_weight: 1.0,
        })
        .expect("hit");
        assert_eq!(hit.clip, ClipId(ids[1]));
    }

    #[test]
    fn a_mismatched_feature_list_matches_nothing() {
        let (store, ids) = lane(vec![clip_at(0.0, 1.0)]);
        let clips = store.read_cold_lane::<SkelClip>().expect("clips");
        let query = MotionQuery {
            positions: vec![Vec3::ZERO, Vec3::ZERO],
            velocity: Vec3::ZERO,
        };
        let hit = match_motion(&MotionSearch {
            clips: &clips,
            skeleton: &skeleton(),
            samples: &[MotionSample {
                clip: ClipId(ids[0]),
                time: 0.0,
                looping: false,
            }],
            joints: &[JointId::from_raw(0)],
            query: &query,
            step: 0.1,
            velocity_weight: 0.0,
        });
        assert!(hit.is_none());
    }

    #[test]
    fn distance_walks_the_root_translation() {
        let clip = clip_at(0.0, 4.0);
        let root = JointId::from_raw(0);
        assert_eq!(time_at_distance(&clip, root, 0.0), Some(0.0));
        let mid = time_at_distance(&clip, root, 2.0).expect("mid");
        assert!((mid - 0.5).abs() < 1e-5, "time {mid}");
        assert_eq!(time_at_distance(&clip, root, 100.0), Some(1.0));
        assert!(time_at_distance(&clip, root, f32::NAN).is_none());
        let empty = SkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: vec![],
        };
        assert!(time_at_distance(&empty, root, 1.0).is_none());
    }

    #[test]
    fn warp_weight_peaks_at_the_contact() {
        let window = WarpWindow {
            start: 0.0,
            contact: 0.4,
            end: 1.0,
        };
        assert_eq!(warp_weight(-1.0, window), 0.0);
        assert_eq!(warp_weight(0.0, window), 0.0);
        assert!((warp_weight(0.2, window) - 0.5).abs() < 1e-5);
        assert!((warp_weight(0.4, window) - 1.0).abs() < 1e-5);
        assert!((warp_weight(0.7, window) - 0.5).abs() < 1e-5);
        assert_eq!(warp_weight(1.0, window), 0.0);
        let broken = WarpWindow {
            start: 0.5,
            contact: 0.0,
            end: 1.0,
        };
        assert_eq!(warp_weight(0.4, broken), 0.0);
    }

    /// A non-finite cost never qualifies (§8.5): a NaN-pose sample does
    /// not stick as `best` (`cost < NaN` never displaces it), and an
    /// all-NaN query answers `None` — the [`match_motion`] contract.
    #[test]
    fn a_non_finite_cost_never_qualifies() {
        let (ids, hit) = search_two(
            [(f32::NAN, f32::NAN), (10.0, 10.0)],
            [(0, 0.0), (1, 0.0)],
            vec![Vec3::new(10.0, 0.0, 0.0)],
            Vec3::ZERO,
            0.0,
        );
        let hit = hit.expect("the finite candidate");
        assert_eq!(
            hit.clip,
            ClipId(ids[1]),
            "the NaN-pose sample must not stick"
        );
        let (_, all_nan) = search_two(
            [(f32::NAN, f32::NAN), (10.0, 10.0)],
            [(0, 0.0), (1, 0.0)],
            vec![Vec3::NAN],
            Vec3::ZERO,
            0.0,
        );
        assert!(all_nan.is_none());
    }
}
