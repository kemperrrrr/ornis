//! Root motion: the per-frame [`RootMotionDelta`] the played clip carries
//! (design §8.4, gate B — the sampler writes the delta, the host applies
//! it: to the transform, to the simulation, or C-style — accumulate in the
//! variable pass and consume on the fixed tick).

use glam::{Quat, Vec3};

use crate::events::CursorStep;
use crate::{JointId, JointTrack, SkelClip, Skeleton};

/// Steps below this read as no step (float noise).
const STEP_EPSILON: f32 = 1.0e-6;

/// The walk accumulates at most this many within-a-loop pieces and then
/// stops early — a runaway cursor step cannot spin the sampler.
const MAX_PIECES: u32 = 256;

/// One frame's root-motion delta from the played clip (design §8.4 B).
///
/// The motion of the skeleton's root joint over this frame's cursor step,
/// in the skeleton's model space: [`Self::translation`] is the joint's
/// position delta and [`Self::rotation`] the orientation delta
/// (`rotation * previous == current` at full weight). The sampler refreshes
/// the hot lane on the character root every frame (identity while paused or
/// clip-less); the host applies it — to the transform, to the simulation,
/// or C-style: accumulate in the variable pass and consume on the fixed
/// tick (§8.4 C).
///
/// `Clone + Send + Sync` per the hot-lane contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootMotionDelta {
    /// Root-joint translation delta of this frame, skeleton model space.
    pub translation: Vec3,
    /// Root-joint rotation delta of this frame: `rotation * previous ==
    /// current` at full weight.
    pub rotation: Quat,
}

impl RootMotionDelta {
    /// The no-motion delta (a paused or clip-less cursor).
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };
}

/// Integrates `root`'s channels over `cursor`'s step — the frame's
/// [`RootMotionDelta`] (design §8.4 B).
///
/// Looping clips wrap at the ends without losing motion at the seam (a
/// locomotion loop keeps walking forward through the track's snap-back),
/// clamped clips stop at the boundary, reverse cursors retrace. `weight`
/// scales the delta toward none (`0` holds the rest pose and moves
/// nothing). A missing clip, root joint or track reads as no motion, as
/// does a step past [`MAX_PIECES`] clip loops after the pieces already
/// walked.
pub(crate) fn root_motion(
    clip: Option<&SkelClip>,
    cursor: Option<CursorStep>,
    root: Option<JointId>,
    weight: f32,
) -> RootMotionDelta {
    let (Some(clip), Some(cursor), Some(root)) = (clip, cursor, root) else {
        return RootMotionDelta::IDENTITY;
    };
    if cursor.duration <= 0.0 {
        return RootMotionDelta::IDENTITY;
    }
    let Some(track) = clip.tracks.iter().find(|track| track.joint == root) else {
        return RootMotionDelta::IDENTITY;
    };
    let mut translation = Vec3::ZERO;
    let mut rotation = Quat::IDENTITY;
    walk(track, cursor, &mut translation, &mut rotation);
    RootMotionDelta {
        translation: translation * weight,
        rotation: Quat::IDENTITY.slerp(rotation, weight),
    }
}

/// Accumulates every piece of the cursor step into the running deltas.
fn walk(track: &JointTrack, cursor: CursorStep, translation: &mut Vec3, rotation: &mut Quat) {
    let mut at = cursor.prev;
    let mut left = cursor.step;
    for _ in 0..MAX_PIECES {
        let Some((from, to, next, used)) = piece(at, left, cursor.duration, cursor.looping) else {
            break;
        };
        accumulate(track, from, to, translation, rotation);
        at = next;
        left -= used;
    }
}

/// The next walk piece — `from → to` inside one loop — plus where the
/// walk continues (`next_at`) and how much of the step it consumed;
/// [`None`] at the end of the step.
fn piece(at: f32, left: f32, duration: f32, looping: bool) -> Option<(f32, f32, f32, f32)> {
    if left > STEP_EPSILON {
        return forward(at, left, duration, looping);
    }
    if left < -STEP_EPSILON {
        return backward(at, left, duration, looping);
    }
    None
}

/// Forward walk piece: ends the step inside the loop, wraps at the
/// duration (looping) or stops at it (clamped).
fn forward(at: f32, left: f32, duration: f32, looping: bool) -> Option<(f32, f32, f32, f32)> {
    let headroom = duration - at;
    if left < headroom {
        return Some((at, at + left, at + left, left));
    }
    if looping {
        return Some((at, duration, 0.0, headroom));
    }
    Some((at, duration, duration, left))
}

/// Backward walk piece (negative step): retraces inside the loop, wraps
/// down through zero (looping) or stops at it (clamped).
fn backward(at: f32, left: f32, duration: f32, looping: bool) -> Option<(f32, f32, f32, f32)> {
    let headroom = at;
    if -left < headroom {
        return Some((at, at + left, at + left, left));
    }
    if looping {
        return Some((at, 0.0, duration, -headroom));
    }
    Some((at, 0.0, 0.0, left))
}

/// Accumulates one within-a-loop piece `from → to` of `track`'s channels
/// into the running `translation`/`rotation` deltas.
fn accumulate(track: &JointTrack, from: f32, to: f32, translation: &mut Vec3, rotation: &mut Quat) {
    let from_t = track.translation.sample(from).unwrap_or(Vec3::ZERO);
    let to_t = track.translation.sample(to).unwrap_or(Vec3::ZERO);
    *translation += to_t - from_t;
    let from_r = track.rotation.sample(from).unwrap_or(Quat::IDENTITY);
    let to_r = track.rotation.sample(to).unwrap_or(Quat::IDENTITY);
    *rotation = (to_r * from_r.inverse()) * *rotation;
}

/// The skeleton's root joint that carries root motion (§8.4): the first
/// joint without a parent. [`None`] for a skeleton without one.
pub(crate) fn root_joint(skeleton: &Skeleton) -> Option<JointId> {
    skeleton
        .parents
        .iter()
        .position(|parent| parent.is_none())
        .map(JointId::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Key, KeyTrack};

    fn clip_over(translation: KeyTrack<Vec3>, rotation: KeyTrack<Quat>) -> SkelClip {
        SkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: vec![JointTrack {
                joint: JointId::from_raw(0),
                node: None,
                translation,
                rotation,
                scale: KeyTrack::linear(vec![]),
            }],
        }
    }

    fn walking() -> SkelClip {
        clip_over(
            KeyTrack::linear(vec![
                Key {
                    time: 0.0,
                    value: Vec3::ZERO,
                },
                Key {
                    time: 1.0,
                    value: Vec3::X,
                },
            ]),
            KeyTrack::linear(vec![]),
        )
    }

    fn step_of(prev: f32, step: f32, looping: bool) -> Option<CursorStep> {
        Some(CursorStep {
            prev,
            step,
            duration: 1.0,
            looping,
        })
    }

    fn run(clip: &SkelClip, cursor: Option<CursorStep>, weight: f32) -> RootMotionDelta {
        root_motion(Some(clip), cursor, Some(JointId::from_raw(0)), weight)
    }

    #[test]
    fn forward_step_translates_by_the_track() {
        let delta = run(&walking(), step_of(0.25, 0.5, false), 1.0);
        assert!((delta.translation - Vec3::X * 0.5).length() < 1e-5);
        assert_eq!(delta.rotation, Quat::IDENTITY);
    }

    #[test]
    fn clamped_step_stops_at_the_end() {
        let delta = run(&walking(), step_of(0.8, 0.5, false), 1.0);
        assert!((delta.translation - Vec3::X * 0.2).length() < 1e-5);
    }

    #[test]
    fn looping_step_keeps_walking_through_the_seam() {
        // A locomotion loop: the track snaps back at the loop point, but
        // the character keeps walking forward.
        let delta = run(&walking(), step_of(0.9, 0.2, true), 1.0);
        assert!((delta.translation - Vec3::X * 0.2).length() < 1e-5);
    }

    #[test]
    fn reverse_step_retraces_the_track() {
        let delta = run(&walking(), step_of(0.3, -0.2, false), 1.0);
        assert!((delta.translation - Vec3::X * -0.2).length() < 1e-5);
    }

    #[test]
    fn rotation_delta_composes_the_turn() {
        let quarter = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let clip = clip_over(
            KeyTrack::linear(vec![]),
            KeyTrack::linear(vec![
                Key {
                    time: 0.0,
                    value: Quat::IDENTITY,
                },
                Key {
                    time: 1.0,
                    value: quarter,
                },
            ]),
        );
        let delta = run(&clip, step_of(0.0, 1.0, false), 1.0);
        assert!((delta.rotation - quarter).length() < 1e-5);
    }

    #[test]
    fn weight_scales_the_delta_to_none() {
        let half = run(&walking(), step_of(0.25, 0.5, false), 0.5);
        assert!((half.translation - Vec3::X * 0.25).length() < 1e-5);
        let none = run(&walking(), step_of(0.25, 0.5, false), 0.0);
        assert_eq!(none, RootMotionDelta::IDENTITY);
    }

    #[test]
    fn missing_pieces_read_as_no_motion() {
        let root = Some(JointId::from_raw(0));
        let cursor = step_of(0.0, 0.5, false);
        assert_eq!(
            root_motion(None, cursor, root, 1.0),
            RootMotionDelta::IDENTITY
        );
        assert_eq!(
            root_motion(Some(&walking()), None, root, 1.0),
            RootMotionDelta::IDENTITY
        );
        assert_eq!(
            root_motion(Some(&walking()), cursor, None, 1.0),
            RootMotionDelta::IDENTITY
        );
        let empty = clip_over(KeyTrack::linear(vec![]), KeyTrack::linear(vec![]));
        assert_eq!(
            root_motion(Some(&empty), cursor, root, 1.0),
            RootMotionDelta::IDENTITY
        );
    }
}
