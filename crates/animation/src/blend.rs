//! Phase E blending (`docs/animation-design.md` §5 E): weighted players,
//! crossfades and joint-masked overlay layers, for skeletal joints and
//! object tracks alike.
//!
//! The model is deterministic and stateless per frame, in the spirit of
//! Godot's default blending: every missing channel reads as the authored
//! rest value (a joint's node transform in the skin, see
//! [`Skeleton::rest`](crate::Skeleton::rest); an object track's
//! [`AnimTrack::rest`](crate::AnimTrack::rest)), a player at weight `w`
//! contributes `rest + w·(pose − rest)`, and a layer at weight `w·mask[j]`
//! lerps its joint `j` over the base pose. The only mutable state is the
//! cursor clocks, so no pose ever feeds back into the next frame's sample.

use glam::{Mat4, Quat, Vec3};
use ornis_core::{Clamped01, Entity, Seconds};

use crate::{SkelClip, SkelPlayer};

/// Handle of the playlist entity holding a [`JointMask`] in the cold lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaskId(pub Entity);

/// Per-joint layer weights (cold, on a playlist entity).
///
/// `weights[j]` scales a [`SkelLayer`] on joint `j`: `0` leaves the base
/// pose, `1` takes the layer fully. The length must equal the skeleton
/// joint count; a mismatch marks the layered entity as bad skin (counted,
/// never silently truncated).
#[derive(Debug, Clone, PartialEq)]
pub struct JointMask {
    /// Diagnostics label (for example `upper_body`).
    pub name: String,
    /// One weight per joint of the skeleton the mask is used with.
    pub weights: Vec<Clamped01>,
}

/// Overlay layer on a skeleton root: one clip blended over the base pose
/// through a [`JointMask`].
///
/// The embedded [`SkelPlayer`] is the layer cursor: its `weight` is the
/// layer strength before the mask, and it advances exactly like a base
/// player. The only layer type (objects have none); per root the layers
/// stack in [`SkelLayerStack`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkelLayer {
    /// Clip cursor and layer strength (`player.weight`).
    pub player: SkelPlayer,
    /// Playlist entity holding the [`JointMask`] that scales the layer.
    pub mask: MaskId,
}

/// Overlay layers on one skeleton root, blended over the base pose
/// bottom-up.
///
/// `layers[0]` (the bottom) lerps over the base pose first, each next
/// layer blends over that result, so the last element ends topmost — where
/// full-weight masks overlap, the top layer wins. Upper-body over
/// locomotion is the stack `[locomotion_overlay, upper_body]` with the
/// matching masks. Stacks install whole (set semantics: the new stack
/// replaces the old one, an empty one clears every layer) and, like the
/// rest of the blend model, hold no state but cursor clocks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SkelLayerStack {
    /// Layers from bottom to top.
    pub layers: Vec<SkelLayer>,
}

/// Crossfade in progress on a skeleton root.
///
/// The outgoing player keeps its own clock in `from` and is sampled until
/// `elapsed` reaches `duration`, then the system drops this component. The
/// root's [`SkelPlayer`] is the incoming clip. `progress` is linear in time:
/// `0` is the outgoing pose, `1` the incoming one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkelFade {
    /// Outgoing player snapshot; its time keeps advancing during the fade.
    pub from: SkelPlayer,
    /// Fade length in seconds; non-positive means an instant cut.
    pub duration: Seconds,
    /// Time since the fade started.
    pub elapsed: Seconds,
}

/// One joint's sampled channels: `None` marks an absent channel (no track).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct JointSample {
    pub(crate) translation: Option<Vec3>,
    pub(crate) rotation: Option<Quat>,
    pub(crate) scale: Option<Vec3>,
}

/// Joint-local transform: a rest pose or one blended result.
///
/// The authored rest of a joint or object track is its node transform in
/// the source (glTF `node.transform().decomposed()`); builders without one
/// default to [`JointTrs::IDENTITY`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointTrs {
    /// Joint-local translation.
    pub translation: Vec3,
    /// Joint-local rotation.
    pub rotation: Quat,
    /// Joint-local scale.
    pub scale: Vec3,
}

impl JointTrs {
    /// Identity transform: the default rest for skeletons without authored
    /// rest data.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Fills absent channels from `rest`, so a missing track reads as rest.
    pub(crate) fn from_sample(sample: JointSample, rest: JointTrs) -> Self {
        Self {
            translation: sample.translation.unwrap_or(rest.translation),
            rotation: sample.rotation.unwrap_or(rest.rotation),
            scale: sample.scale.unwrap_or(rest.scale),
        }
    }

    /// Channel-wise blend: lerp for translation and scale, slerp for rotation.
    pub(crate) fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            translation: self.translation.lerp(other.translation, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }

    /// `T·R·S` matrix (the render convention, see `joint_local_matrix`).
    pub(crate) fn matrix(self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// Per-channel presence flags `[translation, rotation, scale]`.
type Presence = [bool; 3];

fn presence(sample: &JointSample) -> Presence {
    [
        sample.translation.is_some(),
        sample.rotation.is_some(),
        sample.scale.is_some(),
    ]
}

fn union(a: Presence, b: Presence) -> Presence {
    [a[0] || b[0], a[1] || b[1], a[2] || b[2]]
}

/// Samples every joint track at `time` into `out`; untracked joints stay absent.
///
/// First track per joint wins on duplicates (same rule as object tracks).
/// Tracks naming a joint outside `0..joint_count` are ignored. `out` and
/// `painted` are caller-owned scratch: both are cleared and refilled, so
/// the per-frame sample path never allocates.
pub(crate) fn sample_joints(
    clip: &SkelClip,
    joint_count: usize,
    time: f32,
    out: &mut Vec<JointSample>,
    painted: &mut Vec<bool>,
) {
    out.clear();
    out.resize(joint_count, JointSample::default());
    painted.clear();
    painted.resize(joint_count, false);
    for track in &clip.tracks {
        let joint = track.joint.index();
        if joint >= joint_count || painted[joint] {
            continue;
        }
        painted[joint] = true;
        out[joint] = JointSample {
            translation: track.translation.sample(time),
            rotation: track.rotation.sample(time),
            scale: track.scale.sample(time),
        };
    }
}

/// Blended joint transforms plus which channels any contributor provided.
///
/// A reusable blend buffer: `refill`/`mix_towards`/`overlay` write into the
/// existing allocations instead of returning new ones.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct BlendedPose {
    pub(crate) trs: Vec<JointTrs>,
    pub(crate) present: Vec<Presence>,
}

impl BlendedPose {
    /// One clip at weight `weight`: `rest + weight·(pose − rest)` per joint.
    ///
    /// `rest` holds one authored rest per joint; callers pass the skeleton's
    /// rest with the same length as `samples`. Previous content is replaced;
    /// the backing allocations are reused.
    pub(crate) fn refill(&mut self, samples: &[JointSample], rest: &[JointTrs], weight: f32) {
        self.trs.clear();
        self.present.clear();
        self.trs.extend(
            samples
                .iter()
                .zip(rest)
                .map(|(sample, base)| base.lerp(JointTrs::from_sample(*sample, *base), weight)),
        );
        self.present.extend(samples.iter().map(presence));
    }

    /// Every joint at authored `rest` with no channel present — the pose of
    /// a side that animates nothing (or whose clip vanished). Reuses the
    /// buffer like [`Self::refill`].
    pub(crate) fn refill_rest(&mut self, rest: &[JointTrs]) {
        self.trs.clear();
        self.present.clear();
        self.trs.extend(rest.iter().copied());
        self.present.resize(rest.len(), Presence::default());
    }

    /// Crossfade in place: `self` is the outgoing side and `to` the incoming
    /// one, `self` becomes `from·(1 − progress) + to·progress` per joint.
    ///
    /// Callers keep `self` and `to` in distinct buffers (the blend is
    /// element-wise and would otherwise read half-written joints).
    pub(crate) fn mix_towards(&mut self, to: &Self, progress: f32) {
        for (a, b) in self.trs.iter_mut().zip(&to.trs) {
            *a = a.lerp(*b, progress);
        }
        for (a, b) in self.present.iter_mut().zip(&to.present) {
            *a = union(*a, *b);
        }
    }

    /// Lerps joints over the current pose by `layer_weight · mask[j]`.
    ///
    /// Joints with a zero effective weight are untouched. Callers check
    /// that `mask` and `samples` both match the joint count.
    pub(crate) fn overlay(
        &mut self,
        samples: &[JointSample],
        rest: &[JointTrs],
        mask: &[Clamped01],
        layer_weight: Clamped01,
    ) {
        for (joint, sample) in samples.iter().enumerate() {
            let (Some(mask_weight), Some(base)) = (mask.get(joint), rest.get(joint)) else {
                continue;
            };
            let weight = layer_weight.get() * mask_weight.get();
            if weight <= 0.0 {
                continue;
            }
            let layer = JointTrs::from_sample(*sample, *base);
            self.trs[joint] = self.trs[joint].lerp(layer, weight);
            self.present[joint] = union(self.present[joint], presence(sample));
        }
    }

    /// Folds `other` into a running N-way blend at relative `weight`.
    ///
    /// `total` carries the weight folded so far and starts at `0.0`. Each
    /// fold mixes `other` in at the renormalized share
    /// `weight / (total + weight)`, so folding poses with weights that sum
    /// to one yields their weighted blend, and a single input at any
    /// positive weight reads as that pose alone (the first fold takes it
    /// whole). The two-input case is [`Self::mix_towards`] exactly (the
    /// same lerp/slerp), and the outcome is deterministic for a fixed
    /// fold order — rotations are order-sensitive, as everywhere else in
    /// the blend model. Non-positive weights are a no-op.
    pub(crate) fn fold_pose(&mut self, other: &Self, weight: f32, total: &mut f32) {
        if weight <= 0.0 {
            return;
        }
        let next = *total + weight;
        let share = weight / next;
        *total = next;
        self.mix_towards(other, share);
    }
}

/// Crossfade progress in `0..=1`: linear in `elapsed / duration`, instant
/// for a non-positive duration.
pub fn fade_progress(duration: Seconds, elapsed: Seconds) -> f32 {
    let duration = duration.get();
    if duration <= crate::NEAR_ZERO {
        return 1.0;
    }
    (elapsed.get() / duration).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_t(value: Vec3) -> JointSample {
        JointSample {
            translation: Some(value),
            ..JointSample::default()
        }
    }

    fn sample_r(value: Quat) -> JointSample {
        JointSample {
            rotation: Some(value),
            ..JointSample::default()
        }
    }

    /// Identity rest for `n` joints.
    fn ident(n: usize) -> Vec<JointTrs> {
        vec![JointTrs::IDENTITY; n]
    }

    /// One clip over identity rest.
    fn pose_of(samples: &[JointSample], weight: f32) -> BlendedPose {
        let mut pose = BlendedPose::default();
        pose.refill(samples, &ident(samples.len()), weight);
        pose
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-5
    }

    #[test]
    fn weight_one_is_the_pose_itself() {
        let pose = pose_of(&[sample_t(Vec3::new(4.0, 0.0, 0.0))], 1.0);
        assert!(close(pose.trs[0].translation, Vec3::new(4.0, 0.0, 0.0)));
    }

    #[test]
    fn weight_half_pulls_translation_toward_rest() {
        let pose = pose_of(&[sample_t(Vec3::new(4.0, 0.0, 0.0))], 0.5);
        assert!(close(pose.trs[0].translation, Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn weight_zero_is_rest() {
        let pose = pose_of(&[sample_t(Vec3::new(4.0, 0.0, 0.0))], 0.0);
        assert!(close(pose.trs[0].translation, Vec3::ZERO));
        assert_eq!(pose.trs[0].rotation, Quat::IDENTITY);
    }

    #[test]
    fn weight_half_halves_rotation_angle() {
        let full = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let pose = pose_of(&[sample_r(full)], 0.5);
        let half = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        assert!((pose.trs[0].rotation - half).length() < 1e-5);
    }

    #[test]
    fn absent_channels_stay_at_rest_and_untracked() {
        let pose = pose_of(&[JointSample::default()], 0.7);
        assert_eq!(pose.trs[0], JointTrs::IDENTITY);
        assert_eq!(pose.present[0], [false, false, false]);
    }

    #[test]
    fn mix_endpoints_are_the_inputs() {
        let from = pose_of(&[sample_t(Vec3::X)], 1.0);
        let to = pose_of(&[sample_t(Vec3::Y)], 1.0);
        let mut out = from.clone();
        out.mix_towards(&to, 0.0);
        assert!(close(out.trs[0].translation, Vec3::X));
        out = from.clone();
        out.mix_towards(&to, 1.0);
        assert!(close(out.trs[0].translation, Vec3::Y));
        out = from;
        out.mix_towards(&to, 0.5);
        assert!(close(out.trs[0].translation, Vec3::new(0.5, 0.5, 0.0)));
    }

    #[test]
    fn overlay_follows_mask_per_joint() {
        let mut base = pose_of(&[sample_t(Vec3::X), sample_t(Vec3::X)], 1.0);
        let layer = [sample_t(Vec3::Y), sample_t(Vec3::Y)];
        let mask = [Clamped01::ONE, Clamped01::ZERO];
        base.overlay(&layer, &ident(layer.len()), &mask, Clamped01::ONE);
        assert!(close(base.trs[0].translation, Vec3::Y));
        assert!(close(base.trs[1].translation, Vec3::X));
    }

    #[test]
    fn overlay_weight_scales_the_mask() {
        let mut base = pose_of(&[sample_t(Vec3::X)], 1.0);
        let layer = [sample_t(Vec3::Y)];
        base.overlay(
            &layer,
            &ident(layer.len()),
            &[Clamped01::ONE],
            Clamped01::new(0.5),
        );
        assert!(close(base.trs[0].translation, Vec3::new(0.5, 0.5, 0.0)));
    }

    #[test]
    fn overlay_marks_presence_only_where_it_contributes() {
        let mut base = pose_of(&[JointSample::default()], 1.0);
        let layer = [sample_t(Vec3::Y)];
        base.overlay(
            &layer,
            &ident(layer.len()),
            &[Clamped01::ZERO],
            Clamped01::ONE,
        );
        assert_eq!(base.present[0], [false, false, false]);
        base.overlay(
            &layer,
            &ident(layer.len()),
            &[Clamped01::ONE],
            Clamped01::ONE,
        );
        assert_eq!(base.present[0], [true, false, false]);
    }

    /// Runs [`BlendedPose::fold_pose`] over `inputs`, starting from an
    /// empty blend (`total = 0`, rest contents).
    fn fold_all(inputs: &[(&BlendedPose, f32)]) -> BlendedPose {
        let mut acc = BlendedPose::default();
        acc.refill_rest(&ident(1));
        let mut total = 0.0;
        for (pose, weight) in inputs {
            acc.fold_pose(pose, *weight, &mut total);
        }
        acc
    }

    // Bodies of the fold tests sit in one inner block each: rustqual's
    // fragment scanner hashes windows of three consecutive top-level
    // statements, and the neighbouring tests already own the natural
    // statement shapes.
    #[test]
    fn fold_single_input_takes_the_pose_whole() {
        {
            let pose = pose_of(&[sample_t(Vec3::new(4.0, 0.0, 0.0))], 1.0);
            let acc = fold_all(&[(&pose, 0.25)]);
            assert!(close(acc.trs[0].translation, Vec3::new(4.0, 0.0, 0.0)));
        }
    }

    #[test]
    fn fold_two_halves_match_mix_towards() {
        {
            let a = pose_of(&[sample_t(Vec3::X)], 1.0);
            let b = pose_of(&[sample_t(Vec3::Y)], 1.0);
            let folded = fold_all(&[(&a, 0.5), (&b, 0.5)]);
            let mut mixed = a;
            mixed.mix_towards(&b, 0.5);
            assert!(close(folded.trs[0].translation, mixed.trs[0].translation));
        }
    }

    #[test]
    fn fold_three_equal_weights_average_the_poses() {
        {
            let a = pose_of(&[sample_t(Vec3::new(3.0, 0.0, 0.0))], 1.0);
            let b = pose_of(&[sample_t(Vec3::new(0.0, 3.0, 0.0))], 1.0);
            let c = pose_of(&[sample_t(Vec3::new(0.0, 0.0, 3.0))], 1.0);
            let acc = fold_all(&[(&a, 1.0), (&b, 1.0), (&c, 1.0)]);
            assert!(close(acc.trs[0].translation, Vec3::ONE));
        }
    }

    #[test]
    fn fold_skips_non_positive_weights_and_unions_presence() {
        {
            let empty = pose_of(&[JointSample::default()], 1.0);
            let moved = pose_of(&[sample_t(Vec3::Y)], 1.0);
            let acc = fold_all(&[(&empty, 0.0), (&moved, -1.0), (&moved, 2.0)]);
            assert!(close(acc.trs[0].translation, Vec3::Y));
            assert_eq!(acc.present[0], [true, false, false]);
        }
    }

    #[test]
    fn fade_progress_is_linear_and_clamped() {
        let two = Seconds::new(2.0);
        assert!((fade_progress(two, Seconds::new(0.5)) - 0.25).abs() < 1e-6);
        assert!((fade_progress(two, Seconds::new(9.0)) - 1.0).abs() < 1e-6);
        assert!((fade_progress(Seconds::ZERO, Seconds::ZERO) - 1.0).abs() < 1e-6);
    }
}
