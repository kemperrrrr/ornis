//! Phase E skeletal blending (`docs/animation-design.md` §5 E): weighted
//! players, crossfades and joint-masked overlay layers.
//!
//! The model is deterministic and stateless per frame, in the spirit of
//! Godot's default blending: every missing channel reads as the rest value
//! (identity for joints, see [`JointTrs::IDENTITY`]), a player at weight `w`
//! contributes `rest + w·(pose − rest)`, and a layer at weight `w·mask[j]`
//! lerps its joint `j` over the base pose. The only mutable state is the
//! cursor clocks, so no pose ever feeds back into the next frame's sample.
//!
//! Object (non-skeletal) tracks are out of scope for this module: they write
//! into live placement components and need an authored rest pose first.

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

/// Overlay layer on a skeleton root: a second clip blended over the base
/// pose through a [`JointMask`].
///
/// Hot lane, at most one layer per root. The embedded [`SkelPlayer`] is
/// the layer cursor: its `weight` is the layer strength before the mask,
/// and it advances exactly like a base player.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkelLayer {
    /// Clip cursor and layer strength (`player.weight`).
    pub player: SkelPlayer,
    /// Playlist entity holding the [`JointMask`] that scales the layer.
    pub mask: MaskId,
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

/// Joint-local transform with channel defaults at rest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JointTrs {
    pub(crate) translation: Vec3,
    pub(crate) rotation: Quat,
    pub(crate) scale: Vec3,
}

impl JointTrs {
    /// Rest value of every channel: the value an absent track reads as.
    pub(crate) const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Fills absent channels with rest values.
    pub(crate) fn from_sample(sample: JointSample) -> Self {
        Self {
            translation: sample.translation.unwrap_or(Vec3::ZERO),
            rotation: sample.rotation.unwrap_or(Quat::IDENTITY),
            scale: sample.scale.unwrap_or(Vec3::ONE),
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

/// Samples every joint track at `time`; untracked joints stay absent.
///
/// First track per joint wins on duplicates (same rule as object tracks).
/// Tracks naming a joint outside `0..joint_count` are ignored.
pub(crate) fn sample_joints(clip: &SkelClip, joint_count: usize, time: f32) -> Vec<JointSample> {
    let mut out = vec![JointSample::default(); joint_count];
    let mut painted = vec![false; joint_count];
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
    out
}

/// Blended joint transforms plus which channels any contributor provided.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BlendedPose {
    pub(crate) trs: Vec<JointTrs>,
    pub(crate) present: Vec<Presence>,
}

impl BlendedPose {
    /// One clip at weight `weight`: `rest + weight·(pose − rest)` per joint.
    pub(crate) fn from_samples(samples: &[JointSample], weight: f32) -> Self {
        Self {
            trs: samples
                .iter()
                .map(|sample| JointTrs::IDENTITY.lerp(JointTrs::from_sample(*sample), weight))
                .collect(),
            present: samples.iter().map(presence).collect(),
        }
    }

    /// Crossfade: `from` at `1 − progress`, `to` at `progress`, per joint.
    pub(crate) fn mix(from: &Self, to: &Self, progress: f32) -> Self {
        Self {
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
                .map(|(a, b)| union(*a, *b))
                .collect(),
        }
    }

    /// Lerps joints over the current pose by `layer_weight · mask[j]`.
    ///
    /// Joints with a zero effective weight are untouched. Callers check
    /// that `mask` and `samples` both match the joint count.
    pub(crate) fn overlay(
        &mut self,
        samples: &[JointSample],
        mask: &[Clamped01],
        layer_weight: Clamped01,
    ) {
        for (joint, sample) in samples.iter().enumerate() {
            let Some(mask_weight) = mask.get(joint) else {
                continue;
            };
            let weight = layer_weight.get() * mask_weight.get();
            if weight <= 0.0 {
                continue;
            }
            self.trs[joint] = self.trs[joint].lerp(JointTrs::from_sample(*sample), weight);
            self.present[joint] = union(self.present[joint], presence(sample));
        }
    }
}

/// Crossfade progress in `0..=1`: linear in `elapsed / duration`, instant
/// for a non-positive duration.
pub(crate) fn fade_progress(fade: &SkelFade) -> f32 {
    let duration = fade.duration.get();
    if duration <= crate::NEAR_ZERO {
        return 1.0;
    }
    (fade.elapsed.get() / duration).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ClipId;

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

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-5
    }

    #[test]
    fn weight_one_is_the_pose_itself() {
        let pose = BlendedPose::from_samples(&[sample_t(Vec3::new(4.0, 0.0, 0.0))], 1.0);
        assert!(close(pose.trs[0].translation, Vec3::new(4.0, 0.0, 0.0)));
    }

    #[test]
    fn weight_half_pulls_translation_toward_rest() {
        let pose = BlendedPose::from_samples(&[sample_t(Vec3::new(4.0, 0.0, 0.0))], 0.5);
        assert!(close(pose.trs[0].translation, Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn weight_zero_is_rest() {
        let pose = BlendedPose::from_samples(&[sample_t(Vec3::new(4.0, 0.0, 0.0))], 0.0);
        assert!(close(pose.trs[0].translation, Vec3::ZERO));
        assert_eq!(pose.trs[0].rotation, Quat::IDENTITY);
    }

    #[test]
    fn weight_half_halves_rotation_angle() {
        let full = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let pose = BlendedPose::from_samples(&[sample_r(full)], 0.5);
        let half = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        assert!((pose.trs[0].rotation - half).length() < 1e-5);
    }

    #[test]
    fn absent_channels_stay_at_rest_and_untracked() {
        let pose = BlendedPose::from_samples(&[JointSample::default()], 0.7);
        assert_eq!(pose.trs[0], JointTrs::IDENTITY);
        assert_eq!(pose.present[0], [false, false, false]);
    }

    #[test]
    fn mix_endpoints_are_the_inputs() {
        let from = BlendedPose::from_samples(&[sample_t(Vec3::X)], 1.0);
        let to = BlendedPose::from_samples(&[sample_t(Vec3::Y)], 1.0);
        assert!(close(
            BlendedPose::mix(&from, &to, 0.0).trs[0].translation,
            Vec3::X
        ));
        assert!(close(
            BlendedPose::mix(&from, &to, 1.0).trs[0].translation,
            Vec3::Y
        ));
        assert!(close(
            BlendedPose::mix(&from, &to, 0.5).trs[0].translation,
            Vec3::new(0.5, 0.5, 0.0)
        ));
    }

    #[test]
    fn overlay_follows_mask_per_joint() {
        let mut base = BlendedPose::from_samples(&[sample_t(Vec3::X), sample_t(Vec3::X)], 1.0);
        let layer = [sample_t(Vec3::Y), sample_t(Vec3::Y)];
        let mask = [Clamped01::ONE, Clamped01::ZERO];
        base.overlay(&layer, &mask, Clamped01::ONE);
        assert!(close(base.trs[0].translation, Vec3::Y));
        assert!(close(base.trs[1].translation, Vec3::X));
    }

    #[test]
    fn overlay_weight_scales_the_mask() {
        let mut base = BlendedPose::from_samples(&[sample_t(Vec3::X)], 1.0);
        let layer = [sample_t(Vec3::Y)];
        base.overlay(&layer, &[Clamped01::ONE], Clamped01::new(0.5));
        assert!(close(base.trs[0].translation, Vec3::new(0.5, 0.5, 0.0)));
    }

    #[test]
    fn overlay_marks_presence_only_where_it_contributes() {
        let mut base = BlendedPose::from_samples(&[JointSample::default()], 1.0);
        let layer = [sample_t(Vec3::Y)];
        base.overlay(&layer, &[Clamped01::ZERO], Clamped01::ONE);
        assert_eq!(base.present[0], [false, false, false]);
        base.overlay(&layer, &[Clamped01::ONE], Clamped01::ONE);
        assert_eq!(base.present[0], [true, false, false]);
    }

    #[test]
    fn fade_progress_is_linear_and_clamped() {
        let store = ornis_core::SmartStore::new();
        let clip_entity = store.create_entity();
        let from = SkelPlayer {
            clip: ClipId(clip_entity),
            time: Seconds::ZERO,
            speed: Seconds::new(1.0),
            weight: Clamped01::ONE,
            playing: true,
            looping: true,
        };
        let mut fade = SkelFade {
            from,
            duration: Seconds::new(2.0),
            elapsed: Seconds::new(0.5),
        };
        assert!((fade_progress(&fade) - 0.25).abs() < 1e-6);
        fade.elapsed = Seconds::new(9.0);
        assert!((fade_progress(&fade) - 1.0).abs() < 1e-6);
        fade.duration = Seconds::ZERO;
        assert!((fade_progress(&fade) - 1.0).abs() < 1e-6);
    }
}
