//! Retarget map: explicit «source joint → target joint» correspondence
//! (design §8.3).
//!
//! A [`RetargetMap`] lives in the cold lane on the retargeted (target)
//! skeleton root. Sampling resolves every clip track through it: the
//! sampled local channels land on the mapped target joint — rotation and
//! scale as authored, translation scaled by the rest-bone length ratio
//! (§8.3 «Перенос»). A `None` entry leaves the target joint at rest.
//! The same map drives every skeletal contributor on the target root:
//! the base [`crate::SkelPlayer`], its crossfade, overlay layers and
//! [`crate::SkelTree`] leaves. Hierarchical node writes stay untouched
//! (no node re-binds).

use crate::{JointId, Skeleton};

/// Rest lengths at or below this read as degenerate and take the unit
/// scale instead of a ratio (§8.3).
const MIN_REST_LENGTH: f32 = 1.0e-6;

/// Explicit «source joint → target joint» correspondence (design §8.3).
///
/// `joints[s]` names the target joint for source [`JointId`] `s`; `None`
/// means the bone is not transferred and the target joint stays at rest.
/// `scales[s]` is the translation scale of that pair — the target's
/// rest-bone length over the source's (§8.3), `1.0` where either length
/// is degenerate. No auto-matching by names or inverse bind (the ragdoll
/// doctrine): the map is authored explicitly and baked by
/// [`RetargetMap::bake`].
///
/// `Clone + Send + Sync` per the cold-lane contract.
#[derive(Debug, Clone, PartialEq)]
pub struct RetargetMap {
    /// Index = source [`JointId`] → target joint; `None` = the bone is not
    /// transferred (the target joint stays at rest).
    pub joints: Vec<Option<JointId>>,
    /// Translation scale per source joint: the target rest-bone length over
    /// the source rest-bone length (§8.3); `1.0` where degenerate. Parallel
    /// to [`Self::joints`].
    pub scales: Vec<f32>,
}

impl RetargetMap {
    /// Bakes `joints` into a validated map for playing `source`-authored
    /// clips on `target`'s skeleton.
    ///
    /// Returns [`None`] when the mapping does not fit: `joints` needs
    /// exactly one entry per source joint and every `Some(target)` must
    /// name a joint of `target`.
    #[must_use]
    pub fn bake(
        source: &Skeleton,
        target: &Skeleton,
        joints: Vec<Option<JointId>>,
    ) -> Option<Self> {
        if joints.len() != source.joint_count() {
            return None;
        }
        if joints
            .iter()
            .flatten()
            .any(|joint| joint.index() >= target.joint_count())
        {
            return None;
        }
        let scales = joints
            .iter()
            .enumerate()
            .map(|(source_id, target_id)| match target_id {
                Some(target_id) => rest_ratio(source, source_id, target, target_id.index()),
                None => 1.0,
            })
            .collect();
        Some(Self { joints, scales })
    }
}

/// Translation scale for one mapped pair: the target rest-bone length over
/// the source's (§8.3), `1.0` when either length is degenerate or
/// non-finite.
fn rest_ratio(source: &Skeleton, source_id: usize, target: &Skeleton, target_id: usize) -> f32 {
    let Some(source_rest) = source.rest.get(source_id) else {
        return 1.0;
    };
    let Some(target_rest) = target.rest.get(target_id) else {
        return 1.0;
    };
    let source_len = source_rest.translation.length();
    let target_len = target_rest.translation.length();
    if source_len > MIN_REST_LENGTH && source_len.is_finite() && target_len.is_finite() {
        target_len / source_len
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::*;
    use crate::JointTrs;

    fn bone(translation: Vec3) -> JointTrs {
        JointTrs {
            translation,
            ..JointTrs::IDENTITY
        }
    }

    fn skeleton(rest: Vec<JointTrs>) -> Skeleton {
        let count = rest.len();
        Skeleton::new(
            (0..count).map(|_| None).collect(),
            (0..count).map(|_| Mat4::IDENTITY).collect(),
            (0..count).map(|_| String::new()).collect(),
        )
        .with_rest(rest)
    }

    #[test]
    fn bake_transfers_and_scales_by_rest_lengths() {
        // Source bones along +X of lengths 1 and 2; target bones 2 and 2.
        let source = skeleton(vec![bone(Vec3::X), bone(Vec3::X * 2.0)]);
        let target = skeleton(vec![bone(Vec3::X * 2.0), bone(Vec3::Y * 2.0)]);
        let map =
            RetargetMap::bake(&source, &target, vec![Some(JointId::from_raw(1)), None]).unwrap();
        assert_eq!(map.joints, vec![Some(JointId::from_raw(1)), None]);
        assert_eq!(map.scales, vec![2.0, 1.0]);
    }

    #[test]
    fn bake_rejects_wrong_source_width() {
        let source = skeleton(vec![bone(Vec3::X), bone(Vec3::X)]);
        let target = skeleton(vec![bone(Vec3::X)]);
        assert!(RetargetMap::bake(&source, &target, vec![Some(JointId::from_raw(0))]).is_none());
    }

    #[test]
    fn bake_rejects_target_out_of_range() {
        let source = skeleton(vec![bone(Vec3::X)]);
        let target = skeleton(vec![bone(Vec3::X)]);
        assert!(RetargetMap::bake(&source, &target, vec![Some(JointId::from_raw(3))]).is_none());
    }

    #[test]
    fn bake_gives_degenerate_bones_unit_scale() {
        let source = skeleton(vec![bone(Vec3::ZERO)]);
        let target = skeleton(vec![bone(Vec3::X * 4.0)]);
        let map = RetargetMap::bake(&source, &target, vec![Some(JointId::from_raw(0))]).unwrap();
        assert_eq!(map.scales, vec![1.0]);
    }
}
