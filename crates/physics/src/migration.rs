//! Solver-independent joint assembly state and deterministic dense remapping.
//! Migrations discard numerical warm starts, never the physical rest pose.

use glam::{Quat, Vec3};

use crate::body::BodyHandle;
use crate::joint::{JointKind, ResolvedJoint};

/// Physical assembly references, distinct from impulses/penalty warm starts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JointReference {
    pub angle: f32,
    pub length: f32,
    pub distance: f32,
    pub rotation: Quat,
    pub anchor_delta: Vec3,
}

impl Default for JointReference {
    fn default() -> Self {
        Self {
            angle: 0.0,
            length: 0.0,
            distance: 0.0,
            rotation: Quat::IDENTITY,
            anchor_delta: Vec3::ZERO,
        }
    }
}

impl From<ResolvedJoint> for JointReference {
    fn from(r: ResolvedJoint) -> Self {
        Self {
            angle: r.ref_angle,
            length: r.ref_length,
            distance: r.ref_distance,
            rotation: r.ref_quat,
            anchor_delta: r.ref_anchor_delta,
        }
    }
}

/// A joint in a registry's handle space, including its original rest state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JointSnapshot {
    pub a: BodyHandle,
    pub b: BodyHandle,
    pub spec: JointKind,
    pub reference: JointReference,
}

/// Remove dependent gears before constructing the dense old-to-new map.
/// Building a map first and then removing gears silently retargets later refs.
pub(crate) fn joint_remap(kinds: &[JointKind], mut removed: Vec<bool>) -> Vec<Option<usize>> {
    loop {
        let mut changed = false;
        for (i, kind) in kinds.iter().enumerate() {
            if removed[i] {
                continue;
            }
            if let JointKind::Gear {
                joint_a, joint_b, ..
            } = kind
            {
                let invalid = [*joint_a, *joint_b].into_iter().any(|h| {
                    removed.get(h).copied().unwrap_or(true)
                        || !matches!(
                            kinds.get(h),
                            Some(JointKind::Revolute { .. } | JointKind::Prismatic { .. })
                        )
                });
                if invalid {
                    removed[i] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut next = 0;
    removed
        .into_iter()
        .map(|drop| {
            if drop {
                None
            } else {
                let index = next;
                next += 1;
                Some(index)
            }
        })
        .collect()
}

/// Rewrite the references of a surviving gear using a validated dense map.
pub(crate) fn remap_gear(kind: &mut JointKind, remap: &[Option<usize>]) {
    if let JointKind::Gear {
        joint_a, joint_b, ..
    } = kind
    {
        *joint_a = remap[*joint_a].expect("surviving gear has a surviving first joint");
        *joint_b = remap[*joint_b].expect("surviving gear has a surviving second joint");
    }
}

/// Unwrap an angular coordinate relative to stored raw/continuous values.
/// Linear coordinates are never periodic.
pub fn gear_coordinate(raw: f32, angular: bool, previous: Option<(f32, f32)>) -> f32 {
    match previous {
        Some((old_raw, continuous)) if angular => {
            continuous
                + ((raw - old_raw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                    - std::f32::consts::PI)
        }
        _ => raw,
    }
}

/// Completed-step contact baselines, transported separately from event queues.
#[derive(Clone, Debug, Default)]
pub(crate) struct EventState {
    pub contacts: std::collections::BTreeSet<(BodyHandle, BodyHandle)>,
    pub triggers: std::collections::BTreeSet<(BodyHandle, BodyHandle)>,
}

/// Reject parameters that would make a clamp, normalization or spring undefined.
/// Zero axes retain each solver's documented fallback/rejection policy.
pub(crate) fn valid_joint(kind: &JointKind) -> bool {
    use crate::joint::AxisConfig;
    let bounds = |lo: f32, hi: f32| lo.is_finite() && hi.is_finite() && lo <= hi;
    let motor = |speed: f32, cap: f32| speed.is_finite() && cap.is_finite() && cap >= 0.0;
    let axis = |v: Vec3| v.is_finite();
    let config = |c: &AxisConfig| match *c {
        AxisConfig::Free | AxisConfig::Locked => true,
        AxisConfig::Limited { min, max } => bounds(min, max),
    };
    let anchors = match *kind {
        JointKind::Gear { ratio, .. } => return ratio.is_finite(),
        JointKind::Ball {
            local_anchor_a,
            local_anchor_b,
        }
        | JointKind::Fixed {
            local_anchor_a,
            local_anchor_b,
        }
        | JointKind::Distance {
            local_anchor_a,
            local_anchor_b,
        }
        | JointKind::Revolute {
            local_anchor_a,
            local_anchor_b,
            ..
        }
        | JointKind::Prismatic {
            local_anchor_a,
            local_anchor_b,
            ..
        }
        | JointKind::Wheel {
            local_anchor_a,
            local_anchor_b,
            ..
        }
        | JointKind::SixDof {
            local_anchor_a,
            local_anchor_b,
            ..
        } => axis(local_anchor_a) && axis(local_anchor_b),
    };
    anchors
        && match *kind {
            JointKind::Revolute {
                local_axis_a,
                local_axis_b,
                limit,
                motor: drive,
                ..
            } => {
                axis(local_axis_a)
                    && axis(local_axis_b)
                    && limit.is_none_or(|l| bounds(l.min, l.max))
                    && drive.is_none_or(|m| motor(m.target_speed, m.max_torque))
            }
            JointKind::Prismatic {
                local_axis_a,
                local_axis_b,
                limit,
                motor: drive,
                ..
            } => {
                axis(local_axis_a)
                    && axis(local_axis_b)
                    && limit.is_none_or(|l| bounds(l.min, l.max))
                    && drive.is_none_or(|m| motor(m.target_speed, m.max_force))
            }
            JointKind::Wheel {
                local_suspension_a,
                local_suspension_b,
                local_axle_a,
                local_axle_b,
                suspension,
                motor: drive,
                ..
            } => {
                [
                    local_suspension_a,
                    local_suspension_b,
                    local_axle_a,
                    local_axle_b,
                ]
                .into_iter()
                .all(axis)
                    && suspension.frequency_hz.is_finite()
                    && suspension.frequency_hz >= 0.0
                    && suspension.damping_ratio.is_finite()
                    && suspension.damping_ratio >= 0.0
                    && drive.is_none_or(|m| motor(m.target_speed, m.max_torque))
            }
            JointKind::SixDof {
                linear, angular, ..
            } => linear.iter().chain(&angular).all(config),
            _ => true,
        }
}

/// Registry transfer without recapturing rest poses or driver history.
pub(crate) struct SceneSnapshot {
    pub bodies: Vec<crate::RigidBody>,
    pub joints: Vec<JointSnapshot>,
    pub previous: Vec<crate::broadphase::PrevPose>,
    pub events: EventState,
}
