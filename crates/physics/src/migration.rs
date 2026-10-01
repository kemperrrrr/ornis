//! Solver-independent joint assembly state and deterministic dense remapping.
//! Migrations discard numerical warm starts, never the physical rest pose.

use glam::{Quat, Vec3};

use crate::body::BodyHandle;
use crate::flags::CoordKind;
use crate::invariants::{Meters, Radians};
use crate::joint::{JointHandle, JointKind, ResolvedJoint};

/// Physical assembly references, distinct from impulses/penalty warm starts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JointReference {
    /// Hinge twist at assembly (rad).
    pub angle: Radians,
    /// Slide/spring rest length at assembly (m).
    pub length: Meters,
    /// Distance-rod / gear constraint constant at assembly (m).
    pub distance: Meters,
    /// Relative orientation at assembly.
    pub rotation: Quat,
    /// Anchor separation at assembly (m, body-A frame).
    pub anchor_delta: Vec3,
}

impl Default for JointReference {
    fn default() -> Self {
        Self {
            angle: Radians(0.0),
            length: Meters(0.0),
            distance: Meters(0.0),
            rotation: Quat::IDENTITY,
            anchor_delta: Vec3::ZERO,
        }
    }
}

impl From<ResolvedJoint> for JointReference {
    fn from(r: ResolvedJoint) -> Self {
        Self {
            angle: Radians(r.ref_angle),
            length: Meters(r.ref_length),
            distance: Meters(r.ref_distance),
            rotation: r.ref_quat,
            anchor_delta: r.ref_anchor_delta,
        }
    }
}

/// A joint in a registry's handle space, including its original rest state.
///
/// Handle-space policy: at the [`crate::Engine`] boundary this is the
/// GLOBAL registry space ([`BodyHandle`](crate::body::BodyHandle) /
/// [`JointHandle`](crate::joint::JointHandle)); inside an engine's
/// `joint_snapshots` the same struct carries that engine's LOCAL table
/// indices. The split registry (`split.rs`) converts between the two with
/// the explicit [`LocalAvbdBody`](crate::body::LocalAvbdBody) /
/// [`LocalSiBody`](crate::body::LocalSiBody) (and joint) newtypes —
/// never by reinterpreting a bare `u32`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JointSnapshot {
    pub a: BodyHandle,
    pub b: BodyHandle,
    pub spec: JointKind,
    pub reference: JointReference,
}

/// Remove dependent gears before constructing the dense old-to-new map.
/// Building a map first and then removing gears silently retargets later refs.
pub(crate) fn joint_remap(kinds: &[JointKind], mut removed: Vec<bool>) -> Vec<Option<JointHandle>> {
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
                    removed.get(h.index()).copied().unwrap_or(true)
                        || !matches!(
                            kinds.get(h.index()),
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
    let mut next = 0u32;
    removed
        .into_iter()
        .map(|drop| {
            if drop {
                None
            } else {
                let handle = JointHandle::from_raw(next);
                next += 1;
                Some(handle)
            }
        })
        .collect()
}

/// Rewrite the references of a surviving gear using a validated dense map.
pub(crate) fn remap_gear(kind: &mut JointKind, remap: &[Option<JointHandle>]) {
    if let JointKind::Gear {
        joint_a, joint_b, ..
    } = kind
        && let (Some(a), Some(b)) = (remap[joint_a.index()], remap[joint_b.index()])
    {
        *joint_a = a;
        *joint_b = b;
    }
}

/// Unwrap an angular coordinate relative to stored raw/continuous values.
/// Linear coordinates are never periodic.
pub fn gear_coordinate(raw: f32, kind: CoordKind, previous: Option<(f32, f32)>) -> f32 {
    match previous {
        Some((old_raw, continuous)) if kind.is_angular() => {
            continuous
                + ((raw - old_raw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                    - std::f32::consts::PI)
        }
        _ => raw,
    }
}

/// Completed-step contact baselines, transported separately from event queues.
///
/// Same handle-space policy as [`JointSnapshot`]: global at the
/// [`crate::Engine`] boundary, engine-local inside an engine's
/// `event_state`. Remapped explicitly at the split boundary.
#[derive(Clone, Debug, Default)]
pub(crate) struct EventState {
    pub contacts: std::collections::BTreeSet<(BodyHandle, BodyHandle)>,
    pub triggers: std::collections::BTreeSet<(BodyHandle, BodyHandle)>,
}

/// Typed joint validation: `Ok(())` admits the joint, `Err` names the flaw.
///
/// # Errors
///
/// [`crate::errors::JointError`] with `BadBounds` (inverted/non-finite limits),
/// `NonFinite` (bad scalars), or `BadAxis` (non-finite anchors/axes/configs).
/// Gear ratios surface as `NonFinite { field: "ratio" }`.
pub(crate) fn validate_joint(kind: &JointKind) -> Result<(), crate::errors::JointError> {
    use crate::errors::JointError;
    use crate::joint::AxisConfig;
    let bounds = |min: f32, max: f32| {
        if min.is_finite() && max.is_finite() && min <= max {
            Ok(())
        } else {
            Err(JointError::BadBounds { min, max })
        }
    };
    let motor = |speed: f32, cap: f32, field: &str| {
        if speed.is_finite() && cap.is_finite() && cap >= 0.0 {
            Ok(())
        } else {
            Err(JointError::NonFinite {
                field: field.to_string(),
            })
        }
    };
    let axis = |v: Vec3, what: &str| {
        if v.is_finite() {
            Ok(())
        } else {
            Err(JointError::BadAxis {
                detail: what.to_string(),
            })
        }
    };
    let config = |c: &AxisConfig| match *c {
        AxisConfig::Free | AxisConfig::Locked => Ok(()),
        AxisConfig::Limited { min, max } => bounds(min, max),
    };
    let anchors = match *kind {
        JointKind::Gear { ratio, .. } => {
            return if ratio.is_finite() {
                Ok(())
            } else {
                Err(JointError::NonFinite {
                    field: "ratio".to_string(),
                })
            };
        }
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
        } => {
            axis(local_anchor_a, "local_anchor_a")?;
            axis(local_anchor_b, "local_anchor_b")?;
            Ok(())
        }
    };
    anchors?;
    match *kind {
        JointKind::Revolute {
            local_axis_a,
            local_axis_b,
            limit,
            motor: drive,
            ..
        } => {
            axis(local_axis_a, "local_axis_a")?;
            axis(local_axis_b, "local_axis_b")?;
            if let Some(l) = limit {
                bounds(l.min, l.max)?;
            }
            if let Some(m) = drive {
                motor(m.target_speed, m.max_torque, "revolute motor")?;
            }
            Ok(())
        }
        JointKind::Prismatic {
            local_axis_a,
            local_axis_b,
            limit,
            motor: drive,
            ..
        } => {
            axis(local_axis_a, "local_axis_a")?;
            axis(local_axis_b, "local_axis_b")?;
            if let Some(l) = limit {
                bounds(l.min, l.max)?;
            }
            if let Some(m) = drive {
                motor(m.target_speed, m.max_force, "prismatic motor")?;
            }
            Ok(())
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
            axis(local_suspension_a, "local_suspension_a")?;
            axis(local_suspension_b, "local_suspension_b")?;
            axis(local_axle_a, "local_axle_a")?;
            axis(local_axle_b, "local_axle_b")?;
            if !suspension.frequency_hz.is_finite() || suspension.frequency_hz < 0.0 {
                return Err(JointError::NonFinite {
                    field: "suspension.frequency_hz".to_string(),
                });
            }
            if !suspension.damping_ratio.is_finite() || suspension.damping_ratio < 0.0 {
                return Err(JointError::NonFinite {
                    field: "suspension.damping_ratio".to_string(),
                });
            }
            if let Some(m) = drive {
                motor(m.target_speed, m.max_torque, "wheel motor")?;
            }
            Ok(())
        }
        JointKind::SixDof {
            linear, angular, ..
        } => {
            for c in linear.iter().chain(&angular) {
                config(c)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Registry transfer without recapturing rest poses or driver history.
pub(crate) struct SceneSnapshot {
    pub bodies: Vec<crate::RigidBody>,
    pub joints: Vec<JointSnapshot>,
    pub previous: Vec<crate::broadphase::PrevPose>,
    pub events: EventState,
    /// Soft bodies in handle order (dense [`crate::soft::SoftHandle`]
    /// order). The XPBD path owns them live; every other solver parks them
    /// untouched — order (and therefore handles) is preserved either way.
    pub soft_bodies: Vec<crate::soft::SoftBody>,
    /// Live soft↔rigid `(soft, particle, body)` touch triples. Indices
    /// survive the ordered rebuild on both registries, so the set transfers
    /// verbatim and the XPBD target emits no manufactured begins. Empty
    /// whenever soft bodies are parked (parked bodies do not step, so they
    /// hold no live touch state worth keeping — the orchestrator clears it
    /// on any structural edit instead of remapping stale indices).
    pub soft_touch: std::collections::BTreeSet<(usize, usize, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::JointError;
    use glam::Vec3;

    #[test]
    fn validate_joint_names_the_flaw() {
        let bad = JointKind::Ball {
            local_anchor_a: Vec3::NAN,
            local_anchor_b: Vec3::ZERO,
        };
        assert!(matches!(
            validate_joint(&bad),
            Err(JointError::BadAxis { .. })
        ));
        assert!(validate_joint(&bad).is_err());
        let inverted = JointKind::Revolute {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            local_axis_a: Vec3::Y,
            local_axis_b: Vec3::Y,
            limit: Some(crate::joint::RevoluteLimit {
                min: 1.0,
                max: -1.0,
            }),
            motor: None,
        };
        assert!(matches!(
            validate_joint(&inverted),
            Err(JointError::BadBounds { .. })
        ));
        let bad_bounds = JointKind::Distance {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
        };
        assert!(validate_joint(&bad_bounds).is_ok());
        let gear = JointKind::Gear {
            joint_a: JointHandle::from_raw(0),
            joint_b: JointHandle::from_raw(1),
            ratio: f32::NAN,
        };
        assert!(matches!(
            validate_joint(&gear),
            Err(JointError::NonFinite { .. })
        ));
    }
}
