//! Gizmo snap settings (PLAN §i, slice E4): quantize a drag preview to a
//! translate/rotate/scale step.
//!
//! Pure over [`TransformDesc`], like the drag math itself: the replica owns
//! the settings (a UI preference, not scene content — same footing as the
//! gizmo mode/space) and applies them to the preview before it is written
//! and committed. Disabled by default; a non-positive step leaves that
//! channel unquantized even when snap is on.

use glam::{Quat, Vec3};
use ornis_assets::scene::TransformDesc;
use ornis_core::units::UnitQuat;

/// Floor for a snapped scale component (never zero or negative), matching
/// the drag math's `MIN_SCALE`.
const MIN_SCALE: f32 = 1e-3;

/// Per-mode snap steps. `enabled` gates all three; each step is in world
/// units (translate), degrees (rotate) or a scale factor (scale).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SnapSettings {
    /// Whether snapping applies at all.
    pub enabled: bool,
    /// Translation step (world units).
    pub translate: f32,
    /// Rotation step (degrees).
    pub rotate_deg: f32,
    /// Scale step (factor).
    pub scale: f32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            translate: 0.5,
            rotate_deg: 15.0,
            scale: 0.25,
        }
    }
}

impl SnapSettings {
    /// Quantizes `transform` to the configured steps. With snap disabled the
    /// transform is returned unchanged; a non-positive step skips its
    /// channel. Rotation snaps the orientation's swept angle to the step.
    #[must_use]
    pub fn snap_transform(&self, transform: TransformDesc) -> TransformDesc {
        let steps = self.active_steps();
        TransformDesc {
            translation: snap_vec3(transform.translation, steps.translate),
            scale: floor_scale(snap_vec3(transform.scale, steps.scale), steps.scale),
            rotation: snap_rotation(transform.rotation, steps.rotate_deg),
        }
    }

    /// Steps with snapping disabled zeroed out, so a disabled setting leaves
    /// every channel unquantized (each leaf operation treats a non-positive
    /// step as identity).
    fn active_steps(&self) -> Self {
        if self.enabled {
            *self
        } else {
            Self {
                enabled: false,
                translate: 0.0,
                rotate_deg: 0.0,
                scale: 0.0,
            }
        }
    }
}

/// Rounds each component to the nearest multiple of `step`; a non-positive
/// step leaves the vector unchanged.
fn snap_vec3(v: Vec3, step: f32) -> Vec3 {
    if step <= 0.0 {
        return v;
    }
    Vec3::new(
        (v.x / step).round() * step,
        (v.y / step).round() * step,
        (v.z / step).round() * step,
    )
}

/// Floors a snapped scale at [`MIN_SCALE`]; a non-positive step (snapping
/// off for this channel) leaves the scale untouched.
fn floor_scale(v: Vec3, step: f32) -> Vec3 {
    if step <= 0.0 {
        return v;
    }
    Vec3::new(
        f32::max(v.x, MIN_SCALE),
        f32::max(v.y, MIN_SCALE),
        f32::max(v.z, MIN_SCALE),
    )
}

/// Rounds the orientation's angle to the nearest multiple of `step_deg`,
/// keeping its axis. A degenerate normalize keeps the original orientation.
fn snap_rotation(rotation: UnitQuat, step_deg: f32) -> UnitQuat {
    let (axis, angle) = snapped_axis_angle(rotation.get(), step_deg);
    UnitQuat::normalize(Quat::from_axis_angle(axis, angle)).unwrap_or(rotation)
}

/// The orientation's axis and its swept angle rounded to `step_deg`.
fn snapped_axis_angle(quat: Quat, step_deg: f32) -> (Vec3, f32) {
    let step = step_deg.to_radians();
    let (axis, angle) = quat.to_axis_angle();
    (axis, (angle / step).round() * step)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform(translation: Vec3, scale: Vec3) -> TransformDesc {
        TransformDesc {
            translation,
            scale,
            ..TransformDesc::IDENTITY
        }
    }

    #[test]
    fn disabled_snap_is_identity() {
        let snap = SnapSettings::default();
        let t = transform(Vec3::new(0.37, 1.23, -2.71), Vec3::new(1.13, 0.87, 2.42));
        assert_eq!(snap.snap_transform(t.clone()), t);
    }

    #[test]
    fn translation_lands_on_multiples_of_the_step() {
        let snap = SnapSettings {
            enabled: true,
            translate: 0.5,
            ..SnapSettings::default()
        };
        let t = snap.snap_transform(transform(Vec3::new(0.37, 1.23, -2.71), Vec3::ONE));
        assert!((t.translation - Vec3::new(0.5, 1.0, -2.5)).length() < 1e-6);
        for c in t.translation.to_array() {
            assert!((c / 0.5).round() * 0.5 - c < 1e-6, "{c} not a multiple");
        }
    }

    #[test]
    fn scale_lands_on_multiples_and_is_floored() {
        let snap = SnapSettings {
            enabled: true,
            scale: 0.25,
            ..SnapSettings::default()
        };
        let t = snap.snap_transform(transform(Vec3::ZERO, Vec3::new(1.13, 0.87, 0.0001)));
        assert!((t.scale - Vec3::new(1.25, 0.75, MIN_SCALE)).length() < 1e-6);
    }

    #[test]
    fn rotation_lands_on_multiples_of_the_step() {
        let snap = SnapSettings {
            enabled: true,
            rotate_deg: 15.0,
            ..SnapSettings::default()
        };
        let start = TransformDesc {
            rotation: UnitQuat::normalize(Quat::from_rotation_y(38.0_f32.to_radians()))
                .expect("unit"),
            ..TransformDesc::IDENTITY
        };
        let snapped = snap.snap_transform(start);
        let (_, angle) = snapped.rotation.get().to_axis_angle();
        let degrees = angle.to_degrees();
        assert!(
            (degrees - 45.0).abs() < 1e-3,
            "38° must snap to 45°, got {degrees}"
        );
        assert!(
            ((degrees / 15.0).round() * 15.0 - degrees).abs() < 1e-3,
            "{degrees} not a multiple of 15°"
        );
    }

    #[test]
    fn non_positive_step_skips_its_channel() {
        let snap = SnapSettings {
            enabled: true,
            translate: 0.0,
            rotate_deg: 0.0,
            scale: 0.0,
        };
        let t = transform(Vec3::new(0.37, 1.23, -2.71), Vec3::new(1.13, 0.87, 2.42));
        assert_eq!(snap.snap_transform(t.clone()), t);
    }
}
