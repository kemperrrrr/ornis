//! Gizmo snap settings (PLAN §i, slice E4): quantize the active drag channel
//! to a translate/rotate/scale step.
//!
//! Pure over `Vec3` and angles: the replica owns the settings (a UI
//! preference, not scene content — same footing as the gizmo mode/space) and
//! applies them to the preview's active channel before it is written and
//! committed. Disabled by default; a step that is not finite and positive —
//! or so tiny that dividing by it overflows — leaves that channel unquantized
//! even when snap is on.

use glam::Vec3;

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
    /// Rounds a translation to the translate step (world units). Snap off or
    /// an unusable step returns `v` unchanged.
    #[must_use]
    pub fn snap_translate(&self, v: Vec3) -> Vec3 {
        if !self.enabled || !usable_step(self.translate) {
            return v;
        }
        snap_vec3(v, self.translate)
    }

    /// Rounds a scale to the scale step (factor) and floors it at
    /// `MIN_SCALE`. Snap off or an unusable step returns `v` unchanged.
    #[must_use]
    pub fn snap_scale(&self, v: Vec3) -> Vec3 {
        if !self.enabled || !usable_step(self.scale) {
            return v;
        }
        floor_scale(snap_vec3(v, self.scale))
    }

    /// Rounds a swept angle (radians) to the rotate step (degrees). Snap off
    /// or an unusable step returns `angle` unchanged.
    #[must_use]
    pub fn snap_angle(&self, angle: f32) -> f32 {
        if !self.enabled || !usable_step(self.rotate_deg) {
            return angle;
        }
        let step = self.rotate_deg.to_radians();
        (angle / step).round() * step
    }
}

/// Whether a snap step can quantize without overflow: finite, positive, and
/// large enough that its reciprocal stays finite. A denormal step (e.g.
/// `1e-45`) passes `> 0.0` but overflows `v / step` to infinity, so it is
/// rejected just like a non-finite or non-positive step.
fn usable_step(step: f32) -> bool {
    step.is_finite() && step > 0.0 && (1.0 / step).is_finite()
}

/// Rounds each component to the nearest multiple of `step` (`step > 0`).
fn snap_vec3(v: Vec3, step: f32) -> Vec3 {
    Vec3::new(
        (v.x / step).round() * step,
        (v.y / step).round() * step,
        (v.z / step).round() * step,
    )
}

/// Floors each scale component at [`MIN_SCALE`].
fn floor_scale(v: Vec3) -> Vec3 {
    Vec3::new(
        f32::max(v.x, MIN_SCALE),
        f32::max(v.y, MIN_SCALE),
        f32::max(v.z, MIN_SCALE),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled() -> SnapSettings {
        SnapSettings {
            enabled: true,
            ..SnapSettings::default()
        }
    }

    #[test]
    fn disabled_snap_is_identity() {
        let snap = SnapSettings::default();
        let v = Vec3::new(0.37, 1.23, -2.71);
        assert_eq!(snap.snap_translate(v), v);
        assert_eq!(snap.snap_scale(v), v);
        assert_eq!(snap.snap_angle(0.38), 0.38);
    }

    #[test]
    fn translation_lands_on_multiples_of_the_step() {
        let snap = SnapSettings {
            translate: 0.5,
            ..enabled()
        };
        let t = snap.snap_translate(Vec3::new(0.37, 1.23, -2.71));
        assert!((t - Vec3::new(0.5, 1.0, -2.5)).length() < 1e-6);
        for c in t.to_array() {
            assert!(
                ((c / 0.5).round() * 0.5 - c).abs() < 1e-6,
                "{c} not a multiple"
            );
        }
    }

    #[test]
    fn scale_lands_on_multiples_and_is_floored() {
        let snap = SnapSettings {
            scale: 0.25,
            ..enabled()
        };
        let s = snap.snap_scale(Vec3::new(1.13, 0.87, 0.0001));
        assert!((s - Vec3::new(1.25, 0.75, MIN_SCALE)).length() < 1e-6);
    }

    #[test]
    fn angle_lands_on_multiples_of_the_step() {
        let snap = SnapSettings {
            rotate_deg: 15.0,
            ..enabled()
        };
        let positive = snap.snap_angle(38.0_f32.to_radians()).to_degrees();
        assert!(
            (positive - 45.0).abs() < 1e-3,
            "38° must snap to 45°, got {positive}"
        );
        let negative = snap.snap_angle((-38.0_f32).to_radians()).to_degrees();
        assert!(
            (negative + 45.0).abs() < 1e-3,
            "-38° must snap to -45°, got {negative}"
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
        let v = Vec3::new(0.37, 1.23, -2.71);
        assert_eq!(snap.snap_translate(v), v);
        assert_eq!(snap.snap_scale(v), v);
        assert_eq!(snap.snap_angle(0.38), 0.38);
    }

    /// A non-finite step (NaN/±inf) leaves every channel unquantized: the
    /// browser can relay one through `set_snap`, and dividing by it would
    /// poison the preview with NaN (E4 review).
    #[test]
    fn non_finite_step_skips_its_channel() {
        let v = Vec3::new(0.37, 1.23, -2.71);
        for step in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let snap = SnapSettings {
                enabled: true,
                translate: step,
                rotate_deg: step,
                scale: step,
            };
            assert_eq!(snap.snap_translate(v), v, "translate step {step}");
            assert_eq!(snap.snap_scale(v), v, "scale step {step}");
            assert_eq!(snap.snap_angle(0.38), 0.38, "rotate step {step}");
        }
    }

    /// A denormal step (e.g. `1e-45`) is finite and positive but overflows
    /// `v / step` to infinity, so it must skip the channel rather than emit
    /// an infinite preview (E4 review).
    #[test]
    fn denormal_step_skips_its_channel() {
        let tiny = f32::from_bits(1); // smallest positive denormal
        let snap = SnapSettings {
            enabled: true,
            translate: tiny,
            rotate_deg: tiny,
            scale: tiny,
        };
        let v = Vec3::new(0.37, 1.23, -2.71);
        assert_eq!(snap.snap_translate(v), v);
        assert_eq!(snap.snap_scale(v), v);
        assert_eq!(snap.snap_angle(0.38), 0.38);
    }
}
