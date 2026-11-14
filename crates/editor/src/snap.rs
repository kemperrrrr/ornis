//! Gizmo snap settings (PLAN §i, slice E4): quantize the active drag channel
//! to a translate/rotate/scale step.
//!
//! Pure over scalars and angles: the replica owns the settings (a UI
//! preference, not scene content — same footing as the gizmo mode/space) and
//! applies them to the preview's active channel before it is written and
//! committed. Disabled by default; a step that is not finite and positive —
//! or so tiny that dividing by it overflows — leaves that channel unquantized
//! even when snap is on.
//!
//! Every entry point quantizes ONE scalar: the offset along the dragged axis
//! (or one in-plane axis) for Move, the dragged scale component for Scale,
//! the swept angle for Rotate. The caller keeps every other component
//! bit-exact, so a single-axis drag never moves the axes it did not grab.

/// Floor for a snapped scale component's magnitude (never zero), matching
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
    /// Rounds a translation offset (world units) measured along the dragged
    /// axis, or along one in-plane axis of a plane handle. Snap off or an
    /// unusable step returns `offset` unchanged.
    #[must_use]
    pub fn snap_offset(&self, offset: f32) -> f32 {
        if !self.enabled || !usable_step(self.translate) {
            return offset;
        }
        (offset / self.translate).round() * self.translate
    }

    /// Rounds one dragged scale component to the scale step and floors its
    /// magnitude at `MIN_SCALE`, preserving the sign: a mirrored (negative)
    /// component stays mirrored instead of collapsing to the floor. Snap off
    /// or an unusable step returns `value` unchanged.
    #[must_use]
    pub fn snap_scale_component(&self, value: f32) -> f32 {
        if !self.enabled || !usable_step(self.scale) {
            return value;
        }
        let magnitude = (value.abs() / self.scale).round() * self.scale;
        magnitude.max(MIN_SCALE).copysign(value)
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
/// `1e-45`) passes `> 0.0` but overflows `value / step` to infinity, so it is
/// rejected just like a non-finite or non-positive step.
fn usable_step(step: f32) -> bool {
    step.is_finite() && step > 0.0 && (1.0 / step).is_finite()
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
        assert_eq!(snap.snap_offset(0.37), 0.37);
        assert_eq!(snap.snap_scale_component(1.13), 1.13);
        assert_eq!(snap.snap_angle(0.38), 0.38);
    }

    #[test]
    fn offset_lands_on_multiples_of_the_step() {
        let snap = SnapSettings {
            translate: 0.5,
            ..enabled()
        };
        for (offset, expected) in [(0.37, 0.5), (1.23, 1.0), (-2.71, -2.5)] {
            let got = snap.snap_offset(offset);
            assert!((got - expected).abs() < 1e-6, "{offset} -> {got}");
            assert!(
                ((got / 0.5).round() * 0.5 - got).abs() < 1e-6,
                "{got} not a multiple"
            );
        }
    }

    #[test]
    fn scale_lands_on_multiples_and_is_floored() {
        let snap = SnapSettings {
            scale: 0.25,
            ..enabled()
        };
        for (value, expected) in [(1.13, 1.25), (0.87, 0.75), (0.0001, MIN_SCALE)] {
            let got = snap.snap_scale_component(value);
            assert!((got - expected).abs() < 1e-6, "{value} -> {got}");
        }
    }

    /// A mirrored component is quantized by magnitude and keeps its sign:
    /// the floor applies to the magnitude, so `-1` never becomes `+0.001`
    /// (E4 review — a Scale-Y drag collapsed the mirrored X).
    #[test]
    fn scale_preserves_the_sign_of_a_mirrored_component() {
        let snap = SnapSettings {
            scale: 0.25,
            ..enabled()
        };
        for (value, expected) in [
            (-1.13, -1.25),
            (-1.0, -1.0),
            (-0.87, -0.75),
            (-0.0001, -MIN_SCALE),
        ] {
            let got = snap.snap_scale_component(value);
            assert!((got - expected).abs() < 1e-6, "{value} -> {got}");
            assert!(got < 0.0, "{value} lost its sign: {got}");
        }
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
        assert_eq!(snap.snap_offset(0.37), 0.37);
        assert_eq!(snap.snap_scale_component(1.13), 1.13);
        assert_eq!(snap.snap_angle(0.38), 0.38);
    }

    /// A non-finite step (NaN/±inf) leaves every channel unquantized: the
    /// browser can relay one through `set_snap`, and dividing by it would
    /// poison the preview with NaN (E4 review).
    #[test]
    fn non_finite_step_skips_its_channel() {
        for step in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let snap = SnapSettings {
                enabled: true,
                translate: step,
                rotate_deg: step,
                scale: step,
            };
            assert_eq!(snap.snap_offset(0.37), 0.37, "translate step {step}");
            assert_eq!(snap.snap_scale_component(1.13), 1.13, "scale step {step}");
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
        assert_eq!(snap.snap_offset(0.37), 0.37);
        assert_eq!(snap.snap_scale_component(1.13), 1.13);
        assert_eq!(snap.snap_angle(0.38), 0.38);
    }
}
