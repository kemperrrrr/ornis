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
//! Every entry point quantizes ONE scalar to a grid anchored at zero, not at
//! the drag's press point: the coordinate along the dragged handle axis (or
//! one in-plane axis) for Move, the dragged scale component for Scale, the
//! absolute angle around the handle axis for Rotate. The caller keeps every
//! other component bit-exact, so a single-axis drag never moves the axes it
//! did not grab.
//!
//! The Move grid is measured along the handle axis: a world basis vector in
//! World space, `R·e` in Local space. A Local drag therefore lands on the grid
//! of the rotated axis, not on the world-axis-aligned grid lines — rounding
//! world components instead would slide the object off its own axis.
//!
//! The Rotate grid lives on a circle, so its step is coerced to the nearest
//! divisor of 360° ([`SnapSettings::snap_angle`]): a grid that does not close
//! the turn has a seam where the snapped angle jumps by less than a step.

/// Floor for a snapped scale component's magnitude (never zero), matching
/// the drag math's `MIN_SCALE`.
const MIN_SCALE: f32 = 1e-3;

/// Degrees in a full turn. The rotate grid has to close the circle, so its
/// step is measured against this rather than against the canonical angle
/// range.
const FULL_TURN_DEG: f32 = 360.0;

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
    /// Rounds a translation coordinate (world units) measured along the
    /// dragged handle axis, or along one in-plane axis of a plane handle, to
    /// the grid of planes perpendicular to that axis, spaced by the step and
    /// anchored at the world origin. In Local space the axis is `R·e`, so the
    /// grid follows the rotated axis instead of the world basis. Snap off or
    /// an unusable step returns `coordinate` unchanged.
    #[must_use]
    pub fn snap_coordinate(&self, coordinate: f32) -> f32 {
        if !self.enabled || !usable_step(self.translate) {
            return coordinate;
        }
        (coordinate / self.translate).round() * self.translate
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

    /// Rounds an absolute angle (radians) around the dragged axis to the
    /// rotate grid, anchored at zero twist rather than at the press point.
    ///
    /// The grid closes the circle: the step is coerced to the nearest value
    /// that divides a full turn evenly, `360 / round(360 / rotate_deg)`. A
    /// step that already divides 360° — the 15° default among them — is used
    /// unchanged; 25° becomes 360/14 ≈ 25.714°. Without that coercion the
    /// canonical `[-180°, 180°]` range would cut the circle between two grid
    /// points, and a drag crossing ±180° would jump by `360 mod step` instead
    /// of one step. Snap off or an unusable step returns `angle` unchanged.
    #[must_use]
    pub fn snap_angle(&self, angle: f32) -> f32 {
        if !self.enabled {
            return angle;
        }
        let Some(step) = self.rotate_step() else {
            return angle;
        };
        (angle / step).round() * step
    }

    /// Rotate step in radians, coerced to a divisor of a full turn.
    ///
    /// `None` when the step cannot define a circular grid: unusable (see
    /// [`usable_step`]), so small that `360 / step` overflows, or so large
    /// that `round(360 / step)` comes out zero.
    fn rotate_step(&self) -> Option<f32> {
        if !usable_step(self.rotate_deg) {
            return None;
        }
        let divisions = (FULL_TURN_DEG / self.rotate_deg).round();
        (divisions.is_finite() && divisions >= 1.0)
            .then(|| (FULL_TURN_DEG / divisions).to_radians())
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
        assert_eq!(snap.snap_coordinate(0.37), 0.37);
        assert_eq!(snap.snap_scale_component(1.13), 1.13);
        assert_eq!(snap.snap_angle(0.38), 0.38);
    }

    #[test]
    fn coordinate_lands_on_multiples_of_the_step() {
        let snap = SnapSettings {
            translate: 0.5,
            ..enabled()
        };
        for (coordinate, expected) in [(0.37, 0.5), (1.23, 1.0), (-2.71, -2.5)] {
            let got = snap.snap_coordinate(coordinate);
            assert!((got - expected).abs() < 1e-6, "{coordinate} -> {got}");
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
        assert_eq!(snap.snap_coordinate(0.37), 0.37);
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
            assert_eq!(snap.snap_coordinate(0.37), 0.37, "translate step {step}");
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
        assert_eq!(snap.snap_coordinate(0.37), 0.37);
        assert_eq!(snap.snap_scale_component(1.13), 1.13);
        assert_eq!(snap.snap_angle(0.38), 0.38);
    }

    /// The rotate grid closes the circle (E4 review, Editor a): sweeping the
    /// canonical angle range visits every grid point once, with exactly one
    /// step between neighbours — including the wrap from the last point back
    /// to zero. A step that does not divide 360° (25°) used to leave a
    /// `360 mod step` gap at ±180°, so a drag crossing the seam jumped by
    /// less than a step.
    #[test]
    fn rotate_grid_closes_the_circle_without_a_seam() {
        for step in [15.0, 25.0, 7.0, 45.0] {
            let snap = SnapSettings {
                rotate_deg: step,
                ..enabled()
            };
            let divisions = (360.0 / step).round();
            let expected = 360.0 / divisions;
            let mut grid: Vec<f32> = (-180..180)
                .map(|deg| snap.snap_angle((deg as f32).to_radians()).to_degrees())
                .map(|deg| deg.rem_euclid(360.0))
                .collect();
            grid.sort_by(f32::total_cmp);
            grid.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
            assert_eq!(grid.len(), divisions as usize, "step {step}: grid points");
            for pair in grid.windows(2) {
                assert!(
                    (pair[1] - pair[0] - expected).abs() < 1e-2,
                    "step {step}: gap {} != {expected}",
                    pair[1] - pair[0]
                );
            }
            let wrap = grid[0] + 360.0 - grid[grid.len() - 1];
            assert!(
                (wrap - expected).abs() < 1e-2,
                "step {step}: the wrap gap {wrap} != {expected}"
            );
        }
    }

    /// A step that already divides 360° is used unchanged; one that does not
    /// is coerced to the nearest divisor. A step that cannot define a
    /// circular grid at all leaves the angle alone.
    #[test]
    fn rotate_step_is_coerced_to_a_divisor_of_a_full_turn() {
        let exact = SnapSettings {
            rotate_deg: 15.0,
            ..enabled()
        };
        assert!(
            (exact.snap_angle(38.0_f32.to_radians()).to_degrees() - 45.0).abs() < 1e-3,
            "a divisor step must stay unchanged"
        );

        let coerced = SnapSettings {
            rotate_deg: 25.0,
            ..enabled()
        };
        let step = 360.0 / 14.0;
        let got = coerced.snap_angle(38.0_f32.to_radians()).to_degrees();
        assert!(
            (got - step).abs() < 1e-2,
            "38° must snap to {step}°, got {got}"
        );

        for step in [1000.0, 1e-38] {
            let snap = SnapSettings {
                rotate_deg: step,
                ..enabled()
            };
            assert_eq!(snap.snap_angle(0.38), 0.38, "rotate step {step}");
        }
    }
}
