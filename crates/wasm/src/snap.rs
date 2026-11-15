//! Gizmo snap viewport glue (PLAN §i, slice E4): the browser-owned snap
//! steps, a UI preference like the gizmo mode/space. The replica reads them
//! on each drag preview; the toolbar writes them through [`set_snap`].

use std::cell::Cell;

use ornis_editor::SnapSettings;
use wasm_bindgen::prelude::*;

thread_local! {
    /// Active gizmo snap steps (E4); a UI preference, like mode/space.
    static SNAP: Cell<SnapSettings> = Cell::new(SnapSettings::default());
}

/// Active snap steps (read by the frame loop's drag step).
pub(crate) fn snap_settings() -> SnapSettings {
    SNAP.with(Cell::get)
}

/// Sets gizmo snap steps (E4): `enabled` gates all three channels;
/// `translate` is in world units, `rotate_deg` in degrees, `scale` a factor.
/// Applied to the next drag preview; a non-positive step leaves that channel
/// unquantized even when enabled.
// qual:api — wasm-bindgen export; called from the browser snap controls.
#[wasm_bindgen]
pub fn set_snap(enabled: bool, translate: f32, rotate_deg: f32, scale: f32) {
    SNAP.with(|cell| {
        cell.set(SnapSettings {
            enabled,
            translate,
            rotate_deg,
            scale,
        });
    });
}

/// The rotate step the grid actually uses, in degrees, or `null` when the
/// channel is unquantized (snap off, or an unusable step).
///
/// The grid coerces the requested step to a divisor of 360° and clamps it to
/// a half-turn at the coarse end, so the toolbar reads this back to show the
/// effective step next to the field instead of duplicating the coercion.
// qual:api — wasm-bindgen export; called from the browser snap controls.
#[wasm_bindgen]
pub fn snap_effective_rotate_deg() -> Option<f32> {
    snap_settings().effective_rotate_deg()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_settings_round_trips_through_the_thread_local() {
        set_snap(true, 0.25, 10.0, 0.5);
        let snap = snap_settings();
        assert!(snap.enabled);
        assert_eq!(snap.translate, 0.25);
        assert_eq!(snap.rotate_deg, 10.0);
        assert_eq!(snap.scale, 0.5);
        // Restore the default for the other tests on this thread.
        set_snap(false, 0.5, 15.0, 0.25);
        assert_eq!(snap_settings(), SnapSettings::default());
    }

    /// A zero step reaches `SnapSettings` verbatim (E4 review): the browser
    /// controls pass 0 through, so the wasm glue must not substitute a
    /// default, and each zeroed channel stays unquantized even when enabled.
    #[test]
    fn zero_steps_pass_through_and_disable_their_channel() {
        set_snap(true, 0.0, 0.0, 0.0);
        let snap = snap_settings();
        assert!(snap.enabled);
        assert_eq!(snap.translate, 0.0);
        assert_eq!(snap.rotate_deg, 0.0);
        assert_eq!(snap.scale, 0.0);
        assert_eq!(snap.snap_coordinate(0.37), 0.37);
        assert_eq!(snap.snap_scale_component(1.13), 1.13);
        assert_eq!(snap.snap_angle(0.38), 0.38);
        // Restore the default for the other tests on this thread.
        set_snap(false, 0.5, 15.0, 0.25);
    }

    /// A non-finite step relayed from the browser (NaN/inf, or a denormal
    /// that overflows division) reaches `SnapSettings` verbatim and leaves
    /// every channel unquantized rather than poisoning the preview (E4
    /// review).
    #[test]
    fn non_finite_steps_relay_through_and_disable_their_channel() {
        for step in [f32::NAN, f32::INFINITY, f32::from_bits(1)] {
            set_snap(true, step, step, step);
            let snap = snap_settings();
            assert_eq!(snap.snap_coordinate(0.37), 0.37, "translate step {step}");
            assert_eq!(snap.snap_scale_component(1.13), 1.13, "scale step {step}");
            assert_eq!(snap.snap_angle(0.38), 0.38, "rotate step {step}");
        }
        // Restore the default for the other tests on this thread.
        set_snap(false, 0.5, 15.0, 0.25);
    }

    /// The toolbar reads the effective rotate step back from the replica
    /// (E4 review, Editor 2): a requested step that does not divide 360°, or
    /// is wider than a half-turn, must be reported as the grid in use, and an
    /// unquantized channel as `null`.
    #[test]
    fn effective_rotate_step_is_readable_from_the_browser() {
        set_snap(true, 0.5, 25.0, 0.25);
        let coerced = snap_effective_rotate_deg().expect("a quantized channel");
        assert!(
            (coerced - 360.0 / 14.0).abs() < 1e-4,
            "25° must report 360/14, got {coerced}"
        );

        set_snap(true, 0.5, 300.0, 0.25);
        assert_eq!(snap_effective_rotate_deg(), Some(180.0));

        set_snap(true, 0.5, 15.0, 0.25);
        assert_eq!(snap_effective_rotate_deg(), Some(15.0));

        set_snap(true, 0.5, 0.0, 0.25);
        assert_eq!(snap_effective_rotate_deg(), None, "a zero step is off");

        set_snap(false, 0.5, 15.0, 0.25);
        assert_eq!(snap_effective_rotate_deg(), None, "snap is off");
    }
}
