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
        let v = glam::Vec3::new(0.37, 1.23, -2.71);
        assert_eq!(snap.snap_translate(v), v);
        assert_eq!(snap.snap_scale(v), v);
        assert_eq!(snap.snap_angle(0.38), 0.38);
        // Restore the default for the other tests on this thread.
        set_snap(false, 0.5, 15.0, 0.25);
    }
}
