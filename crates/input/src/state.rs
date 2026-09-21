//! Backend-neutral per-frame input state.
//!
//! Platform adapters (winit, browser events and future integrations) update
//! [`InputState`] between frame calls. Systems read the same resource during
//! the frame; transient pointer and wheel deltas are cleared after the
//! schedule, while held keys/buttons remain active.

use std::collections::BTreeSet;

use crate::{KeyCode, MouseButton};

/// Input snapshot exposed to systems through the logical world.
///
/// Key and mouse-button identifiers are platform-neutral numeric codes. A
/// native adapter can use physical key codes, while a browser adapter can
/// use its DOM `code` mapping. The core intentionally does not depend on a
/// windowing or DOM crate.
///
/// Prefer the named helpers ([`InputState::keycode_down`],
/// [`InputState::button_down`]) and [`crate::InputMap`] on the read edge;
/// producers keep writing raw codes so the wire format never changes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputState {
    pressed_keys: BTreeSet<u32>,
    pressed_mouse_buttons: BTreeSet<u8>,
    pointer_position: [f32; 2],
    pointer_delta: [f32; 2],
    wheel_delta: f32,
}

impl InputState {
    /// Creates an input state with no held keys/buttons and zero deltas.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks a platform-neutral key code as pressed or released.
    pub fn set_key(&mut self, code: u32, pressed: bool) {
        if pressed {
            self.pressed_keys.insert(code);
        } else {
            self.pressed_keys.remove(&code);
        }
    }

    /// Whether the key code is currently held.
    pub fn key_down(&self, code: u32) -> bool {
        self.pressed_keys.contains(&code)
    }

    /// Marks every raw alias of `key` as pressed or released.
    ///
    /// Mirrors what the browser adapter does when it expands one DOM `code`
    /// into the physical + ASCII pair: a semantic press touches all aliases
    /// so both read paths observe it.
    pub fn set_keycode(&mut self, key: KeyCode, pressed: bool) {
        for &code in key.codes() {
            self.set_key(code, pressed);
        }
    }

    /// Whether any raw alias of `key` is currently held.
    pub fn keycode_down(&self, key: KeyCode) -> bool {
        key.codes().iter().any(|code| self.key_down(*code))
    }

    /// Marks a platform-neutral mouse-button code as pressed or released.
    pub fn set_mouse_button(&mut self, code: u8, pressed: bool) {
        if pressed {
            self.pressed_mouse_buttons.insert(code);
        } else {
            self.pressed_mouse_buttons.remove(&code);
        }
    }

    /// Whether the mouse-button code is currently held.
    pub fn mouse_button_down(&self, code: u8) -> bool {
        self.pressed_mouse_buttons.contains(&code)
    }

    /// Marks a named mouse button as pressed or released.
    pub fn set_button(&mut self, button: MouseButton, pressed: bool) {
        self.set_mouse_button(button.code(), pressed);
    }

    /// Whether the named mouse button is currently held.
    pub fn button_down(&self, button: MouseButton) -> bool {
        self.mouse_button_down(button.code())
    }

    /// Records an absolute pointer position and accumulates its frame delta.
    pub fn set_pointer_position(&mut self, position: [f32; 2]) {
        self.pointer_delta[0] += position[0] - self.pointer_position[0];
        self.pointer_delta[1] += position[1] - self.pointer_position[1];
        self.pointer_position = position;
    }

    /// Sets the pointer position without generating movement delta.
    ///
    /// Pointer-down adapters should use this to establish a drag anchor so
    /// the click location itself does not rotate a camera.
    pub fn set_pointer_anchor(&mut self, position: [f32; 2]) {
        self.pointer_position = position;
    }

    /// Absolute pointer position from the latest platform event.
    pub fn pointer_position(&self) -> [f32; 2] {
        self.pointer_position
    }

    /// Pointer movement accumulated since the previous frame boundary.
    pub fn pointer_delta(&self) -> [f32; 2] {
        self.pointer_delta
    }

    /// Adds a wheel amount to the current frame's accumulated delta.
    pub fn add_wheel_delta(&mut self, delta: f32) {
        if delta.is_finite() {
            self.wheel_delta += delta;
        }
    }

    /// Wheel movement accumulated since the previous frame boundary.
    pub fn wheel_delta(&self) -> f32 {
        self.wheel_delta
    }

    /// Currently held key codes, sorted ascending.
    ///
    /// Snapshots for the browser→server input channel (`POST /api/input`,
    /// WebSocket) are built from this; hot ECS loops keep using
    /// [`InputState::key_down`].
    pub fn pressed_keys(&self) -> Vec<u32> {
        self.pressed_keys.iter().copied().collect()
    }

    /// Currently held mouse button codes, sorted ascending.
    pub fn pressed_mouse_buttons(&self) -> Vec<u8> {
        self.pressed_mouse_buttons.iter().copied().collect()
    }

    /// Clears pointer and wheel deltas after a frame has consumed them.
    /// Held keys/buttons and the last absolute pointer position persist.
    pub fn clear_frame_transients(&mut self) {
        self.pointer_delta = [0.0, 0.0];
        self.wheel_delta = 0.0;
    }

    /// Replace the entire snapshot from a browser input frame.
    ///
    /// The browser is the authoritative source for this frame: held keys and
    /// buttons are replaced wholesale, and pointer/wheel deltas are set
    /// exactly to the values delivered over the WS / `POST /api/input`
    /// channel (no accumulation). This is the unified runtime's input side
    /// of `World/Engine/Schedule` without polling.
    pub fn apply_snapshot(
        &mut self,
        pressed_keys: &[u32],
        pressed_mouse_buttons: &[u8],
        pointer_position: [f32; 2],
        pointer_delta: [f32; 2],
        wheel_delta: f32,
    ) {
        self.pressed_keys = pressed_keys.iter().copied().collect();
        self.pressed_mouse_buttons = pressed_mouse_buttons.iter().copied().collect();
        self.pointer_position = pointer_position;
        self.pointer_delta = pointer_delta;
        self.wheel_delta = if wheel_delta.is_finite() {
            wheel_delta
        } else {
            0.0
        };
    }

    /// Releases all held keys/buttons and resets pointer/wheel state.
    ///
    /// Platform adapters should call this on focus loss or window teardown
    /// so a key released outside the window cannot remain logically stuck.
    pub fn clear_all(&mut self) {
        self.pressed_keys.clear();
        self.pressed_mouse_buttons.clear();
        self.pointer_position = [0.0, 0.0];
        self.clear_frame_transients();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_state_accumulates_events_and_clears_transients() {
        let mut input = InputState::new();
        input.set_key(17, true);
        input.set_mouse_button(1, true);
        input.set_pointer_anchor([10.0, 20.0]);
        input.set_pointer_position([13.0, 18.0]);
        input.add_wheel_delta(2.5);

        assert!(input.key_down(17));
        assert!(input.mouse_button_down(1));
        assert_eq!(input.pointer_position(), [13.0, 18.0]);
        assert_eq!(input.pointer_delta(), [3.0, -2.0]);
        assert_eq!(input.wheel_delta(), 2.5);
        assert_eq!(input.pressed_keys(), vec![17]);
        assert_eq!(input.pressed_mouse_buttons(), vec![1]);

        input.clear_frame_transients();
        assert_eq!(input.pointer_delta(), [0.0, 0.0]);
        assert_eq!(input.wheel_delta(), 0.0);
        assert!(input.key_down(17));
        assert!(input.mouse_button_down(1));

        input.clear_all();
        assert!(!input.key_down(17));
        assert!(!input.mouse_button_down(1));
        assert_eq!(input.pointer_position(), [0.0, 0.0]);
    }

    #[test]
    fn named_key_helpers_cover_all_raw_aliases() {
        let mut input = InputState::new();
        input.set_keycode(KeyCode::KeyW, true);
        assert!(input.keycode_down(KeyCode::KeyW));
        // Both the legacy physical and the ASCII alias land on the wire.
        assert_eq!(input.pressed_keys(), vec![17, 87]);

        // Releasing clears every alias at once.
        input.set_keycode(KeyCode::KeyW, false);
        assert!(!input.keycode_down(KeyCode::KeyW));
        assert!(input.pressed_keys().is_empty());

        // A single raw alias is enough for the named read to fire.
        input.set_key(87, true);
        assert!(input.keycode_down(KeyCode::KeyW));
        assert!(!input.keycode_down(KeyCode::KeyS));
    }

    #[test]
    fn named_button_helpers_match_raw_codes() {
        let mut input = InputState::new();
        input.set_button(MouseButton::Left, true);
        assert!(input.button_down(MouseButton::Left));
        assert!(input.mouse_button_down(0));
        assert!(!input.button_down(MouseButton::Right));

        input.set_button(MouseButton::Left, false);
        assert!(!input.button_down(MouseButton::Left));
    }

    #[test]
    fn snapshot_replaces_held_state_wholesale() {
        let mut input = InputState::new();
        input.set_key(17, true);
        input.set_mouse_button(0, true);
        input.set_pointer_position([5.0, 5.0]);
        input.add_wheel_delta(1.0);

        input.apply_snapshot(&[87], &[1], [1.0, 2.0], [3.0, 4.0], -2.0);
        assert!(!input.key_down(17));
        assert!(input.key_down(87));
        assert!(!input.mouse_button_down(0));
        assert!(input.mouse_button_down(1));
        assert_eq!(input.pointer_position(), [1.0, 2.0]);
        assert_eq!(input.pointer_delta(), [3.0, 4.0]);
        assert_eq!(input.wheel_delta(), -2.0);
        // Wire snapshots stay sorted ascending.
        input.set_key(17, true);
        assert_eq!(input.pressed_keys(), vec![17, 87]);
    }

    #[test]
    fn snapshot_sanitizes_non_finite_wheel_delta() {
        let mut input = InputState::new();
        input.apply_snapshot(&[], &[], [0.0, 0.0], [0.0, 0.0], f32::NAN);
        assert_eq!(input.wheel_delta(), 0.0);
        input.add_wheel_delta(f32::INFINITY);
        assert_eq!(input.wheel_delta(), 0.0);
    }
}
