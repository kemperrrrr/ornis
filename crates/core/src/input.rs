//! Backend-neutral per-frame input state.
//!
//! Platform adapters (winit, browser events and future integrations) update
//! [`InputState`] between frame calls. Systems read the same resource during
//! [`crate::Engine::run_frame`]; transient pointer and wheel deltas are
//! cleared after the schedule, while held keys/buttons remain active.

use std::collections::BTreeSet;

/// Input snapshot exposed to systems through the logical [`crate::World`].
///
/// Key and mouse-button identifiers are platform-neutral numeric codes. A
/// native adapter can use physical key codes, while a browser adapter can
/// use its DOM `code` mapping. The core intentionally does not depend on a
/// windowing or DOM crate.
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

    /// Clears pointer and wheel deltas after a frame has consumed them.
    /// Held keys/buttons and the last absolute pointer position persist.
    pub fn clear_frame_transients(&mut self) {
        self.pointer_delta = [0.0, 0.0];
        self.wheel_delta = 0.0;
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
    use crate::{Engine, Resources, System, SystemAccess};
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Captured {
        key_down: bool,
        button_down: bool,
        pointer_position: [f32; 2],
        pointer_delta: [f32; 2],
        wheel_delta: f32,
    }

    struct CaptureInput(Arc<Mutex<Vec<Captured>>>);

    impl System for CaptureInput {
        fn name(&self) -> &'static str {
            "capture_input"
        }

        fn access(&self) -> SystemAccess {
            SystemAccess::new().reads::<InputState>()
        }

        fn run(&self, resources: &Resources) {
            let input = resources.get::<InputState>().expect("input resource");
            self.0.lock().expect("capture lock").push(Captured {
                key_down: input.key_down(17),
                button_down: input.mouse_button_down(1),
                pointer_position: input.pointer_position(),
                pointer_delta: input.pointer_delta(),
                wheel_delta: input.wheel_delta(),
            });
        }
    }

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
    fn engine_publishes_input_to_systems_before_clearing_deltas() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut engine = Engine::new();
        engine
            .schedule_mut()
            .add_system(CaptureInput(captured.clone()));
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("engine publishes input");
            input.set_key(17, true);
            input.set_pointer_position([4.0, 5.0]);
            input.add_wheel_delta(-1.0);
        }

        engine.run_frame(1.0 / 60.0);

        assert_eq!(
            captured.lock().expect("capture lock").as_slice(),
            &[Captured {
                key_down: true,
                button_down: false,
                pointer_position: [4.0, 5.0],
                pointer_delta: [4.0, 5.0],
                wheel_delta: -1.0,
            }]
        );
        let input = engine
            .world()
            .resources()
            .get::<InputState>()
            .expect("input resource");
        assert_eq!(input.pointer_delta(), [0.0, 0.0]);
        assert_eq!(input.wheel_delta(), 0.0);
        assert!(input.key_down(17));
    }
}
