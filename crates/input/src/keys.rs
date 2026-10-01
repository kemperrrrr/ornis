//! Named keys and mouse buttons over the raw numeric wire codes.
//!
//! The engine intentionally keeps input codes as plain numbers
//! ([`crate::InputState`] stores `u32` key codes and `u8` mouse-button
//! codes) so the core never depends on a windowing or DOM crate. This
//! module gives those numbers stable names plus the alias sets the current
//! adapters emit.
//!
//! # Raw code table
//!
//! | Key | Raw `u32` aliases | Origin |
//! |---|---|---|
//! | `KeyCode::KeyW` | 17, 87 | legacy winit-physical `W` (evdev-style), ASCII `'W'` |
//! | `KeyCode::KeyA` | 30, 65 | legacy winit-physical `A`, ASCII `'A'` |
//! | `KeyCode::KeyS` | 31, 83 | legacy winit-physical `S`, ASCII `'S'` |
//! | `KeyCode::KeyD` | 32, 68 | legacy winit-physical `D`, ASCII `'D'` |
//! | `KeyCode::ArrowUp` | 38 | legacy DOM `keyCode` for `ArrowUp`, shared by both paths |
//! | `KeyCode::ArrowDown` | 40 | legacy DOM `keyCode` for `ArrowDown` |
//! | `KeyCode::ArrowLeft` | 37 | legacy DOM `keyCode` for `ArrowLeft` |
//! | `KeyCode::ArrowRight` | 39 | legacy DOM `keyCode` for `ArrowRight` |
//! | `KeyCode::Space` | 62, 32 | winit 0.30 physical `Space`, ASCII space (see collision note) |
//!
//! | Button | Raw `u8` | Origin |
//! |---|---|---|
//! | `MouseButton::Left` | 0 | winit `MouseButton::Left` / browser primary button |
//! | `MouseButton::Right` | 1 | winit `MouseButton::Right` |
//! | `MouseButton::Middle` | 2 | winit `MouseButton::Middle` |
//! | `MouseButton::Back` | 3 | winit `MouseButton::Back` |
//! | `MouseButton::Forward` | 4 | winit `MouseButton::Forward` |
//!
//! The "legacy winit-physical" codes (17/30/31/32) are the evdev-style
//! values the adapters and the browser `game_key_codes` table have always
//! emitted. They are intentionally *not* the `winit` 0.30
//! `KeyCode as u32` discriminants (verified: `KeyW = 41`, `KeyA = 19`,
//! `KeyS = 37`, `KeyD = 22` there — plain enum order). Adapters must keep
//! emitting the table above; do not switch producers to enum order.
//!
//! `Space` needs care: ASCII space (32) collides numerically with legacy
//! physical `D` (32) in the flat `u32` namespace, so matching 32 also
//! matches a held physical `D`. The default gameplay map therefore never
//! binds `Space`; bind it explicitly only when that aliasing is acceptable.

/// Semantic key covering every raw alias the adapters emit for it.
///
/// Use [`KeyCode::codes`] to expand a key into the raw codes producers
/// write, or [`crate::InputState::keycode_down`] to test a key on read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyCode {
    /// `W`: forward. Raw aliases 17 (legacy physical) and 87 (ASCII).
    KeyW,
    /// `A`: strafe left. Raw aliases 30 (legacy physical) and 65 (ASCII).
    KeyA,
    /// `S`: back. Raw aliases 31 (legacy physical) and 83 (ASCII).
    KeyS,
    /// `D`: strafe right. Raw aliases 32 (legacy physical) and 68 (ASCII).
    KeyD,
    /// Up arrow. Single raw alias 38 (legacy DOM `keyCode`).
    ArrowUp,
    /// Down arrow. Single raw alias 40 (legacy DOM `keyCode`).
    ArrowDown,
    /// Left arrow. Single raw alias 37 (legacy DOM `keyCode`).
    ArrowLeft,
    /// Right arrow. Single raw alias 39 (legacy DOM `keyCode`).
    ArrowRight,
    /// Space bar. Raw aliases 62 (winit 0.30 physical) and 32 (ASCII).
    ///
    /// ASCII 32 collides with legacy physical `D` (also 32); see the
    /// module-level table before binding this key.
    Space,
}

/// winit 0.30 physical scan code for W.
const WIN_SCAN_W: u32 = 17;
/// Legacy DOM `keyCode` for W.
const DOM_KEY_W: u32 = 87;
/// winit 0.30 physical scan code for A.
const WIN_SCAN_A: u32 = 30;
/// Legacy DOM `keyCode` for A.
const DOM_KEY_A: u32 = 65;
/// winit 0.30 physical scan code for S.
const WIN_SCAN_S: u32 = 31;
/// Legacy DOM `keyCode` for S.
const DOM_KEY_S: u32 = 83;
/// winit 0.30 physical scan code for D (also ASCII Space — see module docs).
const WIN_SCAN_D: u32 = 32;
/// Legacy DOM `keyCode` for D.
const DOM_KEY_D: u32 = 68;
/// Legacy DOM `keyCode` for ArrowUp.
const DOM_ARROW_UP: u32 = 38;
/// Legacy DOM `keyCode` for ArrowDown.
const DOM_ARROW_DOWN: u32 = 40;
/// Legacy DOM `keyCode` for ArrowLeft.
const DOM_ARROW_LEFT: u32 = 37;
/// Legacy DOM `keyCode` for ArrowRight.
const DOM_ARROW_RIGHT: u32 = 39;
/// winit 0.30 physical scan code for Space.
const WIN_SCAN_SPACE: u32 = 62;
/// ASCII / DOM code for Space (collides with [`WIN_SCAN_D`]).
const ASCII_SPACE: u32 = 32;
/// Raw wire code for mouse Back.
const MOUSE_BACK: u8 = 3;
/// Raw wire code for mouse Forward.
const MOUSE_FORWARD: u8 = 4;

impl KeyCode {
    /// All raw wire codes that count as this key, in canonical order.
    ///
    /// The first entry is the primary code reported by
    /// [`KeyCode::primary_code`].
    pub fn codes(self) -> &'static [u32] {
        match self {
            KeyCode::KeyW => &[WIN_SCAN_W, DOM_KEY_W],
            KeyCode::KeyA => &[WIN_SCAN_A, DOM_KEY_A],
            KeyCode::KeyS => &[WIN_SCAN_S, DOM_KEY_S],
            KeyCode::KeyD => &[WIN_SCAN_D, DOM_KEY_D],
            KeyCode::ArrowUp => &[DOM_ARROW_UP],
            KeyCode::ArrowDown => &[DOM_ARROW_DOWN],
            KeyCode::ArrowLeft => &[DOM_ARROW_LEFT],
            KeyCode::ArrowRight => &[DOM_ARROW_RIGHT],
            KeyCode::Space => &[WIN_SCAN_SPACE, ASCII_SPACE],
        }
    }

    /// Primary raw wire code for this key (first entry of
    /// [`KeyCode::codes`]).
    pub fn primary_code(self) -> u32 {
        self.codes()[0]
    }

    /// DOM-style short name of the key (`"KeyW"`, `"ArrowUp"`, `"Space"`).
    pub fn name(self) -> &'static str {
        match self {
            KeyCode::KeyW => "KeyW",
            KeyCode::KeyA => "KeyA",
            KeyCode::KeyS => "KeyS",
            KeyCode::KeyD => "KeyD",
            KeyCode::ArrowUp => "ArrowUp",
            KeyCode::ArrowDown => "ArrowDown",
            KeyCode::ArrowLeft => "ArrowLeft",
            KeyCode::ArrowRight => "ArrowRight",
            KeyCode::Space => "Space",
        }
    }

    /// Resolves a raw wire code to its named key.
    ///
    /// Returns `None` for codes no named key covers. Ambiguous code 32
    /// resolves to [`KeyCode::KeyD`] (canonical order: letters before
    /// `Space`).
    pub fn from_code(code: u32) -> Option<Self> {
        const ALL: [KeyCode; 9] = [
            KeyCode::KeyW,
            KeyCode::KeyA,
            KeyCode::KeyS,
            KeyCode::KeyD,
            KeyCode::ArrowUp,
            KeyCode::ArrowDown,
            KeyCode::ArrowLeft,
            KeyCode::ArrowRight,
            KeyCode::Space,
        ];
        ALL.into_iter().find(|key| key.codes().contains(&code))
    }
}

/// Named mouse button over the raw `u8` wire codes.
///
/// The discriminants mirror the native adapter mapping
/// (`Left = 0`, `Right = 1`, `Middle = 2`, `Back = 3`, `Forward = 4`);
/// higher codes from `MouseButton::Other` stay raw-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MouseButton {
    /// Primary (left) button, raw code 0.
    Left,
    /// Secondary (right) button, raw code 1.
    Right,
    /// Middle button, raw code 2.
    Middle,
    /// Back (fourth) button, raw code 3.
    Back,
    /// Forward (fifth) button, raw code 4.
    Forward,
}

impl MouseButton {
    /// Raw wire code for this button.
    pub fn code(self) -> u8 {
        match self {
            MouseButton::Left => 0,
            MouseButton::Right => 1,
            MouseButton::Middle => 2,
            MouseButton::Back => MOUSE_BACK,
            MouseButton::Forward => MOUSE_FORWARD,
        }
    }

    /// Resolves a raw wire code to its named button.
    ///
    /// Returns `None` for codes above `Forward` (native `Other` buttons
    /// stay raw-only).
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(MouseButton::Left),
            1 => Some(MouseButton::Right),
            2 => Some(MouseButton::Middle),
            3 => Some(MouseButton::Back),
            4 => Some(MouseButton::Forward),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_aliases_match_legacy_wire_codes() {
        assert_eq!(KeyCode::KeyW.codes(), &[17, 87]);
        assert_eq!(KeyCode::KeyA.codes(), &[30, 65]);
        assert_eq!(KeyCode::KeyS.codes(), &[31, 83]);
        assert_eq!(KeyCode::KeyD.codes(), &[32, 68]);
        assert_eq!(KeyCode::ArrowUp.codes(), &[38]);
        assert_eq!(KeyCode::ArrowDown.codes(), &[40]);
        assert_eq!(KeyCode::ArrowLeft.codes(), &[37]);
        assert_eq!(KeyCode::ArrowRight.codes(), &[39]);
        assert_eq!(KeyCode::Space.codes(), &[62, 32]);
    }

    #[test]
    fn primary_codes_are_canonical_first_aliases() {
        assert_eq!(KeyCode::KeyW.primary_code(), 17);
        assert_eq!(KeyCode::ArrowUp.primary_code(), 38);
        assert_eq!(KeyCode::Space.primary_code(), 62);
    }

    #[test]
    fn from_code_round_trips_and_flags_unknowns() {
        assert_eq!(KeyCode::from_code(17), Some(KeyCode::KeyW));
        assert_eq!(KeyCode::from_code(87), Some(KeyCode::KeyW));
        assert_eq!(KeyCode::from_code(38), Some(KeyCode::ArrowUp));
        assert_eq!(KeyCode::from_code(62), Some(KeyCode::Space));
        // Ambiguous 32 resolves to KeyD by canonical order.
        assert_eq!(KeyCode::from_code(32), Some(KeyCode::KeyD));
        assert_eq!(KeyCode::from_code(999), None);
    }

    #[test]
    fn mouse_codes_mirror_native_adapter_mapping() {
        assert_eq!(MouseButton::Left.code(), 0);
        assert_eq!(MouseButton::Right.code(), 1);
        assert_eq!(MouseButton::Middle.code(), 2);
        assert_eq!(MouseButton::Back.code(), 3);
        assert_eq!(MouseButton::Forward.code(), 4);
        assert_eq!(MouseButton::from_code(0), Some(MouseButton::Left));
        assert_eq!(MouseButton::from_code(4), Some(MouseButton::Forward));
        assert_eq!(MouseButton::from_code(5), None);
    }
}
