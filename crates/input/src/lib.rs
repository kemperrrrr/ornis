//! Backend-neutral per-frame input: raw state, named keys and action mapping.
//!
//! Platform adapters (winit, browser events and future integrations) write
//! raw numeric codes into [`InputState`] between frame calls; gameplay reads
//! through named [`KeyCode`]/[`MouseButton`] aliases or through an
//! [`InputMap`] action binding. The engine frame loop is unchanged: held
//! keys/buttons persist across frames while pointer and wheel deltas are
//! cleared after each frame.
//!
//! # Wire stability
//!
//! The on-the-wire snapshot (`pressed_keys: Vec<u32>`,
//! `pressed_mouse_buttons: Vec<u8>` over WS / `POST /api/input` and the
//! editor-backend IPC) is unchanged. Mapping always happens on the read
//! edge: producers keep writing raw codes, consumers query
//! [`InputMap::action_down`] or [`InputState::keycode_down`].
//!
//! The raw code table lives on [`KeyCode`]; each variant lists every alias
//! the current adapters emit for that key.
#![warn(missing_docs)]

mod keys;
mod map;
mod state;

pub use keys::{KeyCode, MouseButton};
pub use map::{ActionId, GameAction, InputBinding, InputMap};
pub use state::InputState;
