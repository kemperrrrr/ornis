#![warn(missing_docs)]
//! Editor backend for Ornis: the HTTP server that bridges the browser
//! editor (`editor/`) and the engine's ECS world.
//!
//! This crate holds the editor↔engine IPC protocol (`ipc`) and the remote
//! HTTP server (`remote`) so both can be unit- and integration-tested as a
//! normal library (the `ornis` binary depends on it).

pub mod ipc;
/// HTTP server bridging the browser editor and the engine world.
pub mod remote;

// Re-export the common protocol types at the crate root for convenience.
pub use ipc::{GameConnection, GameEvent, IpcChannel, UiCommand};
pub use remote::RemoteEditor;
