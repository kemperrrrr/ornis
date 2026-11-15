#![warn(missing_docs)]
//! Desktop shell for the Ornis editor.
//!
//! Production mode next to `editor-only`: instead of serving the editor
//! frontend over loopback HTTP (`RemoteEditor` on 127.0.0.1:3420), this
//! crate hosts it in a native OS window ([`wry::WebView`]) loaded from a
//! custom `ornis://app/…` scheme mapped onto the editor directory on disk.
//! No localhost server is started in embedded mode.
//!
//! The IPC bridge mirrors the `/api` semantics of
//! `editor_backend::remote` without touching that transport: page commands
//! arrive as `window.ipc.postMessage(JSON)` envelopes
//! (`{"type", "request_id"?, "data"}`), are queued as
//! [`UiCommand::WithRequestId`](editor_backend::ipc::UiCommand::WithRequestId),
//! and receive an explicit `accepted` acknowledgement; engine
//! [`GameEvent`](editor_backend::ipc::GameEvent)s flow back with a transport
//! `sequence` cursor. All of that is pure functions in [`bridge`], unit
//! tested without opening a window; the `wry` wiring lives in the binary.
//!
//! The shell-level channel is [`viewport_ipc`]: the page reports its
//! viewport rectangle (`viewport_bounds`) and forwards pointer/wheel/key
//! state over it (`viewport_input`). Both are pure parses here, so the native
//! viewport prototype (`--native-viewport`) shares the exact protocol the
//! browser page already speaks and CI tests it without a GPU.

pub mod assets;
pub mod bridge;
pub mod error;
pub mod native_viewport;
pub mod notice;
pub mod options;
pub mod probe;
pub mod serve;
pub mod viewport_ipc;

pub use assets::{
    AssetResponse, content_type_for, read_asset, request_file_path, resolve_editor_dir, start_url,
};
pub use bridge::{
    AckOutcome, IncomingPost, ack_json, ack_message_json, allocate_request_id, dispatch_script,
    event_json, events_message_json, init_script, init_script_for, next_event_sequence,
    parse_incoming,
};
pub use error::{AssetError, BridgeError};
pub use native_viewport::{ViewFrame, ViewportPlacement};
pub use notice::{ShellCapabilities, WEBGPU_UNAVAILABLE_NOTICE};
pub use options::{OptionAction, ParseOutcome, ShellOptions, USAGE, parse_args};
pub use probe::probe_page;
pub use serve::{STATUS_NOT_FOUND, STATUS_OK, STATUS_SERVER_ERROR, serve_response};
pub use viewport_ipc::{
    MAX_PRESSED, MAX_SCALE, MAX_VIEWPORT_EDGE, MIN_SCALE, MIN_VIEWPORT_EDGE, PhysicalRect,
    SCENE_EVENT_TAG, ViewportBounds, ViewportInput, ViewportMessage, WINDOW_HEIGHT, WINDOW_WIDTH,
    full_window_bounds, parse_viewport_message, scene_snapshot_json,
};
