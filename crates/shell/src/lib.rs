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

pub mod assets;
pub mod bridge;
pub mod error;
pub mod probe;

pub use assets::{
    AssetResponse, content_type_for, read_asset, request_file_path, resolve_editor_dir, start_url,
};
pub use bridge::{
    AckOutcome, IncomingPost, ack_json, ack_message_json, allocate_request_id, dispatch_script,
    event_json, events_message_json, init_script, parse_incoming,
};
pub use error::{AssetError, BridgeError};
pub use probe::probe_page;
