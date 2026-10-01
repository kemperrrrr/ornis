//! Typed failures of the shell bridge and asset serving.
//!
//! Every fallible shell operation reports one of these enums — never a bare
//! string or `bool` — so hosts can match on outcomes and tests can assert on
//! variants.

use thiserror::Error;

/// Why an inbound `window.ipc.postMessage` payload was rejected.
///
/// Mirrors the `/api/command` acknowledgement semantics of
/// `editor_backend::remote`: malformed input and a disconnected engine
/// channel are distinct, typed outcomes; only a successfully queued command
/// reports `accepted: true` (see [`crate::bridge::AckOutcome`]).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BridgeError {
    /// The payload is not valid JSON.
    #[error("request body is not valid JSON")]
    InvalidJson,
    /// The JSON is valid but has no string `type` field.
    #[error("command type must be a string")]
    BadEnvelope,
    /// The `type`/`data` pair does not build a [`UiCommand`](editor_backend::ipc::UiCommand).
    #[error("invalid command data for type '{command}'")]
    InvalidCommand {
        /// The `type` tag that failed to parse.
        command: String,
    },
    /// The engine command channel is disconnected.
    #[error("engine command channel is disconnected")]
    Disconnected,
}

/// Why a custom-scheme asset request failed.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AssetError {
    /// The URL path escapes the editor root (`..`, absolute path) or is
    /// otherwise not mappable to a file inside it.
    #[error("asset path escapes the editor root: {path}")]
    Forbidden {
        /// The raw URL path that was rejected.
        path: String,
    },
    /// No such file under the editor root.
    #[error("asset not found: {path}")]
    NotFound {
        /// The raw URL path that was requested.
        path: String,
    },
    /// The file could not be read.
    #[error("cannot read asset {path}: {reason}")]
    Io {
        /// The raw URL path that failed.
        path: String,
        /// The underlying I/O cause.
        reason: String,
    },
}
