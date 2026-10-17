//! Identity of the reference encoder binary, mixed into the bake cache key.
//!
//! The identity is the trimmed `-version` stdout plus `bin:` and an FNV-1a
//! digest of the file. Version text alone collides: two builds that both
//! print `basisu 2.10.0` (different commits, or SSE on versus off) would
//! share a cache entry and could replay the wrong container. The digest
//! separates those builds. It is a cache key, not a security hash — the same
//! FNV-1a the entry key uses. When `-version` fails, the digest is the whole
//! identity. When the file cannot be read after a successful `-version`, the
//! version text is used alone.
//!
//! The result is remembered for the process, keyed by the canonical path,
//! mtime and length, so a replaced tool misses the previous entry without
//! re-reading the file on every bake. `-version` runs once per stamp; a
//! failure is not probed a second time just to classify the error.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, SystemTime};

use super::BakeError;
use super::cache::fnv_hex;

/// Argument the tool prints its version for.
const VERSION_FLAG: &str = "-version";

/// Identity of one encoder binary, remembered for the process.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IdentityStamp {
    /// Canonical program path.
    path: PathBuf,
    /// `mtime` when the platform reports one.
    modified: Option<SystemTime>,
    /// File length, so a same-timestamp rewrite still misses.
    len: u64,
}

/// Versions already read, keyed by [`IdentityStamp`].
static IDENTITIES: LazyLock<Mutex<std::collections::HashMap<IdentityStamp, String>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

/// Tool identity: `-version` stdout plus a content hash of the binary.
///
/// # Errors
///
/// [`BakeError::TimedOut`] when `-version` exceeds `timeout`.
/// [`BakeError::Spawn`] when the tool cannot be started and its bytes cannot
/// be read. A tool that starts but yields no identity and cannot be hashed
/// is [`BakeError::Exit`].
pub(super) fn tool_identity(program: &Path, timeout: Duration) -> Result<String, BakeError> {
    if let Some(hit) = remembered(program) {
        return Ok(hit);
    }
    let identity = identity_of(program, timeout)?;
    remember(program, &identity);
    Ok(identity)
}

/// Version text and binary digest, without a second process spawn.
fn identity_of(program: &Path, timeout: Duration) -> Result<String, BakeError> {
    let version = match probe_version(program, timeout) {
        Ok(version) => version,
        Err(error @ BakeError::Spawn { .. }) => return file_hash(program).ok_or(error),
        Err(error) => return Err(error),
    };
    match (version, file_hash(program)) {
        (Some(text), Some(digest)) => Ok(format!("{text}\n{digest}")),
        (None, Some(digest)) => Ok(digest),
        (Some(text), None) => Ok(text),
        (None, None) => Err(no_identity(program)),
    }
}

/// Trimmed `-version` stdout, or `None` when the tool exits without one.
///
/// # Errors
///
/// [`BakeError::Spawn`] and [`BakeError::TimedOut`] from the process run.
fn probe_version(program: &Path, timeout: Duration) -> Result<Option<String>, BakeError> {
    let output = super::process::run_timed(program, &[OsString::from(VERSION_FLAG)], timeout)?;
    if !output.status.success() {
        return Ok(None);
    }
    Ok(nonempty_trim(&output.stdout))
}

/// Trimmed stdout, or `None` when it is empty.
fn nonempty_trim(bytes: &[u8]) -> Option<String> {
    let trimmed = String::from_utf8_lossy(bytes).trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// Reported when neither `-version` nor the file bytes yield an identity.
fn no_identity(program: &Path) -> BakeError {
    BakeError::Exit {
        program: program.display().to_string(),
        code: 1,
        stderr: "encoder -version produced no identity".to_string(),
    }
}

/// `bin:` plus the FNV-1a digest of the tool's bytes.
fn file_hash(program: &Path) -> Option<String> {
    let bytes = std::fs::read(program).ok()?;
    Some(format!("bin:{}", fnv_hex(&bytes)))
}

/// Cached identity for `program`, when its stamp still matches.
fn remembered(program: &Path) -> Option<String> {
    let stamp = identity_stamp(program)?;
    let guard = IDENTITIES
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.get(&stamp).cloned()
}

/// Stores `identity` under the current stamp of `program`.
fn remember(program: &Path, identity: &str) {
    let Some(stamp) = identity_stamp(program) else {
        return;
    };
    let mut guard = IDENTITIES
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.insert(stamp, identity.to_string());
}

/// Stamp of `program`, or `None` when the file cannot be stated.
fn identity_stamp(program: &Path) -> Option<IdentityStamp> {
    let meta = std::fs::metadata(program).ok()?;
    Some(IdentityStamp {
        path: program.to_path_buf(),
        modified: meta.modified().ok(),
        len: meta.len(),
    })
}
