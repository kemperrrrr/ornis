//! Identity of the reference encoder binary, mixed into the bake cache key.
//!
//! A successful `-version` with non-empty stdout wins. Otherwise the key uses
//! a content hash of the binary. The result is remembered for the process,
//! keyed by the canonical path, mtime and length, so a replaced tool misses
//! the previous entry without re-reading the file on every bake.

use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::SystemTime;

use super::BakeError;
use super::cache::{fnv_hex, io_error};

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

/// Tool identity: `-version` stdout, else a content hash of the binary.
///
/// # Errors
///
/// [`BakeError::Spawn`] when the tool cannot be started and its bytes cannot
/// be read. A tool that starts but yields no identity is [`BakeError::Exit`],
/// or an I/O error when the content-hash fallback cannot read the file.
pub(super) fn tool_identity(program: &Path) -> Result<String, BakeError> {
    if let Some(hit) = remembered(program) {
        return Ok(hit);
    }
    let identity = version_stdout(program)
        .or_else(|| file_hash(program))
        .ok_or_else(|| identity_failure(program))?;
    remember(program, &identity);
    Ok(identity)
}

/// Trimmed `-version` stdout, when the tool exits successfully with some.
fn version_stdout(program: &Path) -> Option<String> {
    let output = std::process::Command::new(program)
        .arg(VERSION_FLAG)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// `bin:` plus the FNV-1a digest of the tool's bytes.
fn file_hash(program: &Path) -> Option<String> {
    let bytes = std::fs::read(program).ok()?;
    Some(format!("bin:{}", fnv_hex(&bytes)))
}

/// Failure reported when neither `-version` nor the content hash works.
fn identity_failure(program: &Path) -> BakeError {
    match std::process::Command::new(program)
        .arg(VERSION_FLAG)
        .output()
    {
        Err(source) => BakeError::Spawn {
            program: program.display().to_string(),
            source,
        },
        Ok(_) => match std::fs::read(program) {
            Err(source) => io_error(program, source),
            Ok(_) => BakeError::Exit {
                program: program.display().to_string(),
                code: 1,
                stderr: "encoder -version produced no identity".to_string(),
            },
        },
    }
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
