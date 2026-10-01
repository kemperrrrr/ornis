//! Editor assets behind the `ornis://app/…` custom scheme.
//!
//! Resolution order mirrors `editor_backend::remote::assets_root` (CLI
//! `--editor-dir`, then `ORNIS_EDITOR_DIR`, then `<workspace>/editor`), so
//! the shell loads exactly the frontend the browser mode would serve —
//! both `editor/` and the future `editor-v2/` mockup, selected the same
//! way. The difference is delivery: files are read from disk by the custom
//! protocol handler instead of over loopback HTTP, and no localhost server
//! is started in embedded mode.
//!
//! All mapping here is pure functions over explicit inputs (no ambient
//! `std::env::args` reads), so scheme/path handling is unit-tested without
//! a window. The binary feeds the real CLI/env values in.

use std::path::{Path, PathBuf};

/// Custom scheme the shell serves the editor frontend from.
pub const SCHEME: &str = "ornis";
/// Host/path prefix the editor app lives under (`ornis://app/…`).
pub const APP_HOST: &str = "app";
/// First page loaded into the webview.
pub const START_PAGE: &str = "index.html";

/// URL the webview opens at startup.
///
/// # Examples
///
/// ```
/// assert_eq!(ornis_shell::start_url(), "ornis://app/index.html");
/// ```
pub fn start_url() -> String {
    format!("{SCHEME}://{APP_HOST}/{START_PAGE}")
}

/// Resolves the editor directory with the same precedence as the HTTP
/// transport (`editor_backend::remote::assets_root`):
///
/// 1. `cli_dir` — the `--editor-dir <path>` CLI argument, when present;
/// 2. `env_dir` — the `ORNIS_EDITOR_DIR` variable, when non-empty;
/// 3. `<manifest_dir>/../../editor` — the workspace checkout default (this
///    crate lives at `crates/shell`, two levels below the root).
///
/// The caller passes the already-read CLI/env values plus
/// `CARGO_MANIFEST_DIR`; nothing here touches ambient process state.
pub fn resolve_editor_dir(
    cli_dir: Option<&str>,
    env_dir: Option<&str>,
    manifest_dir: &Path,
) -> PathBuf {
    if let Some(dir) = cli_dir {
        return PathBuf::from(dir);
    }
    if let Some(dir) = env_dir
        && !dir.is_empty()
    {
        return PathBuf::from(dir);
    }
    normalize_lexically(&manifest_dir.join("../../editor"))
}

/// Collapse `.`/`..`/duplicate separators without touching the filesystem.
///
/// The default editor root is built by joining `../../editor` onto
/// `CARGO_MANIFEST_DIR`; without normalization every log line and test
/// assertion carries the raw `crates/shell/../../editor` segments even
/// though the OS resolves them fine at open time.
fn normalize_lexically(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Maps a custom-scheme URL path to a file relative to the editor root.
///
/// Strips the query string, treats an empty path or a trailing `/` as
/// `index.html` (same convention as the HTTP static server), and returns
/// `None` for anything that would escape the root (`..` segments, absolute
/// paths) — the caller reports [`crate::error::AssetError::Forbidden`].
pub fn request_file_path(url_path: &str) -> Option<PathBuf> {
    let path = url_path.split('?').next().unwrap_or(url_path);
    let rel = path.trim_start_matches('/');
    if rel.contains("..") {
        return None;
    }
    if rel.is_empty() || rel.ends_with('/') {
        return Some(PathBuf::from(START_PAGE));
    }
    Some(PathBuf::from(rel))
}

/// Served asset: file bytes plus their MIME type for the protocol response.
#[derive(Debug, Clone)]
pub struct AssetResponse {
    /// Raw file bytes.
    pub bytes: Vec<u8>,
    /// Value for the `Content-Type` header.
    pub content_type: &'static str,
}

/// Reads one editor asset from disk.
///
/// Directory requests resolve to `index.html`; see [`request_file_path`]
/// for the traversal rules.
///
/// # Errors
///
/// [`crate::error::AssetError::Forbidden`] for escaping paths,
/// [`crate::error::AssetError::NotFound`] for missing files,
/// [`crate::error::AssetError::Io`] when the file cannot be read.
pub fn read_asset(root: &Path, url_path: &str) -> Result<AssetResponse, crate::error::AssetError> {
    use crate::error::AssetError;

    let Some(rel) = request_file_path(url_path) else {
        return Err(AssetError::Forbidden {
            path: url_path.to_owned(),
        });
    };
    let mut full = root.join(&rel);
    if full.is_dir() {
        full = root.join(START_PAGE);
    }
    let bytes = std::fs::read(&full).map_err(|e| {
        if full.exists() {
            AssetError::Io {
                path: url_path.to_owned(),
                reason: e.to_string(),
            }
        } else {
            AssetError::NotFound {
                path: url_path.to_owned(),
            }
        }
    })?;
    Ok(AssetResponse {
        content_type: content_type_for(&full),
        bytes,
    })
}

/// MIME type for an asset path, by extension.
///
/// Covers the editor frontend (HTML/CSS/JS/WASM/JSON/fonts/images); unknown
/// extensions fall back to `application/octet-stream`, as on the HTTP side.
pub fn content_type_for(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js" | "mjs") => "application/javascript; charset=utf-8",
        Some("json" | "map") => "application/json; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("gif") => "image/gif",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("ttf") => "font/ttf",
        Some("ron") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_dir_cli_arg_wins_over_env_and_default() {
        let manifest = Path::new("/repo/crates/shell");
        assert_eq!(
            resolve_editor_dir(Some("/v1"), Some("/v2"), manifest),
            PathBuf::from("/v1")
        );
        assert_eq!(
            resolve_editor_dir(None, Some("/v2"), manifest),
            PathBuf::from("/v2")
        );
        // An empty env value falls through to the workspace default.
        assert_eq!(
            resolve_editor_dir(None, Some(""), manifest),
            PathBuf::from("/repo/editor")
        );
        assert_eq!(
            resolve_editor_dir(None, None, manifest),
            PathBuf::from("/repo/editor")
        );
    }

    #[test]
    fn request_paths_map_to_index_and_reject_traversal() {
        assert_eq!(
            request_file_path("/index.html"),
            Some(PathBuf::from("index.html"))
        );
        assert_eq!(
            request_file_path("/pkg/editor.js?v=4.0"),
            Some(PathBuf::from("pkg/editor.js"))
        );
        assert_eq!(request_file_path(""), Some(PathBuf::from("index.html")));
        assert_eq!(request_file_path("/"), Some(PathBuf::from("index.html")));
        assert_eq!(
            request_file_path("/sub/"),
            Some(PathBuf::from("index.html"))
        );
        assert_eq!(request_file_path("/../etc/passwd"), None);
        assert_eq!(request_file_path("/a/../../secret.ron"), None);
    }

    #[test]
    fn content_types_cover_the_editor_frontend() {
        let mime = |name: &str| content_type_for(Path::new(name));
        assert!(mime("index.html").starts_with("text/html"));
        assert!(mime("editor.css").starts_with("text/css"));
        assert!(mime("editor.js").starts_with("application/javascript"));
        assert!(mime("chunk.mjs").starts_with("application/javascript"));
        assert_eq!(mime("viewport.wasm"), "application/wasm");
        assert!(mime("scene.json").starts_with("application/json"));
        assert!(mime("icons.svg").starts_with("image/svg+xml"));
        assert!(mime("gabe.png").starts_with("image/png"));
        assert!(mime("font.woff2").starts_with("font/woff2"));
        assert_eq!(mime("blob.bin"), "application/octet-stream");
    }

    #[test]
    fn read_asset_serves_files_and_blocks_escape() {
        let dir = std::env::temp_dir().join(format!("ornis-shell-assets-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("pkg")).expect("fixture root");
        std::fs::write(dir.join("index.html"), b"<html></html>").expect("fixture page");
        std::fs::write(dir.join("pkg/app.wasm"), b"\0asm").expect("fixture wasm");

        let page = read_asset(&dir, "/index.html").expect("page serves");
        assert_eq!(page.bytes, b"<html></html>");
        assert!(page.content_type.starts_with("text/html"));

        let root = read_asset(&dir, "/").expect("root serves index");
        assert_eq!(root.bytes, b"<html></html>");

        let wasm = read_asset(&dir, "/pkg/app.wasm").expect("wasm serves");
        assert_eq!(wasm.content_type, "application/wasm");

        let missing = read_asset(&dir, "/nope.js").unwrap_err();
        assert!(matches!(missing, crate::error::AssetError::NotFound { .. }));
        let escape = read_asset(&dir, "/../etc/passwd").unwrap_err();
        assert!(matches!(escape, crate::error::AssetError::Forbidden { .. }));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
