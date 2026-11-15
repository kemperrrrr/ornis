//! `ornis://` responses: the shell's asset handler as a tested function.
//!
//! The webview's custom-protocol handler is a closure `wry` calls on the main
//! thread, so the status/header decision it makes is the one part of the shell
//! that used to live only in the binary — untestable without a window. It is
//! pure enough to belong in the library: given a root and a URL path, answer
//! with a [`Response`].
//!
//! Status vocabulary matches the HTTP transport in
//! `editor_backend::remote`: missing and escaping paths both answer 404 (the
//! server deliberately does not distinguish them either), unreadable files 500.
//! That is the security-relevant behaviour, so it is pinned here rather than in
//! a manual test.

use std::borrow::Cow;
use std::path::Path;

use wry::http::{Response, StatusCode};

use crate::assets::read_asset;
use crate::error::AssetError;

/// HTTP status for a served asset.
pub const STATUS_OK: u16 = 200;
/// HTTP status for a missing or escaping path.
pub const STATUS_NOT_FOUND: u16 = 404;
/// HTTP status for a file that exists but cannot be read.
pub const STATUS_SERVER_ERROR: u16 = 500;

/// Content type for plain-text responses (status bodies, fallbacks).
const TEXT_PLAIN: &str = "text/plain; charset=utf-8";

/// Serves one `ornis://app/…` request from `root`.
///
/// A missing file and a path escaping `root` both answer 404: the editor
/// frontend only ever asks for files it shipped, so telling the two apart would
/// leak the directory layout for free. An existing-but-unreadable file answers
/// 500, which the page logs instead of silently rendering a blank panel.
///
/// `Access-Control-Allow-Origin: *` is required: the page runs on the custom
/// scheme and fetches `pkg/*.wasm` and module scripts, which WebKit treats as
/// cross-origin against that scheme.
pub fn serve_response(root: &Path, url_path: &str) -> Response<Cow<'static, [u8]>> {
    match read_asset(root, url_path) {
        Ok(asset) => response(STATUS_OK, asset.content_type, asset.bytes),
        Err(AssetError::NotFound { .. } | AssetError::Forbidden { .. }) => {
            response(STATUS_NOT_FOUND, TEXT_PLAIN, b"404 Not Found".to_vec())
        }
        Err(AssetError::Io { .. }) => response(
            STATUS_SERVER_ERROR,
            TEXT_PLAIN,
            b"500 Cannot Read Asset".to_vec(),
        ),
    }
}

/// Builds one response, degrading without panicking.
///
/// `http::Response::builder` only rejects a malformed header value or status.
/// Every value here is a compile-time constant from this module, so a failure
/// would be a bug — but the handler runs on the webview's main thread, where a
/// panic takes the whole shell down for a cosmetic problem. A bare
/// [`Response::new`] (no headers at all) is always valid and still answers the
/// page, which is the honest degradation.
fn response(
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
) -> Response<Cow<'static, [u8]>> {
    Response::builder()
        .status(status)
        .header("Content-Type", content_type)
        .header("Access-Control-Allow-Origin", "*")
        .body(Cow::Owned(body))
        .unwrap_or_else(|_| {
            let mut fallback = Response::new(Cow::Borrowed(b"internal error".as_slice()));
            *fallback.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
            fallback
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_root(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ornis-serve-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("pkg")).expect("fixture root");
        std::fs::write(dir.join("index.html"), b"<html>editor</html>").expect("fixture page");
        std::fs::write(dir.join("pkg/app.js"), b"export const x = 1;").expect("fixture script");
        dir
    }

    #[test]
    fn served_files_carry_their_content_type_and_cors_header() {
        let root = fixture_root("ok");
        let page = serve_response(&root, "/index.html");
        assert_eq!(page.status(), STATUS_OK);
        assert!(
            page.headers()["Content-Type"]
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
        assert_eq!(page.headers()["Access-Control-Allow-Origin"], "*");
        assert_eq!(page.body().as_ref(), b"<html>editor</html>");

        // A root request maps to the start page, like the HTTP server.
        let index = serve_response(&root, "/");
        assert_eq!(index.status(), STATUS_OK);
        assert!(
            index.headers()["Content-Type"]
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );

        let script = serve_response(&root, "/pkg/app.js?v=2");
        assert_eq!(script.status(), STATUS_OK);
        assert!(
            script.headers()["Content-Type"]
                .to_str()
                .unwrap()
                .starts_with("application/javascript")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_and_escaping_paths_both_answer_404() {
        let root = fixture_root("miss");
        for path in ["/nope.js", "/../etc/passwd", "/pkg/../../secret.ron"] {
            let response = serve_response(&root, path);
            assert_eq!(response.status(), STATUS_NOT_FOUND, "path {path}");
            assert_eq!(response.body().as_ref(), b"404 Not Found");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_unreadable_file_answers_500() {
        use std::os::unix::fs::PermissionsExt;

        let root = fixture_root("io");
        let locked = root.join("locked.bin");
        std::fs::write(&locked, b"secret").expect("fixture file");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000))
            .expect("drop read permission");
        // A directory request maps to the start page (a deliberate alias), so
        // the unreadable-file branch needs a real file with no read bit.
        let response = serve_response(&root, "/locked.bin");
        assert_eq!(response.status(), STATUS_SERVER_ERROR);
        assert_eq!(response.body().as_ref(), b"500 Cannot Read Asset");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644))
            .expect("restore permission");
        let _ = std::fs::remove_dir_all(&root);
    }
}
