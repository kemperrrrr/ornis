//! What the shell tells the page about the viewport it cannot render.
//!
//! The embedded WKWebView has no `navigator.gpu` (PLAN.md §j, spike ⓪), so
//! the WASM viewport cannot boot there. The page owns the visible result —
//! it swaps the red failure canvas for a written explanation — but the shell
//! owns the wording, because only the shell knows *which* environment the
//! page is running in (embedded webview vs Chrome over `cargo xtask editor`).
//!
//! The text travels in the initialization script (`window.ornisShell`), so
//! one Rust constant is the single source of truth and the page can never
//! drift into a stale instruction. [`init_script`] embeds it; the tests below
//! pin the wording and the page's use of it without a window or a GPU.
//!
//! In native-viewport mode the message is not used: the shell renders the
//! scene itself, so there is nothing to explain.

/// Message shown in place of the viewport when WebGPU is unavailable in the
/// embedded webview.
///
/// Deliberately names the working alternative: a dead end with no way
/// forward is what the bare red canvas already was. Kept short — it is
/// rendered inside the viewport box, not in a dialog.
pub const WEBGPU_UNAVAILABLE_NOTICE: &str = "WebGPU недоступен в WKWebView — живой вьюпорт \
     открывается в Chrome через `cargo xtask editor`";

/// Shell marker the page reads to pick its rendering path.
///
/// `native_viewport` is `true` when the shell renders the scene itself in a
/// native child view under a transparent webview region (PLAN.md §j, вариант
/// 1): the page then reports its viewport rectangle and forwards input
/// instead of starting the WASM renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShellCapabilities {
    /// The shell renders the viewport natively; the page only lays it out.
    pub native_viewport: bool,
}

impl ShellCapabilities {
    /// Plain embedded webview, no native viewport (today's default).
    pub const WEBVIEW_ONLY: Self = Self {
        native_viewport: false,
    };

    /// Embedded webview with a native wgpu viewport under it.
    pub const NATIVE_VIEWPORT: Self = Self {
        native_viewport: true,
    };

    /// JS object literal spliced into the initialization script.
    ///
    /// Hand-built rather than `serde_json` so the notice text is escaped by
    /// the same single place that owns it; the result is a JS expression, so
    /// it must stay free of literal newlines and unescaped quotes (pinned by
    /// the tests).
    pub(crate) fn marker_js(self) -> String {
        format!(
            "{{scheme:\"ornis\",api:1,nativeViewport:{},webgpuNotice:{}}}",
            self.native_viewport,
            json_string(WEBGPU_UNAVAILABLE_NOTICE)
        )
    }
}

/// Double-quoted JSON string literal for `value`.
///
/// Only the escapes a JS/JSON string needs are emitted; the notice text is
/// static and short, so this stays a two-branch replacement instead of a
/// serializer dependency in a crate that otherwise needs none.
fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notice_points_at_a_working_alternative() {
        assert!(
            WEBGPU_UNAVAILABLE_NOTICE.contains("WKWebView"),
            "must name the environment that lacks WebGPU"
        );
        assert!(
            WEBGPU_UNAVAILABLE_NOTICE.contains("WebGPU"),
            "must name the missing capability"
        );
        assert!(
            WEBGPU_UNAVAILABLE_NOTICE.contains("cargo xtask editor"),
            "must name the command that does work"
        );
        assert!(
            WEBGPU_UNAVAILABLE_NOTICE.contains("Chrome"),
            "must name the browser that does work"
        );
        // Short enough to sit inside a viewport box without scrolling.
        assert!(WEBGPU_UNAVAILABLE_NOTICE.chars().count() <= 160);
        assert!(!WEBGPU_UNAVAILABLE_NOTICE.contains('\n'));
    }

    #[test]
    fn marker_is_a_js_object_literal_with_the_notice() {
        let js = ShellCapabilities::WEBVIEW_ONLY.marker_js();
        assert!(js.starts_with('{') && js.ends_with('}'), "{js}");
        assert!(js.contains("nativeViewport:false"), "{js}");
        assert!(js.contains("\"ornis\""), "{js}");
        assert!(
            js.contains(&json_string(WEBGPU_UNAVAILABLE_NOTICE)),
            "notice must be embedded verbatim as a JS string: {js}"
        );
        // No raw newline can leak into the script (it would break the literal).
        assert!(!js.contains('\n'));
        assert!(
            ShellCapabilities::NATIVE_VIEWPORT
                .marker_js()
                .contains("nativeViewport:true")
        );
    }

    #[test]
    fn json_string_escapes_the_characters_js_needs() {
        assert_eq!(json_string("plain"), "\"plain\"");
        assert_eq!(json_string("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_string("a\\b"), "\"a\\\\b\"");
        assert_eq!(json_string("a\nb\tc\rd"), "\"a\\nb\\tc\\rd\"");
        assert_eq!(json_string("\u{1}"), "\"\\u0001\"");
        // Cyrillic stays literal: the marker is injected as UTF-8 JS source.
        assert_eq!(json_string("вьюпорт"), "\"вьюпорт\"");
    }

    #[test]
    fn the_page_reads_the_notice_and_the_native_flag_from_the_marker() {
        // The wording above is only useful if the embedded page consumes it;
        // this is the contract edge between the Rust side and editor-v2.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../editor-v2");
        let viewport = std::fs::read_to_string(root.join("viewport.js"))
            .expect("editor-v2/viewport.js must exist next to the shell crate");
        assert!(
            viewport.contains("webgpuNotice"),
            "viewport.js must show window.ornisShell.webgpuNotice when WebGPU is missing"
        );
        assert!(
            viewport.contains("nativeViewport"),
            "viewport.js must branch on window.ornisShell.nativeViewport"
        );
        let index = std::fs::read_to_string(root.join("index.html")).expect("editor-v2 index");
        assert!(
            index.contains("id=\"viewport-notice\""),
            "index.html must carry the #viewport-notice element the fallback paints into"
        );
    }
}
