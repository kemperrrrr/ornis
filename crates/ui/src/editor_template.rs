//! Unified Editor Template
//!
//! Loads the shared HTML/CSS editor sources — the exact same files that are
//! served to the remote/website editor — so the in-game (Vello) editor and the
//! browser editor share one common source of truth.

pub use crate::unified_editor::UnifiedEditorConfig;

use std::path::PathBuf;

/// Editor template loader - reads the shared HTML/CSS editor sources.
pub struct EditorTemplate {
    config: UnifiedEditorConfig,
}

impl EditorTemplate {
    /// Create a new editor template with the given configuration
    pub fn new(config: UnifiedEditorConfig) -> Self {
        Self { config }
    }

    /// Generate the full HTML for the editor
    pub fn generate_html(&self) -> String {
        Self::generate_html_with_theme(&self.config)
    }

    /// Generate the CSS for the editor
    pub fn generate_css(&self) -> String {
        Self::generate_css_with_theme(&self.config)
    }

    /// Generate HTML from the shared editor source
    pub fn generate_html_with_theme(_config: &UnifiedEditorConfig) -> String {
        read_asset("index.html")
    }

    pub fn generate_css_with_theme(config: &UnifiedEditorConfig) -> String {
        let mut css = read_asset("index.css");
        // The editor's layout-critical rules live in inline `<style>` blocks
        // inside `index.html` (the external `index.css` only holds variables
        // and base styles). The browser uses both, so for the in-game editor
        // we must concatenate the inline styles too — otherwise `.app`,
        // `.header`, etc. have no rules and collapse into a vertical block
        // stack.
        let html = read_asset("index.html");
        for block in extract_style_blocks(&html) {
            css.push('\n');
            css.push_str(&block);
        }
        // Apply the (darker) theme `:root` overrides AFTER the asset rules so
        // the in-game editor matches the themed browser editor.
        css.push('\n');
        css.push_str(&crate::unified_editor::UnifiedEditor::theme_root_css(
            &config.theme,
        ));
        css
    }
}

/// Extracts the text content of every `<style>...</style>` block from an HTML
/// document (attributes on the opening tag are ignored).
fn extract_style_blocks(html: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut rest = html;
    while let Some(open) = rest.find("<style") {
        // Skip past the opening tag to its closing `>`.
        let after_open = match rest[open..].find('>') {
            Some(i) => open + i + 1,
            None => break,
        };
        let tail = &rest[after_open..];
        match tail.find("</style>") {
            Some(close_rel) => {
                blocks.push(tail[..close_rel].to_string());
                rest = &tail[close_rel + "</style>".len()..];
            }
            None => break,
        }
    }
    blocks
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/editor")
}

fn read_asset(name: &str) -> String {
    let path = assets_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| format!("<!-- failed to load {name}: {e} -->"))
}
