//! Exports the generated editor HTML+CSS to files so Chromium can load it.
//!
//! ```sh
//! cargo run -p ornis-ui --example export_editor_html
//! # -> writes editor_test.html + editor_test.css
//! ```

use ornis_ui::editor_template::EditorTemplate;
use ornis_ui::unified_editor::UnifiedEditorConfig;
use std::fs;

fn main() {
    let config = UnifiedEditorConfig::default();
    let html = EditorTemplate::generate_html_with_theme(&config);
    let css = EditorTemplate::generate_css_with_theme(&config);

    // Embed CSS directly into HTML for simpler serving
    let full_html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{css}</style>
</head>
<body>
{html}
</body>
</html>"#,
        css = css,
        html = html,
    );

    fs::write("editor_test.html", &full_html).expect("write editor_test.html");
    println!("wrote editor_test.html  ({} bytes)", full_html.len());
}
