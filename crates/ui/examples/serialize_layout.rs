//! Dumps the Ornis UI layout engine's computed tree to JSON so it can be
//! diffed against a real browser via `browser_probe.py` + `diff.py`.
//!
//! ```sh
//! cargo run -p ornis-ui --features serialize --example serialize_layout -- 1280 800
//! # -> writes ui_layout.json
//! ```

use ornis_ui::css::Stylesheet;
use ornis_ui::editor_template::EditorTemplate;
use ornis_ui::html::parse_html;
use ornis_ui::layout::LayoutTree;
use ornis_ui::unified_editor::UnifiedEditorConfig;
use std::io::Read;
use vello::peniko::FontData;

fn load_font() -> FontData {
    let candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ];
    for c in candidates {
        if let Ok(mut f) = std::fs::File::open(c) {
            let mut buf = Vec::new();
            if f.read_to_end(&mut buf).is_ok() {
                return FontData::new(vello::peniko::Blob::from(buf), 0);
            }
        }
    }
    panic!("no system font found");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vw: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1280.0);
    let vh: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(800.0);

    let config = UnifiedEditorConfig::default();
    let html = EditorTemplate::generate_html_with_theme(&config);
    let doc = parse_html(&html);
    let css = EditorTemplate::generate_css_with_theme(&config);
    let sheet = Stylesheet::parse(&css).expect("css parses");
    let font = load_font();

    let tree =
        LayoutTree::build_with_viewport(&doc, &[sheet], vw, vh, &font).expect("layout builds");

    let json = tree.to_json();
    let out = serde_json::to_string_pretty(&json).expect("serialize");
    std::fs::write("ui_layout.json", &out).expect("write");
    println!(
        "wrote ui_layout.json  (viewport {}x{}, {} nodes)",
        vw as i32, vh as i32, json["node_count"]
    );
}
