//! Standalone binary that renders the Ornis editor HTML via Blitz (Stylo + Vello).
//!
//! Usage:
//!   cargo run --bin ornis-blitz --no-default-features -F blitz -- [width] [height] [output.png]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let width: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1920);
    let height: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1080);
    let output = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "/tmp/blitz_ornis.png".to_string());

    // Build full HTML with inlined CSS (Blitz doesn't load <link> resources)
    let css = include_str!("../crates/ui/assets/editor/index.css");
    let html_raw = include_str!("../crates/ui/assets/editor/index.html");
    let body_start = html_raw.find("<body").unwrap_or(0);
    let body_end = html_raw.find("</body>").unwrap_or(html_raw.len());
    let body_content = &html_raw[body_start..body_end];

    let inline = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>{css}</style>
</head>
{body_content}
</body>
</html>"#
    );

    println!("Blitz renderer: {}x{} -> {}", width, height, output);
    match ornis_ui_blitz::render_to_png(&inline, &output, width, height) {
        Ok(()) => println!("✅ Saved: {}", output),
        Err(e) => eprintln!("❌ Error: {e}"),
    }
}
