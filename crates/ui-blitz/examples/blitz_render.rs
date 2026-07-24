/// Render Ornis editor HTML via Blitz (Stylo + Vello) to PNG.
///
/// Usage:
///   cargo run -p ornis-ui-blitz --example blitz_render -- <width> <height> <output.png>
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let width: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1920);
    let height: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1080);
    let output = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "/tmp/blitz_output.png".to_string());

    let css = include_str!("../../ui/assets/editor/index.css");
    let html_raw = include_str!("../../ui/assets/editor/index.html");
    let body_start = html_raw.find("<body").unwrap_or(0);
    let body_end = html_raw.find("</body>").unwrap_or(html_raw.len());
    let body_content = &html_raw[body_start..body_end];

    let inline = format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\"><style>{css}</style></head>\n{body_content}\n</body>\n</html>"
    );

    println!("Rendering {}x{} -> {}", width, height, output);

    // Create a tokio runtime that outlives the blitz net provider
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();

    match ornis_ui_blitz::render_to_png(&inline, &output, width, height) {
        Ok(()) => println!("Done: {}", output),
        Err(e) => eprintln!("Error: {e}"),
    }
}
