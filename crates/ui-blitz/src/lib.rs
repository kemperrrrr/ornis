//! Ornis UI rendering via Blitz (Stylo + Taffy + Vello).

use anyrender::ImageRenderer;
use anyrender_vello::VelloImageRenderer;
use blitz_dom::DocumentConfig;
use blitz_html::HtmlDocument;
use blitz_paint::paint_scene;
use blitz_traits::shell::{ColorScheme, Viewport};
use std::path::Path;
use std::sync::Arc;

/// Render the Ornis editor HTML via Blitz and save as PNG.
pub fn render_to_png(
    html: &str,
    output_path: impl AsRef<Path>,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let net = Arc::new(blitz_net::Provider::new(None));

    let mut doc = HtmlDocument::from_html(
        html,
        DocumentConfig {
            base_url: Some("file:///Users/a0000/AI-Projects/ornis/crates/ui/assets/editor/".to_string()),
            net_provider: Some(Arc::clone(&net) as _),
            viewport: Some(Viewport::new(width, height, 1.0, ColorScheme::Dark)),
            ..Default::default()
        },
    );

    for _ in 0..500 {
        doc.resolve(0.0);
        if net.is_empty() {
            break;
        }
    }
    doc.as_mut().resolve(0.0);
    doc.as_mut().resolve_layout();

    let mut renderer = VelloImageRenderer::new(width, height);
    let mut buf = Vec::new();
    renderer.render_to_vec(
        |scene| {
            use anyrender::PaintScene;
            use vello::peniko::{kurbo::Rect, Color, Fill};
            scene.fill(
                Fill::NonZero,
                Default::default(),
                Color::BLACK,
                Default::default(),
                &Rect::new(0.0, 0.0, width as f64, height as f64),
            );
            paint_scene(scene, doc.as_mut(), 1.0, width, height, 0, 0);
        },
        &mut buf,
    );

    let mut file = std::fs::File::create(output_path.as_ref())
        .map_err(|e| format!("create output: {e}"))?;
    let mut encoder = png::Encoder::new(&mut file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("png header: {e}"))?;
    writer
        .write_image_data(&buf)
        .map_err(|e| format!("png data: {e}"))?;
    writer
        .finish()
        .map_err(|e| format!("png finish: {e}"))?;

    Ok(())
}
