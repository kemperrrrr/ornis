//! Blitz-based UI rendering backend.
//!
//! Uses [Blitz](https://github.com/DioxusLabs/blitz) (Stylo + Taffy + Vello)
//! to parse HTML+CSS and render it to a GPU buffer.  Replaces our hand-rolled
//! CSS engine (`css.rs`+`layout.rs`+`paint.rs`) when the `blitz-backend` feature
//! is enabled.

use anyrender::{ImageRenderer, render_to_buffer};
use anyrender_vello::VelloImageRenderer;
use blitz_dom::DocumentConfig;
use blitz_html::HtmlDocument;
use blitz_paint::paint_scene;
use blitz_traits::shell::{ColorScheme, Viewport};
use std::sync::Arc;

/// Render an HTML document to an RGBA pixel buffer using Blitz (GPU-accelerated).
pub fn render_html(html: &str, width: u32, height: u32) -> Vec<u8> {
    let net = Arc::new(blitz_net::Provider::new(None));

    let mut doc = HtmlDocument::from_html(
        html,
        DocumentConfig {
            base_url: None,
            net_provider: Some(Arc::clone(&net) as _),
            viewport: Some(Viewport::new(width, height, 1.0, ColorScheme::Dark)),
            ..Default::default()
        },
    );

    let mut timeout = 500;
    loop {
        doc.resolve(0.0);
        if net.is_empty() || timeout == 0 {
            break;
        }
        timeout -= 1;
    }

    doc.as_mut().resolve(0.0);

    // GPU render via VelloImageRenderer (wgpu-based)
    let buffer = render_to_buffer::<VelloImageRenderer, _>(
        |scene| {
            use anyrender::PaintScene;
            use vello::peniko::{Fill, Color, kurbo::Rect};

            // Dark background
            scene.fill(
                Fill::NonZero,
                Default::default(),
                Color::BLACK,
                Default::default(),
                &Rect::new(0.0, 0.0, width as f64, height as f64),
            );
            // 7-arg paint_scene (0.3.0-beta.1 API) — scene, doc, scale, w, h, offset_x, offset_y
            paint_scene(scene, doc.as_mut(), 1.0, width, height, 0, 0);
        },
        width,
        height,
    );

    buffer
}
