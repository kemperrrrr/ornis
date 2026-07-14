use crate::css;
use crate::layout::{LayoutNodeId, LayoutTree};
use crate::render::UIRenderer;
use vello::peniko::{Color, FontData};

pub fn paint_layout(tree: &LayoutTree, renderer: &mut UIRenderer, font: &FontData) {
    paint_node(tree, tree.root, renderer, font);
}

/// Walks up the ancestor chain to resolve an inherited property
/// (e.g. `color`, `font-size`) for a node that has no own declaration.
fn resolve_inherited<'a>(tree: &'a LayoutTree, id: LayoutNodeId, prop: &str) -> Option<&'a String> {
    let mut cur = Some(id);
    while let Some(c) = cur {
        let node = &tree.arena[c];
        if let Some(v) = node.styles.get(prop) {
            return Some(v);
        }
        cur = node.parent;
    }
    None
}

/// Effective opacity for a node = product of `opacity` on the node and all
/// its ancestors (CSS `opacity` compounds down the tree). Values outside
/// `[0,1]` are clamped. Missing/unparseable `opacity` is treated as `1.0`.
fn effective_opacity(tree: &LayoutTree, id: LayoutNodeId) -> f32 {
    let mut acc = 1.0_f32;
    let mut cur = Some(id);
    while let Some(c) = cur {
        let node = &tree.arena[c];
        if let Some(v) = node.styles.get("opacity") {
            if let Ok(o) = v.trim().parse::<f32>() {
                acc *= o.clamp(0.0, 1.0);
            }
        }
        cur = node.parent;
    }
    acc
}

/// Resolves an SVG icon `fill` to a concrete color.
///
/// When an explicit fill is given, `currentColor` resolves to the inherited
/// `color`; any other explicit color is used directly. When no explicit fill is
/// present (the common case — the `<path>` inherits `fill` from a parent such as
/// an inline `style="fill:#5796e8"` on the `.icon` container, since `fill` is an
/// inherited CSS property) we walk up the ancestor chain for an inherited
/// `fill`, then `color`, falling back to white.
fn resolve_svg_fill(tree: &LayoutTree, id: LayoutNodeId, fill: Option<&str>) -> Color {
    if let Some(f) = fill {
        if f.eq_ignore_ascii_case("currentColor") {
            if let Some(v) = resolve_inherited(tree, id, "color") {
                if let Some(c) = css::parse_css_color(v) {
                    return c;
                }
            }
        } else if let Some(c) = css::parse_css_color(f) {
            return c;
        }
    }
    if let Some(v) = resolve_inherited(tree, id, "fill") {
        if let Some(c) = css::parse_css_color(v) {
            return c;
        }
    }
    if let Some(v) = resolve_inherited(tree, id, "color") {
        if let Some(c) = css::parse_css_color(v) {
            return c;
        }
    }
    Color::new([1.0, 1.0, 1.0, 1.0])
}

/// Composes a CSS `transform` (currently `rotate(<deg>)` and/or
/// `scale(<x> [<y>])`) onto an existing affine `base`, rotating/scaling around
/// the center of the node's box (CSS `transform-origin: center`, the default).
///
/// Without this, `.icon.new-tab { transform: rotate(45deg) }` would render as a
/// plain close-X instead of the intended plus sign.
fn apply_css_transform(
    node: &crate::layout::LayoutNode,
    rect: crate::layout::Rect,
    base: vello::peniko::kurbo::Affine,
) -> vello::peniko::kurbo::Affine {
    let Some(transform_str) = node.styles.get("transform") else {
        return base;
    };
    let mut extra = vello::peniko::kurbo::Affine::IDENTITY;
    // Tokens like "rotate(45deg)" / "scale(1.2)" separated by whitespace.
    for token in transform_str.split_whitespace() {
        let (func, rest) = match token.find('(') {
            Some(i) => (&token[..i], &token[i + 1..]),
            None => continue,
        };
        let args: Vec<f64> = rest
            .trim_end_matches(')')
            .split(',')
            .filter_map(|a| a.trim().trim_end_matches("deg").parse::<f64>().ok())
            .collect();
        match func {
            "rotate" if !args.is_empty() => {
                let rad = args[0].to_radians();
                extra *= vello::peniko::kurbo::Affine::rotate(rad);
            }
            "scale" if !args.is_empty() => {
                let sx = args[0];
                let sy = args.get(1).copied().unwrap_or(sx);
                extra *= vello::peniko::kurbo::Affine::scale_non_uniform(sx, sy);
            }
            _ => {}
        }
    }
    if extra == vello::peniko::kurbo::Affine::IDENTITY {
        return base;
    }
    let cx = rect.x as f64 + rect.width as f64 / 2.0;
    let cy = rect.y as f64 + rect.height as f64 / 2.0;
    let around_center = vello::peniko::kurbo::Affine::translate((cx, cy))
        * extra
        * vello::peniko::kurbo::Affine::translate((-cx, -cy));
    around_center * base
}

fn paint_node(tree: &LayoutTree, id: LayoutNodeId, renderer: &mut UIRenderer, font: &FontData) {
    let node = &tree.arena[id];
    let rect = node.rect;

    // Vector icons: draw the SVG path scaled into this node's box.
    if let Some((d, fill)) = &node.svg_path {
        if fill.as_deref() != Some("none") {
            if let Some(path) = crate::svg::parse_svg_path(d) {
                let (vx, vy, vbw, vbh) =
                    node.svg_view_box
                        .unwrap_or((0.0, 0.0, rect.width, rect.height));
                let vbw = if vbw > 0.0 {
                    vbw as f64
                } else {
                    rect.width as f64
                };
                let vbh = if vbh > 0.0 {
                    vbh as f64
                } else {
                    rect.height as f64
                };
                let s = (rect.width as f64 / vbw).min(rect.height as f64 / vbh);
                // Safety cap: never draw an icon larger than 48px on its longest
                // axis, so a stretched/oversized container (e.g. an absolutely
                // positioned overlay laid out in normal flow at fullscreen)
                // can't blow the glyph up to fill the screen.
                let s = s.min(48.0 / vbw).min(48.0 / vbh);
                // Center the viewBox contents inside the node box, accounting
                // for a non-zero viewBox origin (e.g. "0 -960 960 960").
                let tx = rect.x as f64 + (rect.width as f64 - vbw * s) / 2.0 - vx as f64 * s;
                let ty = rect.y as f64 + (rect.height as f64 - vbh * s) / 2.0 - vy as f64 * s;
                let mut transform = vello::peniko::kurbo::Affine::new([s, 0.0, 0.0, s, tx, ty]);
                // Apply any CSS `transform` (e.g. `.icon.new-tab { rotate(45deg) }`
                // turns the close-X into a plus). Rotates around the box center.
                transform = apply_css_transform(node, rect, transform);
                let color = resolve_svg_fill(tree, id, fill.as_deref());
                renderer.fill_bez_path(&path, transform, color);
            }
        }
    }

    // `<img>` nodes: draw the decoded image into this node's box.
    // Raster images are scaled to fill the box; SVG images are drawn as a path
    // (same scaling logic as inline `<svg>`), so both icon formats paint.
    if let Some(decoded) = &node.image {
        match decoded {
            crate::image_loader::DecodedImage::Raster(data) => {
                if data.width > 0 && data.height > 0 && rect.width > 0.0 && rect.height > 0.0 {
                    let (dw, dh) = match node.object_fit.as_deref() {
                        Some("cover") => {
                            let s = (rect.width as f64 / data.width as f64)
                                .max(rect.height as f64 / data.height as f64);
                            (data.width as f64 * s, data.height as f64 * s)
                        }
                        Some("fill") => (rect.width as f64, rect.height as f64),
                        _ => {
                            let s = (rect.width as f64 / data.width as f64)
                                .min(rect.height as f64 / data.height as f64);
                            (data.width as f64 * s, data.height as f64 * s)
                        }
                    };
                    let dx = rect.x as f64 + (rect.width as f64 - dw) / 2.0;
                    let dy = rect.y as f64 + (rect.height as f64 - dh) / 2.0;
                    let sx = dw / data.width as f64;
                    let sy = dh / data.height as f64;

                    // For significant downscaling (<50%), pre-resize with Lanczos3
                    // for better quality than vello's linear filter.
                    let (image_to_draw, sx_final, sy_final) = if sx < 0.5 && sy < 0.5 {
                        let new_w = (data.width as f64 * sx) as u32;
                        let new_h = (data.height as f64 * sy) as u32;
                        if new_w > 0 && new_h > 0 {
                            match crate::image_loader::resize_lanczos(data, new_w, new_h) {
                                Some(resized) => (resized, 1.0f64, 1.0f64),
                                None => (data.clone(), sx, sy),
                            }
                        } else {
                            (data.clone(), sx, sy)
                        }
                    } else {
                        (data.clone(), sx, sy)
                    };

                    let transform = vello::peniko::kurbo::Affine::new([
                        sx_final, 0.0, 0.0, sy_final, dx, dy,
                    ]);
                    let quality = match node.image_rendering.as_deref() {
                        Some("pixelated") | Some("crisp-edges") => vello::peniko::ImageQuality::Low,
                        _ => vello::peniko::ImageQuality::High,
                    };
                    renderer.draw_image(&image_to_draw, transform, quality);
                }
            }
            crate::image_loader::DecodedImage::Svg {
                path_d,
                view_box,
                fill,
                ..
            } => {
                if fill.as_deref() != Some("none") {
                    if let Some(path) = crate::svg::parse_svg_path(path_d) {
                        let (vx, vy, vbw, vbh) = *view_box;
                        let vbw = if vbw > 0.0 {
                            vbw as f64
                        } else {
                            rect.width as f64
                        };
                        let vbh = if vbh > 0.0 {
                            vbh as f64
                        } else {
                            rect.height as f64
                        };
                        let s = (rect.width as f64 / vbw).min(rect.height as f64 / vbh);
                        let s = s.min(48.0 / vbw).min(48.0 / vbh);
                        let tx =
                            rect.x as f64 + (rect.width as f64 - vbw * s) / 2.0 - vx as f64 * s;
                        let ty =
                            rect.y as f64 + (rect.height as f64 - vbh * s) / 2.0 - vy as f64 * s;
                        let mut transform =
                            vello::peniko::kurbo::Affine::new([s, 0.0, 0.0, s, tx, ty]);
                        transform = apply_css_transform(node, rect, transform);
                        let color = resolve_svg_fill(tree, id, fill.as_deref());
                        renderer.fill_bez_path(&path, transform, color);
                    }
                }
            }
        }
    }

    if node.tag == "#text" {
        if let Some(ref text) = node.text {
            let color = resolve_inherited(tree, id, "color")
                .and_then(|v| css::parse_css_color(v))
                .unwrap_or(Color::new([0.0, 0.0, 0.0, 1.0]))
                .multiply_alpha(effective_opacity(tree, id));
            let font_size = resolve_inherited(tree, id, "font-size")
                .and_then(|v| css::parse_css_length(v))
                .map(|v| v as f32)
                .unwrap_or(16.0);
            let font_weight = node
                .styles
                .get("font-weight")
                .and_then(|w| w.parse::<f32>().ok())
                .unwrap_or(400.0);
            let bold = font_weight >= 500.0;
            renderer.fill_text(
                rect.x as f64,
                rect.y as f64 + font_size as f64,
                text,
                font_size,
                color,
                font,
                bold,
                Some(font_weight),
            );
        }
        return;
    }

    let styles = &node.styles;

    let bg = styles
        .get("background-color")
        .or_else(|| styles.get("background"))
        .and_then(|v| css::parse_css_color(v));
    let border_radius = styles
        .get("border-radius")
        .and_then(|v| css::parse_css_border_radius(v).first().copied());
    let border_width = styles
        .get("border")
        .and_then(|v| css::parse_css_border_width(v));
    let border_color = styles
        .get("border-color")
        .and_then(|v| css::parse_css_color(v));

    if let Some(color) = bg {
        let color = color.multiply_alpha(effective_opacity(tree, id));
        match border_radius {
            Some(r) if r > 0.0 => {
                renderer.fill_rounded_rect(
                    rect.x as f64,
                    rect.y as f64,
                    rect.width as f64,
                    rect.height as f64,
                    r as f64,
                    color,
                );
            }
            _ => {
                renderer.fill_rect(
                    rect.x as f64,
                    rect.y as f64,
                    rect.width as f64,
                    rect.height as f64,
                    color,
                );
            }
        }
    }

    // `background-image`: draw the decoded raster image clipped to this box,
    // sized per `background-size` (cover/contain; anything else paints at the
    // image's intrinsic pixel size). The source was decoded + cached during
    // layout (see `build_node`).
    if let Some(img) = &node.background_image {
        if rect.width > 0.0 && rect.height > 0.0 {
            if let Some(r) = border_radius {
                renderer.push_rounded_clip(
                    rect.x as f64,
                    rect.y as f64,
                    rect.width as f64,
                    rect.height as f64,
                    r as f64,
                );
            }
            if let Some(decoded) = img.raster() {
                let (iw, ih) = (decoded.width as f64, decoded.height as f64);
                let (dw, dh) = match node.background_size.as_deref() {
                    Some("cover") => {
                        let s = (rect.width as f64 / iw).max(rect.height as f64 / ih);
                        (iw * s, ih * s)
                    }
                    Some("contain") => {
                        let s = (rect.width as f64 / iw).min(rect.height as f64 / ih);
                        (iw * s, ih * s)
                    }
                    _ => (iw, ih),
                };
                let dx = rect.x as f64 + (rect.width as f64 - dw) / 2.0;
                let dy = rect.y as f64 + (rect.height as f64 - dh) / 2.0;
                let sx = dw / iw;
                let sy = dh / ih;
                let transform = vello::peniko::kurbo::Affine::new([sx, 0.0, 0.0, sy, dx, dy]);
                renderer.draw_image(&decoded, transform, vello::peniko::ImageQuality::High);
            }
            if border_radius.is_some() {
                renderer.pop_clip();
            }
        }
    }

    for &child_id in &node.children {
        paint_node(tree, child_id, renderer, font);
    }

    if let Some(w) = border_width {
        if let Some(color) = border_color {
            renderer.stroke_rect(
                rect.x as f64,
                rect.y as f64,
                rect.width as f64,
                rect.height as f64,
                w as f64,
                color,
            );
        }
    }
}
