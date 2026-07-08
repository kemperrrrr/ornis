use crate::css;
use crate::layout::{LayoutNodeId, LayoutTree};
use crate::render::UIRenderer;
use vello::peniko::{Color, FontData};

pub fn paint_layout(tree: &LayoutTree, renderer: &mut UIRenderer, font: &FontData) {
    paint_node(tree, tree.root, renderer, font);
}

fn paint_node(tree: &LayoutTree, id: LayoutNodeId, renderer: &mut UIRenderer, font: &FontData) {
    let node = &tree.arena[id];
    let rect = node.rect;

    if node.tag == "#text" {
        if let Some(ref text) = node.text {
            let color = node.styles.get("color").and_then(|v| css::parse_css_color(v)).unwrap_or(Color::new([0.0, 0.0, 0.0, 1.0]));
            let font_size = node.styles.get("font-size").and_then(|v| css::parse_css_length(v)).map(|v| v as f32).unwrap_or(16.0);
            renderer.fill_text(rect.x as f64, rect.y as f64 + font_size as f64, text, font_size, color, font);
        }
        return;
    }

    let styles = &node.styles;

    let bg = styles.get("background-color").and_then(|v| css::parse_css_color(v));
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
