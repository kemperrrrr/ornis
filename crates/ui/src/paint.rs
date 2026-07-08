use crate::css;
use crate::layout::{LayoutNodeId, LayoutTree};
use crate::render::UIRenderer;

pub fn paint_layout(tree: &LayoutTree, renderer: &mut UIRenderer) {
    paint_node(tree, tree.root, renderer);
}

fn paint_node(tree: &LayoutTree, id: LayoutNodeId, renderer: &mut UIRenderer) {
    let node = &tree.arena[id];
    if node.tag == "#text" {
        return;
    }

    let styles = &node.styles;
    let rect = node.rect;

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

    // Draw background
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

    // Draw children
    for &child_id in &node.children {
        paint_node(tree, child_id, renderer);
    }

    // Draw border
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
