import re

with open('crates/ui/src/paint.rs', 'r') as f:
    content = f.read()

# 1. Fix is_interactive to include interactive classes
old_is = '''/// Returns true for elements that should show a hover highlight.
fn is_interactive(node: &crate::layout::LayoutNode) -> bool {
    match node.tag.as_str() {
        "a" | "button" | "input" | "textarea" | "select" | "summary" | "details" | "label" => true,
        _ => {
            node.styles.get("cursor")
                .map(|c| c.trim() == "pointer")
                .unwrap_or(false)
        }
    }
}'''

new_is = '''/// Returns true for elements that should show a hover highlight.
fn is_interactive(node: &crate::layout::LayoutNode) -> bool {
    match node.tag.as_str() {
        "a" | "button" | "input" | "textarea" | "select" | "summary" | "details" | "label" => true,
        _ => {
            let has_interactive_class = node.dom_classes.iter().any(|c| {
                matches!(c.as_str(), "tab" | "close" | "icon" | "new-tab" | "arrow" | "file" | "play-options" | "logs" | "slider")
            });
            has_interactive_class || node.styles.get("cursor")
                .map(|c| c.trim() == "pointer")
                .unwrap_or(false)
        }
    }
}'''

content = content.replace(old_is, new_is)

# 2. Add arrow drawing before children loop
# Find the line "    // Collect children sorted by z-index." and insert arrow drawing before it
arrow_code = '''    // Draw .arrow chevrons (original uses ::before which we don't support).
    if node.dom_classes.iter().any(|c| c == "arrow") {
        if let Some(parent_id) = node.parent {
            let parent = &tree.arena[parent_id];
            if parent.tag == "summary" {
                let is_open = parent.parent
                    .map(|gp| interaction.map(|i| i.is_details_open(gp)).unwrap_or(true))
                    .unwrap_or(true);
                let color = resolve_inherited(tree, id, "color")
                    .and_then(|v| css::parse_css_color(v))
                    .unwrap_or(Color::new([1.0, 1.0, 1.0, 0.5]));
                let cx = rect.x + rect.width / 2.0;
                let cy = rect.y + rect.height / 2.0;
                let sz = 4.0_f32;
                use vello::peniko::kurbo::{BezPath, PathEl};
                let mut path = BezPath::new();
                if is_open {
                    path.push(PathEl::MoveTo((cx - sz) as f64, (cy - sz * 0.3) as f64));
                    path.push(PathEl::LineTo((cx + sz) as f64, (cy - sz * 0.3) as f64));
                    path.push(PathEl::LineTo(cx as f64, (cy + sz * 0.7) as f64));
                } else {
                    path.push(PathEl::MoveTo((cx - sz * 0.3) as f64, (cy - sz) as f64));
                    path.push(PathEl::LineTo((cx - sz * 0.3) as f64, (cy + sz) as f64));
                    path.push(PathEl::LineTo((cx + sz * 0.7) as f64, cy as f64));
                }
                path.push(PathEl::ClosePath);
                renderer.fill_bez_path(&path, vello::peniko::kurbo::Affine::IDENTITY, color);
            }
        }
    }

    // Collect children sorted by z-index.'''

content = content.replace('    // Collect children sorted by z-index.', arrow_code)

with open('crates/ui/src/paint.rs', 'w') as f:
    f.write(content)

print('Fixed paint.rs')

# 3. Fix about_to_wait — remove unconditional request_redraw
with open('src/main.rs', 'r') as f:
    content = f.read()

old_about = '''    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ctx) = &mut self.context {
            Self::process_remote_commands(ctx);
            ctx.window.request_redraw();
        }
    }'''

new_about = '''    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ctx) = &mut self.context {
            Self::process_remote_commands(ctx);
            // redraw is requested only when state changes (hover, click, scroll, resize, key)
        }
    }'''

content = content.replace(old_about, new_about)

with open('src/main.rs', 'w') as f:
    f.write(content)

print('Fixed main.rs about_to_wait')
