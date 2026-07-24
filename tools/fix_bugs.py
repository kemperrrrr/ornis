import re

# Fix paint.rs — hover only on interactive elements
with open('crates/ui/src/paint.rs', 'r') as f:
    content = f.read()

old_hover = '''    // Hover highlight (subtle white overlay)
    if let Some(inter) = interaction {
        if inter.hovered == Some(id) {
            let hover_color = Color::new([1.0, 1.0, 1.0, 0.06]);
            renderer.fill_rect(
                rect.x as f64,
                rect.y as f64,
                rect.width as f64,
                rect.height as f64,
                hover_color,
            );
        }
    }'''

new_hover = '''    // Hover highlight — only on interactive elements
    if let Some(inter) = interaction {
        if inter.hovered == Some(id) && is_interactive(node) {
            let hover_color = Color::new([1.0, 1.0, 1.0, 0.06]);
            renderer.fill_rect(
                rect.x as f64,
                rect.y as f64,
                rect.width as f64,
                rect.height as f64,
                hover_color,
            );
        }
    }'''

content = content.replace(old_hover, new_hover)

# Add is_interactive function at the end of paint_node function, before the closing brace
# Find the last closing brace of paint_node and insert before it
# Actually, let's add it as a standalone function at the end of the file
is_interactive_fn = '''
/// Returns true for elements that should show a hover highlight.
fn is_interactive(node: &crate::layout::LayoutNode) -> bool {
    match node.tag.as_str() {
        "a" | "button" | "input" | "textarea" | "select" | "summary" | "details" | "label" => true,
        _ => {
            node.styles.get("cursor")
                .map(|c| c.trim() == "pointer")
                .unwrap_or(false)
        }
    }
}
'''

content = content.rstrip() + is_interactive_fn

with open('crates/ui/src/paint.rs', 'w') as f:
    f.write(content)

print('Fixed paint.rs hover')

# Fix main.rs — scale factor for CursorMoved
with open('src/main.rs', 'r') as f:
    content = f.read()

old_cursor = '''            WindowEvent::CursorMoved { position, .. } => {
                let x = position.x as f32;
                let y = position.y as f32;'''

new_cursor = '''            WindowEvent::CursorMoved { position, .. } => {
                let scale = ctx.window.scale_factor() as f32;
                let x = position.x as f32 / scale;
                let y = position.y as f32 / scale;'''

content = content.replace(old_cursor, new_cursor)

with open('src/main.rs', 'w') as f:
    f.write(content)

print('Fixed main.rs scale factor')
