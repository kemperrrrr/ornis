with open('src/main.rs', 'r') as f:
    content = f.read()

old = '''            WindowEvent::CursorMoved { position, .. } => {
                let scale = ctx.window.scale_factor() as f32;
                let x = position.x as f32 / scale;
                let y = position.y as f32 / scale;
                if let Some(ref tree) = ctx.layout_tree {
                    let changed = ctx.interaction.handle_mouse_move(tree, x, y);
                    if changed {
                        ctx.window.request_redraw();
                    }
                }
            }'''

new = '''            WindowEvent::CursorMoved { position, .. } => {
                let scale = ctx.window.scale_factor() as f32;
                let x = position.x as f32 / scale;
                let y = position.y as f32 / scale;
                ctx.interaction.last_mouse_pos = (x, y);
            }'''

content = content.replace(old, new)

with open('src/main.rs', 'w') as f:
    f.write(content)

print('Removed redraw on CursorMoved')
