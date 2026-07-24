with open('crates/ui/src/paint.rs', 'r') as f:
    content = f.read()

# Fix PathEl calls to use Point::new instead of two arguments
content = content.replace(
    'path.push(PathEl::MoveTo((cx - sz) as f64, (cy - sz * 0.3) as f64));',
    'path.push(PathEl::MoveTo(vello::peniko::kurbo::Point::new((cx - sz) as f64, (cy - sz * 0.3) as f64)));'
)
content = content.replace(
    'path.push(PathEl::LineTo((cx + sz) as f64, (cy - sz * 0.3) as f64));',
    'path.push(PathEl::LineTo(vello::peniko::kurbo::Point::new((cx + sz) as f64, (cy - sz * 0.3) as f64)));'
)
content = content.replace(
    'path.push(PathEl::LineTo(cx as f64, (cy + sz * 0.7) as f64));',
    'path.push(PathEl::LineTo(vello::peniko::kurbo::Point::new(cx as f64, (cy + sz * 0.7) as f64)));'
)
content = content.replace(
    'path.push(PathEl::MoveTo((cx - sz * 0.3) as f64, (cy - sz) as f64));',
    'path.push(PathEl::MoveTo(vello::peniko::kurbo::Point::new((cx - sz * 0.3) as f64, (cy - sz) as f64)));'
)
content = content.replace(
    'path.push(PathEl::LineTo((cx - sz * 0.3) as f64, (cy + sz) as f64));',
    'path.push(PathEl::LineTo(vello::peniko::kurbo::Point::new((cx - sz * 0.3) as f64, (cy + sz) as f64)));'
)
content = content.replace(
    'path.push(PathEl::LineTo((cx + sz * 0.7) as f64, cy as f64));',
    'path.push(PathEl::LineTo(vello::peniko::kurbo::Point::new((cx + sz * 0.7) as f64, cy as f64)));'
)

with open('crates/ui/src/paint.rs', 'w') as f:
    f.write(content)

print('Fixed kurbo PathEl Point API')
