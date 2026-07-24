with open('src/main.rs', 'r') as f:
    content = f.read()

old = '''    fn render_frame(ctx: &mut GameContext) {
        let w = ctx.surface_config.width as f64;
        let h = ctx.surface_config.height as f64;
        let aspect = w as f32 / h as f32;

        ctx.renderer.begin_frame();

        ctx.renderer.fill_rect(0.0, 0.0, w, h, Color::new([0.0, 0.0, 0.0, 0.0]));
        if ctx.layout_tree.is_some() {'''

new = '''    fn render_frame(ctx: &mut GameContext) {
        let frame_start = std::time::Instant::now();
        let w = ctx.surface_config.width as f64;
        let h = ctx.surface_config.height as f64;
        let aspect = w as f32 / h as f32;

        let t0 = std::time::Instant::now();
        ctx.renderer.begin_frame();
        let t_begin = t0.elapsed().as_millis();

        ctx.renderer.fill_rect(0.0, 0.0, w, h, Color::new([0.0, 0.0, 0.0, 0.0]));
        let t1 = std::time::Instant::now();
        #[cfg(not(feature = "blitz"))]
        if let Some(ref tree) = ctx.layout_tree {
            paint_layout(tree, &mut ctx.renderer, &ctx.font, Some(&ctx.interaction));
        }
        let t_paint = t1.elapsed().as_millis();

        let t2 = std::time::Instant::now();
        ctx.renderer.render_scene(&ctx.device, &ctx.queue).ok();
        let t_render = t2.elapsed().as_millis();

        let total = frame_start.elapsed().as_millis();
        if total > 16 {
            eprintln!("[perf] frame={total}ms | begin={t_begin}ms | paint={t_paint}ms | render_scene={t_render}ms");
        }

        // 3D setup'''

content = content.replace(old, new)

with open('src/main.rs', 'w') as f:
    f.write(content)

print('Added frame timing to render_frame')
