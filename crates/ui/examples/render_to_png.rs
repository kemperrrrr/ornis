//! Headless render of the in-game editor to a PNG, for offline visual
//! verification. Mirrors `src/main.rs`'s editor build path so the PNG matches
//! what the live game window draws.
//!
//! Usage:
//!   cargo run -p ornis-ui --features serialize --example render_to_png <W> <H> [out.png]

use ornis_ui::css::Stylesheet;
use ornis_ui::editor_template::{EditorTemplate, UnifiedEditorConfig, read_asset};
use ornis_ui::html::parse_html;
use ornis_ui::layout::LayoutTree;
use ornis_ui::paint::paint_layout;
use ornis_ui::render::UIRenderer;
use ornis_ui::text::load_inter_font;

use vello::peniko::{Color, FontData as VelloFontData};

fn load_font() -> VelloFontData {
    load_inter_font()
}

async fn run(w: u32, h: u32, out: &str) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("png device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 8,
                ..wgpu::Limits::downlevel_defaults()
            },
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("device");

    let surface_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: w,
        height: h,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    let mut renderer = UIRenderer::new(&device, &surface_config, surface_format).expect("renderer");
    let font = load_font();

    let editor_config = UnifiedEditorConfig::default();
    let editor_template = EditorTemplate::new(editor_config);
    let editor_html = read_asset("index.html");
    let editor_css = editor_template.generate_css();

    let doc = parse_html(&editor_html);
    let stylesheet = Stylesheet::parse(&editor_css).unwrap_or_else(|_| Stylesheet {
        rules: vec![],
        custom_properties: std::collections::HashMap::new(),
    });
    let tree = LayoutTree::build_with_viewport(&doc, &[stylesheet], w as f32, h as f32, &font, &[], None)
        .expect("layout build");
    
    renderer.begin_frame();
    renderer.fill_rect(
        0.0,
        0.0,
        w as f64,
        h as f64,
        Color::new([0.12, 0.12, 0.14, 1.0]),
    );
    paint_layout(&tree, &mut renderer, &font, None);
    match renderer.save_png(&device, &queue, out) {
        Ok(_) => println!("wrote {out} ({w}x{h})"),
        Err(e) => eprintln!("png save failed: {e:?}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let w: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1280);
    let h: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(800);
    let out = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "/tmp/ornis_editor.png".to_string());
    pollster::block_on(run(w, h, &out));
}
