use vello::peniko::{Color, FontData};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{WindowEvent, KeyEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::WindowAttributes;

use ornis_ui::components::EcsBridge;
use ornis_ui::css::Stylesheet;
use ornis_ui::editor::EditorOverlay;
use ornis_ui::js::JsRuntime;
use ornis_ui::layout::LayoutTree;
use ornis_ui::paint::paint_layout;
use ornis_ui::render::UIRenderer;

const UI_JS: &str = r##"
(function() {
var container = document.createElement("div");
container.setAttribute("class", "container");

var button = document.createElement("div");
button.setAttribute("class", "button");

var icon = document.createElement("div");
icon.setAttribute("class", "icon-play");

var label = document.createElement("span");
label.setAttribute("class", "label");
label.textContent = "Play";

button.appendChild(icon);
button.appendChild(label);
container.appendChild(button);
document.body.appendChild(container);
})();
"##;

const UI_CSS: &str = r#"
body {
  display: flex;
  width: 100%;
  height: 100%;
}
.container {
  display: flex;
  width: 100%;
  height: 100%;
  align-items: center;
  justify-content: center;
  background-color: #1e1e24;
}
.button {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 200px;
  height: 60px;
  background-color: #3b6ff0;
  border-radius: 12px;
}
.icon-play {
  width: 18px;
  height: 18px;
  background-color: #ffffff;
  border-radius: 9px;
}
.label {
  color: #ffffff;
  font-size: 16px;
}
"#;

struct GameApp {
    context: Option<GameContext>,
}

struct GameContext {
    window: winit::window::Window,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: UIRenderer,
    js: JsRuntime,
    layout_tree: Option<LayoutTree>,
    font: FontData,
    editor: EditorOverlay,
}

impl GameApp {
    fn new() -> Self {
        GameApp { context: None }
    }

    fn context(&mut self) -> &mut GameContext {
        self.context.as_mut().unwrap()
    }

    fn build_layout(ctx: &mut GameContext) {
        let _ = ctx.js.eval(UI_JS);
        let root = ctx.js.document_node();
        let doc = ornis_ui::dom::Document { root };
        if let Ok(stylesheet) = Stylesheet::parse(UI_CSS) {
            if let Ok(tree) = LayoutTree::build_with_viewport(
                &doc,
                &[stylesheet],
                ctx.surface_config.width as f32,
                ctx.surface_config.height as f32,
            ) {
                ctx.layout_tree = Some(tree);
            }
        }
    }

    fn render_frame(ctx: &mut GameContext) {
        let w = ctx.surface_config.width as f64;
        let h = ctx.surface_config.height as f64;

        ctx.renderer.begin_frame();

        // --- Game world (background) ---
        ctx.renderer.fill_rect(0.0, 0.0, w, h, Color::new([0.12, 0.12, 0.14, 1.0]));

        // Starfield / decorative elements
        let stars = [(100.0, 80.0, 2.0), (300.0, 150.0, 1.5), (500.0, 60.0, 2.5),
                     (700.0, 200.0, 1.0), (200.0, 400.0, 2.0), (600.0, 500.0, 1.5)];
        for &(sx, sy, sr) in &stars {
            ctx.renderer.fill_circle(sx, sy, sr, Color::new([1.0, 1.0, 1.0, 0.3]));
        }

        // Grid lines
        for i in 0..20 {
            let y = (i as f64) * 60.0 + 30.0;
            ctx.renderer.stroke_rect(0.0, y, w, 1.0, 1.0, Color::new([1.0, 1.0, 1.0, 0.04]));
        }

        // --- UI overlay (game HUD / buttons) ---
        if let Some(ref tree) = ctx.layout_tree {
            paint_layout(tree, &mut ctx.renderer, &ctx.font);
        }

        // --- Editor overlay ---
        ctx.editor.set_entity_count(0);
        ctx.editor.paint(&mut ctx.renderer, w, h, &ctx.font);

        if let Err(e) = ctx.renderer.end_frame(&ctx.device, &ctx.queue, &ctx.surface) {
            eprintln!("render error: {e:?}");
        }
    }
}

impl ApplicationHandler for GameApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = WindowAttributes::default()
            .with_title("Ornis Engine")
            .with_inner_size(PhysicalSize::new(800, 600));
        let window = event_loop.create_window(window_attrs).unwrap();

        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface: wgpu::Surface<'static> = unsafe {
            instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(&window).unwrap(),
            ).unwrap()
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })).unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ornis device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            },
        )).unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter().copied().find(|f| {
            matches!(f, wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb)
        }).unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let renderer = UIRenderer::new(&device, &surface_config, surface_format).unwrap();

        let font = load_font();
        let bridge = EcsBridge::new();
        let js = JsRuntime::new(bridge);

        let mut ctx = GameContext {
            window,
            device,
            queue,
            surface,
            surface_config,
            renderer,
            js,
            layout_tree: None,
            font,
            editor: EditorOverlay::new(),
        };

        Self::build_layout(&mut ctx);
        self.context = Some(ctx);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let ctx = self.context();
                Self::render_frame(ctx);
            }
            WindowEvent::Resized(size) => {
                let ctx = self.context();
                ctx.surface_config.width = size.width.max(1);
                ctx.surface_config.height = size.height.max(1);
                ctx.surface.configure(&ctx.device, &ctx.surface_config);
                ctx.renderer.resize(&ctx.device, size.width.max(1), size.height.max(1));
                Self::build_layout(ctx);
                Self::render_frame(ctx);
            }
            WindowEvent::KeyboardInput { event: KeyEvent { logical_key, state, .. }, .. } => {
                if state == winit::event::ElementState::Pressed {
                    let key_str = match &logical_key {
                        Key::Named(NamedKey::Escape) => "Escape",
                        Key::Character(c) => c.as_str(),
                        _ => return,
                    };
                    let ctx = self.context();
                    if ctx.editor.handle_key(key_str) {
                        Self::build_layout(ctx);
                        ctx.window.request_redraw();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ctx) = &self.context {
            ctx.window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.context = None;
    }
}

fn load_font() -> FontData {
    let paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ];
    for path in &paths {
        if let Ok(data) = std::fs::read(path) {
            return ornis_ui::text::load_font_from_bytes(&data);
        }
    }
    eprintln!("warning: no system font found, text will be invisible");
    FontData::new(vello::peniko::Blob::from(Vec::new()), 0)
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = GameApp::new();
    event_loop.run_app(&mut app).unwrap();
}
