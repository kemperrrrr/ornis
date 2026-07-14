use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::WindowAttributes;

use ornis_ui::components::EcsBridge;
use ornis_ui::css::Stylesheet;
use ornis_ui::js::JsRuntime;
use ornis_ui::layout::LayoutTree;
use ornis_ui::paint::paint_layout;
use ornis_ui::render::UIRenderer;

use vello::peniko::{Color, FontData};

const JS_CODE: &str = r##"
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
"##;

const CSS: &str = r#"
.container {
  display: flex;
  width: 800px;
  height: 600px;
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

struct DemoApp {
    context: Option<DemoContext>,
}

struct DemoContext {
    window: winit::window::Window,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: UIRenderer,
    js: JsRuntime,
    layout_tree: Option<LayoutTree>,
    font: FontData,
}

impl DemoApp {
    fn new() -> Self {
        DemoApp { context: None }
    }

    fn context(&mut self) -> &mut DemoContext {
        self.context.as_mut().unwrap()
    }
}

impl ApplicationHandler for DemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = WindowAttributes::default()
            .with_title("Ornis UI Demo")
            .with_inner_size(PhysicalSize::new(800, 600));
        let window = event_loop.create_window(window_attrs).unwrap();

        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface: wgpu::Surface<'static> = unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(&window).unwrap())
                .unwrap()
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("demo device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    f,
                    wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb
                )
            })
            .unwrap_or(surface_caps.formats[0]);

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

        let font = load_demo_font();
        let bridge = EcsBridge::new();
        let js = JsRuntime::new(bridge);

        self.context = Some(DemoContext {
            window,
            device,
            queue,
            surface,
            surface_config,
            renderer,
            js,
            layout_tree: None,
            font,
        });

        self.build_layout(size.width, size.height);
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
                self.render_frame();
            }
            WindowEvent::Resized(size) => {
                let ctx = self.context();
                ctx.surface_config.width = size.width.max(1);
                ctx.surface_config.height = size.height.max(1);
                ctx.surface.configure(&ctx.device, &ctx.surface_config);
                ctx.renderer
                    .resize(&ctx.device, size.width.max(1), size.height.max(1));
                self.build_layout(size.width, size.height);
                self.render_frame();
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

impl DemoApp {
    fn build_layout(&mut self, viewport_width: u32, viewport_height: u32) {
        let ctx = self.context();
        let _ = ctx.js.eval(JS_CODE);

        let root = ctx.js.document_node();
        let doc = ornis_ui::dom::Document { root };
        let stylesheets = vec![Stylesheet::parse(CSS).unwrap()];
        let font = ornis_ui::text::load_font_from_bytes(&[]);
        let tree = LayoutTree::build_with_viewport(
            &doc,
            &stylesheets,
            viewport_width as f32,
            viewport_height as f32,
            &font,
        )
        .unwrap();
        ctx.layout_tree = Some(tree);
    }

    fn render_frame(&mut self) {
        let ctx = self.context();

        ctx.renderer.begin_frame();

        let w = ctx.surface_config.width as f64;
        let h = ctx.surface_config.height as f64;

        // Clear background
        ctx.renderer
            .fill_rect(0.0, 0.0, w, h, Color::new([0.12, 0.12, 0.14, 1.0]));

        // Paint layout tree
        if let Some(ref tree) = ctx.layout_tree {
            paint_layout(tree, &mut ctx.renderer, &ctx.font);
        }

        if let Err(e) = ctx
            .renderer
            .end_frame(&ctx.device, &ctx.queue, &ctx.surface)
        {
            eprintln!("render error: {e:?}");
        }
    }
}

fn load_demo_font() -> FontData {
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
    let mut app = DemoApp::new();
    event_loop.run_app(&mut app).unwrap();
}
