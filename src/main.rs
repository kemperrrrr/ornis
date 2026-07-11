use crossbeam_channel::{Receiver, Sender, unbounded};
use glam::{Mat4, Vec3};
use vello::peniko::{Color, FontData};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{WindowEvent, KeyEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::WindowAttributes;

use ornis_render::{Renderer3D, OpenPBRMaterial, Mesh, create_sphere, InstanceData};
use ornis_ui::components::EcsBridge;
use ornis_ui::css::Stylesheet;
use ornis_ui::editor::EditorOverlay;
use ornis_ui::ipc::{GameEvent, UiCommand};
use ornis_ui::js::JsRuntime;
use ornis_ui::layout::LayoutTree;
use ornis_ui::paint::paint_layout;
use ornis_ui::render::UIRenderer;

mod remote;

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
    remote_editor: Option<remote::RemoteEditor>,
}

struct GameContext {
    window: winit::window::Window,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: UIRenderer,
    renderer3d: Renderer3D,
    sphere_mesh: Mesh,
    materials: Vec<OpenPBRMaterial>,
    instance_data: Vec<InstanceData>,
    js: JsRuntime,
    layout_tree: Option<LayoutTree>,
    font: FontData,
    editor: EditorOverlay,
    remote_cmd_rx: Receiver<UiCommand>,
    remote_ev_tx: Sender<GameEvent>,
}

impl GameApp {
    fn new() -> Self {
        GameApp { context: None, remote_editor: None }
    }

    fn context(&mut self) -> Option<&mut GameContext> {
        self.context.as_mut()
    }

    fn initialize(event_loop: &ActiveEventLoop,
        remote_cmd_rx: Receiver<UiCommand>,
        remote_ev_tx: Sender<GameEvent>,
    ) -> Result<GameContext, String> {
        let window_attrs = WindowAttributes::default()
            .with_title("Ornis Engine")
            .with_inner_size(PhysicalSize::new(800, 600));
        let window = event_loop.create_window(window_attrs)
            .map_err(|e| format!("window creation: {e}"))?;

        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface_target = unsafe { wgpu::SurfaceTargetUnsafe::from_window(&window) }
            .map_err(|e| format!("surface target: {e}"))?;
        let surface: wgpu::Surface<'static> = unsafe {
            instance.create_surface_unsafe(surface_target)
                .map_err(|e| format!("surface creation: {e}"))?
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })).map_err(|_| "no adapter found".to_string())?;

        let mut limits = adapter.limits();
        limits.max_storage_buffers_per_shader_stage = 8;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ornis device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            },
        )).map_err(|e| format!("device request: {e}"))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter().copied().find(|f| {
            matches!(f, wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb)
        }).unwrap_or_else(|| surface_caps.formats.first().copied()
            .unwrap_or(wgpu::TextureFormat::Rgba8UnormSrgb));

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

        let renderer = UIRenderer::new(&device, &surface_config, surface_format)
            .map_err(|e| format!("renderer init: {e:?}"))?;

        let renderer3d = Renderer3D::new(&device, &surface_config, 1);
        let sphere_mesh = create_sphere(&device, 1.0, 32, 24);

        let materials = vec![
            OpenPBRMaterial::dielectric()
                .base_color_rgb([0.8, 0.2, 0.2]),
            OpenPBRMaterial::dielectric()
                .base_color_rgb([0.2, 0.8, 0.2])
                .specular_roughness(0.7),
            OpenPBRMaterial::dielectric()
                .base_color_rgb([0.2, 0.2, 0.8])
                .specular_roughness(0.1),
            OpenPBRMaterial::metal()
                .base_color_rgb([0.9, 0.7, 0.1])
                .specular_roughness(0.2),
            OpenPBRMaterial::coat()
                .base_color_rgb([0.9, 0.9, 0.9])
                .coat_weight(1.0)
                .coat_roughness(0.1),
        ];

        let spacing = 2.8;
        let instance_data: Vec<InstanceData> = materials.iter().enumerate().map(|(i, _)| {
            let x = (i as f32 - 2.0) * spacing;
            let pos = Vec3::new(x, 0.0, 0.0);
            let model = Mat4::from_translation(pos);
            let normal_matrix = Mat4::IDENTITY;
            InstanceData {
                model_matrix: model,
                normal_matrix,
                material_index: i as u32,
            }
        }).collect();

        let font = load_font();
        let bridge = EcsBridge::new();
        let js = JsRuntime::new(bridge);

        Ok(GameContext {
            window,
            device,
            queue,
            surface,
            surface_config,
            renderer,
            renderer3d,
            sphere_mesh,
            materials,
            instance_data,
            js,
            layout_tree: None,
            font,
            editor: EditorOverlay::new(),
            remote_cmd_rx,
            remote_ev_tx,
            })
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

    fn process_remote_commands(ctx: &mut GameContext) {
        while let Ok(cmd) = ctx.remote_cmd_rx.try_recv() {
            match cmd {
                UiCommand::Custom { cmd_type, json_data: _ } => {
                    match cmd_type.as_str() {
                        "create_entity" => {
                            let id = ctx.js.bridge.create_entity();
                            ctx.remote_ev_tx.send(GameEvent::CustomEvent {
                                cmd_type: "entity_created".into(),
                                json_data: format!(r#"{{"entity_id":{id}}}"#),
                            }).ok();
                        }
                        "list_entities" => {
                            ctx.remote_ev_tx.send(GameEvent::CustomEvent {
                                cmd_type: "entity_list".into(),
                                json_data: format!(r#"{{"count":{}}}"#, entity_count(ctx)),
                            }).ok();
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    fn render_frame(ctx: &mut GameContext) {
        let w = ctx.surface_config.width as f64;
        let h = ctx.surface_config.height as f64;
        let aspect = w as f32 / h as f32;

        // --- Build Vello UI scene (transparent bg) ---
        ctx.renderer.begin_frame();

        // Starfield / decorative elements
        ctx.renderer.fill_rect(0.0, 0.0, w, h, Color::new([0.0, 0.0, 0.0, 0.0]));
        let stars = [(100.0, 80.0, 2.0), (300.0, 150.0, 1.5), (500.0, 60.0, 2.5),
                     (700.0, 200.0, 1.0), (200.0, 400.0, 2.0), (600.0, 500.0, 1.5)];
        for &(sx, sy, sr) in &stars {
            ctx.renderer.fill_circle(sx, sy, sr, Color::new([1.0, 1.0, 1.0, 0.3]));
        }
        for i in 0..20 {
            let y = (i as f64) * 60.0 + 30.0;
            ctx.renderer.stroke_rect(0.0, y, w, 1.0, 1.0, Color::new([1.0, 1.0, 1.0, 0.04]));
        }
        if let Some(ref tree) = ctx.layout_tree {
            paint_layout(tree, &mut ctx.renderer, &ctx.font);
        }
        ctx.editor.set_entity_count(entity_count(ctx));
        ctx.editor.paint(&mut ctx.renderer, w, h, &ctx.font);

        // Render Vello scene to internal texture
        ctx.renderer.render_scene(&ctx.device, &ctx.queue).ok();

        // --- 3D PBR setup ---
        let view = Mat4::look_at_rh(Vec3::new(0.0, 2.5, 9.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, aspect, 0.1, 100.0);
        let view_proj = proj * view;

        ctx.renderer3d.set_camera(&ctx.queue, &view_proj.to_cols_array_2d(), [0.0, 2.5, 9.0]);
        ctx.renderer3d.set_lights(
            &ctx.queue,
            [0.10, 0.10, 0.15],
            &[
                ([1.0, 1.0, 1.0], 0.6, [1.0, 1.0, 1.0]),
                ([-0.5, 0.5, -0.5], 0.3, [0.8, 0.8, 1.0]),
            ],
        );
        ctx.renderer3d.upload_materials(&ctx.queue, &ctx.materials);
        ctx.renderer3d.upload_instances(&ctx.queue, &ctx.instance_data);

        // --- Render 3D to PBR texture, composite with UI onto surface ---
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ornis frame"),
        });

        let frame = match ctx.surface.get_current_texture() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("surface error: {e:?}");
                return;
            }
        };
        let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        ctx.renderer3d.render_scene(
            &ctx.device,
            &mut encoder,
            &frame_view,
            &ctx.sphere_mesh,
            ctx.instance_data.len() as u32,
        );

        ctx.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

impl ApplicationHandler for GameApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (cmd_tx, cmd_rx) = unbounded();
            let (ev_tx, ev_rx) = unbounded();
            let editor = remote::RemoteEditor::start(3420, cmd_tx, ev_rx);
            self.remote_editor = Some(editor);
            match Self::initialize(event_loop, cmd_rx, ev_tx) {
                Ok(mut ctx) => {
                    Self::build_layout(&mut ctx);
                    self.context = Some(ctx);
                }
                Err(e) => {
                    eprintln!("ornis: failed to initialize: {e}");
                }
            }
        }));
        if let Err(e) = result {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown cause".to_string()
            };
            eprintln!("ornis: initialization panicked: {msg}");
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let ctx = self.context();
        let Some(ctx) = ctx else { return };
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                Self::render_frame(ctx);
            }
            WindowEvent::Resized(size) => {
                ctx.surface_config.width = size.width.max(1);
                ctx.surface_config.height = size.height.max(1);
                ctx.surface.configure(&ctx.device, &ctx.surface_config);
                ctx.renderer.resize(&ctx.device, size.width.max(1), size.height.max(1));
                ctx.renderer3d.resize(&ctx.device, size.width.max(1), size.height.max(1));
                Self::build_layout(ctx);
                ctx.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event: KeyEvent { logical_key, state, .. }, .. } => {
                if state == winit::event::ElementState::Pressed {
                    let key_str = match &logical_key {
                        Key::Named(NamedKey::Escape) => "Escape",
                        Key::Named(NamedKey::F1) => "F1",
                        Key::Character(c) => c.as_str(),
                        _ => return,
                    };
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
        if let Some(ctx) = &mut self.context {
            Self::process_remote_commands(ctx);
            ctx.window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.context = None;
    }
}

fn entity_count(ctx: &GameContext) -> u32 {
    ctx.js.bridge.entity_count()
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
