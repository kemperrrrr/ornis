use crossbeam_channel::unbounded;

mod remote;

// ═══════════════════════════════════════════════════════════════════════════
// РЕЖИМ "ТОЛЬКО РЕДАКТОР В БРАУЗЕРЕ" (editor-only)
// ═══════════════════════════════════════════════════════════════════════════
// Запуск: cargo run --features editor-only
// При этом нативное winit-окно НЕ создаётся. Работает только HTTP-сервер
// RemoteEditor на порту 3420. Разработчик открывает браузер по адресу
// http://127.0.0.1:3420 и получает полноценный редактор.
//
// См. STRATEGY_PIVOT.md — стратегический поворот от нативного UI
// к браузерному редактору (июль 2026).
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "editor-only")]
fn main() {
    let (cmd_tx, _cmd_rx) = unbounded();
    let (_ev_tx, ev_rx) = unbounded();

    let editor = remote::RemoteEditor::start(3420, cmd_tx, ev_rx);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Ornis Engine — Browser Editor Mode                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Editor:  http://127.0.0.1:3420                               ║");
    println!("║  Status:  http://127.0.0.1:3420/api/status                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Press Ctrl+C to stop                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Бесконечное ожидание — сервер работает в отдельном потоке.
    loop {
        std::thread::park();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ПОЛНЫЙ ДВИЖОК С НАТИВНЫМ ОКНОМ (режим по умолчанию)
// ═══════════════════════════════════════════════════════════════════════════
// Запуск: cargo run
// Классический режим: winit-окно, wgpu-рендеринг, Vello UI overlay,
// 3D-сцена со сферами. RemoteEditor тоже работает на порту 3420.
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "editor-only"))]
mod native {
    pub use crossbeam_channel::{Receiver, Sender, unbounded};
    pub use glam::{Mat4, Vec3};
    pub use vello::peniko::{Color, FontData};
    pub use winit::application::ApplicationHandler;
    pub use winit::dpi::PhysicalSize;
    pub use winit::event::{WindowEvent, KeyEvent, MouseButton as WinitMouseButton, MouseScrollDelta};
    pub use winit::event_loop::{ActiveEventLoop, EventLoop};
    pub use winit::keyboard::{Key, NamedKey};
    pub use winit::window::WindowAttributes;

    pub use ornis_render::{OpenPBRMaterial, Mesh, create_sphere, InstanceData, RenderBackend, RenderBackendConfig, create_render_backend, LegacyCompositePass};
    pub use ornis_ui::editor_template::{EditorTemplate, UnifiedEditorConfig, read_asset};
    pub use ornis_ui::events::{InteractionState, MouseButton};
    pub use ornis_ui::ipc::{GameEvent, UiCommand};
    pub use ornis_ui::render::UIRenderer;

    #[cfg(not(feature = "blitz"))]
    pub use ornis_ui::components::EcsBridge;
    #[cfg(not(feature = "blitz"))]
    pub use ornis_ui::js::JsRuntime;
    #[cfg(not(feature = "blitz"))]
    pub use ornis_ui::layout::LayoutTree;
    #[cfg(not(feature = "blitz"))]
    pub use ornis_ui::paint::paint_layout;
}

#[cfg(not(feature = "editor-only"))]
use native::*;

#[cfg(not(feature = "editor-only"))]
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

#[cfg(not(feature = "editor-only"))]
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
  height: 60px
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

#[cfg(not(feature = "editor-only"))]
struct GameApp {
    context: Option<GameContext>,
    remote_editor: Option<remote::RemoteEditor>,
}

#[cfg(not(feature = "editor-only"))]
struct GameContext {
    window: winit::window::Window,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: UIRenderer,
    renderer3d: Box<dyn RenderBackend>,
    sphere_mesh: Mesh,
    materials: Vec<OpenPBRMaterial>,
    instance_data: Vec<InstanceData>,
    #[cfg(not(feature = "blitz"))]
    js: JsRuntime,
    #[cfg(not(feature = "blitz"))]
    layout_tree: Option<LayoutTree>,
    #[cfg(not(feature = "blitz"))]
    font: FontData,
    editor_html: String,
    #[cfg(not(feature = "blitz"))]
    editor_css: String,
    #[cfg(not(feature = "blitz"))]
    cached_doc: Option<ornis_ui::dom::Document>,
    #[cfg(not(feature = "blitz"))]
    cached_stylesheet: Option<ornis_ui::css::Stylesheet>,
    remote_cmd_rx: Receiver<UiCommand>,
    remote_ev_tx: Sender<GameEvent>,
    pbr_target: wgpu::Texture,
    composite_pass: LegacyCompositePass,
    interaction: InteractionState,
}

#[cfg(not(feature = "editor-only"))]
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
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), flags: wgpu::InstanceFlags::empty(), memory_budget_thresholds: Default::default(), backend_options: Default::default(), display: None });

        let surface_target = unsafe { wgpu::SurfaceTargetUnsafe::from_display_and_window(event_loop, &window) }
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
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
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

        let backend_config = RenderBackendConfig {
            surface_config: surface_config.clone(),
            sample_count: 1,
            max_objects: 256,
            max_materials: 64,
        };
        let renderer3d: Box<dyn RenderBackend> = create_render_backend(&device, &backend_config);
        let sphere_mesh = create_sphere(&device, 1.0, 32, 24);

        let pbr_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbr intermediate"),
            size: wgpu::Extent3d {
                width: surface_config.width.max(1),
                height: surface_config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let composite_pass = LegacyCompositePass::new(&device, surface_format);

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

        #[cfg(not(feature = "blitz"))]
        let font = load_font();
        #[cfg(not(feature = "blitz"))]
        let bridge = EcsBridge::new();
        #[cfg(not(feature = "blitz"))]
        let js = JsRuntime::new(bridge);

        let editor_template = EditorTemplate::new(UnifiedEditorConfig::default());
        let editor_html = read_asset("index.html");
        #[cfg(not(feature = "blitz"))]
        let editor_css = editor_template.generate_css();

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
            #[cfg(not(feature = "blitz"))]
            js,
            #[cfg(not(feature = "blitz"))]
            layout_tree: None,
            #[cfg(not(feature = "blitz"))]
            font,
            editor_html,
            #[cfg(not(feature = "blitz"))]
            editor_css,
            #[cfg(not(feature = "blitz"))]
            cached_doc: None,
            #[cfg(not(feature = "blitz"))]
            cached_stylesheet: None,
            remote_cmd_rx,
            remote_ev_tx,
            pbr_target,
            composite_pass,
            interaction: InteractionState::new(),
        })
    }

    #[cfg(not(feature = "blitz"))]
    fn build_layout(ctx: &mut GameContext) {
        let doc = ctx.cached_doc.get_or_insert_with(|| ornis_ui::html::parse_html(&ctx.editor_html));
        let stylesheet = ctx.cached_stylesheet.get_or_insert_with(|| {
            ornis_ui::css::Stylesheet::parse(&ctx.editor_css)
                .unwrap_or_else(|_| ornis_ui::css::Stylesheet { rules: Vec::new(), custom_properties: std::collections::HashMap::new() })
        });

        let existing_cache = ctx.layout_tree.as_ref().map(|t| t.image_cache.clone());
        let closed_details = ctx.interaction.closed_details_indices();
        match LayoutTree::build_with_viewport(
            doc,
            &[stylesheet.clone()],
            ctx.surface_config.width as f32,
            ctx.surface_config.height as f32,
            &ctx.font,
            &closed_details,
            existing_cache,
        ) {
            Ok(tree) => ctx.layout_tree = Some(tree),
            Err(e) => eprintln!("ornis: editor layout build failed: {e:?}"),
        }
    }

    #[cfg(feature = "blitz")]
    fn build_layout(ctx: &mut GameContext) {
        let _ = ctx;
    }

    fn process_remote_commands(ctx: &mut GameContext) {
        while let Ok(cmd) = ctx.remote_cmd_rx.try_recv() {
            match cmd {
                UiCommand::Custom { cmd_type, json_data: _ } => {
                    match cmd_type.as_str() {
                        "create_entity" => {
                            #[cfg(not(feature = "blitz"))]
                            {
                                let id = ctx.js.bridge.create_entity();
                                ctx.remote_ev_tx.send(GameEvent::CustomEvent {
                                    cmd_type: "entity_created".into(),
                                    json_data: format!(r#"{{"entity_id":{id}}}"#),
                                }).ok();
                            }
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

        let stars = [(100.0, 80.0, 2.0), (300.0, 150.0, 1.5), (500.0, 60.0, 2.5),
                     (700.0, 200.0, 1.0), (200.0, 400.0, 2.0), (600.0, 500.0, 1.5)];
        for &(sx, sy, sr) in &stars {
            ctx.renderer.fill_circle(sx, sy, sr, Color::new([1.0, 1.0, 1.0, 0.3]));
        }
        for i in 0..20 {
            let y = (i as f64) * 60.0 + 30.0;
            ctx.renderer.stroke_rect(0.0, y, w, 1.0, 1.0, Color::new([1.0, 1.0, 1.0, 0.04]));
        }

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

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ornis frame"),
        });

        let frame = match ctx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated => {
                ctx.surface.configure(&ctx.device, &ctx.surface_config);
                return;
            }
            err => {
                if !matches!(err, wgpu::CurrentSurfaceTexture::Occluded) {
                    eprintln!("surface error: {err:?}");
                }
                return;
            }
        };
        let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let pbr_view = ctx.pbr_target.create_view(&wgpu::TextureViewDescriptor::default());

        let context = ornis_render::RenderContext {
            device: &ctx.device,
            queue: &ctx.queue,
            encoder: &mut encoder,
            target: &pbr_view,
        };

        ctx.renderer3d.render_scene(
            context,
            &ctx.sphere_mesh,
            ctx.instance_data.len() as u32,
        );

        let ui_view = ctx.renderer.get_internal_texture_view();
        ctx.composite_pass.compose(&ctx.device, &mut encoder, &frame_view, &pbr_view, &ui_view);

        ctx.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

#[cfg(not(feature = "editor-only"))]
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
                ctx.pbr_target = ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("pbr intermediate"),
                    size: wgpu::Extent3d {
                        width: size.width.max(1),
                        height: size.height.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: ctx.surface_config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                Self::build_layout(ctx);
                ctx.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let scale = ctx.window.scale_factor() as f32;
                let x = position.x as f32 / scale;
                let y = position.y as f32 / scale;
                ctx.interaction.last_mouse_pos = (x, y);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let (x, y) = ctx.interaction.last_mouse_pos;
                let btn = match button {
                    WinitMouseButton::Left => MouseButton::Left,
                    WinitMouseButton::Right => MouseButton::Right,
                    WinitMouseButton::Middle => MouseButton::Middle,
                    _ => MouseButton::Left,
                };
                if let Some(ref tree) = ctx.layout_tree {
                    match state {
                        winit::event::ElementState::Pressed => {
                            ctx.interaction.handle_mouse_down(tree, x, y, btn);
                        }
                        winit::event::ElementState::Released => {
                            if let Some(clicked) = ctx.interaction.handle_mouse_up(tree, x, y, btn) {
                                Self::handle_click(ctx, clicked);
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (dx, dy) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x * 20.0, y * 20.0),
                    MouseScrollDelta::PixelDelta(pos) => (pos.x as f32, pos.y as f32),
                };
                if let Some(node_id) = ctx.interaction.hovered {
                    ctx.interaction.scroll_node(node_id, dx, dy);
                    ctx.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event: KeyEvent { logical_key, state, text, .. }, .. } => {
                if state == winit::event::ElementState::Pressed {
                    #[cfg(not(feature = "blitz"))]
                    if let Some(focused_id) = ctx.interaction.focused {
                        if let Some(ref tree) = ctx.layout_tree {
                            let node = &tree.arena[focused_id];
                            if node.tag == "input" || node.tag == "textarea" {
                                let input = ctx.interaction.text_input_mut(focused_id);
                                match &logical_key {
                                    Key::Named(NamedKey::Backspace) => {
                                        input.backspace();
                                        ctx.window.request_redraw();
                                        return;
                                    }
                                    Key::Named(NamedKey::Delete) => {
                                        input.delete();
                                        ctx.window.request_redraw();
                                        return;
                                    }
                                    Key::Named(NamedKey::ArrowLeft) => {
                                        input.move_caret(-1);
                                        ctx.window.request_redraw();
                                        return;
                                    }
                                    Key::Named(NamedKey::ArrowRight) => {
                                        input.move_caret(1);
                                        ctx.window.request_redraw();
                                        return;
                                    }
                                    Key::Named(NamedKey::Home) => {
                                        input.move_caret_home();
                                        ctx.window.request_redraw();
                                        return;
                                    }
                                    Key::Named(NamedKey::End) => {
                                        input.move_caret_end();
                                        ctx.window.request_redraw();
                                        return;
                                    }
                                    _ => {}
                                }
                                if let Some(t) = text {
                                    input.insert(t.as_str());
                                    ctx.window.request_redraw();
                                    return;
                                }
                            }
                        }
                    }

                    let key_str = match &logical_key {
                        Key::Named(NamedKey::Escape) => "Escape",
                        Key::Named(NamedKey::F1) => "F1",
                        Key::Character(c) => c.as_str(),
                        _ => return,
                    };
                    #[cfg(not(feature = "blitz"))]
                    if key_str == "`" || key_str == "Backtick" || key_str == "F1" {
                        let _ = ctx.js.eval("window.OrnisEditor?.toggle?.()");
                        Self::build_layout(ctx);
                        ctx.window.request_redraw();
                    } else if key_str == "Escape" {
                        let _ = ctx.js.eval("window.OrnisEditor?.close?.()");
                        Self::build_layout(ctx);
                        ctx.window.request_redraw();
                    } else if key_str == "1" {
                        let _ = ctx.js.eval("window.OrnisEditor?.setGizmoMode?.('translate')");
                    } else if key_str == "2" {
                        let _ = ctx.js.eval("window.OrnisEditor?.setGizmoMode?.('rotate')");
                    } else if key_str == "3" {
                        let _ = ctx.js.eval("window.OrnisEditor?.setGizmoMode?.('scale')");
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ctx) = &mut self.context {
            Self::process_remote_commands(ctx);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.context = None;
    }
}

#[cfg(not(feature = "editor-only"))]
impl GameApp {
    fn handle_click(ctx: &mut GameContext, node_id: ornis_ui::layout::LayoutNodeId) {
        let tree = match ctx.layout_tree.as_ref() {
            Some(t) => t,
            None => return,
        };

        let mut cur = Some(node_id);
        while let Some(cid) = cur {
            let node = &tree.arena[cid];
            if node.tag == "summary" {
                if let Some(pid) = node.parent {
                    if let Some(details_node) = tree.arena.get(pid) {
                        if details_node.tag == "details" {
                            if let Some(didx) = details_node.details_index {
                                ctx.interaction.toggle_details(didx);
                                let closed = ctx.interaction.closed_details_indices();
                                if let Some(layout) = &mut ctx.layout_tree {
                                    let _ = layout.apply_details_states(&closed, &ctx.font);
                                }
                                ctx.window.request_redraw();
                            }
                            break;
                        }
                    }
                }
            }
            if node.tag == "input" || node.tag == "textarea" {
                if ctx.interaction.text_input_ref(cid).is_none() {
                    let val = node.attributes.get("value").cloned().unwrap_or_default();
                    ctx.interaction.set_input_value(cid, val);
                }
                break;
            }
            cur = node.parent;
        }
    }
}

#[cfg(not(feature = "editor-only"))]
fn entity_count(ctx: &GameContext) -> u32 {
    ctx.js.bridge.entity_count()
}

#[cfg(not(feature = "editor-only"))]
#[cfg(not(feature = "blitz"))]
fn load_font() -> FontData {
    let inter = ornis_ui::text::load_inter_font();
    let _ = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ];
    inter
}

#[cfg(not(feature = "editor-only"))]
fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = GameApp::new();
    event_loop.run_app(&mut app).unwrap();
}
