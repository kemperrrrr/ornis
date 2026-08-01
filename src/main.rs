use crossbeam_channel::unbounded;

mod ipc;
mod remote;

// ═══════════════════════════════════════════════════════════════════════════
// РЕЖИМ "ТОЛЬКО РЕДАКТОР В БРАУЗЕРЕ" (editor-only)
// ═══════════════════════════════════════════════════════════════════════════
// Запуск: cargo run --features editor-only
// При этом нативное winit-окно НЕ создаётся. Работает только HTTP-сервер
// RemoteEditor на порту 3420. Разработчик открывает браузер по адресу
// http://127.0.0.1:3420 и получает полноценный редактор.
//
// См. docs/archive/STRATEGY_PIVOT.md — стратегический поворот от нативного UI
// к браузерному редактору (июль 2026). Нативный UI-крейт удалён (август 2026).
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
// winit-окно, wgpu-рендеринг, 3D-сцена со сферами (OpenPBR).
// RemoteEditor тоже работает на порту 3420.
// Нативный UI overlay удалён вместе с крейтом ornis-ui — редактор живёт
// в браузере (см. режим editor-only / cargo xtask editor).
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "editor-only"))]
mod native {
    pub use crossbeam_channel::{Receiver, Sender};
    pub use glam::{Mat4, Vec3};
    pub use winit::application::ApplicationHandler;
    pub use winit::dpi::PhysicalSize;
    pub use winit::event::WindowEvent;
    pub use winit::event_loop::{ActiveEventLoop, EventLoop};
    pub use winit::window::WindowAttributes;

    pub use ornis_render::{
        InstanceData, Mesh, OpenPBRMaterial, RenderBackend, RenderBackendConfig,
        create_render_backend, create_sphere,
    };

    pub use crate::ipc::{GameEvent, UiCommand};
}

#[cfg(not(feature = "editor-only"))]
use native::*;

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
    renderer3d: Box<dyn RenderBackend>,
    sphere_mesh: Mesh,
    materials: Vec<OpenPBRMaterial>,
    instance_data: Vec<InstanceData>,
    remote_cmd_rx: Receiver<UiCommand>,
    remote_ev_tx: Sender<GameEvent>,
    entity_count: u32,
}

#[cfg(not(feature = "editor-only"))]
impl GameApp {
    fn new() -> Self {
        GameApp {
            context: None,
            remote_editor: None,
        }
    }

    fn context(&mut self) -> Option<&mut GameContext> {
        self.context.as_mut()
    }

    fn initialize(
        event_loop: &ActiveEventLoop,
        remote_cmd_rx: Receiver<UiCommand>,
        remote_ev_tx: Sender<GameEvent>,
    ) -> Result<GameContext, String> {
        let window_attrs = WindowAttributes::default()
            .with_title("Ornis Engine")
            .with_inner_size(PhysicalSize::new(800, 600));
        let window = event_loop
            .create_window(window_attrs)
            .map_err(|e| format!("window creation: {e}"))?;

        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let surface_target =
            unsafe { wgpu::SurfaceTargetUnsafe::from_display_and_window(event_loop, &window) }
                .map_err(|e| format!("surface target: {e}"))?;
        let surface: wgpu::Surface<'static> = unsafe {
            instance
                .create_surface_unsafe(surface_target)
                .map_err(|e| format!("surface creation: {e}"))?
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .map_err(|_| "no adapter found".to_string())?;

        let mut limits = adapter.limits();
        limits.max_storage_buffers_per_shader_stage = 8;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ornis device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("device request: {e}"))?;

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
            .unwrap_or_else(|| {
                surface_caps
                    .formats
                    .first()
                    .copied()
                    .unwrap_or(wgpu::TextureFormat::Rgba8UnormSrgb)
            });

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

        let backend_config = RenderBackendConfig {
            surface_config: surface_config.clone(),
            sample_count: 1,
            max_objects: 256,
            max_materials: 64,
        };
        let renderer3d: Box<dyn RenderBackend> = create_render_backend(&device, &backend_config);
        let sphere_mesh = create_sphere(&device, 1.0, 32, 24);

        let materials = vec![
            OpenPBRMaterial::dielectric().base_color_rgb([0.8, 0.2, 0.2]),
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
        let instance_data: Vec<InstanceData> = materials
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let x = (i as f32 - 2.0) * spacing;
                let pos = Vec3::new(x, 0.0, 0.0);
                let model = Mat4::from_translation(pos);
                let normal_matrix = Mat4::IDENTITY;
                InstanceData {
                    model_matrix: model,
                    normal_matrix,
                    material_index: i as u32,
                }
            })
            .collect();

        Ok(GameContext {
            window,
            device,
            queue,
            surface,
            surface_config,
            renderer3d,
            sphere_mesh,
            materials,
            instance_data,
            remote_cmd_rx,
            remote_ev_tx,
            entity_count: 0,
        })
    }

    fn process_remote_commands(ctx: &mut GameContext) {
        while let Ok(cmd) = ctx.remote_cmd_rx.try_recv() {
            if let UiCommand::Custom {
                cmd_type,
                json_data: _,
            } = cmd
            {
                match cmd_type.as_str() {
                    "create_entity" => {
                        ctx.entity_count += 1;
                        let id = ctx.entity_count;
                        ctx.remote_ev_tx
                            .send(GameEvent::CustomEvent {
                                cmd_type: "entity_created".into(),
                                json_data: format!(r#"{{"entity_id":{id}}}"#),
                            })
                            .ok();
                    }
                    "list_entities" => {
                        ctx.remote_ev_tx
                            .send(GameEvent::CustomEvent {
                                cmd_type: "entity_list".into(),
                                json_data: format!(r#"{{"count":{}}}"#, ctx.entity_count),
                            })
                            .ok();
                    }
                    _ => {}
                }
            }
        }
    }

    fn render_frame(ctx: &mut GameContext) {
        let w = ctx.surface_config.width as f64;
        let h = ctx.surface_config.height as f64;
        let aspect = w as f32 / h as f32;

        let view = Mat4::look_at_rh(Vec3::new(0.0, 2.5, 9.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, aspect, 0.1, 100.0);
        let view_proj = proj * view;

        ctx.renderer3d
            .set_camera(&ctx.queue, &view_proj.to_cols_array_2d(), [0.0, 2.5, 9.0]);
        ctx.renderer3d.set_lights(
            &ctx.queue,
            [0.10, 0.10, 0.15],
            &[
                ([1.0, 1.0, 1.0], 0.6, [1.0, 1.0, 1.0]),
                ([-0.5, 0.5, -0.5], 0.3, [0.8, 0.8, 1.0]),
            ],
        );
        ctx.renderer3d.upload_materials(&ctx.queue, &ctx.materials);
        ctx.renderer3d
            .upload_instances(&ctx.queue, &ctx.instance_data);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let context = ornis_render::RenderContext {
            device: &ctx.device,
            queue: &ctx.queue,
            encoder: &mut encoder,
            target: &frame_view,
        };

        ctx.renderer3d
            .render_scene(context, &ctx.sphere_mesh, ctx.instance_data.len() as u32);

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
                Ok(ctx) => {
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
                ctx.renderer3d
                    .resize(&ctx.device, size.width.max(1), size.height.max(1));
                ctx.window.request_redraw();
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

#[cfg(not(feature = "editor-only"))]
fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = GameApp::new();
    event_loop.run_app(&mut app).unwrap();
}
