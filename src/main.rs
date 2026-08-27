//! Бинарь `ornis`: нативный режим (winit + wgpu) и `editor-only`
//! (HTTP-сервер редактора на порту 3420 без нативного окна).

#![warn(missing_docs)]

use crossbeam_channel::unbounded;
use editor_backend::RemoteEditor;
// Only the native-mode loop types the channels explicitly.
#[cfg(not(feature = "editor-only"))]
use editor_backend::{GameEvent, UiCommand};
#[cfg(not(feature = "editor-only"))]
use engine_runtime::{RenderExtracted, install_render_extract};

// Compiled in both modes so its unit tests run under a plain `cargo test`;
// in native mode nothing calls it yet (the native loop is a counter stub).
#[cfg_attr(not(feature = "editor-only"), allow(dead_code))]
mod editor_world;
mod engine_runtime;

// ═══════════════════════════════════════════════════════════════════════════
// "BROWSER-ONLY EDITOR" MODE (editor-only)
// ═══════════════════════════════════════════════════════════════════════════
// Run with: cargo run --features editor-only
// No native winit window is created; only the RemoteEditor HTTP server
// on port 3420 runs. The developer opens http://127.0.0.1:3420 in a
// browser and gets the full editor.
//
// Strategic pivot from native UI to the browser editor
// (July 2026, see PLAN.md). The native UI crate was removed (August 2026).
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "editor-only")]
fn main() {
    let (cmd_tx, cmd_rx) = unbounded();
    let (ev_tx, ev_rx) = unbounded();

    // Live ECS world on a dedicated thread: executes commands from
    // POST /api/command and publishes status/scene snapshots + events.
    editor_world::run(cmd_rx, ev_tx);

    // The binding keeps RemoteEditor alive until main ends (Drop stops the server).
    let _editor = RemoteEditor::start(3420, cmd_tx, ev_rx);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Ornis Engine — Browser Editor Mode                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Editor:  http://127.0.0.1:3420                               ║");
    println!("║  Status:  http://127.0.0.1:3420/api/status                    ║");
    println!("║  Scene:   http://127.0.0.1:3420/api/scene                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Press Ctrl+C to stop                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Wait forever — the server runs on a separate thread.
    loop {
        std::thread::park();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FULL ENGINE WITH A NATIVE WINDOW (default mode)
// ═══════════════════════════════════════════════════════════════════════════
// Run with: cargo run
// winit window, wgpu rendering, a 3D scene of spheres (OpenPBR).
// The browser editor server is opt-in here: `cargo run -- --remote-editor`
// serves it on port 3420 (off by default — audit §6.2, backlog #16).
// The native UI overlay was removed with the ornis-ui crate — the editor
// lives in the browser (see editor-only mode / cargo xtask editor).
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

    pub use ornis_core::Engine;
    pub use ornis_render::scene::{MaterialDesc, MeshDesc, TransformDesc};
    pub use ornis_render::{
        Mesh, RenderBackend, RenderBackendConfig, create_render_backend, create_sphere,
    };
}

#[cfg(not(feature = "editor-only"))]
use native::*;

#[cfg(not(feature = "editor-only"))]
struct GameApp {
    context: Option<GameContext>,
    remote_editor: Option<RemoteEditor>,
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
    engine: Engine,
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
            MaterialDesc::Dielectric {
                base_color: [0.8, 0.2, 0.2],
                roughness: 0.5,
            },
            MaterialDesc::Dielectric {
                base_color: [0.2, 0.8, 0.2],
                roughness: 0.7,
            },
            MaterialDesc::Dielectric {
                base_color: [0.2, 0.2, 0.8],
                roughness: 0.1,
            },
            MaterialDesc::Metal {
                base_color: [0.9, 0.7, 0.1],
                roughness: 0.2,
            },
            MaterialDesc::Coat {
                base_color: [0.9, 0.9, 0.9],
                coat_weight: 1.0,
                coat_roughness: 0.1,
            },
        ];
        let entity_count = materials.len() as u32;
        let mut engine = Engine::new();
        {
            let store = engine.world_mut().store_mut().expect("world store");
            for (i, material) in materials.into_iter().enumerate() {
                let entity = store.create_entity();
                let x = (i as f32 - 2.0) * 2.8;
                store.insert(
                    entity,
                    TransformDesc {
                        translation: [x, 0.0, 0.0],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        scale: [1.0, 1.0, 1.0],
                    },
                );
                store.insert(
                    entity,
                    MeshDesc::Sphere {
                        radius: 1.0,
                        segments: 32,
                        rings: 24,
                    },
                );
                store.insert(entity, material);
            }
        }
        install_render_extract(&mut engine);

        Ok(GameContext {
            window,
            device,
            queue,
            surface,
            surface_config,
            renderer3d,
            sphere_mesh,
            engine,
            remote_cmd_rx,
            remote_ev_tx,
            entity_count,
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
        ctx.engine.run_frame(1.0 / 60.0);
        let extracted = ctx
            .engine
            .world()
            .resources()
            .get::<std::sync::Mutex<RenderExtracted>>()
            .expect("render extraction resource")
            .lock()
            .expect("render extraction lock")
            .clone();
        if extracted.mesh_params != (32, 24) {
            ctx.sphere_mesh = create_sphere(
                &ctx.device,
                1.0,
                extracted.mesh_params.0,
                extracted.mesh_params.1,
            );
        }

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
        ctx.renderer3d
            .upload_materials(&ctx.queue, &extracted.materials);
        ctx.renderer3d
            .upload_instances(&ctx.queue, &extracted.instances);

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
            .render_scene(context, &ctx.sphere_mesh, extracted.instances.len() as u32);

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
            // The remote editor server is a dev-tool: opt-in in native
            // mode; when off, the channels simply idle (`process_remote_commands`
            // polls an empty/disconnected receiver, sends are `.ok()`-dropped).
            if remote_editor_requested() {
                self.remote_editor = Some(RemoteEditor::start(3420, cmd_tx, ev_rx));
            }
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

/// Native mode: the remote editor HTTP server is opt-in (`--remote-editor`,
/// serves on port 3420), not on by default — the engine binary runs fine
/// without the editor dev-tool (audit §6.2, backlog #16). The `editor-only`
/// mode always serves it: that mode IS the editor.
#[cfg(not(feature = "editor-only"))]
fn remote_editor_requested() -> bool {
    std::env::args().any(|a| a == "--remote-editor")
}

#[cfg(not(feature = "editor-only"))]
fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = GameApp::new();
    event_loop.run_app(&mut app).unwrap();
}
