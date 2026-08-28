//! Бинарь `ornis`: нативный режим (winit + wgpu) и `editor-only`
//! (HTTP-сервер редактора на порту 3420 без нативного окна).

#![warn(missing_docs)]

use crossbeam_channel::unbounded;
use editor_backend::RemoteEditor;
// Only the native-mode loop types the channels explicitly.
#[cfg(not(feature = "editor-only"))]
use editor_backend::{GameEvent, UiCommand};
#[cfg(not(feature = "editor-only"))]
use engine_runtime::install_physics;

// Compiled in both modes so its unit tests run under a plain `cargo test`;
// native mode also installs the physics systems into the showcase Engine.
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
    pub use ornis_core::InputState;
    pub use ornis_physics::RigidBody;
    pub use winit::application::ApplicationHandler;
    pub use winit::dpi::PhysicalSize;
    pub use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
    pub use winit::event_loop::{ActiveEventLoop, EventLoop};
    pub use winit::keyboard::PhysicalKey;
    pub use winit::window::WindowAttributes;

    pub use ornis_render::scene::Scene;
    pub use ornis_render::{
        install_orbit_camera, read_orbit_camera, Mesh, OrbitCamera, RenderContext, RenderFrame3D,
        RenderWorld, Renderer3D, Technique, create_sphere,
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
    renderer3d: Renderer3D,
    frame_plan: RenderFrame3D,
    sphere_mesh: Mesh,
    render_world: RenderWorld,
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

        let renderer3d = Renderer3D::new(&device, &surface_config, 1);
        let frame_plan = RenderFrame3D::new_with(
            surface_format,
            (surface_config.width, surface_config.height),
            Technique::Hybrid,
            false,
        );
        let sphere_mesh = create_sphere(&device, 1.0, 32, 24);

        let (render_world, entity_count) = Self::showcase_engine();

        Ok(GameContext {
            window,
            device,
            queue,
            surface,
            surface_config,
            renderer3d,
            frame_plan,
            sphere_mesh,
            render_world,
            remote_cmd_rx,
            remote_ev_tx,
            entity_count,
        })
    }

    fn showcase_engine() -> (RenderWorld, u32) {
        let scene = Scene::from_ron(include_str!("../assets/scene.ron"))
            .expect("shipped showcase scene must parse");
        let entity_count = scene.entities.len() as u32;
        let mut render_world = RenderWorld::from_scene(&scene);
        install_orbit_camera(
            render_world.engine_mut(),
            OrbitCamera::from_desc(&scene.camera),
        );
        install_physics(render_world.engine_mut(), Vec3::new(0.0, -9.81, 0.0));
        {
            let entities = render_world.entities().to_vec();
            let store = render_world
                .engine_mut()
                .world_mut()
                .store_mut()
                .expect("render world store");
            for (index, entity) in entities.into_iter().enumerate() {
                let description = &scene.entities[index];
                let radius = match &description.mesh {
                    ornis_render::scene::MeshDesc::Sphere { radius, .. } => *radius,
                };
                let mass = if index == 0 { 1.0 } else { 0.0 };
                store.insert(
                    entity,
                    RigidBody::new_sphere(
                        Vec3::from_array(description.transform.translation),
                        radius,
                        mass,
                    ),
                );
            }
            // Hidden static floor: it has a physics component but no render
            // components, so it does not enter RenderExtracted.
            let floor = store.create_entity();
            store.insert(
                floor,
                RigidBody::new_box(Vec3::new(0.0, -2.0, 0.0), Vec3::new(20.0, 1.0, 20.0), 0.0),
            );
        }
        render_world.run_frame(0.0);
        (render_world, entity_count)
    }

    fn update_input(ctx: &mut GameContext, update: impl FnOnce(&mut InputState)) {
        if let Some(input) = ctx
            .render_world
            .engine_mut()
            .world_mut()
            .resources_mut()
            .get_mut::<InputState>()
        {
            update(input);
        }
    }

    fn process_remote_commands(ctx: &mut GameContext) {
        while let Ok(command) = ctx.remote_cmd_rx.try_recv() {
            let (request_id, command) = match command {
                UiCommand::WithRequestId {
                    request_id,
                    command,
                } => (Some(request_id), *command),
                command => (None, command),
            };
            let success = Self::execute_native_command(ctx, &command);
            if let Some(request_id) = request_id {
                ctx.remote_ev_tx
                    .send(GameEvent::CommandCompleted {
                        request_id,
                        command: Self::command_name(&command),
                        success,
                        error: (!success).then_some("native showcase command is a stub".into()),
                    })
                    .ok();
            }
        }
    }

    fn execute_native_command(ctx: &mut GameContext, command: &UiCommand) -> bool {
        let UiCommand::Custom {
            cmd_type,
            json_data: _,
        } = command
        else {
            return false;
        };
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
                true
            }
            "list_entities" => {
                ctx.remote_ev_tx
                    .send(GameEvent::CustomEvent {
                        cmd_type: "entity_list".into(),
                        json_data: format!(r#"{{"count":{}}}"#, ctx.entity_count),
                    })
                    .ok();
                true
            }
            _ => false,
        }
    }

    fn command_name(command: &UiCommand) -> String {
        match command {
            UiCommand::CreateEntity => "create_entity".into(),
            UiCommand::DestroyEntity { .. } => "destroy_entity".into(),
            UiCommand::SetComponent { .. } => "set_component".into(),
            UiCommand::Custom { cmd_type, .. } => cmd_type.clone(),
            UiCommand::WithRequestId { command, .. } => Self::command_name(command),
        }
    }

    fn render_frame(ctx: &mut GameContext) {
        ctx.render_world.run_frame(1.0 / 60.0);
        let extracted = ctx.render_world.extracted();
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

        let orbit = read_orbit_camera(ctx.render_world.engine())
            .expect("native showcase installs orbit camera");
        let (cam_pos, cam_target, cam_up, fov, near, far) = orbit.view_parameters();
        let view = Mat4::look_at_rh(cam_pos, cam_target, cam_up);
        let proj = Mat4::perspective_rh(fov.to_radians(), aspect, near, far);
        let view_proj = proj * view;

        ctx.renderer3d.set_camera(
            &ctx.queue,
            &view_proj.to_cols_array_2d(),
            cam_pos.to_array(),
        );
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

        let context = RenderContext {
            device: &ctx.device,
            queue: &ctx.queue,
            encoder: &mut encoder,
            target: &frame_view,
        };

        ctx.frame_plan.render(
            context,
            &ctx.renderer3d,
            &ctx.sphere_mesh,
            extracted.instances.len() as u32,
        );

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
                ctx.frame_plan
                    .set_surface_size(size.width.max(1), size.height.max(1));
                ctx.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = matches!(event.state, ElementState::Pressed);
                if let PhysicalKey::Code(code) = event.physical_key {
                    Self::update_input(ctx, |input| input.set_key(code as u32, pressed));
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let code = match button {
                    MouseButton::Left => 0,
                    MouseButton::Right => 1,
                    MouseButton::Middle => 2,
                    MouseButton::Back => 3,
                    MouseButton::Forward => 4,
                    MouseButton::Other(code) => code.min(u16::from(u8::MAX)) as u8,
                };
                let pressed = matches!(state, ElementState::Pressed);
                Self::update_input(ctx, |input| {
                    if pressed {
                        input.clear_frame_transients();
                    }
                    input.set_mouse_button(code, pressed);
                });
            }
            WindowEvent::CursorMoved { position, .. } => {
                Self::update_input(ctx, |input| {
                    input.set_pointer_position([position.x as f32, position.y as f32]);
                });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 100.0,
                };
                Self::update_input(ctx, |input| input.add_wheel_delta(amount));
            }
            WindowEvent::Focused(false) => {
                Self::update_input(ctx, InputState::clear_all);
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
