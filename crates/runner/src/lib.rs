//! Native shell: winit window + wgpu rendering + frame loop over one
//! [`GameWorld`](ornis_app::GameWorld).
//!
//! Extracted from the `ornis` binary so games and examples reuse the same
//! shell instead of forking it: the caller builds a world (spheres
//! showcase, animation demo, real game) and hands it to
//! [`run_native`]. Window creation, GPU init, input routing, the
//! opt-in remote-editor server and the `--frames` smoke-test budget all
//! live here; world content stays with the caller.

#![warn(missing_docs)]

use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use ornis_app::GameWorld;
use ornis_app::session::clamp_frame_dt;
use ornis_core::InputState;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::WindowAttributes;

/// Default editor HTTP port (loopback).
pub const EDITOR_HTTP_PORT: u16 = 3420;
/// Native window width (px).
pub const WINDOW_WIDTH: u32 = 800;
/// Native window height (px).
pub const WINDOW_HEIGHT: u32 = 600;
/// Storage buffers required by the render shader stages.
const MAX_STORAGE_BUFFERS_PER_STAGE: u32 = 8;
/// Pixel-delta → line-delta scale for mouse wheel.
const WHEEL_PIXELS_PER_LINE: f32 = 100.0;
/// Raw wire code for mouse Back.
const MOUSE_BACK: u8 = 3;
/// Raw wire code for mouse Forward.
const MOUSE_FORWARD: u8 = 4;

/// World builder handed to [`run_native`]: a ready-to-run [`GameWorld`]
/// plus its entity count (window title/diagnostics).
pub type BuildWorld = Box<dyn FnOnce() -> (GameWorld, u32)>;

/// Native run options: window title plus the two CLI conventions
/// (`--remote-editor`, `--frames N`); see [`NativeOptions::from_env`].
pub struct NativeOptions {
    /// Window title.
    pub title: &'static str,
    /// Smoke-test budget: exit after N presented frames (`None` = forever).
    pub frames: Option<u64>,
    /// Serve the browser editor alongside the native window.
    pub remote_editor: bool,
}

impl NativeOptions {
    /// Parses the shell CLI conventions with the given window title:
    /// `--remote-editor` serves the editor on
    /// [`EDITOR_HTTP_PORT`], `--frames N` exits after N frames.
    pub fn from_env(title: &'static str) -> Self {
        let mut args = std::env::args();
        let mut frames = None;
        let mut remote_editor = false;
        while let Some(arg) = args.next() {
            if arg == "--remote-editor" {
                remote_editor = true;
            } else if arg == "--frames" {
                frames = args.next().and_then(|n| n.parse().ok());
            }
        }
        Self {
            title,
            frames,
            remote_editor,
        }
    }
}

/// Runs one [`GameWorld`] in a native window: builds the world via
/// `build`, opens the window, and pumps frames until close (or the
/// `--frames` budget runs out).
///
/// # Errors
///
/// When the event loop or GPU init fails; runtime frame errors never
/// surface here (initialization failures print to stderr instead).
pub fn run_native(
    build: impl FnOnce() -> (GameWorld, u32) + 'static,
    options: NativeOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = GameApp {
        context: None,
        remote_editor: None,
        build: Some(Box::new(build)),
        options,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct GameApp {
    context: Option<GameContext>,
    remote_editor: Option<editor_backend::RemoteEditor>,
    build: Option<BuildWorld>,
    options: NativeOptions,
}

struct GameContext {
    window: winit::window::Window,
    runtime: GameWorld,
    remote_cmd_rx: Receiver<editor_backend::ipc::UiCommand>,
    remote_ev_tx: Sender<editor_backend::ipc::GameEvent>,
    entity_count: u32,
    /// Wall clock of the last presented frame: `render_frame` measures the
    /// real delta against it (clamped, see [`clamp_frame_dt`]).
    last_frame: Instant,
    /// `--frames N` smoke-test budget: `None` runs forever.
    frames_left: Option<u64>,
}

impl GameApp {
    fn context(&mut self) -> Option<&mut GameContext> {
        self.context.as_mut()
    }

    #[allow(clippy::too_many_lines)]
    fn initialize(
        event_loop: &ActiveEventLoop,
        remote_cmd_rx: Receiver<editor_backend::ipc::UiCommand>,
        remote_ev_tx: Sender<editor_backend::ipc::GameEvent>,
        build: BuildWorld,
        title: &'static str,
        frames: Option<u64>,
    ) -> Result<GameContext, String> {
        let window_attrs = WindowAttributes::default()
            .with_title(title)
            .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));
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
            apply_limit_buckets: false,
        }))
        .map_err(|_| "no adapter found".to_string())?;

        let mut limits = adapter.limits();
        limits.max_storage_buffers_per_shader_stage = MAX_STORAGE_BUFFERS_PER_STAGE;
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
            color_space: wgpu::SurfaceColorSpace::Auto,
        };

        // Initial configure: without it the first acquire hits a
        // validation error instead of the Outdated/Lost recovery path.
        surface.configure(&device, &surface_config);

        let renderer3d = ornis_render::Renderer3D::new(&device, &surface_config, 1);
        let frame3d = ornis_render::RenderFrame3D::new_with(
            surface_format,
            (surface_config.width, surface_config.height),
            ornis_render::Technique::Hybrid,
            ornis_render::Bloom::Off,
        );
        let sphere_mesh = ornis_render::create_sphere(
            &device,
            1.0,
            DEFAULT_SPHERE_SEGMENTS,
            DEFAULT_SPHERE_RINGS,
        );

        let (mut runtime, entity_count) = build();
        // S7: GPU state lives as Engine resources, RenderSubmit/RenderPresent run in schedule.
        {
            use ornis_render::gpu_resources::{
                GpuFrameState, GpuMesh, GpuSurfaceState, install_gpu_resources,
            };
            install_gpu_resources(
                runtime.engine_mut(),
                device.clone(),
                queue.clone(),
                surface,
                GpuSurfaceState {
                    size: (surface_config.width, surface_config.height),
                    format: surface_format,
                },
                GpuFrameState {
                    renderer: renderer3d,
                    frame3d,
                },
                GpuMesh {
                    mesh: sphere_mesh,
                    params: (DEFAULT_SPHERE_SEGMENTS, DEFAULT_SPHERE_RINGS),
                },
            );
        }

        Ok(GameContext {
            window,
            runtime,
            remote_cmd_rx,
            remote_ev_tx,
            entity_count,
            last_frame: Instant::now(),
            frames_left: frames,
        })
    }

    fn render_frame(ctx: &mut GameContext) {
        // S7 step 2: the whole GPU frame (upload + acquire → record → submit → present)
        // runs in Engine::schedule as RenderSubmit/RenderPresent. Only the
        // frame stays here (fixed + variable schedules + CPU extraction); the Present
        // system acquires via surface.get_current_texture and renders via frame3d itself.
        //
        // Vsync is OFF: the surface is configured with
        // `wgpu::PresentMode::AutoNoVsync` (see `initialize` and the
        // `Resized` handler), so frames are unthrottled and the wall-clock
        // interval varies with load. The simulation therefore measures the
        // real delta since the last frame (clamped against hitches by
        // `clamp_frame_dt`, at most ~6 fixed steps) instead of assuming
        // 1/60 s — sim speed is independent of FPS; the engine's bounded
        // fixed accumulator absorbs the residual jitter.
        let elapsed = ctx.last_frame.elapsed();
        ctx.last_frame = Instant::now();
        ctx.runtime.frame_secs(clamp_frame_dt(elapsed));
    }
}

/// Default procedural sphere resolution (matches the showcase): sector count.
const DEFAULT_SPHERE_SEGMENTS: u32 = 32;
/// Default procedural sphere resolution (matches the showcase): stack count.
const DEFAULT_SPHERE_RINGS: u32 = 24;

impl GameApp {
    fn update_input(ctx: &mut GameContext, update: impl FnOnce(&mut InputState)) {
        if let Some(input) = ctx
            .runtime
            .engine_mut()
            .world_mut()
            .resources_mut()
            .get_mut::<InputState>()
        {
            update(input);
        }
    }

    fn process_remote_commands(ctx: &mut GameContext) {
        use editor_backend::ipc::{GameEvent, UiCommand};
        while let Ok(command) = ctx.remote_cmd_rx.try_recv() {
            // Browser input channel (WS bidirectionally + POST /api/input):
            // replace authoritative InputState in the unified World (single
            // World/Engine/Schedule, no polling / scene.ron fallback). The
            // unwrap path handles WithRequestId(Input) for completeness even
            // though the transport never wraps input.
            let input = match &command {
                UiCommand::Input { input } => Some(input.clone()),
                UiCommand::WithRequestId { command, .. } => match &**command {
                    UiCommand::Input { input } => Some(input.clone()),
                    _ => None,
                },
                _ => None,
            };
            if let Some(input) = input {
                Self::apply_browser_input(ctx, &input);
                continue;
            }
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

    fn execute_native_command(
        ctx: &mut GameContext,
        command: &editor_backend::ipc::UiCommand,
    ) -> bool {
        use editor_backend::ipc::UiCommand;
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
                    .send(editor_backend::ipc::GameEvent::CustomEvent {
                        cmd_type: "entity_created".into(),
                        json_data: format!(r#"{{"entity_id":{id}}}"#),
                    })
                    .ok();
                true
            }
            "list_entities" => {
                ctx.remote_ev_tx
                    .send(editor_backend::ipc::GameEvent::CustomEvent {
                        cmd_type: "entity_list".into(),
                        json_data: format!(r#"{{"count":{}}}"#, ctx.entity_count),
                    })
                    .ok();
                true
            }
            _ => false,
        }
    }

    fn command_name(
        command: &editor_backend::ipc::UiCommand,
    ) -> editor_backend::ipc::EditorCommand {
        use editor_backend::ipc::EditorCommand;
        use editor_backend::ipc::UiCommand;
        match command {
            UiCommand::CreateEntity => EditorCommand::CreateEntity,
            UiCommand::DestroyEntity { .. } => EditorCommand::DestroyEntity,
            UiCommand::SetComponent { .. } => EditorCommand::SetComponent,
            UiCommand::Custom { cmd_type, .. } => cmd_type.clone(),
            UiCommand::Input { .. } => EditorCommand::Input,
            UiCommand::WithRequestId { command, .. } => Self::command_name(command),
        }
    }

    fn apply_browser_input(ctx: &mut GameContext, input: &editor_backend::BrowserInput) {
        let world = ctx.runtime.engine_mut().world_mut();
        let state = world.resources_mut().get_mut::<InputState>();
        // Always ensure the resource exists even if never initialized elsewhere.
        let state = if let Some(s) = state {
            s
        } else {
            world.resources_mut().insert(InputState::default());
            let Some(s) = world.resources_mut().get_mut::<InputState>() else {
                return;
            };
            s
        };
        state.apply_snapshot(
            &input.pressed_keys,
            &input.pressed_mouse_buttons,
            input.pointer_position,
            input.pointer_delta,
            input.wheel_delta,
        );
    }
}

impl ApplicationHandler for GameApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
            let (ev_tx, ev_rx) = crossbeam_channel::unbounded();
            // The remote editor server is a dev-tool: opt-in in native
            // mode; when off, the channels simply idle (`process_remote_commands`
            // polls an empty/disconnected receiver, sends are `.ok()`-dropped).
            if self.options.remote_editor {
                self.remote_editor = Some(editor_backend::RemoteEditor::start(
                    EDITOR_HTTP_PORT,
                    cmd_tx,
                    ev_rx,
                ));
            }
            let build = self.build.take();
            let options_title = self.options.title;
            let options_frames = self.options.frames;
            match build {
                Some(build) => match Self::initialize(
                    event_loop,
                    cmd_rx,
                    ev_tx,
                    build,
                    options_title,
                    options_frames,
                ) {
                    Ok(ctx) => {
                        self.context = Some(ctx);
                    }
                    Err(e) => {
                        eprintln!("ornis: failed to initialize: {e}");
                    }
                },
                None => eprintln!("ornis: resumed without a world builder"),
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
                let (w, h) = (size.width.max(1), size.height.max(1));
                // Sync the ECS resources with the new size — reconfigure the
                // Surface (held in a resource) and update GpuFrameState.
                let device = ctx
                    .runtime
                    .engine()
                    .world()
                    .resources()
                    .get::<ornis_render::gpu_resources::GpuDevice>()
                    .map(|d| d.0.clone());
                if let Some(state) = ctx
                    .runtime
                    .engine_mut()
                    .world_mut()
                    .resources_mut()
                    .get_mut::<ornis_render::gpu_resources::GpuSurfaceState>()
                {
                    state.size = (w, h);
                }
                if let (Some(device), Some(surface)) = (
                    device,
                    ctx.runtime
                        .engine()
                        .world()
                        .resources()
                        .get::<ornis_render::gpu_resources::GpuSurface>(),
                ) {
                    let guard = surface.0.lock().unwrap_or_else(|e| e.into_inner());
                    // Take the format from the updated GpuSurfaceState.
                    let format = ctx
                        .runtime
                        .engine()
                        .world()
                        .resources()
                        .get::<ornis_render::gpu_resources::GpuSurfaceState>()
                        .map(|s| s.format)
                        .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);
                    guard.configure(
                        &device,
                        &wgpu::SurfaceConfiguration {
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            format,
                            width: w,
                            height: h,
                            present_mode: wgpu::PresentMode::AutoNoVsync,
                            alpha_mode: wgpu::CompositeAlphaMode::Auto,
                            view_formats: vec![],
                            desired_maximum_frame_latency: 2,
                            color_space: wgpu::SurfaceColorSpace::Auto,
                        },
                    );
                }
                if let (Some(fs), Some(dev)) = (
                    ctx.runtime
                        .engine()
                        .world()
                        .resources()
                        .get::<std::sync::Mutex<ornis_render::gpu_resources::GpuFrameState>>(),
                    ctx.runtime
                        .engine()
                        .world()
                        .resources()
                        .get::<ornis_render::gpu_resources::GpuDevice>(),
                ) {
                    let mut fs = fs.lock().unwrap_or_else(|e| e.into_inner());
                    fs.renderer.resize(&dev.0, w, h);
                    fs.frame3d.set_surface_size(w, h);
                }
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
                    MouseButton::Back => MOUSE_BACK,
                    MouseButton::Forward => MOUSE_FORWARD,
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
                    MouseScrollDelta::PixelDelta(position) => {
                        position.y as f32 / WHEEL_PIXELS_PER_LINE
                    }
                };
                Self::update_input(ctx, |input| input.add_wheel_delta(amount));
            }
            WindowEvent::Focused(false) => {
                Self::update_input(ctx, InputState::clear_all);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(ctx) = &mut self.context {
            Self::process_remote_commands(ctx);
            if let Some(left) = ctx.frames_left.as_mut() {
                if *left == 0 {
                    event_loop.exit();
                    return;
                }
                *left -= 1;
            }
            ctx.window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.context = None;
    }
}
