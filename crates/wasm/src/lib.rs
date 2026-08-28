//! Ornis WASM — WebGPU entry point for the browser editor.

#![warn(missing_docs)]
//!
//! Renders the live scene from `/api/scene` (polled ~1/s) when the remote
//! server provides it; otherwise falls back to `assets/scene.ron` through
//! the shared ECS [`RenderWorld`], [`RenderExtracted`], and
//! [`RenderFrame3D`] frame contract. The orbit
//! camera is client-side only.
//!
//! Build: `wasm-pack build crates/wasm --target web`.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use glam::{Mat4, Vec3};
use ornis_core::InputState;
use wasm_bindgen::prelude::*;
use web_sys::console;

use ornis_render::scene::{CameraDesc, LightDesc, Scene};
use ornis_render::{
    RenderContext, RenderExtracted, RenderFrame3D, RenderWorld, Renderer3D, Technique,
};

mod scene_api;

use scene_api::LiveScene;

/// Shared handle to the requestAnimationFrame closure (self-rescheduling
/// render loop needs to reference itself).
type FrameCallback = Rc<RefCell<Option<Closure<dyn FnMut()>>>>;

/// Compiled-in fallback for the scene when fetch('scene.ron') is unavailable
/// (e.g. opened without the ornis remote server).
const FALLBACK_SCENE_RON: &str = include_str!("../../../assets/scene.ron");

/// Poll `/api/scene` about once per second (~60 animation frames).
const LIVE_POLL_INTERVAL_FRAMES: u64 = 60;

#[wasm_bindgen(start)]
/// wasm-bindgen entry point: installs the panic hook and logs module load.
pub fn start() {
    console_error_panic_hook::set_once();
    console::log_1(&"[ornis-wasm] module loaded".into());
}

/// Wrapper to give HtmlCanvasElement a raw window handle for wgpu.
struct CanvasWindow(web_sys::HtmlCanvasElement);

impl raw_window_handle::HasWindowHandle for CanvasWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        use core::ptr::NonNull;
        use raw_window_handle::{RawWindowHandle, WebCanvasWindowHandle, WindowHandle};

        let js_value: &wasm_bindgen::JsValue = &self.0;
        let obj = NonNull::from(js_value).cast();
        let web_handle = WebCanvasWindowHandle::new(obj);
        let raw = RawWindowHandle::WebCanvas(web_handle);
        Ok(unsafe { WindowHandle::borrow_raw(raw) })
    }
}

/// Fetch scene.ron from the server; fall back to the compiled-in copy.
async fn load_scene_ron() -> String {
    if let Some(window) = web_sys::window() {
        let resp_value =
            wasm_bindgen_futures::JsFuture::from(window.fetch_with_str("scene.ron")).await;
        if let Ok(resp_value) = resp_value {
            let resp: web_sys::Response = match resp_value.dyn_into() {
                Ok(r) => r,
                Err(_) => return FALLBACK_SCENE_RON.to_string(),
            };
            if resp.ok()
                && let Ok(text_promise) = resp.text()
                && let Ok(text) = wasm_bindgen_futures::JsFuture::from(text_promise).await
                && let Some(s) = text.as_string()
            {
                console::log_1(&"[ornis-wasm] scene.ron fetched from server".into());
                return s;
            }
        }
    }
    console::warn_1(&"[ornis-wasm] fetch(scene.ron) failed, using embedded scene".into());
    FALLBACK_SCENE_RON.to_string()
}

/// Fetch and parse `/api/scene`. Returns `None` on any failure — network
/// error, non-OK status, malformed JSON or the reduced server variant
/// without per-entity transform/mesh/material — so the caller can fall back
/// to the static scene.ron path.
async fn fetch_live_scene() -> Option<LiveScene> {
    let text = fetch_api_text("/api/scene").await?;
    match scene_api::parse_scene_json(&text) {
        Ok(live) => Some(live),
        Err(e) => {
            console::warn_1(&format!("[ornis-wasm] /api/scene parse failed: {e}").into());
            None
        }
    }
}

/// Client-side orbit camera: azimuth/elevation around a target plus a zoom
/// radius. Initialized from the scene camera; never sent to the server.
struct OrbitCamera {
    target: Vec3,
    up: Vec3,
    azimuth: f32,
    elevation: f32,
    radius: f32,
    fov: f32,
    near: f32,
    far: f32,
}

impl OrbitCamera {
    const MIN_RADIUS: f32 = 0.5;
    const MAX_RADIUS: f32 = 1000.0;
    /// Keep elevation off the poles so `look_at` never degenerates.
    const ELEVATION_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    const ROTATE_SPEED: f32 = 0.005;
    const ZOOM_SPEED: f32 = 0.001;

    fn from_desc(cam: &CameraDesc) -> Self {
        let target = Vec3::from(cam.target);
        let offset = Vec3::from(cam.position) - target;
        let radius = offset.length().max(Self::MIN_RADIUS);
        // offset = radius * (cos(el)*cos(az), sin(el), cos(el)*sin(az))
        let elevation = (offset.y / radius).clamp(-1.0, 1.0).asin();
        let azimuth = offset.z.atan2(offset.x);
        Self {
            target,
            up: Vec3::from(cam.up),
            azimuth,
            elevation,
            radius,
            fov: cam.fov,
            near: cam.near,
            far: cam.far,
        }
    }

    fn position(&self) -> Vec3 {
        let (ce, se) = (self.elevation.cos(), self.elevation.sin());
        let (ca, sa) = (self.azimuth.cos(), self.azimuth.sin());
        self.target + self.radius * Vec3::new(ce * ca, se, ce * sa)
    }

    fn rotate(&mut self, dx: f32, dy: f32) {
        self.azimuth -= dx * Self::ROTATE_SPEED;
        self.elevation = (self.elevation + dy * Self::ROTATE_SPEED)
            .clamp(-Self::ELEVATION_LIMIT, Self::ELEVATION_LIMIT);
    }

    /// `delta_y` from a wheel event: positive scrolls down/away (zoom out).
    fn zoom(&mut self, delta_y: f32) {
        self.radius = (self.radius * (delta_y * Self::ZOOM_SPEED).exp())
            .clamp(Self::MIN_RADIUS, Self::MAX_RADIUS);
    }
}

/// GPU-side scene built from the ECS extraction snapshot. The scene
/// description remains the serialization boundary; renderable components
/// are inserted into [`RenderWorld`] and extracted by its scheduled
/// `Engine` frame before this GPU adapter runs.
struct GpuScene {
    mesh: ornis_render::Mesh,
    /// (segments, rings) of the shared unit sphere — recreate the mesh only
    /// when these change.
    mesh_params: (u32, u32),
    extracted: RenderExtracted,
    lights: Vec<([f32; 3], f32, [f32; 3])>,
    ambient: [f32; 3],
}

fn build_gpu_scene(device: &wgpu::Device, render_world: &RenderWorld, scene: &Scene) -> GpuScene {
    let extracted = render_world.extracted();
    // RenderFrame3D draws one shared mesh instanced. Each sphere's radius is
    // already folded into its extracted model scale, so the maximum
    // tessellation is sufficient for every entity.
    let mesh_params = extracted.mesh_params;
    let mesh = ornis_render::create_sphere(device, 1.0, mesh_params.0, mesh_params.1);
    let lights = scene
        .lights
        .iter()
        .map(|light| match light {
            LightDesc::Directional {
                direction,
                intensity,
                color,
            } => (*direction, *intensity, *color),
        })
        .collect();

    GpuScene {
        mesh,
        mesh_params,
        extracted,
        lights,
        ambient: scene.ambient,
    }
}

/// Attach orbit-camera pointer/wheel listeners to the canvas. The closures
/// are leaked intentionally — they live as long as the page.
fn attach_orbit_controls(canvas: &web_sys::HtmlCanvasElement, input: &Rc<RefCell<InputState>>) {
    let on_pointerdown: Closure<dyn FnMut(web_sys::PointerEvent)> = {
        let input = input.clone();
        let canvas = canvas.clone();
        Closure::new(move |e: web_sys::PointerEvent| {
            let mut input = input.borrow_mut();
            input.set_mouse_button(0, true);
            input.set_pointer_position([e.client_x() as f32, e.client_y() as f32]);
            // Capture so the drag continues when the pointer leaves the canvas.
            let _ = canvas.set_pointer_capture(e.pointer_id());
        })
    };
    let on_pointermove: Closure<dyn FnMut(web_sys::PointerEvent)> = {
        let input = input.clone();
        Closure::new(move |e: web_sys::PointerEvent| {
            input
                .borrow_mut()
                .set_pointer_position([e.client_x() as f32, e.client_y() as f32]);
        })
    };
    let on_pointerup: Closure<dyn FnMut(web_sys::PointerEvent)> = {
        let input = input.clone();
        Closure::new(move |_e: web_sys::PointerEvent| {
            input.borrow_mut().set_mouse_button(0, false);
        })
    };
    let on_wheel: Closure<dyn FnMut(web_sys::WheelEvent)> = {
        let input = input.clone();
        Closure::new(move |e: web_sys::WheelEvent| {
            e.prevent_default();
            input.borrow_mut().add_wheel_delta(e.delta_y() as f32);
        })
    };

    for (kind, cb) in [
        ("pointerdown", &on_pointerdown),
        ("pointermove", &on_pointermove),
        ("pointerup", &on_pointerup),
        ("pointercancel", &on_pointerup),
    ] {
        if let Err(e) = canvas.add_event_listener_with_callback(kind, cb.as_ref().unchecked_ref()) {
            console::warn_1(&format!("[ornis-wasm] failed to attach {kind}: {e:?}").into());
        }
    }
    if let Err(e) =
        canvas.add_event_listener_with_callback("wheel", on_wheel.as_ref().unchecked_ref())
    {
        console::warn_1(&format!("[ornis-wasm] failed to attach wheel: {e:?}").into());
    }

    on_pointerdown.forget();
    on_pointermove.forget();
    on_pointerup.forget();
    on_wheel.forget();
}

/// Look up the canvas element by id and cast it to `HtmlCanvasElement`.
fn get_canvas(canvas_id: &str) -> Result<web_sys::HtmlCanvasElement, JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| JsValue::from_str("canvas not found"))?
        .dyn_into()
        .map_err(Into::into)
}

/// Resize the canvas to fill its parent element.
fn resize_canvas_to_parent(canvas: &web_sys::HtmlCanvasElement) {
    if let Some(parent) = canvas.parent_element() {
        canvas.set_width(parent.client_width() as u32);
        canvas.set_height(parent.client_height() as u32);
    }
}

fn make_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        flags: wgpu::InstanceFlags::empty(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    })
}

/// Everything the render loop needs from WebGPU initialization.
struct GpuContext {
    // SAFETY invariant: the phantom window lifetime is pinned to `'static`;
    // the canvas handle is owned (`CanvasWindow(canvas.clone())`) and lives
    // as long as the page.
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
}

async fn init_webgpu(
    instance: &wgpu::Instance,
    canvas: &web_sys::HtmlCanvasElement,
) -> Result<GpuContext, JsValue> {
    // SAFETY: the canvas outlives the surface — both are kept alive for the
    // lifetime of the page (the loop closure is mem::forget'ed).
    let surface: wgpu::Surface<'static> = unsafe {
        instance
            .create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(&CanvasWindow(canvas.clone()))
                    .map_err(|e| format!("surface target: {:?}", e))?,
            )
            .map_err(|e| format!("surface: {:?}", e))?
    };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(|_| JsValue::from_str("adapter not found"))?;

    // WebGPU spec defaults: max_storage_buffers_per_shader_stage = 8, which is
    // what Renderer3D's bind groups need (camera + per-object + materials +
    // lighting). Downlevel (WebGL2) defaults would zero out storage buffers.
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("ornis-wasm"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("device: {:?}", e))?;

    // Route uncaptured WebGPU validation errors into console.error explicitly.
    // (The browser also prints them, but this makes them greppable and keeps
    // them visible if the page's console filtering hides GPU messages.)
    device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
        console::error_1(&format!("[ornis-wasm] wgpu error: {:?}", e).into());
    }));

    console::log_1(
        &format!(
            "[ornis-wasm] adapter='{}' backend={:?}, limits: storage_buffers/stage={}",
            adapter.get_info().name,
            adapter.get_info().backend,
            device.limits().max_storage_buffers_per_shader_stage
        )
        .into(),
    );

    let caps = surface.get_capabilities(&adapter);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: pick_surface_format(&caps),
        width: canvas.width().max(1),
        height: canvas.height().max(1),
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    console::log_1(
        &format!(
            "[ornis-wasm] WebGPU ready, format={:?}, surface={}x{}",
            config.format, config.width, config.height
        )
        .into(),
    );

    Ok(GpuContext {
        surface,
        device,
        queue,
        config,
    })
}

/// Pick an sRGB swap-chain format when available, else the first supported.
fn pick_surface_format(caps: &wgpu::SurfaceCapabilities) -> wgpu::TextureFormat {
    caps.formats
        .iter()
        .copied()
        .find(|f| {
            matches!(
                f,
                wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb
            )
        })
        .unwrap_or_else(|| {
            caps.formats
                .first()
                .copied()
                .unwrap_or(wgpu::TextureFormat::Rgba8UnormSrgb)
        })
}

/// Load the initial scene: live `/api/scene` first, `scene.ron` as fallback.
/// Returns the scene, its version, and whether live polling should run.
async fn load_initial_scene() -> Result<(Scene, u64, bool), JsValue> {
    if let Some(live) = fetch_live_scene().await {
        console::log_1(
            &format!(
                "[ornis-wasm] live scene from /api/scene: version={}, {} entities, {} lights",
                live.version,
                live.scene.entities.len(),
                live.scene.lights.len()
            )
            .into(),
        );
        let LiveScene { scene, version } = live;
        return Ok((scene, version, true));
    }

    console::log_1(
        &"[ornis-wasm] /api/scene unavailable or reduced, falling back to scene.ron".into(),
    );
    let ron_text = load_scene_ron().await;
    let scene = Scene::from_ron(&ron_text).map_err(|e| format!("scene parse: {:?}", e))?;
    console::log_1(
        &format!(
            "[ornis-wasm] scene '{}' loaded: {} entities, {} lights",
            scene.name,
            scene.entities.len(),
            scene.lights.len()
        )
        .into(),
    );
    Ok((scene, 0, false))
}

/// Mutable state carried across animation frames by the render loop.
struct FrameState<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer3D,
    frame_plan: RenderFrame3D,
    render_world: RenderWorld,
    mesh: ornis_render::Mesh,
    mesh_params: (u32, u32),
    instance_count: u32,
    input: Rc<RefCell<InputState>>,
    orbit: Rc<RefCell<OrbitCamera>>,
}

impl<'a> FrameState<'a> {
    /// Match the canvas (and surface) size to its parent element.
    fn handle_resize(&mut self, canvas: &web_sys::HtmlCanvasElement) {
        let pw = canvas
            .parent_element()
            .map(|p| p.client_width() as u32)
            .unwrap_or(canvas.width())
            .max(1);
        let ph = canvas
            .parent_element()
            .map(|p| p.client_height() as u32)
            .unwrap_or(canvas.height())
            .max(1);
        if canvas.width() != pw || canvas.height() != ph {
            canvas.set_width(pw);
            canvas.set_height(ph);
            self.config.width = pw;
            self.config.height = ph;
            self.surface.configure(&self.device, &self.config);
            self.renderer.resize(&self.device, pw, ph);
            self.frame_plan.set_surface_size(pw, ph);
            console::log_1(&format!("[ornis-wasm] resized surface to {}x{}", pw, ph).into());
        }
    }

    /// Rebuild the client ECS scene through the shared `Engine` schedule,
    /// extract its components and re-upload the resulting GPU snapshot. No
    /// device/surface recreation is needed for a live scene update.
    fn apply_live_scene(&mut self, live: &LiveScene, applied_version: &Cell<u64>) {
        self.render_world.replace_scene(&live.scene);
        self.render_world.run_frame(0.0);
        let gpu = build_gpu_scene(&self.device, &self.render_world, &live.scene);
        if gpu.mesh_params != self.mesh_params {
            self.mesh = gpu.mesh;
            self.mesh_params = gpu.mesh_params;
        }
        self.renderer
            .upload_materials(&self.queue, &gpu.extracted.materials);
        self.renderer
            .upload_instances(&self.queue, &gpu.extracted.instances);
        self.renderer
            .set_lights(&self.queue, live.scene.ambient, &gpu.lights);
        self.instance_count = gpu.extracted.instances.len() as u32;
        applied_version.set(live.version);
        console::log_1(
            &format!(
                "[ornis-wasm] live scene v{} applied through Engine/FramePlan ({} instances)",
                live.version, self.instance_count
            )
            .into(),
        );
    }

    /// Move platform input into the browser-side Engine resource and consume
    /// its camera-facing controls. The shared InputState is copied into the
    /// logical world before `Engine::run_frame`, so custom browser systems
    /// can read the same snapshot without taking ownership of DOM events.
    fn sync_input(&mut self) {
        let snapshot = self.input.borrow().clone();
        {
            let mut orbit = self.orbit.borrow_mut();
            if snapshot.mouse_button_down(0) {
                let [dx, dy] = snapshot.pointer_delta();
                orbit.rotate(dx, dy);
            }
            orbit.zoom(snapshot.wheel_delta());
        }
        if let Some(input) = self
            .render_world
            .engine_mut()
            .world_mut()
            .resources_mut()
            .get_mut::<InputState>()
        {
            *input = snapshot;
        }
        self.input.borrow_mut().clear_frame_transients();
    }

    /// Upload the orbit-derived camera for the current aspect ratio.
    fn update_camera(&mut self) {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let (cam_pos, cam_target, cam_up, fov, near, far) = {
            let orbit = self.orbit.borrow();
            (
                orbit.position(),
                orbit.target,
                orbit.up,
                orbit.fov,
                orbit.near,
                orbit.far,
            )
        };
        let view = Mat4::look_at_rh(cam_pos, cam_target, cam_up);
        let proj = Mat4::perspective_rh(fov.to_radians(), aspect, near, far);
        let view_proj = proj * view;
        self.renderer.set_camera(
            &self.queue,
            &view_proj.to_cols_array_2d(),
            cam_pos.to_array(),
        );
    }

    /// Draw one frame into the given swap-chain view through the shared
    /// [`RenderFrame3D`] plan, not the legacy `render_scene` shortcut.
    fn draw(&mut self, target_view: &wgpu::TextureView) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });
        self.frame_plan.render(
            RenderContext {
                device: &self.device,
                queue: &self.queue,
                encoder: &mut encoder,
                target: target_view,
            },
            &self.renderer,
            &self.mesh,
            self.instance_count,
        );
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Reconfigure a surface whose configuration was lost or outdated.
    fn reconfigure(&mut self) {
        self.surface.configure(&self.device, &self.config);
    }
}

/// Log milestone frame counts: the very first frame and periodic heartbeats.
fn log_frame_milestone(frame_count: u64, config: &wgpu::SurfaceConfiguration, instances: u32) {
    if frame_count == 1 {
        console::log_1(
            &format!(
                "[ornis-wasm] first frame rendered ({}x{}, {} instances)",
                config.width, config.height, instances
            )
            .into(),
        );
    } else if frame_count.is_multiple_of(600) {
        console::log_1(&format!("[ornis-wasm] frame {frame_count} rendered").into());
    }
}

/// Whether a candidate snapshot is newer than both the applied and queued
/// versions. The server's version is monotonic; rejecting older responses
/// prevents an out-of-order fetch from rolling the viewport back.
fn accept_live_scene_version(
    applied_version: u64,
    pending_version: Option<u64>,
    candidate_version: u64,
) -> bool {
    candidate_version > applied_version
        && pending_version.is_none_or(|pending| candidate_version > pending)
}

/// Kick off a `/api/scene` poll unless one is already in flight. Deposits a
/// newer parsed scene into `pending_scene`; cleared on completion.
fn poll_live_scene(
    pending_scene: &Rc<RefCell<Option<LiveScene>>>,
    fetch_in_flight: &Rc<Cell<bool>>,
    applied_version: &Rc<Cell<u64>>,
) {
    fetch_in_flight.set(true);
    let pending_scene = pending_scene.clone();
    let fetch_in_flight = fetch_in_flight.clone();
    let applied_version = applied_version.clone();
    wasm_bindgen_futures::spawn_local(async move {
        if let Some(live) = fetch_live_scene().await {
            let applied = applied_version.get();
            let pending = pending_scene.borrow().as_ref().map(|scene| scene.version);
            if accept_live_scene_version(applied, pending, live.version) {
                console::log_1(
                    &format!(
                        "[ornis-wasm] /api/scene changed: v{} -> v{}",
                        applied, live.version
                    )
                    .into(),
                );
                *pending_scene.borrow_mut() = Some(live);
            }
        }
        fetch_in_flight.set(false);
    });
}

/// Non-GPU handles shared with the render-loop closure.
struct LoopHandles {
    window: web_sys::Window,
    canvas: web_sys::HtmlCanvasElement,
    pending_scene: Rc<RefCell<Option<LiveScene>>>,
    fetch_in_flight: Rc<Cell<bool>>,
    live_mode: bool,
    input: Rc<RefCell<InputState>>,
}

/// Browser entry point: initializes WebGPU on the canvas with `canvas_id`,
/// loads the initial scene (`/api/scene` or the compiled-in fallback), and
/// spawns the render loop with orbit controls and live scene polling.
///
/// # Errors
///
/// Returns a `JsValue` error if the canvas is missing, WebGPU initialization
/// or scene loading fails, or no `window` object is available.
#[wasm_bindgen]
pub async fn start_renderer(canvas_id: String) -> Result<(), JsValue> {
    console::log_1(&format!("[ornis-wasm] init canvas={}", canvas_id).into());

    let canvas = get_canvas(&canvas_id)?;
    resize_canvas_to_parent(&canvas);

    let instance = make_instance();
    let ctx = init_webgpu(&instance, &canvas).await?;

    let (scene, initial_version, live_mode) = load_initial_scene().await?;
    // Keep the browser-side ECS explicitly separate from the server's
    // authoritative world: the serialized scene crosses the boundary once,
    // then the same Engine/RenderExtract contract as native is used locally.
    let mut render_world = RenderWorld::from_scene(&scene);
    render_world.run_frame(0.0);
    let gpu_scene = build_gpu_scene(&ctx.device, &render_world, &scene);

    let renderer = Renderer3D::new(&ctx.device, &ctx.config, 1);
    let frame_plan = RenderFrame3D::new_with(
        ctx.config.format,
        (ctx.config.width, ctx.config.height),
        Technique::Hybrid,
        false,
    );

    // Client-side orbit camera, initialized from the scene camera. DOM events
    // first enter the backend-neutral InputState; the frame loop consumes
    // them, so the camera and future browser systems share one input path.
    let orbit = Rc::new(RefCell::new(OrbitCamera::from_desc(&scene.camera)));
    let input = Rc::new(RefCell::new(InputState::new()));
    attach_orbit_controls(&canvas, &input);

    // Live-update plumbing (single-threaded): the render loop spawns a
    // fetch every LIVE_POLL_INTERVAL_FRAMES; the fetch task deposits the
    // parsed scene into `pending_scene` and the loop applies it on the next
    // frame, so the renderer/device are only ever touched by the loop.
    let handles = LoopHandles {
        window: web_sys::window().ok_or("no window")?,
        canvas,
        pending_scene: Rc::new(RefCell::new(None)),
        fetch_in_flight: Rc::new(Cell::new(false)),
        live_mode,
        input,
    };

    spawn_render_loop(
        handles,
        ctx,
        renderer,
        frame_plan,
        render_world,
        gpu_scene,
        orbit,
        initial_version,
    )?;
    Ok(())
}

/// Build and leak the requestAnimationFrame closure driving the render loop.
/// The closures are leaked intentionally so the JS callback stays valid.
#[allow(clippy::too_many_arguments)]
fn spawn_render_loop(
    handles: LoopHandles,
    ctx: GpuContext,
    renderer: Renderer3D,
    frame_plan: RenderFrame3D,
    render_world: RenderWorld,
    gpu_scene: GpuScene,
    orbit: Rc<RefCell<OrbitCamera>>,
    initial_version: u64,
) -> Result<(), JsValue> {
    renderer.upload_materials(&ctx.queue, &gpu_scene.extracted.materials);
    renderer.upload_instances(&ctx.queue, &gpu_scene.extracted.instances);
    renderer.set_lights(&ctx.queue, gpu_scene.ambient, &gpu_scene.lights);

    let applied_version = Rc::new(Cell::new(initial_version));

    let LoopHandles {
        window,
        canvas,
        pending_scene,
        fetch_in_flight,
        live_mode,
        input,
    } = handles;

    let mut frame = FrameState {
        surface: ctx.surface,
        device: ctx.device,
        queue: ctx.queue,
        config: ctx.config,
        renderer,
        frame_plan,
        render_world,
        mesh: gpu_scene.mesh,
        mesh_params: gpu_scene.mesh_params,
        instance_count: gpu_scene.extracted.instances.len() as u32,
        input,
        orbit,
    };

    let f: FrameCallback = Rc::new(RefCell::new(None));
    let f_clone = f.clone();

    let window_for_loop = window.clone();
    let mut frame_count: u64 = 0;
    let f_inner = f.clone();

    *f_clone.borrow_mut() = Some(Closure::new(move || {
        frame.handle_resize(&canvas);
        frame.sync_input();
        frame.render_world.run_frame(1.0 / 60.0);

        // ── Live scene polling (~1/s) ────────────────────────────────
        if live_mode
            && frame_count.is_multiple_of(LIVE_POLL_INTERVAL_FRAMES)
            && !fetch_in_flight.get()
        {
            poll_live_scene(&pending_scene, &fetch_in_flight, &applied_version);
        }

        // Apply a freshly polled scene, if any.
        if let Some(live) = pending_scene.borrow_mut().take() {
            frame.apply_live_scene(&live, &applied_version);
        }

        frame.update_camera();

        match frame.surface.get_current_texture() {
            // Surface lost its configuration — reconfigure and retry next frame.
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                frame.reconfigure();
            }
            // Skip frame, will retry next animation frame.
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {}
            wgpu::CurrentSurfaceTexture::Success(presentable)
            | wgpu::CurrentSurfaceTexture::Suboptimal(presentable) => {
                let view = presentable
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                frame.draw(&view);
                presentable.present();

                frame_count += 1;
                log_frame_milestone(frame_count, &frame.config, frame.instance_count);
            }
        }

        // Schedule next frame
        window_for_loop
            .request_animation_frame(f_inner.borrow().as_ref().unwrap().as_ref().unchecked_ref())
            .unwrap();
    }));

    window.request_animation_frame(f_clone.borrow().as_ref().unwrap().as_ref().unchecked_ref())?;

    std::mem::forget(f);
    std::mem::forget(f_clone);

    Ok(())
}

/// Fetch a URL and return its response body as a string, or `None` on any
/// network / status / decoding failure.
async fn fetch_api_text(url: &str) -> Option<String> {
    let window = web_sys::window()?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url))
        .await
        .ok()?;
    let resp: web_sys::Response = resp_value.dyn_into().ok()?;
    if !resp.ok() {
        return None;
    }
    let text = wasm_bindgen_futures::JsFuture::from(resp.text().ok()?)
        .await
        .ok()?;
    text.as_string()
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn live_snapshot_crosses_serialization_boundary_into_shared_render_world() {
        let live = scene_api::parse_scene_json(scene_api::FULL_CONTRACT)
            .expect("the shared API contract must parse");
        let mut render_world = RenderWorld::from_scene(&live.scene);

        assert_eq!(render_world.entity_count(), live.scene.entities.len());
        assert_eq!(render_world.engine().schedule().len(), 1);

        render_world.run_frame(0.0);
        let extracted = render_world.extracted();

        assert_eq!(extracted.mesh_params, (32, 24));
        assert_eq!(extracted.materials.len(), live.scene.entities.len());
        assert_eq!(extracted.instances.len(), live.scene.entities.len());
        assert_eq!(extracted.instances[0].material_index, 0);
        assert_eq!(
            extracted.instances[0].model_matrix.w_axis.truncate(),
            Vec3::new(-5.6, 0.0, 0.0)
        );
    }

    #[test]
    fn stale_live_snapshot_versions_are_rejected() {
        assert!(!accept_live_scene_version(5, None, 4));
        assert!(!accept_live_scene_version(5, Some(7), 6));
        assert!(!accept_live_scene_version(5, Some(7), 7));
        assert!(accept_live_scene_version(5, Some(7), 8));
        assert!(accept_live_scene_version(0, None, 1));
    }
}
