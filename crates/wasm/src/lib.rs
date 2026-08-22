use std::cell::{Cell, RefCell};
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use wasm_bindgen::prelude::*;
use web_sys::console;

use ornis_core::OpenPBRMaterial;
use ornis_render::scene::{CameraDesc, LightDesc, MaterialDesc, MeshDesc, Scene};
use ornis_render::{
    InstanceData, RenderBackend, RenderBackendConfig, RenderContext, create_render_backend,
};

mod scene_api;

use scene_api::LiveScene;

// ═══════════════════════════════════════════════════════════════════════════
// Ornis WASM — WebGPU entry point for browser editor
// Renders the live scene from /api/scene (polled ~1/s) when the remote
// server provides it; otherwise falls back to assets/scene.ron through the
// shared Renderer3D (RenderBackend). Orbit camera is client-side only.
// Build: wasm-pack build crates/wasm --target web
// ═══════════════════════════════════════════════════════════════════════════

/// Compiled-in fallback for the scene when fetch('scene.ron') is unavailable
/// (e.g. opened without the ornis remote server).
const FALLBACK_SCENE_RON: &str = include_str!("../../../assets/scene.ron");

/// Poll `/api/scene` about once per second (~60 animation frames).
const LIVE_POLL_INTERVAL_FRAMES: u64 = 60;

#[wasm_bindgen(start)]
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
    let window = web_sys::window()?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str("/api/scene"))
        .await
        .ok()?;
    let resp: web_sys::Response = resp_value.dyn_into().ok()?;
    if !resp.ok() {
        return None;
    }
    let text = wasm_bindgen_futures::JsFuture::from(resp.text().ok()?)
        .await
        .ok()?
        .as_string()?;
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

/// GPU-side scene built from a parsed [`Scene`]: shared mesh, materials,
/// per-entity instances and lights in the tuple form RenderBackend expects.
struct GpuScene {
    mesh: ornis_render::Mesh,
    /// (segments, rings) of the shared unit sphere — recreate the mesh only
    /// when these change.
    mesh_params: (u32, u32),
    materials: Vec<OpenPBRMaterial>,
    instances: Vec<InstanceData>,
    lights: Vec<([f32; 3], f32, [f32; 3])>,
}

fn build_gpu_scene(device: &wgpu::Device, scene: &Scene) -> GpuScene {
    // Renderer3D::render_scene draws ONE mesh instanced, so all entities
    // share a single unit sphere (radius 1) and each entity's radius is
    // folded into its instance scale. A sphere of radius r is exactly the
    // unit sphere scaled by r, so this is lossless and supports entities
    // with different radii. Tessellation uses the max segments/rings.
    let mut segments = 32u32;
    let mut rings = 24u32;
    for entity in &scene.entities {
        let MeshDesc::Sphere {
            segments: s,
            rings: r,
            ..
        } = &entity.mesh;
        segments = segments.max(*s);
        rings = rings.max(*r);
    }
    let mesh = ornis_render::create_sphere(device, 1.0, segments, rings);

    let mut materials = Vec::with_capacity(scene.entities.len());
    let mut instances = Vec::with_capacity(scene.entities.len());
    for (i, entity) in scene.entities.iter().enumerate() {
        let material = match &entity.material {
            MaterialDesc::Dielectric {
                base_color,
                roughness,
            } => OpenPBRMaterial::dielectric()
                .base_color_rgb(*base_color)
                .specular_roughness(*roughness),
            MaterialDesc::Metal {
                base_color,
                roughness,
            } => OpenPBRMaterial::metal()
                .base_color_rgb(*base_color)
                .specular_roughness(*roughness),
            MaterialDesc::Coat {
                base_color,
                coat_weight,
                coat_roughness,
            } => OpenPBRMaterial::coat()
                .base_color_rgb(*base_color)
                .coat_weight(*coat_weight)
                .coat_roughness(*coat_roughness),
        };
        materials.push(material);

        let MeshDesc::Sphere { radius, .. } = &entity.mesh;
        let t = &entity.transform;
        let model = Mat4::from_scale_rotation_translation(
            Vec3::from(t.scale) * *radius,
            Quat::from_xyzw(t.rotation[0], t.rotation[1], t.rotation[2], t.rotation[3]).normalize(),
            Vec3::from(t.translation),
        );
        let normal_matrix = model.inverse().transpose();
        instances.push(InstanceData {
            model_matrix: model,
            normal_matrix,
            material_index: i as u32,
        });
    }

    let lights = scene
        .lights
        .iter()
        .map(|l| match l {
            LightDesc::Directional {
                direction,
                intensity,
                color,
            } => (*direction, *intensity, *color),
        })
        .collect();

    GpuScene {
        mesh,
        mesh_params: (segments, rings),
        materials,
        instances,
        lights,
    }
}

/// Attach orbit-camera pointer/wheel listeners to the canvas. The closures
/// are leaked intentionally — they live as long as the page.
fn attach_orbit_controls(canvas: &web_sys::HtmlCanvasElement, orbit: &Rc<RefCell<OrbitCamera>>) {
    // Some((last_x, last_y)) while a drag is in progress.
    let drag = Rc::new(RefCell::new(None::<(f32, f32)>));

    let on_pointerdown: Closure<dyn FnMut(web_sys::PointerEvent)> = {
        let drag = drag.clone();
        let canvas = canvas.clone();
        Closure::new(move |e: web_sys::PointerEvent| {
            *drag.borrow_mut() = Some((e.client_x() as f32, e.client_y() as f32));
            // Capture so the drag continues when the pointer leaves the canvas.
            let _ = canvas.set_pointer_capture(e.pointer_id());
        })
    };
    let on_pointermove: Closure<dyn FnMut(web_sys::PointerEvent)> = {
        let drag = drag.clone();
        let orbit = orbit.clone();
        Closure::new(move |e: web_sys::PointerEvent| {
            let mut drag = drag.borrow_mut();
            if let Some((last_x, last_y)) = *drag {
                let (x, y) = (e.client_x() as f32, e.client_y() as f32);
                orbit.borrow_mut().rotate(x - last_x, y - last_y);
                *drag = Some((x, y));
            }
        })
    };
    let on_pointerup: Closure<dyn FnMut(web_sys::PointerEvent)> = {
        let drag = drag.clone();
        Closure::new(move |_e: web_sys::PointerEvent| {
            *drag.borrow_mut() = None;
        })
    };
    let on_wheel: Closure<dyn FnMut(web_sys::WheelEvent)> = {
        let orbit = orbit.clone();
        Closure::new(move |e: web_sys::WheelEvent| {
            e.prevent_default();
            orbit.borrow_mut().zoom(e.delta_y() as f32);
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

/// Initialize WebGPU on a canvas, load the scene (live /api/scene with
/// scene.ron fallback) and start the render loop.
#[wasm_bindgen]
pub async fn start_renderer(canvas_id: String) -> Result<(), JsValue> {
    console::log_1(&format!("[ornis-wasm] init canvas={}", canvas_id).into());

    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let canvas: web_sys::HtmlCanvasElement = document
        .get_element_by_id(&canvas_id)
        .ok_or("canvas not found")?
        .dyn_into()?;

    // Resize canvas to match viewport
    let resize = |c: &web_sys::HtmlCanvasElement| {
        let parent = c.parent_element().unwrap();
        c.set_width(parent.client_width() as u32);
        c.set_height(parent.client_height() as u32);
    };
    resize(&canvas);

    // ── wgpu WebGPU setup ─────────────────────────────────────────────
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        flags: wgpu::InstanceFlags::empty(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });

    let surface = unsafe {
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
        .map_err(|_| "adapter not found")?;

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

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
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
            surface_format, config.width, config.height
        )
        .into(),
    );

    // ── Scene: live /api/scene first, scene.ron as fallback ───────────
    let live = fetch_live_scene().await;
    let live_mode = live.is_some();
    let (scene, initial_version) = match live {
        Some(live) => {
            console::log_1(
                &format!(
                    "[ornis-wasm] live scene from /api/scene: version={}, {} entities, {} lights",
                    live.version,
                    live.scene.entities.len(),
                    live.scene.lights.len()
                )
                .into(),
            );
            (live.scene, live.version)
        }
        None => {
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
            (scene, 0)
        }
    };

    let gpu_scene = build_gpu_scene(&device, &scene);

    // ── Renderer3D via the RenderBackend trait ────────────────────────
    let backend_config = RenderBackendConfig {
        surface_config: config.clone(),
        sample_count: 1,
        max_objects: 256,
        max_materials: 64,
    };
    let mut renderer: Box<dyn RenderBackend> = create_render_backend(&device, &backend_config);

    renderer.upload_materials(&queue, &gpu_scene.materials);
    renderer.upload_instances(&queue, &gpu_scene.instances);
    renderer.set_lights(&queue, scene.ambient, &gpu_scene.lights);

    let mut instance_count = gpu_scene.instances.len() as u32;
    let mut mesh = gpu_scene.mesh;
    let mut mesh_params = gpu_scene.mesh_params;

    // Client-side orbit camera, initialized from the scene camera.
    let orbit = Rc::new(RefCell::new(OrbitCamera::from_desc(&scene.camera)));
    attach_orbit_controls(&canvas, &orbit);

    // Live-update plumbing (single-threaded): the render loop spawns a
    // fetch every LIVE_POLL_INTERVAL_FRAMES; the fetch task deposits the
    // parsed scene into `pending_scene` and the loop applies it on the next
    // frame, so the renderer/device are only ever touched by the loop.
    let pending_scene: Rc<RefCell<Option<LiveScene>>> = Rc::new(RefCell::new(None));
    let fetch_in_flight = Rc::new(Cell::new(false));
    // Version of the scene already uploaded to the renderer. 0 means "no
    // live version applied" (static scene.ron mode never polls anyway).
    let applied_version = Rc::new(Cell::new(initial_version));

    // ── Render loop ───────────────────────────────────────────────────
    let f = Rc::new(RefCell::new(None as Option<Closure<dyn FnMut()>>));
    let f_clone = f.clone();

    let window_for_loop = window.clone();
    let canvas_for_loop = canvas.clone();

    let mut frame_count: u64 = 0;

    let f_inner = f.clone();
    *f_clone.borrow_mut() = Some(Closure::new(move || {
        // Handle resize
        let pw = canvas_for_loop
            .parent_element()
            .map(|p| p.client_width() as u32)
            .unwrap_or(canvas_for_loop.width())
            .max(1);
        let ph = canvas_for_loop
            .parent_element()
            .map(|p| p.client_height() as u32)
            .unwrap_or(canvas_for_loop.height())
            .max(1);
        if canvas_for_loop.width() != pw || canvas_for_loop.height() != ph {
            canvas_for_loop.set_width(pw);
            canvas_for_loop.set_height(ph);
            config.width = pw;
            config.height = ph;
            surface.configure(&device, &config);
            renderer.resize(&device, pw, ph);
            console::log_1(&format!("[ornis-wasm] resized surface to {}x{}", pw, ph).into());
        }

        // ── Live scene polling (~1/s) ────────────────────────────────
        if live_mode
            && frame_count.is_multiple_of(LIVE_POLL_INTERVAL_FRAMES)
            && !fetch_in_flight.get()
        {
            fetch_in_flight.set(true);
            let pending_scene = pending_scene.clone();
            let fetch_in_flight = fetch_in_flight.clone();
            let applied_version = applied_version.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(live) = fetch_live_scene().await
                    && live.version != applied_version.get()
                {
                    console::log_1(
                        &format!(
                            "[ornis-wasm] /api/scene changed: v{} -> v{}",
                            applied_version.get(),
                            live.version
                        )
                        .into(),
                    );
                    *pending_scene.borrow_mut() = Some(live);
                }
                fetch_in_flight.set(false);
            });
        }

        // Apply a freshly polled scene: rebuild materials/instances (and the
        // shared mesh if its tessellation changed) and re-upload. No
        // device/surface recreation.
        if let Some(live) = pending_scene.borrow_mut().take() {
            let gpu = build_gpu_scene(&device, &live.scene);
            if gpu.mesh_params != mesh_params {
                mesh = gpu.mesh;
                mesh_params = gpu.mesh_params;
            }
            renderer.upload_materials(&queue, &gpu.materials);
            renderer.upload_instances(&queue, &gpu.instances);
            renderer.set_lights(&queue, live.scene.ambient, &gpu.lights);
            instance_count = gpu.instances.len() as u32;
            applied_version.set(live.version);
            console::log_1(
                &format!(
                    "[ornis-wasm] live scene v{} applied ({} instances)",
                    live.version, instance_count
                )
                .into(),
            );
        }

        // Camera for the current aspect ratio, from the orbit state.
        let aspect = config.width as f32 / config.height as f32;
        let (cam_pos, cam_target, cam_up, fov, near, far) = {
            let orbit = orbit.borrow();
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
        renderer.set_camera(&queue, &view_proj.to_cols_array_2d(), cam_pos.to_array());

        match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render_encoder"),
                });

                renderer.render_scene(
                    RenderContext {
                        device: &device,
                        queue: &queue,
                        encoder: &mut encoder,
                        target: &view,
                    },
                    &mesh,
                    instance_count,
                );

                queue.submit(std::iter::once(encoder.finish()));
                frame.present();

                frame_count += 1;
                if frame_count == 1 {
                    console::log_1(
                        &format!(
                            "[ornis-wasm] first frame rendered ({}x{}, {} instances)",
                            config.width, config.height, instance_count
                        )
                        .into(),
                    );
                } else if frame_count.is_multiple_of(600) {
                    console::log_1(&format!("[ornis-wasm] frame {} rendered", frame_count).into());
                }
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                // Surface lost its configuration — reconfigure and retry next frame
                surface.configure(&device, &config);
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                // Skip frame, will retry next animation frame
            }
        }

        // Schedule next frame
        window_for_loop
            .request_animation_frame(f_inner.borrow().as_ref().unwrap().as_ref().unchecked_ref())
            .unwrap();
    }));

    window.request_animation_frame(f_clone.borrow().as_ref().unwrap().as_ref().unchecked_ref())?;

    // Prevent dropping the closure so the JS callback stays valid
    std::mem::forget(f);
    std::mem::forget(f_clone);

    Ok(())
}
