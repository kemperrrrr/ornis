use glam::{Mat4, Quat, Vec3};
use wasm_bindgen::prelude::*;
use web_sys::console;

use ornis_core::OpenPBRMaterial;
use ornis_render::scene::{LightDesc, MaterialDesc, MeshDesc, Scene};
use ornis_render::{
    InstanceData, RenderBackend, RenderBackendConfig, RenderContext, create_render_backend,
};

// ═══════════════════════════════════════════════════════════════════════════
// Ornis WASM — WebGPU entry point for browser editor
// Renders assets/scene.ron through the shared Renderer3D (RenderBackend).
// Build: wasm-pack build crates/wasm --target web
// ═══════════════════════════════════════════════════════════════════════════

/// Compiled-in fallback for the scene when fetch('scene.ron') is unavailable
/// (e.g. opened without the ornis remote server).
const FALLBACK_SCENE_RON: &str = include_str!("../../../assets/scene.ron");

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
            if resp.ok() {
                if let Ok(text_promise) = resp.text() {
                    if let Ok(text) = wasm_bindgen_futures::JsFuture::from(text_promise).await {
                        if let Some(s) = text.as_string() {
                            console::log_1(&"[ornis-wasm] scene.ron fetched from server".into());
                            return s;
                        }
                    }
                }
            }
        }
    }
    console::warn_1(&"[ornis-wasm] fetch(scene.ron) failed, using embedded scene".into());
    FALLBACK_SCENE_RON.to_string()
}

/// GPU-side scene built from a parsed [`Scene`]: shared mesh, materials,
/// per-entity instances and lights in the tuple form RenderBackend expects.
struct GpuScene {
    mesh: ornis_render::Mesh,
    materials: Vec<OpenPBRMaterial>,
    instances: Vec<InstanceData>,
    lights: Vec<([f32; 3], f32, [f32; 3])>,
}

fn build_gpu_scene(device: &wgpu::Device, scene: &Scene) -> Result<GpuScene, JsValue> {
    // Current scene format only has Sphere meshes and Renderer3D::render_scene
    // draws a single mesh instanced — all entities share one mesh. Use the
    // first entity's mesh parameters.
    let first = scene.entities.first().ok_or("scene has no entities")?;
    let mesh = match &first.mesh {
        MeshDesc::Sphere {
            radius,
            segments,
            rings,
        } => ornis_render::create_sphere(device, *radius, *segments, *rings),
    };

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

        let t = &entity.transform;
        let model = Mat4::from_scale_rotation_translation(
            Vec3::from(t.scale),
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

    Ok(GpuScene {
        mesh,
        materials,
        instances,
        lights,
    })
}

/// Initialize WebGPU on a canvas, load scene.ron and start the render loop.
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

    // ── Scene ─────────────────────────────────────────────────────────
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

    let gpu_scene = build_gpu_scene(&device, &scene)?;

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

    let instance_count = gpu_scene.instances.len() as u32;
    let mesh = gpu_scene.mesh;

    // Camera from the scene description.
    let cam = scene.camera.clone();
    let cam_pos = Vec3::from(cam.position);
    let cam_target = Vec3::from(cam.target);
    let cam_up = Vec3::from(cam.up);

    // ── Render loop ───────────────────────────────────────────────────
    let f = std::rc::Rc::new(std::cell::RefCell::new(
        None as Option<Closure<dyn FnMut()>>,
    ));
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

        // Camera for the current aspect ratio
        let aspect = config.width as f32 / config.height as f32;
        let view = Mat4::look_at_rh(cam_pos, cam_target, cam_up);
        let proj = Mat4::perspective_rh(cam.fov.to_radians(), aspect, cam.near, cam.far);
        let view_proj = proj * view;
        renderer.set_camera(&queue, &view_proj.to_cols_array_2d(), cam.position);

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
                } else if frame_count % 600 == 0 {
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
