//! GPU resources as ECS singletons — дизайн единого scheduler'а (S6→S7).
//!
//! Цель: `Device`/`Queue`/`Surface`/`SurfaceConfiguration`/`Renderer3D`/`RenderFrame3D`
//! как ресурсы `World`, чтобы `RenderSubmit` (upload) и `RenderPresent`
//! (acquire → record → submit → present) стали обычными `System` в
//! `Engine::schedule` вместо императива `GameContext::render_frame`.
//! Пассы уже типизированы (`FramePass` с `Reads`/`Writes`),
//! их уровни — `bitset_level_plan` из `ornis-schedule` (единый движок с
//! `Schedule`). Здесь фиксируется контракт, как GPU-объекты входят в мир.
//!
//! # Контракт
//! - `GpuDevice`/`GpuQueue` — тонкие обёртки над `wgpu` объектами, `Send+Sync`.
//! - `GpuSurfaceState` — размер + формат + present mode, мутируется на resize.
//! - `GpuSurface` — `wgpu::Surface` в `Mutex` (внутренняя изменяемость как у
//!   `RenderExtracted`/`OrbitCamera`). Хранится отдельно от `GpuSurfaceState`,
//!   чтобы `RenderSubmit` мог читать размер без блокировки `Surface`.
//! - `GpuFrameState` — `Renderer3D` + `RenderFrame3D` + `Mesh` в `Mutex`.
//!   Хранит пул слотов `FrameExecutor` между кадрами.
//! - `RenderSubmit` — `System` читает `RenderExtracted` + `OrbitCamera`, пишет
//!   `GpuFrameState` (через `Mutex`), внутри делает `set_camera`/`upload_*`.
//!   Зависимость от `RenderExtract` выводится автоматически (RaW на
//!   `Mutex<RenderExtracted>`), порядок — уровень после extraction.
//! - `RenderPresent` — `System` читает `GpuSurface`/`GpuSurfaceState`/
//!   `GpuDevice`/`GpuQueue` + `Mutex<RenderExtracted>` (instance count) и пишет
//!   `GpuFrameState` (`&mut RenderFrame3D` для записи команд). Делает
//!   `surface.get_current_texture → create_view → frame3d.render →
//!   queue.submit → queue.present`. Ошибки `Outdated`/`Lost` —
//!   реконфигурируют `Surface` на месте; `Occluded`/`Timeout`/`Validation` —
//!   пропускают кадр. `Suboptimal` трактуется как `Success`.
//!   После этого `GameApp::render_frame` сводится к `run_frame`.

use std::sync::Mutex;

use ornis_core::{Resources, System, SystemAccess};

use crate::extraction::RenderExtracted;
use crate::frame_exec::RenderFrame3D;
use crate::mesh::Mesh;
use crate::renderer::Renderer3D;

/// Обёртка над `wgpu::Device` как ECS-ресурс.
pub struct GpuDevice(pub wgpu::Device);

/// Обёртка над `wgpu::Queue` как ECS-ресурс.
pub struct GpuQueue(pub wgpu::Queue);

/// Поверхностное состояние, которое меняется на resize.
#[derive(Debug, Clone)]
pub struct GpuSurfaceState {
    /// Текущий размер поверхности.
    pub size: (u32, u32),
    /// Формат поверхности.
    pub format: wgpu::TextureFormat,
}

/// GPU-поверхность как ECS-ресурс.
///
/// Хранится в `Mutex`, чтобы `System::run(&Resources)` мог вызывать
/// `get_current_texture` и `configure` через interior mutability.
pub struct GpuSurface(pub Mutex<wgpu::Surface<'static>>);

/// GPU-состояние кадра: renderer + frame plan + mesh, pooled между кадрами.
///
/// Хранится как `Mutex<GpuFrameState>` ресурс, чтобы `System::run(&Resources)`
/// мог мутировать его через interior mutability.
pub struct GpuFrameState {
    /// Deferred renderer (pipelines + buffers).
    pub renderer: Renderer3D,
    /// Frame plan с пулом текстур (`FrameExecutor` внутри).
    pub frame3d: RenderFrame3D,
    /// Сфера-меш кадра.
    pub mesh: Mesh,
    /// Кешированные параметры меша для пересоздания.
    pub mesh_params: (u32, u32),
}

/// Регистрирует GPU-ресурсы в `engine`.
///
/// Вызывать после создания `Device`/`Queue`/`Surface`/`Renderer3D`/
/// `RenderFrame3D`/`Mesh` в `GameApp::initialize` — до первого `run_frame`.
/// После этого `RenderSubmit` и `RenderPresent` в `schedule` видят те же
/// объекты без копирования.
pub fn install_gpu_resources(
    engine: &mut ornis_core::Engine,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_state: GpuSurfaceState,
    frame_state: GpuFrameState,
) {
    let _ = engine.world_mut().insert(GpuDevice(device));
    let _ = engine.world_mut().insert(GpuQueue(queue));
    let _ = engine.world_mut().insert(GpuSurface(Mutex::new(surface)));
    let _ = engine.world_mut().insert(surface_state);
    let _ = engine.world_mut().insert(Mutex::new(frame_state));
    engine.schedule_mut().add_system(RenderSubmit);
    engine.schedule_mut().add_system(RenderPresent);
}

/// Система сабмита кадра: читает extraction + камеру, пишет GPU-состояние.
///
/// S7-шаг 1: делает `set_camera`/`upload_*` и пересоздаёт меш при смене
/// `mesh_params`. `frame3d.render` + `queue.submit`/`present` — в
/// `RenderPresent` (S7-шаг 2).
struct RenderSubmit;

impl System for RenderSubmit {
    fn name(&self) -> &'static str {
        "render_submit"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<Mutex<RenderExtracted>>()
            .reads::<Mutex<crate::camera::OrbitCamera>>()
            .writes::<Mutex<GpuFrameState>>()
            .reads::<GpuDevice>()
            .reads::<GpuQueue>()
            .reads::<GpuSurfaceState>()
    }

    fn run(&self, resources: &Resources) {
        let Some(extracted) = resources
            .get::<Mutex<RenderExtracted>>()
            .map(|m| m.lock().expect("render extraction lock").clone())
        else {
            return;
        };
        let Some(orbit) = resources
            .get::<Mutex<crate::camera::OrbitCamera>>()
            .map(|m| m.lock().expect("orbit camera lock").clone())
        else {
            return;
        };
        let Some(device) = resources.get::<GpuDevice>() else {
            return;
        };
        let Some(queue) = resources.get::<GpuQueue>() else {
            return;
        };
        let Some(surface_state) = resources.get::<GpuSurfaceState>() else {
            return;
        };
        let Some(frame_state) = resources.get::<Mutex<GpuFrameState>>() else {
            return;
        };
        let mut fs = frame_state.lock().expect("gpu frame state lock");

        // Пересоздать меш если extraction требует другую тесселяцию.
        if extracted.mesh_params != fs.mesh_params {
            fs.mesh = crate::mesh::create_sphere(
                &device.0,
                1.0,
                extracted.mesh_params.0,
                extracted.mesh_params.1,
            );
            fs.mesh_params = extracted.mesh_params;
        }

        let (w, h) = (surface_state.size.0 as f64, surface_state.size.1 as f64);
        let aspect = if h > 0.0 { w as f32 / h as f32 } else { 1.0 };
        let (cam_pos, cam_target, cam_up, fov, near, far) = orbit.view_parameters();
        let view = glam::camera::rh::view::look_at_mat4(cam_pos, cam_target, cam_up);
        let proj =
            glam::camera::rh::proj::directx::perspective(fov.to_radians(), aspect, near, far);
        let view_proj = proj * view;

        fs.renderer
            .set_camera(&queue.0, &view_proj.to_cols_array_2d(), cam_pos.to_array());
        fs.renderer.set_lights(
            &queue.0,
            [0.10, 0.10, 0.15],
            &[
                ([1.0, 1.0, 1.0], 0.6, [1.0, 1.0, 1.0]),
                ([-0.5, 0.5, -0.5], 0.3, [0.8, 0.8, 1.0]),
            ],
        );
        fs.renderer.upload_materials(&queue.0, &extracted.materials);
        fs.renderer.upload_instances(&queue.0, &extracted.instances);
    }
}

/// Система презента кадра: acquire → record → submit → present.
///
/// S7-шаг 2: переносит `Surface` acquire/present из `GameApp::render_frame`
/// в `Engine::schedule`. Зависимость от `RenderSubmit` выводится как WaW по
/// `Mutex<GpuFrameState>`; порядок — регистрация после `RenderSubmit`.
struct RenderPresent;

impl System for RenderPresent {
    fn name(&self) -> &'static str {
        "render_present"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<GpuDevice>()
            .reads::<GpuQueue>()
            .reads::<GpuSurface>()
            .reads::<GpuSurfaceState>()
            .reads::<Mutex<RenderExtracted>>()
            .writes::<Mutex<GpuFrameState>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(device) = resources.get::<GpuDevice>() else {
            return;
        };
        let Some(queue) = resources.get::<GpuQueue>() else {
            return;
        };
        let Some(surface_state) = resources.get::<GpuSurfaceState>().cloned() else {
            return;
        };
        let Some(surface) = resources.get::<GpuSurface>() else {
            return;
        };
        let Some(extracted) = resources
            .get::<Mutex<RenderExtracted>>()
            .map(|m| m.lock().expect("render extraction lock").clone())
        else {
            return;
        };
        let Some(frame_state) = resources.get::<Mutex<GpuFrameState>>() else {
            return;
        };

        // Acquire swapchain texture. Hold the Surface lock only for the acquire
        // and for an optional reconfigure on Outdated/Lost.
        let frame = {
            let guard = surface.0.lock().expect("gpu surface lock");
            match guard.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(frame)
                | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => Some(frame),
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    drop(guard);
                    // Reconfigure with the last known size/format and default
                    // present parameters (matches GameApp::initialize).
                    let guard = surface.0.lock().expect("gpu surface lock");
                    guard.configure(
                        &device.0,
                        &wgpu::SurfaceConfiguration {
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            format: surface_state.format,
                            width: surface_state.size.0.max(1),
                            height: surface_state.size.1.max(1),
                            present_mode: wgpu::PresentMode::AutoNoVsync,
                            alpha_mode: wgpu::CompositeAlphaMode::Auto,
                            view_formats: vec![],
                            desired_maximum_frame_latency: 2,
                            color_space: wgpu::SurfaceColorSpace::Auto,
                        },
                    );
                    return;
                }
                wgpu::CurrentSurfaceTexture::Occluded
                | wgpu::CurrentSurfaceTexture::Timeout
                | wgpu::CurrentSurfaceTexture::Validation => {
                    return;
                }
            }
        };

        let Some(frame) = frame else {
            return;
        };

        let mut encoder = device
            .0
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ornis frame"),
            });
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let instance_count = extracted.instances.len() as u32;
        let context = crate::RenderContext {
            device: &device.0,
            queue: &queue.0,
            encoder: &mut encoder,
            target: &frame_view,
        };

        {
            let mut fs = frame_state.lock().expect("gpu frame state lock");
            // renderer/mesh + frame3d from the same Mutex< GpuFrameState>.
            // Raw pointers avoid double &mut borrow of disjoint fields through
            // a single MutexGuard (safe: different fields).
            let renderer = &fs.renderer as *const Renderer3D;
            let mesh = &fs.mesh as *const Mesh;
            let frame3d = &mut fs.frame3d as *mut RenderFrame3D;
            unsafe {
                (*frame3d).render(context, &*renderer, &*mesh, instance_count);
            }
        }

        queue.0.submit(Some(encoder.finish()));
        queue.0.present(frame);
    }
}
