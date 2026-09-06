//! GPU resources as ECS singletons — дизайн единого scheduler'а (S6→S7).
//!
//! Цель: `Device`/`Queue`/`SurfaceConfiguration`/`Renderer3D`/`RenderFrame3D`
//! как ресурсы `World`, чтобы `RenderSubmit` стал обычным `System` в
//! `Engine::schedule` вместо императива `GameContext::render_frame`.
//! Пассы `FramePlan` уже типизированы (`FramePass` с `Reads`/`Writes`),
//! их уровни — `bitset_level_plan` из `ornis-schedule` (единый движок с
//! `Schedule`). Здесь фиксируется контракт, как GPU-объекты входят в мир.
//!
//! # Контракт
//! - `GpuDevice`/`GpuQueue` — тонкие обёртки над `wgpu` объектами, `Send+Sync`.
//! - `GpuSurfaceState` — размер + формат + present mode, мутируется на resize.
//! - `GpuFrameState` — `Renderer3D` + `RenderFrame3D` + `Mesh` в `Mutex` (внутренняя
//!   изменяемость как у `RenderExtracted`/`OrbitCamera`). Хранит пул слотов
//!   `FrameExecutor` между кадрами.
//! - `RenderSubmit` — `System` читает `RenderExtracted` + `OrbitCamera`, пишет
//!   `GpuFrameState` (через `Mutex`), внутри делает `set_camera`/`upload_*`.
//!   Зависимость от `RenderExtract` выводится автоматически (RaW на
//!   `Mutex<RenderExtracted>`), порядок — уровень после extraction. Поверхностный
//!   `frame_plan.render` + `queue.submit`/`present` остаётся в `GameApp` на
//!   S7-шаге 1 (acquire вне системы); шаг 2 перенесёт `Surface` в ресурс.

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

/// GPU-состояние кадра: renderer + frame plan + mesh, pooled между кадрами.
///
/// Хранится как `Mutex<GpuFrameState>` ресурс, чтобы `System::run(&Resources)`
/// мог мутировать его через interior mutability.
pub struct GpuFrameState {
    /// Deferred renderer (pipelines + buffers).
    pub renderer: Renderer3D,
    /// Frame plan с пулом текстур (`FrameExecutor` внутри).
    pub frame_plan: RenderFrame3D,
    /// Сфера-меш кадра.
    pub mesh: Mesh,
    /// Кешированные параметры меша для пересоздания.
    pub mesh_params: (u32, u32),
}

/// Регистрирует GPU-ресурсы в `engine`.
///
/// Вызывать после создания `Device`/`Queue`/`Renderer3D`/`RenderFrame3D`/`Mesh`
/// в `GameApp::initialize` — до первого `run_frame`. После этого `RenderSubmit`
/// в `schedule` видит те же объекты без копирования.
pub fn install_gpu_resources(
    engine: &mut ornis_core::Engine,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_state: GpuSurfaceState,
    frame_state: GpuFrameState,
) {
    let _ = engine.world_mut().insert(GpuDevice(device));
    let _ = engine.world_mut().insert(GpuQueue(queue));
    let _ = engine.world_mut().insert(surface_state);
    let _ = engine.world_mut().insert(Mutex::new(frame_state));
    engine.schedule_mut().add_system(RenderSubmit);
}

/// Система сабмита кадра: читает extraction + камеру, пишет GPU-состояние.
///
/// S7-шаг 1: делает `set_camera`/`upload_*` и пересоздаёт меш при смене
/// `mesh_params`. `frame_plan.render` + `queue.submit`/`present` остаётся в
/// `GameApp::render_frame` после `engine.run_frame`, чтобы `Surface` acquire
/// не требовал `Mutex<Surface>` в `System`.
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
