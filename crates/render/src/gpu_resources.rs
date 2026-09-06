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
//!   `GpuFrameState` (через `Mutex`), внутри делает `set_camera`/`upload_*`/
//!   `FrameExecutor::execute` и `queue.submit`. Зависимость от `RenderExtract`
//!   выводится автоматически (RaW на `Mutex<RenderExtracted>`), порядок —
//!   уровень после extraction.
//! - `Surface`/`get_current_texture` остаётся вне `System` на S7-шаге 1
//!   (acquire/present — imperative в `GameApp`), на шаге 2 — `Mutex<Surface>`
//!   ресурс внутри `GpuFrameState` с обработкой `Outdated`/`Lost` внутри системы.
//!
//! Пока модуль — дизайн + типы без интеграции в `GameApp` (чтобы не ломать
//! `render_frame`). Интеграция — следующий шаг `integrate`.

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
/// Сейчас — stub (только контракт доступов и уровней). Тело, делающее
/// `set_camera`/`upload_*`/`frame_plan.render` + `queue.submit`, подключается
/// на шаге `integrate` (требует `Surface`/`Target` + `queue.present`).
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
        // Stub: проверяет, что все ресурсы объявлены и доступны, но пока не
        // трогает GPU (surface acquire остаётся в GameApp). Реальное тело —
        // следующий коммит `integrate`.
        let _ = resources.get::<Mutex<RenderExtracted>>();
        let _ = resources.get::<Mutex<crate::camera::OrbitCamera>>();
        let _ = resources.get::<Mutex<GpuFrameState>>();
        let _ = resources.get::<GpuDevice>();
        let _ = resources.get::<GpuQueue>();
        let _ = resources.get::<GpuSurfaceState>();
    }
}
