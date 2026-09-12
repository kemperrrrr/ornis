//! GPU resources as ECS singletons — дизайн единого scheduler'а (S6→S7).
//!
//! Цель: `Device`/`Queue`/`Surface`/`SurfaceConfiguration`/`Renderer3D`/`RenderFrame3D`
//! как ресурсы `World`, чтобы `RenderSubmit` (upload), `RenderPresent`
//! (acquire → record) и `RenderFlush` (ordered submit → present) стали
//! обычными `System` в `Engine::schedule` вместо императива
//! `GameContext::render_frame`.
//! Пассы уже типизированы (`FramePass` с `Reads`/`Writes`),
//! их уровни — `bitset_level_plan` из `ornis-schedule` (единый движок с
//! `Schedule`). Здесь фиксируется контракт, как GPU-объекты входят в мир.
//!
//! # Контракт
//! - `GpuDevice`/`GpuQueue` — тонкие обёртки над `wgpu` объектами, `Send+Sync`.
//! - `GpuSurfaceState` — размер + формат + present mode, мутируется на resize.
//! - `GpuSurface` — `wgpu::Surface` в `Mutex` (внутренняя изменяемость как у
//!   `OrbitCamera`). Хранится отдельно от `GpuSurfaceState`,
//!   чтобы `RenderSubmit` мог читать размер без блокировки `Surface`.
//! - `GpuFrameState` — `Renderer3D` + `RenderFrame3D` в `Mutex`.
//!   Хранит пул слотов `FrameExecutor` между кадрами.
//! - `GpuMesh` — `Mesh` + кеш тесселяции в `Mutex` (X2): отдельный от
//!   `GpuFrameState` ресурс, пересоздание из лейна не держит лок
//!   renderer'а. Порядок локов: `GpuMesh` раньше `GpuFrameState`.
//! - `RenderSubmit` — `System` читает лейны `TransformDesc`/`MeshDesc`/
//!   `MaterialDesc` напрямую (`reads_lane`, X1/Extract-free, канон S5d)
//!   + `OrbitCamera` + `RenderLights` (X3: ambient/направленные источники
//!   из мира, не хардкод), пишет `GpuFrameState` (через `Mutex`), внутри
//!   делает `set_camera`/`upload_*`. Зависимость от пишущих лейны систем —
//!   RaW по лейнам; снапшотов больше нет (X4) — канон `extract_render_data`
//!   читает лейны напрямую.
//! - `RenderMesh` — `System` (X2) пишет только `Mutex<GpuMesh>`:
//!   пересоздаёт сферу, когда `max_mesh_params` из лейна (тот же канон,
//!   что у `extract_render_data`) отличается от кеша.
//! - `RenderPresent` — `System` читает `GpuSurface`/`GpuSurfaceState`/
//!   `GpuDevice`/`GpuQueue` + лейны (X4: instance count — прямой
//!   `extract_render_data`) + `Mutex<GpuMesh>` (X2) и пишет
//!   `GpuFrameState` (`&mut RenderFrame3D` для записи команд) + оба
//!   E2-handover-ресурса. Делает `surface.get_current_texture → create_view →
//!   frame3d.render_to_buffers` (per-pass encoders → `FrameCommandBuffers`,
//!   acquired frame → `FramePresentTarget`). Ошибки `Outdated`/`Lost` —
//!   реконфигурируют `Surface` на месте; `Occluded`/`Timeout`/`Validation` —
//!   пропускают кадр. `Suboptimal` трактуется как `Success`.
//! - `RenderFlush` — `System` (E2) сливает `FrameCommandBuffers` одним
//!   ordered submit (`FrameCommandBuffers::flush`) и презентует acquired
//!   frame из `FramePresentTarget`. Порядок после `RenderPresent`
//!   гарантирован WaW по обоим handover-ресурсам (регистрация следом).
//!   После этого `GameApp::render_frame` сводится к `run_frame`.

use std::sync::Mutex;

use ornis_core::{Resources, SmartStore, System, SystemAccess};

use crate::camera::camera_view_projection;
use crate::extraction::{RenderLights, extract_render_data, max_mesh_params};
use crate::frame_exec::{BufferRenderContext, RenderFrame3D};
use crate::mesh::Mesh;
use crate::renderer::Renderer3D;
use crate::scene::{MaterialDesc, MeshDesc, TransformDesc};

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

/// Готовые command-буферы кадра, ожидающие submit (стадия 1 handover
/// encoder'а в `World`, No-encoder doubling).
///
/// Живой `wgpu::CommandEncoder` остаётся frame-локальным: его создаёт
/// `RenderPresent` на каждый кадр (`device.create_command_encoder`), потому
/// что encoder — короткоживущее незавершённое состояние записи, а не
/// разделяемый синглтон. Через `World` передаётся только завершённый
/// продукт — `Vec<wgpu::CommandBuffer>` за `Mutex` (interior mutability,
/// как у `GpuSurface`/`GpuFrameState`).
/// Стадия 2 (не входит сюда): `RenderPresent` пушит сюда вместо прямого
/// `queue.submit`, а отдельная система сливает буферы в порядке
/// регистрации — тогда ни одна система не владеет encoder'ом напрямую.
#[derive(Debug, Default)]
pub struct FrameCommandBuffers(pub Mutex<Vec<wgpu::CommandBuffer>>);

impl FrameCommandBuffers {
    /// E2 (S5e): drains the recorded buffers in registration order and
    /// submits them with a single ordered submit — the flush half of the
    /// encoder-as-frame-resource handover (the runtime's `RenderFlush`
    /// system). Empty handover is a no-op submit.
    pub fn flush(&self, queue: &wgpu::Queue) {
        let mut guard = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        queue.submit(guard.drain(..));
    }
}

/// Acquired swapchain texture of the current frame, handed from the
/// acquiring system ([`RenderPresent`]) to the submitting one
/// ([`RenderFlush`]) — the present half of the E2 handover.
///
/// `Mutex<Option<…>>` interior mutability, as with the other GPU
/// resources: systems run against `&Resources`.
#[derive(Debug, Default)]
pub struct FramePresentTarget(pub Mutex<Option<wgpu::SurfaceTexture>>);

/// Регистрирует [`FrameCommandBuffers`] в мире движка (стадия 1 handover).
///
/// Вызывать один раз до первого `run_frame`; повторный вызов заменяет
/// ресурс пустым (потеря pending-буферов — только при неверном порядке
/// инициализации, в steady state не вызывается).
pub fn install_frame_buffers(engine: &mut ornis_core::Engine) {
    let _ = engine.world_mut().insert(FrameCommandBuffers::default());
}
/// GPU-состояние кадра: renderer + frame plan, pooled между кадрами.
///
/// Хранится как `Mutex<GpuFrameState>` ресурс, чтобы `System::run(&Resources)`
/// мог мутировать его через interior mutability. Меш — отдельный ресурс
/// [`GpuMesh`] (X2): пересоздание из лейна не держит лок renderer'а.
pub struct GpuFrameState {
    /// Deferred renderer (pipelines + buffers).
    pub renderer: Renderer3D,
    /// Frame plan с пулом текстур (`FrameExecutor` внутри).
    pub frame3d: RenderFrame3D,
}

/// GPU-меш кадра как отдельный ресурс (X2, Extract-free).
///
/// `params` — тесселяция, из которой `mesh` создан; критерий пересоздания —
/// `max_mesh_params` из `MeshDesc`-лейна (тот же канон, что у
/// `extract_render_data`). Система `RenderMesh` пишет только этот
/// ресурс; читатели (`RenderPresent`) упорядочены RaW. Хранится как
/// `Mutex<GpuMesh>` — interior mutability, как у `GpuFrameState`.
pub struct GpuMesh {
    /// Сфера-меш кадра.
    pub mesh: Mesh,
    /// Тесселяция `mesh` — кеш критерия пересоздания.
    pub params: (u32, u32),
}

/// Регистрирует [`GpuMesh`] в мире и систему его пересоздания (X2).
///
/// `RenderMesh` читает лейны (`reads_lane`, канон S5d) и пишет только
/// `Mutex<GpuMesh>`; для `create_sphere` нужен `GpuDevice` в ресурсах —
/// вставьте его до первого `run_frame` (`install_gpu_resources` делает
/// это сам). Повторный вызов заменяет ресурс (steady state — не
/// вызывается).
pub fn install_render_mesh(engine: &mut ornis_core::Engine, mesh: GpuMesh) {
    let _ = engine.world_mut().insert(Mutex::new(mesh));
    engine.schedule_mut().add_system(RenderMesh);
}

/// Регистрирует GPU-ресурсы в `engine`.
///
/// Вызывать после создания `Device`/`Queue`/`Surface`/`Renderer3D`/
/// `RenderFrame3D`/`Mesh` в `GameApp::initialize` — до первого `run_frame`.
/// После этого `RenderMesh`/`RenderSubmit`/`RenderPresent`/`RenderFlush`
/// в `schedule` видят те же объекты без копирования.
pub fn install_gpu_resources(
    engine: &mut ornis_core::Engine,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_state: GpuSurfaceState,
    frame_state: GpuFrameState,
    mesh: GpuMesh,
) {
    install_frame_buffers(engine);
    let _ = engine.world_mut().insert(GpuDevice(device));
    let _ = engine.world_mut().insert(GpuQueue(queue));
    let _ = engine.world_mut().insert(GpuSurface(Mutex::new(surface)));
    let _ = engine.world_mut().insert(surface_state);
    let _ = engine.world_mut().insert(Mutex::new(frame_state));
    let _ = engine.world_mut().insert(FramePresentTarget::default());
    let _ = engine.world_mut().insert(RenderLights::default());
    install_render_mesh(engine, mesh);
    engine.schedule_mut().add_system(RenderSubmit);
    engine.schedule_mut().add_system(RenderPresent);
    engine.schedule_mut().add_system(RenderFlush);
}

/// Система пересоздания меша (X2): тесселяция — из лейна, не из снапшота.
///
/// `max_mesh_params` — тот же канон, что у `extract_render_data`
/// (полные сущности, пол (32, 24)). Пишет только `Mutex<GpuMesh>`;
/// читатели (`RenderPresent`) упорядочены RaW, с `RenderSubmit` общих
/// ресурсов нет. Порядок локов: `GpuMesh` раньше `GpuFrameState`
/// (иначе — только здесь, один лок на систему).
struct RenderMesh;

impl System for RenderMesh {
    fn name(&self) -> &'static str {
        "render_mesh"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<TransformDesc>()
            .reads_lane::<MeshDesc>()
            .reads_lane::<MaterialDesc>()
            .reads::<GpuDevice>()
            .writes::<Mutex<GpuMesh>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(device) = resources.get::<GpuDevice>() else {
            return;
        };
        let Some(mesh_resource) = resources.get::<Mutex<GpuMesh>>() else {
            return;
        };
        let params = max_mesh_params(store);
        let mut mesh_state = mesh_resource
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if params != mesh_state.params {
            mesh_state.mesh = crate::mesh::create_sphere(&device.0, 1.0, params.0, params.1);
            mesh_state.params = params;
        }
    }
}

/// Система сабмита кадра: читает лейны + камеру + свет, пишет GPU-состояние.
///
/// S7-шаг 1: делает `set_camera`/`upload_*`. X1 (Extract-free): данные
/// материалов/инстансов — прямое чтение `TransformDesc`/`MeshDesc`/
/// `MaterialDesc`-лейн (канон S5d) через `extract_render_data`.
/// X2: пересоздание меша ушло в `RenderMesh` (`GpuMesh`-ресурс).
/// X3: свет — из `RenderLights`-ресурса (сцено-загрузчик), не хардкод.
/// `frame3d.render` — в `RenderPresent` (S7-шаг 2).
struct RenderSubmit;

impl System for RenderSubmit {
    fn name(&self) -> &'static str {
        "render_submit"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<TransformDesc>()
            .reads_lane::<MeshDesc>()
            .reads_lane::<MaterialDesc>()
            .reads::<Mutex<crate::camera::OrbitCamera>>()
            .reads::<RenderLights>()
            .writes::<Mutex<GpuFrameState>>()
            .reads::<GpuDevice>()
            .reads::<GpuQueue>()
            .reads::<GpuSurfaceState>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(orbit) = resources
            .get::<Mutex<crate::camera::OrbitCamera>>()
            .map(|m| m.lock().expect("orbit camera lock").clone())
        else {
            return;
        };
        let Some(lights) = resources.get::<RenderLights>() else {
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
        let fs = frame_state.lock().expect("gpu frame state lock");

        // X1/X4: direct lane read through the shared canon — no snapshot.
        let extracted = extract_render_data(store);
        // S7: the camera math lives in `camera_view_projection` — `run`
        // stays pure orchestration (IOSP).
        let (view_proj, cam_pos) =
            camera_view_projection(orbit.view_parameters(), surface_state.size);

        fs.renderer
            .set_camera(&queue.0, &view_proj.to_cols_array_2d(), cam_pos.to_array());
        fs.renderer
            .set_lights(&queue.0, lights.ambient, &lights.set_lights_args());
        fs.renderer
            .upload_materials(&device.0, &queue.0, &extracted.materials);
        fs.renderer
            .upload_instances(&device.0, &queue.0, &extracted.instances);
    }
}

/// Система презента кадра: acquire → record (E2: encoder как frame-ресурс).
///
/// S7-шаг 2: переносит `Surface` acquire из `GameApp::render_frame`
/// в `Engine::schedule`. E2 (S5e): запись идёт через per-pass encoders в
/// `FrameCommandBuffers` (`RenderFrame3D::render_to_buffers`), acquired
/// frame уходит в `FramePresentTarget`; submit + present выполняет
/// отдельная система `RenderFlush` (регистрация следом — WaW по обоим
/// handover-ресурсам). X2: меш читается из `GpuMesh`-ресурса (RaW после
/// `RenderMesh`). X4: instance count — прямое чтение лейнов
/// (`extract_render_data`), не снапшот. Зависимость от `RenderSubmit`
/// выводится как WaW по `Mutex<GpuFrameState>`; порядок — регистрация
/// после `RenderSubmit`.
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
            .reads::<SmartStore>()
            .reads_lane::<TransformDesc>()
            .reads_lane::<MeshDesc>()
            .reads_lane::<MaterialDesc>()
            .reads::<Mutex<GpuMesh>>()
            .writes::<Mutex<GpuFrameState>>()
            .writes::<FrameCommandBuffers>()
            .writes::<FramePresentTarget>()
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
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(frame_state) = resources.get::<Mutex<GpuFrameState>>() else {
            return;
        };
        let Some(mesh_resource) = resources.get::<Mutex<GpuMesh>>() else {
            return;
        };
        let Some(buffers) = resources.get::<FrameCommandBuffers>() else {
            return;
        };
        let Some(present_target) = resources.get::<FramePresentTarget>() else {
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

        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // X4: instance count straight from the lane canon — no snapshot.
        let instance_count = extract_render_data(store).instances.len() as u32;

        {
            // X2: the mesh is its own resource — lock order mesh before
            // frame state (RenderMesh holds only the mesh lock).
            let mesh_state = mesh_resource
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut fs = frame_state.lock().expect("gpu frame state lock");
            // renderer + frame3d from the same Mutex<GpuFrameState>.
            // Raw pointers avoid double &mut borrow of disjoint fields through
            // a single MutexGuard (safe: different fields).
            let renderer = &fs.renderer as *const Renderer3D;
            let frame3d = &mut fs.frame3d as *mut RenderFrame3D;
            unsafe {
                // E2: encoder context as frame data — per-pass encoders
                // land in FrameCommandBuffers; submit + present live in
                // RenderFlush now.
                let context = BufferRenderContext {
                    device: &device.0,
                    queue: &queue.0,
                    target: &frame_view,
                    renderer: &*renderer,
                    mesh: &mesh_state.mesh,
                    instance_count,
                    buffers,
                };
                (*frame3d)
                    .render_to_buffers(context)
                    .expect("E2 render_to_buffers: projection failed");
            }
        }

        // Hand the acquired frame over: recording and the ordered
        // submit + present now compose inside one schedule.
        let mut slot = present_target
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *slot = Some(frame);
    }
}

/// Система слива кадра: ordered submit + present (E2).
///
/// Отдельная система после `RenderPresent`: сливает `FrameCommandBuffers`
/// в порядке регистрации (один submit) и презентует acquired frame из
/// `FramePresentTarget`. Порядок гарантирован WaW по обоим
/// handover-ресурсам при регистрации следом за `RenderPresent`.
struct RenderFlush;

impl System for RenderFlush {
    fn name(&self) -> &'static str {
        "render_flush"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<GpuQueue>()
            .writes::<FrameCommandBuffers>()
            .writes::<FramePresentTarget>()
    }

    fn run(&self, resources: &Resources) {
        let Some(queue) = resources.get::<GpuQueue>() else {
            return;
        };
        let Some(buffers) = resources.get::<FrameCommandBuffers>() else {
            return;
        };
        let Some(present_target) = resources.get::<FramePresentTarget>() else {
            return;
        };
        let frame = present_target
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        buffers.flush(&queue.0);
        if let Some(frame) = frame {
            queue.0.present(frame);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::Engine;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn frame_command_buffers_are_world_resources() {
        // Compile-time proof of the handover design: finished buffers and
        // the acquired frame are `Send + Sync` (hence `Resources`-
        // compatible), unlike the live encoder which stays frame-local.
        assert_send_sync::<FrameCommandBuffers>();
        assert_send_sync::<wgpu::CommandBuffer>();
        assert_send_sync::<FramePresentTarget>();
        assert_send_sync::<wgpu::SurfaceTexture>();

        let mut engine = Engine::new();
        install_frame_buffers(&mut engine);
        let buffers = engine
            .world()
            .resources()
            .get::<FrameCommandBuffers>()
            .expect("frame buffers resource");
        assert!(
            buffers.0.lock().expect("frame buffers lock").is_empty(),
            "fresh install holds no pending buffers"
        );
    }
}
