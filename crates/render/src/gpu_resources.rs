//! GPU resources as ECS singletons — unified-scheduler design (S6→S7).
//!
//! Goal: `Device`/`Queue`/`Surface`/`SurfaceConfiguration`/`Renderer3D`/`RenderFrame3D`
//! as `World` resources, so `RenderSubmit` (upload), `RenderPresent`
//! (acquire → record) and `RenderFlush` (ordered submit → present) become
//! ordinary `System`s in `Engine::schedule` instead of imperative
//! `GameContext::render_frame`.
//! Passes are already typed (`FramePass` with `Reads`/`Writes`),
//! their levels come from `bitset_level_plan` in `ornis-schedule` (one engine
//! shared with `Schedule`). This module pins the contract for how GPU
//! objects enter the world.
//!
//! # Contract
//! - `GpuDevice`/`GpuQueue` — thin wrappers over `wgpu` objects, `Send+Sync`.
//! - `GpuSurfaceState` — size + format + present mode, mutated on resize.
//! - `GpuSurface` — `wgpu::Surface` behind a `Mutex` (interior mutability as
//!   in `OrbitCamera`). Stored separately from `GpuSurfaceState` so
//!   `RenderSubmit` can read the size without locking the `Surface`.
//! - `GpuFrameState` — `Renderer3D` + `RenderFrame3D` behind a `Mutex`.
//!   Holds the `FrameExecutor` pool slots across frames.
//! - `GpuMesh` — `Mesh` + tessellation cache behind a `Mutex` (X2): a
//!   resource separate from `GpuFrameState`, so rebuilding from the lane
//!   never holds the renderer lock. Lock order: `GpuMesh` before
//!   `GpuFrameState`.
//! - `RenderSubmit` — a `System` reading the `TransformDesc`/`MeshDesc`/
//!   `MaterialDesc` lanes directly (`reads_lane`, X1/Extract-free, S5d canon)
//!   + `OrbitCamera` + `RenderLights` (X3: ambient/directional lights from
//!   the world, not hardcoded), writing `GpuFrameState` (through the
//!   `Mutex`); internally it runs `set_camera`/`upload_*`. The dependency
//!   on lane-writing systems is RaW over lanes; snapshots are gone (X4) —
//!   the `extract_render_data` canon reads lanes directly.
//! - `RenderMesh` — a `System` (X2) writing only `Mutex<GpuMesh>`:
//!   rebuilds the sphere when `max_mesh_params` from the lane (same canon
//!   as `extract_render_data`) differs from the cache.
//! - `RenderPresent` — a `System` reading `GpuSurface`/`GpuSurfaceState`/
//!   `GpuDevice`/`GpuQueue` + lanes (X4: instance count — direct
//!   `extract_render_data`) + `Mutex<GpuMesh>` (X2) and writing
//!   `GpuFrameState` (`&mut RenderFrame3D` for command recording) + both
//!   E2-handover resources. Runs `surface.get_current_texture → create_view →
//!   frame3d.render_to_buffers` (per-pass encoders → `FrameCommandBuffers`,
//!   acquired frame → `FramePresentTarget`). `Outdated`/`Lost` errors
//!   reconfigure the `Surface` in place; `Occluded`/`Timeout`/`Validation`
//!   skip the frame. `Suboptimal` counts as `Success`.
//! - `RenderFlush` — a `System` (E2) draining `FrameCommandBuffers` with one
//!   ordered submit (`FrameCommandBuffers::flush`) and presenting the
//!   acquired frame from `FramePresentTarget`. Ordering after `RenderPresent`
//!   is guaranteed by WaW over both handover resources (registered right
//!   after). After this `GameApp::render_frame` reduces to `GameWorld::frame`
//!   (`run_frame` in the schedule + an off-plan `frame_upload` whose result
//!   the native path discards — the systems read the lanes themselves).

use std::sync::Mutex;

use ornis_core::{Resources, SmartStore, System, SystemAccess};

use crate::camera::camera_view_projection;
use crate::extraction::{RenderLights, extract_render_data, max_mesh_params};
use crate::frame_exec::{BufferRenderContext, RenderFrame3D};
use crate::mesh::Mesh;
use crate::renderer::{CustomGbufferDraw, Renderer3D, StagedCustomMesh, custom_draw_items};
use ornis_animation::{JointPose, Skeleton, SkinnedMesh};
use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};

/// Wrapper over `wgpu::Device` as an ECS resource.
pub struct GpuDevice(pub wgpu::Device);

/// Wrapper over `wgpu::Queue` as an ECS resource.
pub struct GpuQueue(pub wgpu::Queue);

/// Surface state that changes on resize.
#[derive(Debug, Clone)]
pub struct GpuSurfaceState {
    /// Current surface size.
    pub size: (u32, u32),
    /// Surface format.
    pub format: wgpu::TextureFormat,
}

/// GPU surface as an ECS resource.
///
/// Held behind a `Mutex` so `System::run(&Resources)` can call
/// `get_current_texture` and `configure` through interior mutability.
pub struct GpuSurface(pub Mutex<wgpu::Surface<'static>>);

/// Finished per-frame command buffers awaiting submit (stage 1 of the
/// encoder handover into `World`, no encoder doubling).
///
/// The live `wgpu::CommandEncoder` stays frame-local: `RenderPresent`
/// creates it every frame (`device.create_command_encoder`) because an
/// encoder is short-lived unfinished recording state, not a shared
/// singleton. Only the finished product crosses `World` —
/// `Vec<wgpu::CommandBuffer>` behind a `Mutex` (interior mutability,
/// as in `GpuSurface`/`GpuFrameState`).
/// Stage 2 (not here): `RenderPresent` pushes here instead of calling
/// `queue.submit` directly, and a separate system drains the buffers in
/// registration order — so no system owns an encoder outright.
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

/// Registers [`FrameCommandBuffers`] in the engine world (stage 1 handover).
///
/// Call once before the first `run_frame`; a repeat call replaces the
/// resource with an empty one (losing pending buffers only happens on a
/// wrong initialization order, never in steady state).
pub fn install_frame_buffers(engine: &mut ornis_core::Engine) {
    let _ = engine.world_mut().insert(FrameCommandBuffers::default());
}
/// Per-frame GPU state: renderer + frame plan, pooled across frames.
///
/// Stored as a `Mutex<GpuFrameState>` resource so `System::run(&Resources)`
/// can mutate it through interior mutability. The mesh is a separate
/// [`GpuMesh`] resource (X2): rebuilding from the lane never holds the
/// renderer lock.
pub struct GpuFrameState {
    /// Deferred renderer (pipelines + buffers).
    pub renderer: Renderer3D,
    /// Frame plan with the texture pool (`FrameExecutor` inside).
    pub frame3d: RenderFrame3D,
}

/// Per-frame GPU mesh as a separate resource (X2, Extract-free).
///
/// `params` is the tessellation `mesh` was built from; the rebuild
/// criterion is `max_mesh_params` from the `MeshDesc` lane (same canon as
/// `extract_render_data`). The `RenderMesh` system is the only writer of
/// this resource; readers (`RenderPresent`) order after it via RaW. Stored
/// as `Mutex<GpuMesh>` — interior mutability, as in `GpuFrameState`.
pub struct GpuMesh {
    /// Per-frame sphere mesh.
    pub mesh: Mesh,
    /// Tessellation of `mesh` — rebuild-criterion cache.
    pub params: (u32, u32),
}

/// Per-frame staged custom meshes as a separate resource (custom-geometry
/// half of X2, Extract-free).
///
/// `RenderSubmit` rebuilds this every frame from the `custom_meshes` lane
/// of [`extract_render_data`](crate::extraction::extract_render_data)
/// (see [`Renderer3D::stage_custom_meshes`](crate::renderer::Renderer3D::stage_custom_meshes)):
/// per-entity GPU meshes plus joint palettes, in dense lane order.
/// `RenderPresent` draws it after the shared sphere batch (custom instances
/// were uploaded contiguously above the sphere slots, so slots stay dense).
/// Stored as `Mutex<Vec<…>>` — interior mutability, as in `GpuFrameState`.
/// Rebuilt per frame (correctness fallback — see the caching note on
/// `stage_custom_meshes`); a repeat install replaces the resource (never
/// called in steady state).
#[derive(Default)]
pub struct GpuCustomMeshes(pub Mutex<Vec<StagedCustomMesh>>);

/// Registers [`GpuMesh`] in the world plus its rebuild system (X2).
///
/// `RenderMesh` reads the lanes (`reads_lane`, S5d canon) and writes only
/// `Mutex<GpuMesh>`; `create_sphere` needs `GpuDevice` in the resources —
/// insert it before the first `run_frame` (`install_gpu_resources` does
/// that itself). A repeat call replaces the resource (never called in
/// steady state).
pub fn install_render_mesh(engine: &mut ornis_core::Engine, mesh: GpuMesh) {
    let _ = engine.world_mut().insert(Mutex::new(mesh));
    engine.schedule_mut().add_system(RenderMesh);
}

/// Registers the GPU resources in `engine`.
///
/// Call after creating `Device`/`Queue`/`Surface`/`Renderer3D`/
/// `RenderFrame3D`/`Mesh` in the shell initialize — before the first `run_frame`.
/// After that `RenderMesh`/`RenderSubmit`/`RenderPresent`/`RenderFlush`
/// in the `schedule` observe the same objects without copying.
///
/// No light rig is published here: worlds start dark, lights arrive
/// explicitly (scene descriptions, [`GameWorld::add_directional_light`](ornis_app::GameWorld::add_directional_light)).
/// Viewport lighting for lightless scenes is the editor's job (Blender-style
/// shading modes), not a silent engine default.
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
    let _ = engine.world_mut().insert(GpuCustomMeshes::default());
    let _ = engine.world_mut().insert(FramePresentTarget::default());
    install_render_mesh(engine, mesh);
    engine.schedule_mut().add_system(RenderSubmit);
    engine.schedule_mut().add_system(RenderPresent);
    engine.schedule_mut().add_system(RenderFlush);
}

/// Mesh rebuild system (X2): tessellation comes from the lane, not a snapshot.
///
/// `max_mesh_params` is the same canon as `extract_render_data`
/// (full entities, floor (32, 24)). Writes only `Mutex<GpuMesh>`;
/// readers (`RenderPresent`) order after it via RaW, with no resources
/// shared with `RenderSubmit`. Lock order: `GpuMesh` before `GpuFrameState`
/// (otherwise — only here, one lock per system).
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

/// Frame submit system: reads lanes + camera + lights, writes GPU state.
///
/// S7 step 1: runs `set_camera`/`upload_*`. X1 (Extract-free): material and
/// instance data come from direct `TransformDesc`/`MeshDesc`/`MaterialDesc`
/// lane reads (S5d canon) through `extract_render_data`.
/// X2: mesh rebuild moved to `RenderMesh` (the `GpuMesh` resource).
/// X3: lights come from the `RenderLights` resource (scene loader), not
/// hardcoded. Custom geometry (`custom_meshes`, incl. skinned) is staged
/// into the `GpuCustomMeshes` resource
/// ([`Renderer3D::stage_custom_meshes`](crate::renderer::Renderer3D::stage_custom_meshes))
/// with its instances uploaded contiguously above the sphere slots.
/// `frame3d.render_to_buffers` lives in `RenderPresent` (S7 step 2).
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
            .reads_lane::<SkinnedMesh>()
            .reads_lane::<Skeleton>()
            .reads_lane::<JointPose>()
            .reads::<Mutex<crate::camera::OrbitCamera>>()
            .reads::<RenderLights>()
            .writes::<Mutex<GpuFrameState>>()
            .writes::<GpuCustomMeshes>()
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
            .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()).clone())
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
        let Some(custom_meshes) = resources.get::<GpuCustomMeshes>() else {
            return;
        };
        let fs = frame_state.lock().unwrap_or_else(|e| e.into_inner());

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
        // Custom geometry (loaded `.glb` / sculpted soups, incl. skinned):
        // stage per-entity meshes + palettes, then upload sphere and custom
        // instances contiguously — spheres occupy `0..n`, customs follow in
        // dense lane order, so `RenderPresent` draws slot ranges.
        let staged = fs
            .renderer
            .stage_custom_meshes(&device.0, &queue.0, &extracted.custom_meshes);
        let mut instances = Vec::with_capacity(extracted.instances.len() + staged.len());
        instances.extend_from_slice(&extracted.instances);
        instances.extend(staged.iter().map(|entry| entry.instance));
        fs.renderer
            .upload_instances(&device.0, &queue.0, &instances);
        *custom_meshes
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = staged;
    }
}

/// Frame present system: acquire → record (E2: encoder as a frame resource).
///
/// S7 step 2: moves the `Surface` acquire from `GameApp::render_frame`
/// into `Engine::schedule`. E2 (S5e): recording goes through per-pass
/// encoders into `FrameCommandBuffers` (`RenderFrame3D::render_to_buffers`);
/// the acquired frame travels in `FramePresentTarget`; a separate
/// `RenderFlush` system performs submit + present (registered right after —
/// WaW over both handover resources). X2: the mesh is read from the
/// `GpuMesh` resource (RaW after `RenderMesh`). Custom geometry staged by
/// `RenderSubmit` is read from `GpuCustomMeshes` (RaW after `RenderSubmit`)
/// and drawn after the sphere batch through the `*_with_custom` passes.
/// X4: instance count is a
/// direct lane read (`extract_render_data`), not a snapshot. The dependency
/// on `RenderSubmit` derives as WaW over `Mutex<GpuFrameState>`; ordering
/// comes from registering after `RenderSubmit`.
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
            .reads_lane::<SkinnedMesh>()
            .reads_lane::<Skeleton>()
            .reads_lane::<JointPose>()
            .reads::<Mutex<GpuMesh>>()
            .reads::<GpuCustomMeshes>()
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
        let Some(custom_meshes) = resources.get::<GpuCustomMeshes>() else {
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
            let guard = surface.0.lock().unwrap_or_else(|e| e.into_inner());
            match guard.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(frame)
                | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => Some(frame),
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    drop(guard);
                    // Reconfigure with the last known size/format and default
                    // present parameters (matches GameApp::initialize).
                    let guard = surface.0.lock().unwrap_or_else(|e| e.into_inner());
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
            let customs_state = custom_meshes
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Custom instances were uploaded contiguously above the sphere
            // slots, so each item draws its single-instance range.
            let customs: Vec<CustomGbufferDraw<'_>> =
                custom_draw_items(&customs_state, instance_count);
            let mut fs = frame_state.lock().unwrap_or_else(|e| e.into_inner());
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
                    customs: &customs,
                };
                let _ = (*frame3d).render_to_buffers(context);
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

/// Frame drain system: ordered submit + present (E2).
///
/// A separate system after `RenderPresent`: drains `FrameCommandBuffers`
/// in registration order (single submit) and presents the acquired frame
/// from `FramePresentTarget`. Ordering is guaranteed by WaW over both
/// handover resources when registered right after `RenderPresent`.
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
    use ornis_assets::scene::ShadowCast;
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
        assert_send_sync::<GpuCustomMeshes>();

        let mut engine = Engine::new();
        install_frame_buffers(&mut engine);
        let buffers = engine
            .world()
            .resources()
            .get::<FrameCommandBuffers>()
            .expect("frame buffers resource");
        assert!(
            buffers
                .0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty(),
            "fresh install holds no pending buffers"
        );
    }

    #[test]
    fn install_keeps_scene_published_lights() {
        // `replace_scene` runs before `install_gpu_resources` in
        // `GameApp::initialize`: a published scene rig must survive the
        // GPU install, while a sceneless runtime still gets the legacy
        // default (X3 zero-diff gate).
        let mut engine = Engine::new();
        let custom = RenderLights {
            ambient: [0.5, 0.4, 0.3],
            lights: vec![ornis_assets::scene::LightDesc::Directional {
                direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(0.0, -1.0, 0.0))
                    .expect("non-zero direction"),
                intensity: 2.0,
                color: [1.0, 0.9, 0.8],
                shadow: ShadowCast::Disabled,
            }],
            ambient_intensity: 1.0,
            exposure: 1.0,
        };
        let _ = engine.world_mut().insert(custom);
        install_frame_buffers(&mut engine);
        let kept = engine
            .world()
            .resources()
            .get::<RenderLights>()
            .expect("lights resource");
        assert_eq!(kept.ambient, [0.5, 0.4, 0.3]);
        assert_eq!(kept.lights.len(), 1);

        let mut fresh = Engine::new();
        install_frame_buffers(&mut fresh);
        assert!(
            fresh.world().resources().get::<RenderLights>().is_none(),
            "no scene, no lights: the world starts dark"
        );
    }

    #[test]
    fn render_system_accesses_pin_the_dag_contract() {
        use std::any::TypeId;

        // E1–E3/X1–X4 pin: the whole native frame orders through these
        // declarations — no hidden edges. All lane readers declare the
        // full S5d canon; RenderMesh shares nothing with RenderSubmit;
        // RenderPresent follows via RaW on `Mutex<GpuMesh>` plus WaW on
        // `Mutex<GpuFrameState>`; RenderFlush closes via WaW on both
        // handover resources.
        for access in [
            RenderMesh.access(),
            RenderSubmit.access(),
            RenderPresent.access(),
        ] {
            for lane in [
                TypeId::of::<TransformDesc>(),
                TypeId::of::<MeshDesc>(),
                TypeId::of::<MaterialDesc>(),
            ] {
                assert!(
                    access.reads_lanes.contains(&lane),
                    "render systems read the lane canon"
                );
            }
        }

        // RenderMesh writes only the mesh slot (X2).
        let mesh = RenderMesh.access();
        assert_eq!(mesh.writes, vec![TypeId::of::<Mutex<GpuMesh>>()]);
        assert!(mesh.reads.contains(&TypeId::of::<GpuDevice>()));

        // RenderSubmit writes the frame slot plus the staged custom meshes
        // (X1/X3: lanes and lights in, no snapshot).
        let submit = RenderSubmit.access();
        assert_eq!(
            submit.writes,
            vec![
                TypeId::of::<Mutex<GpuFrameState>>(),
                TypeId::of::<GpuCustomMeshes>()
            ]
        );
        assert!(submit.reads.contains(&TypeId::of::<RenderLights>()));

        // RenderPresent bridges into both handover resources (E2) and reads
        // the staged custom meshes (RaW after RenderSubmit).
        let present = RenderPresent.access();
        assert!(present.reads.contains(&TypeId::of::<Mutex<GpuMesh>>()));
        assert!(present.reads.contains(&TypeId::of::<GpuCustomMeshes>()));
        assert!(
            present
                .writes
                .contains(&TypeId::of::<Mutex<GpuFrameState>>())
        );
        assert!(
            present
                .writes
                .contains(&TypeId::of::<FrameCommandBuffers>())
        );
        assert!(present.writes.contains(&TypeId::of::<FramePresentTarget>()));

        // RenderFlush drains exactly the handover pair (E2).
        let flush = RenderFlush.access();
        assert_eq!(flush.reads, vec![TypeId::of::<GpuQueue>()]);
        assert!(flush.writes.contains(&TypeId::of::<FrameCommandBuffers>()));
        assert!(flush.writes.contains(&TypeId::of::<FramePresentTarget>()));
    }

    #[test]
    fn render_systems_level_as_mesh_submit_then_present_then_flush() {
        // Registration order is the execution contract
        // (`install_gpu_resources` registers in this exact order): mesh
        // and submit share level 0, present follows (RaW on the mesh
        // slot, WaW on the frame slot), flush closes (WaW on both
        // handovers). Needs no GPU — levels come from declarations only.
        let mut engine = Engine::new();
        engine.schedule_mut().add_system(RenderMesh);
        engine.schedule_mut().add_system(RenderSubmit);
        engine.schedule_mut().add_system(RenderPresent);
        engine.schedule_mut().add_system(RenderFlush);
        assert_eq!(
            engine.schedule().levels(),
            vec![vec![0, 1], vec![2], vec![3]]
        );
        let mermaid = engine.schedule().mermaid();
        for name in [
            "render_mesh",
            "render_submit",
            "render_present",
            "render_flush",
        ] {
            assert!(mermaid.contains(name), "schedule mentions {name}");
        }
    }
}
