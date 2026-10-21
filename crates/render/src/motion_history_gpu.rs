//! Temporal-history gate for the indirect path (review #100): the
//! records staged for the GPU culler must carry the previous frame's
//! transform and the per-instance motion-validity flag, so the g-buffer
//! motion vectors stay meaningful. The gate drives the real submit
//! system over two frames with a moved entity and inspects both the
//! staged records and the compacted GPU batch.
#![cfg(test)]
use super::{read_lock, test_util};
use crate::gpu_culling::{Culler, INDIRECT_ARGS_WORD_COUNT};
use crate::gpu_resources::{
    GpuCustomMeshes, GpuDevice, GpuFrameState, GpuQueue, GpuSurfaceState, RenderSubmit,
};
use crate::renderer::{PerObjectGpu, Renderer3D};
use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
use ornis_core::units::{Clamped01, PositiveF32, UnitQuat};
use ornis_core::{Entity, SmartStore, System};

/// Frame size of the headless history gate (the camera aspect follows).
const GATE_SIZE: (u32, u32) = (320, 180);
/// World position of the sphere that moves between the two frames.
const MOVED_FROM: [f32; 3] = [0.0, 0.0, 0.0];
/// World position the moving sphere reaches in the second frame.
const MOVED_TO: [f32; 3] = [0.6, 0.0, 0.2];
/// World position of the sphere that never moves.
const STATIC_AT: [f32; 3] = [2.5, 0.5, -1.0];
/// Tolerance of the world-position comparison (float round-trip).
const POSITION_EPSILON: f32 = 1e-5;
/// Index count of the indirect submit (the culler only rewrites the
/// instance count).
const INDEX_COUNT: u32 = 42;

/// The records of the two-frame scene: the batch staged for the culler
/// in each frame and the compacted GPU batch of the second one.
struct HistoryProbe {
    /// Records staged in the first frame (the entity is seen for the
    /// first time — no history yet).
    staged_first: Vec<PerObjectGpu>,
    /// Records staged in the second frame (the sphere moved).
    staged_second: Vec<PerObjectGpu>,
    /// Records of the compacted GPU batch of the second frame.
    compacted: Vec<PerObjectGpu>,
}

/// The world transform of a unit sphere at `position`.
fn transform_of(position: [f32; 3]) -> TransformDesc {
    TransformDesc {
        translation: glam::Vec3::from(position),
        rotation: UnitQuat::IDENTITY,
        scale: glam::Vec3::splat(1.0),
    }
}

/// Inserts a unit sphere with a material at `position`; returns the
/// entity (the history gate moves it between the two frames).
fn insert_sphere(store: &mut SmartStore, position: [f32; 3]) -> Entity {
    let entity = store.create_entity();
    store.insert(entity, transform_of(position));
    store.insert(
        entity,
        MeshDesc::Sphere {
            radius: PositiveF32::expect_valid(1.0),
            segments: 16,
            rings: 12,
        },
    );
    store.insert(
        entity,
        MaterialDesc::Dielectric {
            base_color: [0.8, 0.8, 0.8],
            roughness: Clamped01::new(0.5),
            emission: [0.0, 0.0, 0.0],
            metallic: ornis_core::Metallic::new(0.0),
        },
    );
    entity
}

/// The engine with the two-sphere scene and an armed indirect renderer;
/// returns it together with the entity that moves between the frames.
fn history_engine(device: &wgpu::Device, queue: &wgpu::Queue) -> (ornis_core::Engine, Entity) {
    let mut engine = ornis_core::Engine::new();
    let moved = {
        let store = engine.world_mut().store_mut().expect("scene store");
        let moved = insert_sphere(store, MOVED_FROM);
        insert_sphere(store, STATIC_AT);
        moved
    };
    let surface = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        width: GATE_SIZE.0,
        height: GATE_SIZE.1,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    let renderer = Renderer3D::new(device, &surface, 1);
    renderer.set_indirect_draw(true);
    let world = engine.world_mut();
    world.insert(std::sync::Mutex::new(
        crate::camera::OrbitCamera::looking_at(
            glam::Vec3::new(0.0, 2.5, 9.0),
            glam::Vec3::new(0.5, 0.0, 0.0),
        ),
    ));
    world.insert(crate::extraction::RenderLights::default());
    world.insert(GpuDevice(device.clone()));
    world.insert(GpuQueue(queue.clone()));
    world.insert(GpuSurfaceState {
        size: GATE_SIZE,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
    });
    world.insert(std::sync::Mutex::new(GpuFrameState {
        renderer,
        frame3d: crate::frame_exec::RenderFrame3D::new(
            wgpu::TextureFormat::Rgba8UnormSrgb,
            GATE_SIZE,
        ),
    }));
    world.insert(GpuCustomMeshes::default());
    (engine, moved)
}

/// Moves an entity to `position` (the transform lane is replaced, so the
/// next frame extracts a different world transform).
fn move_entity(engine: &mut ornis_core::Engine, entity: Entity, position: [f32; 3]) {
    let store = engine.world_mut().store_mut().expect("scene store");
    store.insert(entity, transform_of(position));
}

/// Runs the real submit system and returns the records it staged for
/// the indirect culler.
fn submit_and_stage(engine: &ornis_core::Engine) -> Vec<PerObjectGpu> {
    RenderSubmit.run(engine.world().resources());
    let frame = engine
        .world()
        .resources()
        .get::<std::sync::Mutex<GpuFrameState>>()
        .expect("frame state resource");
    let frame = frame
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let staged = read_lock(&frame.renderer.staged_batch);
    staged.as_ref().expect("batch staged while armed").1.clone()
}

/// Prepares the staged batch, dispatches the compaction and returns the
/// compacted records of the visible instances.
fn compact_indirect_batch(
    engine: &ornis_core::Engine,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Vec<PerObjectGpu> {
    let frame = engine
        .world()
        .resources()
        .get::<std::sync::Mutex<GpuFrameState>>()
        .expect("frame state resource");
    let frame = frame
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    frame
        .renderer
        .prepare_indirect_batch(device, queue, INDEX_COUNT);
    assert!(
        frame.renderer.indirect_batch_ready(),
        "the indirect batch must be armed after the prepare"
    );
    let culler = read_lock(&frame.renderer.culler);
    let culler = culler.as_ref().expect("culler built");
    test_util::dispatch_culler(device, queue, culler);
    let visible = args_words(device, queue, culler)[1] as usize;
    compacted_records(device, queue, culler, visible)
}

/// The indirect-args words of the culler (index count, instance count,
/// first index, base vertex, first instance).
fn args_words(device: &wgpu::Device, queue: &wgpu::Queue, culler: &Culler) -> Vec<u32> {
    let raw = crate::read_bytes_back(
        device,
        queue,
        &culler.args_buffer,
        INDIRECT_ARGS_WORD_COUNT as u64 * 4,
    )
    .expect("indirect args readback");
    raw.chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte word")))
        .collect()
}

/// The compacted records of the culler's dense destination buffer.
fn compacted_records(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    culler: &Culler,
    visible: usize,
) -> Vec<PerObjectGpu> {
    let bytes = (visible * std::mem::size_of::<PerObjectGpu>()) as u64;
    let raw =
        crate::read_bytes_back(device, queue, &culler.visible_buffer, bytes).expect("readback");
    bytemuck::cast_slice(&raw).to_vec()
}

/// The record of the instance standing at `position`.
fn record_at(records: &[PerObjectGpu], position: [f32; 3]) -> PerObjectGpu {
    records
        .iter()
        .find(|record| {
            let translation = [record.model[3][0], record.model[3][1], record.model[3][2]];
            (translation[0] - position[0]).abs() < POSITION_EPSILON
                && (translation[1] - position[1]).abs() < POSITION_EPSILON
                && (translation[2] - position[2]).abs() < POSITION_EPSILON
        })
        .copied()
        .expect("the instance is staged")
}

/// Drives the two-frame scene: the first frame sees the entity for the
/// first time, the second one finds it moved, and the second frame is
/// compacted on the GPU.
fn run_history_probe(device: &wgpu::Device, queue: &wgpu::Queue) -> HistoryProbe {
    let (mut engine, moved) = history_engine(device, queue);
    let staged_first = submit_and_stage(&engine);
    move_entity(&mut engine, moved, MOVED_TO);
    let staged_second = submit_and_stage(&engine);
    let compacted = compact_indirect_batch(&engine, device, queue);
    HistoryProbe {
        staged_first,
        staged_second,
        compacted,
    }
}

/// Review #100: the records the submit system stages for the indirect
/// culler must carry the previous frame's transform and the motion
/// validity of the entity history. The first frame has no history (the
/// motion is invalid), the second one keeps the transform of the first
/// frame (the motion is valid). Fails if the staging path drops the
/// history (`previous_model = model`, a constant `motion_valid`) or if
/// the records are rebuilt by
/// [`convert_instances`](crate::gpu_culling::convert_instances), which
/// knows nothing about the entity history.
#[test]
fn indirect_staging_keeps_the_temporal_history() {
    let Some((device, queue)) = test_util::try_device() else {
        eprintln!("SKIPPED (indirect_staging_keeps_the_temporal_history): no wgpu adapter");
        return;
    };
    let probe = run_history_probe(&device, &queue);
    let moved_first = record_at(&probe.staged_first, MOVED_FROM);
    assert_eq!(
        moved_first.motion_valid, 0,
        "the first frame has no motion history"
    );
    assert_eq!(
        moved_first.previous_model, moved_first.model,
        "without history the previous transform falls back to the current one"
    );
    let moved_second = record_at(&probe.staged_second, MOVED_TO);
    assert_ne!(
        moved_second.model, moved_first.model,
        "the entity really moved between the frames"
    );
    assert_eq!(
        moved_second.previous_model, moved_first.model,
        "the staged record must keep the previous frame's transform"
    );
    assert_eq!(
        moved_second.motion_valid, 1,
        "a known entity has a valid motion history"
    );
    // The static sphere keeps its own transform as the previous one.
    let static_second = record_at(&probe.staged_second, STATIC_AT);
    assert_eq!(
        static_second.previous_model, static_second.model,
        "a sphere that did not move keeps its transform"
    );
    // The compacted GPU batch carries the same history: the compute
    // pass copies the whole record.
    let moved_compacted = record_at(&probe.compacted, MOVED_TO);
    assert_eq!(
        moved_compacted.previous_model, moved_first.model,
        "the compacted batch must keep the previous frame's transform"
    );
    assert_eq!(
        moved_compacted.motion_valid, 1,
        "the compacted batch must keep the motion validity"
    );
}
