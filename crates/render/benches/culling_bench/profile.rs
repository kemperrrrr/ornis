//! RAT-48 culling stage profile: three scenes (stands, moves 1%, moves
//! all) x three sizes (10k, 100k, 1M) x the two staging paths of the
//! issue — `compact`, and `compact_records` fed by the renderer — as one
//! markdown row per cell.
//!
//! Each frame splits into the issue's stages. CPU: build (record
//! conversion — the renderer's on the records path), `upload_per_objects`
//! (the renderer's full-frame write, records path only), diff against the
//! previous frame (records path: incl. the staging hand-off copy), the
//! dirty-run `write_buffer` fill, and the uniform/args pre-write. GPU: the
//! culling compute pass and the indirect draw, read back through
//! `wgpu::Features::TIMESTAMP_QUERY` pass timestamps — cells print "—"
//! when the adapter lacks the feature or the readback fails. The stages
//! are the production hooks themselves, called in the real frame's
//! execution order; this bench measures, it does not optimize.
//!
//! Output: the table on stderr (copied into
//! `crates/render/docs/culling-profile.md`), plus one criterion bench per
//! cell so the wall time carries the harness statistics too.
use std::time::{Duration, Instant};

use criterion::{BenchmarkGroup, Criterion, measurement::WallTime};
use glam::{Mat4, Vec3};
use ornis_core::Entity;
use ornis_render::{Culler, InstanceData, Renderer3D, gpu_culling::profile};

use super::probe::{PROBE_HEIGHT, PROBE_WIDTH, Probe};
use super::{INDEX_COUNT, SIZE_10K, SIZE_100K, camera_view_proj, scene_line};
use model::{
    BenchPath, DRIFT_STEP, Frame, MOVE_STRIDE, PROFILE_FRAMES, PROFILE_SAMPLE_SIZE, Scenario,
    Setup, Stages,
};

mod model;

/// Instances in the largest profiled scene (one culler covers them all).
const SIZE_1M: usize = 1_000_000;

/// Move this frame's movers on X: nothing when the scene stands, every
/// `MOVE_STRIDE`-th instance at 1%, all of them otherwise. The sign
/// alternates with the step, so the scene drifts in place instead of
/// walking out of the camera frame.
fn apply_motion(instances: &mut [InstanceData], scenario: Scenario, step: usize) {
    let stride = match scenario {
        Scenario::Static => return,
        Scenario::MoveOnePercent => MOVE_STRIDE,
        Scenario::MoveAll => 1,
    };
    let drift = if step.is_multiple_of(2) {
        DRIFT_STEP
    } else {
        -DRIFT_STEP
    };
    let shift = Mat4::from_translation(Vec3 {
        x: drift,
        y: 0.0,
        z: 0.0,
    });
    for chunk in instances.chunks_mut(stride) {
        if let Some(instance) = chunk.first_mut() {
            instance.model_matrix = shift * instance.model_matrix;
        }
    }
}

// qual:allow(test_quality, untested) reason: "bench-only frame pipeline, driven by the criterion harness, never under libtest"
/// One profiled frame: motion, then the issue's CPU stages in the real
/// frame's execution order (build, upload, diff, fill, pre-write), then
/// both GPU passes in one encoder — submit, blocking poll, stamp
/// readback. Every stage times its own production hook; `wall` covers
/// the whole frame including that readback.
fn profile_frame(frame: &mut Frame) -> Stages {
    let mut stages = Stages::default();
    let wall = Instant::now();
    apply_motion(&mut frame.instances, frame.scenario, frame.step);
    frame.step += 1;

    let clock = Instant::now();
    match frame.path {
        BenchPath::Compact => {
            frame.next.clear();
            profile::convert(&mut frame.next, &frame.instances);
        }
        BenchPath::Records => {
            frame.records = frame.setup.renderer.profile_stage_records(
                frame.setup.device,
                &frame.instances,
                &[],
                &frame.entities,
            );
        }
    }
    stages.build = clock.elapsed();

    if frame.path == BenchPath::Records {
        let clock = Instant::now();
        frame
            .setup
            .renderer
            .profile_write_records(frame.setup.queue, &frame.records);
        stages.upload = clock.elapsed();
        stages.upload_measured = true;
    }

    let clock = Instant::now();
    if frame.path == BenchPath::Records {
        frame.next.clear();
        frame.next.extend_from_slice(&frame.records);
    }
    let ranges = profile::diff(&frame.prev, &frame.next);
    stages.diff = clock.elapsed();
    let staged = frame.next.len();

    let clock = Instant::now();
    frame
        .setup
        .culler
        .profile_write_dirty(frame.setup.queue, &frame.next, &ranges);
    stages.fill = clock.elapsed();
    std::mem::swap(&mut frame.prev, &mut frame.next);

    let clock = Instant::now();
    frame.setup.culler.profile_prewrite(
        frame.setup.queue,
        &frame.setup.view_proj,
        INDEX_COUNT,
        staged as u32,
    );
    stages.prewrite = clock.elapsed();

    let (compute, draw) =
        frame
            .setup
            .probe
            .timed_passes(frame.setup.device, frame.setup.queue, frame.setup.culler);
    stages.compute = compute.unwrap_or(Duration::ZERO);
    stages.draw = draw.unwrap_or(Duration::ZERO);
    stages.gpu_measured = compute.is_some() && draw.is_some();
    stages.wall = wall.elapsed();
    stages
}

// qual:allow(test_quality, untested) reason: "bench-only cell runner, driven by the criterion harness, never under libtest"
/// One (scenario, path) cell: a discarded warm-up frame, then
/// `PROFILE_FRAMES` frames averaged into the returned row, then one
/// criterion bench (a full frame per iteration) for the cell's wall
/// time under the harness's own statistics.
fn profile_cell(group: &mut BenchmarkGroup<WallTime>, frame: &mut Frame) -> Stages {
    let _warm = profile_frame(frame);
    let mut row = Stages::default();
    for _ in 0..PROFILE_FRAMES {
        row.accumulate(&profile_frame(frame));
    }
    let label = format!(
        "{}_{}_{}",
        frame.scenario.label(),
        frame.instances.len(),
        frame.path.label(),
    );
    group.bench_function(label, |b| {
        b.iter(|| std::hint::black_box(profile_frame(frame)))
    });
    row.averaged(PROFILE_FRAMES)
}

// qual:allow(test_quality, untested) reason: "bench-only size loop, driven by the criterion harness, never under libtest"
/// One size through the profile: every scenario x path cell, its row
/// printed in loop order under the caller's header.
fn profile_size(
    group: &mut BenchmarkGroup<WallTime>,
    setup: &Setup,
    base: &[InstanceData],
    entities: &[Entity],
) {
    for scenario in [
        Scenario::Static,
        Scenario::MoveOnePercent,
        Scenario::MoveAll,
    ] {
        for path in [BenchPath::Compact, BenchPath::Records] {
            let mut frame = Frame::new(setup, base, entities, scenario, path);
            let row = profile_cell(group, &mut frame);
            eprintln!("{}", row.row(scenario.label(), base.len(), path.label()));
        }
    }
}

// qual:allow(iosp) reason: "criterion entry by design — dispatch into shared setup and the size loop, same dispatch-only premise as bench_culling; measured by the harness, not a layer crossing"
// qual:allow(test_quality, untested) reason: "bench-only criterion entry, driven by the criterion harness, never under libtest"
/// The RAT-48 profile entry: one shared GPU setup, then every
/// size x scenario x path cell with its markdown row. A host without an
/// adapter (or without 1M culler capacity) prints a skip note instead of
/// failing the bench, like the GPU groups of the bench above.
pub(super) fn bench_culling_profile(c: &mut Criterion) {
    let (device, queue) = match ornis_render::try_profile_headless_device() {
        Some(pair) => pair,
        None => {
            eprintln!("SKIP culling_profile: no wgpu adapter (run on a GPU machine)");
            return;
        }
    };
    let culler = match Culler::try_new(&device, SIZE_1M as u32) {
        Ok(culler) => culler,
        Err(err) => {
            eprintln!("SKIP culling_profile: 1M-capacity culler unavailable ({err})");
            return;
        }
    };
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: PROBE_WIDTH,
        height: PROBE_HEIGHT,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 1,
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    let renderer = Renderer3D::new(&device, &surface_config, 1);
    let probe = Probe::new(&device, &queue);
    let setup = Setup {
        device: &device,
        queue: &queue,
        renderer: &renderer,
        culler: &culler,
        probe: &probe,
        view_proj: camera_view_proj(),
    };

    let mut group = c.benchmark_group("culling_profile");
    group.sample_size(PROFILE_SAMPLE_SIZE);
    eprintln!("culling profile — ms per frame, mean of {PROFILE_FRAMES} frames (— = not measured)");
    eprintln!(
        "| scenario | size | path | build | diff | fill | upload_per_objects | prewrite | compute | draw | wall |"
    );
    eprintln!("|---|---|---|---|---|---|---|---|---|---|---|");
    for size in [SIZE_10K, SIZE_100K, SIZE_1M] {
        let base = scene_line(size);
        let entities: Vec<Entity> = (0..size).map(|i| Entity::new(i as u32)).collect();
        profile_size(&mut group, &setup, &base, &entities);
    }
    group.finish();
}
