//! RAT-48/RAT-52 culling stage profile: three scenes (stands, moves 1%,
//! moves all) x three sizes (10k, 100k, 1M) x the three staging paths of
//! the issue — `compact`, `compact_records` (the pre-#185 reference) and
//! `compact_changed` (the incremental record build fed by the ECS change
//! feed) — as one markdown row per cell.
//!
//! Each frame splits into the issue's stages. CPU: build (record
//! conversion — the renderer's on the record paths; on `compact_changed`
//! only the changed entities are rebuilt), `upload_per_objects` (the
//! renderer's per-object write), diff against the previous frame (the two
//! full-frame paths; incl. the staging hand-off copy on
//! `compact_records`), the dirty-run `write_buffer` fill, and the
//! uniform/args pre-write. GPU: the
//! culling compute pass and the indirect draw, read back through
//! `wgpu::Features::TIMESTAMP_QUERY` pass timestamps — cells print "—"
//! when the adapter lacks the feature or the readback fails. The stages
//! are the production hooks themselves, called in the real frame's
//! execution order; this bench measures, it does not optimize.
//!
//! Output: the table on stderr (copied into
//! `crates/render/docs/culling-profile.md`), plus one criterion bench per
//! cell so the wall time carries the harness statistics too.
use std::time::Instant;

use criterion::{BenchmarkGroup, Criterion, measurement::WallTime};
use glam::{Mat4, Vec3};
use ornis_core::Entity;
use ornis_render::{Culler, InstanceData, PerObjectGpu};

use super::probe::{PROBE_HEIGHT, PROBE_WIDTH, Probe};
use super::{INDEX_COUNT, SIZE_10K, SIZE_100K, camera_view_proj, scene_line};
use model::{
    BenchPath, DRIFT_STEP, Frame, MOVE_STRIDE, PROFILE_FRAMES, PROFILE_SAMPLE_SIZE,
    PROFILE_WARM_FRAMES, Scenario, Setup, Stages,
};

mod model;
mod stages;

use stages::{build_records, fill_culling, prewrite};

/// Instances in the largest profiled scene (one culler covers them all).
const SIZE_1M: usize = 1_000_000;

/// Move this frame's movers on X: nothing when the scene stands, every
/// `MOVE_STRIDE`-th instance at 1%, all of them otherwise. The sign
/// alternates with the step, so the scene drifts in place instead of
/// walking out of the camera frame.
///
/// Returns the mover indices — exactly the change feed an ECS transform
/// lane would report, which the changed path consumes instead of
/// byte-diffing the whole frame.
fn apply_motion(instances: &mut [InstanceData], scenario: Scenario, step: usize) -> Vec<u32> {
    let stride = match scenario {
        Scenario::Static => return Vec::new(),
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
    let mut moved = Vec::with_capacity(instances.len().div_ceil(stride));
    for (index, instance) in instances.iter_mut().enumerate() {
        if index.is_multiple_of(stride) {
            instance.model_matrix = shift * instance.model_matrix;
            moved.push(index as u32);
        }
    }
    moved
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
    let first_frame = frame.step == 0;
    let moved = apply_motion(&mut frame.instances, frame.scenario, frame.step);
    frame.step += 1;

    stages.build = build_records(frame, &moved, first_frame);
    if frame.path != BenchPath::Compact {
        let clock = Instant::now();
        frame
            .renderer
            .profile_write_records(frame.setup.queue, &frame.records);
        stages.upload = clock.elapsed();
        stages.upload_measured = true;
    }

    let (diff, staged) = fill_culling(frame);
    if frame.path != BenchPath::Changed {
        stages.diff = diff;
        stages.diff_measured = true;
    }
    stages.fill = diff;
    stages.prewrite = prewrite(frame, staged);
    let (compute, draw) =
        frame
            .setup
            .probe
            .timed_passes(frame.setup.device, frame.setup.queue, frame.setup.culler);
    if let (Some(compute), Some(draw)) = (compute, draw) {
        stages.compute = compute;
        stages.draw = draw;
        stages.gpu_measured = true;
    }
    stages.wall = wall.elapsed();
    stages
}

// qual:allow(test_quality, untested) reason: "bench-only cell runner, driven by the criterion harness, never under libtest"
/// One (scenario, path) cell: `PROFILE_WARM_FRAMES` discarded warm-up
/// frames, then `PROFILE_FRAMES` frames averaged into the returned row,
/// then one criterion bench (a full frame per iteration) for the cell's
/// wall time under the harness's own statistics.
fn profile_cell(group: &mut BenchmarkGroup<WallTime>, frame: &mut Frame) -> Stages {
    for _ in 0..PROFILE_WARM_FRAMES {
        let _ = profile_frame(frame);
    }
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
        for path in [BenchPath::Compact, BenchPath::Records, BenchPath::Changed] {
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
/// adapter prints a skip note instead of failing the bench, like the GPU
/// groups of the bench above; a host whose
/// `max_storage_buffer_binding_size` cannot bind a scene's records
/// drops that size (the drop is printed) instead of tripping a wgpu
/// validation error — see the size filter below.
pub(super) fn bench_culling_profile(c: &mut Criterion) {
    let (device, queue) = match ornis_render::try_profile_headless_device() {
        Some(pair) => pair,
        None => {
            eprintln!("SKIP culling_profile: no wgpu adapter (run on a GPU machine)");
            return;
        }
    };
    // The binding limit governs both GPU objects of a cell: the culler
    // binds `size * size_of::<PerObjectGpu>()` exactly, and the
    // renderer's per-object buffer grows by doubling from 256, i.e. to
    // `size.next_power_of_two()`. A size whose binding would exceed the
    // device limit is dropped with a printed note rather than failing
    // the bench at `create_bind_group` (e.g. a 128 MiB GL adapter
    // cannot bind the 1M scene: 208 MB > 134 MB).
    let binding_limit = device.limits().max_storage_buffer_binding_size;
    let record_bytes = std::mem::size_of::<PerObjectGpu>() as u64;
    let fits = |size: usize| (size.next_power_of_two() as u64) * record_bytes <= binding_limit;
    let all_sizes = [SIZE_10K, SIZE_100K, SIZE_1M];
    let sizes: Vec<usize> = all_sizes.into_iter().filter(|&size| fits(size)).collect();
    if sizes.is_empty() {
        eprintln!(
            "SKIP culling_profile: no scene fits max_storage_buffer_binding_size \
             {binding_limit} B ({record_bytes} B per record)"
        );
        return;
    }
    let dropped: Vec<usize> = all_sizes.into_iter().filter(|&size| !fits(size)).collect();
    if !dropped.is_empty() {
        eprintln!(
            "culling_profile: adapter max_storage_buffer_binding_size {binding_limit} B \
             drops sizes {dropped:?}"
        );
    }
    let largest = *sizes.last().expect("sizes is non-empty");
    let culler = match Culler::try_new(&device, largest as u32) {
        Ok(culler) => culler,
        Err(err) => {
            eprintln!("SKIP culling_profile: {largest}-capacity culler unavailable ({err})");
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
    let probe = Probe::new(&device, &queue);
    let setup = Setup {
        device: &device,
        queue: &queue,
        surface_config: &surface_config,
        culler: &culler,
        probe: &probe,
        view_proj: camera_view_proj(),
    };

    let mut group = c.benchmark_group("culling_profile");
    group.sample_size(PROFILE_SAMPLE_SIZE);
    eprintln!(
        "culling profile — ms per frame, mean of {PROFILE_FRAMES} frames after \
         {PROFILE_WARM_FRAMES} warm-ups (— = not measured)"
    );
    eprintln!(
        "| scenario | size | path | build | diff | fill | upload_per_objects | prewrite | compute | draw | wall |"
    );
    eprintln!("|---|---|---|---|---|---|---|---|---|---|---|");
    for size in sizes {
        let base = scene_line(size);
        let entities: Vec<Entity> = (0..size).map(|i| Entity::new(i as u32)).collect();
        profile_size(&mut group, &setup, &base, &entities);
    }
    group.finish();
}
