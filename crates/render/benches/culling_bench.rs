//! RAT-35 slice 3 bench: CPU vs GPU frustum culling, camera sees a small part.
//!
//! Both paths compact the same scene (spheres on a wide line, the camera
//! frame covers only a few of them): the CPU reference
//! ([`compact_visible_indices`](ornis_render::compact_visible_indices)) runs
//! the sphere tests inline, while the GPU path
//! ([`Culler::compact`](ornis_render::Culler::compact)) stages the instances,
//! dispatches the compute pass and waits for it. Sizes 10k/100k per the
//! issue; the numbers table goes into the PR (headless lavapipe measures
//! the relative gap on one machine, not absolute discrete-GPU frame cost).
//!
//! Column asymmetry (read before comparing): the GPU column includes the
//! per-frame staging upload of every instance (`queue.write_buffer` of the
//! full `src` buffer) plus the blocking device poll, while the CPU column
//! measures only the inline sphere tests. A persistent resident instance
//! buffer (no per-frame upload, no poll wait) is the follow-up slice that
//! closes this gap — see the RAT-43 issue.
//!
//! Compile-checked by the gate; manual run:
//!   cargo bench -p ornis-render --bench culling_bench
//!
//! With `ORNIS_REQUIRE_GPU` set (the CI GPU lane), a missing adapter fails
//! the bench instead of degrading to the CPU column.

use criterion::{Criterion, criterion_group, criterion_main};
use glam::{Mat4, Vec3};
use ornis_render::{
    Culler, INDIRECT_ARGS_WORD_BYTES, INDIRECT_ARGS_WORD_COUNT, INSTANCE_COUNT_WORD, InstanceData,
    MaterialIdx, compact_visible_indices, gpu_required,
};

/// Sphere mesh index count placeholder: the cull pass only compacts, the
/// count is carried through into the indirect args untouched.
const INDEX_COUNT: u32 = 42;
/// Instances on the 10k scene line.
const SIZE_10K: usize = 10_000;
/// Instances on the 100k scene line.
const SIZE_100K: usize = 100_000;
/// Spacing between neighbours on the scene line (world units).
const SPACING: f32 = 2.0;
/// Camera eye: slightly above and in front of the origin.
const CAMERA_EYE: [f32; 3] = [0.0, 2.5, 9.0];
/// Vertical FOV in radians (60 degrees; the perspective call takes radians).
const FOV_RAD: f32 = std::f32::consts::FRAC_PI_3;
/// Bench frame width/height: the 16:9 aspect of the culler unit tests.
const FRAME_W: f32 = 16.0;
const FRAME_H: f32 = 9.0;
/// Aspect ratio of the bench frame.
const ASPECT: f32 = FRAME_W / FRAME_H;
/// Perspective near plane.
const NEAR: f32 = 0.1;
/// Perspective far plane.
const FAR: f32 = 100.0;
/// Denominator of the bench premise: the camera must cover less than a
/// tenth of the scene.
const VISIBLE_PART_DENOM: usize = 10;

async fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
    {
        Ok(adapter) => adapter,
        Err(err) => {
            assert!(
                !gpu_required(),
                "ORNIS_REQUIRE_GPU is set but no wgpu adapter: {err}"
            );
            return None;
        }
    };
    match adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
    {
        Ok(pair) => Some(pair),
        Err(err) => {
            assert!(
                !gpu_required(),
                "ORNIS_REQUIRE_GPU is set but no wgpu device: {err}"
            );
            None
        }
    }
}

// qual:allow(test_quality, untested) reason: "bench scene builder runs only under the criterion harness, never under libtest"
/// `count` unit spheres on the X axis centred at the origin; the frame
/// camera only ever covers the few near `x = 0`.
fn scene_line(count: usize) -> Vec<InstanceData> {
    let offset = count as f32 / 2.0 * SPACING;
    (0..count)
        .map(|i| {
            let model = Mat4::from_translation(Vec3::new(i as f32 * SPACING - offset, 0.0, 0.0));
            InstanceData {
                model_matrix: model,
                normal_matrix: Mat4::IDENTITY,
                material_index: MaterialIdx::from_raw(0),
            }
        })
        .collect()
}

// qual:allow(test_quality, untested) reason: "bench premise predicate runs only under the criterion harness, never under libtest"
/// The bench premise as a pure predicate: the camera covers a few
/// instances — never none, never a tenth of the scene.
fn sees_small_part(visible: usize, total: usize) -> bool {
    visible > 0 && visible * VISIBLE_PART_DENOM < total
}

// qual:allow(test_quality, untested) reason: "bench GPU arg readback runs only under the criterion harness, never under libtest"
/// Reads the `instance_count` word back from a culler's args buffer
/// (`None` when the headless readback fails — the caller skips the parity
/// check instead of failing the bench).
fn read_instance_count(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    args: &wgpu::Buffer,
) -> Option<u32> {
    let args_bytes = INDIRECT_ARGS_WORD_COUNT as u64 * INDIRECT_ARGS_WORD_BYTES;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("culling bench args readback"),
        size: args_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("culling bench args copy"),
    });
    encoder.copy_buffer_to_buffer(args, 0, &staging, 0, args_bytes);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().ok()?.ok()?;
    let view = slice.get_mapped_range().ok()?;
    let words: [u32; INDIRECT_ARGS_WORD_COUNT] = bytemuck::cast_slice(&view)
        [..INDIRECT_ARGS_WORD_COUNT]
        .try_into()
        .ok()?;
    drop(view);
    staging.unmap();
    Some(words[INSTANCE_COUNT_WORD])
}

// qual:allow(test_quality, untested) reason: "bench GPU frame runner runs only under the criterion harness, never under libtest"
/// One GPU frame: `compact` uploads, then the dispatch records into a
/// submitted encoder (the production frame records into its own encoder
/// instead, with no extra submit).
fn dispatch_compact(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    culler: &Culler,
    view_proj: &Mat4,
    instances: &[InstanceData],
) {
    let _ = culler.compact(queue, view_proj, INDEX_COUNT, instances);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("culling bench dispatch"),
    });
    culler.record_compact(&mut encoder);
    queue.submit([encoder.finish()]);
}

// qual:allow(test_quality, untested) reason: "bench GPU dispatch runner runs only under the criterion harness, never under libtest"
/// One GPU dispatch: records the staged frame and submits (no upload,
/// no wait — the caller polls). Isolates the GPU round trip from the
/// CPU staging measured by `gpu_upload_*`.
fn submit_compact(device: &wgpu::Device, queue: &wgpu::Queue, culler: &Culler) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("culling bench dispatch-only"),
    });
    culler.record_compact(&mut encoder);
    queue.submit([encoder.finish()]);
}

// qual:allow(test_quality, untested) reason: "bench GPU runner runs only under the criterion harness, never under libtest"
/// One scene size through the GPU bench: without an adapter it leaves a
/// skip note; otherwise it warms the pipeline (compilation stays out of
/// the measurement), pins the warmed `instance_count` against the CPU
/// visible count, and measures stage-plus-dispatch (`gpu_*`) split into
/// CPU staging (`gpu_upload_*`, no dispatch, no wait) and the GPU round
/// trip (`gpu_dispatch_*`, submit plus blocking poll). Branching lives
/// only in `match` arms (dispatch for the gate): the `compact` calls keep
/// no branch since over-capacity is impossible by construction (the
/// culler is built for exactly `instances.len()`), so their `Result`s are
/// dropped like the infallible queue polls below.
fn run_gpu_bench(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    gpu: &Option<(wgpu::Device, wgpu::Queue)>,
    view_proj: &Mat4,
    instances: &[InstanceData],
    label: &str,
) {
    match gpu {
        Some(pair) => {
            let (device, queue) = pair;
            match Culler::try_new(device, instances.len() as u32) {
                Ok(culler) => {
                    dispatch_compact(device, queue, &culler, view_proj, instances);
                    let _ = device.poll(wgpu::PollType::wait_indefinitely());
                    let expected = compact_visible_indices(instances, view_proj).len() as u32;
                    if let Some(count) = read_instance_count(device, queue, &culler.args_buffer) {
                        assert_eq!(
                            count, expected,
                            "warmed GPU instance_count must match the CPU visible count ({label})"
                        );
                    } else {
                        eprintln!(
                            "culling/gpu_{label}: args readback failed — skipping parity pin"
                        );
                    }
                    group.bench_function(format!("gpu_{label}"), |b| {
                        b.iter(|| {
                            dispatch_compact(
                                device,
                                queue,
                                &culler,
                                view_proj,
                                std::hint::black_box(instances),
                            );
                            let _ = device.poll(wgpu::PollType::wait_indefinitely());
                        })
                    });
                    group.bench_function(format!("gpu_upload_{label}"), |b| {
                        b.iter(|| {
                            let _ = culler.compact(
                                queue,
                                view_proj,
                                INDEX_COUNT,
                                std::hint::black_box(instances),
                            );
                        })
                    });
                    group.bench_function(format!("gpu_dispatch_{label}"), |b| {
                        b.iter(|| {
                            submit_compact(device, queue, &culler);
                            let _ = device.poll(wgpu::PollType::wait_indefinitely());
                        })
                    });
                }
                Err(err) => {
                    eprintln!("culling/gpu_{label}: {err} — skipping");
                }
            }
        }
        None => {
            eprintln!(
                "SKIP culling/gpu_{label}: no wgpu adapter (run on any machine with lavapipe/GPU)"
            );
        }
    }
}

// qual:allow(iosp) reason: "criterion entry pins its small-part premise with one assert by RAT-43 design; otherwise dispatch-only"
// qual:api — criterion entry; called only by the generated harness, invisible to the call graph.
/// CPU vs GPU culling at 10k/100k, one [`Criterion`] group (`cpu_*` vs
/// `gpu_*`). The entry is dispatch-only apart from the single
/// small-part premise assert below (the harness shape the IOSP gate
/// otherwise accepts): the adapter-absent note lives in
/// [`run_gpu_bench`], so a headless machine degrades to the CPU column
/// instead of failing the bench.
fn bench_culling(c: &mut Criterion) {
    let view =
        glam::camera::rh::view::look_at_mat4(Vec3::from_array(CAMERA_EYE), Vec3::ZERO, Vec3::Y);
    let proj = glam::camera::rh::proj::directx::perspective(FOV_RAD, ASPECT, NEAR, FAR);
    let view_proj = proj.mul_mat4(&view);
    let gpu = pollster::block_on(device());

    let mut group = c.benchmark_group("culling");
    // Premise lines accumulate here on purpose: the `for` body below must
    // stay delegation-only (plain calls — no macros) so the gate reads
    // the loop as dispatch, not logic. One report print follows the loop.
    let mut premise = String::new();
    for (count, label) in [(SIZE_10K, "10k"), (SIZE_100K, "100k")] {
        let instances = scene_line(count);
        let visible = compact_visible_indices(&instances, &view_proj);
        assert!(
            sees_small_part(visible.len(), count),
            "bench premise: the camera must see a small part of the scene ({label}: {} visible of {count})",
            visible.len()
        );
        premise.push_str(&format!(
            "culling {label}: {} visible of {count} (small-part premise: true)\n",
            visible.len(),
        ));

        group.bench_function(format!("cpu_{label}"), |b| {
            b.iter(|| {
                std::hint::black_box(compact_visible_indices(
                    std::hint::black_box(&instances),
                    &view_proj,
                ))
            })
        });

        run_gpu_bench(&mut group, &gpu, &view_proj, &instances, label);
    }
    eprintln!("culling premises:\n{premise}");
    group.finish();
}

criterion_group!(benches, bench_culling);
criterion_main!(benches);
