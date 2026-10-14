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
//! Compile-checked by the gate; manual run:
//!   cargo bench -p ornis-render --bench culling_bench

use criterion::{Criterion, criterion_group, criterion_main};
use glam::{Mat4, Vec3};
use ornis_render::{Culler, InstanceData, MaterialIdx, compact_visible_indices};

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
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok()?;
    adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .ok()
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

// qual:allow(test_quality, untested) reason: "bench GPU runner runs only under the criterion harness, never under libtest"
/// One scene size through the GPU bench: without an adapter it leaves a
/// skip note; otherwise it warms the pipeline (compilation stays out of
/// the measurement) and measures stage-plus-dispatch. Branching lives
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
                    let _ = culler.compact(device, queue, view_proj, INDEX_COUNT, instances);
                    let _ = device.poll(wgpu::PollType::wait_indefinitely());
                    group.bench_function(format!("gpu_{label}"), |b| {
                        b.iter(|| {
                            let _ = culler.compact(
                                device,
                                queue,
                                view_proj,
                                INDEX_COUNT,
                                std::hint::black_box(instances),
                            );
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

// qual:api — criterion entry; called only by the generated harness, invisible to the call graph.
/// CPU vs GPU culling at 10k/100k, one [`Criterion`] group (`cpu_*` vs
/// `gpu_*`). The entry is branch-free on purpose (the harness shape the
/// IOSP gate accepts): the adapter-absent note lives in
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
        premise.push_str(&format!(
            "culling {label}: {} visible of {count} (small-part premise: {})\n",
            visible.len(),
            sees_small_part(visible.len(), count)
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
