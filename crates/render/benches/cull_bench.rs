//! [RAT-35](/RAT/issues/RAT-35): cost of the CPU culling compaction at
//! 10k/100k instances where the camera sees a small part of the scene —
//! the CPU side of the CPU-vs-GPU culling table (the GPU column lands
//! with the compute-dispatch wiring slice).
//!
//! Run: `cargo bench -p ornis-render --bench cull_bench`.

use criterion::{Criterion, criterion_group, criterion_main};
use glam::Mat4;
use glam::Vec3;
use ornis_assets::scene::CameraProjection;
use ornis_core::units::Degrees;
use ornis_core::units::Meters;
use ornis_render::CameraView;
use ornis_render::InstanceData;
use ornis_render::MaterialIdx;
use ornis_render::camera_view_projection;
use ornis_render::compact_visible_indices;

/// Grid spacing in world units between instances.
const GRID_SPACING: f32 = 2.0;

/// Camera over the grid center: sees the middle of the field, culling
/// the vast majority at 10k/100k.
fn narrow_view() -> (CameraView, (u32, u32)) {
    (
        CameraView {
            eye: Vec3::new(0.0, 2.5, 9.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            projection: CameraProjection::Perspective,
            fov: Degrees::new(60.0),
            near: Meters::new(0.1),
            far: Meters::new(100.0),
        },
        (160, 90),
    )
}

/// `count` translation-only instances on a centered XZ grid.
fn grid_scene(count: usize) -> Vec<InstanceData> {
    let side = count.isqrt().max(1);
    let half = side as f32 * GRID_SPACING * 0.5;
    let mut instances = Vec::with_capacity(count);
    for i in 0..count {
        let gx = (i % side) as f32 * GRID_SPACING - half;
        let gz = (i / side) as f32 * GRID_SPACING - half;
        let model = Mat4::from_translation(Vec3::new(gx, 0.0, gz));
        instances.push(InstanceData {
            model_matrix: model,
            normal_matrix: Mat4::IDENTITY,
            material_index: MaterialIdx::from_raw(0),
        });
    }
    instances
}

fn bench_compact(c: &mut Criterion) {
    let (view, size) = narrow_view();
    let (view_proj, _) = camera_view_projection(&view, size);
    let mut group = c.benchmark_group("cull/compact_visible_indices");
    for count in [10_000, 100_000] {
        let scene = grid_scene(count);
        let visible = compact_visible_indices(&scene, &view_proj).len();
        group.bench_function(format!("{count}_instances_{visible}_visible"), |b| {
            b.iter(|| {
                std::hint::black_box(compact_visible_indices(
                    std::hint::black_box(&scene),
                    std::hint::black_box(&view_proj),
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compact);
criterion_main!(benches);
