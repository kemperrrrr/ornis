//! S0 baseline (PLAN.md, Приложение C): how expensive is one
//! `compute_layout` on the three technique graphs, and what does the S1
//! cache cost on a hit.
//!
//! Run: `cargo bench -p ornis-render`. Record the numbers in
//! `docs/rendering/unified-scheduler.md` (S0 table).

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use ornis_render::{RenderGraph3D, Technique};

/// The three production wirings (bloom on — the heaviest variant):
/// Forward 7 passes / Deferred 8 / Hybrid 9, 10–12 declared resources.
fn make(technique: Technique) -> RenderGraph3D {
    RenderGraph3D::new_with(
        wgpu::TextureFormat::Rgba8Unorm,
        (1920, 1080),
        technique,
        true,
    )
}

fn bench_layout_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout/compute");
    for (name, technique) in [
        ("forward_7_passes", Technique::Forward),
        ("deferred_8_passes", Technique::Deferred),
        ("hybrid_9_passes", Technique::Hybrid),
    ] {
        let mut g3 = make(technique);
        group.bench_function(name, |b| {
            b.iter(|| {
                // A mutation-driven recompute: the cost the cache removes
                // from steady-state frames.
                g3.graph_mut().invalidate();
                black_box(g3.graph_mut().layout());
            });
        });
    }
    group.finish();
}

fn bench_layout_cache_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout/cache_hit");
    for (name, technique) in [
        ("forward_7_passes", Technique::Forward),
        ("deferred_8_passes", Technique::Deferred),
        ("hybrid_9_passes", Technique::Hybrid),
    ] {
        let mut g3 = make(technique);
        let _ = g3.graph_mut().layout(); // warm the cache
        group.bench_function(name, |b| {
            b.iter(|| {
                // Steady-state frame: no mutations → cache hit.
                black_box(g3.graph_mut().layout());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_layout_compute, bench_layout_cache_hit);
criterion_main!(benches);
