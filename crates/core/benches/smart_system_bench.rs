//! Benchmark: hand-written `impl System` vs `#[smart_system]` vs naive AoS (RAT-31, IDEAS §32).
//!
//! The same per-object logic runs in three forms over 10k/100k entities:
//! (1) manual lane-guard loops registered as `impl System` (today's style),
//! (2) the generated `#[smart_system]` kernel through the same `Schedule`,
//! (3) a plain `Vec` + `for` loop as the absolute baseline.
//! Scenarios: (a) linear integration, (b) `if`/`return` filter passing ~50%
//! and ~1% of entities, (c) cross-entity reads through `target: Entity`
//! with adjacent (cache-friendly) and scattered targets.
//!
//! Scaffolding lives in `support.rs` so this file stays under the
//! module-length cap. Not part of the CI gate: the Performance workflow
//! runs it manually and the numbers go into the PR description (frame
//! time + ratio to AoS).

#[path = "smart_system_bench/support.rs"]
mod support;

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use ornis_core::{Entity, Pack, Resources, SmartStore, System, SystemAccess};
use support::{
    ActiveLane, Body, ENTITY_COUNTS, FilterDensity, LinkLayout, MEASURE_SECS, PosLane, SAMPLE_SIZE,
    TargetLane, VelLane, WARMUP_SECS, aos_filter, aos_integrate, aos_link, build_aos, build_world,
    schedule_for,
};
use support::{
    Body__active__PackLane, Body__position__PackLane, Body__target__PackLane,
    Body__velocity__PackLane,
};
use support::{DT, GRAVITY, LINK_PULL};

// ---------------------------------------------------------------------------
// (a) linear integration
// ---------------------------------------------------------------------------

/// `#[smart_system]` kernel (consumed by the macro; invisible to the
/// call graph, like the `comparison_bench` helpers in baseline).
#[allow(dead_code)]
#[ornis_macros::smart_system]
fn integrate_body(b: &mut Body) {
    b.velocity += GRAVITY * DT;
    b.position += b.velocity * DT;
}

/// Today's style: one guard per lane, tight zip over dense data.
struct IntegrateManual;

impl System for IntegrateManual {
    fn name(&self) -> &'static str {
        "integrate_manual"
    }

    fn access(&self) -> SystemAccess {
        let mut access = SystemAccess::new().reads::<SmartStore>();
        for lane in Body::pack_lane_ids() {
            access = access.writes_lane_id(lane);
        }
        access
    }

    // qual:allow(complexity, error_handling) reason: "bench-only lane fetch — lanes are guaranteed by build_world setup; mirrors production pack_put which skips missing lanes instead"
    fn run(&self, resources: &Resources) {
        let store = resources.get::<SmartStore>().expect("store");
        let mut positions = store.write_lane::<PosLane>().expect("position lane");
        let mut velocities = store.write_lane::<VelLane>().expect("velocity lane");
        debug_assert_eq!(positions.data.len(), velocities.data.len());
        for (position, velocity) in positions.data.iter_mut().zip(velocities.data.iter_mut()) {
            velocity.0 += GRAVITY * DT;
            position.0 += velocity.0 * DT;
        }
    }
}

// ---------------------------------------------------------------------------
// (b) filter via early return
// ---------------------------------------------------------------------------

/// `#[smart_system]` kernel (consumed by the macro; invisible to the
/// call graph, like the `comparison_bench` helpers in baseline).
#[allow(dead_code)]
#[ornis_macros::smart_system]
fn filter_body(b: &mut Body) {
    if !b.active {
        return;
    }
    b.velocity += GRAVITY * DT;
    b.position += b.velocity * DT;
}

/// Today's style with an explicit skip over inactive lanes.
struct FilterManual;

impl System for FilterManual {
    fn name(&self) -> &'static str {
        "filter_manual"
    }

    fn access(&self) -> SystemAccess {
        let mut access = SystemAccess::new().reads::<SmartStore>();
        for lane in Body::pack_lane_ids() {
            access = access.writes_lane_id(lane);
        }
        access
    }

    // qual:allow(complexity, error_handling) reason: "bench-only lane fetch — lanes are guaranteed by build_world setup; mirrors production pack_put which skips missing lanes instead"
    fn run(&self, resources: &Resources) {
        let store = resources.get::<SmartStore>().expect("store");
        let mut positions = store.write_lane::<PosLane>().expect("position lane");
        let mut velocities = store.write_lane::<VelLane>().expect("velocity lane");
        let actives = store.read_lane::<ActiveLane>().expect("active lane");
        debug_assert_eq!(positions.data.len(), velocities.data.len());
        debug_assert_eq!(positions.data.len(), actives.data.len());
        for ((position, velocity), active) in positions
            .data
            .iter_mut()
            .zip(velocities.data.iter_mut())
            .zip(actives.data.iter())
        {
            if !active.0 {
                continue;
            }
            velocity.0 += GRAVITY * DT;
            position.0 += velocity.0 * DT;
        }
    }
}

// ---------------------------------------------------------------------------
// (c) cross-entity read through `target`
// ---------------------------------------------------------------------------

/// `#[smart_system]` kernel (consumed by the macro; invisible to the
/// call graph, like the `comparison_bench` helpers in baseline).
#[allow(dead_code)]
#[ornis_macros::smart_system]
fn link_body(b: &mut Body) {
    let goal = b.target.map(|t| t.position).unwrap_or(0.0);
    b.position += (goal - b.position) * LINK_PULL;
}

/// Today's style: per-entity target gather through the pack API.
struct LinkManual;

impl System for LinkManual {
    fn name(&self) -> &'static str {
        "link_manual"
    }

    fn access(&self) -> SystemAccess {
        let mut access = SystemAccess::new().reads::<SmartStore>();
        for lane in Body::pack_lane_ids() {
            access = access.writes_lane_id(lane);
        }
        access
    }

    // qual:allow(complexity, error_handling) reason: "bench-only lane fetch — lanes are guaranteed by build_world setup; mirrors production pack_put which skips missing lanes instead"
    fn run(&self, resources: &Resources) {
        let store = resources.get::<SmartStore>().expect("store");
        let entities: Vec<Entity> = {
            let positions = store.read_lane::<PosLane>().expect("position lane");
            positions.iter_entities().collect()
        };
        for entity in entities {
            let goal = {
                let targets = store.read_lane::<TargetLane>().expect("target lane");
                targets
                    .get(entity)
                    .and_then(|link| Body::pack_get(store, link.0))
                    .map_or(0.0, |body| body.position)
            };
            let mut positions = store.write_lane::<PosLane>().expect("position lane");
            if let Some(position) = positions.get_mut(entity) {
                position.0 += (goal - position.0) * LINK_PULL;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// criterion groups (one trio per scenario tag)
// ---------------------------------------------------------------------------

/// Runs the three variants (smart/manual/AoS) for one scenario tag.
// qual:allow(test_quality, untested) reason: "benchmark harness wiring — exercised by criterion runs, mirroring comparison_bench helpers"
fn bench_trio(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tag: &str,
    count: usize,
    smart: impl Fn(),
    manual: impl Fn(),
    aos: impl FnMut(),
) {
    group.throughput(Throughput::Elements(count as u64));
    group.bench_with_input(BenchmarkId::new("smart_system", tag), &count, |b, _| {
        b.iter(&smart);
    });
    group.bench_with_input(BenchmarkId::new("manual_system", tag), &count, |b, _| {
        b.iter(&manual);
    });
    let mut aos = aos;
    group.bench_with_input(BenchmarkId::new("naive_aos", tag), &count, |b, _| {
        b.iter(&mut aos);
    });
}

/// Configures sample/measure/warm-up budgets shared by all groups.
// qual:allow(test_quality, untested) reason: "benchmark harness wiring — exercised by criterion runs, mirroring comparison_bench helpers"
fn bench_group<'a>(
    c: &'a mut Criterion,
    name: &str,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    group
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(MEASURE_SECS))
        .warm_up_time(Duration::from_secs(WARMUP_SECS));
    group
}

/// `#[allow(dead_code)]`: registered with criterion, invisible to the call graph.
#[allow(dead_code)]
fn bench_linear(c: &mut Criterion) {
    let mut group = bench_group(c, "smart_system_linear");
    for count in ENTITY_COUNTS {
        let density = FilterDensity::Half;
        let layout = LinkLayout::Adjacent;
        let (resources, _) = build_world(count, &density, &layout);
        let smart = schedule_for(IntegrateBodySystem::new());
        let (resources_manual, _) = build_world(count, &density, &layout);
        let manual = schedule_for(IntegrateManual);
        let mut bodies = build_aos(count, &density, &layout);
        bench_trio(
            &mut group,
            &count.to_string(),
            count,
            || smart.run(&resources),
            || manual.run(&resources_manual),
            || aos_integrate(&mut bodies),
        );
    }
    group.finish();
}

/// `#[allow(dead_code)]`: registered with criterion, invisible to the call graph.
#[allow(dead_code)]
fn bench_filter(c: &mut Criterion) {
    let mut group = bench_group(c, "smart_system_filter");
    for density in [FilterDensity::Half, FilterDensity::OnePercent] {
        for count in ENTITY_COUNTS {
            let layout = LinkLayout::Adjacent;
            let name = format!("{}_{}", density.id(), count);
            let (resources, _) = build_world(count, &density, &layout);
            let smart = schedule_for(FilterBodySystem::new());
            let (resources_manual, _) = build_world(count, &density, &layout);
            let manual = schedule_for(FilterManual);
            let mut bodies = build_aos(count, &density, &layout);
            bench_trio(
                &mut group,
                &name,
                count,
                || smart.run(&resources),
                || manual.run(&resources_manual),
                || aos_filter(&mut bodies),
            );
        }
    }
    group.finish();
}

/// `#[allow(dead_code)]`: registered with criterion, invisible to the call graph.
#[allow(dead_code)]
fn bench_link(c: &mut Criterion) {
    let mut group = bench_group(c, "smart_system_link");
    for layout in [LinkLayout::Adjacent, LinkLayout::Scattered] {
        for count in ENTITY_COUNTS {
            let density = FilterDensity::Half;
            let name = format!("{}_{}", layout.id(), count);
            let (resources, _) = build_world(count, &density, &layout);
            let smart = schedule_for(LinkBodySystem::new());
            let (resources_manual, _) = build_world(count, &density, &layout);
            let manual = schedule_for(LinkManual);
            let mut bodies = build_aos(count, &density, &layout);
            bench_trio(
                &mut group,
                &name,
                count,
                || smart.run(&resources),
                || manual.run(&resources_manual),
                || aos_link(&mut bodies),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_linear, bench_filter, bench_link);
criterion_main!(benches);
