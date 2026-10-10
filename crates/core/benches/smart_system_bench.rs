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
//! Not part of the CI gate: the Performance workflow runs it manually and
//! the numbers go into the PR description (frame time + ratio to AoS).

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use ornis_core::{Entity, Pack, Resources, Schedule, SmartStore, System, SystemAccess};

/// Per-object bundle under test: one generated lane per field.
#[derive(Clone, Debug, ornis_macros::Pack)]
struct Body {
    position: f32,
    velocity: f32,
    active: bool,
    target: Option<Entity>,
}

#[allow(non_camel_case_types)]
type PosLane = Body__position__PackLane__0;
#[allow(non_camel_case_types)]
type VelLane = Body__velocity__PackLane__1;
#[allow(non_camel_case_types)]
type ActiveLane = Body__active__PackLane__2;
#[allow(non_camel_case_types)]
type TargetLane = Body__target__PackLane__3;

const DT: f32 = 1.0 / 60.0;
const GRAVITY: f32 = -9.81;
/// Pull toward the link target per step (scenario c).
const LINK_PULL: f32 = 0.1;

// ---------------------------------------------------------------------------
// (a) linear integration
// ---------------------------------------------------------------------------

#[ornis_macros::smart_system]
fn integrate_body(b: &mut Body) {
    b.velocity += GRAVITY * DT;
    b.position += b.velocity * DT;
}

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

#[ornis_macros::smart_system]
fn filter_body(b: &mut Body) {
    if !b.active {
        return;
    }
    b.velocity += GRAVITY * DT;
    b.position += b.velocity * DT;
}

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

#[ornis_macros::smart_system]
fn link_body(b: &mut Body) {
    let goal = b.target.map(|t| t.position).unwrap_or(0.0);
    b.position += (goal - b.position) * LINK_PULL;
}

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
// (3) naive AoS baseline
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct AosBody {
    position: f32,
    velocity: f32,
    active: bool,
    target: Option<usize>,
}

fn aos_integrate(bodies: &mut [AosBody]) {
    for body in bodies.iter_mut() {
        body.velocity += GRAVITY * DT;
        body.position += body.velocity * DT;
    }
}

fn aos_filter(bodies: &mut [AosBody]) {
    for body in bodies.iter_mut() {
        if !body.active {
            continue;
        }
        body.velocity += GRAVITY * DT;
        body.position += body.velocity * DT;
    }
}

fn aos_link(bodies: &mut [AosBody]) {
    for index in 0..bodies.len() {
        let goal = bodies[index]
            .target
            .map_or(0.0, |target| bodies[target].position);
        bodies[index].position += (goal - bodies[index].position) * LINK_PULL;
    }
}

// ---------------------------------------------------------------------------
// setup (not measured)
// ---------------------------------------------------------------------------

/// Deterministic xorshift for scattered targets (fixed seed, no extra deps).
struct XorShift(u64);

/// Xorshift64 step constants (Marsaglia).
const XOR_A: u32 = 13;
const XOR_B: u32 = 7;
const XOR_C: u32 = 17;
/// Fixed seed: layouts must be identical across variants.
const XOR_SEED: u64 = 0x9E3779B97F4A7C15;

impl XorShift {
    fn next(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << XOR_A;
        self.0 ^= self.0 >> XOR_B;
        self.0 ^= self.0 << XOR_C;
        (self.0 as usize) % bound
    }
}

enum FilterDensity {
    Half,
    OnePercent,
}

impl FilterDensity {
    fn active(&self, index: usize) -> bool {
        match self {
            FilterDensity::Half => index.is_multiple_of(2),
            FilterDensity::OnePercent => index.is_multiple_of(100),
        }
    }

    fn id(&self) -> &'static str {
        match self {
            FilterDensity::Half => "50pct",
            FilterDensity::OnePercent => "1pct",
        }
    }
}

enum LinkLayout {
    Adjacent,
    Scattered,
}

impl LinkLayout {
    fn id(&self) -> &'static str {
        match self {
            LinkLayout::Adjacent => "adjacent",
            LinkLayout::Scattered => "scattered",
        }
    }
}

fn build_world(
    count: usize,
    density: &FilterDensity,
    layout: &LinkLayout,
) -> (Resources, Vec<Entity>) {
    let mut store = SmartStore::new();
    Body::pack_register(&mut store);
    let mut bodies = Vec::with_capacity(count);
    for index in 0..count {
        let entity = store.create_entity();
        bodies.push((
            entity,
            Body {
                position: index as f32,
                velocity: 1.0,
                active: density.active(index),
                target: None,
            },
        ));
    }
    let mut rng = XorShift(XOR_SEED);
    let entities: Vec<Entity> = bodies.iter().map(|(entity, _)| *entity).collect();
    for (index, (entity, body)) in bodies.iter_mut().enumerate() {
        let target = match layout {
            LinkLayout::Adjacent => entities[(index + 1) % count],
            LinkLayout::Scattered => entities[rng.next(count)],
        };
        body.target = Some(target);
        body.pack_insert(&mut store, *entity);
    }
    let mut resources = Resources::new();
    resources.insert(store);
    (resources, entities)
}

fn build_aos(count: usize, density: &FilterDensity, layout: &LinkLayout) -> Vec<AosBody> {
    let mut rng = XorShift(XOR_SEED);
    (0..count)
        .map(|index| {
            let target = match layout {
                LinkLayout::Adjacent => (index + 1) % count,
                LinkLayout::Scattered => rng.next(count),
            };
            AosBody {
                position: index as f32,
                velocity: 1.0,
                active: density.active(index),
                target: Some(target),
            }
        })
        .collect()
}

fn schedule_for(system: impl System + 'static) -> Schedule {
    let mut schedule = Schedule::new();
    schedule.add_system(system);
    schedule
}

// ---------------------------------------------------------------------------
// criterion groups
// ---------------------------------------------------------------------------

fn bench_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("smart_system_linear");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(4))
        .warm_up_time(Duration::from_secs(1));
    for count in [10_000, 100_000] {
        group.throughput(Throughput::Elements(count as u64));
        let density = FilterDensity::Half;
        let layout = LinkLayout::Adjacent;

        let (resources, _) = build_world(count, &density, &layout);
        let schedule = schedule_for(IntegrateBodySystem::new());
        group.bench_with_input(BenchmarkId::new("smart_system", count), &count, |b, _| {
            b.iter(|| schedule.run(&resources));
        });

        let (resources, _) = build_world(count, &density, &layout);
        let schedule = schedule_for(IntegrateManual);
        group.bench_with_input(BenchmarkId::new("manual_system", count), &count, |b, _| {
            b.iter(|| schedule.run(&resources));
        });

        let mut bodies = build_aos(count, &density, &layout);
        group.bench_with_input(BenchmarkId::new("naive_aos", count), &count, |b, _| {
            b.iter(|| aos_integrate(&mut bodies));
        });
    }
    group.finish();
}

fn bench_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("smart_system_filter");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(4))
        .warm_up_time(Duration::from_secs(1));
    for density in [FilterDensity::Half, FilterDensity::OnePercent] {
        for count in [10_000, 100_000] {
            group.throughput(Throughput::Elements(count as u64));
            let layout = LinkLayout::Adjacent;
            let name = format!("{}_{}", density.id(), count);

            let (resources, _) = build_world(count, &density, &layout);
            let schedule = schedule_for(FilterBodySystem::new());
            group.bench_with_input(BenchmarkId::new("smart_system", &name), &count, |b, _| {
                b.iter(|| schedule.run(&resources));
            });

            let (resources, _) = build_world(count, &density, &layout);
            let schedule = schedule_for(FilterManual);
            group.bench_with_input(BenchmarkId::new("manual_system", &name), &count, |b, _| {
                b.iter(|| schedule.run(&resources));
            });

            let mut bodies = build_aos(count, &density, &layout);
            group.bench_with_input(BenchmarkId::new("naive_aos", &name), &count, |b, _| {
                b.iter(|| aos_filter(&mut bodies));
            });
        }
    }
    group.finish();
}

fn bench_link(c: &mut Criterion) {
    let mut group = c.benchmark_group("smart_system_link");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(4))
        .warm_up_time(Duration::from_secs(1));
    for layout in [LinkLayout::Adjacent, LinkLayout::Scattered] {
        for count in [10_000, 100_000] {
            group.throughput(Throughput::Elements(count as u64));
            let density = FilterDensity::Half;
            let name = format!("{}_{}", layout.id(), count);

            let (resources, _) = build_world(count, &density, &layout);
            let schedule = schedule_for(LinkBodySystem::new());
            group.bench_with_input(BenchmarkId::new("smart_system", &name), &count, |b, _| {
                b.iter(|| schedule.run(&resources));
            });

            let (resources, _) = build_world(count, &density, &layout);
            let schedule = schedule_for(LinkManual);
            group.bench_with_input(BenchmarkId::new("manual_system", &name), &count, |b, _| {
                b.iter(|| schedule.run(&resources));
            });

            let mut bodies = build_aos(count, &density, &layout);
            group.bench_with_input(BenchmarkId::new("naive_aos", &name), &count, |b, _| {
                b.iter(|| aos_link(&mut bodies));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_linear, bench_filter, bench_link);
criterion_main!(benches);
