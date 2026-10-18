//! Shared scaffolding for the `smart_system` benchmark (RAT-31).
//!
//! World builders, the naive AoS baseline, and benchmark constants live
//! here so the benchmark file itself stays under the module-length cap.
//! None of this runs under `cargo test` (criterion harness); the helpers
//! are exercised by the benchmarks, mirroring `comparison_bench`.

use ornis_core::{Entity, Pack, Resources, Schedule, SmartStore, System};

/// Per-object bundle under test: one generated lane per field.
#[derive(Clone, Debug, ornis_macros::Pack)]
pub struct Body {
    pub position: f32,
    pub velocity: f32,
    pub active: bool,
    pub target: Option<Entity>,
}

#[allow(non_camel_case_types)]
pub type PosLane = Body__position__PackLane;
#[allow(non_camel_case_types)]
pub type VelLane = Body__velocity__PackLane;
#[allow(non_camel_case_types)]
pub type ActiveLane = Body__active__PackLane;
#[allow(non_camel_case_types)]
pub type TargetLane = Body__target__PackLane;

/// Fixed step shared by all variants (seconds).
pub const DT: f32 = 1.0 / 60.0;
/// Gravity applied by the linear scenario (m/s²).
pub const GRAVITY: f32 = -9.81;
/// Pull toward the link target per step (scenario c).
pub const LINK_PULL: f32 = 0.1;
/// Entity counts: the owner-mandated 10k/100k scenarios.
pub const ENTITY_COUNTS: [usize; 2] = [10_000, 100_000];
/// Criterion samples per benchmark (keeps the full run to minutes).
pub const SAMPLE_SIZE: usize = 20;
/// Measurement and warm-up budget per benchmark (seconds).
pub const MEASURE_SECS: u64 = 4;
/// Warm-up budget per benchmark (seconds).
pub const WARMUP_SECS: u64 = 1;

/// Naive AoS baseline body (plain struct, no lanes/scheduler).
#[derive(Clone, Debug)]
pub struct AosBody {
    pub position: f32,
    pub velocity: f32,
    pub active: bool,
    pub target: Option<usize>,
}

// qual:allow(test_quality, untested) reason: "benchmark-only kernels — exercised by criterion runs, mirroring comparison_bench helpers which carry the same note via baseline"
/// Linear integration over the AoS baseline.
pub fn aos_integrate(bodies: &mut [AosBody]) {
    for body in bodies.iter_mut() {
        body.velocity += GRAVITY * DT;
        body.position += body.velocity * DT;
    }
}

// qual:allow(test_quality, untested) reason: "benchmark-only kernels — exercised by criterion runs, mirroring comparison_bench helpers which carry the same note via baseline"
/// Filtered integration over the AoS baseline.
pub fn aos_filter(bodies: &mut [AosBody]) {
    for body in bodies.iter_mut() {
        if !body.active {
            continue;
        }
        body.velocity += GRAVITY * DT;
        body.position += body.velocity * DT;
    }
}

// qual:allow(test_quality, untested) reason: "benchmark-only kernels — exercised by criterion runs, mirroring comparison_bench helpers which carry the same note via baseline"
/// Link-following over the AoS baseline.
pub fn aos_link(bodies: &mut [AosBody]) {
    for index in 0..bodies.len() {
        let goal = bodies[index]
            .target
            .map_or(0.0, |target| bodies[target].position);
        bodies[index].position += (goal - bodies[index].position) * LINK_PULL;
    }
}

/// Deterministic xorshift for scattered targets (fixed seed, no extra deps).
pub struct XorShift(u64);

/// Xorshift64 step constants (Marsaglia).
const XOR_A: u32 = 13;
const XOR_B: u32 = 7;
const XOR_C: u32 = 17;
/// Fixed seed: layouts must be identical across variants.
const XOR_SEED: u64 = 0x9E3779B97F4A7C15;

impl XorShift {
    /// Next pseudo-random index below `bound`.
    pub fn next(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << XOR_A;
        self.0 ^= self.0 >> XOR_B;
        self.0 ^= self.0 << XOR_C;
        (self.0 as usize) % bound
    }

    /// Fresh generator at the fixed seed.
    // qual:allow(test_quality, untested) reason: "benchmark-only setup — exercised by criterion runs, mirroring comparison_bench helpers"
    pub fn seeded() -> Self {
        Self(XOR_SEED)
    }
}

/// Filter densities for scenario (b).
pub enum FilterDensity {
    /// Roughly half the entities pass.
    Half,
    /// Roughly one percent passes.
    OnePercent,
}

impl FilterDensity {
    /// Whether `index` passes the filter.
    pub fn active(&self, index: usize) -> bool {
        match self {
            FilterDensity::Half => index.is_multiple_of(2),
            FilterDensity::OnePercent => index.is_multiple_of(100),
        }
    }

    /// Criterion benchmark-id fragment.
    pub fn id(&self) -> &'static str {
        match self {
            FilterDensity::Half => "50pct",
            FilterDensity::OnePercent => "1pct",
        }
    }
}

/// Target layouts for scenario (c).
pub enum LinkLayout {
    /// Targets are adjacent (cache-friendly gather).
    Adjacent,
    /// Targets are xorshift-scattered.
    Scattered,
}

impl LinkLayout {
    /// Criterion benchmark-id fragment.
    pub fn id(&self) -> &'static str {
        match self {
            LinkLayout::Adjacent => "adjacent",
            LinkLayout::Scattered => "scattered",
        }
    }
}

// qual:allow(test_quality, untested) reason: "benchmark-only setup — exercised by criterion runs, mirroring comparison_bench helpers which carry the same note via baseline"
/// Builds a world of `count` bodies (setup, not measured).
pub fn build_world(
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
    let mut rng = XorShift::seeded();
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

// qual:allow(test_quality, untested) reason: "benchmark-only setup — exercised by criterion runs, mirroring comparison_bench helpers which carry the same note via baseline"
/// Builds the AoS baseline vector (setup, not measured).
pub fn build_aos(count: usize, density: &FilterDensity, layout: &LinkLayout) -> Vec<AosBody> {
    let mut rng = XorShift::seeded();
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

// qual:allow(test_quality, untested) reason: "benchmark-only setup — exercised by criterion runs, mirroring comparison_bench helpers which carry the same note via baseline"
/// Wraps one system in a single-system schedule.
pub fn schedule_for(system: impl System + 'static) -> Schedule {
    let mut schedule = Schedule::new();
    schedule.add_system(system);
    schedule
}
