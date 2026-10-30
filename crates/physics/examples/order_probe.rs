//! Body-order locality probe v2 (issue #178, review follow-up).
//!
//! The question: does re-ordering `bodies` by island / Morton locality make
//! the solver faster? v1 measured the falling transient (`settle: 0`), used a
//! fixed ±64 m Morton quantiser, and only produced a comparison-sort cost.
//! This version addresses the review:
//!
//! * every scene is warmed for [`DISCOVERY_STEPS`] steps before any timing,
//!   then every timed run starts from that settled pose set (review item 1);
//! * equivalence is measured, not assumed: each layout runs the same
//!   settled scene and we report the final-position divergence and the
//!   contact-count difference against `spatial` (item 2);
//! * Morton keys use the scene's own AABB, so `body_grid_10000` keeps its
//!   full ±100 m extent (item 3);
//! * an `islands` layout groups bodies by the contact/island graph measured
//!   after settling (item 4);
//! * the `pile_2000` churn scene runs 600 timed steps with a rebuild-based
//!   re-sort every N ∈ {1, 10, 60} steps (item 5);
//! * ordering cost is split into Morton-key computation, comparison sort,
//!   LSD radix sort over precomputed keys, permutation apply, and the
//!   handle-table indirection a stable external handle needs (item 6);
//! * `morton` is compared against `shuffled_all` as well as `spatial` (item 7);
//! * [`REPEATS`] interleaved repeats, median + IQR, a paired Wilcoxon
//!   signed-rank test, one Rayon worker, and a raw CSV (item 8);
//! * the same matrix runs on [`AvbdEngine`] (item 10).
//!
//! Run single-threaded in release on an otherwise idle machine:
//!
//! ```text
//! cargo run --release -p ornis-physics --example order_probe
//! ```
//!
//! The probe forces `RAYON_NUM_THREADS=1` itself and writes four CSV files
//! under `target/order_probe/` (`raw.csv`, `summary.csv`, `equivalence.csv`,
//! `ordering_cost.csv`, `churn.csv`, `diagnostics.csv`).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

use glam::Vec3;
use ornis_physics::{
    AvbdEngine, BodyHandle, BodyType, ContactEvent, ContactEventKind, PhysicsEngine, RigidBody,
    SequentialImpulseEngine,
};

const DT: f32 = 1.0 / 60.0;
const BOX_HALF: f32 = 0.4;
const PITCH: f32 = 2.0;
const LEVEL_PITCH: f32 = 0.8;
const REST_Y: f32 = 0.4;
const FLOOR_HALF_Y: f32 = 0.5;
const TILE_HALF: f32 = 5.05;

/// Steps run once per scene to reach a settled pose set (review item 1).
const DISCOVERY_STEPS: u32 = 150;
/// Short re-warm after rebuilding a settled scene in a new order.
const REWARM_STEPS: u32 = 3;
/// Interleaved measurement repeats per (engine, scene, layout) cell.
const REPEATS: usize = 30;
/// Timed steps inside one repeat.
const MEASURE_STEPS_SMALL: u32 = 8;
const MEASURE_STEPS_LARGE: u32 = 6;
/// Timed churn steps (review item 5).
const CHURN_STEPS: u32 = 600;
const CHURN_WARM_STEPS: u32 = 120;
const CHURN_REPEATS: usize = 5;

fn gravity() -> Vec3 {
    Vec3::new(0.0, -9.81, 0.0)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EngineKind {
    Si,
    Avbd,
}

impl EngineKind {
    fn name(self) -> &'static str {
        match self {
            Self::Si => "si",
            Self::Avbd => "avbd",
        }
    }
}

fn make_engine(kind: EngineKind) -> Box<dyn PhysicsEngine> {
    match kind {
        EngineKind::Si => Box::new(SequentialImpulseEngine::new(gravity())),
        EngineKind::Avbd => Box::new(AvbdEngine::new(gravity())),
    }
}

/// Build an engine whose physical order is `order` (`order[handle] = logical`).
fn build(kind: EngineKind, order: &[usize], bodies: &[RigidBody]) -> Box<dyn PhysicsEngine> {
    let mut engine = make_engine(kind);
    for &logical in order {
        engine.add_body(bodies[logical].clone());
    }
    engine
}

/// Build an engine whose physical body order is already `bodies`.
fn build_direct(kind: EngineKind, bodies: &[RigidBody]) -> Box<dyn PhysicsEngine> {
    let mut engine = make_engine(kind);
    for body in bodies {
        engine.add_body(body.clone());
    }
    engine
}

fn sync_pose(dst: &mut RigidBody, src: &RigidBody) {
    dst.position = src.position;
    dst.orientation = src.orientation;
    dst.velocity = src.velocity;
    dst.angular_velocity = src.angular_velocity;
}

/// Copy the current pose of every handle back into `out` (`out[logical]`).
fn capture_poses(
    engine: &dyn PhysicsEngine,
    order: &[usize],
    out: &mut [RigidBody],
    fallback: &[RigidBody],
) {
    for (handle, &logical) in order.iter().enumerate() {
        if let Some(body) = engine.get_body(BodyHandle::from(handle)) {
            sync_pose(&mut out[logical], body);
        } else {
            sync_pose(&mut out[logical], &fallback[logical]);
        }
    }
}

fn logical_of(handle: usize, order: &[usize]) -> usize {
    order[handle]
}

fn pair_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// Fold drained contact transitions into the live touching-pair set, in
/// logical body ids so the set is layout-independent.
fn track_contacts(
    touch: &mut BTreeSet<(usize, usize)>,
    events: Vec<ContactEvent>,
    order: &[usize],
) {
    for event in events {
        let a = logical_of(event.body_a.index(), order);
        let b = logical_of(event.body_b.index(), order);
        let key = pair_key(a, b);
        match event.kind {
            ContactEventKind::Begin | ContactEventKind::Hit { .. } => {
                touch.insert(key);
            }
            ContactEventKind::End => {
                touch.remove(&key);
            }
        }
    }
}

fn union_find(n: usize, pairs: &BTreeSet<(usize, usize)>, dynamic: &[bool]) -> Vec<u32> {
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let grand = parent[parent[x as usize] as usize];
            parent[x as usize] = grand;
            x = grand;
        }
        x
    }
    for &(a, b) in pairs {
        // The engine keeps statics out of dynamic islands; a shared floor
        // must not fuse every stack into one island.
        if !dynamic[a] || !dynamic[b] {
            continue;
        }
        let ra = find(&mut parent, a as u32);
        let rb = find(&mut parent, b as u32);
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            parent[hi as usize] = lo;
        }
    }
    (0..n).map(|i| find(&mut parent, i as u32)).collect()
}

struct Discovered {
    /// Settled pose set in logical order (masses from the original bodies).
    settled: Vec<RigidBody>,
    /// 30-bit Morton code per logical body, over the settled scene AABB.
    morton_keys: Vec<u64>,
    /// Island root per logical body.
    island_root: Vec<u32>,
    /// Live solid-contact pairs at the settled state (logical ids).
    contacts: BTreeSet<(usize, usize)>,
}

fn aabb_of(bodies: &[RigidBody]) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for body in bodies {
        min = min.min(body.position);
        max = max.max(body.position);
    }
    let pad = Vec3::splat(1.0e-3);
    (min - pad, max + pad)
}

fn spread_10bit(mut x: u64) -> u64 {
    x &= 0x3ff;
    x = (x | (x << 16)) & 0x0300_00ff_0000_00ff;
    x = (x | (x << 8)) & 0x000f_00f0_00f0_00f0;
    x = (x | (x << 4)) & 0x0c30_c30c_30c3_0c30;
    x = (x | (x << 2)) & 0x2492_4924_9249_2492;
    x
}

/// 10-bit-per-axis Morton code using the scene AABB (review item 3).
fn morton_keys(bodies: &[RigidBody]) -> Vec<u64> {
    let (min, max) = aabb_of(bodies);
    let span = (max - min).max(Vec3::splat(1.0e-6));
    bodies
        .iter()
        .map(|body| {
            let q = (body.position - min) / span;
            let q = (q * 1023.0).clamp(Vec3::ZERO, Vec3::splat(1023.0));
            let x = q.x.round() as u64;
            let y = q.y.round() as u64;
            let z = q.z.round() as u64;
            spread_10bit(x) | (spread_10bit(y) << 1) | (spread_10bit(z) << 2)
        })
        .collect()
}

fn discover(kind: EngineKind, bodies: &[RigidBody], steps: u32) -> Discovered {
    let n = bodies.len();
    let order: Vec<usize> = (0..n).collect();
    let mut engine = build(kind, &order, bodies);
    let mut touch = BTreeSet::new();
    for _ in 0..steps {
        engine.step(DT);
        track_contacts(&mut touch, engine.drain_contact_events(), &order);
    }
    let mut settled = bodies.to_vec();
    capture_poses(engine.as_ref(), &order, &mut settled, bodies);
    let keys = morton_keys(&settled);
    let dynamic: Vec<bool> = bodies
        .iter()
        .map(|body| body.body_type != BodyType::Static)
        .collect();
    let roots = union_find(n, &touch, &dynamic);
    Discovered {
        settled,
        morton_keys: keys,
        island_root: roots,
        contacts: touch,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    Spatial,
    ShuffledAll,
    ShuffledDynamic,
    ShuffledStatic,
    Morton,
    Islands,
}

const LAYOUTS: [Layout; 6] = [
    Layout::Spatial,
    Layout::ShuffledAll,
    Layout::ShuffledDynamic,
    Layout::ShuffledStatic,
    Layout::Morton,
    Layout::Islands,
];

impl Layout {
    fn name(self) -> &'static str {
        match self {
            Self::Spatial => "spatial",
            Self::ShuffledAll => "shuffled_all",
            Self::ShuffledDynamic => "shuffled_dynamic",
            Self::ShuffledStatic => "shuffled_static",
            Self::Morton => "morton",
            Self::Islands => "islands",
        }
    }
}

fn shuffle_positions(values: &mut [usize], seed: u64) {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize
    };
    for i in (1..values.len()).rev() {
        let j = next() % (i + 1);
        values.swap(i, j);
    }
}

fn shuffled_order(bodies: &[RigidBody], layout: Layout, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..bodies.len()).collect();
    let movable: Vec<usize> = (0..bodies.len())
        .filter(|&i| match layout {
            Layout::ShuffledAll => true,
            Layout::ShuffledDynamic => bodies[i].body_type == BodyType::Dynamic,
            Layout::ShuffledStatic => bodies[i].body_type == BodyType::Static,
            _ => false,
        })
        .collect();
    let mut values: Vec<usize> = movable.iter().map(|&p| order[p]).collect();
    shuffle_positions(&mut values, seed);
    for (slot, &position) in movable.iter().enumerate() {
        order[position] = values[slot];
    }
    order
}

/// Group bodies into islands and order each island breadth-first from its
/// lowest Morton key, islands by their minimum key (review item 4).
fn island_order(discovered: &Discovered) -> Vec<usize> {
    let n = discovered.settled.len();
    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (logical, &root) in discovered.island_root.iter().enumerate() {
        groups.entry(root).or_default().push(logical);
    }
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in &discovered.contacts {
        adjacency[a].push(b);
        adjacency[b].push(a);
    }
    let mut islands: Vec<(u64, Vec<usize>)> = Vec::with_capacity(groups.len());
    for members in groups.into_values() {
        let start = *members
            .iter()
            .min_by_key(|&&m| (discovered.morton_keys[m], m))
            .expect("island has at least one body");
        let mut has_member = vec![false; n];
        for &m in &members {
            has_member[m] = true;
        }
        let mut seen = vec![false; n];
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        let mut ordered = Vec::with_capacity(members.len());
        while let Some(body) = queue.pop_front() {
            ordered.push(body);
            for &next in &adjacency[body] {
                if has_member[next] && !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
        islands.push((discovered.morton_keys[start], ordered));
    }
    islands.sort_by_key(|(key, _)| *key);
    islands
        .into_iter()
        .flat_map(|(_, ordered)| ordered)
        .collect()
}

fn layout_order(
    layout: Layout,
    discovered: &Discovered,
    bodies: &[RigidBody],
    seed: u64,
) -> Vec<usize> {
    match layout {
        Layout::Spatial => (0..bodies.len()).collect(),
        Layout::ShuffledAll | Layout::ShuffledDynamic | Layout::ShuffledStatic => {
            shuffled_order(bodies, layout, seed)
        }
        Layout::Morton => {
            let mut order: Vec<usize> = (0..bodies.len()).collect();
            order.sort_by_key(|&i| (discovered.morton_keys[i], i));
            order
        }
        Layout::Islands => island_order(discovered),
    }
}

fn timed_run(
    kind: EngineKind,
    discovered: &Discovered,
    order: &[usize],
    measure_steps: u32,
    rewarm_steps: u32,
) -> f64 {
    let mut engine = build(kind, order, &discovered.settled);
    for _ in 0..rewarm_steps {
        engine.step(DT);
    }
    // Wake every island so the sample is solve work, not the sleep fast path.
    // `measure_steps` stays below the shortest island sleep delay (~0.2 s),
    // so the whole timed window is awake.
    for handle in 0..discovered.settled.len() {
        engine.wake_body(BodyHandle::from(handle));
    }
    let start = Instant::now();
    for _ in 0..measure_steps {
        engine.step(black_box(DT));
    }
    start.elapsed().as_secs_f64() * 1e9 / f64::from(measure_steps)
}

struct Equivalence {
    final_positions: Vec<Vec3>,
    contacts: usize,
}

fn equivalence_run(
    kind: EngineKind,
    discovered: &Discovered,
    order: &[usize],
    measure_steps: u32,
    rewarm_steps: u32,
) -> Equivalence {
    let mut engine = build(kind, order, &discovered.settled);
    let mut touch = BTreeSet::new();
    for _ in 0..rewarm_steps {
        engine.step(DT);
        track_contacts(&mut touch, engine.drain_contact_events(), order);
    }
    for handle in 0..discovered.settled.len() {
        engine.wake_body(BodyHandle::from(handle));
    }
    for _ in 0..measure_steps {
        engine.step(DT);
        track_contacts(&mut touch, engine.drain_contact_events(), order);
    }
    if std::env::var("ORDER_PROBE_DEBUG").is_ok() {
        eprintln!(
            "debug equivalence contact_set={} last_counts={:?}",
            touch.len(),
            {
                let mut counts = Vec::new();
                for handle in (0..discovered.settled.len()).take(3) {
                    if let Some(body) = engine.get_body(BodyHandle::from(handle)) {
                        counts.push(body.position.y);
                    }
                }
                counts
            }
        );
    }
    let mut final_bodies = discovered.settled.clone();
    capture_poses(
        engine.as_ref(),
        order,
        &mut final_bodies,
        &discovered.settled,
    );
    Equivalence {
        final_positions: final_bodies.iter().map(|body| body.position).collect(),
        contacts: touch.len(),
    }
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let index = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[index]
}

fn median_iqr(samples: &mut [f64]) -> (f64, f64, f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN samples"));
    (
        quantile(samples, 0.5),
        quantile(samples, 0.25),
        quantile(samples, 0.75),
        samples[0],
        samples[samples.len() - 1],
    )
}

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7 — enough for a p-value.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Two-sided paired Wilcoxon signed-rank p-value (normal approximation,
/// tie-averaged ranks). Returns 1.0 for fewer than six non-zero pairs.
fn wilcoxon_p(a: &[f64], b: &[f64]) -> f64 {
    let diffs: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| x - y)
        .filter(|d| d.abs() > 1.0e-9)
        .collect();
    let n = diffs.len();
    if n < 6 {
        return 1.0;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        diffs[i]
            .abs()
            .partial_cmp(&diffs[j].abs())
            .expect("finite diffs")
    });
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (diffs[order[j]].abs() - diffs[order[i]].abs()).abs() <= 1.0e-12 {
            j += 1;
        }
        let rank = (i + j + 1) as f64 / 2.0;
        for &k in &order[i..j] {
            ranks[k] = rank;
        }
        i = j;
    }
    let w_plus: f64 = diffs
        .iter()
        .enumerate()
        .filter(|(_, d)| **d > 0.0)
        .map(|(k, _)| ranks[k])
        .sum();
    let n_f = n as f64;
    let mean = n_f * (n_f + 1.0) / 4.0;
    let var = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0;
    let z = (w_plus - mean) / var.sqrt();
    let p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z.abs() / std::f64::consts::SQRT_2)));
    p.clamp(0.0, 1.0)
}

/// LSD radix sort of `0..n` by 8-bit passes over precomputed keys.
fn radix_order(keys: &[u32]) -> Vec<u32> {
    let n = keys.len();
    let mut index: Vec<u32> = (0..n as u32).collect();
    let mut buffer = vec![0u32; n];
    for shift in [0u32, 8, 16, 24] {
        let mut counts = [0usize; 256];
        for &i in &index {
            counts[((keys[i as usize] >> shift) & 0xff) as usize] += 1;
        }
        let mut sum = 0;
        for count in &mut counts {
            let taken = *count;
            *count = sum;
            sum += taken;
        }
        for &i in &index {
            let bucket = ((keys[i as usize] >> shift) & 0xff) as usize;
            buffer[counts[bucket]] = i;
            counts[bucket] += 1;
        }
        std::mem::swap(&mut index, &mut buffer);
    }
    index
}

struct OrderingCost {
    key_ns: f64,
    comparison_sort_ns: f64,
    radix_sort_ns: f64,
    permute_ns: f64,
    handle_indirection_ns_per_body: f64,
}

fn measure_ordering_cost(discovered: &Discovered, iterations: usize) -> OrderingCost {
    let bodies = &discovered.settled;
    let n = bodies.len();
    let mut key_samples = Vec::with_capacity(iterations);
    let mut comparison_samples = Vec::with_capacity(iterations);
    let mut radix_samples = Vec::with_capacity(iterations);
    let mut permute_samples = Vec::with_capacity(iterations);
    let mut table_samples = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let keys = black_box(morton_keys(bodies));
        key_samples.push(start.elapsed().as_secs_f64() * 1e9);

        let start = Instant::now();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by_key(|&i| keys[i]);
        black_box(&order);
        comparison_samples.push(start.elapsed().as_secs_f64() * 1e9);

        let u32_keys: Vec<u32> = keys.iter().map(|&k| k as u32).collect();
        let start = Instant::now();
        let radix = radix_order(black_box(&u32_keys));
        black_box(&radix);
        radix_samples.push(start.elapsed().as_secs_f64() * 1e9);

        let start = Instant::now();
        let permuted: Vec<RigidBody> = radix.iter().map(|&i| bodies[i as usize].clone()).collect();
        black_box(&permuted);
        permute_samples.push(start.elapsed().as_secs_f64() * 1e9);

        // Stable external handles need `handle -> physical slot` indirection.
        // Build the inverse table and walk it once per body.
        let mut table = vec![0u32; n];
        for (slot, &logical) in order.iter().enumerate() {
            table[logical] = slot as u32;
        }
        let start = Instant::now();
        let mut sum = 0.0f32;
        for logical in 0..n {
            sum += permuted[table[logical] as usize].inv_mass;
        }
        black_box(sum);
        table_samples.push(start.elapsed().as_secs_f64() * 1e9 / n as f64);
    }

    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        v[v.len() / 2]
    };
    OrderingCost {
        key_ns: median(key_samples),
        comparison_sort_ns: median(comparison_samples),
        radix_sort_ns: median(radix_samples),
        permute_ns: median(permute_samples),
        handle_indirection_ns_per_body: median(table_samples),
    }
}

struct ChurnResult {
    per_step_ns: Vec<f64>,
    reorder_ns: Vec<f64>,
    total_ns: f64,
}

/// Long churn (review item 5): warm once, then 600 timed steps, rebuilding
/// the engine in a Morton order every `interval` steps (`0` = never).
/// A public-API implementation can only re-order by rebuilding; that cost is
/// reported separately.
fn churn_run(
    kind: EngineKind,
    scene_bodies: &[RigidBody],
    interval: u32,
    steps: u32,
) -> ChurnResult {
    let n = scene_bodies.len();
    let mut logical_order: Vec<usize> = (0..n).collect();
    let mut physical: Vec<RigidBody> = scene_bodies.to_vec();
    let mut engine = build_direct(kind, &physical);
    for _ in 0..CHURN_WARM_STEPS {
        engine.step(DT);
    }

    let mut per_step = Vec::with_capacity(steps as usize);
    let mut reorder = Vec::new();
    let start = Instant::now();
    for step in 0..steps {
        // Keep the heap awake: a settled pile otherwise sleeps and the timed
        // window measures the fast path. The wake loop is outside the timed
        // region but identical for every reorder interval.
        for handle in 0..n {
            engine.wake_body(BodyHandle::from(handle));
        }
        let t = Instant::now();
        engine.step(DT);
        per_step.push(t.elapsed().as_secs_f64() * 1e9);

        if interval > 0 && (step + 1) % interval == 0 {
            let t = Instant::now();
            for (h, slot) in physical.iter_mut().enumerate() {
                if let Some(body) = engine.get_body(BodyHandle::from(h)) {
                    *slot = body.clone();
                }
            }
            let keys = morton_keys(&physical);
            let mut sort_order: Vec<usize> = (0..n).collect();
            sort_order.sort_unstable_by_key(|&i| (keys[i], i));
            let mut shuffled = Vec::with_capacity(n);
            let mut new_logical = Vec::with_capacity(n);
            for &old in &sort_order {
                shuffled.push(physical[old].clone());
                new_logical.push(logical_order[old]);
            }
            physical = shuffled;
            logical_order = new_logical;
            engine = build_direct(kind, &physical);
            reorder.push(t.elapsed().as_secs_f64() * 1e9);
        }
    }
    ChurnResult {
        per_step_ns: per_step,
        reorder_ns: reorder,
        total_ns: start.elapsed().as_secs_f64() * 1e9,
    }
}

struct Scene {
    name: &'static str,
    bodies: Vec<RigidBody>,
    measure_steps: u32,
}

fn floor(half_xz: f32) -> RigidBody {
    RigidBody::new_box(
        Vec3::new(0.0, -FLOOR_HALF_Y, 0.0),
        Vec3::new(half_xz, FLOOR_HALF_Y, half_xz),
        0.0,
    )
}

fn scene_islands_grid(g: u32) -> Vec<RigidBody> {
    let mut bodies = vec![floor(100.0)];
    let half = Vec3::splat(BOX_HALF);
    for gx in 0..g {
        for gz in 0..g {
            let x = (gx as f32 - g as f32 / 2.0) * PITCH;
            let z = (gz as f32 - g as f32 / 2.0) * PITCH;
            for level in 0..4 {
                bodies.push(RigidBody::new_box(
                    Vec3::new(x, REST_Y + 0.001 + level as f32 * LEVEL_PITCH, z),
                    half,
                    1.0,
                ));
            }
        }
    }
    bodies
}

fn scene_body_grid(n: u32) -> Vec<RigidBody> {
    let side = (n as f32).sqrt().ceil() as u32;
    let span = side as f32 * PITCH;
    let tiles = (span / (2.0 * TILE_HALF)).ceil() as i32;
    let mut bodies = Vec::new();
    for tx in 0..tiles {
        for tz in 0..tiles {
            let x = (tx as f32 - tiles as f32 / 2.0 + 0.5) * 2.0 * TILE_HALF;
            let z = (tz as f32 - tiles as f32 / 2.0 + 0.5) * 2.0 * TILE_HALF;
            bodies.push(RigidBody::new_box(
                Vec3::new(x, -FLOOR_HALF_Y, z),
                Vec3::new(TILE_HALF, FLOOR_HALF_Y, TILE_HALF),
                0.0,
            ));
        }
    }
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = (gx as f32 - side as f32 / 2.0) * PITCH;
        let z = (gz as f32 - side as f32 / 2.0) * PITCH;
        bodies.push(RigidBody::new_box(
            Vec3::new(x, REST_Y + 0.001, z),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    bodies
}

fn scene_stack(n: u32) -> Vec<RigidBody> {
    let mut bodies = vec![floor(10.0)];
    for level in 0..n {
        bodies.push(RigidBody::new_box(
            Vec3::new(0.0, REST_Y + 0.001 + level as f32 * LEVEL_PITCH, 0.0),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    bodies
}

fn scene_pile(n: u32) -> Vec<RigidBody> {
    let side = (n as f32).sqrt().ceil() as u32;
    let mut bodies = vec![floor(40.0)];
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = (gx as f32 - side as f32 / 2.0) * 1.1;
        let z = (gz as f32 - side as f32 / 2.0) * 1.1;
        let y = 3.0 + ((i * 37) % 97) as f32 * 0.11;
        bodies.push(RigidBody::new_box(
            Vec3::new(x, y, z),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    bodies
}

struct RawWriter {
    rows: String,
}

impl RawWriter {
    fn new() -> Self {
        Self {
            rows: String::from("engine,scene,bodies,layout,repeat,ns_per_step\n"),
        }
    }

    fn push(
        &mut self,
        engine: &str,
        scene: &str,
        bodies: usize,
        layout: &str,
        repeat: usize,
        ns: f64,
    ) {
        let _ = writeln!(
            self.rows,
            "{engine},{scene},{bodies},{layout},{repeat},{ns:.1}"
        );
    }
}

struct Diagnostics {
    kind: EngineKind,
    scene: &'static str,
    bodies: usize,
    islands: usize,
    contacts: usize,
    broad_phase_ms: f64,
    narrow_phase_ms: f64,
    solver_ms: f64,
    island_ms: f64,
    substeps: u32,
}

fn measure_diagnostics(kind: EngineKind, scene: &Scene, discovered: &Discovered) -> Diagnostics {
    let order: Vec<usize> = (0..scene.bodies.len()).collect();
    let mut islands = discovered.island_root.clone();
    islands.sort_unstable();
    islands.dedup();
    let mut diag = Diagnostics {
        kind,
        scene: scene.name,
        bodies: scene.bodies.len(),
        islands: islands.len(),
        contacts: discovered.contacts.len(),
        broad_phase_ms: 0.0,
        narrow_phase_ms: 0.0,
        solver_ms: 0.0,
        island_ms: 0.0,
        substeps: 0,
    };
    if kind == EngineKind::Si {
        let mut concrete = SequentialImpulseEngine::new(gravity());
        for &logical in &order {
            concrete.add_body(discovered.settled[logical].clone());
        }
        for _ in 0..REWARM_STEPS {
            concrete.step(DT);
        }
        for handle in 0..order.len() {
            concrete.wake_body(BodyHandle::from(handle));
        }
        concrete.step(DT);
        let timing = concrete.step_timing();
        diag.broad_phase_ms = timing.broad_phase_ms;
        diag.narrow_phase_ms = timing.narrow_phase_ms;
        diag.solver_ms = timing.solver_ms;
        diag.island_ms = timing.island_ms;
        diag.substeps = timing.substeps;
    }
    diag
}

fn main() {
    // SAFETY: called at process start before any Rayon worker exists; the
    // review requires a single-threaded measurement run.
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }

    let repeats: usize = std::env::var("ORDER_PROBE_REPEATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(REPEATS);
    let discovery_steps: u32 = std::env::var("ORDER_PROBE_DISCOVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DISCOVERY_STEPS);
    let measure_override: Option<u32> = std::env::var("ORDER_PROBE_MEASURE")
        .ok()
        .and_then(|v| v.parse().ok());
    let rewarm_steps: u32 = std::env::var("ORDER_PROBE_REWARM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(REWARM_STEPS);
    let churn_repeats: usize = std::env::var("ORDER_PROBE_CHURN_REPEATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CHURN_REPEATS);
    let run_churn = std::env::var("ORDER_PROBE_CHURN")
        .map(|v| v != "0")
        .unwrap_or(true);
    let scene_filter = std::env::var("ORDER_PROBE_SCENES").ok();
    let engine_filter = std::env::var("ORDER_PROBE_ENGINES").ok();
    let out_dir = std::path::PathBuf::from(
        std::env::var("ORDER_PROBE_OUT").unwrap_or_else(|_| "target/order_probe".into()),
    );
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let scenes = vec![
        Scene {
            name: "islands_grid_16x16",
            bodies: scene_islands_grid(16),
            measure_steps: MEASURE_STEPS_SMALL,
        },
        Scene {
            name: "small_grid_4x4",
            bodies: scene_islands_grid(4),
            measure_steps: MEASURE_STEPS_SMALL,
        },
        Scene {
            name: "body_grid_1000",
            bodies: scene_body_grid(1_000),
            measure_steps: MEASURE_STEPS_SMALL,
        },
        Scene {
            name: "body_grid_10000",
            bodies: scene_body_grid(10_000),
            measure_steps: MEASURE_STEPS_LARGE,
        },
        Scene {
            name: "stack_128_single_island",
            bodies: scene_stack(128),
            measure_steps: MEASURE_STEPS_LARGE,
        },
        Scene {
            name: "pile_2000_active",
            bodies: scene_pile(2_000),
            measure_steps: MEASURE_STEPS_SMALL,
        },
    ];

    let mut raw = RawWriter::new();
    let mut summary = String::from(
        "engine,scene,bodies,layout,repeats,median_ns,q1_ns,q3_ns,min_ns,max_ns,ratio_over_spatial,p_vs_spatial,ratio_over_shuffled_all,p_vs_shuffled_all\n",
    );
    let mut equivalence = String::from(
        "engine,scene,bodies,layout,max_position_delta_m,contacts,contact_delta_vs_spatial\n",
    );
    let mut ordering = String::from(
        "engine,scene,bodies,key_ns,comparison_sort_ns,radix_sort_ns,permute_ns,handle_indirection_ns_per_body\n",
    );
    let mut churn = String::from(
        "engine,scene,bodies,interval,repeats,median_step_ns,mean_step_ns,total_ms,reorder_count,median_reorder_ns\n",
    );
    let mut diagnostics = String::from(
        "engine,scene,bodies,islands,contacts,broad_phase_ms,narrow_phase_ms,solver_ms,island_ms,substeps\n",
    );

    for kind in [EngineKind::Si, EngineKind::Avbd] {
        if let Some(filter) = &engine_filter
            && !filter.split(',').any(|part| part.trim() == kind.name())
        {
            continue;
        }
        for scene in &scenes {
            if let Some(filter) = &scene_filter
                && !filter.split(',').any(|part| scene.name == part.trim())
            {
                continue;
            }
            let measure_steps = measure_override.unwrap_or(scene.measure_steps);
            let discovered = discover(kind, &scene.bodies, discovery_steps);
            if std::env::var("ORDER_PROBE_DEBUG").is_ok() {
                let mean_y: f32 = discovered
                    .settled
                    .iter()
                    .map(|body| body.position.y)
                    .sum::<f32>()
                    / discovered.settled.len() as f32;
                eprintln!(
                    "debug {}/{} settled_mean_y={mean_y:.3} contacts={}",
                    kind.name(),
                    scene.name,
                    discovered.contacts.len()
                );
            }
            let diag = measure_diagnostics(kind, scene, &discovered);
            let _ = writeln!(
                diagnostics,
                "{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{}",
                diag.kind.name(),
                diag.scene,
                diag.bodies,
                diag.islands,
                diag.contacts,
                diag.broad_phase_ms,
                diag.narrow_phase_ms,
                diag.solver_ms,
                diag.island_ms,
                diag.substeps
            );

            let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(repeats); LAYOUTS.len()];
            for repeat in 0..repeats {
                let seed = 0x9E37_79B9_7F4A_7C15u64
                    .wrapping_add((repeat as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
                for (slot, layout) in LAYOUTS.iter().enumerate() {
                    let order = layout_order(*layout, &discovered, &scene.bodies, seed);
                    let ns = timed_run(kind, &discovered, &order, measure_steps, rewarm_steps);
                    raw.push(
                        kind.name(),
                        scene.name,
                        scene.bodies.len(),
                        layout.name(),
                        repeat,
                        ns,
                    );
                    samples[slot].push(ns);
                }
            }

            let mut medians = Vec::with_capacity(LAYOUTS.len());
            let mut iqrs = Vec::with_capacity(LAYOUTS.len());
            let mut mins = Vec::with_capacity(LAYOUTS.len());
            let mut maxs = Vec::with_capacity(LAYOUTS.len());
            for sample in &mut samples {
                let (median, q1, q3, min, max) = median_iqr(sample);
                medians.push(median);
                iqrs.push((q1, q3));
                mins.push(min);
                maxs.push(max);
            }
            let spatial_median = medians[0];
            let shuffled_median = medians[1];
            for (slot, layout) in LAYOUTS.iter().enumerate() {
                let (q1, q3) = iqrs[slot];
                let p_spatial = if slot == 0 {
                    1.0
                } else {
                    wilcoxon_p(&samples[slot], &samples[0])
                };
                let p_shuffled = if slot == 0 || slot == 1 {
                    1.0
                } else {
                    wilcoxon_p(&samples[slot], &samples[1])
                };
                let _ = writeln!(
                    summary,
                    "{},{},{},{},{},{:.0},{:.0},{:.0},{:.0},{:.0},{:.3},{:.4},{:.3},{:.4}",
                    kind.name(),
                    scene.name,
                    scene.bodies.len(),
                    layout.name(),
                    repeats,
                    medians[slot],
                    q1,
                    q3,
                    mins[slot],
                    maxs[slot],
                    medians[slot] / spatial_median,
                    p_spatial,
                    medians[slot] / shuffled_median,
                    p_shuffled
                );
            }

            let spatial_equiv = equivalence_run(
                kind,
                &discovered,
                &(0..scene.bodies.len()).collect::<Vec<_>>(),
                measure_steps,
                rewarm_steps,
            );
            for layout in LAYOUTS {
                let order = layout_order(layout, &discovered, &scene.bodies, 0xDEAD_BEEF);
                let equiv = equivalence_run(kind, &discovered, &order, measure_steps, rewarm_steps);
                let max_delta = equiv
                    .final_positions
                    .iter()
                    .zip(&spatial_equiv.final_positions)
                    .map(|(a, b)| (*a - *b).length() as f64)
                    .fold(0.0, f64::max);
                let _ = writeln!(
                    equivalence,
                    "{},{},{},{},{:.6},{},{}",
                    kind.name(),
                    scene.name,
                    scene.bodies.len(),
                    layout.name(),
                    max_delta,
                    equiv.contacts,
                    equiv.contacts as i64 - spatial_equiv.contacts as i64
                );
            }

            let cost = measure_ordering_cost(&discovered, 30);
            let _ = writeln!(
                ordering,
                "{},{},{},{:.0},{:.0},{:.0},{:.0},{:.2}",
                kind.name(),
                scene.name,
                scene.bodies.len(),
                cost.key_ns,
                cost.comparison_sort_ns,
                cost.radix_sort_ns,
                cost.permute_ns,
                cost.handle_indirection_ns_per_body
            );

            eprintln!(
                "{}/{} done ({} bodies, {} islands, {} contacts)",
                kind.name(),
                scene.name,
                scene.bodies.len(),
                diag.islands,
                diag.contacts
            );
        }

        if !run_churn {
            continue;
        }
        // Long churn on the pile only (review item 5).
        let pile = scenes
            .iter()
            .find(|scene| scene.name == "pile_2000_active")
            .expect("pile scene");
        let mut per_interval: Vec<(u32, Vec<ChurnResult>)> = [0u32, 1, 10, 60]
            .into_iter()
            .map(|i| (i, Vec::new()))
            .collect();
        for (interval, results) in per_interval.iter_mut() {
            for repeat in 0..churn_repeats {
                let result = churn_run(kind, &pile.bodies, *interval, CHURN_STEPS);
                results.push(result);
                eprintln!(
                    "{}/churn interval={} repeat={}/{}",
                    kind.name(),
                    interval,
                    repeat + 1,
                    churn_repeats
                );
            }
        }
        for (interval, results) in &per_interval {
            let mut medians = Vec::with_capacity(results.len());
            let mut means = Vec::with_capacity(results.len());
            let mut totals = Vec::with_capacity(results.len());
            let mut reorder_medians = Vec::with_capacity(results.len());
            let mut reorder_count = 0;
            for result in results {
                let (median, _, _, _, _) = median_iqr(&mut result.per_step_ns.clone());
                medians.push(median);
                let mean = result.per_step_ns.iter().sum::<f64>() / result.per_step_ns.len() as f64;
                means.push(mean);
                totals.push(result.total_ns);
                reorder_count = result.reorder_ns.len();
                if !result.reorder_ns.is_empty() {
                    let (m, _, _, _, _) = median_iqr(&mut result.reorder_ns.clone());
                    reorder_medians.push(m);
                }
            }
            let (median, _, _, _, _) = median_iqr(&mut medians);
            let (mean, _, _, _, _) = median_iqr(&mut means);
            let (total, _, _, _, _) = median_iqr(&mut totals);
            let reorder = if reorder_medians.is_empty() {
                0.0
            } else {
                median_iqr(&mut reorder_medians).0
            };
            let _ = writeln!(
                churn,
                "{},{},{},{},{},{:.0},{:.0},{:.1},{},{}",
                kind.name(),
                pile.name,
                pile.bodies.len(),
                interval,
                churn_repeats,
                median,
                mean,
                total / 1e6,
                reorder_count,
                reorder
            );
        }
    }

    std::fs::write(out_dir.join("raw.csv"), raw.rows).expect("write raw.csv");
    std::fs::write(out_dir.join("summary.csv"), summary).expect("write summary.csv");
    std::fs::write(out_dir.join("equivalence.csv"), equivalence).expect("write equivalence.csv");
    std::fs::write(out_dir.join("ordering_cost.csv"), ordering).expect("write ordering_cost.csv");
    std::fs::write(out_dir.join("churn.csv"), churn).expect("write churn.csv");
    std::fs::write(out_dir.join("diagnostics.csv"), diagnostics).expect("write diagnostics.csv");
    println!(
        "wrote {}/raw.csv, summary.csv, equivalence.csv, ordering_cost.csv, churn.csv, diagnostics.csv",
        out_dir.display()
    );
}
