//! Scheduler frontend parity (backlog #19, anti-drift canon): the same
//! access topology on `ornis_core::Schedule` (systems, TypeId keys — resources
//! and lanes) and on `ornis_render::FramePlan` (passes, ResourceId keys) must
//! produce bitwise identical levels: both consumers are considered a single
//! `ornis-schedule` engine. Semantic drift on either side = red CI.

use ornis_core::{Resources, Schedule, System, SystemAccess};
use ornis_render::{FramePlan, ResourceId, SizePolicy, TextureSpec};

/// Core resource namespace keys (marker types in this file,
/// real singleton resources are not needed for the plan).
struct K0;
struct K1;
struct K2;
struct K3;

/// `SmartStore` lane keys — second core namespace; render has no
/// analogue: the render namespace (ResourceId) is single, keys
/// 4..8 simply map to its elements.
struct L0;
struct L1;
struct L2;
struct L3;

/// Stub system: parity only needs name and accesses.
struct Stub(&'static str, SystemAccess);

impl System for Stub {
    fn name(&self) -> &'static str {
        self.0
    }

    fn access(&self) -> SystemAccess {
        self.1.clone()
    }

    fn run(&self, _: &Resources) {}
}

fn spec() -> TextureSpec {
    TextureSpec {
        format: wgpu::TextureFormat::Rgba8Unorm,
        samples: 1,
        size: SizePolicy::Fixed {
            width: 4,
            height: 4,
        },
    }
}

/// Key 0..8 → core declaration: 0..4 — resource namespace,
/// 4..8 — lanes (mirrors r0..r7 elements on the render side).
fn push_access(access: SystemAccess, key: usize, write: bool) -> SystemAccess {
    match (key, write) {
        (0, false) => access.reads::<K0>(),
        (0, true) => access.writes::<K0>(),
        (1, false) => access.reads::<K1>(),
        (1, true) => access.writes::<K1>(),
        (2, false) => access.reads::<K2>(),
        (2, true) => access.writes::<K2>(),
        (3, false) => access.reads::<K3>(),
        (3, true) => access.writes::<K3>(),
        (4, false) => access.reads_lane::<L0>(),
        (4, true) => access.writes_lane::<L0>(),
        (5, false) => access.reads_lane::<L1>(),
        (5, true) => access.writes_lane::<L1>(),
        (6, false) => access.reads_lane::<L2>(),
        (6, true) => access.writes_lane::<L2>(),
        (7, false) => access.reads_lane::<L3>(),
        (7, true) => access.writes_lane::<L3>(),
        (_, _) => unreachable!("key space is 0..8"),
    }
}

/// Builds mirrored plans for one `reads`/`writes` topology (keys 0..8)
/// on both frontends and applies explicit named edges; returns
/// levels (core, render).
fn mirrored_levels(
    reads: &[Vec<usize>],
    writes: &[Vec<usize>],
    edges: &[(&'static str, &'static str)],
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    const NAMES: [&str; 12] = [
        "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    ];
    assert_eq!(reads.len(), writes.len(), "parallel access slices");
    // Render domain rule "first touch must be a write"
    // (`FramePlan::build`): a resource whose first touch is a read without
    // an earlier (including own) write must be an import.
    // Imports do not affect levels (out-of-pool, not access semantics),
    // core side has no such rule — mirror honestly.
    let mut import = [false; 8];
    for (key, slot) in import.iter_mut().enumerate() {
        let first_use =
            (0..reads.len()).find(|&i| reads[i].contains(&key) || writes[i].contains(&key));
        if let Some(i) = first_use {
            let written = (0..=i).any(|j| writes[j].contains(&key));
            if reads[i].contains(&key) && !written {
                *slot = true;
            }
        }
    }
    let spec = spec();
    let mut sched = Schedule::new();
    let mut plan = FramePlan::new((640, 480));
    let ids: Vec<ResourceId> = (0..8)
        .map(|i| {
            if import[i] {
                plan.import_resource(format!("r{i}"), spec)
            } else {
                plan.create_resource(format!("r{i}"), spec)
            }
        })
        .collect();
    for i in 0..reads.len() {
        let mut access = SystemAccess::new();
        for &k in &reads[i] {
            access = push_access(access, k, false);
        }
        for &k in &writes[i] {
            access = push_access(access, k, true);
        }
        sched.add_system(Stub(NAMES[i], access));
        let mut pass = plan.add_pass(NAMES[i]);
        for &k in &reads[i] {
            pass = pass.read(ids[k]);
        }
        for &k in &writes[i] {
            pass = pass.write(ids[k]);
        }
    }
    for &(before, after) in edges {
        sched.order_before(before, after);
        plan.order_before_named(before, after);
    }
    (sched.levels(), plan.build().levels())
}

/// Basic conflict classes and explicit edges: frontend levels identical.
#[test]
fn fixed_topologies_match_across_frontends() {
    // Independent writers → one level.
    let (core, render) =
        mirrored_levels(&[vec![], vec![], vec![]], &[vec![0], vec![1], vec![2]], &[]);
    assert_eq!(core, vec![vec![0, 1, 2]]);
    assert_eq!(core, render);

    // RaW chain.
    let (core, render) = mirrored_levels(
        &[vec![], vec![0], vec![1]],
        &[vec![0], vec![1], vec![]],
        &[],
    );
    assert_eq!(core, vec![vec![0], vec![1], vec![2]]);
    assert_eq!(core, render);

    // WaR (anti-dependency): reader first; key from core lane
    // namespace (7 → r7 on render side).
    let (core, render) = mirrored_levels(&[vec![7], vec![]], &[vec![], vec![7]], &[]);
    assert_eq!(core, vec![vec![0], vec![1]]);
    assert_eq!(core, render);

    // Lanes (4..8) and resources (0..4) — separate namespaces:
    // resource writer and lane reader do not conflict.
    let (core, render) = mirrored_levels(&[vec![], vec![4]], &[vec![0], vec![]], &[]);
    assert_eq!(core, vec![vec![0, 1]]);
    assert_eq!(core, render);

    // Explicit edge splits a shared level on both frontends.
    let (core, render) = mirrored_levels(&[vec![], vec![]], &[vec![0], vec![1]], &[("s0", "s1")]);
    assert_eq!(core, vec![vec![0], vec![1]]);
    assert_eq!(core, render);
}

/// Differential parity: pseudo-random access slices over 8 keys
/// (both core namespaces) with and without explicit edges — frontend
/// levels must match bitwise.
#[test]
fn lcg_scenarios_match_across_frontends() {
    let mut lcg = 0x9E37_79B9u64;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        lcg
    };
    let mut reads: Vec<Vec<usize>> = Vec::new();
    let mut writes: Vec<Vec<usize>> = Vec::new();
    for _ in 0..12 {
        let mut r = Vec::new();
        let mut w = Vec::new();
        for key in 0..8 {
            if next() % 5 == 0 {
                r.push(key);
            }
            if next() % 5 == 0 {
                w.push(key);
            }
        }
        reads.push(r);
        writes.push(w);
    }
    let (core, render) = mirrored_levels(&reads, &writes, &[]);
    assert_eq!(core, render, "parity without explicit edges");
    let (core, render) = mirrored_levels(&reads, &writes, &[("s2", "s8"), ("s0", "s11")]);
    assert_eq!(core, render, "parity with explicit edges");
}
