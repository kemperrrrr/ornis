//! Scheduler frontend parity (backlog #19, anti-drift canon): the same
//! access topology on `ornis_core::Schedule` (systems, TypeId keys — resources
//! and lanes) and on `ornis_render::SystemSet` (passes, ResourceId keys) must
//! produce bitwise identical levels: both consumers are considered a single
//! `ornis-schedule` engine. Semantic drift on either side = red CI.

use ornis_core::{Resources, Schedule, System, SystemAccess};
use ornis_render::{
    FrameResource, ProjectionError, ResourceId, ResourceKind, SizePolicy, SystemSet, TextureSpec,
    try_project_schedule,
};

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
    // (`SystemSet::build`): a resource whose first touch is a read without
    // an earlier (including own) write must be an import.
    // Imports do not affect levels (out-of-pool, not access semantics),
    // core side has no such rule — mirror honestly.
    const RES_NAMES: [&str; 8] = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"];
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
    let mut plan = SystemSet::new();
    plan.set_surface_size((640, 480));
    let ids: Vec<ResourceId> = (0..8)
        .map(|i| {
            if import[i] {
                plan.import_resource(RES_NAMES[i], spec)
            } else {
                plan.create_resource(RES_NAMES[i], spec)
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

// ── E1 (S5e): passes projected as core `Schedule` systems ─────────────
//
// The gate of the E1 decomposition step: `try_project_schedule` mirrors a
// typed-registered `SystemSet` into a core `Schedule` whose levels must
// equal `FrameLayout::levels()` bitwise — the same anti-drift contract as
// above, now through the pass-as-system adapter (`schedule_bridge`).

/// Typed frame resources for the projection (the bridge keys core
/// accesses by `FrameResource` `TypeId`s, so registration must be typed).
macro_rules! typed_resources {
    ($($r:ident => $name:literal),+ $(,)?) => {
        $(
            struct $r;
            impl FrameResource for $r {
                const NAME: &'static str = $name;
                fn kind() -> ResourceKind {
                    ResourceKind::FrameOwned
                }
                fn spec(_: wgpu::TextureFormat) -> TextureSpec {
                    spec()
                }
            }
        )+
    };
}

typed_resources!(
    R0 => "r0", R1 => "r1", R2 => "r2", R3 => "r3",
    R4 => "r4", R5 => "r5", R6 => "r6", R7 => "r7",
);

/// Builds a typed-registered plan for one topology and returns
/// (projected core Schedule levels, frame layout levels).
fn projected_levels(
    reads: &[Vec<usize>],
    writes: &[Vec<usize>],
    edges: &[(&'static str, &'static str)],
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    const NAMES: [&str; 6] = ["s0", "s1", "s2", "s3", "s4", "s5"];
    assert_eq!(reads.len(), writes.len(), "parallel access slices");
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut plan = SystemSet::new();
    plan.set_surface_size((640, 480));
    let ids = [
        plan.register_resource::<R0>(format),
        plan.register_resource::<R1>(format),
        plan.register_resource::<R2>(format),
        plan.register_resource::<R3>(format),
        plan.register_resource::<R4>(format),
        plan.register_resource::<R5>(format),
        plan.register_resource::<R6>(format),
        plan.register_resource::<R7>(format),
    ];
    for i in 0..reads.len() {
        let mut pass = plan.add_pass(NAMES[i]);
        for &k in &reads[i] {
            pass = pass.read(ids[k]);
        }
        for &k in &writes[i] {
            pass = pass.write(ids[k]);
        }
    }
    for &(before, after) in edges {
        plan.order_before_named(before, after);
    }
    let schedule = try_project_schedule(&plan).expect("typed registration projects");
    (schedule.levels(), plan.build().levels())
}

/// E1 gate: adapter levels == `FrameLayout::levels()` on representative
/// topologies (chains, shared levels, explicit-edge splits).
#[test]
fn projected_pass_system_levels_match_layout_levels() {
    // Chain: RaW/WaW dependencies serialize level by level.
    let (projected, layout) = projected_levels(
        &[vec![], vec![0], vec![1]],
        &[vec![0], vec![1], vec![2]],
        &[],
    );
    assert_eq!(projected, layout, "chain: adapter levels != layout levels");
    assert_eq!(projected, vec![vec![0], vec![1], vec![2]]);

    // Independent writers share a level; the reader closes the frame.
    let (projected, layout) = projected_levels(
        &[vec![], vec![], vec![0, 1]],
        &[vec![0], vec![1], vec![2]],
        &[],
    );
    assert_eq!(
        projected, layout,
        "shared level: adapter levels != layout levels"
    );
    assert_eq!(projected, vec![vec![0, 1], vec![2]]);

    // An explicit edge splits a shared level on both sides.
    let (projected, layout) =
        projected_levels(&[vec![], vec![]], &[vec![0], vec![1]], &[("s0", "s1")]);
    assert_eq!(
        projected, layout,
        "edge split: adapter levels != layout levels"
    );
    assert_eq!(projected, vec![vec![0], vec![1]]);
}

/// Disabled passes drop out of the projection together with their edges —
/// mirroring `TransientPool::layout_levels` exactly.
#[test]
fn projection_skips_disabled_passes_and_their_edges() {
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut plan = SystemSet::new();
    plan.set_surface_size((640, 480));
    let r0 = plan.register_resource::<R0>(format);
    let r1 = plan.register_resource::<R1>(format);
    let mid = plan.add_pass("mid").write(r0).id();
    let last = plan.add_pass("last").write(r1).id();
    plan.order_before(mid, last);
    plan.set_pass_enabled(mid, false);

    let schedule = try_project_schedule(&plan).expect("typed registration projects");
    assert_eq!(schedule.levels(), vec![vec![0]]);
    assert_eq!(
        schedule.levels(),
        plan.build().levels(),
        "disabled pass culling must match the layout"
    );
}

/// The projection is honest about what it cannot mirror: resources
/// declared without a typed identity have no core `TypeId` to project.
#[test]
fn projection_rejects_untyped_resources() {
    let mut plan = SystemSet::new();
    plan.set_surface_size((640, 480));
    let untyped = plan.create_resource("untyped", spec());
    plan.add_pass("user").read(untyped);

    let error = try_project_schedule(&plan).unwrap_err();
    assert_eq!(
        error,
        ProjectionError::UntypedResource {
            pass: "user",
            resource: untyped,
        }
    );
}
