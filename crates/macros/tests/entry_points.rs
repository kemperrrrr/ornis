//! Compile-test for all public macro entry points.
//!
//! One positive expansion per public entry point (`#[stage]`,
//! `#[gpu_pipeline]` (legacy + full-shader), `#[wgsl_fn]`,
//! `#[derive(WgslStruct)]` / `#[derive(WgslInterface)]`,
//! `#[smart_pipeline]`), asserting the generated surface
//! (`wgsl_source`, `pipeline_label`, `entry_point`, layout constants)
//! and naga-validating the complete compute/helper modules.
//! Error paths stay in `tests/ui` (trybuild).

use ornis_core::SmartStore;
use ornis_macros::{WgslInterface, WgslStruct, gpu_pipeline, smart_pipeline, stage, wgsl_fn};

// ── #[stage] ─────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct EntryMirror {
    _x: u32,
}
#[allow(unused_imports)]
use EntryMirror as EntryOut;

/// Vertex entry: builtin param, constructor call, aliased mirror.
#[stage(vertex, entry = "vs_main")]
fn entry_vs(#[wgsl(builtin = "vertex_index")] idx: u32) -> EntryOut {
    return EntryOut { x: QUAD[idx] };
}

/// Fragment entry: location param, located value return.
#[stage(fragment, entry = "fs_main", returns = "@location(0) vec4<f32>")]
fn entry_fs(#[wgsl(location = 0)] uv: Vec2) -> Vec4 {
    return Vec4(uv.x, uv.y, 0.0, 1.0);
}

#[test]
fn stage_entries_expose_wgsl_sources() {
    let vs = entry_vs::wgsl_source();
    assert!(
        vs.starts_with("@vertex\nfn vs_main(@builtin(vertex_index) idx: u32)"),
        "{vs}"
    );
    assert_eq!(entry_vs::entry_point(), "vs_main");
    let fs = entry_fs::wgsl_source();
    assert!(
        fs.starts_with("@fragment\nfn fs_main(@location(0) uv: vec2<f32>)"),
        "{fs}"
    );
    assert!(fs.contains("-> @location(0) vec4<f32>"), "{fs}");
}

// ── #[wgsl_fn] ───────────────────────────────────────────────────────────

/// Plain helper: no stage wrapper, passthrough builtin call.
#[wgsl_fn]
fn entry_helper(n: Vec3, k: f32) -> Vec3 {
    let base = Vec3::new(1.0);
    return mix(base, n, k);
}

#[test]
fn wgsl_fn_helper_shape() {
    let src = entry_helper::wgsl_source();
    assert!(
        src.starts_with("fn entry_helper(n: vec3<f32>, k: f32) -> vec3<f32>"),
        "{src}"
    );
    assert!(
        !src.contains("@vertex") && !src.contains("@fragment"),
        "{src}"
    );
}

// ── WgslStruct / WgslInterface ────────────────────────────────────────────

/// Two vec3s with explicit WGSL padding.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, WgslStruct)]
struct EntryBody {
    pos: [f32; 3],
    pad0: f32,
    vel: [f32; 3],
    pad1: f32,
}

/// Builtin position plus one UV location.
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "EntryVarying")]
#[allow(dead_code)]
struct EntryVarying {
    #[wgsl(builtin = "position")]
    clip: [f32; 4],
    #[wgsl(location = 0)]
    uv: [f32; 2],
}

#[test]
fn wgsl_struct_and_interface_sources() {
    assert!(EntryBody::WGSL_SOURCE.contains("struct EntryBody"));
    assert!(EntryBody::WGSL_SOURCE.contains("pos: vec3<f32>"));
    assert_eq!(std::mem::size_of::<EntryBody>(), 32);
    let src = EntryVarying::WGSL_SOURCE;
    assert!(src.contains("struct EntryVarying"), "{src}");
    assert!(src.contains("@builtin(position) clip: vec4<f32>"), "{src}");
    assert!(src.contains("@location(0) uv: vec2<f32>"), "{src}");
}

// ── #[gpu_pipeline] ──────────────────────────────────────────────────────

/// Legacy mode: params become storage buffers, tail expr is the output.
#[gpu_pipeline]
fn entry_add(a: f32, b: f32) -> f32 {
    a + b * 2.0
}

/// Full-shader mode: bindings + workgroup size + builtin in the attribute,
/// the body is the compute entry body.
#[gpu_pipeline(
    workgroup_size = 4,
    storage(buf: [f32; 64], read_write),
    uniform(params: [u32; 4]),
    builtin(gid: workgroup_id),
)]
fn entry_scale() {
    let i = gid.x;
    if i < params.x {
        buf[i] = buf[i] * 2.0;
    }
}

fn validate_module(name: &str, source: &str) {
    let module =
        naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("{name} must parse: {e}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{name} must validate: {e:?}"));
}

#[test]
fn gpu_pipeline_legacy_and_full_shader_modes() {
    let legacy = entry_add::wgsl_source();
    assert!(legacy.contains("storage"), "{legacy}");
    assert!(legacy.contains("a + b * 2.0"), "{legacy}");
    assert_eq!(entry_add::pipeline_label(), "entry_add");

    let full = entry_scale::wgsl_source();
    assert!(
        full.contains("@group(0) @binding(0) var<storage, read_write> buf: array<f32>;"),
        "{full}"
    );
    assert!(
        full.contains("@group(0) @binding(1) var<uniform> params: vec4<u32>;"),
        "{full}"
    );
    assert!(full.contains("@compute @workgroup_size(4)"), "{full}");
    assert_eq!(entry_scale::pipeline_label(), "entry_scale");
    validate_module("entry_scale compute module", full);
    validate_module("entry_helper helper", entry_helper::wgsl_source());
}

// ── #[smart_pipeline] ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct EntryPos {
    x: f32,
}

#[derive(Debug, Clone)]
struct EntryVel {
    x: f32,
}

#[smart_pipeline]
fn entry_integrate(store: &SmartStore, dt: f32) -> usize {
    let mut positions = store.write_lane::<EntryPos>().expect("pos lane");
    let velocities = store.read_lane::<EntryVel>().expect("vel lane");
    for (pos, vel) in positions.iter_mut().zip(velocities.iter()) {
        pos.x += vel.x * dt;
    }
    positions.len()
}

/// PLAN m, R3 (variant a): `#[smart_pipeline]` also understands a store
/// alias bound from `resources.get::<SmartStore>()`.
#[smart_pipeline]
fn entry_over_resources(resources: &ornis_core::Resources) -> usize {
    let store = resources.get::<SmartStore>().expect("store");
    let mut positions = store.write_lane::<EntryPos>().expect("pos");
    for pos in positions.iter_mut() {
        pos.x += 1.0;
    }
    positions.len()
}

#[test]
fn smart_pipeline_over_resources_alias() {
    let mut store = SmartStore::new();
    store.register::<EntryPos>();
    for _ in 0..4 {
        let e = store.create_entity();
        store.insert(e, EntryPos { x: 0.0 });
    }
    let mut resources = ornis_core::Resources::new();
    resources.insert(store);
    assert_eq!(entry_over_resources(&resources), 4);
    assert_eq!(
        __SMART_PIPELINE_ACCESS_entry_over_resources,
        &[("EntryPos", true)]
    );
    let store = resources.get::<SmartStore>().expect("store");
    let lane = store.read_lane::<EntryPos>().expect("pos");
    for pos in lane.iter() {
        assert_eq!(pos.x, 1.0);
    }
}

#[test]
fn smart_pipeline_entry_compiles_and_runs() {
    let mut store = SmartStore::new();
    store.register::<EntryPos>();
    store.register::<EntryVel>();
    for _ in 0..4 {
        let e = store.create_entity();
        store.insert(e, EntryPos { x: 0.0 });
        store.insert(e, EntryVel { x: 2.0 });
    }
    assert_eq!(entry_integrate(&store, 0.5), 4);
    let lane = store.read_lane::<EntryPos>().expect("pos lane");
    for pos in lane.iter() {
        assert_eq!(pos.x, 1.0);
    }
}

/// R3: the macro exports the access set derived from the lane bindings as a
/// sorted `(type, is_write)` constant next to the function.
#[test]
fn smart_pipeline_access_const_matches_lane_bindings() {
    // Sorted by type name; a write covers a read of the same lane.
    assert_eq!(
        __SMART_PIPELINE_ACCESS_entry_integrate,
        &[("EntryPos", true), ("EntryVel", false)]
    );
}

#[smart_pipeline]
fn entry_readonly(store: &SmartStore) -> usize {
    let velocities = store.read_lane::<EntryVel>().expect("vel lane");
    velocities.len()
}

/// R3: a read-only single lane exports one read entry.
#[test]
fn smart_pipeline_access_const_single_read_lane() {
    assert_eq!(
        __SMART_PIPELINE_ACCESS_entry_readonly,
        &[("EntryVel", false)]
    );
}

#[smart_pipeline]
fn entry_no_lanes(dt: f32) -> f32 {
    dt * 2.0
}

/// R3: no lane bindings export an empty access set (still deterministic).
#[test]
fn smart_pipeline_access_const_empty_without_lanes() {
    assert!(__SMART_PIPELINE_ACCESS_entry_no_lanes.is_empty());
    assert_eq!(entry_no_lanes(21.0), 42.0);
}
