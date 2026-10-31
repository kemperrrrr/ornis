//! Dispatch the lowered `cull` kernel and A/B it against a hand twin of
//! the production `sphere_visible` test.
//!
//! The kernel flags eight sphere slots — six guarded by `count` — against
//! six planes: a clear pass, a clear cut, a `dot + r == 0` boundary, a
//! NaN radius that fails open like the production WGSL, a cut by a second
//! plane, and two sentinel slots past `count`. Both shaders run on the
//! same bytes: the outputs must match [`EXPECTED`], [`derived_flags`]
//! re-derives that constant from the production expression
//! (`render/src/gpu_culling.rs`), and the line this test prints shows
//! both GPU times from the CI log.

use wasm2gpu_testkit::cull::{
    ACTIVE_COUNT, EXPECTED, LANES, PLANES, RUNS, SENTINEL, SLOTS, SPHERES, WORKGROUPS,
    lowered_kernel, run,
};
use wasm2gpu_testkit::device;

/// The hand twin runs the same guard, the same `dot + w + radius < 0.0`
/// order, and the same workgroup, so bitwise agreement is the contract.
const TWIN_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> planes: array<vec4<f32>, 24>;
@group(0) @binding(1) var<storage, read_write> spheres: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> sphere_count: array<u32, 4>;

@compute @workgroup_size(4, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id < sphere_count[0]) {
        let center = spheres[id];
        var visible = 1u;
        var plane = 0u;
        while (plane < 6u) {
            let at = plane * 4u;
            let px = planes[at].x;
            let py = planes[at + 1u].x;
            let pz = planes[at + 2u].x;
            let pw = planes[at + 3u].x;
            if (px * center.x + py * center.y + pz * center.z + pw + center.w < 0.0) {
                visible = 0u;
            }
            plane += 1u;
        }
        flags[id] = visible;
    }
}
"#;

/// The production `sphere_visible` from `render/src/gpu_culling.rs`,
/// expression for expression: `dot(plane.xyz, center) + plane.w < -radius`
/// with the same early exit. The non-finite guards are CPU-only, so a NaN
/// radius fails open here — the semantics [`EXPECTED`] pins.
fn production_visible(center: [f32; 4]) -> bool {
    let radius = center[3];
    let mut plane = 0usize;
    while plane < PLANES.len() {
        let at = PLANES[plane];
        let dot = at[0] * center[0] + at[1] * center[1] + at[2] * center[2];
        if dot + at[3] < -radius {
            return false;
        }
        plane += 1;
    }
    true
}

/// Flags the production expression writes for each slot: visible up to
/// [`ACTIVE_COUNT`], then the sentinel the count guard must not touch.
fn derived_flags() -> [u32; SLOTS] {
    let mut flags = [SENTINEL; SLOTS];
    let mut slot = 0usize;
    while slot < ACTIVE_COUNT {
        flags[slot] = u32::from(production_visible(SPHERES[slot]));
        slot += 1;
    }
    flags
}

#[test]
fn cull_kernel_matches_the_hand_twin() {
    assert_eq!(LANES * WORKGROUPS as usize, SPHERES.len());
    let wgsl = lowered_kernel();
    let derived = derived_flags();
    assert_eq!(EXPECTED, derived, "production-formula cross-check");
    match device() {
        Some((device, queue)) => {
            let (flags, wasm_time) = run(&device, &queue, &wgsl);
            assert_eq!(flags, EXPECTED, "wasm2gpu kernel output");
            let (twin, twin_time) = run(&device, &queue, TWIN_WGSL);
            assert_eq!(twin, EXPECTED, "hand twin output");
            assert_eq!(flags, twin, "bitwise A/B");
            eprintln!("cull {flags:?}: wasm {wasm_time:?} twin {twin_time:?} x{RUNS}");
        }
        None if std::env::var_os("ORNIS_REQUIRE_GPU").is_some() => {
            panic!("ORNIS_REQUIRE_GPU is set but no wgpu adapter was found");
        }
        None => eprintln!("cull_kernel_matches_the_hand_twin: no wgpu adapter — skipped"),
    }
}
