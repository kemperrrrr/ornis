//! Integration tests for `#[stage]`: entry wrapper emission, parameter
//! interface attributes, `returns` for located value returns, and
//! struct-literal constructors. The input functions are DSL-only — `Foo`,
//! `QUAD` and friends intentionally do not exist as Rust items. Renamed
//! mirrors are spelled through Rust import aliases (see below), so the macro
//! itself needs no name mapping.

use ornis_macros::stage;

#[allow(dead_code)]
struct RustMirror {
    _x: u32,
}
// Only referenced inside DSL-only stage bodies (dropped before codegen).
#[allow(unused_imports)]
use RustMirror as WgslName;

/// Quad-vertex shape: builtin param, constructor call, aliased mirror.
#[stage(vertex, entry = "vs_main")]
fn quad_vs_entry(#[wgsl(builtin = "vertex_index")] idx: u32) -> WgslName {
    return WgslName { x: QUAD[idx] };
}

/// Fragment shape: location param, located value return.
#[stage(fragment, entry = "fs_main", returns = "@location(0) vec4<f32>")]
fn quad_fs_entry(#[wgsl(location = 0)] uv: Vec2) -> Vec4 {
    return Vec4(uv.x, uv.y, 0.0, 1.0);
}

#[test]
fn stage_vertex_entry_shape() {
    let src = quad_vs_entry::wgsl_source();
    assert!(
        src.starts_with("@vertex\nfn vs_main(@builtin(vertex_index) idx: u32)"),
        "{src}"
    );
    assert!(src.contains("-> WgslName"), "{src}");
    assert!(src.contains("return WgslName(QUAD[idx]) /* x */;"), "{src}");
    assert_eq!(quad_vs_entry::entry_point(), "vs_main");
}

#[test]
fn stage_fragment_location_param() {
    let src = quad_fs_entry::wgsl_source();
    assert!(
        src.starts_with("@fragment\nfn fs_main(@location(0) uv: vec2<f32>)"),
        "{src}"
    );
    assert!(src.contains("-> @location(0) vec4<f32>"), "{src}");
}
