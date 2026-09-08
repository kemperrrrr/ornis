//! Integration tests for `#[stage]`: entry wrapper emission, parameter
//! interface attributes, `returns` rename (including constructor calls) and
//! struct-literal constructors. The input functions are DSL-only — `Foo`,
//! `QUAD` and friends intentionally do not exist as Rust items.

use ornis_macros::stage;

/// Quad-vertex shape: builtin param, renamed mirror return + constructor.
#[stage(vertex, entry = "vs_main", returns = "CompositeVertexOutput")]
fn composite_vertex_entry(#[wgsl(builtin = "vertex_index")] idx: u32) -> HdrVertexOutput {
    return HdrVertexOutput {
        clip_position: QUAD[idx],
        uv: UVS[idx],
    };
}

/// Fragment shape: location param, verbatim mirror return.
#[stage(fragment, entry = "fs_main")]
fn composite_fragment_entry(#[wgsl(location = 0)] uv: Vec2) -> Vec4 {
    return Vec4(uv.x, uv.y, 0.0, 1.0);
}

#[test]
fn stage_vertex_entry_shape() {
    let src = composite_vertex_entry::wgsl_source();
    assert!(
        src.starts_with("@vertex\nfn vs_main(@builtin(vertex_index) idx: u32)"),
        "{src}"
    );
    assert!(src.contains("-> CompositeVertexOutput"), "{src}");
    assert!(
        src.contains("return CompositeVertexOutput(QUAD[idx], UVS[idx]);"),
        "{src}"
    );
    assert_eq!(composite_vertex_entry::entry_point(), "vs_main");
}

#[test]
fn stage_fragment_location_param() {
    let src = composite_fragment_entry::wgsl_source();
    assert!(
        src.starts_with("@fragment\nfn fs_main(@location(0) uv: vec2<f32>)"),
        "{src}"
    );
    assert!(src.contains("-> vec4<f32>"), "{src}");
}
