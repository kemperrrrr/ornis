//! Integration tests for `#[derive(WgslInterface)]`: the derive generates a
//! WGSL shader-interface struct declaration from field attributes. Missing
//! assignments fail at compile time.

use ornis_macros::WgslInterface;

/// Fullscreen-quad varying: builtin position plus one UV location.
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "QuadVertexOutput")]
#[allow(dead_code)]
struct QuadVarying {
    #[wgsl(builtin = "position")]
    clip_position: [f32; 4],
    #[wgsl(location = 0)]
    uv: [f32; 2],
}

/// Instance vertex output with a flat integer varying.
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "VertexOutput")]
#[allow(dead_code)]
struct InstanceVarying {
    #[wgsl(builtin = "position")]
    clip_position: [f32; 4],
    #[wgsl(location = 0)]
    world_position: [f32; 3],
    #[wgsl(location = 4, interpolate = "flat")]
    material_index: u32,
}

#[test]
fn quad_varying_wgsl_source() {
    let src = QuadVarying::WGSL_SOURCE;
    assert!(src.contains("struct QuadVertexOutput"));
    assert!(src.contains("@builtin(position) clip_position: vec4<f32>"));
    assert!(src.contains("@location(0) uv: vec2<f32>"));
}

#[test]
fn instance_varying_wgsl_source() {
    let src = InstanceVarying::WGSL_SOURCE;
    assert!(src.contains("struct VertexOutput"));
    assert!(src.contains("@builtin(position) clip_position: vec4<f32>"));
    assert!(src.contains("@location(0) world_position: vec3<f32>"));
    assert!(src.contains("@location(4) @interpolate(flat) material_index: u32"));
}
