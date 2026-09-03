//! Composite shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module; WGSL is assembled
//! from constants + `srgb_to_linear::wgsl_source()` (kernel from
//! `crates/render/src/shaders/math.rs` via `#[kernel]`). The handwritten
//! `shaders/wgsl/composite.wgsl` remains as a reference/legacy, but
//! `composite.rs` (LegacyCompositePass) now uses only this module.

use crate::shaders::math::srgb_to_linear;

/// WGSL bindings + quad constants + vertex/fragment entry points.
///
/// Assembled at runtime as a `String`, but the source is Rust: constants and
/// `srgb_to_linear::wgsl_source()` — the single `srgb_to_linear` in the
/// system. This removes duplication of the WGSL literal from `composite.rs`.
fn composite_wgsl_body() -> String {
    // Header: bindings, VertexOutput, QUAD/UVS, vertex entry.
    // Format is identical to `shaders/wgsl/composite.wgsl`; entry point names `vs`/`fs`
    // are kept for compatibility with `CompositePass::new`.
    let header = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var pbr_tex: texture_2d<f32>;
@group(0) @binding(1) var pbr_sampler: sampler;
@group(0) @binding(2) var ui_tex: texture_2d<f32>;
@group(0) @binding(3) var ui_sampler: sampler;

const QUAD: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0,  1.0, 0.0, 1.0),
    vec4<f32>( 1.0, -1.0, 0.0, 1.0),
    vec4<f32>( 1.0,  1.0, 0.0, 1.0),
);
const UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),
);

@vertex
fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = QUAD[idx];
    out.uv = UVS[idx];
    return out;
}
"#;

    // Fragment entry: sampling + sRGB decode + mix. Uses `srgb_to_linear`
    // from the kernel (same name in WGSL).
    let fragment = r#"
@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let bg = textureSampleLevel(pbr_tex, pbr_sampler, input.uv, 0.0);
    let ui = textureSampleLevel(ui_tex, ui_sampler, input.uv, 0.0);
    let ui_linear = srgb_to_linear(ui.rgb);
    return vec4<f32>(mix(bg.rgb, ui_linear, ui.a), 1.0);
}
"#;

    // Kernel WGSL already contains `fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> { ... }`
    let kernel = srgb_to_linear::wgsl_source();
    format!("{header}\n{kernel}\n{fragment}\n")
}

/// Full WGSL source for the composite pass, assembled from Rust.
pub fn wgsl_source() -> String {
    composite_wgsl_body()
}

/// Static view for naga validation in tests (cloned from `wgsl_source()`).
/// Also used for deterministic snapshot testing.
pub fn wgsl_source_static() -> String {
    wgsl_source()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_valid_wgsl(name: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name} must parse: {e}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} must validate: {e}"));
    }

    #[test]
    fn composite_generated_validates_with_naga() {
        assert_valid_wgsl("composite_generated", &wgsl_source());
    }

    #[test]
    fn composite_generated_contains_expected_bindings() {
        let src = wgsl_source();
        assert!(src.contains("@group(0) @binding(0) var pbr_tex"));
        assert!(src.contains("@group(0) @binding(1) var pbr_sampler"));
        assert!(src.contains("@group(0) @binding(2) var ui_tex"));
        assert!(src.contains("@group(0) @binding(3) var ui_sampler"));
        assert!(src.contains("fn vs("));
        assert!(src.contains("fn fs("));
        assert!(src.contains("fn srgb_to_linear"));
    }
}
