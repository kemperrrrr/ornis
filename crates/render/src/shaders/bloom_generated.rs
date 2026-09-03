//! Bloom shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module; WGSL is assembled
//! from constants + `luminance::wgsl_source()` (kernel from
//! `crates/render/src/shaders/math.rs` via `#[kernel]`). The handwritten
//! `shaders/wgsl/bloom_fragment.wgsl` remains as a reference/legacy, but
//! `renderer::create_bloom_pass` now uses only this module.

use crate::shaders::math::luminance;

/// WGSL bindings + quad constants + vertex/fragment entry points.
///
/// Assembled at runtime as a `String`, but the source is Rust: constants and
/// `luminance::wgsl_source()` — the single `luminance` in the system.
fn bloom_wgsl_body() -> String {
    let header = r#"
struct BloomParams {
    threshold: f32,
    intensity: f32,
    mode: u32,
    _pad: f32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> bloom_params: BloomParams;

const QUAD: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>( 1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0,  1.0, 0.0, 1.0),
    vec4<f32>( 1.0,  1.0, 0.0, 1.0),
);

const UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
);

struct BloomVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> BloomVertexOutput {
    return BloomVertexOutput(QUAD[idx], UVS[idx]);
}
"#;

    let fragment = r#"
@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let color = textureSample(src_tex, src_sampler, uv).rgb;
    let luma = luminance(color);
    let keep = smoothstep(bloom_params.threshold, bloom_params.threshold + 0.05, luma);
    return vec4<f32>(color * keep, 1.0);
}
"#;

    let kernel = luminance::wgsl_source();
    format!("{header}\n{kernel}\n{fragment}\n")
}

/// Full WGSL source for the bloom pass, assembled from Rust.
pub fn wgsl_source() -> String {
    bloom_wgsl_body()
}

/// Static view for naga validation in tests.
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
    fn bloom_generated_validates_with_naga() {
        assert_valid_wgsl("bloom_generated", &wgsl_source());
    }

    #[test]
    fn bloom_generated_contains_expected_bindings() {
        let src = wgsl_source();
        assert!(src.contains("@group(0) @binding(0) var src_tex"));
        assert!(src.contains("@group(0) @binding(1) var src_sampler"));
        assert!(src.contains("@group(0) @binding(2) var<uniform> bloom_params"));
        assert!(src.contains("fn vs_main("));
        assert!(src.contains("fn fs_main("));
        assert!(src.contains("fn luminance"));
    }
}
