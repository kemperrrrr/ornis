//! Bloom shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module; WGSL is assembled
//! from constants + `luminance::wgsl_source()` (kernel from
//! `crates/render/src/shaders/math.rs` via `#[kernel]`). The former
//! handwritten `shaders/wgsl/bloom_fragment.wgsl` was deleted after the
//! `#[stage]` translation; `renderer::create_bloom_pass` now uses only this module.

use super::interface::BloomVertexOut as BloomVertexOutput;
use super::wgsl_decl;
use crate::renderer::BloomUniform;
use crate::shaders::math::luminance;
use ornis_macros::stage;

/// Bloom vertex entry, translated by [`stage`](ornis_macros::stage).
/// DSL-only — replaced by `bloom_vertex_entry::wgsl_source()`.
#[stage(vertex, entry = "vs_main")]
fn bloom_vertex_entry(#[wgsl(builtin = "vertex_index")] idx: u32) -> BloomVertexOutput {
    return BloomVertexOutput {
        clip_position: QUAD[idx],
        uv: UVS[idx],
    };
}

/// Bloom fragment entry, translated by [`stage`](ornis_macros::stage):
/// bright-pass reselection. DSL-only — free texture/uniform identifiers.
/// `Vec4::new(vec3, scalar)` spells the WGSL `vec4<f32>(vec3, f32)`
/// constructor; the `returns` override carries the `@location` return.
#[stage(fragment, entry = "fs_main", returns = "@location(0) vec4<f32>")]
fn bloom_fragment_entry(#[wgsl(location = 0)] uv: glam::Vec2) -> glam::Vec4 {
    let color = textureSample(src_tex, src_sampler, uv).rgb;
    let luma = luminance(color);
    let keep = smoothstep(bloom_params.threshold, bloom_params.threshold + 0.05, luma);
    return glam::Vec4::new(color * keep, 1.0);
}

/// WGSL bindings + quad constants + vertex/fragment entry points.
///
/// Assembled at runtime as a `String`, but the source is Rust: constants and
/// `luminance::wgsl_source()` — the single `luminance` in the system.
fn bloom_wgsl_body() -> String {
    // Derived `BloomParams` layout and `BloomVertexOut` varying spliced into
    // the handwritten header rest (bindings + quad + entries). Byte-identical
    // to the previous assembly except the dropped `_pad` line
    // (`#[wgsl(skip)]` pads are not shader-visible).
    let header = format!(
        "\n{bloom}\n{head}\n{vout}\n{vs}",
        bloom = wgsl_decl(BloomUniform::WGSL_SOURCE),
        head = BLOOM_HEADER_HEAD,
        vout = wgsl_decl(BloomVertexOutput::WGSL_SOURCE),
        vs = bloom_vertex_entry::wgsl_source(),
    );

    const BLOOM_HEADER_HEAD: &str = r#"@group(0) @binding(0) var src_tex: texture_2d<f32>;
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
"#;

    let fragment = bloom_fragment_entry::wgsl_source();

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

    /// Translated entries keep the legacy shape (naga + pixels carry the
    /// rest; generated entries are single-line).
    #[test]
    fn bloom_entries_match_legacy_shape() {
        let vs = bloom_vertex_entry::wgsl_source();
        assert!(vs.starts_with("@vertex\nfn vs_main(@builtin(vertex_index) idx: u32)"));
        assert!(vs.contains("return BloomVertexOutput(QUAD[idx], UVS[idx]);"));
        let fs = bloom_fragment_entry::wgsl_source();
        assert!(fs.starts_with("@fragment\nfn fs_main(@location(0) uv: vec2<f32>)"));
        assert!(fs.contains("-> @location(0) vec4<f32>"));
        assert!(fs.contains("let color = textureSample(src_tex, src_sampler, uv).rgb;"));
        assert!(fs.contains("return vec4<f32>(color * keep, 1.0);"));
    }
}
