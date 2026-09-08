//! HDR composite shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module: the quad-vertex
//! boilerplate and the HDR-mix fragment skeleton live here as Rust strings,
//! and the `aces_tonemap` / `luminance` kernels are spliced in from
//! [`crate::shaders::math`] (single source of truth via `#[kernel]`).
//! The handwritten `shaders/wgsl/composite_vertex.wgsl` and
//! `shaders/wgsl/composite_fragment.wgsl` remain as references; the
//! `hdr_generated_parity_with_legacy_assembly` test pins this module
//! byte-identical to them.

use super::interface::{HdrFragmentOut, HdrVertexOutput};
use super::wgsl_decl;
use crate::renderer::{BloomUniform, CameraUniform};
use crate::shaders::math::{aces_tonemap, luminance};
use ornis_macros::stage;

/// HDR composite vertex entry, written in Rust and translated to WGSL by
/// [`stage`](ornis_macros::stage): fullscreen-quad corner passthrough.
/// DSL-only (free `QUAD`/`UVS` binding identifiers) — replaced by
/// `composite_vertex_entry::wgsl_source()`, never compiled as Rust.
#[stage(vertex, entry = "vs_main", returns = "CompositeVertexOutput")]
fn composite_vertex_entry(#[wgsl(builtin = "vertex_index")] idx: u32) -> HdrVertexOutput {
    return HdrVertexOutput {
        clip_position: QUAD[idx],
        uv: UVS[idx],
    };
}

/// HDR composite vertex shader: fullscreen quad from `vertex_index`.
///
/// Assembled from the quad constants, the derived [`HdrVertexOutput`]
/// layout and the translated [`composite_vertex_entry`] body. Structurally
/// identical to `shaders/wgsl/composite_vertex.wgsl` (same signature and
/// constructor call; the generated entry is single-line) — pinned by
/// `hdr_vertex_entry_matches_legacy_shape` plus naga and the pixel probes.
pub fn wgsl_vertex_source() -> String {
    format!(
        "\n{quad}\n{vout}\n{body}",
        quad = WGSL_VERTEX_QUAD,
        vout = wgsl_decl(HdrVertexOutput::WGSL_SOURCE),
        body = composite_vertex_entry::wgsl_source(),
    )
}

/// HDR composite fragment shader: deferred/forward layer mix + bloom.
///
/// Assembled from the derived `Camera`/`BloomParams` layouts plus the
/// fragment skeleton and the ACES/luminance kernels; entry point `fs_main`
/// is kept. Byte-identical to the legacy assembly except the dropped
/// `_pad` line (`#[wgsl(skip)]` pads are not shader-visible).
pub fn wgsl_source() -> String {
    format!(
        "\n{cam}\n{bloom}\n{head}\n{qo}\n{tail}\n{aces}\n{lum}",
        cam = wgsl_decl(CameraUniform::WGSL_SOURCE),
        bloom = wgsl_decl(BloomUniform::WGSL_SOURCE),
        head = WGSL_FRAGMENT_HEAD,
        qo = wgsl_decl(HdrFragmentOut::WGSL_SOURCE),
        tail = WGSL_FRAGMENT_TAIL,
        aces = aces_tonemap::wgsl_source(),
        lum = luminance::wgsl_source()
    )
}

/// Static view for naga validation in tests.
pub fn wgsl_source_static() -> String {
    wgsl_source()
}

const WGSL_VERTEX_QUAD: &str = r#"const QUAD: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
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

const WGSL_FRAGMENT_HEAD: &str = r#"@group(0) @binding(0) var deferred_tex: texture_2d<f32>;
@group(0) @binding(1) var forward_tex: texture_2d<f32>;
@group(0) @binding(2) var composite_sampler: sampler;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> bloom_params: BloomParams;

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

const WGSL_FRAGMENT_TAIL: &str = r#"@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> QuadVertexOutput {
    return QuadVertexOutput(QUAD[idx], UVS[idx]);
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let deferred_color = textureSample(deferred_tex, composite_sampler, uv).rgb;
    let forward_color = textureSample(forward_tex, composite_sampler, uv).rgba;

    // Layer mix depends on the technique: 0 = deferred-only, 1 = forward-only,
    // 2 = hybrid. The dead layer is bound to the live one, so the mode is
    // what disambiguates the two inputs.
    var combined = deferred_color;
    if (bloom_params.mode == 1u) {
        combined = forward_color.rgb * forward_color.a;
    } else if (bloom_params.mode == 2u) {
        combined = deferred_color + forward_color.rgb * forward_color.a;
    }
    let bloom = textureSample(bloom_tex, composite_sampler, uv).rgb;
    let tonemapped = aces_tonemap(combined + bloom * bloom_params.intensity);

    // The composited scene is opaque; forward_color.a is 0 where no forward
    // geometry was drawn, which would make the whole frame transparent on a
    // canvas/surface. Native compositing (LegacyCompositePass) also forces 1.0.
    return vec4<f32>(tonemapped, 1.0);
}
"#;

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
    fn hdr_composite_generated_validates_with_naga() {
        assert_valid_wgsl("hdr_composite_vertex", &wgsl_vertex_source());
        assert_valid_wgsl("hdr_composite_fragment", &wgsl_source());
    }

    #[test]
    fn hdr_composite_generated_contains_expected_bindings() {
        let src = wgsl_source();
        assert!(src.contains("@group(0) @binding(0) var deferred_tex"));
        assert!(src.contains("@group(0) @binding(4) var<uniform> bloom_params"));
        assert!(src.contains("fn vs_main("));
        assert!(src.contains("fn fs_main("));
        assert!(src.contains("fn aces_tonemap"));
        assert!(src.contains("fn luminance"));
    }

    /// The translated vertex entry must keep the legacy shape: same entry
    /// name and signature, same constructor call on the same globals.
    /// (Byte-parity no longer applies — the generated entry is single-line.)
    #[test]
    fn hdr_vertex_entry_matches_legacy_shape() {
        let entry = composite_vertex_entry::wgsl_source();
        assert!(entry.starts_with("@vertex\nfn vs_main(@builtin(vertex_index) idx: u32)"));
        assert!(entry.contains("-> CompositeVertexOutput"));
        assert!(entry.contains("return CompositeVertexOutput(QUAD[idx], UVS[idx]);"));
    }

    #[test]
    fn hdr_generated_parity_with_legacy_assembly() {
        // Fragment only: the vertex entry is translated (see above), the
        // fragment skeleton is still spliced.
        // The only admitted difference: the `_pad` line is gone — skipped
        // padding is not shader-visible (`#[wgsl(skip)]`).
        let legacy_fragment = format!(
            "{}\n{}\n{}",
            include_str!("wgsl/composite_fragment.wgsl").replace("    _pad: f32,\n", ""),
            aces_tonemap::wgsl_source(),
            luminance::wgsl_source()
        );
        assert_eq!(wgsl_source(), legacy_fragment);
    }
}
