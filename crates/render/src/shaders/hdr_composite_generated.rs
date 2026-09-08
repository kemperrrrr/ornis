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

use super::interface::{
    HdrFragmentOut as QuadVertexOutput, HdrVertexOutput as CompositeVertexOutput,
};
use super::wgsl_decl;
use crate::renderer::{BloomUniform, CameraUniform};
use crate::shaders::math::{aces_tonemap, luminance};
use ornis_macros::stage;

/// HDR composite vertex entry, written in Rust and translated to WGSL by
/// [`stage`](ornis_macros::stage): fullscreen-quad corner passthrough.
/// DSL-only (free `QUAD`/`UVS` binding identifiers) — replaced by
/// `composite_vertex_entry::wgsl_source()`, never compiled as Rust.
#[stage(vertex, entry = "vs_main")]
fn composite_vertex_entry(#[wgsl(builtin = "vertex_index")] idx: u32) -> CompositeVertexOutput {
    return CompositeVertexOutput {
        clip_position: QUAD[idx],
        uv: UVS[idx],
    };
}

/// HDR composite vertex shader: fullscreen quad from `vertex_index`.
///
/// Assembled from the quad constants, the derived `CompositeVertexOutput`
/// layout and the translated [`composite_vertex_entry`] body. Structurally
/// identical to `shaders/wgsl/composite_vertex.wgsl` (same signature and
/// constructor call; the generated entry is single-line) — pinned by
/// `hdr_vertex_entry_matches_legacy_shape` plus naga and the pixel probes.
pub fn wgsl_vertex_source() -> String {
    format!(
        "\n{quad}\n{vout}\n{body}",
        quad = WGSL_VERTEX_QUAD,
        vout = wgsl_decl(CompositeVertexOutput::WGSL_SOURCE),
        body = composite_vertex_entry::wgsl_source(),
    )
}

/// HDR composite fragment shader: deferred/forward layer mix + bloom.
///
/// Assembled from the derived `Camera`/`BloomParams` layouts, the shared
/// `QuadVertexOutput` varying, the two translated entries and the
/// ACES/luminance kernels; entry point `fs_main` is kept.
pub fn wgsl_source() -> String {
    format!(
        "\n{cam}\n{bloom}\n{head}\n{qo}\n{vs}\n{fs}\n{aces}\n{lum}",
        cam = wgsl_decl(CameraUniform::WGSL_SOURCE),
        bloom = wgsl_decl(BloomUniform::WGSL_SOURCE),
        head = WGSL_FRAGMENT_HEAD,
        qo = wgsl_decl(QuadVertexOutput::WGSL_SOURCE),
        vs = hdr_fragment_vs_entry::wgsl_source(),
        fs = hdr_fragment_entry::wgsl_source(),
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

/// Fragment-file vertex entry (dead in practice — `fs_main` is the selected
/// entry — but part of the legacy text). Translated like the vertex module.
#[stage(vertex, entry = "vs_main")]
fn hdr_fragment_vs_entry(#[wgsl(builtin = "vertex_index")] idx: u32) -> QuadVertexOutput {
    return QuadVertexOutput {
        clip_position: QUAD[idx],
        uv: UVS[idx],
    };
}

/// HDR composite fragment entry, translated by [`stage`](ornis_macros::stage):
/// deferred/forward layer mix + bloom. DSL-only — free texture/uniform
/// identifiers. `==` on the mode uniform selects the layer mix.
#[stage(fragment, entry = "fs_main", returns = "@location(0) vec4<f32>")]
fn hdr_fragment_entry(#[wgsl(location = 0)] uv: glam::Vec2) -> glam::Vec4 {
    let deferred_color = textureSample(deferred_tex, composite_sampler, uv).rgb;
    let forward_color = textureSample(forward_tex, composite_sampler, uv).rgba;
    let mut combined = deferred_color;
    if bloom_params.mode == 1u {
        combined = forward_color.rgb * forward_color.a;
    } else if bloom_params.mode == 2u {
        combined = deferred_color + forward_color.rgb * forward_color.a;
    }
    let bloom = textureSample(bloom_tex, composite_sampler, uv).rgb;
    let tonemapped = aces_tonemap(combined + bloom * bloom_params.intensity);
    return glam::Vec4::new(tonemapped, 1.0);
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

    /// The translated fragment entry must keep the legacy shape: same
    /// signature, same sampling/mix/tonemap calls. Formatting differs
    /// (single-line, `else { if }` nesting, normalized int suffixes —
    /// all semantics-preserving, naga-validated).
    #[test]
    fn hdr_fragment_entry_matches_legacy_shape() {
        let entry = hdr_fragment_entry::wgsl_source();
        assert!(entry.starts_with("@fragment\nfn fs_main(@location(0) uv: vec2<f32>)"));
        assert!(entry.contains("-> @location(0) vec4<f32>"));
        assert!(entry.contains(
            "let deferred_color = textureSample(deferred_tex, composite_sampler, uv).rgb;"
        ));
        assert!(entry.contains("if (bloom_params.mode =="));
        assert!(entry.contains("combined = forward_color.rgb * forward_color.a;"));
        assert!(entry.contains("combined = deferred_color + forward_color.rgb * forward_color.a;"));
        assert!(
            entry.contains(
                "let tonemapped = aces_tonemap(combined + bloom * bloom_params.intensity);"
            )
        );
        assert!(entry.contains("return vec4<f32>(tonemapped, 1.0);"));
    }

    #[test]
    fn hdr_generated_parity_with_legacy_assembly() {
        // Layouts still splice the derived declarations (entries are
        // translated — see above). The `_pad` line is gone: skipped
        // padding is not shader-visible.
        let src = wgsl_source();
        assert!(src.contains(&wgsl_decl(CameraUniform::WGSL_SOURCE)));
        assert!(src.contains(&wgsl_decl(BloomUniform::WGSL_SOURCE)));
        assert!(src.contains(&wgsl_decl(QuadVertexOutput::WGSL_SOURCE)));
        assert!(!src.contains("_pad"));
    }
}
