//! HDR composite shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module: the quad-vertex
//! boilerplate and the HDR-mix fragment skeleton live here as Rust strings,
//! and the `aces_tonemap` / `luminance` kernels are spliced in from
//! [`crate::shaders::math`] (single source of truth via `#[kernel]`).
//! The former handwritten `shaders/wgsl/composite_*.wgsl` sources were
//! deleted after the `#[stage]` translation; the `hdr_*_matches_legacy_shape`
//! tests pin the entry shapes.

use super::interface::{
    HdrFragmentOut as QuadVertexOutput, HdrVertexOutput as CompositeVertexOutput,
};
use super::wgsl_decl;
use super::{
    Resource, ResourceKind, STANDARD_QUAD, STANDARD_UVS, Sampler, Texture2d, naga_ir,
    resource_decls,
};
use crate::renderer::{BloomUniform, CameraUniform};
use crate::shaders::math::{aces_tonemap, luminance};
use ornis_macros::stage;

/// HDR composite vertex entry, written in Rust and translated to WGSL by
/// [`stage`](ornis_macros::stage): fullscreen-quad corner passthrough.
/// DSL-only (`QUAD`/`UVS` globals declared via `#[wgsl(global)]`) — replaced by
/// `composite_vertex_entry::wgsl_source()`, never compiled as Rust.
#[stage(vertex, entry = "vs_main")]
fn composite_vertex_entry(
    vertex_index: super::VertexIndex,
    ctx: Context<super::QuadContext>,
) -> CompositeVertexOutput {
    return CompositeVertexOutput {
        clip_position: ctx.quad[vertex_index],
        uv: ctx.uvs[vertex_index],
    };
}

/// HDR composite vertex shader: fullscreen quad from `vertex_index`.
///
/// Assembled from the quad constants, the derived `CompositeVertexOutput`
/// layout and the translated [`composite_vertex_entry`] body. Structurally
/// identical to the former handwritten vertex (same signature and
/// constructor call; the generated entry is single-line) — pinned by
/// `hdr_vertex_entry_matches_legacy_shape` plus naga and the pixel probes.
pub fn wgsl_vertex_source() -> String {
    format!(
        "\n{quad}\n{vout}\n{body}",
        quad = vertex_quad(),
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
        head = fragment_head(),
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

/// Quad constants shared by both HDR assemblies, via IR.
fn vertex_quad() -> String {
    naga_ir::const_block(&STANDARD_QUAD, &STANDARD_UVS)
}

/// Resource layout of the HDR composite pass (what `create_composite_pass`
/// builds its layout from). The uniform carries the explicit minimum size
/// the handwritten layout spelled.
pub const HDR_RESOURCES: [Resource; 5] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "deferred_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "forward_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "composite_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 3,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "bloom_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "bloom_params",
        kind: ResourceKind::Uniform(BloomUniform::WGSL_NAME),
        min_size: Some(std::mem::size_of::<BloomUniform>() as u64),
    },
];

/// Fragment resource bindings (0–4) from the table, plus quad constants.
fn fragment_head() -> String {
    let mut out = resource_decls(&HDR_RESOURCES, &[0, 1, 2, 3, 4]);
    out.push('\n');
    out.push_str(&vertex_quad());
    out
}

/// Fragment-file vertex entry (dead in practice — `fs_main` is the selected
/// entry — but part of the legacy text). Translated like the vertex module.
#[stage(vertex, entry = "vs_main")]
fn hdr_fragment_vs_entry(
    vertex_index: super::VertexIndex,
    ctx: Context<super::QuadContext>,
) -> QuadVertexOutput {
    return QuadVertexOutput {
        clip_position: ctx.quad[vertex_index],
        uv: ctx.uvs[vertex_index],
    };
}

/// HDR composite fragment entry, translated by [`stage`](ornis_macros::stage):
/// deferred/forward layer mix + bloom. DSL-only — resource bundle
/// (`ctx: HdrContext`). `==` on the mode uniform selects the layer mix.
/// HDR layer-mix resources as a context bundle (`ctx.deferred_tex`, …).
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct HdrContext {
    pub deferred_tex: Texture2d,
    pub forward_tex: Texture2d,
    pub bloom_tex: Texture2d,
    pub composite_sampler: Sampler,
    pub bloom_params: BloomUniform,
}

#[stage(fragment, entry = "fs_main")]
fn hdr_fragment_entry(
    input: QuadVertexOutput,
    ctx: Context<HdrContext>,
) -> super::Location<0, glam::Vec4> {
    let deferred_color = textureSample(ctx.deferred_tex, ctx.composite_sampler, input.uv).rgb;
    let forward_color = textureSample(ctx.forward_tex, ctx.composite_sampler, input.uv).rgba;
    let mut combined = deferred_color;
    if ctx.bloom_params.mode == 1u {
        combined = forward_color.rgb * forward_color.a;
    } else if ctx.bloom_params.mode == 2u {
        combined = deferred_color + forward_color.rgb * forward_color.a;
    }
    let bloom = textureSample(ctx.bloom_tex, ctx.composite_sampler, input.uv).rgb;
    let tonemapped = aces_tonemap(combined + bloom * ctx.bloom_params.intensity);
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
        assert!(entry.starts_with("@vertex\nfn vs_main(@builtin(vertex_index) vertex_index: u32)"));
        assert!(entry.contains("-> CompositeVertexOutput"));
        assert!(entry.contains(
            "return CompositeVertexOutput(quad[vertex_index], uvs[vertex_index]) /* clip_position, uv */;"
        ));
    }

    /// The translated fragment entry must keep the legacy shape: same
    /// signature, same sampling/mix/tonemap calls. Formatting differs
    /// (single-line, `else { if }` nesting, normalized int suffixes —
    /// all semantics-preserving, naga-validated).
    #[test]
    fn hdr_fragment_entry_matches_legacy_shape() {
        let entry = hdr_fragment_entry::wgsl_source();
        assert!(entry.starts_with("@fragment\nfn fs_main(input: QuadVertexOutput)"));
        assert!(entry.contains("-> @location(0) vec4<f32>"));
        assert!(entry.contains(
            "let deferred_color = textureSample(deferred_tex, composite_sampler, input.uv).rgb;"
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

    /// Every table row's declaration appears in the assembled shader, and
    /// every row maps to a layout entry: shader and pipeline agree.
    #[test]
    fn hdr_resources_cover_shader_and_layout() {
        use super::super::{bgl_entry, resource_decl};
        let src = wgsl_source();
        assert_eq!(HDR_RESOURCES.len(), 5);
        for r in HDR_RESOURCES {
            assert!(src.contains(&resource_decl(&r)), "missing {}", r.name);
            let e = bgl_entry(&r, false);
            assert_eq!((e.binding, e.visibility), (r.binding, r.visibility));
        }
    }
}
