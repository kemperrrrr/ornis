//! Composite shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module; WGSL is assembled
//! from constants + `srgb_to_linear::wgsl_source()` (kernel from
//! `crates/render/src/shaders/math.rs` via `#[kernel]`). The former
//! handwritten `shaders/wgsl/composite.wgsl` was deleted after the `#[stage]`
//! translation; `composite.rs` (LegacyCompositePass) now uses only this module.

use super::interface::UiCompositeOut as VertexOutput;
use super::{Resource, ResourceKind, Sampler, Texture2d, naga_ir, resource_decls, wgsl_decl};
use crate::shaders::math::srgb_to_linear;
use ornis_macros::stage;

/// Composite fragment resources as a context bundle (`ctx.pbr_tex`, …).
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct CompositeContext {
    pub pbr_tex: Texture2d,
    pub pbr_sampler: Sampler,
    pub ui_tex: Texture2d,
    pub ui_sampler: Sampler,
}

/// Legacy composite vertex entry, translated by [`stage`](ornis_macros::stage).
/// DSL-only — replaced by `composite_vs_entry::wgsl_source()`. Uses the
/// `var`-out form (`let mut out: T;` + field assignment).
#[stage(vertex, entry = "vs")]
fn composite_vs_entry(
    vertex_index: super::VertexIndex,
    #[wgsl(context)] ctx: super::QuadContext,
) -> VertexOutput {
    let mut out: VertexOutput;
    out.position = ctx.quad[vertex_index];
    out.uv = ctx.uvs[vertex_index];
    return out;
}

/// Legacy composite fragment entry, translated by [`stage`](ornis_macros::stage).
/// DSL-only — texture bundle (`ctx: CompositeContext`).
#[stage(fragment, entry = "fs", returns = "@location(0) vec4<f32>")]
fn composite_fs_entry(input: VertexOutput, #[wgsl(context)] ctx: CompositeContext) -> glam::Vec4 {
    let bg = textureSampleLevel(ctx.pbr_tex, ctx.pbr_sampler, input.uv, 0.0);
    let ui = textureSampleLevel(ctx.ui_tex, ctx.ui_sampler, input.uv, 0.0);
    let ui_linear = srgb_to_linear(ui.rgb);
    return glam::Vec4::new(mix(bg.rgb, ui_linear, ui.a), 1.0);
}

/// Composite-specific quad corners/UVs (different winding from
/// [`STANDARD_QUAD`](super::STANDARD_QUAD)), as Rust data.
const COMPOSITE_QUAD: [[f32; 4]; 4] = [
    [-1.0, -1.0, 0.0, 1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [1.0, -1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0],
];

/// Composite-specific UVs, as Rust data.
const COMPOSITE_UVS: [[f32; 2]; 4] = [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]];

/// Resource layout of the legacy composite pass (what `LegacyCompositePass`
/// builds its layout from in `composite.rs`).
pub const COMPOSITE_RESOURCES: [Resource; 4] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "pbr_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "pbr_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ui_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 3,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ui_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
];

/// WGSL bindings + quad constants + vertex/fragment entry points.
///
/// Assembled at runtime as a `String`, but the source is Rust: constants and
/// `srgb_to_linear::wgsl_source()` — the single `srgb_to_linear` in the
/// system. This removes duplication of the WGSL literal from `composite.rs`.
fn composite_wgsl_body() -> String {
    // Header: derived `VertexOutput` varying plus bindings, QUAD/UVS, vertex
    // entry. Format matches the former handwritten composite; entry
    // point names `vs`/`fs` are kept for compatibility with
    // `CompositePass::new`.
    let header = format!(
        "\n{vout}\n{rest}\n{vs}",
        vout = wgsl_decl(VertexOutput::WGSL_SOURCE),
        rest = composite_header_rest(),
        vs = composite_vs_entry::wgsl_source(),
    );

    /// Bindings (0–3) from the table + quad constants, all assembled from Rust.
    fn composite_header_rest() -> String {
        let mut out = resource_decls(&COMPOSITE_RESOURCES, &[0, 1, 2, 3]);
        out.push('\n');
        out.push_str(&naga_ir::const_block(&COMPOSITE_QUAD, &COMPOSITE_UVS));
        out
    }

    // Fragment entry: sampling + sRGB decode + mix. Translated above; the
    // kernel (same `srgb_to_linear` name in WGSL) splices in below.
    let fragment = composite_fs_entry::wgsl_source();

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

    /// Every table row's declaration appears in the assembled shader, and
    /// every row maps to a layout entry: shader and pipeline agree.
    #[test]
    fn composite_resources_cover_shader_and_layout() {
        use super::super::{bgl_entry, resource_decl};
        let src = wgsl_source();
        assert_eq!(COMPOSITE_RESOURCES.len(), 4);
        for r in COMPOSITE_RESOURCES {
            assert!(src.contains(&resource_decl(&r)), "missing {}", r.name);
            let e = bgl_entry(&r, false);
            assert_eq!((e.binding, e.visibility), (r.binding, r.visibility));
        }
    }
}
