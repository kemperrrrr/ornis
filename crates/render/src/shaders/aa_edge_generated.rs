//! FXAA and SMAA shaders: the single-pass edge filters over the composited
//! image (raw WGSL).
//!
//! One shader serves both filters: the mode selector in the uniform picks
//! the blend, so the two share the gradient detection and differ only in
//! how the two taps straddling the edge are weighted (see
//! [`crate::antialias::fxaa_blend`] and [`crate::antialias::smaa_blend`] for
//! the CPU reference of the two blends).
//!
//! The body stays raw WGSL (not `#[stage]`): the per-mode branch and the
//! weighted blend are outside the DSL's subset.

use super::aa_common::{LUMA_HELPER, VERTEX_ENTRY};
use super::interface::HdrFragmentOut as QuadVertexOutput;
use super::{
    Resource, ResourceKind, STANDARD_QUAD, STANDARD_UVS, ShaderModule, naga_ir, wgsl_decl,
};
use crate::renderer::AaUniform;

/// Resource layout of the edge filter (what the renderer builds its
/// bind-group layout from).
pub const EDGE_RESOURCES: [Resource; 3] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "src_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "src_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "aa_params",
        kind: ResourceKind::Uniform(AaUniform::WGSL_NAME),
        min_size: Some(std::mem::size_of::<AaUniform>() as u64),
    },
];

/// Group-0 `@binding` indices the edge filter fragment stage declares.
pub const EDGE_FRAGMENT_BINDINGS: &[u32] = &[0, 1, 2];

/// The edge filter fragment body.
///
/// The luma gradient says where the edge runs: the two samples across the
/// dominant axis are blended into the centre, weighted by how strongly the
/// edge shows (FXAA) or by how close each sample is to the centre in
/// brightness (SMAA). A flat pixel is returned unchanged.
const EDGE_BODY: &str = r#"
    let uv = input.uv;
    let dims = vec2<f32>(textureDimensions(src_tex));
    let texel = 1.0 / dims;
    let center = textureSampleLevel(src_tex, src_sampler, uv, 0.0).rgb;
    let luma_center = aa_luma(center);
    let north = textureSampleLevel(src_tex, src_sampler, uv - vec2<f32>(0.0, texel.y), 0.0).rgb;
    let south = textureSampleLevel(src_tex, src_sampler, uv + vec2<f32>(0.0, texel.y), 0.0).rgb;
    let east = textureSampleLevel(src_tex, src_sampler, uv + vec2<f32>(texel.x, 0.0), 0.0).rgb;
    let west = textureSampleLevel(src_tex, src_sampler, uv - vec2<f32>(texel.x, 0.0), 0.0).rgb;
    let gradient_x = aa_luma(east) - aa_luma(west);
    let gradient_y = aa_luma(south) - aa_luma(north);
    let range = max(abs(gradient_x), abs(gradient_y));
    if (range <= aa_params.threshold) {
        return vec4<f32>(center, 1.0);
    }
    // The edge runs across the dominant gradient: blend along it.
    var tap_a = center;
    var tap_b = center;
    if (abs(gradient_x) > abs(gradient_y)) {
        tap_a = east;
        tap_b = west;
    } else {
        tap_a = south;
        tap_b = north;
    }
    var blended = (center + center + tap_a + tap_b) * 0.25;
    if (aa_params.mode == 1u) {
        // SMAA: weight the two taps by their luma similarity to the
        // centre, so the sample on the centre's side keeps the blend and
        // the other one fades out.
        let weight_a = 1.0 / (1.0 + abs(aa_luma(tap_a) - luma_center));
        let weight_b = 1.0 / (1.0 + abs(aa_luma(tap_b) - luma_center));
        blended = (tap_a * weight_a + tap_b * weight_b) / (weight_a + weight_b);
    }
    let weight = clamp((range - aa_params.threshold) / range, 0.0, 1.0);
    return vec4<f32>(mix(center, blended, weight), 1.0);
"#;

/// The edge filter fragment entry.
fn edge_fragment_entry() -> String {
    format!(
        "@fragment\nfn fs_main(input: QuadVertexOutput) -> @location(0) vec4<f32> {{{body}}}\n",
        body = EDGE_BODY
    )
}

/// Full WGSL source of the edge filter (the mode is a uniform selector, so
/// one module serves FXAA and SMAA).
pub fn wgsl_source() -> String {
    ShaderModule::new()
        .decl(wgsl_decl(AaUniform::WGSL_SOURCE))
        .resources(&EDGE_RESOURCES, EDGE_FRAGMENT_BINDINGS)
        .consts(naga_ir::const_block(&STANDARD_QUAD, &STANDARD_UVS))
        .decl(wgsl_decl(QuadVertexOutput::WGSL_SOURCE))
        .helper(crate::shaders::helpers::wgsl_consts())
        .helper(LUMA_HELPER)
        .entry(VERTEX_ENTRY)
        .entry(edge_fragment_entry())
        .emit()
}

/// Vertex entry-point name (the pipeline-convention ABI).
pub mod vs_main {
    /// Pipeline entry-point name.
    pub fn entry_point() -> &'static str {
        "vs_main"
    }
}

/// Fragment entry-point name (the pipeline-convention ABI).
pub mod fs_main {
    /// Pipeline entry-point name.
    pub fn entry_point() -> &'static str {
        "fs_main"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse and validate a WGSL source with naga.
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
    fn edge_shader_validates_with_naga() {
        assert_valid_wgsl("edge", &wgsl_source());
    }

    #[test]
    fn edge_resources_cover_every_binding() {
        assert_eq!(EDGE_RESOURCES.len(), EDGE_FRAGMENT_BINDINGS.len());
        for (index, resource) in EDGE_RESOURCES.iter().enumerate() {
            assert_eq!(resource.binding, index as u32);
        }
    }

    #[test]
    fn edge_shader_selects_the_filter_by_uniform_mode() {
        let source = wgsl_source();
        assert!(source.contains("aa_params.mode"));
        assert!(source.contains("aa_luma"));
    }
}
