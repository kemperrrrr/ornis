//! Temporal anti-aliasing shader: the history resolve (raw WGSL).
//!
//! Canonical source is this module: the fullscreen-quad vertex boilerplate
//! plus the temporal fragment body, assembled with the shared
//! [`ShaderModule`](super::ShaderModule) machinery, so naga pins the result
//! like every other pass.
//!
//! The body stays raw WGSL (not `#[stage]`): the neighbourhood clamp needs
//! an indexed loop and the reprojection a matrix multiply, neither of
//! which the DSL covers. The CPU reference of the same math lives in
//! [`crate::antialias`].
//!
//! The resolve reads the composited image (`src_tex`), the g-buffer depth
//! (`depth_tex`, multisampled at 4x — sample 0 is loaded) and the previous
//! frame's history (`history_tex`), and writes two colour attachments: the
//! presented image and the history the next frame reads.

use super::aa_common::{VERTEX_ENTRY, YCOCG_HELPER};
use super::interface::HdrFragmentOut as QuadVertexOutput;
use super::{
    Resource, ResourceKind, STANDARD_QUAD, STANDARD_UVS, ShaderModule, naga_ir, wgsl_decl,
};
use crate::renderer::AaUniform;

/// Resource layout of the temporal resolve (what the renderer builds its
/// bind-group layout from). The uniform carries the explicit minimum size.
pub const TAA_RESOURCES: [Resource; 6] = [
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
        name: "depth_tex",
        kind: ResourceKind::TextureDepth,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 3,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "motion_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "history_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 5,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "aa_params",
        kind: ResourceKind::Uniform(AaUniform::WGSL_NAME),
        min_size: Some(std::mem::size_of::<AaUniform>() as u64),
    },
];

/// Group-0 `@binding` indices the temporal fragment stage declares.
pub const TAA_FRAGMENT_BINDINGS: &[u32] = &[0, 1, 2, 3, 4, 5];

/// The two colour attachments of the temporal resolve: the presented image
/// and the history the next frame reads.
const TAA_OUTPUT_DECL: &str = r#"
struct TaaOutput {
    @location(0) color: vec4<f32>,
    @location(1) history: vec4<f32>,
}
"#;

/// The temporal fragment body: the history resolve.
///
/// The current pixel is reprojected into the previous frame through the
/// depth buffer, the history sample is clamped into the colour range of
/// the current neighbourhood (YCoCg), and the two are blended. A pixel
/// with no surface behind it (the cleared depth) and a reprojection that
/// leaves the screen keep the current colour — the history is not read
/// there.
const TAA_BODY: &str = r#"
    let uv = input.uv;
    let dims_u = textureDimensions(src_tex);
    let dims = vec2<f32>(dims_u);
    let coord_u = min(vec2<u32>(uv * dims), dims_u - vec2<u32>(1u));
    let coord = vec2<i32>(coord_u);
    let current = textureSampleLevel(src_tex, src_sampler, uv, 0.0).rgb;
    let depth = textureLoad(depth_tex, coord, 0);
    if (depth >= CLEAR_DEPTH) {
        return TaaOutput(vec4<f32>(current, 1.0), vec4<f32>(current, 1.0));
    }
    // The g-buffer motion is current UV minus the previous UV. Its B
    // channel is the depth this surface had in the previous frame, so the
    // sampled history can reject disocclusions before colour blending.
    let motion = textureLoad(motion_tex, coord, 0).rgba;
    let prev_uv = uv - motion.xy;
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        return TaaOutput(vec4<f32>(current, 1.0), vec4<f32>(current, depth));
    }
    // The colour range of the current neighbourhood: the history is
    // clamped into it, so a stale sample cannot drag its colour along.
    let radius = i32(aa_params.clamp_radius);
    var lower = vec3<f32>(1.0e30);
    var upper = vec3<f32>(-1.0e30);
    var sum = vec3<f32>(0.0);
    var sum_squares = vec3<f32>(0.0);
    var count = 0.0;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let tap_uv = clamp(uv + vec2<f32>(f32(dx), f32(dy)) / dims, vec2<f32>(0.0), vec2<f32>(1.0));
            let tap = aa_ycocg_encode(textureSampleLevel(src_tex, src_sampler, tap_uv, 0.0).rgb);
            lower = min(lower, tap);
            upper = max(upper, tap);
            sum = sum + tap;
            sum_squares = sum_squares + tap * tap;
            count = count + 1.0;
        }
    }
    let mean = sum / count;
    let sigma = sqrt(max(sum_squares / count - mean * mean, vec3<f32>(0.0)));
    let bounds_low = max(lower, mean - 1.25 * sigma);
    let bounds_high = min(upper, mean + 1.25 * sigma);
    var result = current;
    if (aa_params.history_valid > 0.5 && motion.a > 0.5) {
        let history = textureSampleLevel(history_tex, src_sampler, prev_uv, 0.0);
        // The motion vector carries the previous NDC depth. A mismatch
        // means the previous surface was occluded or this is a disocclusion:
        // rejecting it is the depth test that keeps moving objects from
        // leaving long history trails.
        let depth_gap = abs(history.a - motion.z);
        if (depth_gap < 0.002) {
            let clamped = aa_ycocg_decode(
                clamp(aa_ycocg_encode(history.rgb), bounds_low, bounds_high)
            );
            result = mix(current, clamped, aa_params.feedback);
        }
    }
    return TaaOutput(vec4<f32>(result, 1.0), vec4<f32>(result, depth));
"#;

/// The temporal fragment entry.
fn taa_fragment_entry() -> String {
    format!(
        "@fragment\nfn fs_main(input: QuadVertexOutput) -> TaaOutput {{{body}}}\n",
        body = TAA_BODY
    )
}

/// Full WGSL source of the temporal resolve at a sample count: the depth
/// binding spells the multisampled type at 4x (sample 0 is loaded, the
/// lighting MSAA contract).
pub fn wgsl_source_for_samples(sample_count: u32) -> String {
    ShaderModule::new()
        .decl(wgsl_decl(AaUniform::WGSL_SOURCE))
        .decl(TAA_OUTPUT_DECL)
        .resources_for_samples(&TAA_RESOURCES, TAA_FRAGMENT_BINDINGS, sample_count)
        .consts(naga_ir::const_block(&STANDARD_QUAD, &STANDARD_UVS))
        .decl(wgsl_decl(QuadVertexOutput::WGSL_SOURCE))
        .helper(crate::shaders::helpers::wgsl_consts())
        .helper(YCOCG_HELPER)
        .entry(VERTEX_ENTRY)
        .entry(taa_fragment_entry())
        .emit()
}

/// Full WGSL source of the temporal resolve (single-sample).
pub fn wgsl_source() -> String {
    wgsl_source_for_samples(1)
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
    fn taa_shader_validates_with_naga() {
        assert_valid_wgsl("taa", &wgsl_source());
        assert_valid_wgsl("taa_msaa", &wgsl_source_for_samples(4));
    }

    #[test]
    fn taa_resources_cover_every_binding() {
        // The fragment stage declares every row of the table, so the
        // shader and the pipeline layout cannot drift apart.
        assert_eq!(TAA_RESOURCES.len(), TAA_FRAGMENT_BINDINGS.len());
        for (index, resource) in TAA_RESOURCES.iter().enumerate() {
            assert_eq!(resource.binding, index as u32);
            assert_eq!(resource.group, 0);
        }
        let source = wgsl_source();
        for resource in TAA_RESOURCES {
            assert!(
                source.contains(&format!("@binding({})", resource.binding)),
                "binding {} missing",
                resource.binding
            );
        }
    }

    #[test]
    fn taa_shader_reads_the_history_and_writes_both_targets() {
        let source = wgsl_source();
        assert!(source.contains("history_tex"));
        assert!(source.contains("@location(1) history"));
        assert!(source.contains("motion.a > 0.5"));
        assert!(source.contains("depth_gap < 0.002"));
        assert!(source.contains("aa_ycocg_encode"));
    }
}
