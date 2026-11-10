//! Screen-space AO shaders: classic SSAO and GTAO raw passes plus the
//! bilateral blur.
//!
//! Every fragment body is a checked-in llvm2gpu artifact built from a
//! `crates/gpu-*` kernel: `ssao_generated.wgsl`, `gtao_generated.wgsl`,
//! `ao_blur_generated.wgsl`, and the `--multisampled` builds used by the 4x
//! lighting path (`ssao_msaa_generated.wgsl`, `gtao_msaa_generated.wgsl`).
//! This module owns what is left of the hand-written WGSL here: the shared
//! fullscreen-quad vertex stage, the bind-group layouts, and the mapping from
//! a sample count to the artifact.
//!
//! At 4x the depth and the octahedral normal stay per-sample (the normal must
//! not go through the hardware box filter: a cleared texel is `(0, 0)`, which
//! decodes to +Z and would fringe every silhouette — the lighting MSAA
//! contract, see `keeps_per_sample` there). The kernels keep the same body and
//! only change the declared texture types.
//!
//! Depth source is the g-buffer `Depth` buffer (hardware `Depth32Float`,
//! linearized through `Camera::inv_view_proj` by the shared
//! `reconstruct_world_pos`); normals decode from the octahedral layer.

use super::interface::HdrFragmentOut as QuadVertexOutput;
use super::{
    Resource, ResourceKind, STANDARD_QUAD, STANDARD_UVS, ShaderModule, naga_ir,
    resource_stays_multisampled, wgsl_decl,
};
use crate::renderer::{AoUniform, CameraUniform};

/// Resource layout of the AO raw pass (what `Renderer3D::create_ao_passes`
/// builds its layout from). Depth and normals are `textureLoad`ed, so no
/// sampler is bound here. The uniform carries the explicit minimum size.
pub const AO_RAW_RESOURCES: [Resource; 4] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "depth_tex",
        kind: ResourceKind::TextureDepth,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "normal_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "camera",
        kind: ResourceKind::Uniform(CameraUniform::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 3,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ao_params",
        kind: ResourceKind::Uniform(AoUniform::WGSL_NAME),
        min_size: Some(std::mem::size_of::<AoUniform>() as u64),
    },
];

/// Resource layout of the bilateral blur pass.
pub const AO_BLUR_RESOURCES: [Resource; 5] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ao_raw_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "blur_sampler",
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
        name: "ao_params",
        kind: ResourceKind::Uniform(AoUniform::WGSL_NAME),
        min_size: Some(std::mem::size_of::<AoUniform>() as u64),
    },
    Resource {
        group: 0,
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "camera",
        kind: ResourceKind::Uniform(CameraUniform::WGSL_NAME),
        min_size: None,
    },
];

/// Depth and the octahedral normal stay per-sample at 4x (the normal must
/// not go through the hardware box filter: a cleared texel is `(0, 0)`,
/// which decodes to +Z and would fringe every silhouette — the lighting
/// MSAA contract, see `keeps_per_sample` there).
pub(crate) fn ao_keeps_per_sample(name: &str, kind: &ResourceKind) -> bool {
    name == "normal_tex" || resource_stays_multisampled(kind)
}

/// Whether an AO binding keeps per-sample storage in MSAA mode.
pub(crate) fn ao_per_sample_flag(sample_count: u32, keeps: bool) -> bool {
    sample_count > 1 && keeps
}

/// Fullscreen vertex entry shared by all AO stages (triangle-strip quad).
const AO_VERTEX_ENTRY: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> QuadVertexOutput {
    return QuadVertexOutput(quad[vertex_index], uvs[vertex_index]);
}
"#;

/// Full WGSL source for the AO raw fragment stage at a sample count.
///
/// The kernels come from the llvm2gpu crates: one artifact for 1x and the
/// `--multisampled` build for 4x, where depth and the normal declare the
/// multisampled types (both stay per-sample: depth has no resolve target and
/// the normal must not go through the hardware box filter).
pub fn wgsl_source_raw_for_samples(method: crate::ao::AoMethod, sample_count: u32) -> String {
    if method.is_ssao() {
        if sample_count > 1 {
            include_str!("ssao_msaa_generated.wgsl").to_string()
        } else {
            include_str!("ssao_generated.wgsl").to_string()
        }
    } else if sample_count > 1 {
        include_str!("gtao_msaa_generated.wgsl").to_string()
    } else {
        include_str!("gtao_generated.wgsl").to_string()
    }
}

/// Full WGSL source for the AO raw fragment stage (single-sample).
pub fn wgsl_source_raw(method: crate::ao::AoMethod) -> String {
    wgsl_source_raw_for_samples(method, 1)
}

/// Full WGSL source for the bilateral blur stage (always single-sample:
/// the pooled AO targets never multisample).
pub fn wgsl_source_blur() -> String {
    include_str!("ao_blur_generated.wgsl").to_string()
}

/// Full WGSL source for the shared AO vertex stage.
pub fn wgsl_vertex_source() -> String {
    ShaderModule::new()
        .consts(naga_ir::const_block(&STANDARD_QUAD, &STANDARD_UVS))
        .decl(wgsl_decl(QuadVertexOutput::WGSL_SOURCE))
        .entry(AO_VERTEX_ENTRY)
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
    use crate::ao::AoMethod;

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
    fn ao_shaders_validate_with_naga() {
        assert_valid_wgsl("ao_vertex", &wgsl_vertex_source());
        assert_valid_wgsl("ao_raw_ssao", &wgsl_source_raw(AoMethod::Ssao));
        assert_valid_wgsl("ao_raw_gtao", &wgsl_source_raw(AoMethod::Gtao));
        assert_valid_wgsl("ao_blur", &wgsl_source_blur());
        assert_valid_wgsl(
            "ao_raw_ssao_msaa",
            &wgsl_source_raw_for_samples(AoMethod::Ssao, 4),
        );
        assert_valid_wgsl(
            "ao_raw_gtao_msaa",
            &wgsl_source_raw_for_samples(AoMethod::Gtao, 4),
        );
    }

    #[test]
    fn ao_raw_kernels_spell_their_math() {
        let ssao = wgsl_source_raw(AoMethod::Ssao);
        assert!(ssao.contains("@fragment"), "{ssao}");
        assert!(ssao.contains("fn fs_main("), "{ssao}");
        assert!(ssao.contains("texture_depth_2d"), "{ssao}");
        assert!(ssao.contains("texture_2d<f32>"), "{ssao}");
        assert!(ssao.contains("textureLoad("), "{ssao}");
        assert!(ssao.contains("loop"), "ssao taps: {ssao}");
        assert!(!ssao.contains("horizon"), "ssao must not horizon-march");
        let gtao = wgsl_source_raw(AoMethod::Gtao);
        assert!(gtao.contains("@fragment"), "{gtao}");
        assert!(gtao.contains("fn fs_main("), "{gtao}");
        assert!(gtao.contains("@location(0)"), "{gtao}");
        assert!(gtao.contains("texture_depth_2d"), "{gtao}");
        assert!(gtao.contains("texture_2d<f32>"), "{gtao}");
        assert!(gtao.contains("textureLoad("), "{gtao}");
        assert!(gtao.contains("acos("), "{gtao}");
        assert!(gtao.contains("cross("), "{gtao}");
        assert!(gtao.contains("loop"), "gtao slices: {gtao}");
        assert!(
            !ssao.contains("gtao"),
            "ssao must not integrate horizons: {ssao}"
        );
    }

    #[test]
    fn ao_msaa_artifacts_declare_multisampled_textures() {
        for method in [AoMethod::Ssao, AoMethod::Gtao] {
            let source = wgsl_source_raw_for_samples(method, 4);
            assert!(
                source.contains("texture_depth_multisampled_2d"),
                "{method:?} msaa depth: {source}"
            );
            assert!(
                source.contains("texture_multisampled_2d<f32>"),
                "{method:?} msaa normal: {source}"
            );
            assert!(source.contains("textureLoad("), "{source}");
        }
    }

    #[test]
    fn gtao_horizon_clamps_the_edge_texel() {
        for (label, source) in [
            ("gtao", wgsl_source_raw(AoMethod::Gtao)),
            ("gtao-4x", wgsl_source_raw_for_samples(AoMethod::Gtao, 4)),
        ] {
            assert!(
                source.contains("min(u32("),
                "{label} must clamp uv=1 to the last texel, like master min(u32(uv * dims), dims - 1)"
            );
        }
        // Master SSAO has no such min: `suv == 1` stays an out-of-bounds index.
        let ssao = wgsl_source_raw(AoMethod::Ssao);
        assert!(
            !ssao.contains("min(u32("),
            "ssao must keep master's unclamped tap index: {ssao}"
        );
    }

    #[test]
    fn ao_blur_is_depth_aware() {
        let blur = wgsl_source_blur();
        assert!(blur.contains("textureSampleLevel("), "{blur}");
        assert!(blur.contains("exp("), "{blur}");
        assert!(blur.contains("textureLoad("), "{blur}");
        assert!(blur.contains("@fragment"), "{blur}");
        assert!(blur.contains("fn fs_main("), "{blur}");
        assert!(
            !blur.contains("texture_multisampled_2d"),
            "the blur stage is single-sample: {blur}"
        );
    }
}
