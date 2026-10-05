//! Distance-fog shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module: the fullscreen-quad
//! vertex boilerplate and the fog-mix fragment skeleton live here as `#[stage]`
//! entries, and the world-position reconstruction splices from
//! [`crate::shaders::helpers`] (the same `reconstruct_world_pos` lighting
//! uses — single source of truth). No handwritten WGSL.
//!
//! Depth source: the g-buffer `Depth` buffer (hardware `Depth32Float`,
//! non-linear). It is linearized through `Camera::inv_view_proj` and the
//! Euclidean distance to `Camera::camera_pos` feeds the exponential mix
//! `color + (fog.color - color) * (1 - exp(-density * depth))` — the same
//! math as [`crate::frame_passes::apply_fog`]. The `world_position` layer
//! is NOT the source: it only stores xy (`Rg16Float`, z comes from depth
//! anyway), so sampling depth directly is authoritative.
//!
//! Placement: this stage always reads `hdr` and writes `target` (see
//! [`crate::frame_passes::FogPlacement`] — after `composite` is
//! recommended; the composite clear discards a before-`composite` write).

use super::helpers;
use super::interface::HdrFragmentOut as QuadVertexOutput;
use super::{
    DepthTexture, Resource, ResourceKind, STANDARD_QUAD, STANDARD_UVS, Sampler, ShaderModule,
    Texture2d, naga_ir, wgsl_decl,
};
use crate::renderer::{CameraUniform, FogUniform};
use ornis_macros::stage;

/// Fog fragment uniforms as a context bundle (`ctx.camera`, `ctx.fog_params`).
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct FogContext {
    /// Frame camera (view reconstruction + eye position).
    pub camera: CameraUniform,
    /// Fog color + density.
    pub fog_params: FogUniform,
}

/// Sampled fog maps as a separate bundle (`maps.hdr_tex`, …).
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct FogMaps {
    /// Deferred HDR color layer.
    pub hdr_tex: Texture2d,
    /// G-buffer hardware depth.
    pub depth_tex: DepthTexture,
    /// Linear clamp sampler for the HDR fetch.
    pub fog_sampler: Sampler,
}

/// Fog vertex entry, translated by [`stage`](ornis_macros::stage).
/// DSL-only — replaced by `vs_main::wgsl_source()`.
#[stage(vertex)]
fn vs_main(
    vertex_index: super::VertexIndex,
    consts: Context<super::QuadConsts>,
) -> QuadVertexOutput {
    return QuadVertexOutput {
        clip_position: consts.quad[vertex_index],
        uv: consts.uvs[vertex_index],
    };
}

/// Fog fragment entry, translated by [`stage`](ornis_macros::stage):
/// HDR fetch + depth linearization + exponential fog mix. DSL-only —
/// resource bundles (`ctx: FogContext`, `maps: FogMaps`).
#[stage(fragment)]
fn fs_main(
    input: QuadVertexOutput,
    ctx: Context<FogContext>,
    maps: Context<FogMaps>,
) -> super::Location<0, glam::Vec4> {
    let hdr = textureSampleLevel(maps.hdr_tex, maps.fog_sampler, input.uv, 0.0).rgb;
    let depth = textureLoad(
        maps.depth_tex,
        UVec2::new(input.uv * Vec2::new(textureDimensions(maps.depth_tex))),
        0,
    );
    let world_pos = reconstruct_world_pos(input.uv, depth, ctx.camera);
    let dist = length(ctx.camera.camera_pos.xyz - world_pos);
    let factor = 1.0 - exp(0.0 - ctx.fog_params.density * dist);
    let mixed = hdr + (ctx.fog_params.color - hdr) * factor;
    return glam::Vec4::new(mixed, 1.0);
}

/// Resource layout of the fog pass (what `Renderer3D::create_fog_pass`
/// builds its layout from). The uniform carries the explicit minimum size.
pub const FOG_RESOURCES: [Resource; 5] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "hdr_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "fog_sampler",
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
        name: "camera",
        kind: ResourceKind::Uniform(CameraUniform::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "fog_params",
        kind: ResourceKind::Uniform(FogUniform::WGSL_NAME),
        min_size: Some(std::mem::size_of::<FogUniform>() as u64),
    },
];

/// Full WGSL source for the fog fragment stage, assembled from Rust.
pub fn wgsl_source() -> String {
    wgsl_source_for_samples(1)
}

/// Full WGSL source for the fog fragment stage at a sample count: at 1x
/// byte-identical to [`wgsl_source`]; in MSAA mode the g-buffer depth
/// declares the multisampled type (it has no resolve target — see
/// [`super::resource_stays_multisampled`]) while HDR stays single-sample.
pub fn wgsl_source_for_samples(sample_count: u32) -> String {
    ShaderModule::new()
        .decl(wgsl_decl(CameraUniform::WGSL_SOURCE))
        .decl(wgsl_decl(FogUniform::WGSL_SOURCE))
        .resources_for_samples(&FOG_RESOURCES, &[0, 1, 2, 3, 4], sample_count)
        .consts(naga_ir::const_block(&STANDARD_QUAD, &STANDARD_UVS))
        .decl(wgsl_decl(QuadVertexOutput::WGSL_SOURCE))
        .helper(helpers::wgsl_lighting_decode())
        .entry(fs_main::wgsl_source())
        .emit()
}

/// Full WGSL source for the fog vertex stage (fullscreen quad).
pub fn wgsl_vertex_source() -> String {
    ShaderModule::new()
        .consts(naga_ir::const_block(&STANDARD_QUAD, &STANDARD_UVS))
        .decl(wgsl_decl(QuadVertexOutput::WGSL_SOURCE))
        .entry(vs_main::wgsl_source())
        .emit()
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
    fn fog_generated_validates_with_naga() {
        assert_valid_wgsl("fog_vertex", &wgsl_vertex_source());
        assert_valid_wgsl("fog_fragment", &wgsl_source());
    }

    #[test]
    fn fog_generated_contains_expected_bindings() {
        let src = wgsl_source();
        assert!(src.contains("@group(0) @binding(0) var hdr_tex"));
        assert!(src.contains("@group(0) @binding(1) var fog_sampler"));
        assert!(src.contains("@group(0) @binding(2) var depth_tex"));
        assert!(src.contains("@group(0) @binding(3) var<uniform> camera"));
        assert!(src.contains("@group(0) @binding(4) var<uniform> fog_params"));
        assert!(src.contains("fn fs_main("));
        assert!(src.contains("fn reconstruct_world_pos"));
        assert!(wgsl_vertex_source().contains("fn vs_main("));
    }

    #[test]
    fn fog_fragment_entry_matches_pinned_math() {
        // The GPU mix must spell the same math as `apply_fog`:
        // depth linearization, exponential factor, color mix.
        let entry = fs_main::wgsl_source();
        assert!(entry.starts_with("@fragment\nfn fs_main(input: QuadVertexOutput)"));
        assert!(entry.contains("-> @location(0) vec4<f32>"));
        assert!(entry.contains(
            "textureLoad(depth_tex, vec2<u32>(input.uv * vec2<f32>(textureDimensions(depth_tex))), 0)"
        ));
        assert!(entry.contains("reconstruct_world_pos(input.uv, depth, camera)"));
        assert!(entry.contains("length(camera.camera_pos.xyz - world_pos)"));
        assert!(entry.contains("exp("));
        assert!(entry.contains("fog_params.density"));
        assert!(entry.contains("fog_params.color"));
        assert!(entry.contains("return vec4<f32>(mixed, 1.0);"));
    }

    /// Every table row's declaration appears in the assembled shader, and
    /// every row maps to a layout entry: shader and pipeline agree.
    #[test]
    fn fog_resources_cover_shader_and_layout() {
        use super::super::{bgl_entry, resource_decl};
        let src = wgsl_source();
        assert_eq!(FOG_RESOURCES.len(), 5);
        for r in FOG_RESOURCES {
            assert!(src.contains(&resource_decl(&r)), "missing {}", r.name);
            let e = bgl_entry(&r, false);
            assert_eq!((e.binding, e.visibility), (r.binding, r.visibility));
        }
        assert!(matches!(
            FOG_RESOURCES[4].min_size,
            Some(s) if s == std::mem::size_of::<FogUniform>() as u64
        ));
    }

    /// The MSAA source validates with naga: depth goes multisampled (loaded
    /// sample 0, like lighting), HDR stays single-sample (it binds the
    /// already-resolved layer).
    #[test]
    fn fog_msaa_source_validates_with_multisampled_depth() {
        let src = wgsl_source_for_samples(4);
        assert_valid_wgsl("fog_fragment_msaa", &src);
        assert!(src.contains("texture_depth_multisampled_2d"), "{src}");
        assert!(src.contains("textureSampleLevel"), "{src}");
        let plain = wgsl_source();
        assert!(!plain.contains("multisampled"), "{plain}");
    }
}
