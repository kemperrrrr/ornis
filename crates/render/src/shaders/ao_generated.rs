//! Screen-space AO shaders: classic SSAO and GTAO raw passes plus the
//! bilateral blur (Render path 2 shape, raw-WGSL bodies).
//!
//! Canonical source is this module: fullscreen-quad vertex boilerplate plus
//! three fragment bodies over the g-buffer depth and octahedral normals,
//! assembled with the shared [`ShaderModule`] machinery (types → resources
//! → consts → helpers → entry), so naga plus the pixel probes pin the
//! result like every other pass.
//!
//! Bodies stay raw WGSL (not `#[stage]`): the tap loops and the horizon
//! march need indexed loops and `continue`, which the DSL does not cover.
//! The single-sample `textureLoad(…, 0)` call text serves both modes: at 1x
//! it loads mip 0, at 4x it loads sample 0 (the lighting MSAA trick) — only
//! the type declarations differ (see [`wgsl_source_raw_for_samples`]).
//!
//! Depth source is the g-buffer `Depth` buffer (hardware `Depth32Float`,
//! linearized through `Camera::inv_view_proj` by the shared
//! `reconstruct_world_pos`); normals decode from the octahedral layer.

use super::ao_gtao::{AO_GTAO_BODY, AO_GTAO_HELPER};
use super::helpers;
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

/// Group-0 `@binding` indices the AO raw fragment stage declares.
const AO_RAW_FRAGMENT_BINDINGS: &[u32] = &[0, 1, 2, 3];

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

/// Group-0 `@binding` indices the blur fragment stage declares.
const AO_BLUR_FRAGMENT_BINDINGS: &[u32] = &[0, 1, 2, 3, 4];

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

/// Shared prologue of both raw kernels: background early-out, normal
/// decode, world-space center and view direction.
const AO_PROLOGUE: &str = r#"
    let dims = vec2<f32>(textureDimensions(depth_tex));
    let coord = vec2<u32>(input.uv * dims);
    let depth = textureLoad(depth_tex, coord, 0);
    if (depth >= CLEAR_DEPTH) {
        return vec4<f32>(1.0);
    }
    let normal_enc = textureLoad(normal_tex, coord, 0);
    let n = octahedral_decode(normal_enc.rg);
    let center = reconstruct_world_pos(input.uv, depth, camera);
"#;

/// Normal-aligned tangent the SSAO hemisphere needs. Split from the
/// prologue so the GTAO kernel does not carry an unused basis.
const AO_SSAO_BASIS: &str = r#"
    let t = normalize(cross(n, vec3<f32>(0.0, 1.0, 0.0)) + vec3<f32>(0.0, 0.0, EPS));
    let b = cross(n, t);
"#;

/// Classic SSAO body: 8 spiral taps on the normal hemisphere, reprojected
/// through `view_proj`; a tap occludes when the scene in front of the probe
/// point is nearer the eye than the probe itself (minus bias), weighted by
/// `(1 - range / radius)`.
const AO_SSAO_BODY: &str = r#"
    let angle = ao_interleaved_gradient(coord) * 6.28318530718;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    var occluded = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let fi = f32(i) + 0.5;
        let tap_r = fi / 8.0;
        let tap_a = fi * 2.399963;
        var tap = vec2<f32>(cos(tap_a), sin(tap_a)) * tap_r;
        tap = vec2<f32>(tap.x * cos_a - tap.y * sin_a, tap.x * sin_a + tap.y * cos_a);
        let dir = normalize(t * tap.x + b * tap.y + n * 0.75);
        let probe = center + dir * ao_params.radius;
        let clip = camera.view_proj * vec4<f32>(probe, 1.0);
        let ndc = clip.xyz / max(clip.w, EPS);
        if (ndc.z > 1.0) {
            continue;
        }
        let suv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) {
            continue;
        }
        let sdepth = textureLoad(depth_tex, vec2<u32>(suv * dims), 0);
        if (sdepth >= CLEAR_DEPTH) {
            continue;
        }
        let sample_pos = reconstruct_world_pos(suv, sdepth, camera);
        let range = length(probe - sample_pos);
        if (range < EPS) {
            continue;
        }
        let eye_probe = length(camera.camera_pos.xyz - probe);
        let eye_sample = length(camera.camera_pos.xyz - sample_pos);
        if (eye_sample < eye_probe - ao_params.bias) {
            occluded += max(0.0, 1.0 - range / ao_params.radius);
        }
    }
    let ao = clamp(1.0 - (occluded / 8.0) * ao_params.intensity, 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
"#;

/// Eye distance in meters. Raw NDC is not a depth the bilateral sigma
/// (meters) can see; this is the GPU form of [`crate::ao::linear_view_depth`].
const AO_BLUR_HELPER: &str = r#"
fn ao_linear_depth(uv: vec2<f32>, depth: f32, camera: Camera) -> f32 {
    let world = reconstruct_world_pos(uv, depth, camera);
    return length(camera.camera_pos.xyz - world);
}
"#;

/// Bilateral blur body: 4-tap cross over the half-res raw factor, weighted
/// by the metric eye-distance step (the [`crate::ao::bilateral_weight`]
/// curve), so occlusion never bleeds over silhouette edges.
const AO_BLUR_BODY: &str = r#"
    let raw = textureSampleLevel(ao_raw_tex, blur_sampler, input.uv, 0.0).r;
    let dims_u = textureDimensions(depth_tex);
    let dims = vec2<f32>(dims_u);
    let center_coord = min(vec2<u32>(input.uv * dims), dims_u - vec2<u32>(1u));
    let center_z = ao_linear_depth(input.uv, textureLoad(depth_tex, center_coord, 0), camera);
    let texel = 1.0 / vec2<f32>(textureDimensions(ao_raw_tex));
    let sigma = max(ao_params.blur_sigma, EPS);
    var acc = raw;
    var weight = 1.0;
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(1.5, 0.0),
        vec2<f32>(-1.5, 0.0),
        vec2<f32>(0.0, 1.5),
        vec2<f32>(0.0, -1.5),
    );
    for (var i = 0u; i < 4u; i = i + 1u) {
        let off = offsets[i] * texel;
        let sample_uv = input.uv + off;
        let clamped = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        let sample_coord = min(vec2<u32>(clamped * dims), dims_u - vec2<u32>(1u));
        let sample_z = ao_linear_depth(clamped, textureLoad(depth_tex, sample_coord, 0), camera);
        let delta = (center_z - sample_z) / sigma;
        let w = exp(-delta * delta);
        acc += textureSampleLevel(ao_raw_tex, blur_sampler, sample_uv, 0.0).r * w;
        weight += w;
    }
    let ao = acc / weight;
    return vec4<f32>(ao, ao, ao, 1.0);
"#;

/// Per-pixel hash shared by both kernels (Jimenez interleaved gradient
/// noise): decorrelates the tap spiral into noise instead of banding, with
/// no noise texture to upload or bind.
const AO_HASH_HELPER: &str = r#"
fn ao_interleaved_gradient(pixel: vec2<u32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(vec2<f32>(pixel), magic.xy)));
}
"#;

/// Fullscreen vertex entry shared by all AO stages (triangle-strip quad).
const AO_VERTEX_ENTRY: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> QuadVertexOutput {
    return QuadVertexOutput(quad[vertex_index], uvs[vertex_index]);
}
"#;

/// Raw fragment entry for `method`: prologue, the SSAO tangent basis when
/// that kernel is selected, and the kernel body.
fn ao_raw_entry(method: crate::ao::AoMethod) -> String {
    let (basis, body) = if method.is_ssao() {
        (AO_SSAO_BASIS, AO_SSAO_BODY)
    } else {
        ("", AO_GTAO_BODY)
    };
    format!(
        "@fragment\nfn fs_main(input: QuadVertexOutput) -> @location(0) vec4<f32> {{{prologue}{basis}{body}}}\n",
        prologue = AO_PROLOGUE,
        basis = basis,
        body = body,
    )
}

/// Blur fragment entry.
fn ao_blur_entry() -> String {
    format!(
        "@fragment\nfn fs_main(input: QuadVertexOutput) -> @location(0) vec4<f32> {{{body}}}\n",
        body = AO_BLUR_BODY
    )
}

/// Full WGSL source for the AO raw fragment stage at a sample count: at 1x
/// byte-identical to the plain source; at 4x depth and the normal declare
/// the multisampled types (both stay per-sample: depth has no resolve
/// target and the normal must not go through the hardware box filter).
pub fn wgsl_source_raw_for_samples(method: crate::ao::AoMethod, sample_count: u32) -> String {
    let module = ShaderModule::new()
        .decl(wgsl_decl(CameraUniform::WGSL_SOURCE))
        .decl(wgsl_decl(AoUniform::WGSL_SOURCE))
        .resources_where(&AO_RAW_RESOURCES, AO_RAW_FRAGMENT_BINDINGS, |r| {
            ao_per_sample_flag(sample_count, ao_keeps_per_sample(r.name, &r.kind))
        })
        .consts(naga_ir::const_block(&STANDARD_QUAD, &STANDARD_UVS))
        .consts(crate::shaders::math::wgsl_consts())
        .decl(wgsl_decl(QuadVertexOutput::WGSL_SOURCE))
        .helper(helpers::wgsl_consts())
        .helper(helpers::wgsl_lighting_decode())
        .helper(AO_HASH_HELPER);
    let module = if method.is_ssao() {
        module
    } else {
        module.helper(AO_GTAO_HELPER)
    };
    module.entry(ao_raw_entry(method)).emit()
}

/// Full WGSL source for the AO raw fragment stage (single-sample).
pub fn wgsl_source_raw(method: crate::ao::AoMethod) -> String {
    wgsl_source_raw_for_samples(method, 1)
}

/// Full WGSL source for the bilateral blur stage (always single-sample:
/// the pooled AO targets never multisample).
pub fn wgsl_source_blur() -> String {
    ShaderModule::new()
        .decl(wgsl_decl(CameraUniform::WGSL_SOURCE))
        .decl(wgsl_decl(AoUniform::WGSL_SOURCE))
        .resources(&AO_BLUR_RESOURCES, AO_BLUR_FRAGMENT_BINDINGS)
        .consts(naga_ir::const_block(&STANDARD_QUAD, &STANDARD_UVS))
        .consts(crate::shaders::math::wgsl_consts())
        .decl(wgsl_decl(QuadVertexOutput::WGSL_SOURCE))
        .helper(helpers::wgsl_consts())
        .helper(helpers::wgsl_lighting_decode())
        .helper(AO_BLUR_HELPER)
        .entry(ao_blur_entry())
        .emit()
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
            "ao_raw_msaa",
            &wgsl_source_raw_for_samples(AoMethod::Gtao, 4),
        );
    }

    #[test]
    fn ao_raw_kernels_spell_their_math() {
        let ssao = wgsl_source_raw(AoMethod::Ssao);
        assert!(ssao.contains("ao_interleaved_gradient("), "{ssao}");
        assert!(ssao.contains("reconstruct_world_pos(input.uv, depth, camera)"));
        assert!(ssao.contains("octahedral_decode(normal_enc.rg)"));
        assert!(ssao.contains("ao_params.radius"), "{ssao}");
        assert!(ssao.contains("ao_params.bias"), "{ssao}");
        assert!(ssao.contains("ao_params.intensity"), "{ssao}");
        assert!(ssao.contains("i < 8u"), "ssao taps: {ssao}");
        assert!(!ssao.contains("horizon"), "ssao must not horizon-march");
        let gtao = wgsl_source_raw(AoMethod::Gtao);
        assert!(gtao.contains("horizon_cos_pos"), "{gtao}");
        assert!(gtao.contains("horizon_cos_neg"), "both slice sides: {gtao}");
        assert!(gtao.contains("slice_dir"), "{gtao}");
        assert!(gtao.contains("s < 2u"), "gtao slices: {gtao}");
        assert!(gtao.contains("j <= 4u"), "gtao steps: {gtao}");
        assert!(gtao.contains("fn gtao_arc("), "{gtao}");
        assert!(
            gtao.contains("0.25 * (-cos(2.0 * h - n) + cos(n) + 2.0 * h * sin(n))"),
            "XeGTAO integral: {gtao}"
        );
        assert!(
            gtao.contains("1.0 - r / max(radius, EPS)"),
            "radius falloff: {gtao}"
        );
        assert!(gtao.contains("input.uv - off"), "negative horizon: {gtao}");
        assert!(!gtao.contains("ao_params.bias"), "gtao needs no depth bias");
        assert!(
            !ssao.contains("gtao_arc"),
            "ssao must not integrate horizons"
        );
    }

    #[test]
    fn ao_blur_is_depth_aware() {
        let blur = wgsl_source_blur();
        assert!(blur.contains("ao_raw_tex"), "{blur}");
        assert!(blur.contains("blur_sampler"), "{blur}");
        assert!(blur.contains("ao_params.blur_sigma"), "{blur}");
        assert!(blur.contains("exp(-delta * delta)"), "{blur}");
        assert!(blur.contains("fn ao_linear_depth("), "{blur}");
        assert!(
            blur.contains("reconstruct_world_pos(uv, depth, camera)"),
            "blur must linearize through inv_view_proj, not raw NDC: {blur}"
        );
        assert!(
            !blur.contains("center_depth - textureLoad"),
            "raw NDC difference must not weight the blur: {blur}"
        );
    }

    #[test]
    fn ao_resources_cover_shader_and_layout() {
        use super::super::{bgl_entry, resource_decl};
        assert_eq!(AO_RAW_RESOURCES.len(), 4);
        for r in AO_RAW_RESOURCES {
            assert!(
                wgsl_source_raw(AoMethod::Gtao).contains(&resource_decl(&r)),
                "missing {}",
                r.name
            );
            let e = bgl_entry(&r, false);
            assert_eq!((e.binding, e.visibility), (r.binding, r.visibility));
        }
        assert_eq!(AO_BLUR_RESOURCES.len(), 5);
        for r in AO_BLUR_RESOURCES {
            assert!(
                wgsl_source_blur().contains(&resource_decl(&r)),
                "missing {}",
                r.name
            );
        }
        assert!(matches!(
            AO_RAW_RESOURCES[3].min_size,
            Some(s) if s == std::mem::size_of::<AoUniform>() as u64
        ));
    }

    #[test]
    fn ao_msaa_keeps_depth_and_normal_per_sample() {
        let src = wgsl_source_raw_for_samples(AoMethod::Ssao, 4);
        assert!(src.contains("texture_depth_multisampled_2d"), "{src}");
        assert!(src.contains("texture_multisampled_2d<f32>"), "{src}");
        assert!(src.contains("textureLoad(normal_tex"), "{src}");
        let plain = wgsl_source_raw(AoMethod::Ssao);
        assert!(!plain.contains("multisampled"), "{plain}");
    }
}
