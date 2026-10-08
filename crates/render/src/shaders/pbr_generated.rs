//! Forward-PBR shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module: the full OpenPBR
//! fragment skeleton (layer evaluators + `fs_main`) lives here as a Rust
//! string, and the BRDF math kernels are spliced in from
//! [`crate::shaders::math`] (single source of truth via `#[kernel]`).
//! The former handwritten `shaders/wgsl/pbr_*.wgsl` sources were deleted
//! after the `#[stage]` translation of `fs_main`; the
//! `fs_main_matches_legacy_shape` test pins the entry shape.
//!
//! Note: the vertex stage is shared with the g-buffer pass (same instance
//! transform), so [`wgsl_vertex_source`] reuses
//! [`crate::shaders::gbuffer_generated::wgsl_vertex_source`] instead of
//! duplicating it.

use super::interface::GbufferFragmentInput as FragmentInput;
use super::{
    ComparisonSampler, DepthTextureArray, DepthTextureCubeArray, OPENPBR_WGSL_NAME, Resource,
    ResourceKind, Sampler, ShaderModule, Texture2d, TextureCube, helpers, openpbr_material_decl,
    wgsl_decl,
};
use crate::renderer::{CameraUniform, GpuLight, LightingUniform, PerObjectGpu};
use crate::shaders::{gbuffer_generated, math};
use ornis_core::material::OpenPBRMaterial;
use ornis_macros::stage;

/// Forward-PBR vertex shader: instance transforms + world position.
///
/// Delegates to [`gbuffer_generated::wgsl_vertex_source`] — the two legacy
/// vertex files are byte-identical. Entry point name `vs_main` is kept for
/// compatibility with `create_pbr_pass`.
pub fn wgsl_vertex_source() -> String {
    gbuffer_generated::wgsl_vertex_source()
}

/// Group-0 `@binding` indices the forward-PBR fragment stage declares.
///
/// Bindings 7–10 are the split-sum IBL set (prefilter, irradiance, BRDF LUT,
/// sampler) added with the image-based light path.
const PBR_FRAGMENT_BINDINGS: &[u32] = &[0, 2, 3, 4, 5, 6, 7, 8, 9, 10];

/// Forward-PBR fragment shader: full OpenPBR evaluation.
///
/// Assembled as `{skeleton}\\n{kernels}`, exactly like the legacy
/// `shaders::pbr_fragment()`; entry point `fs_main` is kept.
pub fn wgsl_source() -> String {
    let kernels = [
        math::luminance::wgsl_source(),
        math::fresnel0_from_ior::wgsl_source(),
        math::fresnel_schlick::wgsl_source(),
        math::fresnel_schlick_vec::wgsl_source(),
        math::fresnel_f82_tint::wgsl_source(),
        math::ggx_ndf::wgsl_source(),
        math::ggx_ndf_aniso::wgsl_source(),
        math::openpbr_anisotropy::wgsl_source(),
        math::smith_ggx_correlated::wgsl_source(),
        math::smith_ggx_aniso::wgsl_source(),
        math::oren_nayar_brdf::wgsl_source(),
        math::base_diffuse_energy::wgsl_source(),
        math::thin_film_weight_mix::wgsl_source(),
        math::coated_emission::wgsl_source(),
        math::coat_base_darkening::wgsl_source(),
        math::coat_blend_darkened::wgsl_source(),
        math::thin_film_modulation::wgsl_source(),
        math::sheen_brdf::wgsl_source(),
        math::transmission_color_to_extinction::wgsl_source(),
        math::subsurface_brdf::wgsl_source(),
        math::srgb_to_linear::wgsl_source(),
        math::evaluate_ibl::wgsl_source(),
    ];
    let module = ShaderModule::new()
        .decl(CameraUniform::WGSL_SOURCE)
        .decl(wgsl_decl(GpuLight::WGSL_SOURCE))
        .decl(wgsl_decl(LightingUniform::WGSL_SOURCE))
        .decl(openpbr_material_decl())
        .resources(&PBR_RESOURCES, PBR_FRAGMENT_BINDINGS)
        .decl(wgsl_decl(FragmentInput::WGSL_SOURCE))
        .consts(helpers::wgsl_consts())
        .consts(math::wgsl_consts())
        .helper(helpers::wgsl_shared_helpers())
        .entry(fs_main::wgsl_source())
        .helpers(kernels);
    module.emit()
}

/// Forward-PBR fragment entry, translated by [`stage`](ornis_macros::stage):
/// full OpenPBR evaluation over the light array. DSL-only —
/// `camera`/`lighting`/`materials` globals declared via `#[wgsl(global)]`.
/// Layer evaluators
/// stay handwritten below and are called by name.
/// Forward-PBR resources as a context bundle (`ctx.materials`, …).
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct PbrContext {
    pub materials: Vec<OpenPBRMaterial>,
    pub camera: CameraUniform,
    pub lighting: LightingUniform,
    pub shadow_tex: DepthTextureArray,
    pub shadow_sampler: ComparisonSampler,
    pub shadow_cube_tex: DepthTextureCubeArray,
    pub prefilter_cube: TextureCube,
    pub irradiance_cube: TextureCube,
    pub brdf_lut: Texture2d,
    pub ibl_sampler: Sampler,
}

#[stage(fragment)]
fn fs_main(input: FragmentInput, ctx: Context<PbrContext>) -> super::Location<0, glam::Vec4> {
    let mat = ctx.materials[input.material_index];
    let n = normalize(input.world_normal);
    let v = normalize(ctx.camera.camera_pos.xyz - input.world_position);
    let nov = max(dot(n, v), EPS);
    let t = normalize(input.world_tangent);
    let b = cross(n, t);
    let base_weight = mat.base_params.x;
    let base_color = mat.base_color.rgb;
    let metalness = mat.base_params.z;
    let diffuse_roughness = mat.base_params.y;
    let specular_weight = mat.specular_params.x;
    let specular_roughness = specular_aa_roughness(mat.specular_params.y, n);
    let specular_ior = mat.specular_params.z;
    let specular_anisotropy = mat.specular_params.w;
    let specular_edge_tint = mat.specular_color.rgb;
    let transmission_weight = mat.transmission_params.x;
    let transmission_depth = mat.transmission_params.y;
    let transmission_dispersion_scale = mat.transmission_params.z;
    let transmission_dispersion_abbe = mat.transmission_params.w;
    let transmission_color = mat.transmission_color.rgb;
    let transmission_scatter = mat.transmission_scatter.rgb;
    let transmission_scatter_anisotropy = mat.transmission_scatter.a;
    let subsurface_weight = mat.subsurface_params.x;
    let subsurface_radius = mat.subsurface_params.y;
    let subsurface_radius_scale_r = mat.subsurface_params.z;
    let subsurface_scatter_anisotropy = mat.subsurface_params.w;
    let subsurface_color = mat.subsurface_color.rgb;
    let subsurface_radius_scale_g = mat.subsurface_radius_scale_gb.x;
    let subsurface_radius_scale_b = mat.subsurface_radius_scale_gb.y;
    let fuzz_weight = mat.fuzz_params.x;
    let fuzz_roughness = mat.fuzz_params.y;
    let fuzz_color = mat.fuzz_color.rgb;
    let coat_weight = mat.coat_params.x;
    let coat_roughness = mat.coat_params.y;
    let coat_anisotropy = mat.coat_params.z;
    let coat_darkening = mat.coat_params.w;
    let coat_color = mat.coat_color.rgb;
    let coat_ior = mat.coat_ior.x;
    let thin_film_weight = mat.thin_film_params.x;
    let thin_film_thickness_um = mat.thin_film_params.y;
    let thin_film_ior = mat.thin_film_params.z;
    let emission_luminance = mat.emission_params.x;
    let emission_color = mat.emission_color.rgb;
    let opacity = mat.geometry_params.x;
    let thin_walled = mat.geometry_params.y;
    let mut lo = Vec3::new(0.0, 0.0, 0.0);
    let thin_film_mod = thin_film_weight_mix(
        thin_film_weight,
        thin_film_modulation(nov, thin_film_ior, thin_film_thickness_um, 1.0),
    );
    let coat_darken = evaluate_coat_darkening(
        coat_weight,
        coat_darkening,
        coat_ior,
        metalness,
        base_color,
        base_weight,
        specular_weight,
        subsurface_weight,
        subsurface_color,
    );
    for i in 0u..ctx.lighting.light_count {
        let light = ctx.lighting.lights[i];
        let kind = light.kind.x;
        // Point/spot: vector from the surface to the light + range cutoff.
        // Directionals keep the legacy infinite-light path (pixel-identical).
        let to_light = light.position.xyz - input.world_position;
        let dist = length(to_light);
        let l_point = to_light / max(dist, EPS);
        let use_point = step(POINT_OR_SPOT_KIND_EDGE, kind);
        let l_dir = normalize(mix(normalize(light.direction.xyz), l_point, use_point));
        // smoothstep(edge0 == edge1) is undefined in WGSL — widen the
        // cutoff edge by 1%: coshaped but defined on every driver.
        let edge0 = light.params.x * RANGE_CUTOFF_INNER_FRACTION;
        let range_cut = mix(
            1.0,
            1.0 - smoothstep(edge0, light.params.x, dist),
            use_point,
        );
        let l = l_dir;
        let h = normalize(v + l + Vec3::new(0.0, 0.0, EPS));
        let light_color = light.color.rgb;
        let intensity = light.color.w;
        // Spot cone (kind == 2): full inside inner, soft edge to outer.
        let cos_theta = dot(
            normalize(light.direction.xyz),
            normalize(input.world_position - light.position.xyz),
        );
        let cone = mix(
            1.0,
            smoothstep(light.params.z, light.params.y, cos_theta),
            step(SPOT_KIND_EDGE, kind),
        );
        let attenuation = mix(1.0, 1.0 / max(dist * dist, EPS), use_point);
        let mut radiance = light_color * intensity * attenuation * range_cut * cone;
        // Point lights (kind == 1) sample the cube pool; dir/spot use
        // the 2D layers. `params.w` indexes the active pool.
        let is_point = step(POINT_OR_SPOT_KIND_EDGE, kind) * (1.0 - step(SPOT_KIND_EDGE, kind));
        // Shadow: project into the light's clip space and compare
        // against its map layer (hardware 2x2 PCF via the comparison
        // sampler). Unshadowed lights keep `params.w = -1.0` and skip
        // the lookup; single-mip depth needs no LOD, so the branch is
        // safe in non-uniform control flow.
        if light.params.w >= 0.0 {
            if is_point <= LIGHT_SELECTOR_SPLIT {
                let shadow_clip = light.shadow_vp
                    * Vec4::new(
                        input.world_position.x,
                        input.world_position.y,
                        input.world_position.z,
                        1.0,
                    );
                let shadow_ndc = shadow_clip.xyz / shadow_clip.w;
                // V is mirrored: rasterization puts NDC y+1 at texture
                // row 0 while `shadow_uv` v=0 reads from the top, so an
                // unmirrored lookup samples the mirrored texel (shadows
                // land overturned — darkness tests are blind to it).
                let shadow_uv = Vec2::new(
                    shadow_ndc.x * NDC_TO_UV_HALF + NDC_TO_UV_HALF,
                    NDC_TO_UV_HALF - shadow_ndc.y * NDC_TO_UV_HALF,
                );
                radiance = radiance
                    * textureSampleCompare(
                        ctx.shadow_tex,
                        ctx.shadow_sampler,
                        shadow_uv,
                        i32(light.params.w),
                        shadow_ndc.z - SHADOW_REF_BIAS,
                    );
            }
            if is_point > LIGHT_SELECTOR_SPLIT {
                // Cube sample: the hardware picks the face from the
                // fragment→light vector's major axis; the reference is
                // the 90°-perspective depth for that axis
                // (`SHADOW_CUBE_NEAR`, far = light range — the same
                // formula the face renders use, so no VP uniform).
                let to_frag = input.world_position - light.position.xyz;
                let major = max(max(abs(to_frag.x), abs(to_frag.y)), abs(to_frag.z));
                let far = max(light.params.x, 1.0);
                let denom = far - SHADOW_CUBE_NEAR;
                let cube_ref =
                    (far / denom) - (SHADOW_CUBE_NEAR * far) / (denom * major) - SHADOW_REF_BIAS;
                radiance = radiance
                    * textureSampleCompare(
                        ctx.shadow_cube_tex,
                        ctx.shadow_sampler,
                        to_frag,
                        i32(light.params.w),
                        cube_ref,
                    );
            }
        }
        let nol = max(dot(n, l), EPS);
        let noh = max(dot(n, h), EPS);
        let voh = max(dot(v, h), EPS);
        if nol <= EPS {
            continue;
        }
        let base_bsdf = evaluate_base_layer(
            n,
            v,
            l,
            h,
            nov,
            nol,
            noh,
            voh,
            mat,
            base_weight,
            base_color,
            metalness,
            diffuse_roughness,
            specular_weight,
            specular_roughness,
            specular_ior,
            specular_anisotropy,
            specular_edge_tint,
            t,
            b,
            thin_film_mod,
        ) * coat_darken;
        let coat_bsdf = evaluate_coat_layer(
            n,
            v,
            l,
            h,
            nov,
            nol,
            noh,
            voh,
            mat,
            coat_weight,
            coat_roughness,
            coat_anisotropy,
            coat_darkening,
            coat_ior,
            coat_color,
            metalness,
            base_color,
            base_weight,
            specular_weight,
            subsurface_weight,
            subsurface_color,
            t,
            b,
        );
        let fuzz_bsdf = evaluate_fuzz_layer(
            n,
            v,
            l,
            h,
            nov,
            nol,
            noh,
            voh,
            fuzz_weight,
            fuzz_roughness,
            fuzz_color,
        );
        let trans_bsdf = evaluate_transmission_layer(
            n,
            v,
            l,
            h,
            nov,
            nol,
            noh,
            voh,
            mat,
            transmission_weight,
            transmission_depth,
            transmission_dispersion_scale,
            transmission_dispersion_abbe,
            transmission_color,
            transmission_scatter,
            transmission_scatter_anisotropy,
            specular_ior,
            specular_roughness,
            specular_anisotropy,
            thin_walled,
        );
        let ss_bsdf = evaluate_subsurface_layer(
            n,
            v,
            l,
            nov,
            nol,
            subsurface_weight,
            subsurface_radius,
            subsurface_radius_scale_r,
            subsurface_radius_scale_g,
            subsurface_radius_scale_b,
            subsurface_scatter_anisotropy,
            subsurface_color,
        );
        let layer_bsdf = base_bsdf + coat_bsdf + fuzz_bsdf + trans_bsdf + ss_bsdf;
        lo = lo + layer_bsdf * radiance * nol;
    }
    let ambient =
        ctx.lighting.ambient_color.rgb * mix(base_color, base_color * specular_weight, metalness);
    let emission = evaluate_emission(
        emission_luminance,
        emission_color,
        coat_weight,
        coat_color,
        nov,
    );
    let f0_dielectric = fresnel0_from_ior(specular_ior);
    let f0 = mix(
        Vec3::new(f0_dielectric, f0_dielectric, f0_dielectric),
        base_color * base_weight,
        metalness,
    );
    let reflect_dir = n * (2.0 * dot(n, v)) - v;
    let prefiltered = textureSampleLevel(
        ctx.prefilter_cube,
        ctx.ibl_sampler,
        reflect_dir,
        specular_roughness * ctx.lighting.ibl_max_mip,
    );
    let irradiance = textureSample(ctx.irradiance_cube, ctx.ibl_sampler, n);
    let lut = textureSample(
        ctx.brdf_lut,
        ctx.ibl_sampler,
        Vec2::new(nov, specular_roughness),
    );
    let ibl = evaluate_ibl(
        prefiltered.rgb,
        irradiance.rgb,
        lut.r,
        lut.g,
        f0,
        base_color * (1.0 - metalness),
        ctx.lighting.ibl_weight,
    );
    // Unlit sprites (geometry `params[2]`, see `ShadingMode`): radiance is
    // the emission alone — no BRDF, light, shadow or IBL term. `select`
    // keeps the lit sum bit-identical when the flag is off.
    let full = ambient + lo + emission + ibl;
    let color = sanitize_hdr(select(full, emission, mat.geometry_params.z >= 0.5));
    return glam::Vec4::new(color, opacity);
}

/// Static view for naga validation in tests.
pub fn wgsl_source_static() -> String {
    wgsl_source()
}

/// Resource layout of the forward-PBR pass (vertex 0–1, fragment 0, 2–3).
/// Shared by `create_pbr_bind_group` and `create_forward_pass`, whose
/// handwritten layouts were identical. Type names come from the Rust side.
pub const PBR_RESOURCES: [Resource; 11] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        name: "camera",
        kind: ResourceKind::Uniform(CameraUniform::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::VERTEX,
        name: "per_objects",
        kind: ResourceKind::StorageReadArray(PerObjectGpu::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "materials",
        kind: ResourceKind::StorageReadArray(OPENPBR_WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 3,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "lighting",
        kind: ResourceKind::Uniform(LightingUniform::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_tex",
        kind: ResourceKind::TextureDepthArray,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 5,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_sampler",
        kind: ResourceKind::SamplerComparison,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 6,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_cube_tex",
        kind: ResourceKind::TextureDepthCubeArray,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 7,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "prefilter_cube",
        kind: ResourceKind::TextureCube,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 8,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "irradiance_cube",
        kind: ResourceKind::TextureCube,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 9,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "brdf_lut",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 10,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ibl_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
];

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
    fn pbr_generated_validates_with_naga() {
        assert_valid_wgsl("pbr_vertex", &wgsl_vertex_source());
        assert_valid_wgsl("pbr_fragment", &wgsl_source());
    }

    #[test]
    fn pbr_generated_contains_expected_bindings() {
        let vs = wgsl_vertex_source();
        assert!(vs.contains("@group(0) @binding(1) var<storage, read> per_objects"));
        assert!(vs.contains("fn vs_main("));
        let fs = wgsl_source();
        assert!(fs.contains("@group(0) @binding(3) var<uniform> lighting"));
        assert!(fs.contains("fn fs_main("));
        assert!(fs.contains("fn evaluate_base_layer"));
        assert!(!fs.contains("aces_tonemap("));
    }

    /// The translated fragment entry must keep the legacy shape: same
    /// signature, same layer-evaluator calls, same light loop with the
    /// early-out. (Byte-parity no longer applies — the generated entry is
    /// single-line with normalized int suffixes.)
    #[test]
    fn fs_main_matches_legacy_shape() {
        let entry = fs_main::wgsl_source();
        assert!(entry.starts_with("@fragment\nfn fs_main(input: FragmentInput)"));
        assert!(entry.contains("-> @location(0) vec4<f32>"));
        assert!(entry.contains("let mat = materials[input.material_index];"));
        assert!(entry.contains("for (var i: u32 = 0; i < lighting.light_count; i = i + 1)"));
        assert!(entry.contains("continue;"));
        assert!(entry.contains("let base_bsdf = evaluate_base_layer("));
        assert!(entry.contains(
            "let layer_bsdf = base_bsdf + coat_bsdf + fuzz_bsdf + trans_bsdf + ss_bsdf;"
        ));
        assert!(entry.contains("return vec4<f32>(color, opacity);"));
    }

    #[test]
    fn pbr_generated_parity_with_legacy_assembly() {
        // Vertex is shared with gbuffer (translated — see
        // `gbuffer_vertex_entry_matches_legacy_shape`); pin the sharing.
        assert_eq!(
            wgsl_vertex_source(),
            super::super::gbuffer_generated::wgsl_vertex_source()
        );
        // Fragment layouts still splice the derived declarations (Camera
        // raw — that declaration terminates with bare `}`).
        let src = wgsl_source();
        assert!(src.contains(CameraUniform::WGSL_SOURCE));
        assert!(src.contains(&wgsl_decl(GpuLight::WGSL_SOURCE)));
        assert!(src.contains(&wgsl_decl(LightingUniform::WGSL_SOURCE)));
        assert!(src.contains(openpbr_material_decl().as_str()));
    }

    /// Every table row's declaration appears across the assembled stages,
    /// and every row maps to a layout entry: shader and pipeline agree.
    #[test]
    fn pbr_resources_cover_stages_and_layout() {
        use super::super::{bgl_entry, resource_decl};
        let src = wgsl_vertex_source() + &wgsl_source();
        assert_eq!(PBR_RESOURCES.len(), 11);
        for r in PBR_RESOURCES {
            assert!(src.contains(&resource_decl(&r)), "missing {}", r.name);
            let e = bgl_entry(&r, false);
            assert_eq!((e.binding, e.visibility), (r.binding, r.visibility));
        }
    }
}
