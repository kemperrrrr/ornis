//! Sampled glTF material textures feeding the forward-PBR evaluator.
//!
//! The upload side ([`crate::textures`]) owns images, formats and CPU
//! mirrors; this module owns the shader side: three pure `#[wgsl_fn]`
//! combinators (one per [`crate::textures::TextureRole`]) plus a dedicated
//! forward-fragment entry (`fs_main_textured`) that samples the bound
//! textures with the material UV (`input.uv`) and folds the texels into
//! the scalar factors before the shared layer evaluators run.
//!
//! The legacy forward entry ([`crate::shaders::pbr_generated::fs_main`])
//! is untouched: textured materials are a new path (new resource table,
//! new pipeline built by
//! [`Renderer3D::ensure_textured_forward`](crate::renderer::Renderer3D::ensure_textured_forward)),
//! so untextured renders stay pixel-identical by construction. The
//! deferred-lighting pass takes no bindings here on purpose: the g-buffer
//! carries no material UV (only screen UV plus ids), so sampling
//! per-material textures there would need a new MRT and break the
//! pixel-parity gate — the combinators below are ready for it once UV
//! storage lands.
//!
//! sRGB/linear contract (mirrors [`crate::textures`]): color roles arrive
//! as `Rgba8UnormSrgb`, so the hardware already decoded the sample to
//! linear light and the multiply stays in linear; the data role arrives
//! as `Rgba8Unorm` (green holds roughness, blue holds metallic) and scales
//! linearly. The scalar `is_metallic` switch never moves — the texture
//! only modulates the metallic factor. Unbound slots sample a 1x1 white
//! fallback (`x * 1.0 == x`, IEEE-exact), which is the legacy no-op.

use super::interface::GbufferFragmentInput as FragmentInput;
use super::{
    ComparisonSampler, DepthTextureArray, DepthTextureCubeArray, OPENPBR_WGSL_NAME, Resource,
    ResourceKind, Sampler, ShaderModule, Texture2d, TextureCube, helpers, openpbr_material_decl,
    wgsl_decl,
};
use crate::renderer::{CameraUniform, GpuLight, LightingUniform, PerObjectGpu};
use crate::shaders::math;
use ornis_core::material::OpenPBRMaterial;
use ornis_macros::stage;

/// Folds a sampled albedo/emissive texel (linear RGB) into a scalar color.
///
/// `map_rgb` is the `.rgb` swizzle of a `textureSample` of a color-role
/// texture (hardware sRGB-decoded, so linear); the multiply stays in
/// linear light. White is the exact no-op.
#[ornis_macros::wgsl_fn]
fn apply_base_color_map(base_color: glam::Vec3, map_rgb: glam::Vec3) -> glam::Vec3 {
    return base_color * map_rgb;
}

/// Folds a sampled metallic-roughness texel into the scalar factors.
///
/// `map_gb` is the `.gb` swizzle of a `textureSample` of the data-role
/// texture (linear, never sRGB-decoded): green scales roughness, blue
/// scales metallic (the glTF factor-times-texel rule). `(1, 1)` is the
/// exact no-op. The scalar `is_metallic` switch is untouched.
#[ornis_macros::wgsl_fn]
fn apply_metallic_roughness_map(roughness: f32, metallic: f32, map_gb: glam::Vec2) -> glam::Vec2 {
    return glam::Vec2::new(roughness * map_gb.x, metallic * map_gb.y);
}

/// Folds a sampled emissive texel (linear RGB) into the emission color.
///
/// Same linear-multiply shape as [`apply_base_color_map`]; luminance stays
/// scalar. White is the exact no-op.
#[ornis_macros::wgsl_fn]
fn apply_emissive_map(emission_color: glam::Vec3, map_rgb: glam::Vec3) -> glam::Vec3 {
    return emission_color * map_rgb;
}

/// Multiplies a scalar base color by a linear map texel (CPU mirror).
///
/// Pure software twin of [`apply_base_color_map`] over the
/// [`crate::textures::sample_albedo_linear`] output, so tests pin the
/// combine without an adapter.
pub fn apply_base_color_map_cpu(base_color: [f32; 3], map_rgb: [f32; 3]) -> [f32; 3] {
    [
        base_color[0] * map_rgb[0],
        base_color[1] * map_rgb[1],
        base_color[2] * map_rgb[2],
    ]
}

/// Scales roughness/metallic by a linear `(G, B)` texel (CPU mirror).
///
/// Pure software twin of [`apply_metallic_roughness_map`] over the
/// [`crate::textures::sample_metallic_roughness`] output.
pub fn apply_metallic_roughness_map_cpu(
    roughness: f32,
    metallic: f32,
    map_gb: [f32; 2],
) -> [f32; 2] {
    [roughness * map_gb[0], metallic * map_gb[1]]
}

/// Multiplies an emission color by a linear map texel (CPU mirror).
///
/// Pure software twin of [`apply_emissive_map`].
pub fn apply_emissive_map_cpu(emission_color: [f32; 3], map_rgb: [f32; 3]) -> [f32; 3] {
    [
        emission_color[0] * map_rgb[0],
        emission_color[1] * map_rgb[1],
        emission_color[2] * map_rgb[2],
    ]
}

/// The three texture-combine sources concatenated (consts excluded).
///
/// Spliced only into [`wgsl_source_textured`]; the legacy assemblies never
/// see these symbols.
pub fn wgsl_material_texture_helpers() -> String {
    [
        apply_base_color_map::wgsl_source(),
        apply_metallic_roughness_map::wgsl_source(),
        apply_emissive_map::wgsl_source(),
    ]
    .join("\n")
}

/// Textured-forward resources as a context bundle (`ctx.base_color_tex`, …).
///
/// The first seven fields mirror
/// [`crate::shaders::pbr_generated::PbrContext`] (same names, same kinds);
/// the last four are the material textures (one shared filtering sampler —
/// all roles share the upload defaults today, see
/// [`crate::textures::sampler_descriptor_for_role`]).
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct PbrTexturedContext {
    pub materials: Vec<OpenPBRMaterial>,
    pub camera: CameraUniform,
    pub lighting: LightingUniform,
    pub shadow_tex: DepthTextureArray,
    pub shadow_sampler: ComparisonSampler,
    pub shadow_cube_tex: DepthTextureCubeArray,
    pub base_color_tex: Texture2d,
    pub metallic_roughness_tex: Texture2d,
    pub emissive_tex: Texture2d,
    pub material_sampler: Sampler,
    pub prefilter_cube: TextureCube,
    pub irradiance_cube: TextureCube,
    pub brdf_lut: Texture2d,
    pub ibl_sampler: Sampler,
}

/// Resource layout of the textured-forward pass: the legacy seven rows
/// verbatim plus one texture row per role and the shared sampler.
/// Type names come from the Rust side.
pub const TEXTURED_PBR_RESOURCES: [Resource; 15] = [
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
        name: "base_color_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 8,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "metallic_roughness_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 9,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "emissive_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 10,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "material_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 11,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "prefilter_cube",
        kind: ResourceKind::TextureCube,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 12,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "irradiance_cube",
        kind: ResourceKind::TextureCube,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 13,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "brdf_lut",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 14,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ibl_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
];

/// Textured forward-PBR fragment entry, translated by
/// [`stage`](ornis_macros::stage): the legacy evaluation with the three
/// role samples folded in up front (material UV, factor-times-texel).
/// DSL-only — globals declared via the [`PbrTexturedContext`] bundle.
#[stage(fragment)]
fn fs_main_textured(
    input: FragmentInput,
    ctx: Context<PbrTexturedContext>,
) -> super::Location<0, glam::Vec4> {
    let mat = ctx.materials[input.material_index];
    let n = normalize(input.world_normal);
    let v = normalize(ctx.camera.camera_pos.xyz - input.world_position);
    let nov = max(dot(n, v), EPS);
    let t = normalize(input.world_tangent);
    let b = cross(n, t);
    let base_weight = mat.base_params.x;
    let base_map = textureSample(ctx.base_color_tex, ctx.material_sampler, input.uv);
    let base_color = apply_base_color_map(mat.base_color.rgb, base_map.rgb);
    let mr_sample = textureSample(ctx.metallic_roughness_tex, ctx.material_sampler, input.uv);
    let mr_factors =
        apply_metallic_roughness_map(mat.specular_params.y, mat.base_params.z, mr_sample.gb);
    let metalness = mr_factors.y;
    let diffuse_roughness = mat.base_params.y;
    let specular_weight = mat.specular_params.x;
    let specular_roughness = mr_factors.x;
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
    let emissive_map = textureSample(ctx.emissive_tex, ctx.material_sampler, input.uv);
    let emission_color = apply_emissive_map(mat.emission_color.rgb, emissive_map.rgb);
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
        let use_point = step(0.5, kind);
        let l_dir = normalize(mix(normalize(light.direction.xyz), l_point, use_point));
        // smoothstep(edge0 == edge1) is undefined in WGSL — widen the
        // cutoff edge by 1%: coshaped but defined on every driver.
        let edge0 = light.params.x * 0.99;
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
            step(1.5, kind),
        );
        let attenuation = mix(1.0, 1.0 / max(dist * dist, EPS), use_point);
        let mut radiance = light_color * intensity * attenuation * range_cut * cone;
        // Point lights (kind == 1) sample the cube pool; dir/spot use
        // the 2D layers. `params.w` indexes the active pool.
        let is_point = step(0.5, kind) * (1.0 - step(1.5, kind));
        // Shadow: project into the light's clip space and compare
        // against its map layer (hardware 2x2 PCF via the comparison
        // sampler). Unshadowed lights keep `params.w = -1.0` and skip
        // the lookup; single-mip depth needs no LOD, so the branch is
        // safe in non-uniform control flow.
        if light.params.w >= 0.0 {
            if is_point <= 0.5 {
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
                let shadow_uv = Vec2::new(shadow_ndc.x * 0.5 + 0.5, 0.5 - shadow_ndc.y * 0.5);
                radiance = radiance
                    * textureSampleCompare(
                        ctx.shadow_tex,
                        ctx.shadow_sampler,
                        shadow_uv,
                        i32(light.params.w),
                        shadow_ndc.z - SHADOW_REF_BIAS,
                    );
            }
            if is_point > 0.5 {
                // Cube sample: the hardware picks the face from the
                // fragment→light vector's major axis; the reference is
                // the 90°-perspective depth for that axis
                // (`SHADOW_CUBE_NEAR`, far = light range — the same
                // formula the face renders use, so no VP uniform).
                let to_frag = input.world_position - light.position.xyz;
                let major = max(max(abs(to_frag.x), abs(to_frag.y)), abs(to_frag.z));
                let far = max(light.params.x, 1.0);
                let denom = far - 0.1;
                let cube_ref = (far / denom) - (0.1 * far) / (denom * major) - SHADOW_REF_BIAS;
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
    let color = ambient + lo + emission + ibl;
    return glam::Vec4::new(color, opacity);
}

/// Textured forward-PBR fragment shader: the legacy evaluation plus the
/// three role samples, assembled exactly like
/// [`crate::shaders::pbr_generated::wgsl_source`] (same kernel set).
/// Entry point `fs_main_textured` is kept distinct from the legacy
/// `fs_main` so both pipelines can coexist.
pub fn wgsl_source_textured() -> String {
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
        .resources(
            &TEXTURED_PBR_RESOURCES,
            &[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        )
        .decl(wgsl_decl(FragmentInput::WGSL_SOURCE))
        .consts(helpers::wgsl_consts())
        .helper(helpers::wgsl_shared_helpers())
        .helper(wgsl_material_texture_helpers())
        .entry(fs_main_textured::wgsl_source())
        .helpers(kernels);
    module.emit()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textures::{
        CpuImage, TextureRole, sample_albedo_linear, sample_metallic_roughness,
        texture_format_for_role,
    };

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
    fn combinator_sources_keep_signatures() {
        assert!(
            apply_base_color_map::wgsl_source()
                .starts_with("fn apply_base_color_map(base_color: vec3<f32>, map_rgb: vec3<f32>)")
        );
        assert!(apply_metallic_roughness_map::wgsl_source().starts_with(
            "fn apply_metallic_roughness_map(roughness: f32, metallic: f32, map_gb: vec2<f32>)"
        ));
        assert!(
            apply_emissive_map::wgsl_source().starts_with(
                "fn apply_emissive_map(emission_color: vec3<f32>, map_rgb: vec3<f32>)"
            )
        );
        // DSL spellings must not leak into WGSL.
        for src in [
            apply_base_color_map::wgsl_source(),
            apply_metallic_roughness_map::wgsl_source(),
            apply_emissive_map::wgsl_source(),
        ] {
            assert!(!src.contains("Vec3"), "glam spelling leaked: {src}");
            assert!(!src.contains("glam"), "glam spelling leaked: {src}");
        }
    }

    #[test]
    fn textured_assembly_validates_with_naga() {
        assert_valid_wgsl(
            "textured_forward_vertex",
            &super::super::pbr_generated::wgsl_vertex_source(),
        );
        assert_valid_wgsl("textured_forward_fragment", &wgsl_source_textured());
    }

    #[test]
    fn textured_entry_samples_all_roles_before_lighting() {
        let entry = fs_main_textured::wgsl_source();
        assert!(entry.starts_with("@fragment\nfn fs_main_textured(input: FragmentInput)"));
        assert!(entry.contains("-> @location(0) vec4<f32>"));
        assert!(entry.contains("let mat = materials[input.material_index];"));
        // One sample per role, folded before the light loop.
        assert!(entry.contains("textureSample(base_color_tex, material_sampler, input.uv)"));
        assert!(
            entry.contains("textureSample(metallic_roughness_tex, material_sampler, input.uv)")
        );
        assert!(entry.contains("textureSample(emissive_tex, material_sampler, input.uv)"));
        assert!(entry.contains("apply_base_color_map(mat.base_color.rgb, base_map.rgb)"));
        assert!(entry.contains("apply_emissive_map(mat.emission_color.rgb, emissive_map.rgb)"));
        // Shared evaluation shape is unchanged past the prelude.
        assert!(entry.contains("for (var i: u32 = 0; i < lighting.light_count; i = i + 1)"));
        assert!(entry.contains("continue;"));
        assert!(entry.contains("let base_bsdf = evaluate_base_layer("));
        assert!(entry.contains(
            "let layer_bsdf = base_bsdf + coat_bsdf + fuzz_bsdf + trans_bsdf + ss_bsdf;"
        ));
        assert!(entry.contains("return vec4<f32>(color, opacity);"));
    }

    /// Every table row's declaration appears in the assembled fragment, and
    /// every row maps to a layout entry: shader and pipeline agree. The
    /// first seven rows mirror the legacy table one-to-one.
    #[test]
    fn textured_resources_cover_stages_and_layout() {
        use super::super::{bgl_entry, resource_decl};
        use crate::shaders::pbr_generated::{PBR_RESOURCES, wgsl_vertex_source};
        // Vertex-only rows (per_objects) live in the shared vertex stage,
        // like the legacy table — cover both assemblies together.
        let src = wgsl_vertex_source() + &wgsl_source_textured();
        assert_eq!(TEXTURED_PBR_RESOURCES.len(), 15);
        for (i, r) in TEXTURED_PBR_RESOURCES.iter().enumerate() {
            assert!(src.contains(&resource_decl(r)), "missing {}", r.name);
            let e = bgl_entry(r, false);
            assert_eq!((e.binding, e.visibility), (r.binding, r.visibility));
            // The original seven rows stay aligned with the untextured
            // table. IBL is appended after the material textures, so it
            // does not share those binding numbers.
            if i < 7 {
                assert_eq!(
                    (r.binding, r.name),
                    (PBR_RESOURCES[i].binding, PBR_RESOURCES[i].name),
                    "legacy rows must not drift"
                );
            }
        }
    }

    #[test]
    fn sampled_factors_compose_cpu_mirrors_without_adapter() {
        // Adapter-free sample→combine round trip through the same CPU
        // mirrors the GPU path uses: a 2x2 RGBA8 image (the `LoadedImage`
        // contract) samples albedo through the sRGB decode and
        // metallic-roughness through the linear (G, B) pick, then folds
        // into scalar factors exactly like the shader prelude.
        let image = CpuImage::from_rgba8(
            2,
            2,
            vec![
                255, 0, 0, 255, //
                0, 255, 0, 255, //
                0, 0, 255, 255, //
                18, 128, 64, 128,
            ],
        )
        .expect("valid 2x2");
        // Albedo role: red texel decodes to linear red, then scales the
        // scalar base color channel-wise.
        let red = sample_albedo_linear(image.texel(0, 0).expect("red"));
        assert_eq!(red, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            apply_base_color_map_cpu([0.5, 0.25, 1.0], [red[0], red[1], red[2]]),
            [0.5, 0.0, 0.0]
        );
        // Data role: (G, B) = (128/255 roughness, 64/255 metallic), linear.
        let (roughness, metallic) = sample_metallic_roughness(image.texel(1, 1).expect("data"));
        assert!((roughness - 128.0 / 255.0).abs() < 1e-6);
        assert!((metallic - 64.0 / 255.0).abs() < 1e-6);
        let scaled = apply_metallic_roughness_map_cpu(0.8, 0.5, [roughness, metallic]);
        assert!((scaled[0] - 0.8 * 128.0 / 255.0).abs() < 1e-6);
        assert!((scaled[1] - 0.5 * 64.0 / 255.0).abs() < 1e-6);
        // Emissive role: same sRGB decode as albedo, then scales emission.
        let green = sample_albedo_linear(image.texel(1, 0).expect("green"));
        assert_eq!(
            apply_emissive_map_cpu([1.0, 0.5, 0.25], [green[0], green[1], green[2]]),
            [0.0, 0.5, 0.0]
        );
        // White fallbacks are the exact no-op for every role (the legacy
        // path binds them, so untextured materials render unchanged).
        assert_eq!(
            apply_base_color_map_cpu([0.2, 0.4, 0.6], [1.0, 1.0, 1.0]),
            [0.2, 0.4, 0.6]
        );
        assert_eq!(
            apply_metallic_roughness_map_cpu(0.3, 0.9, [1.0, 1.0]),
            [0.3, 0.9]
        );
        assert_eq!(
            apply_emissive_map_cpu([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]),
            [0.1, 0.2, 0.3]
        );
        // The formats behind the samples stay pinned: color roles decode
        // in hardware, the data role never does.
        assert_eq!(
            texture_format_for_role(TextureRole::BaseColor),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            texture_format_for_role(TextureRole::Emissive),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            texture_format_for_role(TextureRole::MetallicRoughness),
            wgpu::TextureFormat::Rgba8Unorm
        );
    }
}
