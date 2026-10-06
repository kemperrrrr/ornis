//! Lighting shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module; WGSL is assembled
//! from constants + `math::*::wgsl_source()` kernels (OpenPBR BRDF).
//! The former handwritten `shaders/wgsl/lighting.wgsl` was deleted after the
//! `#[stage]` translation of `fs_main`; `lighting_fragment` is now assembled
//! only from here. Prepares
//! PBR lighting for the full Rust→WGSL transition (path 2).

use super::helpers;
use super::interface::HdrFragmentOut as QuadVertexOutput;
use super::{
    ComparisonSampler, DepthTexture, DepthTextureArray, DepthTextureCubeArray, OPENPBR_WGSL_NAME,
    Resource, ResourceKind, STANDARD_QUAD, STANDARD_UVS, Sampler, ShaderModule, Texture2d,
    Texture2dUint, TextureCube, naga_ir, wgsl_decl,
};
use crate::renderer::{CameraUniform, GpuLight, LightingUniform};
use crate::shaders::math;
use ornis_core::material::OpenPBRMaterial;
use ornis_macros::stage;

/// WGSL boilerplate for deferred lighting: derived layouts plus resource
/// bindings, assembled as naga IR and printed by naga itself (see
/// [`naga_ir`](super::naga_ir)) — no WGSL text is authored here.
///
/// At 1x every global declares the single-sample type. In MSAA mode depth,
/// the material id, and the octahedral normal stay multisampled
/// ([`keeps_per_sample`]) so lighting can load each sample. Other float
/// layers still resolve.
fn lighting_header_type(
    name: &str,
    camera: naga::Handle<naga::Type>,
    lighting: naga::Handle<naga::Type>,
    material: naga::Handle<naga::Type>,
) -> naga::Handle<naga::Type> {
    match name {
        "camera" => camera,
        "lighting" => lighting,
        "materials" => material,
        _ => camera,
    }
}

/// Depth, material id, and the normal layer stay per-sample at 4x.
///
/// The normal must not go through the hardware box filter: a cleared texel
/// is `(0, 0)`, and the octahedral decode of that is +Z, which fringes
/// every silhouette.
pub(crate) fn keeps_per_sample(name: &str, kind: &ResourceKind) -> bool {
    name == "normal_tex" || matches!(kind, ResourceKind::TextureDepth | ResourceKind::TextureUint)
}

/// Whether an MSAA lighting binding keeps per-sample storage.
pub(crate) fn per_sample_flag(sample_count: u32, keeps: bool) -> bool {
    sample_count > 1 && keeps
}

fn add_lighting_global(
    module: &mut naga::Module,
    sample_count: u32,
    resource: &Resource,
    camera: naga::Handle<naga::Type>,
    lighting: naga::Handle<naga::Type>,
    material: naga::Handle<naga::Type>,
) {
    naga_ir::add_global(
        module,
        lighting_header_type(resource.name, camera, lighting, material),
        resource,
        per_sample_flag(
            sample_count,
            keeps_per_sample(resource.name, &resource.kind),
        ),
    );
}

fn lighting_wgsl_header_for_samples(sample_count: u32) -> String {
    let mut module = naga::Module::default();
    let cam = CameraUniform::naga_add_type(&mut module);
    // Inserted explicitly so declaration order stays Camera, Light,
    // Lighting (deduped when the Lighting member recurses into it).
    let _light = GpuLight::naga_add_type(&mut module);
    let lighting = LightingUniform::naga_add_type(&mut module);
    let mat = naga_ir::openpbr_type(&mut module);
    for r in LIGHTING_RESOURCES {
        add_lighting_global(&mut module, sample_count, &r, cam, lighting, mat);
    }
    naga_ir::write_module(&module, &LIGHTING_RESOURCES)
}

/// Resource layout of the deferred-lighting pass: where each resource
/// binds, when it is visible, under what name. Type names come from the
/// Rust side (`WGSL_NAME` / [`OPENPBR_WGSL_NAME`]) — never retyped.
pub const LIGHTING_RESOURCES: [Resource; 17] = [
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
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "lighting",
        kind: ResourceKind::Uniform(LightingUniform::WGSL_NAME),
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
        name: "albedo_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "normal_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 5,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "material_id_tex",
        kind: ResourceKind::TextureUint,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 6,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "world_pos_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 7,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "mat_params_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 8,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "depth_tex",
        kind: ResourceKind::TextureDepth,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 9,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "lighting_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 10,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_tex",
        kind: ResourceKind::TextureDepthArray,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 11,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_sampler",
        kind: ResourceKind::SamplerComparison,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 12,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_cube_tex",
        kind: ResourceKind::TextureDepthCubeArray,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 13,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "prefilter_cube",
        kind: ResourceKind::TextureCube,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 14,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "irradiance_cube",
        kind: ResourceKind::TextureCube,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 15,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "brdf_lut",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 16,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ibl_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
];

fn lighting_fragment_kernels() -> String {
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
        math::evaluate_ibl::wgsl_source(),
        math::evaluate_ibl_specular::wgsl_source(),
        math::evaluate_ibl_diffuse::wgsl_source(),
    ];
    kernels.join("\n")
}

/// Deferred-lighting fragment entry, translated by [`stage`](ornis_macros::stage):
/// g-buffer decode + full OpenPBR evaluation. DSL-only — resource globals
/// declared via `#[wgsl(global)]`. `discard` is a bare path statement;
/// `Vec2/3/4::new` spell WGSL constructors; `lo = lo + …` avoids `+=`,
/// which the DSL does not cover.
/// Deferred-lighting resources as a context bundle (`maps.depth_tex`, …).
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct LightingContext {
    pub materials: Vec<OpenPBRMaterial>,
    pub camera: CameraUniform,
    pub lighting: LightingUniform,
}

/// Sampled g-buffer maps as a separate bundle (`maps.albedo_tex`, …):
/// ten texture/sampler globals sharing a lifetime, split out of
/// [`LightingContext`] so the entry signature groups uniforms (ctx) and
/// sampled maps (maps) by kind instead of one ten-field list.
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct LightingMaps {
    pub depth_tex: DepthTexture,
    pub albedo_tex: Texture2d,
    pub normal_tex: Texture2d,
    pub material_id_tex: Texture2dUint,
    pub world_pos_tex: Texture2d,
    pub mat_params_tex: Texture2d,
    pub lighting_sampler: Sampler,
    pub shadow_tex: DepthTextureArray,
    pub shadow_sampler: ComparisonSampler,
    pub shadow_cube_tex: DepthTextureCubeArray,
    pub prefilter_cube: TextureCube,
    pub irradiance_cube: TextureCube,
    pub brdf_lut: Texture2d,
    pub ibl_sampler: Sampler,
}

#[stage(fragment)]
fn fs_main(
    input: QuadVertexOutput,
    ctx: Context<LightingContext>,
    maps: Context<LightingMaps>,
) -> super::Location<0, glam::Vec4> {
    let depth = textureLoad(
        maps.depth_tex,
        UVec2::new(input.uv * Vec2::new(textureDimensions(maps.depth_tex))),
        0,
    );
    let albedo = textureSampleLevel(maps.albedo_tex, maps.lighting_sampler, input.uv, 0.0);
    let normal_enc = textureSampleLevel(maps.normal_tex, maps.lighting_sampler, input.uv, 0.0);
    let material_id = textureLoad(
        maps.material_id_tex,
        UVec2::new(input.uv * Vec2::new(textureDimensions(maps.material_id_tex))),
        0,
    )
    .r;
    let world_pos_enc =
        textureSampleLevel(maps.world_pos_tex, maps.lighting_sampler, input.uv, 0.0);
    let mat_params = textureSampleLevel(maps.mat_params_tex, maps.lighting_sampler, input.uv, 0.0);
    let mat = ctx.materials[material_id];
    let n = octahedral_decode(normal_enc.rg);
    let world_pos = reconstruct_world_pos(input.uv, depth, ctx.camera);
    let v = normalize(ctx.camera.camera_pos.xyz - world_pos);
    return shade_lit(n, world_pos, v, albedo.a, mat);
}

/// OpenPBR lighting for one decoded sample. `albedo_a` below
/// [`ALPHA_CUTOFF`](super::helpers::ALPHA_CUTOFF) discards the fragment.
/// Globals (`lighting`, shadow maps, IBL) are the lighting bind group.
#[ornis_macros::wgsl_fn]
fn shade_lit(
    n: glam::Vec3,
    world_pos: glam::Vec3,
    v: glam::Vec3,
    albedo_a: f32,
    mat: OpenPBRMaterial,
) -> glam::Vec4 {
    // Derivatives before `discard`: a non-uniform discard makes `dpdx` invalid.
    let specular_roughness = specular_aa_roughness(mat.specular_params.y, n);
    if albedo_a < ALPHA_CUTOFF {
        discard;
    }
    let nov = max(dot(n, v), EPS);
    let base_weight = mat.base_params.x;
    let base_color = mat.base_color.rgb;
    let metalness = mat.base_params.z;
    let diffuse_roughness = mat.base_params.y;
    let specular_weight = mat.specular_params.x;
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
    let mut spec_acc = Vec3::new(0.0, 0.0, 0.0);
    let mut diff_acc = Vec3::new(0.0, 0.0, 0.0);
    let mut shadow_vis = 1.0;
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
    let t = normalize(cross(n, Vec3::new(0.0, 1.0, 0.0)) + Vec3::new(0.0, 0.0, EPS));
    let b = cross(n, t);
    for i in 0u..lighting.light_count {
        let light = lighting.lights[i];
        let kind = light.kind.x;
        // Point/spot: vector from the surface to the light + range cutoff.
        // Directionals keep the legacy infinite-light path (pixel-identical).
        let to_light = light.position.xyz - world_pos;
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
            normalize(world_pos - light.position.xyz),
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
                let shadow_clip =
                    light.shadow_vp * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
                let shadow_ndc = shadow_clip.xyz / shadow_clip.w;
                // V is mirrored: rasterization puts NDC y+1 at texture
                // row 0 while `shadow_uv` v=0 reads from the top, so an
                // unmirrored lookup samples the mirrored texel (shadows
                // land overturned — darkness tests are blind to it).
                let shadow_uv = Vec2::new(
                    shadow_ndc.x * NDC_TO_UV_HALF + NDC_TO_UV_HALF,
                    NDC_TO_UV_HALF - shadow_ndc.y * NDC_TO_UV_HALF,
                );
                let vis = textureSampleCompare(
                    shadow_tex,
                    shadow_sampler,
                    shadow_uv,
                    i32(light.params.w),
                    shadow_ndc.z - SHADOW_REF_BIAS,
                );
                radiance = radiance * vis;
                shadow_vis = min(shadow_vis, vis);
            }
            if is_point > LIGHT_SELECTOR_SPLIT {
                // Cube sample: the hardware picks the face from the
                // fragment→light vector's major axis; the reference is
                // the 90°-perspective depth for that axis
                // (`SHADOW_CUBE_NEAR`, far = light range — the same
                // formula the face renders use, so no VP uniform).
                let to_frag = world_pos - light.position.xyz;
                let major = max(max(abs(to_frag.x), abs(to_frag.y)), abs(to_frag.z));
                let far = max(light.params.x, 1.0);
                let denom = far - SHADOW_CUBE_NEAR;
                let cube_ref =
                    (far / denom) - (SHADOW_CUBE_NEAR * far) / (denom * major) - SHADOW_REF_BIAS;
                let vis = textureSampleCompare(
                    shadow_cube_tex,
                    shadow_sampler,
                    to_frag,
                    i32(light.params.w),
                    cube_ref,
                );
                radiance = radiance * vis;
                shadow_vis = min(shadow_vis, vis);
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
        if lighting.debug_view != SHADING_DEBUG_BEAUTY {
            let spec_lobe = evaluate_base_specular(
                v,
                l,
                h,
                nov,
                nol,
                noh,
                voh,
                base_weight,
                base_color,
                metalness,
                specular_weight,
                specular_roughness,
                specular_ior,
                specular_anisotropy,
                specular_edge_tint,
                t,
                b,
                thin_film_mod,
            ) * coat_darken;
            spec_acc = spec_acc + spec_lobe * radiance * nol;
            diff_acc = diff_acc + (base_bsdf - spec_lobe) * radiance * nol;
        }
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
        lighting.ambient_color.rgb * mix(base_color, base_color * specular_weight, metalness);
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
        prefilter_cube,
        ibl_sampler,
        reflect_dir,
        specular_roughness * lighting.ibl_max_mip,
    );
    let irradiance = textureSample(irradiance_cube, ibl_sampler, n);
    let lut = textureSample(brdf_lut, ibl_sampler, Vec2::new(nov, specular_roughness));
    let ibl = evaluate_ibl(
        prefiltered.rgb,
        irradiance.rgb,
        lut.r,
        lut.g,
        f0,
        base_color * (1.0 - metalness),
        lighting.ibl_weight,
    );
    let raw = ambient + lo + emission + ibl;
    let beauty = sanitize_hdr(raw);
    let mut ibl_spec = Vec3::new(0.0, 0.0, 0.0);
    if lighting.debug_view == SHADING_DEBUG_IBL_SPECULAR
        || lighting.debug_view == SHADING_DEBUG_DIFFUSE
    {
        ibl_spec = evaluate_ibl_specular(prefiltered.rgb, lut.r, lut.g, f0, lighting.ibl_weight);
    }
    let diffuse = diff_acc + ambient + (ibl - ibl_spec);
    let color = select_shading_debug(
        lighting.debug_view,
        beauty,
        spec_acc,
        ibl_spec,
        diffuse,
        specular_roughness,
        metalness,
        shadow_vis,
        raw,
    );
    return glam::Vec4::new(color, opacity);
}

/// Deferred lighting at 4x. Every sample is lit (the fetches stay in
/// uniform control flow), then an edge mask picks the result: interior
/// pixels keep sample 0, and silhouette or disagreement pixels average
/// only the covered samples with a Karis weight `1 / (1 + luma)`, so one
/// grazing firefly cannot paint the whole pixel white. Alpha stays
/// coverage-weighted. Cleared samples (depth [`CLEAR_DEPTH`](super::helpers::CLEAR_DEPTH),
/// octahedral `(0, 0)` → +Z) contribute nothing.
#[stage(fragment, entry = "fs_main")]
fn fs_main_msaa(
    input: QuadVertexOutput,
    ctx: Context<LightingContext>,
    maps: Context<LightingMaps>,
) -> super::Location<0, glam::Vec4> {
    let albedo = textureSampleLevel(maps.albedo_tex, maps.lighting_sampler, input.uv, 0.0);
    let world_pos_enc =
        textureSampleLevel(maps.world_pos_tex, maps.lighting_sampler, input.uv, 0.0);
    let mat_params = textureSampleLevel(maps.mat_params_tex, maps.lighting_sampler, input.uv, 0.0);
    if albedo.a < ALPHA_CUTOFF {
        discard;
    }
    let coord = UVec2::new(input.uv * Vec2::new(textureDimensions(maps.depth_tex)));
    let mut acc = Vec3::new(0.0, 0.0, 0.0);
    let mut alpha = 0.0;
    let mut weight = 0.0;
    let mut single = Vec4::new(0.0, 0.0, 0.0, 0.0);
    let mut covered_count = 0.0;
    let mut have_ref = 0.0;
    let mut disagree = 0.0;
    let mut ref_d = CLEAR_DEPTH;
    let mut ref_id = MSAA_SAMPLES - MSAA_SAMPLES;
    let mut ref_n = Vec3::new(0.0, 0.0, 1.0);
    for s in 0u..MSAA_SAMPLES {
        let depth = textureLoad(maps.depth_tex, coord, s);
        let id = textureLoad(maps.material_id_tex, coord, s).r;
        let n = octahedral_decode(textureLoad(maps.normal_tex, coord, s).rg);
        let covered = select(0.0, 1.0, depth < CLEAR_DEPTH);
        let is_ref = select(0.0, 1.0, have_ref == 0.0 && covered == 1.0);
        let depth_far = select(0.0, 1.0, abs(depth - ref_d) > DEPTH_EDGE);
        let normal_far = select(0.0, 1.0, dot(n, ref_n) < NORMAL_AGREE);
        let id_far = select(0.0, 1.0, id != ref_id);
        let split = max(depth_far, max(normal_far, id_far));
        disagree = max(disagree, split * covered * have_ref);
        ref_n = mix(ref_n, n, is_ref);
        ref_d = mix(ref_d, depth, is_ref);
        ref_id = select(ref_id, id, is_ref == 1.0);
        have_ref = max(have_ref, covered);
        covered_count = covered_count + covered;
        let world_pos = reconstruct_world_pos(input.uv, depth, ctx.camera);
        let v = normalize(ctx.camera.camera_pos.xyz - world_pos);
        let lit = shade_lit(n, world_pos, v, 1.0, ctx.materials[id]);
        let luma = max(luminance(lit.xyz), 0.0);
        let karis = covered / (KARIS_LUMA_BIAS + luma);
        acc = acc + lit.xyz * karis;
        alpha = alpha + lit.w * covered;
        weight = weight + karis;
        let is_zero = select(0.0, 1.0, s == 0u);
        single = mix(single, lit, is_zero);
    }
    let all_clear = select(0.0, 1.0, covered_count == 0.0);
    let all_hit = select(0.0, 1.0, covered_count == MSAA_SAMPLES_F);
    let calm = select(0.0, 1.0, disagree == 0.0);
    let interior = max(all_clear, all_hit * calm);
    let edge = 1.0 - interior;
    let averaged = Vec4::new(acc / max(weight, EPS), alpha / max(weight, EPS));
    return mix(single, averaged, edge);
}

/// Full WGSL source for deferred lighting, assembled from Rust.
pub fn wgsl_source() -> String {
    wgsl_source_for_samples(1)
}

fn lighting_entry(msaa: bool) -> &'static str {
    match msaa {
        true => fs_main_msaa::wgsl_source(),
        false => fs_main::wgsl_source(),
    }
}

/// Full WGSL source for deferred lighting at a sample count: at 1x
/// byte-identical to [`wgsl_source`]; in MSAA mode depth, material-id and
/// the normal declare multisampled types and [`fs_main_msaa`] shades
/// covered samples on edges.
pub fn wgsl_source_for_samples(sample_count: u32) -> String {
    ShaderModule::new()
        .decl(lighting_wgsl_header_for_samples(sample_count))
        .decl(wgsl_decl(QuadVertexOutput::WGSL_SOURCE))
        .consts(math::wgsl_consts())
        .helpers([
            helpers::wgsl_consts(),
            helpers::wgsl_lighting_decode(),
            helpers::wgsl_shared_helpers(),
        ])
        .helper(shade_lit::wgsl_source())
        .entry(lighting_entry(per_sample_flag(sample_count, true)))
        .helper(lighting_fragment_kernels())
        .emit()
}

/// Vertex entry, translated by [`stage`](ornis_macros::stage).
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

/// Vertex WGSL: full-screen quad (triangle strip) — quad constants built
/// from the shared [`STANDARD_QUAD`]/[`STANDARD_UVS`] Rust data; the varying
/// splices the shared `QuadVertexOutput` declaration (`HdrFragmentOut`)
/// and the entry is translated.
pub fn wgsl_vertex_source() -> String {
    ShaderModule::new()
        .consts(naga_ir::const_block(&STANDARD_QUAD, &STANDARD_UVS))
        .decl(wgsl_decl(QuadVertexOutput::WGSL_SOURCE))
        .entry(vs_main::wgsl_source())
        .emit()
}

/// Static view for naga validation and snapshot tests.
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
    fn lighting_generated_validates_with_naga() {
        assert_valid_wgsl("lighting_generated", &wgsl_source());
    }

    #[test]
    fn lighting_generated_contains_expected_kernels() {
        let src = wgsl_source();
        assert!(
            !src.contains("aces_tonemap("),
            "lighting writes scene-linear HDR; ACES belongs to the composite"
        );
        assert!(src.contains("fn fresnel_f82_tint"));
        assert!(src.contains("fn ggx_ndf_aniso"));
        assert!(src.contains("fn fs_main"));
    }

    /// The resource table drives both sides: WGSL declarations and `wgpu`
    /// layout entries agree on numbers, visibility and kinds.
    #[test]
    fn lighting_resources_drive_bgl_and_wgsl() {
        use super::super::{bgl_entry, resource_decl};
        use super::LIGHTING_RESOURCES;
        assert_eq!(LIGHTING_RESOURCES.len(), 17);
        for r in LIGHTING_RESOURCES {
            let decl = resource_decl(&r);
            assert!(decl.starts_with(&format!("@group({}) @binding({}) ", r.group, r.binding)));
            let plain = bgl_entry(&r, false);
            let msaa = bgl_entry(&r, true);
            assert_eq!(plain.binding, r.binding);
            assert_eq!(plain.visibility, r.visibility);
            // Only textures observe the MSAA flag.
            let is_texture = matches!(plain.ty, wgpu::BindingType::Texture { .. });
            assert_eq!(
                format!("{:?}", plain.ty) != format!("{:?}", msaa.ty),
                is_texture
            );
        }
        // Binding 0 (camera) is visible to the vertex stage too.
        assert!(
            LIGHTING_RESOURCES[0]
                .visibility
                .contains(wgpu::ShaderStages::VERTEX)
        );
    }
    #[test]
    fn lighting_struct_blocks_match_derived_layouts() {
        // The header is built as naga IR from the same derives (drift is
        // impossible by construction); pin declaration order and the 13
        // resource bindings in the printed output.
        let src = wgsl_source();
        let cam = src.find("struct Camera").expect("Camera");
        let light = src.find("struct Light").expect("Light");
        let lighting = src.find("struct Lighting").expect("Lighting");
        let mat = src.find("struct OpenPBRMaterial").expect("OpenPBR");
        let b9 = src.find("lighting_sampler").expect("bindings");
        assert!(cam < light && light < lighting && lighting < mat && mat < b9);
        assert_eq!(src.matches("@binding(").count(), 17);
    }

    /// The translated fragment entry must keep the legacy shape: g-buffer
    /// decode with `textureLoad` coords, then [`shade_lit`] (alpha
    /// `discard`, the light loop, the summed layer BSDF).
    #[test]
    fn fs_main_matches_legacy_shape() {
        let entry = fs_main::wgsl_source();
        assert!(entry.starts_with("@fragment\nfn fs_main(input: QuadVertexOutput)"));
        assert!(entry.contains("-> @location(0) vec4<f32>"));
        assert!(entry.contains(
            "textureLoad(depth_tex, vec2<u32>(input.uv * vec2<f32>(textureDimensions(depth_tex))), 0)"
        ));
        assert!(entry.contains("shade_lit("));
        let src = wgsl_source();
        assert!(src.contains("discard;"));
        assert!(src.contains("for (var i: u32 = 0; i < lighting.light_count; i = i + 1)"));
        assert!(src.contains("continue;"));
        assert!(src.contains("let base_bsdf = evaluate_base_layer("));
        assert!(src.contains(
            "let layer_bsdf = base_bsdf + coat_bsdf + fuzz_bsdf + trans_bsdf + ss_bsdf;"
        ));
        assert!(src.contains("return vec4<f32>(color, opacity);"));
        assert!(src.contains("specular_aa_roughness("));
        assert!(src.contains("sanitize_hdr("));
        assert!(src.contains("textureSampleLevel(prefilter_cube, ibl_sampler"));
        assert!(src.contains("textureSample(brdf_lut, ibl_sampler"));
        assert!(src.contains("textureSample(irradiance_cube, ibl_sampler"));
        assert!(src.contains("lighting.ibl_max_mip"));
        assert!(src.contains("thin_film_weight_mix("));
        assert!(src.contains("evaluate_coat_darkening("));
    }

    /// The MSAA source validates with naga. Depth, material-id and the
    /// normal are multisampled; the entry loads every sample and masks
    /// cleared ones. Albedo stays a resolved `texture_2d`.
    #[test]
    fn msaa_source_validates_with_multisampled_depth_and_id() {
        let src = wgsl_source_for_samples(4);
        assert_valid_wgsl("lighting_generated_msaa", &src);
        assert!(src.contains("texture_depth_multisampled_2d"), "{src}");
        assert!(src.contains("texture_multisampled_2d<u32>"), "{src}");
        assert!(src.contains("texture_multisampled_2d<f32>"), "{src}");
        assert!(src.contains("textureLoad(normal_tex"), "{src}");
        assert!(
            src.contains("for (var s: u32 = 0; s < MSAA_SAMPLES; s = s + 1)"),
            "{src}"
        );
        assert!(src.contains("CLEAR_DEPTH"), "{src}");
        assert!(
            src.contains("KARIS_LUMA_BIAS"),
            "edge resolve must Karis-weight fireflies, {src}"
        );
        assert!(src.contains("textureSampleLevel"), "{src}");
        assert!(src.contains("albedo_tex"), "{src}");
        assert!(
            src.contains("var albedo_tex: texture_2d<f32>"),
            "albedo stays resolved, {src}"
        );
        let plain = wgsl_source();
        assert!(!plain.contains("multisampled"), "{plain}");
        assert!(
            !plain.contains("for (var s: u32 = 0; s < MSAA_SAMPLES; s = s + 1)"),
            "1x lighting does not walk samples"
        );
    }
}
