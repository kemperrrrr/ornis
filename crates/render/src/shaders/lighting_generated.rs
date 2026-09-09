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
    OPENPBR_WGSL_NAME, Resource, ResourceKind, STANDARD_QUAD, STANDARD_UVS, const_vec2_array,
    const_vec4_array, openpbr_material_decl, resource_decl, wgsl_decl,
};
use crate::renderer::{CameraUniform, GpuLight, LightingUniform};
use crate::shaders::math;
use ornis_macros::stage;

/// WGSL boilerplate for deferred lighting: derived layouts plus resource
/// bindings, assembled from Rust (no handwritten WGSL remains).
fn lighting_wgsl_header() -> String {
    let mut out = String::new();
    out.push_str(&wgsl_decl(CameraUniform::WGSL_SOURCE));
    out.push_str(&wgsl_decl(GpuLight::WGSL_SOURCE));
    out.push_str(&wgsl_decl(LightingUniform::WGSL_SOURCE));
    out.push_str(openpbr_material_decl().as_str());
    for r in LIGHTING_RESOURCES {
        out.push_str(&resource_decl(&r));
    }
    out
}

/// Resource layout of the deferred-lighting pass: where each resource
/// binds, when it is visible, under what name. Type names come from the
/// Rust side (`WGSL_NAME` / [`OPENPBR_WGSL_NAME`]) — never retyped.
pub const LIGHTING_RESOURCES: [Resource; 10] = [
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
];

fn lighting_fragment_kernels() -> String {
    let kernels = [
        math::luminance::wgsl_source(),
        math::aces_tonemap::wgsl_source(),
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
        math::coat_base_darkening::wgsl_source(),
        math::coat_blend_darkened::wgsl_source(),
        math::thin_film_modulation::wgsl_source(),
        math::sheen_brdf::wgsl_source(),
        math::transmission_color_to_extinction::wgsl_source(),
        math::subsurface_brdf::wgsl_source(),
    ];
    kernels.join("\n")
}

/// Deferred-lighting fragment entry, translated by [`stage`](ornis_macros::stage):
/// g-buffer decode + full OpenPBR evaluation. DSL-only — free texture /
/// uniform / storage identifiers. `discard` is a bare path statement;
/// `Vec2/3/4::new` spell WGSL constructors; `lo = lo + …` avoids `+=`,
/// which the DSL does not cover.
#[stage(fragment, entry = "fs_main", returns = "@location(0) vec4<f32>")]
fn lighting_fragment_entry(#[wgsl(location = 0)] uv: glam::Vec2) -> glam::Vec4 {
    let depth = textureLoad(
        depth_tex,
        UVec2::new(uv * Vec2::new(textureDimensions(depth_tex))),
        0,
    );
    let albedo = textureSampleLevel(albedo_tex, lighting_sampler, uv, 0.0);
    let normal_enc = textureSampleLevel(normal_tex, lighting_sampler, uv, 0.0);
    let material_id = textureLoad(
        material_id_tex,
        UVec2::new(uv * Vec2::new(textureDimensions(material_id_tex))),
        0,
    )
    .r;
    let world_pos_enc = textureSampleLevel(world_pos_tex, lighting_sampler, uv, 0.0);
    let mat_params = textureSampleLevel(mat_params_tex, lighting_sampler, uv, 0.0);
    let mat = materials[material_id];
    if albedo.a < 0.001 {
        discard;
    }
    let n = octahedral_decode(normal_enc.rg);
    let world_pos = reconstruct_world_pos(uv, depth, camera);
    let v = normalize(camera.camera_pos.xyz - world_pos);
    let nov = max(dot(n, v), EPS);
    let base_weight = mat.base_params.x;
    let base_color = mat.base_color.rgb;
    let metalness = mat.base_params.z;
    let diffuse_roughness = mat.base_params.y;
    let specular_weight = mat.specular_params.x;
    let specular_roughness = mat.specular_params.y;
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
    let thin_film_mod = thin_film_modulation(nov, thin_film_ior, thin_film_thickness_um, 1.0);
    let t = normalize(cross(n, Vec3::new(0.0, 1.0, 0.0)));
    let b = cross(n, t);
    for i in 0u..lighting.light_count {
        let l = normalize(lighting.lights[i].direction.xyz);
        let h = normalize(v + l);
        let light_color = lighting.lights[i].color.rgb;
        let intensity = lighting.lights[i].color.w;
        let radiance = light_color * intensity;
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
        );
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
    let color = ambient + lo + emission;
    let tone_mapped = aces_tonemap(color);
    return glam::Vec4::new(tone_mapped, opacity);
}

/// Full WGSL source for deferred lighting, assembled from Rust.
pub fn wgsl_source() -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        lighting_wgsl_header(),
        helpers::wgsl_consts(),
        helpers::wgsl_lighting_decode(),
        helpers::wgsl_shared_helpers(),
        lighting_fragment_entry::wgsl_source(),
        lighting_fragment_kernels(),
    )
}

/// Vertex entry, translated by [`stage`](ornis_macros::stage).
/// DSL-only — replaced by `lighting_vertex_entry::wgsl_source()`.
#[stage(vertex, entry = "vs_main")]
fn lighting_vertex_entry(#[wgsl(builtin = "vertex_index")] idx: u32) -> QuadVertexOutput {
    return QuadVertexOutput {
        clip_position: QUAD[idx],
        uv: UVS[idx],
    };
}

/// Vertex WGSL: full-screen quad (triangle strip) — quad constants built
/// from the shared [`STANDARD_QUAD`]/[`STANDARD_UVS`] Rust data; the varying
/// splices the shared `QuadVertexOutput` declaration (`HdrFragmentOut`)
/// and the entry is translated.
pub fn wgsl_vertex_source() -> String {
    format!(
        "\n{quad}{qo}{body}",
        quad = vertex_quad_uv(),
        qo = wgsl_decl(QuadVertexOutput::WGSL_SOURCE),
        body = lighting_vertex_entry::wgsl_source(),
    )
}

/// Shared quad constants for the lighting vertex stage.
fn vertex_quad_uv() -> String {
    let mut out = const_vec4_array("QUAD", &STANDARD_QUAD);
    out.push('\n');
    out.push_str(&const_vec2_array("UVS", &STANDARD_UVS));
    out
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
        assert!(src.contains("fn aces_tonemap"));
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
        assert_eq!(LIGHTING_RESOURCES.len(), 10);
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
        use super::super::{openpbr_material_decl, wgsl_decl};
        use crate::renderer::{CameraUniform, GpuLight, LightingUniform};
        let src = wgsl_source();
        let cam = src
            .find(&wgsl_decl(CameraUniform::WGSL_SOURCE))
            .expect("Camera");
        let light = src.find(&wgsl_decl(GpuLight::WGSL_SOURCE)).expect("Light");
        let lighting = src
            .find(&wgsl_decl(LightingUniform::WGSL_SOURCE))
            .expect("Lighting");
        let mat = src.find(openpbr_material_decl().as_str()).expect("OpenPBR");
        let b9 = src
            .find("@group(0) @binding(9) var lighting_sampler: sampler;")
            .expect("bindings");
        assert!(cam < light && light < lighting && lighting < mat && mat < b9);
        assert_eq!(src.matches("@group(0) @binding(").count(), 10);
    }

    /// The translated fragment entry must keep the legacy shape: g-buffer
    /// decode with `textureLoad` coords, alpha `discard`, the light loop
    /// with the early-out, and the summed layer BSDF. (Byte-parity no
    /// longer applies — the generated entry is single-line.)
    #[test]
    fn lighting_fragment_entry_matches_legacy_shape() {
        let entry = lighting_fragment_entry::wgsl_source();
        assert!(entry.starts_with("@fragment\nfn fs_main(@location(0) uv: vec2<f32>)"));
        assert!(entry.contains("-> @location(0) vec4<f32>"));
        assert!(entry.contains(
            "textureLoad(depth_tex, vec2<u32>(uv * vec2<f32>(textureDimensions(depth_tex))), 0)"
        ));
        assert!(entry.contains("discard;"));
        assert!(entry.contains("for (var i: u32 = 0; i < lighting.light_count; i = i + 1)"));
        assert!(entry.contains("continue;"));
        assert!(entry.contains("let base_bsdf = evaluate_base_layer("));
        assert!(entry.contains(
            "let layer_bsdf = base_bsdf + coat_bsdf + fuzz_bsdf + trans_bsdf + ss_bsdf;"
        ));
        assert!(entry.contains("return vec4<f32>(tone_mapped, opacity);"));
    }
}
