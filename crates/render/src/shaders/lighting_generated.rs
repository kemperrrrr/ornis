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
use super::wgsl_decl;
use crate::shaders::math;
use ornis_macros::stage;

/// WGSL boilerplate for deferred lighting: structs, bindings, helpers, main.
/// Self-contained since the `#[stage]` translation; entry point name `fs_main`
/// is kept for compatibility.
fn lighting_wgsl_header() -> &'static str {
    // This literal is the only `vec4<f32>` outside `*_generated.rs` that must
    // be absent; here it is inside generated code, which is allowed by the grep rule.
    r#"
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

struct Light {
    direction: vec4<f32>,
    color: vec4<f32>,
};

struct Lighting {
    ambient_color: vec4<f32>,
    lights: array<Light, 4>,
    light_count: u32,
};

struct OpenPBRMaterial {
    base_params: vec4<f32>,
    base_color: vec4<f32>,
    specular_params: vec4<f32>,
    specular_color: vec4<f32>,
    transmission_params: vec4<f32>,
    transmission_color: vec4<f32>,
    transmission_scatter: vec4<f32>,
    subsurface_params: vec4<f32>,
    subsurface_color: vec4<f32>,
    subsurface_radius_scale_gb: vec4<f32>,
    fuzz_params: vec4<f32>,
    fuzz_color: vec4<f32>,
    coat_params: vec4<f32>,
    coat_color: vec4<f32>,
    coat_ior: vec4<f32>,
    thin_film_params: vec4<f32>,
    emission_params: vec4<f32>,
    emission_color: vec4<f32>,
    geometry_params: vec4<f32>,
    geometry_params2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(0) @binding(2) var<storage, read> materials: array<OpenPBRMaterial>;
@group(0) @binding(3) var albedo_tex: texture_2d<f32>;
@group(0) @binding(4) var normal_tex: texture_2d<f32>;
@group(0) @binding(5) var material_id_tex: texture_2d<u32>;
@group(0) @binding(6) var world_pos_tex: texture_2d<f32>;
@group(0) @binding(7) var mat_params_tex: texture_2d<f32>;
@group(0) @binding(8) var depth_tex: texture_depth_2d;
@group(0) @binding(9) var lighting_sampler: sampler;

"#
}

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

/// Vertex WGSL: full-screen quad (triangle strip) — quad constants stay
/// handwritten; the varying splices the shared `QuadVertexOutput`
/// declaration (`HdrFragmentOut`) and the entry is translated.
pub fn wgsl_vertex_source() -> String {
    format!(
        "\n{quad}{qo}{body}",
        quad = WGSL_VERTEX_QUAD_UV,
        qo = wgsl_decl(QuadVertexOutput::WGSL_SOURCE),
        body = lighting_vertex_entry::wgsl_source(),
    )
}

const WGSL_VERTEX_QUAD_UV: &str = r#"const QUAD: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>( 1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0,  1.0, 0.0, 1.0),
    vec4<f32>( 1.0,  1.0, 0.0, 1.0),
);
const UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
);
"#;

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

    /// The handwritten struct blocks in the lighting header must stay
    /// identical to the derived layouts: the field lists on the Rust mirrors
    /// are the authority, this test is the tripwire. (The header keeps its
    /// monolithic shape instead of splices so the 200-line evaluator body is
    /// never retyped.)
    #[test]
    fn lighting_struct_blocks_match_derived_layouts() {
        use super::super::{OPENPBR_MATERIAL_DECL, wgsl_decl};
        use crate::renderer::{CameraUniform, GpuLight, LightingUniform};
        let src = wgsl_source();
        assert!(
            src.contains(&wgsl_decl(CameraUniform::WGSL_SOURCE)),
            "Camera block drifted from CameraUniform layout"
        );
        assert!(
            src.contains(&wgsl_decl(GpuLight::WGSL_SOURCE)),
            "Light block drifted from GpuLight layout"
        );
        assert!(
            src.contains(&wgsl_decl(LightingUniform::WGSL_SOURCE)),
            "Lighting block drifted from LightingUniform layout"
        );
        assert!(
            src.contains(OPENPBR_MATERIAL_DECL),
            "OpenPBR block drifted from the shared declaration"
        );
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
