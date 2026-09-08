//! Lighting shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module; WGSL is assembled
//! from constants + `math::*::wgsl_source()` kernels (OpenPBR BRDF).
//! The handwritten `shaders/wgsl/lighting.wgsl` remains as a reference/legacy,
//! but `lighting_fragment` is now assembled only from here. Prepares
//! PBR lighting for the full Rust→WGSL transition (path 2).

use super::interface::HdrFragmentOut as QuadVertexOutput;
use super::wgsl_decl;
use crate::shaders::math;
use ornis_macros::stage;

/// WGSL boilerplate for deferred lighting: structs, bindings, helpers, main.
/// Identical to `shaders/wgsl/lighting.wgsl`; entry point names `fs_main`
/// are kept for compatibility.
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

const PI: f32 = 3.14159265359;
const EPS: f32 = 1e-6;
const INV_PI: f32 = 0.31830988618;

fn octahedral_decode(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    let t = max(-n.z, 0.0);
    let offset = select(-n.yx, n.yx, n.xy >= vec2<f32>(0.0)) * t;
    n.x += offset.x;
    n.y += offset.y;
    return normalize(n);
}

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32, camera: Camera) -> vec3<f32> {
    let ndc = vec3<f32>(uv * 2.0 - 1.0, depth * 2.0 - 1.0);
    let clip = vec4<f32>(ndc, 1.0);
    let view = camera.inv_view_proj * clip;
    return view.xyz / view.w;
}

fn evaluate_base_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    mat: OpenPBRMaterial,
    base_weight: f32, base_color: vec3<f32>, metalness: f32,
    diffuse_roughness: f32,
    specular_weight: f32, specular_roughness: f32, specular_ior: f32, specular_anisotropy: f32, specular_edge_tint: vec3<f32>,
    T: vec3<f32>, B: vec3<f32>,
    thin_film_mod: vec3<f32>
) -> vec3<f32> {
    let F0_dielectric = vec3<f32>(fresnel0_from_ior(specular_ior));
    let F0_metal = base_color * base_weight;
    let F0 = mix(F0_dielectric, F0_metal, metalness);
    let F = fresnel_f82_tint(VoH, F0, specular_edge_tint);
    let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);
    let alpha_u = alpha.x;
    let alpha_v = alpha.y;
    let D = ggx_ndf_aniso(NoH, H, T, B, alpha_u, alpha_v);
    let G = smith_ggx_aniso(NoV, NoL, V, L, T, B, alpha_u, alpha_v);
    let spec_brdf = D * G * F / max(4.0 * NoV * NoL, EPS);
    let diffuse_color = base_color * (1.0 - metalness);
    let diff_roughness = max(diffuse_roughness, specular_roughness);
    let diff_alpha = diff_roughness * diff_roughness;
    let cos_phi = max(dot(normalize(V - N * NoV), normalize(L - N * NoL)), 0.0);
    let diff_brdf = oren_nayar_brdf(NoV, NoL, cos_phi, diff_alpha);
    let kS = F * specular_weight;
    let kD = (vec3<f32>(1.0) - luminance(kS)) * (1.0 - metalness);
    let base_bsdf = kD * diff_brdf * diffuse_color + kS * spec_brdf;
    return base_bsdf * base_weight * thin_film_mod;
}

fn evaluate_coat_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    mat: OpenPBRMaterial,
    coat_weight: f32, coat_roughness: f32, coat_anisotropy: f32, coat_dark: f32, coat_ior: f32, coat_color: vec3<f32>,
    base_metalness: f32, base_color: vec3<f32>, base_weight: f32, specular_weight: f32,
    subsurface_weight: f32, subsurface_color: vec3<f32>,
    T: vec3<f32>, B: vec3<f32>
) -> vec3<f32> {
    if coat_weight <= 0.0 { return vec3<f32>(0.0); }
    let coat_F0 = vec3<f32>(fresnel0_from_ior(coat_ior));
    let coat_F = fresnel_schlick_vec(VoH, coat_F0);
    let coat_alpha = openpbr_anisotropy(coat_roughness, coat_anisotropy);
    let coat_alpha_u = coat_alpha.x;
    let coat_alpha_v = coat_alpha.y;
    let coat_D = ggx_ndf_aniso(NoH, H, T, B, coat_alpha_u, coat_alpha_v);
    let coat_G = smith_ggx_aniso(NoV, NoL, V, L, T, B, coat_alpha_u, coat_alpha_v);
    let coat_brdf = coat_D * coat_G * coat_F / max(4.0 * NoV * NoL, EPS);
    let mix_factor = coat_weight * coat_dark;
    let base_darkening = coat_base_darkening(
        coat_ior, base_metalness, base_color, base_weight,
        specular_weight, subsurface_weight, subsurface_color
    );
    let darkening = coat_blend_darkened(base_darkening, mix_factor);
    let coat_albedo_approx = coat_color * coat_weight * luminance(coat_F0);
    return coat_color * coat_brdf * coat_weight + darkening * coat_albedo_approx;
}

fn evaluate_fuzz_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    fuzz_weight: f32, fuzz_roughness: f32, fuzz_color: vec3<f32>
) -> vec3<f32> {
    if fuzz_weight <= 0.0 { return vec3<f32>(0.0); }
    let sheen = sheen_brdf(NoV, NoL, NoH, VoH, fuzz_roughness);
    return fuzz_color * sheen * fuzz_weight;
}

fn evaluate_transmission_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    mat: OpenPBRMaterial,
    transmission_weight: f32, transmission_depth: f32, transmission_dispersion_scale: f32, transmission_dispersion_abbe: f32,
    transmission_color: vec3<f32>, transmission_scatter: vec3<f32>, transmission_scatter_anisotropy: f32,
    specular_ior: f32, specular_roughness: f32, specular_anisotropy: f32,
    thin_walled: f32
) -> vec3<f32> {
    if transmission_weight <= 0.0 { return vec3<f32>(0.0); }
    let ior_out = 1.0;
    let ior_in = specular_ior;
    let eta = ior_in / ior_out;
    let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);
    let extinction = transmission_color_to_extinction(transmission_color, transmission_depth);
    let distance = transmission_depth;
    let btdf = transmission_btdf(NoV, NoL, VoH, ior_in, ior_out, alpha.x, extinction, distance);
    return transmission_color * btdf * transmission_weight;
}

fn evaluate_subsurface_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>,
    NoV: f32, NoL: f32,
    subsurface_weight: f32, subsurface_radius: f32, subsurface_radius_scale_r: f32,
    subsurface_radius_scale_g: f32, subsurface_radius_scale_b: f32,
    subsurface_scatter_anisotropy: f32, subsurface_color: vec3<f32>
) -> vec3<f32> {
    if subsurface_weight <= 0.0 { return vec3<f32>(0.0); }
    let radius = vec3<f32>(
        subsurface_radius * subsurface_radius_scale_r,
        subsurface_radius * subsurface_radius_scale_g,
        subsurface_radius * subsurface_radius_scale_b
    );
    let V_proj = V - N * NoV;
    let L_proj = L - N * NoL;
    let distance = length(V_proj - L_proj);
    let ss_brdf = subsurface_brdf(NoV, NoL, distance, radius, subsurface_scatter_anisotropy);
    return subsurface_color * ss_brdf * subsurface_weight;
}

fn evaluate_emission(
    emission_luminance: f32, emission_color: vec3<f32>,
    coat_weight: f32, coat_color: vec3<f32>,
    NoV: f32
) -> vec3<f32> {
    if emission_luminance <= 0.0 { return vec3<f32>(0.0); }
    let base_emission = emission_color * emission_luminance * INV_PI;
    let coat_emission = coat_color * base_emission * (pow(1.0 - NoV, 5.0) * coat_weight + (1.0 - coat_weight));
    return mix(base_emission, coat_emission, coat_weight);
}

fn transmission_btdf(
    NoV: f32, NoL: f32, VoH: f32,
    ior_in: f32, ior_out: f32,
    alpha: f32, extinction: vec3<f32>, distance: f32
) -> vec3<f32> {
    let eta = ior_in / ior_out;
    let cos_theta_t = sqrt(max(1.0 - eta * eta * (1.0 - NoV * NoV), 0.0));
    let cos_theta_i = NoV;
    let f = fresnel_schlick(max(cos_theta_i, EPS), fresnel0_from_ior(ior_in));
    let T = 1.0 - f;
    let D = ggx_ndf(VoH, alpha);
    let G = smith_ggx_correlated(NoV, NoL, alpha);
    let extinction_factor = exp(-extinction * distance);
    return vec3<f32>(D * G * T / max(4.0 * NoV * NoL, EPS)) * extinction_factor;
}
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
        "{}\n{}\n{}\n",
        lighting_wgsl_header(),
        lighting_fragment_entry::wgsl_source(),
        lighting_fragment_kernels()
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
