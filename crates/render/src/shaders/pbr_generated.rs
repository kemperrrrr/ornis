//! Forward-PBR shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module: the full OpenPBR
//! fragment skeleton (layer evaluators + `fs_main`) lives here as a Rust
//! string, and the 19 BRDF math kernels are spliced in from
//! [`crate::shaders::math`] (single source of truth via `#[kernel]`).
//! The handwritten `shaders/wgsl/pbr_vertex.wgsl` and
//! `shaders/wgsl/pbr_fragment.wgsl` remain as references; the
//! `pbr_generated_parity_with_legacy_assembly` test pins this module
//! byte-identical to them.
//!
//! Note: the vertex stage is shared with the g-buffer pass
//! (`shaders/wgsl/pbr_vertex.wgsl` is byte-identical to
//! `shaders/wgsl/gbuffer_vertex.wgsl`), so [`wgsl_vertex_source`] reuses
//! [`crate::shaders::gbuffer_generated::wgsl_vertex_source`] instead of
//! duplicating it.

use super::interface::GbufferFragmentInput;
use super::{OPENPBR_MATERIAL_DECL, wgsl_decl};
use crate::renderer::{CameraUniform, GpuLight, LightingUniform};
use crate::shaders::{gbuffer_generated, math};

/// Forward-PBR vertex shader: instance transforms + world position.
///
/// Delegates to [`gbuffer_generated::wgsl_vertex_source`] — the two legacy
/// vertex files are byte-identical. Entry point name `vs_main` is kept for
/// compatibility with `create_pbr_pass`.
pub fn wgsl_vertex_source() -> String {
    gbuffer_generated::wgsl_vertex_source()
}

/// Forward-PBR fragment shader: full OpenPBR evaluation.
///
/// Assembled as `{skeleton}\\n{kernel × 19}`, exactly like the legacy
/// `shaders::pbr_fragment()`; entry point `fs_main` is kept.
pub fn wgsl_source() -> String {
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
        math::srgb_to_linear::wgsl_source(),
    ];
    let mut src = format!(
        "\n{cam}\n{light}\n{lighting}{mat}\n{restA}\n{fin}\n{restB}",
        cam = CameraUniform::WGSL_SOURCE,
        light = wgsl_decl(GpuLight::WGSL_SOURCE),
        lighting = wgsl_decl(LightingUniform::WGSL_SOURCE),
        mat = OPENPBR_MATERIAL_DECL,
        restA = WGSL_FRAGMENT_HEAD,
        fin = wgsl_decl(GbufferFragmentInput::WGSL_SOURCE),
        restB = WGSL_FRAGMENT_TAIL,
    );
    for k in &kernels {
        src.push('\n');
        src.push_str(k);
    }
    src
}

/// Static view for naga validation in tests.
pub fn wgsl_source_static() -> String {
    wgsl_source()
}

const WGSL_FRAGMENT_HEAD: &str = r#"@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> materials: array<OpenPBRMaterial>;
@group(0) @binding(3) var<uniform> lighting: Lighting;
"#;

const WGSL_FRAGMENT_TAIL: &str = r#"const PI: f32 = 3.14159265359;
const EPS: f32 = 1e-6;
const INV_PI: f32 = 0.31830988618;

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

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let mat = materials[input.material_index];
    let N = normalize(input.world_normal);
    let V = normalize(camera.camera_pos.xyz - input.world_position);
    let NoV = max(dot(N, V), EPS);

    let T = normalize(input.world_tangent);
    let B = cross(N, T);

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

    var Lo = vec3<f32>(0.0);

    let thin_film_mod = thin_film_modulation(NoV, thin_film_ior, thin_film_thickness_um, 1.0);

    for (var i = 0u; i < lighting.light_count; i = i + 1u) {
        let L = normalize(lighting.lights[i].direction.xyz);
        let H = normalize(V + L);
        let light_color = lighting.lights[i].color.rgb;
        let intensity = lighting.lights[i].color.w;
        let radiance = light_color * intensity;

        let NoL = max(dot(N, L), EPS);
        let NoH = max(dot(N, H), EPS);
        let VoH = max(dot(V, H), EPS);

        if (NoL <= EPS) { continue; }

        let base_bsdf = evaluate_base_layer(
            N, V, L, H, NoV, NoL, NoH, VoH, mat,
            base_weight, base_color, metalness, diffuse_roughness,
            specular_weight, specular_roughness, specular_ior, specular_anisotropy, specular_edge_tint,
            T, B, thin_film_mod
        );

        let coat_bsdf = evaluate_coat_layer(
            N, V, L, H, NoV, NoL, NoH, VoH, mat,
            coat_weight, coat_roughness, coat_anisotropy, coat_darkening, coat_ior, coat_color,
            metalness, base_color, base_weight, specular_weight,
            subsurface_weight, subsurface_color,
            T, B
        );

        let fuzz_bsdf = evaluate_fuzz_layer(
            N, V, L, H, NoV, NoL, NoH, VoH,
            fuzz_weight, fuzz_roughness, fuzz_color
        );

        let trans_bsdf = evaluate_transmission_layer(
            N, V, L, H, NoV, NoL, NoH, VoH, mat,
            transmission_weight, transmission_depth, transmission_dispersion_scale, transmission_dispersion_abbe,
            transmission_color, transmission_scatter, transmission_scatter_anisotropy,
            specular_ior, specular_roughness, specular_anisotropy,
            thin_walled
        );

        let ss_bsdf = evaluate_subsurface_layer(
            N, V, L, NoV, NoL,
            subsurface_weight, subsurface_radius, subsurface_radius_scale_r,
            subsurface_radius_scale_g, subsurface_radius_scale_b,
            subsurface_scatter_anisotropy, subsurface_color
        );

        let layer_bsdf = base_bsdf + coat_bsdf + fuzz_bsdf + trans_bsdf + ss_bsdf;
        Lo += layer_bsdf * radiance * NoL;
    }

    let ambient = lighting.ambient_color.rgb * mix(base_color, base_color * specular_weight, metalness);
    let emission = evaluate_emission(emission_luminance, emission_color, coat_weight, coat_color, NoV);

    let color = ambient + Lo + emission;
    let tone_mapped = aces_tonemap(color);
    return vec4<f32>(tone_mapped, opacity);
}
"#;

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
        assert!(fs.contains("fn aces_tonemap"));
    }

    #[test]
    fn pbr_generated_parity_with_legacy_assembly() {
        // Vertex is shared with gbuffer (translated — see
        // `gbuffer_vertex_entry_matches_legacy_shape`); pin the sharing.
        assert_eq!(
            wgsl_vertex_source(),
            super::super::gbuffer_generated::wgsl_vertex_source()
        );
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
            math::srgb_to_linear::wgsl_source(),
        ];
        let mut legacy_fragment = include_str!("wgsl/pbr_fragment.wgsl").to_string();
        for k in &kernels {
            legacy_fragment.push('\n');
            legacy_fragment.push_str(k);
        }
        assert_eq!(wgsl_source(), legacy_fragment);
    }
}
