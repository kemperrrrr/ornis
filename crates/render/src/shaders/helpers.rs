//! Shared OpenPBR lighting helpers translated from Rust.
//!
//! The layer evaluators (`evaluate_*`, `transmission_btdf`) are textually
//! identical in the deferred-lighting and forward-PBR passes, so they live
//! here once: each `#[wgsl_fn]` function is replaced by a module exposing
//! `wgsl_source()`, and the passes splice those into the assembled WGSL.
//! [`octahedral_decode`]/[`reconstruct_world_pos`] are lighting-only
//! (g-buffer decode) and splice only there.
//!
//! Bodies are DSL-only and never compiled as Rust (like
//! [`stage`](ornis_macros::stage) entries): parameter types pass through by
//! name (`mat: OpenPBRMaterial`, `camera: Camera`), kernel calls stay bare
//! identifiers, and constructors are spelled `Vec3::new` / `Vec2::new`.
//! The float constants below are `f32`-exact (`format!` prints the short
//! decimal that round-trips to the same bits as the former literals).

/// Single-precision π, matching the former `3.14159265359` literal bit-wise.
pub const PI: f32 = std::f32::consts::PI;
/// Epsilon guard against division by zero (former `1e-6`).
pub const EPS: f32 = 1e-6;
/// Single-precision 1/π, matching the former `0.31830988618` bit-wise.
pub const INV_PI: f32 = std::f32::consts::FRAC_1_PI;

/// WGSL `const` block for the helpers, generated from the Rust constants.
pub fn wgsl_consts() -> String {
    format!("const PI: f32 = {PI};\nconst EPS: f32 = {EPS};\nconst INV_PI: f32 = {INV_PI};\n")
}

/// Base layer: dielectric/metallic mix with anisotropic GGX + Oren-Nayar.
#[ornis_macros::wgsl_fn]
fn evaluate_base_layer(
    n: glam::Vec3,
    v: glam::Vec3,
    l: glam::Vec3,
    h: glam::Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    mat: OpenPBRMaterial,
    base_weight: f32,
    base_color: glam::Vec3,
    metalness: f32,
    diffuse_roughness: f32,
    specular_weight: f32,
    specular_roughness: f32,
    specular_ior: f32,
    specular_anisotropy: f32,
    specular_edge_tint: glam::Vec3,
    t: glam::Vec3,
    b: glam::Vec3,
    thin_film_mod: glam::Vec3,
) -> glam::Vec3 {
    let f0_dielectric = Vec3::new(fresnel0_from_ior(specular_ior));
    let f0_metal = base_color * base_weight;
    let f0 = mix(f0_dielectric, f0_metal, metalness);
    let f = fresnel_f82_tint(voh, f0, specular_edge_tint);
    let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);
    let alpha_u = alpha.x;
    let alpha_v = alpha.y;
    let d = ggx_ndf_aniso(noh, h, t, b, alpha_u, alpha_v);
    let g = smith_ggx_aniso(nov, nol, v, l, t, b, alpha_u, alpha_v);
    let spec_brdf = d * g * f / max(4.0 * nov * nol, EPS);
    let diffuse_color = base_color * (1.0 - metalness);
    let diff_roughness = max(diffuse_roughness, specular_roughness);
    let diff_alpha = diff_roughness * diff_roughness;
    let cos_phi = max(dot(normalize(v - n * nov), normalize(l - n * nol)), 0.0);
    let diff_brdf = oren_nayar_brdf(nov, nol, cos_phi, diff_alpha);
    let ks = f * specular_weight;
    let kd = (Vec3::new(1.0) - luminance(ks)) * (1.0 - metalness);
    let base_bsdf = kd * diff_brdf * diffuse_color + ks * spec_brdf;
    return base_bsdf * base_weight * thin_film_mod;
}

/// Coat layer with base darkening under the coat.
#[ornis_macros::wgsl_fn]
fn evaluate_coat_layer(
    n: glam::Vec3,
    v: glam::Vec3,
    l: glam::Vec3,
    h: glam::Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    mat: OpenPBRMaterial,
    coat_weight: f32,
    coat_roughness: f32,
    coat_anisotropy: f32,
    coat_dark: f32,
    coat_ior: f32,
    coat_color: glam::Vec3,
    base_metalness: f32,
    base_color: glam::Vec3,
    base_weight: f32,
    specular_weight: f32,
    subsurface_weight: f32,
    subsurface_color: glam::Vec3,
    t: glam::Vec3,
    b: glam::Vec3,
) -> glam::Vec3 {
    if coat_weight <= 0.0 {
        return Vec3::new(0.0);
    }
    let coat_f0 = Vec3::new(fresnel0_from_ior(coat_ior));
    let coat_f = fresnel_schlick_vec(voh, coat_f0);
    let coat_alpha = openpbr_anisotropy(coat_roughness, coat_anisotropy);
    let coat_alpha_u = coat_alpha.x;
    let coat_alpha_v = coat_alpha.y;
    let coat_d = ggx_ndf_aniso(noh, h, t, b, coat_alpha_u, coat_alpha_v);
    let coat_g = smith_ggx_aniso(nov, nol, v, l, t, b, coat_alpha_u, coat_alpha_v);
    let coat_brdf = coat_d * coat_g * coat_f / max(4.0 * nov * nol, EPS);
    let mix_factor = coat_weight * coat_dark;
    let base_darkening = coat_base_darkening(
        coat_ior,
        base_metalness,
        base_color,
        base_weight,
        specular_weight,
        subsurface_weight,
        subsurface_color,
    );
    let darkening = coat_blend_darkened(base_darkening, mix_factor);
    let coat_albedo_approx = coat_color * coat_weight * luminance(coat_f0);
    return coat_color * coat_brdf * coat_weight + darkening * coat_albedo_approx;
}

/// Fuzz (sheen) layer.
#[ornis_macros::wgsl_fn]
fn evaluate_fuzz_layer(
    n: glam::Vec3,
    v: glam::Vec3,
    l: glam::Vec3,
    h: glam::Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    fuzz_weight: f32,
    fuzz_roughness: f32,
    fuzz_color: glam::Vec3,
) -> glam::Vec3 {
    if fuzz_weight <= 0.0 {
        return Vec3::new(0.0);
    }
    let sheen = sheen_brdf(nov, nol, noh, voh, fuzz_roughness);
    return fuzz_color * sheen * fuzz_weight;
}

/// Transmission layer with Beer-Lambert extinction and BTDF.
#[ornis_macros::wgsl_fn]
fn evaluate_transmission_layer(
    n: glam::Vec3,
    v: glam::Vec3,
    l: glam::Vec3,
    h: glam::Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    mat: OpenPBRMaterial,
    transmission_weight: f32,
    transmission_depth: f32,
    transmission_dispersion_scale: f32,
    transmission_dispersion_abbe: f32,
    transmission_color: glam::Vec3,
    transmission_scatter: glam::Vec3,
    transmission_scatter_anisotropy: f32,
    specular_ior: f32,
    specular_roughness: f32,
    specular_anisotropy: f32,
    thin_walled: f32,
) -> glam::Vec3 {
    if transmission_weight <= 0.0 {
        return Vec3::new(0.0);
    }
    let ior_out = 1.0;
    let ior_in = specular_ior;
    let eta = ior_in / ior_out;
    let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);
    let extinction = transmission_color_to_extinction(transmission_color, transmission_depth);
    let distance = transmission_depth;
    let btdf = transmission_btdf(
        nov, nol, voh, ior_in, ior_out, alpha.x, extinction, distance,
    );
    return transmission_color * btdf * transmission_weight;
}

/// Subsurface layer with projected-distance falloff.
#[ornis_macros::wgsl_fn]
fn evaluate_subsurface_layer(
    n: glam::Vec3,
    v: glam::Vec3,
    l: glam::Vec3,
    nov: f32,
    nol: f32,
    subsurface_weight: f32,
    subsurface_radius: f32,
    subsurface_radius_scale_r: f32,
    subsurface_radius_scale_g: f32,
    subsurface_radius_scale_b: f32,
    subsurface_scatter_anisotropy: f32,
    subsurface_color: glam::Vec3,
) -> glam::Vec3 {
    if subsurface_weight <= 0.0 {
        return Vec3::new(0.0);
    }
    let radius = Vec3::new(
        subsurface_radius * subsurface_radius_scale_r,
        subsurface_radius * subsurface_radius_scale_g,
        subsurface_radius * subsurface_radius_scale_b,
    );
    let v_proj = v - n * nov;
    let l_proj = l - n * nol;
    let distance = length(v_proj - l_proj);
    let ss_brdf = subsurface_brdf(nov, nol, distance, radius, subsurface_scatter_anisotropy);
    return subsurface_color * ss_brdf * subsurface_weight;
}

/// Emission with coat-weighted Fresnel modulation.
#[ornis_macros::wgsl_fn]
fn evaluate_emission(
    emission_luminance: f32,
    emission_color: glam::Vec3,
    coat_weight: f32,
    coat_color: glam::Vec3,
    nov: f32,
) -> glam::Vec3 {
    if emission_luminance <= 0.0 {
        return Vec3::new(0.0);
    }
    let base_emission = emission_color * emission_luminance * INV_PI;
    let coat_emission =
        coat_color * base_emission * (pow(1.0 - nov, 5.0) * coat_weight + (1.0 - coat_weight));
    return mix(base_emission, coat_emission, coat_weight);
}

/// Microfacet transmission BTDF with extinction.
#[ornis_macros::wgsl_fn]
fn transmission_btdf(
    nov: f32,
    nol: f32,
    voh: f32,
    ior_in: f32,
    ior_out: f32,
    alpha: f32,
    extinction: glam::Vec3,
    distance: f32,
) -> glam::Vec3 {
    let eta = ior_in / ior_out;
    let cos_theta_t = sqrt(max(1.0 - eta * eta * (1.0 - nov * nov), 0.0));
    let cos_theta_i = nov;
    let f = fresnel_schlick(max(cos_theta_i, EPS), fresnel0_from_ior(ior_in));
    let t = 1.0 - f;
    let d = ggx_ndf(voh, alpha);
    let g = smith_ggx_correlated(nov, nol, alpha);
    let extinction_factor = exp(-extinction * distance);
    return Vec3::new(d * g * t / max(4.0 * nov * nol, EPS)) * extinction_factor;
}

/// Octahedral normal decode (lighting-only g-buffer unpack).
#[ornis_macros::wgsl_fn]
fn octahedral_decode(p: glam::Vec2) -> glam::Vec3 {
    let mut n = Vec3::new(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    let t = max(-n.z, 0.0);
    let offset = select(-n.yx, n.yx, n.xy >= Vec2::new(0.0)) * t;
    n.x = n.x + offset.x;
    n.y = n.y + offset.y;
    return normalize(n);
}

/// World-position reconstruction from depth (lighting-only).
#[ornis_macros::wgsl_fn]
fn reconstruct_world_pos(uv: glam::Vec2, depth: f32, camera: Camera) -> glam::Vec3 {
    let ndc = Vec3::new(uv * 2.0 - 1.0, depth * 2.0 - 1.0);
    let clip = Vec4::new(ndc, 1.0);
    let view = camera.inv_view_proj * clip;
    return view.xyz / view.w;
}

/// All shared evaluator sources concatenated (consts excluded).
pub fn wgsl_shared_helpers() -> String {
    [
        evaluate_base_layer::wgsl_source(),
        evaluate_coat_layer::wgsl_source(),
        evaluate_fuzz_layer::wgsl_source(),
        evaluate_transmission_layer::wgsl_source(),
        evaluate_subsurface_layer::wgsl_source(),
        evaluate_emission::wgsl_source(),
        transmission_btdf::wgsl_source(),
    ]
    .join("\n")
}

/// Lighting-only decode helpers concatenated.
pub fn wgsl_lighting_decode() -> String {
    [
        octahedral_decode::wgsl_source(),
        reconstruct_world_pos::wgsl_source(),
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_sources_keep_legacy_signatures() {
        assert!(
            evaluate_base_layer::wgsl_source()
                .starts_with("fn evaluate_base_layer(n: vec3<f32>, v: vec3<f32>")
        );
        assert!(evaluate_coat_layer::wgsl_source().contains("mat: OpenPBRMaterial"));
        assert!(
            octahedral_decode::wgsl_source()
                .starts_with("fn octahedral_decode(p: vec2<f32>) -> vec3<f32>")
        );
        assert!(reconstruct_world_pos::wgsl_source().contains("camera: Camera"));
        // DSL spellings must not leak into WGSL.
        for src in [
            evaluate_base_layer::wgsl_source(),
            transmission_btdf::wgsl_source(),
            octahedral_decode::wgsl_source(),
        ] {
            assert!(!src.contains("Vec3"), "glam spelling leaked: {src}");
            assert!(!src.contains("glam"), "glam spelling leaked: {src}");
        }
    }

    #[test]
    fn rust_consts_are_f32_exact() {
        // The former decimal literals and the Rust constants round to the
        // same f32 bits (differences sit far below f32 epsilon).
        assert_eq!(PI.to_bits(), (std::f64::consts::PI as f32).to_bits());
        assert_eq!(EPS.to_bits(), 1e-6f32.to_bits());
        assert_eq!(
            INV_PI.to_bits(),
            (std::f64::consts::FRAC_1_PI as f32).to_bits()
        );
        let consts = wgsl_consts();
        assert!(consts.contains("const PI: f32"));
        assert!(consts.contains("const EPS: f32"));
        assert!(consts.contains("const INV_PI: f32"));
    }
}
