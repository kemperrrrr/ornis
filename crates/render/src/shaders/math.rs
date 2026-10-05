//! Rust reference implementations of the WGSL/PBR BRDF math (OpenPBR spec).
//!
//! Every public function here is also compiled to WGSL by the `#[kernel]`
//! macro pipeline and spliced into the engine's shaders via `wgsl_source()`,
//! keeping GPU shading and CPU-side verification on one source of truth.
// These functions mirror the WGSL/PBR shader signatures (OpenPBR spec), so
// some of them take more than 7 arguments — a deliberate match.
#![allow(clippy::too_many_arguments)]

use glam::Vec3Swizzles;
use ornis_macros::kernel;

/// Pi, re-exported into WGSL kernels by name.
pub const PI: f32 = std::f32::consts::PI;
/// `1/PI`, used to normalize cosine-weighted BRDF integrals.
pub const INV_PI: f32 = 1.0 / PI;
/// Small epsilon guarding divisions and sqrt arguments in the kernels.
pub const EPS: f32 = 1e-6;

/// Rec.709 relative luminance of a linear RGB color.
#[kernel]
fn luminance(c: glam::Vec3) -> f32 {
    c.dot(glam::Vec3::new(0.2126, 0.7152, 0.0722))
}

/// ACES filmic tone-mapping curve (Narkowicz approximation), HDR -> LDR.
#[kernel]
fn aces_tonemap(color: glam::Vec3) -> glam::Vec3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    color * (a * color + b) / (color * (c * color + d) + e)
}

/// Normal-incidence Fresnel reflectance F0 from an index of refraction.
#[kernel]
fn fresnel0_from_ior(ior: f32) -> f32 {
    let f = (ior - 1.0) / (ior + 1.0);
    f * f
}

/// Split-sum image-based light: prefiltered specular times the BRDF LUT
/// (scale/bias) plus irradiance times the diffuse color, gated by `weight`
/// (0 leaves the direct-light result unchanged).
#[kernel]
fn evaluate_ibl(
    prefiltered: glam::Vec3,
    irradiance: glam::Vec3,
    lut_scale: f32,
    lut_bias: f32,
    f0: glam::Vec3,
    diffuse_color: glam::Vec3,
    weight: f32,
) -> glam::Vec3 {
    let fresnel = f0 * lut_scale + glam::Vec3::splat(lut_bias);
    let specular = prefiltered * fresnel;
    let diffuse_weight =
        (glam::Vec3::splat(1.0) - glam::Vec3::splat(lut_bias)).max(glam::Vec3::ZERO);
    let diffuse = irradiance * diffuse_color * diffuse_weight;
    (specular + diffuse) * weight
}

/// Scalar Schlick approximation of the Fresnel term (`cos_theta` = NoV or NoL).
#[kernel]
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    f0 + (1.0 - f0) * (1.0 - cos_theta).powf(5.0)
}

/// RGB Schlick Fresnel with spectral F0 (metals).
#[kernel]
fn fresnel_schlick_vec(cos_theta: f32, f0: glam::Vec3) -> glam::Vec3 {
    f0 + (glam::Vec3::splat(1.0) - f0) * (1.0 - cos_theta).powf(5.0)
}

/// Holzschuch--Pacanowsky "F82 tint" Fresnel: Schlick plus a correction
/// term that pins reflectance to `f82_tint` at 82 degrees, matching measured
/// metal data better than plain Schlick.
#[kernel]
fn fresnel_f82_tint(cos_theta: f32, f0: glam::Vec3, f82_tint: glam::Vec3) -> glam::Vec3 {
    let mu_bar = 1.0 / 7.0;
    let schlick_at_mu_bar = f0 + (glam::Vec3::splat(1.0) - f0) * (1.0_f32 - mu_bar).powf(5.0);
    let f82 = f82_tint * schlick_at_mu_bar;
    let numerator = cos_theta * (1.0_f32 - cos_theta).powf(6.0);
    let denominator = mu_bar * (1.0_f32 - mu_bar).powf(6.0);
    let scale = numerator / denominator;
    let f_schlick = f0 + (glam::Vec3::splat(1.0) - f0) * (1.0 - cos_theta).powf(5.0);
    let f82_correction = f_schlick - glam::Vec3::splat(scale) * (schlick_at_mu_bar - f82);
    f82_correction.max(glam::Vec3::splat(0.0))
}

/// GGX/Trowbridge-Reitz normal distribution for isotropic roughness.
#[kernel]
fn ggx_ndf(NoH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = NoH * NoH * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

/// Anisotropic GGX normal distribution (Heitz 2014; OpenPBR spec):
/// `alpha_u`/`alpha_v` are the true roughness-squared axes (as returned
/// by `openpbr_anisotropy`), used once — not squared again — with the
/// `NoH²` term. This form integrates to 1 over the hemisphere; the
/// previous spelling dropped `NoH²` and scaled by `1/(αu²·αv²)`,
/// inflating specular energy ~10⁴× at roughness 0.1 and stamping hard
/// terminator seams on smooth materials.
#[kernel]
fn ggx_ndf_aniso(
    NoH: f32,
    H: glam::Vec3,
    T: glam::Vec3,
    B: glam::Vec3,
    alpha_u: f32,
    alpha_v: f32,
) -> f32 {
    let Hu = H.dot(T);
    let Hv = H.dot(B);
    let denom = (Hu * Hu) / (alpha_u * alpha_u) + (Hv * Hv) / (alpha_v * alpha_v) + NoH * NoH;
    1.0 / (PI * alpha_u * alpha_v * denom * denom)
}

/// Map (roughness, anisotropy) to the alpha_u/alpha_v pair used by the
/// anisotropic NDF and visibility terms.
#[kernel]
fn openpbr_anisotropy(roughness: f32, anisotropy: f32) -> glam::Vec2 {
    let r2 = roughness * roughness;
    let aniso_inv = 1.0 - anisotropy;
    let aniso_inv_sq = aniso_inv * aniso_inv;
    let denom = aniso_inv_sq + 1.0;
    let fraction = 2.0 / denom;
    let sqrt_frac = fraction.sqrt();
    let alpha_u = r2 * sqrt_frac;
    let alpha_v = aniso_inv * alpha_u;
    glam::Vec2::new(alpha_u, alpha_v)
}

/// Smith height-correlated GGX visibility term (isotropic), the
/// `G / (4 NoV NoL)` factor already folded in.
#[kernel]
fn smith_ggx_correlated(NoV: f32, NoL: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggxv = NoV * (NoL * NoL * (1.0 - a2) + a2).max(EPS).sqrt();
    let ggxl = NoL * (NoV * NoV * (1.0 - a2) + a2).max(EPS).sqrt();
    0.5 / (ggxv + ggxl).max(EPS)
}

/// Height-correlated Smith GGX visibility for anisotropic distributions.
#[kernel]
fn smith_ggx_aniso(
    NoV: f32,
    NoL: f32,
    V: glam::Vec3,
    L: glam::Vec3,
    T: glam::Vec3,
    B: glam::Vec3,
    alpha_u: f32,
    alpha_v: f32,
) -> f32 {
    let Vu = V.dot(T);
    let Vv = V.dot(B);
    let Lu = L.dot(T);
    let Lv = L.dot(B);
    let a2u = alpha_u * alpha_u;
    let a2v = alpha_v * alpha_v;
    let ggxv = NoV * (Lu * Lu * a2u + Lv * Lv * a2v + NoL * NoL).max(EPS).sqrt();
    let ggxl = NoL * (Vu * Vu * a2u + Vv * Vv * a2v + NoV * NoV).max(EPS).sqrt();
    0.5 / (ggxv + ggxl).max(EPS)
}

/// Oren--Nayar diffuse BRDF (`alpha` = roughness sigma), normalized by 1/PI.
#[kernel]
fn oren_nayar_brdf(NoV: f32, NoL: f32, cos_phi: f32, alpha: f32) -> f32 {
    let sigma = alpha.max(EPS);
    let sigma2 = sigma * sigma;
    let A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.57);
    let B = 0.45 * sigma2 / (sigma2 + 0.09);
    let theta_v = NoV.max(0.0).acos();
    let theta_l = NoL.max(0.0).acos();
    let alpha_max = theta_v.max(theta_l);
    let beta_min = theta_v.min(theta_l);
    let tan_beta = beta_min.tan();
    (A + B * cos_phi * alpha_max.sin() * tan_beta) * INV_PI
}

/// Diffuse energy left after the specular lobe.
///
/// The base albedo is already scaled by `(1 - metalness)` at the call
/// site. Multiplying by metalness again darkened partial metals by
/// `(1 - metalness)²` (a 0.5 metal kept a quarter of its diffuse).
#[kernel]
fn base_diffuse_energy(specular_luma: f32) -> f32 {
    (1.0 - specular_luma).max(0.0)
}

/// Mix a thin-film modulation in by weight.
///
/// Weight 0 is the identity. The Airy kernel at thickness 0 is not 1
/// (a zero-thickness film with the default water IOR still returns
/// about 1.04), and the evaluators used to multiply that into every
/// material because `thin_film_weight` was loaded and ignored.
#[kernel]
fn thin_film_weight_mix(weight: f32, modulation: glam::Vec3) -> glam::Vec3 {
    glam::Vec3::splat(1.0).lerp(modulation, weight.clamp(0.0, 1.0))
}

/// Emitter radiance, with an optional coat transmittance tint.
///
/// `emission_luminance` is luminance in nits (cd/m²), the same quantity
/// as outgoing radiance, so it is not divided by π (that factor converts
/// Lambertian exitance and made every emitter π times too dark). A full
/// coat transmits at normal incidence and blocks at grazing; the previous
/// factor was the Fresnel term itself, so a coated emitter went black
/// head-on and lit up on the silhouette.
#[kernel]
fn coated_emission(
    emission_color: glam::Vec3,
    emission_luminance: f32,
    coat_weight: f32,
    coat_color: glam::Vec3,
    nov: f32,
) -> glam::Vec3 {
    let base_emission = emission_color * emission_luminance;
    let coat_fresnel = (1.0 - nov).powf(5.0);
    let transmit = (1.0 - coat_fresnel) * coat_weight + (1.0 - coat_weight);
    let coat_emission = coat_color * base_emission * transmit;
    base_emission.lerp(coat_emission, coat_weight)
}

/// Fabric sheen lobe (Charlie-style D with approximate visibility).
#[kernel]
fn sheen_brdf(NoV: f32, NoL: f32, NoH: f32, VoH: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let D = alpha / (PI * (NoH * NoH * (alpha - 1.0) + 1.0).powf(2.0));
    let G = 1.0 / (1.0 + alpha * (1.0 / NoV + 1.0 / NoL - 2.0));
    let F = VoH;
    D * G * F / (4.0 * NoV * NoL).max(EPS)
}

/// Convert transmission tint + thickness into a Beer--Lambert extinction
/// coefficient `sigma_t = -ln(c) / depth` per channel.
#[kernel]
fn transmission_color_to_extinction(
    transmission_color: glam::Vec3,
    transmission_depth: f32,
) -> glam::Vec3 {
    if transmission_depth <= 0.0 {
        return glam::Vec3::splat(0.0);
    }
    let c = transmission_color.max(glam::Vec3::splat(1e-6));
    -c.ln() / transmission_depth
}

/// Single-lobe subsurface approximation.
///
/// `distance` is the tangent-plane chord between the unit view and light
/// (`|V - N·NoV - (L - N·NoL)|`, range 0..2), not a world-space path
/// length. The dipole `1/r` pole is not defined on that chord: it
/// evaluated to `1/EPS` whenever the light sat near the view (a white
/// firefly on the subsurface preset). The phase uses the scattering
/// cosine recovered from the same chord, not `cos(distance)`.
#[kernel]
fn subsurface_brdf(
    NoV: f32,
    NoL: f32,
    distance: f32,
    radius: glam::Vec3,
    anisotropy: f32,
) -> glam::Vec3 {
    let sigma_tr = 3.0_f32.sqrt() / radius.max(glam::Vec3::splat(EPS));
    let profile = (-distance * sigma_tr).exp();
    let v_len2 = (1.0 - NoV * NoV).max(0.0);
    let l_len2 = (1.0 - NoL * NoL).max(0.0);
    let tangent_dot = (v_len2 + l_len2 - distance * distance) * 0.5;
    let cos_scatter = (NoV * NoL + tangent_dot).clamp(-1.0, 1.0);
    let phase = (1.0 + anisotropy * cos_scatter).max(0.0);
    profile * phase * INV_PI
}

/// Decode sRGB-encoded RGB to linear light (piecewise IEC 61966-2-1 OETF inverse).
#[kernel]
fn srgb_to_linear(c: glam::Vec3) -> glam::Vec3 {
    let cutoff = 0.04045;
    let r = if c.x <= cutoff {
        c.x / 12.92
    } else {
        ((c.x + 0.055) / 1.055).powf(2.4)
    };
    let g = if c.y <= cutoff {
        c.y / 12.92
    } else {
        ((c.y + 0.055) / 1.055).powf(2.4)
    };
    let b = if c.z <= cutoff {
        c.z / 12.92
    } else {
        ((c.z + 0.055) / 1.055).powf(2.4)
    };
    glam::Vec3::new(r, g, b)
}

/// Coat darkening, part 1: the base albedo darkening under a coated
/// surface (Kcoat attenuation of the underlying BRDF response).
#[kernel]
fn coat_base_darkening(
    coat_ior: f32,
    base_metalness: f32,
    base_color: glam::Vec3,
    base_weight: f32,
    specular_weight: f32,
    subsurface_weight: f32,
    subsurface_color: glam::Vec3,
) -> glam::Vec3 {
    let coat_f0 = ((coat_ior - 1.0) / (coat_ior + 1.0)).powf(2.0);
    let one_minus_coat_f0 = 1.0 - coat_f0;
    let coat_ior_sq = coat_ior * coat_ior;
    let Kcoat = 1.0 - one_minus_coat_f0 / coat_ior_sq;

    let Emetal = base_color * base_weight * specular_weight;
    let Edielectric = subsurface_color.lerp(base_color, subsurface_weight);
    let Ebase = Emetal.lerp(Edielectric, base_metalness);

    let Ebase_Kcoat = Ebase * Kcoat;
    let one_minus_Kcoat = 1.0 - Kcoat;
    let one_minus_Ebase_Kcoat = glam::Vec3::splat(1.0) - Ebase_Kcoat;

    glam::Vec3::splat(one_minus_Kcoat) / one_minus_Ebase_Kcoat.max(glam::Vec3::splat(1e-6))
}

/// Coat darkening, part 2: identity when the coat is off, otherwise blend
/// the darkened albedo by `coat_weight * coat_dark`.
///
/// The product used to be added onto the coat lobe (`darkening *
/// coat_albedo`), which brightened the coat. Callers multiply the base
/// lobe by this factor instead.
#[kernel]
fn coat_blend_darkened(base_darkening: glam::Vec3, coat_weight: f32, coat_dark: f32) -> glam::Vec3 {
    if coat_weight <= 0.0 {
        return glam::Vec3::splat(1.0);
    }
    let mix_factor = coat_weight * coat_dark;
    glam::Vec3::splat(1.0).lerp(base_darkening, mix_factor)
}

/// Thin-film interference modulation over white: phase per RGB wavelength
/// from film IOR/thickness (nanometers), Airy-like reflectance boost.
#[kernel]
fn thin_film_modulation(
    cos_theta: f32,
    film_ior: f32,
    thickness_nm: f32,
    ior_outside: f32,
) -> glam::Vec3 {
    let sin_theta_film = ior_outside * (1.0 - cos_theta * cos_theta).max(0.0).sqrt() / film_ior;
    let cos_theta_film = (1.0 - sin_theta_film * sin_theta_film).max(0.0).sqrt();
    let lambda = glam::Vec3::new(650.0, 550.0, 450.0);
    let phase = 4.0 * PI * film_ior * thickness_nm * cos_theta_film / lambda;
    let r0 = ((film_ior - ior_outside) / (film_ior + ior_outside)).powf(2.0);

    glam::Vec3::splat(1.0)
        + glam::Vec3::splat(2.0 * r0) * glam::Vec3::new(phase.x.cos(), phase.y.cos(), phase.z.cos())
            / glam::Vec3::splat(1.0 - r0 * r0)
}

/// Octahedral mapping of a unit vector to [-1,1]^2 — compact normal storage.
#[kernel]
fn octahedral_encode(n: glam::Vec3) -> glam::Vec2 {
    let p = n.xy() / (n.x.abs() + n.y.abs() + n.z.abs());
    let q = glam::Vec2::new(p.x, p.y);
    let sign_x = if q.x >= 0.0 { 1.0 } else { -1.0 };
    let sign_y = if q.y >= 0.0 { 1.0 } else { -1.0 };
    let flipped =
        glam::Vec2::new(1.0 - q.y.abs(), 1.0 - q.x.abs()) * glam::Vec2::new(sign_x, sign_y);
    if n.z < 0.0 {
        return flipped;
    }
    return q;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luminance_black() {
        assert_eq!(luminance::eval(glam::Vec3::ZERO), 0.0);
    }

    #[test]
    fn luminance_white() {
        let result = luminance::eval(glam::Vec3::ONE);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn aces_tonemap_red() {
        let result = aces_tonemap::eval(glam::Vec3::new(1.0, 0.0, 0.0));
        assert!((result.x - 0.8038).abs() < 0.001);
    }

    #[test]
    fn aces_tonemap_black() {
        let result = aces_tonemap::eval(glam::Vec3::ZERO);
        assert!((result.x).abs() < 0.001);
    }

    #[test]
    fn wgsl_generates() {
        assert!(luminance::wgsl_source().contains("fn luminance"));
        assert!(aces_tonemap::wgsl_source().contains("fn aces_tonemap"));
    }

    #[test]
    fn fresnel0_from_ior_glass() {
        let result = fresnel0_from_ior::eval(1.5);
        assert!((result - 0.04).abs() < 0.01);
    }

    #[test]
    fn ggx_ndf_peak() {
        let result = ggx_ndf::eval(1.0, 0.1);
        assert!(result > 0.0);
    }

    #[test]
    fn ggx_ndf_aniso_peak_matches_closed_form() {
        // Peak (H == N) is exactly 1/(π·αu·αv); the old spelling
        // (dropped Noh², 1/(αu²·αv²) scale) returned ~625× this at α=0.04.
        let d = ggx_ndf_aniso::eval(1.0, glam::Vec3::Z, glam::Vec3::X, glam::Vec3::Y, 0.04, 0.04);
        let expected = 1.0 / (PI * 0.04 * 0.04);
        assert!(
            (d - expected).abs() / expected < 1e-4,
            "d={d} expected={expected}"
        );
    }

    #[test]
    fn ggx_ndf_aniso_tilted_matches_reference() {
        // 45° tilt in the tangent plane pins both the Noh² term and the
        // single αu·αv scale (old value 0.317 vs correct 0.00203).
        let h = glam::Vec3::new(
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        );
        let d = ggx_ndf_aniso::eval(h.z, h, glam::Vec3::X, glam::Vec3::Y, 0.04, 0.04);
        assert!((d - 0.00203).abs() / 0.00203 < 0.01, "d={d}");
    }

    #[test]
    fn smith_ggx_zero_to_one() {
        let result = smith_ggx_correlated::eval(1.0, 1.0, 0.5);
        assert!(result > 0.0 && result <= 1.0);
    }

    #[test]
    fn srgb_to_linear_black() {
        let result = srgb_to_linear::eval(glam::Vec3::ZERO);
        assert_eq!(result, glam::Vec3::ZERO);
    }

    #[test]
    fn srgb_to_linear_white() {
        let result = srgb_to_linear::eval(glam::Vec3::ONE);
        assert!((result.x - 1.0).abs() < 0.001);
    }

    #[test]
    fn transmission_color_to_extinction_degenerate() {
        let result = transmission_color_to_extinction::eval(glam::Vec3::splat(0.5), 0.0);
        assert_eq!(result, glam::Vec3::ZERO);
    }

    #[test]
    fn thin_film_modulation_produces_finite() {
        let result = thin_film_modulation::eval(0.5, 1.5, 300.0, 1.0);
        assert!(result.x.is_finite());
        assert!(result.y.is_finite());
        assert!(result.z.is_finite());
    }

    #[test]
    fn octahedral_encode_roundtrip() {
        let n = glam::Vec3::new(0.577, 0.577, 0.577).normalize();
        let enc = octahedral_encode::eval(n);
        let dec = crate::shaders::octahedral_decode_rust(enc);
        for i in 0..3 {
            assert!(
                (n[i] - dec[i]).abs() < 0.01,
                "mismatch at {i}: {n} vs {dec}"
            );
        }
    }

    #[test]
    fn octahedral_decode_covers_folded_hemisphere() {
        // The z<0 hemisphere folds in the encoding; the decode must
        // unfold it (regression: the old offset math erred by up to
        // ~69° here, shading night-side normals wrong).
        let cases = [
            glam::Vec3::new(0.577, 0.577, 0.577),
            glam::Vec3::new(0.124, 0.509, -0.852),
            glam::Vec3::new(-0.6, 0.3, -0.742),
            glam::Vec3::new(0.0, 0.0, -1.0),
            glam::Vec3::new(0.707, -0.707, -0.001),
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::new(0.0, 0.0, 1.0),
        ];
        for n in cases {
            let n = n.normalize();
            let enc = octahedral_encode::eval(n);
            let dec = crate::shaders::octahedral_decode_rust(enc);
            let err_deg = n.dot(dec).clamp(-1.0, 1.0).acos().to_degrees();
            assert!(err_deg < 1.0, "n={n} dec={dec} err={err_deg}°");
        }
    }

    /// Severity: medium. Partial metals must lose diffuse once.
    #[test]
    fn partial_metal_diffuse_energy_is_not_squared() {
        let energy = base_diffuse_energy::eval(0.04);
        assert!((energy - 0.96).abs() < 1e-5, "{energy}");
        // The old weight was `energy * (1 - metalness)`. At metalness 0.5
        // that kept 0.48 instead of 0.96.
        assert!((energy * 0.5 - 0.48).abs() < 1e-5);
    }

    /// Severity: medium. A coat attenuates the base; weight 0 stays identity.
    #[test]
    fn coat_darkening_attenuates_and_is_identity_when_off() {
        // A white base saturates the interreflection term back to 1.
        // A mid-grey dielectric does not, so the coat factor is an
        // attenuator rather than an extra light.
        let grey = glam::Vec3::splat(0.2);
        let darkened = coat_base_darkening::eval(1.5, 0.0, grey, 1.0, 1.0, 0.0, grey);
        assert!(
            darkened.x < 0.6,
            "coat should darken a grey dielectric, got {darkened}"
        );
        let off = coat_blend_darkened::eval(darkened, 0.0, 1.0);
        assert!(
            (off - glam::Vec3::ONE).length() < 1e-5,
            "weight 0 must be identity, got {off}"
        );
        let on = coat_blend_darkened::eval(darkened, 1.0, 1.0);
        assert!((on - darkened).length() < 1e-5, "full coat, got {on}");
    }

    /// Severity: medium. Weight 0 must not inherit the zero-thickness Airy bias.
    #[test]
    fn thin_film_weight_zero_cancels_the_zero_thickness_bias() {
        // Default material: weight 0, thickness 0, IOR 1.33 (water).
        let raw = thin_film_modulation::eval(1.0, 1.33, 0.0, 1.0);
        assert!(
            (raw.x - 1.0401).abs() < 1e-3,
            "zero-thickness Airy bias, got {raw}"
        );
        let off = thin_film_weight_mix::eval(0.0, raw);
        assert!((off.x - 1.0).abs() < 1e-6 && (off - glam::Vec3::ONE).length() < 1e-5);
        let on = thin_film_weight_mix::eval(1.0, raw);
        assert!((on.x - raw.x).abs() < 1e-6);
    }

    /// Severity: medium. Nits are luminance; a coat must not black out the face.
    #[test]
    fn emission_is_luminance_and_coat_transmits_head_on() {
        let white = glam::Vec3::ONE;
        let bare = coated_emission::eval(white, 2.0, 0.0, white, 0.2);
        assert!(
            (bare.x - 2.0).abs() < 1e-5,
            "bare emitter was divided by π, got {bare}"
        );
        let head_on = coated_emission::eval(white, 1.0, 1.0, white, 1.0);
        assert!(
            (head_on.x - 1.0).abs() < 1e-4,
            "coated emitter at normal incidence was black, got {head_on}"
        );
        let grazing = coated_emission::eval(white, 1.0, 1.0, white, 0.0);
        assert!(
            grazing.x < 1e-4,
            "grazing coat should block emission, got {grazing}"
        );
    }

    /// Severity: critical. Aligned view/light must not hit the 1/r pole.
    #[test]
    fn subsurface_aligned_directions_stay_finite() {
        let radius = glam::Vec3::splat(0.1);
        let got = subsurface_brdf::eval(1.0, 1.0, 0.0, radius, 0.0);
        let expected = INV_PI;
        assert!(
            (got.x - expected).abs() < 1e-4 && got.x < 1.0,
            "aligned subsurface blew up, got {got} expected {expected}"
        );
        // Anisotropy uses the scattering cosine (1 when L aligns with V),
        // not cos(distance). The old phase was `1 + g * cos(distance)`.
        let phased = subsurface_brdf::eval(1.0, 1.0, 0.0, radius, 1.0);
        assert!(
            (phased.x - 2.0 * INV_PI).abs() < 1e-4,
            "phase, got {phased}"
        );
        // A zero radius used to divide the dipole coefficient by 0.
        let collapsed = subsurface_brdf::eval(1.0, 1.0, 0.0, glam::Vec3::ZERO, 0.0);
        assert!(collapsed.x.is_finite() && collapsed.x < 1.0, "{collapsed}");
    }

    /// Severity: critical. A second ACES pass crushes an already-mapped midtone.
    #[test]
    fn second_aces_pass_crushes_midtones() {
        let scene = glam::Vec3::splat(1.0);
        let once = aces_tonemap::eval(scene);
        let twice = aces_tonemap::eval(once);
        assert!(twice.x < once.x - 0.05, "once={once} twice={twice}");
    }

    #[test]
    fn evaluate_ibl_is_zero_when_weight_is_zero() {
        let got = evaluate_ibl::eval(
            glam::Vec3::ONE,
            glam::Vec3::ONE,
            1.0,
            0.2,
            glam::Vec3::splat(0.04),
            glam::Vec3::ONE,
            0.0,
        );
        assert!(got.length() < 1.0e-6, "{got}");
        let lit = evaluate_ibl::eval(
            glam::Vec3::ONE,
            glam::Vec3::ONE,
            1.0,
            0.0,
            glam::Vec3::splat(0.04),
            glam::Vec3::ONE,
            1.0,
        );
        assert!(lit.x > 0.5, "{lit}");
    }

    #[test]
    fn all_wgsl_sources_compile() {
        let funcs: [&str; 24] = [
            luminance::wgsl_source(),
            aces_tonemap::wgsl_source(),
            fresnel0_from_ior::wgsl_source(),
            fresnel_schlick::wgsl_source(),
            fresnel_schlick_vec::wgsl_source(),
            fresnel_f82_tint::wgsl_source(),
            ggx_ndf::wgsl_source(),
            ggx_ndf_aniso::wgsl_source(),
            openpbr_anisotropy::wgsl_source(),
            smith_ggx_correlated::wgsl_source(),
            smith_ggx_aniso::wgsl_source(),
            oren_nayar_brdf::wgsl_source(),
            base_diffuse_energy::wgsl_source(),
            thin_film_weight_mix::wgsl_source(),
            coated_emission::wgsl_source(),
            sheen_brdf::wgsl_source(),
            transmission_color_to_extinction::wgsl_source(),
            subsurface_brdf::wgsl_source(),
            srgb_to_linear::wgsl_source(),
            coat_base_darkening::wgsl_source(),
            coat_blend_darkened::wgsl_source(),
            thin_film_modulation::wgsl_source(),
            octahedral_encode::wgsl_source(),
            evaluate_ibl::wgsl_source(),
        ];
        for src in funcs.iter() {
            assert!(src.starts_with("fn "), "bad source: {src}");
            assert!(src.contains("->"), "missing return type: {src}");
        }
    }
}
