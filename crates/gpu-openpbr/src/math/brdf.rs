//! Reflectance lobes: Fresnel, the GGX distribution and visibility, and
//! the diffuse and fabric terms.

// The lobes mirror the OpenPBR spec signatures, and an anisotropic visibility
// takes both roughness axes plus both directions — seven arguments is the
// reference's own budget, not an accidental pile-up.
#![allow(clippy::too_many_arguments)]

use super::{
    EPS, F82_CORRECTION_POWER, F82_PIVOT_COSINE, INV_PI, MICROFACET_SPECULAR_SCALE,
    OREN_NAYAR_A_NUMERATOR, OREN_NAYAR_A_OFFSET, OREN_NAYAR_B_NUMERATOR, OREN_NAYAR_B_OFFSET, PI,
    SCHLICK_FRESNEL_POWER, SMITH_GGX_VISIBILITY_SCALE, dot, max3, mul3, scale3, splat3, sub3,
};
use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::acos;
use ornis_gpu_shader::pow;
use ornis_gpu_shader::sin;
use ornis_gpu_shader::sqrt;
use ornis_gpu_shader::tan;

/// Normal-incidence Fresnel reflectance F0 from an index of refraction.
#[inline(always)]
#[must_use]
pub fn fresnel0_from_ior(ior: f32) -> f32 {
    let f = (ior - 1.0) / (ior + 1.0);
    f * f
}

/// Scalar Schlick approximation of the Fresnel term (`cos_theta` = NoV or NoL).
#[inline(always)]
#[must_use]
pub fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    f0 + (1.0 - f0) * pow(1.0 - cos_theta, SCHLICK_FRESNEL_POWER)
}

/// RGB Schlick Fresnel with spectral F0 (metals).
#[inline(always)]
#[must_use]
pub fn fresnel_schlick_vec(cos_theta: f32, f0: Vec3) -> Vec3 {
    let weight = pow(1.0 - cos_theta, SCHLICK_FRESNEL_POWER);
    Vec3::new(
        f0.x + (1.0 - f0.x) * weight,
        f0.y + (1.0 - f0.y) * weight,
        f0.z + (1.0 - f0.z) * weight,
    )
}

/// Holzschuch–Pacanowsky "F82 tint" Fresnel: Schlick plus a correction
/// term that pins reflectance to `f82_tint` at 82 degrees, matching measured
/// metal data better than plain Schlick.
#[inline(always)]
#[must_use]
pub fn fresnel_f82_tint(cos_theta: f32, f0: Vec3, f82_tint: Vec3) -> Vec3 {
    let mu_bar = F82_PIVOT_COSINE;
    let schlick_at_mu_bar = fresnel_schlick_vec(mu_bar, f0);
    let f82 = mul3(f82_tint, schlick_at_mu_bar);
    let numerator = cos_theta * pow(1.0 - cos_theta, F82_CORRECTION_POWER);
    let denominator = mu_bar * pow(1.0 - mu_bar, F82_CORRECTION_POWER);
    let scale = numerator / denominator;
    let f_schlick = fresnel_schlick_vec(cos_theta, f0);
    let correction = sub3(f_schlick, scale3(sub3(schlick_at_mu_bar, f82), scale));
    max3(correction, splat3(0.0))
}

/// GGX/Trowbridge-Reitz normal distribution for isotropic roughness.
#[inline(always)]
#[must_use]
pub fn ggx_ndf(no_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = no_h * no_h * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

/// Anisotropic GGX normal distribution (Heitz 2014; OpenPBR spec):
/// `alpha_u`/`alpha_v` are the true roughness-squared axes (as returned by
/// [`openpbr_anisotropy`]), used once — not squared again — with the `NoH²`
/// term. This form integrates to 1 over the hemisphere; the previous spelling
/// dropped `NoH²` and scaled by `1/(αu²·αv²)`, inflating specular energy ~10⁴×
/// at roughness 0.1 and stamping hard terminator seams on smooth materials.
#[inline(always)]
#[must_use]
// qual:allow(srp, max_parameters=6) reason: "the two roughness axes are separate operands of the spec, next to the three vectors they apply to"
pub fn ggx_ndf_aniso(no_h: f32, h: Vec3, t: Vec3, b: Vec3, alpha_u: f32, alpha_v: f32) -> f32 {
    let hu = dot(h, t);
    let hv = dot(h, b);
    let denom = hu * hu / (alpha_u * alpha_u) + hv * hv / (alpha_v * alpha_v) + no_h * no_h;
    1.0 / (PI * alpha_u * alpha_v * denom * denom)
}

/// `alpha_u` of the anisotropic NDF: roughness squared, stretched along the
/// tangent by the anisotropy.
///
/// Split from [`openpbr_anisotropy`] because a shader-side caller needs the two
/// alphas as scalars: a `(f32, f32)` return is an aggregate in LLVM IR, and an
/// aggregate is a struct the translator would have to take apart again.
#[inline(always)]
#[must_use]
pub fn anisotropy_alpha_u(roughness: f32, anisotropy: f32) -> f32 {
    let r2 = roughness * roughness;
    let aniso_inv = 1.0 - anisotropy;
    let aniso_inv_sq = aniso_inv * aniso_inv;
    let denom = aniso_inv_sq + 1.0;
    let fraction = 2.0 / denom;
    let sqrt_frac = sqrt(fraction);
    r2 * sqrt_frac
}

/// `alpha_v` of the anisotropic NDF: `alpha_u` squeezed across the bitangent.
#[inline(always)]
#[must_use]
pub fn anisotropy_alpha_v(roughness: f32, anisotropy: f32) -> f32 {
    (1.0 - anisotropy) * anisotropy_alpha_u(roughness, anisotropy)
}

/// Map (roughness, anisotropy) to the `alpha_u`/`alpha_v` pair used by the
/// anisotropic NDF and visibility terms.
///
/// Host-side convenience over [`anisotropy_alpha_u`]/[`anisotropy_alpha_v`];
/// the multiplicities are the same, so the pair and the two scalars agree bit
/// for bit.
#[inline(always)]
#[must_use]
pub fn openpbr_anisotropy(roughness: f32, anisotropy: f32) -> (f32, f32) {
    (
        anisotropy_alpha_u(roughness, anisotropy),
        anisotropy_alpha_v(roughness, anisotropy),
    )
}

/// Smith height-correlated GGX visibility term (isotropic), the
/// `G / (4 NoV NoL)` factor already folded in.
#[inline(always)]
#[must_use]
pub fn smith_ggx_correlated(no_v: f32, no_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggx_v = no_v * sqrt((no_l * no_l * (1.0 - a2) + a2).max(EPS));
    let ggx_l = no_l * sqrt((no_v * no_v * (1.0 - a2) + a2).max(EPS));
    SMITH_GGX_VISIBILITY_SCALE / (ggx_v + ggx_l).max(EPS)
}

/// Height-correlated Smith GGX visibility for anisotropic distributions.
// qual:allow(srp, max_parameters=8) reason: "the two roughness axes are separate operands of the spec, next to the three vectors and the two cosines they are applied to"
#[inline(always)]
#[must_use]
// qual:allow(srp, max_parameters=8) reason: "the two roughness axes are separate operands of the spec, next to the three vectors and the two cosines"
pub fn smith_ggx_aniso(
    no_v: f32,
    no_l: f32,
    v: Vec3,
    l: Vec3,
    t: Vec3,
    b: Vec3,
    alpha_u: f32,
    alpha_v: f32,
) -> f32 {
    let vu = dot(v, t);
    let vv = dot(v, b);
    let lu = dot(l, t);
    let lv = dot(l, b);
    let a2u = alpha_u * alpha_u;
    let a2v = alpha_v * alpha_v;
    let ggx_v = no_v * sqrt((lu * lu * a2u + lv * lv * a2v + no_l * no_l).max(EPS));
    let ggx_l = no_l * sqrt((vu * vu * a2u + vv * vv * a2v + no_v * no_v).max(EPS));
    SMITH_GGX_VISIBILITY_SCALE / (ggx_v + ggx_l).max(EPS)
}

/// Oren–Nayar diffuse BRDF (`alpha` = roughness sigma), normalized by 1/π.
#[inline(always)]
#[must_use]
pub fn oren_nayar_brdf(no_v: f32, no_l: f32, cos_phi: f32, alpha: f32) -> f32 {
    // qual:allow(dry, fragment) reason: "the reference lobe in `ornis-render` computes the same roughness fit; the port is compared against it line by line"
    let sigma = alpha.max(EPS);
    let sigma2 = sigma * sigma;
    let a = 1.0 - OREN_NAYAR_A_NUMERATOR * sigma2 / (sigma2 + OREN_NAYAR_A_OFFSET);
    let b = OREN_NAYAR_B_NUMERATOR * sigma2 / (sigma2 + OREN_NAYAR_B_OFFSET);
    let theta_v = acos(no_v.max(0.0));
    let theta_l = acos(no_l.max(0.0));
    let alpha_max = theta_v.max(theta_l);
    let beta_min = theta_v.min(theta_l);
    let tan_beta = tan(beta_min);
    (a + b * cos_phi * sin(alpha_max) * tan_beta) * INV_PI
}

/// Fabric sheen lobe (Charlie-style D with approximate visibility).
#[inline(always)]
#[must_use]
pub fn sheen_brdf(no_v: f32, no_l: f32, no_h: f32, vo_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    // qual:allow(dry, fragment) reason: "the reference lobe in `ornis-render` factors D, G and F identically; the port is compared against it line by line"
    let d = alpha / (PI * pow(no_h * no_h * (alpha - 1.0) + 1.0, 2.0));
    let g = 1.0 / (1.0 + alpha * (1.0 / no_v + 1.0 / no_l - 2.0));
    let f = vo_h;
    d * g * f / (MICROFACET_SPECULAR_SCALE * no_v * no_l).max(EPS)
}

/// Diffuse energy left after the specular lobe.
///
/// The base albedo is already scaled by `(1 - metalness)` at the call site.
/// Multiplying by metalness again darkened partial metals by `(1 - metalness)²`
/// (a 0.5 metal kept a quarter of its diffuse).
#[inline(always)]
#[must_use]
pub fn base_diffuse_energy(specular_luma: f32) -> f32 {
    (1.0 - specular_luma).max(0.0)
}
