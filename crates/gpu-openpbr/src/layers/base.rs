//! Base, coat and fuzz layers, plus the per-pixel screen-space terms.

use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::sqrt;

use super::{
    OFF, add3, debug_view, dot3, dpdx3, dpdy3, factor, finite, max_f32, min_f32, mix3, mul3,
    normalize3, saturate, scale3,
};
use crate::layers::{
    SHADING_DEBUG_HOT_LUMA, SPECULAR_AA_KERNEL_CAP, SPECULAR_AA_KERNEL_SCALE, SPECULAR_AA_VARIANCE,
};
use crate::math::{
    anisotropy_alpha_u, anisotropy_alpha_v, base_diffuse_energy, coat_base_darkening,
    coat_blend_darkened, fresnel_f82_tint, fresnel_schlick_vec, fresnel0_from_ior, ggx_ndf_aniso,
    luminance, oren_nayar_brdf, sheen_brdf, smith_ggx_aniso, splat3, sub3, v3,
};

/// Geometric specular AA (Kaplanyan / Filament): widen perceptual roughness by
/// the screen-space variance of `n`, then return a perceptual roughness the
/// rest of the BRDF squares again.
///
/// Derivatives must run in uniform control flow, so callers invoke this before
/// any `discard`.
#[inline(always)]
#[must_use]
pub fn specular_aa_roughness(roughness: f32, n: Vec3) -> f32 {
    let du = dpdx3(n);
    let dv = dpdy3(n);
    let variance = SPECULAR_AA_VARIANCE * (dot3(du, du) + dot3(dv, dv));
    let kernel = min_f32(SPECULAR_AA_KERNEL_SCALE * variance, SPECULAR_AA_KERNEL_CAP);
    let alpha = roughness * roughness;
    let square = saturate(alpha * alpha + kernel);
    sqrt(sqrt(max_f32(square, OFF)))
}

/// Zero the non-finite channels of an HDR color. Finite highlights stay
/// untouched so ACES in the composite can compress them; NaN/Inf would
/// otherwise resolve to white.
///
/// The channels are *selected* away, not masked by a factor: `NaN * 0` and
/// `Inf * 0` are both NaN, so a `color * ok` form leaves the poisoned channel
/// poisoned and also drops the finite channels of the same pixel.
#[inline(always)]
#[must_use]
pub fn sanitize_hdr(color: Vec3) -> Vec3 {
    v3(
        keep_finite(color.x),
        keep_finite(color.y),
        keep_finite(color.z),
    )
}

/// `value` when it is a number the composite may carry, `0.0` otherwise.
#[inline(always)]
fn keep_finite(value: f32) -> f32 {
    if finite(value) { value } else { OFF }
}

/// Pick a deferred debug term. `view == 0` returns `beauty` unchanged.
/// `raw` is the pre-sanitize color used by the heat view.
// qual:allow(srp, max_parameters=9) reason: "one value per debug view, in the order the views are numbered; a struct would only move the same nine values onto every caller"
#[inline(always)]
#[must_use]
pub fn select_shading_debug(
    view: u32,
    beauty: Vec3,
    direct_spec: Vec3,
    ibl_spec: Vec3,
    diffuse: Vec3,
    roughness: f32,
    metalness: f32,
    shadow: f32,
    raw: Vec3,
) -> Vec3 {
    let is_beauty = factor(view == debug_view::BEAUTY);
    let is_direct = factor(view == debug_view::DIRECT_SPECULAR);
    let is_ibl = factor(view == debug_view::IBL_SPECULAR);
    let is_diffuse = factor(view == debug_view::DIFFUSE);
    let is_rough = factor(view == debug_view::ROUGHNESS);
    let is_metal = factor(view == debug_view::METALLIC);
    let is_shadow = factor(view == debug_view::SHADOW);
    let is_heat = factor(view == debug_view::HEAT);
    let x_bad = factor(!finite(raw.x));
    let y_bad = factor(!finite(raw.y));
    let z_bad = factor(!finite(raw.z));
    let bad = max_f32(x_bad, max_f32(y_bad, z_bad));
    let hot = factor(luminance(beauty) > SHADING_DEBUG_HOT_LUMA);
    let heat = mix3(
        mix3(beauty, v3(SHADING_DEBUG_HOT_LUMA, OFF, OFF), hot),
        v3(SHADING_DEBUG_HOT_LUMA, OFF, SHADING_DEBUG_HOT_LUMA),
        bad,
    );
    let mut sum = scale3(beauty, is_beauty);
    sum = add3(sum, scale3(direct_spec, is_direct));
    sum = add3(sum, scale3(ibl_spec, is_ibl));
    sum = add3(sum, scale3(diffuse, is_diffuse));
    sum = add3(sum, scale3(splat3(roughness), is_rough));
    sum = add3(sum, scale3(splat3(metalness), is_metal));
    sum = add3(sum, scale3(splat3(shadow), is_shadow));
    add3(sum, scale3(heat, is_heat))
}

/// Base specular lobe `F · D · V`, with Smith `V` already including
/// `1 / (4 NoV NoL)`. Fresnel is applied once here; the diffuse energy term
/// uses the same `F`.
// qual:allow(srp, max_parameters=18) reason: "the per-sample terms, the base fields and the tangent frame; the reference takes the same eighteen"
#[inline(always)]
#[must_use]
pub fn evaluate_base_specular(
    v: Vec3,
    l: Vec3,
    h: Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    base_weight: f32,
    base_color: Vec3,
    metalness: f32,
    specular_weight: f32,
    specular_roughness: f32,
    specular_ior: f32,
    specular_anisotropy: f32,
    specular_edge_tint: Vec3,
    t: Vec3,
    b: Vec3,
    thin_film_mod: Vec3,
) -> Vec3 {
    let f0_dielectric = splat3(fresnel0_from_ior(specular_ior));
    let f0_metal = scale3(base_color, base_weight);
    let f0 = mix3(f0_dielectric, f0_metal, metalness);
    let f = fresnel_f82_tint(voh, f0, specular_edge_tint);
    // qual:allow(dry, fragment) reason: "the coat lobe is the base lobe with its own roughness; sharing it needs a two-value return, and an aggregate return is a struct the translator would have to take apart again"
    let alpha_u = anisotropy_alpha_u(specular_roughness, specular_anisotropy);
    let alpha_v = anisotropy_alpha_v(specular_roughness, specular_anisotropy);
    let d = ggx_ndf_aniso(noh, h, t, b, alpha_u, alpha_v);
    // Smith returns V = G / (4 NoV NoL). Do not divide by 4·NoV·NoL again:
    // the light loop already multiplies by NoL, and a second divide blows up at
    // grazing angles.
    let g = smith_ggx_aniso(nov, nol, v, l, t, b, alpha_u, alpha_v);
    let ks = scale3(f, specular_weight);
    let lobe = scale3(scale3(ks, d), g);
    mul3(scale3(lobe, base_weight), thin_film_mod)
}

/// Base layer: dielectric/metallic mix with anisotropic GGX + Oren-Nayar.
// qual:allow(srp, max_parameters=20) reason: "the layer takes the per-sample terms, the base fields and the tangent frame; the reference takes the same twenty"
#[inline(always)]
#[must_use]
pub fn evaluate_base_layer(
    n: Vec3,
    v: Vec3,
    l: Vec3,
    h: Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    base_weight: f32,
    base_color: Vec3,
    metalness: f32,
    diffuse_roughness: f32,
    specular_weight: f32,
    specular_roughness: f32,
    specular_ior: f32,
    specular_anisotropy: f32,
    specular_edge_tint: Vec3,
    t: Vec3,
    b: Vec3,
    thin_film_mod: Vec3,
) -> Vec3 {
    let f0_dielectric = splat3(fresnel0_from_ior(specular_ior));
    let f0_metal = scale3(base_color, base_weight);
    let f0 = mix3(f0_dielectric, f0_metal, metalness);
    let f = fresnel_f82_tint(voh, f0, specular_edge_tint);
    // The specular lobe applies this Fresnel once (`ks` inside
    // [`evaluate_base_specular`]); multiplying it again here squared a
    // dielectric F0 of ~0.04 down to ~0.0016.
    let spec = evaluate_base_specular(
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
    );
    let diffuse_color = scale3(base_color, 1.0 - metalness);
    let diff_roughness = max_f32(diffuse_roughness, specular_roughness);
    let diff_alpha = diff_roughness * diff_roughness;
    let cos_phi = max_f32(
        dot3(
            normalize3(sub3(v, scale3(n, nov))),
            normalize3(sub3(l, scale3(n, nol))),
        ),
        OFF,
    );
    let diff_brdf = oren_nayar_brdf(nov, nol, cos_phi, diff_alpha);
    let ks = scale3(f, specular_weight);
    let kd = splat3(base_diffuse_energy(luminance(ks)));
    let diff = mul3(
        scale3(mul3(scale3(kd, diff_brdf), diffuse_color), base_weight),
        thin_film_mod,
    );
    add3(spec, diff)
}

/// Coat specular lobe. Base darkening is [`evaluate_coat_darkening`], applied
/// to the base lobe by the caller — adding it here used to brighten the coat
/// instead of attenuating the base.
// qual:allow(srp, max_parameters=22) reason: "coat fields plus the base fields the darkening needs plus the tangent frame; the reference takes the same"
#[inline(always)]
#[must_use]
pub fn evaluate_coat_layer(
    n: Vec3,
    v: Vec3,
    l: Vec3,
    h: Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    coat_weight: f32,
    coat_roughness: f32,
    coat_anisotropy: f32,
    coat_dark: f32,
    coat_ior: f32,
    coat_color: Vec3,
    base_metalness: f32,
    base_color: Vec3,
    base_weight: f32,
    specular_weight: f32,
    subsurface_weight: f32,
    subsurface_color: Vec3,
    t: Vec3,
    b: Vec3,
) -> Vec3 {
    let _ = (
        n,
        coat_dark,
        base_metalness,
        base_color,
        base_weight,
        specular_weight,
        subsurface_weight,
        subsurface_color,
    );
    if coat_weight <= OFF {
        return splat3(OFF);
    }
    let coat_f0 = splat3(fresnel0_from_ior(coat_ior));
    let coat_f = fresnel_schlick_vec(voh, coat_f0);
    // qual:allow(dry, fragment) reason: "see the base lobe: same anisotropic facet terms, different roughness input, and a shared helper would return two values"
    let coat_alpha_u = anisotropy_alpha_u(coat_roughness, coat_anisotropy);
    let coat_alpha_v = anisotropy_alpha_v(coat_roughness, coat_anisotropy);
    // Same visibility contract as the base lobe: `coat_g` is already V.
    let coat_d = ggx_ndf_aniso(noh, h, t, b, coat_alpha_u, coat_alpha_v);
    let coat_g = smith_ggx_aniso(nov, nol, v, l, t, b, coat_alpha_u, coat_alpha_v);
    let coat_brdf = scale3(coat_f, coat_d * coat_g);
    scale3(mul3(coat_color, coat_brdf), coat_weight)
}

/// Multiplier applied to the base lobe under a coat (1 when the coat is off).
// qual:allow(srp, max_parameters=9) reason: "the coat weight and darkening plus the base fields the Kcoat fit reads"
#[inline(always)]
#[must_use]
pub fn evaluate_coat_darkening(
    coat_weight: f32,
    coat_dark: f32,
    coat_ior: f32,
    base_metalness: f32,
    base_color: Vec3,
    base_weight: f32,
    specular_weight: f32,
    subsurface_weight: f32,
    subsurface_color: Vec3,
) -> Vec3 {
    coat_blend_darkened(
        coat_base_darkening(
            coat_ior,
            base_metalness,
            base_color,
            base_weight,
            specular_weight,
            subsurface_weight,
            subsurface_color,
        ),
        coat_weight,
        coat_dark,
    )
}

/// Fuzz (sheen) layer.
// qual:allow(srp, max_parameters=11) reason: "the per-sample terms plus the three fuzz fields the reference takes"
#[inline(always)]
#[must_use]
pub fn evaluate_fuzz_layer(
    n: Vec3,
    v: Vec3,
    l: Vec3,
    h: Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    fuzz_weight: f32,
    fuzz_roughness: f32,
    fuzz_color: Vec3,
) -> Vec3 {
    let _ = (n, v, l, h);
    if fuzz_weight <= OFF {
        return splat3(OFF);
    }
    let sheen = sheen_brdf(nov, nol, noh, voh, fuzz_roughness);
    scale3(scale3(fuzz_color, sheen), fuzz_weight)
}
