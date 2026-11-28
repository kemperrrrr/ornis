//! Transmission, subsurface and emission: the layers a light ray leaves
//! through rather than off.

use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::exp;
use ornis_gpu_shader::length3;

use super::{OFF, max_f32, mul3, scale3};
use crate::math::{
    EPS, anisotropy_alpha_u, coated_emission, fresnel_schlick, fresnel0_from_ior, ggx_ndf,
    smith_ggx_correlated, splat3, sub3, subsurface_brdf, transmission_color_to_extinction, v3,
};

/// Microfacet transmission BTDF with extinction.
// qual:allow(srp, max_parameters=9) reason: "the per-sample terms, the two IORs, the roughness axis, the extinction triple and the path length"
#[inline(always)]
#[must_use]
pub fn transmission_btdf(
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    ior_in: f32,
    ior_out: f32,
    alpha: f32,
    extinction: Vec3,
    distance: f32,
) -> Vec3 {
    let _ = voh;
    // The reference also derives the refracted cosine `cos_theta_t` from the
    // IOR ratio; nothing consumes it, and Snell's term survives in the
    // transmission weight the caller applies.
    let _ = ior_out;
    let f = fresnel_schlick(max_f32(nov, EPS), fresnel0_from_ior(ior_in));
    let t = 1.0 - f;
    // The NDF argument is the facet normal `N·H`, not `V·H` (those agree only
    // when the half-vector sits on the normal). `g` is already
    // V = G / (4 NoV NoL); the BTDF is D·T·V, not divided again.
    let d = ggx_ndf(noh, alpha);
    let g = smith_ggx_correlated(nov, nol, alpha);
    mul3(splat3(d * g * t), extinction_factor(extinction, distance))
}

/// Beer-Lambert attenuation of a path through an extincting medium.
#[inline(always)]
#[must_use]
pub fn extinction_factor(extinction: Vec3, distance: f32) -> Vec3 {
    v3(
        exp(-extinction.x * distance),
        exp(-extinction.y * distance),
        exp(-extinction.z * distance),
    )
}

/// Transmission layer with Beer-Lambert extinction and BTDF.
// qual:allow(srp, max_parameters=20) reason: "the per-sample terms plus every transmission field the reference takes, including the ones dispersion will use"
#[inline(always)]
#[must_use]
pub fn evaluate_transmission_layer(
    n: Vec3,
    v: Vec3,
    l: Vec3,
    h: Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    transmission_weight: f32,
    transmission_depth: f32,
    transmission_dispersion_scale: f32,
    transmission_dispersion_abbe: f32,
    transmission_color: Vec3,
    transmission_scatter: Vec3,
    transmission_scatter_anisotropy: f32,
    specular_ior: f32,
    specular_roughness: f32,
    specular_anisotropy: f32,
    thin_walled: f32,
) -> Vec3 {
    let _ = (n, v, l, h, transmission_dispersion_scale);
    let _ = (transmission_dispersion_abbe, transmission_scatter);
    let _ = (transmission_scatter_anisotropy, thin_walled);
    if transmission_weight <= OFF {
        return splat3(OFF);
    }
    let ior_out = 1.0;
    let ior_in = specular_ior;
    let alpha_u = anisotropy_alpha_u(specular_roughness, specular_anisotropy);
    let extinction = transmission_color_to_extinction(transmission_color, transmission_depth);
    let btdf = transmission_btdf(
        nov,
        nol,
        noh,
        voh,
        ior_in,
        ior_out,
        alpha_u,
        extinction,
        transmission_depth,
    );
    scale3(mul3(transmission_color, btdf), transmission_weight)
}

/// Subsurface layer with projected-distance falloff.
// qual:allow(srp, max_parameters=12) reason: "the per-sample terms, the three radius scales and the phase anisotropy the reference takes"
#[inline(always)]
#[must_use]
pub fn evaluate_subsurface_layer(
    n: Vec3,
    v: Vec3,
    l: Vec3,
    nov: f32,
    nol: f32,
    subsurface_weight: f32,
    subsurface_radius: f32,
    subsurface_radius_scale_r: f32,
    subsurface_radius_scale_g: f32,
    subsurface_radius_scale_b: f32,
    subsurface_scatter_anisotropy: f32,
    subsurface_color: Vec3,
) -> Vec3 {
    if subsurface_weight <= OFF {
        return splat3(OFF);
    }
    let radius = v3(
        subsurface_radius * subsurface_radius_scale_r,
        subsurface_radius * subsurface_radius_scale_g,
        subsurface_radius * subsurface_radius_scale_b,
    );
    let v_proj = sub3(v, scale3(n, nov));
    let l_proj = sub3(l, scale3(n, nol));
    let distance = length3(
        v_proj.x - l_proj.x,
        v_proj.y - l_proj.y,
        v_proj.z - l_proj.z,
    );
    let ss_brdf = subsurface_brdf(nov, nol, distance, radius, subsurface_scatter_anisotropy);
    scale3(mul3(subsurface_color, ss_brdf), subsurface_weight)
}

/// Emission with coat-weighted Fresnel modulation.
#[inline(always)]
#[must_use]
pub fn evaluate_emission(
    emission_luminance: f32,
    emission_color: Vec3,
    coat_weight: f32,
    coat_color: Vec3,
    nov: f32,
) -> Vec3 {
    coated_emission(
        emission_color,
        emission_luminance,
        coat_weight,
        coat_color,
        nov,
    )
}
