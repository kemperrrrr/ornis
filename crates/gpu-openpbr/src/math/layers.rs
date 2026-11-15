//! The OpenPBR layer stack: coat, thin film, transmission, subsurface and
//! emission.

use super::{
    COLOR_CHANNEL_FLOOR, EPS, INV_PI, LAW_OF_COSINES_HALF, PI, SCHLICK_FRESNEL_POWER,
    SUBSURFACE_DIFFUSION, THIN_FILM_PHASE_CYCLES, THIN_FILM_WAVELENGTH_B_NM,
    THIN_FILM_WAVELENGTH_G_NM, THIN_FILM_WAVELENGTH_R_NM, add3, attenuate, div3, lerp3, max3, mul3,
    scale3, splat3, sub3,
};
use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::clamp;
use ornis_gpu_shader::cos;
use ornis_gpu_shader::ln;
use ornis_gpu_shader::pow;
use ornis_gpu_shader::sqrt;

/// Mix a thin-film modulation in by weight.
///
/// Weight 0 is the identity. The Airy kernel at thickness 0 is not 1 (a
/// zero-thickness film with the default water IOR still returns about 1.04),
/// and the evaluators used to multiply that into every material because
/// `thin_film_weight` was loaded and ignored.
#[inline(always)]
#[must_use]
pub fn thin_film_weight_mix(weight: f32, modulation: Vec3) -> Vec3 {
    lerp3(splat3(1.0), modulation, clamp(weight, 0.0, 1.0))
}

/// Emitter radiance, with an optional coat transmittance tint.
///
/// `emission_luminance` is luminance in nits (cd/m²), the same quantity as
/// outgoing radiance, so it is not divided by π (that factor converts
/// Lambertian exitance and made every emitter π times too dark). A full coat
/// transmits at normal incidence and blocks at grazing; the previous factor was
/// the Fresnel term itself, so a coated emitter went black head-on and lit up
/// on the silhouette.
#[inline(always)]
#[must_use]
pub fn coated_emission(
    emission_color: Vec3,
    emission_luminance: f32,
    coat_weight: f32,
    coat_color: Vec3,
    nov: f32,
) -> Vec3 {
    if emission_luminance <= 0.0 {
        return splat3(0.0);
    }
    let base_emission = scale3(emission_color, emission_luminance);
    let coat_fresnel = pow(1.0 - nov, SCHLICK_FRESNEL_POWER);
    let transmit = (1.0 - coat_fresnel) * coat_weight + (1.0 - coat_weight);
    let coat_emission = scale3(mul3(coat_color, base_emission), transmit);
    lerp3(base_emission, coat_emission, coat_weight)
}

/// Convert transmission tint + thickness into a Beer–Lambert extinction
/// coefficient `sigma_t = -ln(c) / depth` per channel.
#[inline(always)]
#[must_use]
pub fn transmission_color_to_extinction(transmission_color: Vec3, depth: f32) -> Vec3 {
    if depth <= 0.0 {
        return splat3(0.0);
    }
    let c = max3(transmission_color, splat3(COLOR_CHANNEL_FLOOR));
    Vec3::new(-ln(c.x) / depth, -ln(c.y) / depth, -ln(c.z) / depth)
}

/// Single-lobe subsurface approximation.
///
/// `distance` is the tangent-plane chord between the unit view and light
/// (`|V - N·NoV - (L - N·NoL)|`, range 0..2), not a world-space path length.
/// The dipole `1/r` pole is not defined on that chord: it evaluated to `1/EPS`
/// whenever the light sat near the view (a white firefly on the subsurface
/// preset). The phase uses the scattering cosine recovered from the same chord,
/// not `cos(distance)`.
#[inline(always)]
#[must_use]
pub fn subsurface_brdf(no_v: f32, no_l: f32, distance: f32, radius: Vec3, anisotropy: f32) -> Vec3 {
    let sigma_tr = mul3(
        splat3(sqrt(SUBSURFACE_DIFFUSION)),
        div3(splat3(1.0), max3(radius, splat3(EPS))),
    );
    // qual:allow(dry, fragment) reason: "the reference layer in `ornis-render` derives the same chord terms; the port is compared against it line by line"
    let v_len2 = (1.0 - no_v * no_v).max(0.0);
    let l_len2 = (1.0 - no_l * no_l).max(0.0);
    let tangent_dot = (v_len2 + l_len2 - distance * distance) * LAW_OF_COSINES_HALF;
    let cos_scatter = clamp(no_v * no_l + tangent_dot, -1.0, 1.0);
    let phase = (1.0 + anisotropy * cos_scatter).max(0.0);
    scale3(attenuate(sigma_tr, distance), phase * INV_PI)
}

/// Coat darkening, part 1: the base albedo darkening under a coated surface
/// (`Kcoat` attenuation of the underlying BRDF response).
// qual:allow(srp, max_parameters=7) reason: "the Kcoat fit takes the coat IOR plus the base layer's four weights; a struct would only rename them"
#[inline(always)]
#[must_use]
pub fn coat_base_darkening(
    coat_ior: f32,
    base_metalness: f32,
    base_color: Vec3,
    base_weight: f32,
    specular_weight: f32,
    subsurface_weight: f32,
    subsurface_color: Vec3,
) -> Vec3 {
    let coat_f0 = pow((coat_ior - 1.0) / (coat_ior + 1.0), 2.0);
    let one_minus_coat_f0 = 1.0 - coat_f0;
    let coat_ior_sq = coat_ior * coat_ior;
    let k_coat = 1.0 - one_minus_coat_f0 / coat_ior_sq;
    let e_metal = scale3(mul3(base_color, splat3(base_weight)), specular_weight);
    let e_dielectric = lerp3(subsurface_color, base_color, subsurface_weight);
    let e_base = lerp3(e_metal, e_dielectric, base_metalness);
    let e_base_k_coat = scale3(e_base, k_coat);
    let one_minus_e_base_k_coat = sub3(splat3(1.0), e_base_k_coat);
    div3(
        splat3(1.0 - k_coat),
        max3(one_minus_e_base_k_coat, splat3(COLOR_CHANNEL_FLOOR)),
    )
}

/// Coat darkening, part 2: identity when the coat is off, otherwise blend the
/// darkened albedo by `coat_weight * coat_dark`.
///
/// The product used to be added onto the coat lobe (`darkening * coat_albedo`),
/// which brightened the coat. Callers multiply the base lobe by this factor
/// instead.
#[inline(always)]
#[must_use]
pub fn coat_blend_darkened(base_darkening: Vec3, coat_weight: f32, coat_dark: f32) -> Vec3 {
    if coat_weight <= 0.0 {
        return splat3(1.0);
    }
    lerp3(splat3(1.0), base_darkening, coat_weight * coat_dark)
}

/// Thin-film interference modulation over white: phase per RGB wavelength from
/// film IOR/thickness (nanometers), Airy-like reflectance boost.
#[inline(always)]
#[must_use]
pub fn thin_film_modulation(
    cos_theta: f32,
    film_ior: f32,
    thickness_nm: f32,
    ior_outside: f32,
) -> Vec3 {
    let sin_theta_film = ior_outside * sqrt((1.0 - cos_theta * cos_theta).max(0.0)) / film_ior;
    let cos_theta_film = sqrt((1.0 - sin_theta_film * sin_theta_film).max(0.0));
    let r0 = pow((film_ior - ior_outside) / (film_ior + ior_outside), 2.0);
    let cycles = THIN_FILM_PHASE_CYCLES * PI * film_ior * thickness_nm * cos_theta_film;
    let phase = Vec3::new(
        cycles / THIN_FILM_WAVELENGTH_R_NM,
        cycles / THIN_FILM_WAVELENGTH_G_NM,
        cycles / THIN_FILM_WAVELENGTH_B_NM,
    );
    let oscillation = Vec3::new(cos(phase.x), cos(phase.y), cos(phase.z));
    add3(
        splat3(1.0),
        div3(mul3(splat3(2.0 * r0), oscillation), splat3(1.0 - r0 * r0)),
    )
}
