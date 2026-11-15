//! Light in, light out: luminance, tonemapping, the sRGB decode,
//! the octahedral normal encoding and the split-sum image-based light.

use super::{
    ACES_FIT_A, ACES_FIT_B, ACES_FIT_C, ACES_FIT_D, ACES_FIT_E, REC709_LUMA_B, REC709_LUMA_G,
    REC709_LUMA_R, SRGB_GAMMA, SRGB_LINEAR_CUTOFF, SRGB_LINEAR_SLOPE, SRGB_OFFSET, SRGB_SCALE,
    add3, dot, max3, mul3, scale3, splat3, sub3,
};
use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::abs;
use ornis_gpu_shader::pow;

/// Rec.709 relative luminance of a linear RGB color.
#[inline(always)]
#[must_use]
pub fn luminance(c: Vec3) -> f32 {
    dot(c, Vec3::new(REC709_LUMA_R, REC709_LUMA_G, REC709_LUMA_B))
}

/// ACES filmic tone-mapping curve (Narkowicz approximation), HDR → LDR.
#[inline(always)]
#[must_use]
// qual:test_helper — the Render pass that calls it moves over next; `openpbr_math_parity` is its only caller until then
pub fn aces_tonemap(color: Vec3) -> Vec3 {
    Vec3::new(
        aces_channel(color.x),
        aces_channel(color.y),
        aces_channel(color.z),
    )
}

/// One channel of [`aces_tonemap`].
#[inline(always)]
fn aces_channel(c: f32) -> f32 {
    let numerator = c * (ACES_FIT_A * c + ACES_FIT_B);
    let denominator = c * (ACES_FIT_C * c + ACES_FIT_D) + ACES_FIT_E;
    numerator / denominator
}

/// Specular half of [`evaluate_ibl`]: prefiltered radiance times the LUT
/// Fresnel.
#[inline(always)]
#[must_use]
pub fn evaluate_ibl_specular(
    prefiltered: Vec3,
    lut_scale: f32,
    lut_bias: f32,
    f0: Vec3,
    weight: f32,
) -> Vec3 {
    let fresnel = add3(scale3(f0, lut_scale), splat3(lut_bias));
    scale3(mul3(prefiltered, fresnel), weight)
}

/// Diffuse half of [`evaluate_ibl`]: irradiance times the diffuse color.
#[inline(always)]
#[must_use]
pub fn evaluate_ibl_diffuse(
    irradiance: Vec3,
    lut_bias: f32,
    diffuse_color: Vec3,
    weight: f32,
) -> Vec3 {
    let diffuse_weight = max3(sub3(splat3(1.0), splat3(lut_bias)), splat3(0.0));
    scale3(
        mul3(mul3(irradiance, diffuse_color), diffuse_weight),
        weight,
    )
}

/// Split-sum image-based light: prefiltered specular times the BRDF LUT
/// (scale/bias) plus irradiance times the diffuse color, gated by `weight`
/// (0 leaves the direct-light result unchanged).
#[inline(always)]
#[must_use]
// qual:allow(srp, max_parameters=7) reason: "the split-sum factorization takes each factor of the product; a context struct would only move the same seven values onto the caller"
pub fn evaluate_ibl(
    prefiltered: Vec3,
    irradiance: Vec3,
    lut_scale: f32,
    lut_bias: f32,
    f0: Vec3,
    diffuse_color: Vec3,
    weight: f32,
) -> Vec3 {
    let specular = evaluate_ibl_specular(prefiltered, lut_scale, lut_bias, f0, 1.0);
    let diffuse = evaluate_ibl_diffuse(irradiance, lut_bias, diffuse_color, 1.0);
    scale3(add3(specular, diffuse), weight)
}

/// Decode sRGB-encoded RGB to linear light (piecewise IEC 61966-2-1 OETF
/// inverse).
#[inline(always)]
#[must_use]
// qual:test_helper — the Render pass that calls it moves over next; `openpbr_math_parity` is its only caller until then
pub fn srgb_to_linear(c: Vec3) -> Vec3 {
    Vec3::new(srgb_channel(c.x), srgb_channel(c.y), srgb_channel(c.z))
}

/// Octahedral mapping of a unit vector to `[-1, 1]²` — compact normal storage.
#[inline(always)]
#[must_use]
// qual:test_helper — the Render pass that calls it moves over next; `openpbr_math_parity` is its only caller until then
pub fn octahedral_encode(n: Vec3) -> (f32, f32) {
    let sum = abs(n.x) + abs(n.y) + abs(n.z);
    let px = n.x / sum;
    let py = n.y / sum;
    let sign_x = if px >= 0.0 { 1.0 } else { -1.0 };
    let sign_y = if py >= 0.0 { 1.0 } else { -1.0 };
    let flipped_x = (1.0 - abs(py)) * sign_x;
    let flipped_y = (1.0 - abs(px)) * sign_y;
    if n.z < 0.0 {
        return (flipped_x, flipped_y);
    }
    (px, py)
}

/// One channel of [`srgb_to_linear`].
#[inline(always)]
fn srgb_channel(c: f32) -> f32 {
    if c <= SRGB_LINEAR_CUTOFF {
        c / SRGB_LINEAR_SLOPE
    } else {
        pow((c + SRGB_OFFSET) / SRGB_SCALE, SRGB_GAMMA)
    }
}
