//! OpenPBR shading math: the shader-side twin of `ornis-render`'s
//! `shaders::math` module.
//!
//! Each function takes and returns [`Vec3`] lanes the way the reference takes
//! `glam::Vec3`. `.max(vec)`/`.lerp()` and the componentwise operators become
//! lane-explicit helpers ([`max3`], [`lerp3`], [`mul3`], ...) because the
//! translator must see three scalar operations, not an aggregate.
//!
//! Two functions return a pair of lanes instead of a `Vec2`; the SDK has no
//! value type for it and a tuple costs nothing in LLVM IR.
//!
//! - [`brdf`] is the lobe math: Fresnel, the GGX distribution and visibility,
//!   and the diffuse and fabric lobes.
//! - [`color`] is light in, light out: luminance, tonemapping, the sRGB decode,
//!   the octahedral normal encoding and the split-sum image-based light.
//! - [`layers`] is the OpenPBR layer stack: coat, thin film, transmission,
//!   subsurface and emission.

use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::dot3;
use ornis_gpu_shader::length3;

pub mod brdf;
pub mod color;
pub mod layers;

pub use brdf::*;
pub use color::*;
pub use layers::*;

/// π.
pub const PI: f32 = core::f32::consts::PI;
/// `1/π`, used to normalize cosine-weighted BRDF integrals.
pub const INV_PI: f32 = 1.0 / PI;
/// Small epsilon guarding divisions and sqrt arguments in the kernels.
pub const EPS: f32 = 1e-6;

/// Rec.709 luma weight for linear red.
pub const REC709_LUMA_R: f32 = 0.2126;
/// Rec.709 luma weight for linear green.
pub const REC709_LUMA_G: f32 = 0.7152;
/// Rec.709 luma weight for linear blue.
pub const REC709_LUMA_B: f32 = 0.0722;

/// Narkowicz ACES filmic-fit coefficient `a` (the quadratic numerator).
pub const ACES_FIT_A: f32 = 2.51;
/// Narkowicz ACES filmic-fit coefficient `b` (the linear numerator).
pub const ACES_FIT_B: f32 = 0.03;
/// Narkowicz ACES filmic-fit coefficient `c` (the quadratic denominator).
pub const ACES_FIT_C: f32 = 2.43;
/// Narkowicz ACES filmic-fit coefficient `d` (the linear denominator).
pub const ACES_FIT_D: f32 = 0.59;
/// Narkowicz ACES filmic-fit coefficient `e` (the constant denominator).
pub const ACES_FIT_E: f32 = 0.14;

/// Exponent of the Schlick Fresnel term, `(1 - cos θ)^5`.
pub const SCHLICK_FRESNEL_POWER: f32 = 5.0;
/// Cosine at which the F82-tint Fresnel correction is pinned (`1/7`, about 82°).
pub const F82_PIVOT_COSINE: f32 = 1.0 / 7.0;
/// Exponent of the F82 correction lobe, `cos θ · (1 - cos θ)^6`.
pub const F82_CORRECTION_POWER: f32 = 6.0;

/// Leading scale of the height-correlated Smith GGX visibility term.
pub const SMITH_GGX_VISIBILITY_SCALE: f32 = 0.5;
/// Numerator of the Oren–Nayar `A` roughness fit (`1 - 0.5 σ² / (σ² + 0.57)`).
pub const OREN_NAYAR_A_NUMERATOR: f32 = 0.5;
/// Offset inside the Oren–Nayar `A` roughness fit.
pub const OREN_NAYAR_A_OFFSET: f32 = 0.57;
/// Numerator of the Oren–Nayar `B` roughness fit.
pub const OREN_NAYAR_B_NUMERATOR: f32 = 0.45;
/// Offset inside the Oren–Nayar `B` roughness fit.
pub const OREN_NAYAR_B_OFFSET: f32 = 0.09;

/// Denominator scale of a microfacet specular lobe (`4 NoV NoL`).
pub const MICROFACET_SPECULAR_SCALE: f32 = 4.0;
/// Floor on a color channel before a logarithm or a reciprocal.
pub const COLOR_CHANNEL_FLOOR: f32 = 1e-6;
/// Factor under the square root of the subsurface diffusion profile (`√3 / radius`).
pub const SUBSURFACE_DIFFUSION: f32 = 3.0;
/// Half in the law of cosines that recovers the tangent-plane view·light dot
/// from the chord `distance` (`(‖V‖² + ‖L‖² − d²) / 2`).
pub const LAW_OF_COSINES_HALF: f32 = 0.5;

/// IEC 61966-2-1 cutoff: sRGB channels at or below this decode linearly.
pub const SRGB_LINEAR_CUTOFF: f32 = 0.04045;
/// Slope of the linear segment of the sRGB inverse transfer (`c / 12.92`).
pub const SRGB_LINEAR_SLOPE: f32 = 12.92;
/// Offset inside the sRGB power segment (`(c + 0.055) / 1.055`).
pub const SRGB_OFFSET: f32 = 0.055;
/// Scale of the sRGB power segment.
pub const SRGB_SCALE: f32 = 1.055;
/// Exponent of the sRGB power segment.
pub const SRGB_GAMMA: f32 = 2.4;

/// Representative red wavelength (nanometers) of the thin-film modulation.
pub const THIN_FILM_WAVELENGTH_R_NM: f32 = 650.0;
/// Representative green wavelength (nanometers) of the thin-film modulation.
pub const THIN_FILM_WAVELENGTH_G_NM: f32 = 550.0;
/// Representative blue wavelength (nanometers) of the thin-film modulation.
pub const THIN_FILM_WAVELENGTH_B_NM: f32 = 450.0;
/// Optical-path cycles in the thin-film phase (`4 π n d cos / λ`).
pub const THIN_FILM_PHASE_CYCLES: f32 = 4.0;

/// Build a vector from its lanes.
#[inline(always)]
#[must_use]
pub fn v3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// `a + b`, lane by lane.
#[inline(always)]
#[must_use]
pub fn add3(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

/// `a - b`, lane by lane.
#[inline(always)]
#[must_use]
pub fn sub3(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

/// `a * b`, lane by lane.
#[inline(always)]
#[must_use]
pub fn mul3(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

/// `a / b`, lane by lane.
#[inline(always)]
#[must_use]
pub fn div3(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x / b.x, a.y / b.y, a.z / b.z)
}

/// `a * s` for a scalar `s`.
#[inline(always)]
#[must_use]
pub fn scale3(a: Vec3, s: f32) -> Vec3 {
    Vec3::new(a.x * s, a.y * s, a.z * s)
}

/// Componentwise maximum.
#[inline(always)]
#[must_use]
pub fn max3(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}

/// Componentwise linear interpolation, `a + (b - a) * factor`.
#[inline(always)]
#[must_use]
pub fn lerp3(a: Vec3, b: Vec3, factor: f32) -> Vec3 {
    Vec3::new(
        a.x + (b.x - a.x) * factor,
        a.y + (b.y - a.y) * factor,
        a.z + (b.z - a.z) * factor,
    )
}

/// `(s, s, s)`, the WGSL `vec3<f32>(s)` broadcast.
#[inline(always)]
#[must_use]
pub fn splat3(s: f32) -> Vec3 {
    Vec3::new(s, s, s)
}

/// Dot product of two vectors, through the order-sensitive SDK operation.
#[inline(always)]
#[must_use]
pub fn dot(a: Vec3, b: Vec3) -> f32 {
    dot3(a.x, a.y, a.z, b.x, b.y, b.z)
}

/// Euclidean length of a vector, through the order-sensitive SDK operation.
#[inline(always)]
#[must_use]
pub fn length(v: Vec3) -> f32 {
    length3(v.x, v.y, v.z)
}

/// `exp(-distance * scale)` per channel — the Beer–Lambert profile of
/// [`layers::subsurface_brdf`].
#[inline(always)]
#[must_use]
pub fn attenuate(scale: Vec3, distance: f32) -> Vec3 {
    Vec3::new(
        ornis_gpu_shader::exp(-distance * scale.x),
        ornis_gpu_shader::exp(-distance * scale.y),
        ornis_gpu_shader::exp(-distance * scale.z),
    )
}
