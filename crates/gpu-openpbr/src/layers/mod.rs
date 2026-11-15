//! The OpenPBR layer stack: one function per layer, called once per light.
//!
//! These are the shader-side twin of `ornis-render`'s `shaders::helpers`
//! evaluators, shared by the forward-PBR, deferred-lighting and textured
//! material passes so the three cannot drift. Each takes the per-sample terms
//! the caller already computed (`nov`, `nol`, `noh`, `voh`) plus the material
//! fields it needs, and returns the layer's contribution.
//!
//! Differences from the reference, all deliberate:
//!
//! - The `mat: OpenPBRMaterial` parameter is gone. The reference passes the
//!   whole material record and never reads it; here the caller has already
//!   loaded the fields off the record and hands over the values.
//! - Locals the reference computes but never uses (the `eta` and `cos_theta_t`
//!   of [`transmission_btdf`], the dispersion terms of the transmission layer)
//!   are not recomputed: a dead local in Rust is a warning, and the parameters
//!   stay for call symmetry with the reference.
//! - `mix` goes through the SDK lane by lane rather than through a
//!   `a + (b - a) * t` helper: WGSL `mix` is `a(1 - t) + b t`, and the two
//!   differ in the last bits.
//!
//! Rounding order matches the reference statement for statement; the layer
//! math underneath comes from [`crate::math`].

// Every layer takes the per-sample terms, the material fields it reads and the
// tangent frame. A struct would not shrink that: it would need layout lowering
// the translator does not need for an inlined helper, and the fields would be
// restated at every call site. The signatures stay flat like the WGSL calls
// they become.
#![allow(clippy::too_many_arguments)]

use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::abs;
use ornis_gpu_shader::dpdx;
use ornis_gpu_shader::dpdy;
use ornis_gpu_shader::length3;
use ornis_gpu_shader::mix;

use crate::math::v3;

/// Screen-space normal variance scale for geometric specular AA (Filament
/// `SIGMA2`).
pub const SPECULAR_AA_VARIANCE: f32 = 0.15;
/// Scale from normal variance to the GGX kernel (`2 * variance` in Filament).
pub const SPECULAR_AA_KERNEL_SCALE: f32 = 2.0;
/// Cap on the extra GGX kernel geometric specular AA may add (Filament
/// `KAPPA`).
pub const SPECULAR_AA_KERNEL_CAP: f32 = 0.2;
/// Magnitude past which an HDR channel is treated as non-finite and zeroed.
pub const HDR_FINITE_LIMIT: f32 = 1.0e20;
/// Pre-tonemap luma above which the heat view paints the pixel red.
pub const SHADING_DEBUG_HOT_LUMA: f32 = 1.0;
/// The `select` factor of a view that is not active.
const OFF: f32 = 0.0;
/// The `select` factor of the active view.
const ON: f32 = 1.0;

/// Deferred debug views, matching `ornis_render::renderer::ShadingDebug`.
pub mod debug_view {
    /// Shaded result.
    pub const BEAUTY: u32 = 0;
    /// Direct specular only.
    pub const DIRECT_SPECULAR: u32 = 1;
    /// Image-based specular only.
    pub const IBL_SPECULAR: u32 = 2;
    /// Diffuse only.
    pub const DIFFUSE: u32 = 3;
    /// Perceptual roughness as grey.
    pub const ROUGHNESS: u32 = 4;
    /// Metalness as grey.
    pub const METALLIC: u32 = 5;
    /// Shadow visibility as grey.
    pub const SHADOW: u32 = 6;
    /// Firefly heat map.
    pub const HEAT: u32 = 7;
}

/// `a * b`, lane by lane.
#[inline(always)]
fn mul3(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

/// `a + b`, lane by lane.
#[inline(always)]
fn add3(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

/// `a * s` for a scalar `s`.
#[inline(always)]
fn scale3(a: Vec3, s: f32) -> Vec3 {
    Vec3::new(a.x * s, a.y * s, a.z * s)
}

/// `mix(a, b, t)` lane by lane, through the SDK so the GPU sees one `mix`.
#[inline(always)]
fn mix3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    Vec3::new(mix(a.x, b.x, t), mix(a.y, b.y, t), mix(a.z, b.z, t))
}

/// `max(a, b)` for two scalars.
#[inline(always)]
fn max_f32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// WGSL `min(a, b)` for two scalars.
#[inline(always)]
fn min_f32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// WGSL `saturate(x)`: `clamp(x, 0, 1)`.
#[inline(always)]
fn saturate(x: f32) -> f32 {
    x.clamp(OFF, ON)
}

/// `1.0` when `condition` holds, `0.0` otherwise — the WGSL `select` factor.
#[inline(always)]
fn factor(condition: bool) -> f32 {
    if condition { ON } else { OFF }
}

/// Whether `value` is a number the composite may carry.
#[inline(always)]
// `v == v` is the NaN test the reference writes, and it lowers to the ordered
// compare the translator supports; `is_nan` would ask for a predicate the IR
// path does not cover.
#[allow(clippy::eq_op)]
fn finite(value: f32) -> bool {
    // The magnitude bound catches the infinities a division by a vanishing
    // denominator produces.
    value == value && abs(value) < HDR_FINITE_LIMIT
}

/// Screen-space derivative of a vector along x, lane by lane.
#[inline(always)]
fn dpdx3(v: Vec3) -> Vec3 {
    v3(dpdx(v.x), dpdx(v.y), dpdx(v.z))
}

/// Screen-space derivative of a vector along y, lane by lane.
#[inline(always)]
fn dpdy3(v: Vec3) -> Vec3 {
    v3(dpdy(v.x), dpdy(v.y), dpdy(v.z))
}

/// Lane `LANE` of `normalize(v)`.
#[inline(always)]
#[must_use]
pub fn normalize3(v: Vec3) -> Vec3 {
    scale3(v, 1.0 / length3(v.x, v.y, v.z))
}

/// `dot(a, a)` for a vector.
#[inline(always)]
fn dot3(a: Vec3, b: Vec3) -> f32 {
    ornis_gpu_shader::dot3(a.x, a.y, a.z, b.x, b.y, b.z)
}

/// The base and coat lobe family: geometric specular AA, HDR sanitation, the
/// debug views and the base/coat/fuzz evaluators.
// The whole module mirrors signatures the reference spells flat; the allow
// lives here once instead of on eleven functions.
#[allow(clippy::too_many_arguments)]
pub mod base;
/// The transmission and subsurface family.
#[allow(clippy::too_many_arguments)]
pub mod transmission;

pub use self::base::{
    evaluate_base_layer, evaluate_base_specular, evaluate_coat_darkening, evaluate_coat_layer,
    evaluate_fuzz_layer, sanitize_hdr, select_shading_debug, specular_aa_roughness,
};
pub use self::transmission::{
    evaluate_emission, evaluate_subsurface_layer, evaluate_transmission_layer, extinction_factor,
    transmission_btdf,
};

#[cfg(test)]
mod tests;
