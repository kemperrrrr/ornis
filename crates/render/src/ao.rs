//! Screen-space ambient occlusion: classic SSAO plus GTAO (#124).
//!
//! Contact shadowing in corners and joints: without it the IBL/ambient term
//! looks flat. Both methods run as fullscreen passes over the g-buffer depth
//! and octahedral normals at half resolution, followed by a depth-aware
//! (bilateral) blur; the blurred factor modulates only the ambient and IBL
//! terms of the deferred lighting pass, never the direct lights.
//!
//! Two methods exist because they cover different platforms: GTAO (horizon
//! search per slice) is the quality default on desktop GPUs, while the
//! classic SSAO hemisphere kernel is the cheap fallback for weak adapters
//! (mobile / software rasterizers) where the GTAO slice integration costs
//! too much ALU per pixel. The CPU helpers below pin the response-curve
//! shape both GPU kernels must follow (flat = 1, corner < 1, range [0, 1]).

use ornis_core::units::{Clamped01, PositiveF32};

/// Half-resolution divisor of the AO targets (matches the bloom chain
/// convention: [`crate::transient_pool::SizePolicy::Fraction`]).
pub const AO_DIVISOR: u32 = 2;

/// Format of the pooled AO targets: filterable 8-bit so the lighting pass
/// can bilinearly upsample the half-res factor.
pub const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Default SSAO/GTAO sample radius in meters.
pub const DEFAULT_AO_RADIUS: f32 = 0.5;
/// Default occlusion strength (full effect).
pub const DEFAULT_AO_INTENSITY: f32 = 1.0;
/// Default depth bias in meters (hides acne on flat walls).
pub const DEFAULT_AO_BIAS: f32 = 0.025;
/// Depth sigma of the bilateral blur, in meters.
pub const DEFAULT_AO_BLUR_SIGMA: f32 = 0.1;

/// Which occlusion kernel the raw AO pass runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AoMethod {
    /// Horizon-search occlusion (quality default on desktop GPUs).
    #[default]
    Gtao,
    /// Classic hemisphere sampling (cheap fallback for weak adapters).
    Ssao,
}

impl AoMethod {
    /// `true` for the classic hemisphere kernel.
    pub fn is_ssao(self) -> bool {
        matches!(self, Self::Ssao)
    }
}

/// Validated AO settings (always the enabled state; disabled AO is the
/// absence of [`AoWiring`] on the plan, which leaves the frame
/// pixel-identical).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AoSettings {
    method: AoMethod,
    radius: PositiveF32,
    intensity: Clamped01,
    bias: f32,
}

impl AoSettings {
    /// Checked constructor: `None` for non-finite/non-positive radius,
    /// non-finite intensity, or negative/non-finite bias.
    pub fn new(method: AoMethod, radius: f32, intensity: f32, bias: f32) -> Option<Self> {
        if !(bias.is_finite() && bias >= 0.0) || !intensity.is_finite() {
            return None;
        }
        Some(Self {
            method,
            radius: PositiveF32::try_new(radius)?,
            intensity: Clamped01::new(intensity),
            bias,
        })
    }

    /// Kernel selected by these settings.
    pub const fn method(self) -> AoMethod {
        self.method
    }

    /// Sample radius in meters (always positive).
    pub const fn radius(self) -> f32 {
        self.radius.get()
    }

    /// Occlusion strength in `[0, 1]`.
    pub const fn intensity(self) -> f32 {
        self.intensity.get()
    }

    /// Depth bias in meters (`>= 0`).
    pub const fn bias(self) -> f32 {
        self.bias
    }
}

impl Default for AoSettings {
    /// GTAO at half-meter radius, full strength (the recommended default).
    fn default() -> Self {
        Self {
            method: AoMethod::Gtao,
            radius: PositiveF32::expect_valid(DEFAULT_AO_RADIUS),
            intensity: Clamped01::new(DEFAULT_AO_INTENSITY),
            bias: DEFAULT_AO_BIAS,
        }
    }
}

/// Opt-in AO wiring for [`RenderFrame3D`](crate::frame_exec::RenderFrame3D):
/// the kernel plus its settings in one value. Dropping it registers
/// nothing — default plans stay AO-free and pixel-identical.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AoWiring {
    settings: AoSettings,
}

impl AoWiring {
    /// Wiring value from validated [`AoSettings`].
    #[must_use]
    pub fn new(settings: AoSettings) -> Self {
        Self { settings }
    }

    /// Kernel settings carried by the wiring.
    pub const fn settings(self) -> AoSettings {
        self.settings
    }
}

/// Classic SSAO response of one depth tap: view-space depths in meters,
/// `sample` nearer than `center - bias` occludes, linearly falling to zero
/// at one `radius` of separation. Output in `[0, 1]`.
pub fn ssao_sample(center: f32, sample: f32, radius: f32, bias: f32) -> f32 {
    if !center.is_finite() || !sample.is_finite() || radius <= 0.0 {
        return 0.0;
    }
    let separation = center - sample - bias;
    (separation / radius).clamp(0.0, 1.0)
}

/// GTAO response of one horizon elevation: `elevation` is the cosine of the
/// highest sample above the tangent plane in `[0, 1]` (0 = flat horizon).
/// Output in `[0, 1]`.
pub fn gtao_sample(elevation: f32) -> f32 {
    elevation.clamp(0.0, 1.0)
}

/// Occlusion factor from accumulated taps: `occluded` is the summed
/// response, `taps` the tap count. `1.0` = unoccluded.
pub fn ao_from_sum(occluded: f32, taps: u32, intensity: f32) -> f32 {
    if taps == 0 {
        return 1.0;
    }
    (1.0 - (occluded / taps as f32) * intensity).clamp(0.0, 1.0)
}

/// Applies the AO factor to an ambient/IBL term: `ao` of `1.0` is the exact
/// identity, so the disabled path (white AO texture) cannot drift a pixel.
pub fn apply_ao_to_ambient(ambient: [f32; 3], ao: f32) -> [f32; 3] {
    [ambient[0] * ao, ambient[1] * ao, ambient[2] * ao]
}

/// Half-resolution extent of a surface (floored, clamped to 1).
pub fn half_extent(width: u32, height: u32) -> (u32, u32) {
    ((width / AO_DIVISOR).max(1), (height / AO_DIVISOR).max(1))
}

/// Bilateral depth weight in `[0, 1]`: `1.0` on the same surface, decaying
/// across depth steps so the blur never bleeds occlusion over edges.
pub fn bilateral_weight(center_depth: f32, sample_depth: f32, sigma: f32) -> f32 {
    if !(center_depth.is_finite() && sample_depth.is_finite()) || sigma <= 0.0 {
        return 0.0;
    }
    let d = (center_depth - sample_depth) / sigma;
    (-d * d).exp().clamp(0.0, 1.0)
}

/// Bilateral blur of one AO texel: depth-weighted mean of the center plus
/// its neighbors. A uniform field returns itself exactly.
pub fn bilateral_blur_center(
    center_ao: f32,
    center_depth: f32,
    taps: &[(f32, f32)],
    sigma: f32,
) -> f32 {
    let mut acc = center_ao;
    let mut weight = 1.0;
    for (ao, depth) in taps {
        let w = bilateral_weight(center_depth, *depth, sigma);
        acc += ao * w;
        weight += w;
    }
    acc / weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_ao_is_identity() {
        // The disabled path binds a white texture: factor 1 must be exact.
        let ambient = [0.25, 0.5, 1.0];
        assert_eq!(apply_ao_to_ambient(ambient, 1.0), ambient);
        assert_eq!(ao_from_sum(0.0, 8, 1.0), 1.0);
        assert_eq!(ao_from_sum(3.0, 0, 1.0), 1.0);
    }

    #[test]
    fn flat_wall_has_no_occlusion() {
        // Every tap at the wall depth: both kernels stay at 1.
        let mut occluded = 0.0;
        for _ in 0..8 {
            occluded += ssao_sample(2.0, 2.0, 0.5, 0.025);
        }
        assert_eq!(occluded, 0.0);
        assert_eq!(ao_from_sum(occluded, 8, 1.0), 1.0);
        assert_eq!(gtao_sample(0.0), 0.0);
        assert_eq!(ao_from_sum(gtao_sample(0.0), 1, 1.0), 1.0);
    }

    #[test]
    fn corner_occludes_in_both_kernels() {
        // Half the taps on a nearer face (the "cube in the corner" shape):
        // both kernels must darken.
        let taps = [2.0, 2.0, 1.6, 1.6, 2.0, 1.6, 2.0, 1.6];
        let mut occluded = 0.0;
        for sample in taps {
            occluded += ssao_sample(2.0, sample, 0.5, 0.025);
        }
        let ssao = ao_from_sum(occluded, taps.len() as u32, 1.0);
        assert!(ssao < 1.0 && ssao > 0.0, "ssao {ssao}");
        let gtao = ao_from_sum(gtao_sample(0.6), 1, 1.0);
        assert!(gtao < 1.0 && gtao > 0.0, "gtao {gtao}");
    }

    #[test]
    fn occlusion_stays_in_unit_range() {
        // Sweep separations, elevations, intensities: never out of [0, 1].
        for center in [0.5, 1.0, 5.0] {
            for sample in [0.0, 0.4, 1.0, 5.0, 100.0] {
                let tap = ssao_sample(center, sample, 0.5, 0.025);
                assert!((0.0..=1.0).contains(&tap), "tap {tap}");
                let ao = ao_from_sum(tap * 8.0, 8, 1.0);
                assert!((0.0..=1.0).contains(&ao), "ao {ao}");
            }
        }
        for elevation in [-1.0, 0.0, 0.5, 1.0, 2.0] {
            assert!((0.0..=1.0).contains(&gtao_sample(elevation)));
        }
        // Degenerate inputs occlude nothing instead of NaN.
        assert_eq!(ssao_sample(f32::NAN, 1.0, 0.5, 0.025), 0.0);
        assert_eq!(ssao_sample(1.0, 1.0, 0.0, 0.025), 0.0);
        assert_eq!(bilateral_weight(1.0, 1.0, 0.0), 0.0);
    }

    #[test]
    fn half_extent_floors_and_clamps() {
        assert_eq!(half_extent(1280, 720), (640, 360));
        assert_eq!(half_extent(1, 1), (1, 1));
        assert_eq!(half_extent(3, 5), (1, 2));
    }

    #[test]
    fn bilateral_blur_preserves_uniform_fields() {
        // Uniform AO + depth: blur is the identity (no drift on flat walls).
        let taps = [(0.7, 2.0), (0.7, 2.0), (0.7, 2.0), (0.7, 2.0)];
        assert_eq!(bilateral_blur_center(0.7, 2.0, &taps, 0.1), 0.7);
        // Across a depth step the far tap barely counts.
        let near = bilateral_weight(2.0, 2.0, 0.1);
        let far = bilateral_weight(2.0, 5.0, 0.1);
        assert_eq!(near, 1.0);
        assert!(far < 1.0e-3, "far {far}");
    }

    #[test]
    fn settings_reject_degenerate_input() {
        assert!(AoSettings::new(AoMethod::Gtao, 0.5, 1.0, 0.025).is_some());
        assert!(AoSettings::new(AoMethod::Ssao, 0.5, 0.0, 0.0).is_some());
        assert!(AoSettings::new(AoMethod::Gtao, 0.0, 1.0, 0.025).is_none());
        assert!(AoSettings::new(AoMethod::Gtao, -1.0, 1.0, 0.025).is_none());
        assert!(AoSettings::new(AoMethod::Gtao, 0.5, f32::NAN, 0.025).is_none());
        assert!(AoSettings::new(AoMethod::Gtao, 0.5, 1.0, -0.1).is_none());
        let defaults = AoSettings::default();
        assert!(matches!(defaults.method(), AoMethod::Gtao));
        assert_eq!(defaults.radius(), DEFAULT_AO_RADIUS);
        assert_eq!(defaults.intensity(), DEFAULT_AO_INTENSITY);
        assert_eq!(defaults.bias(), DEFAULT_AO_BIAS);
        assert!(AoMethod::Ssao.is_ssao());
    }
}
