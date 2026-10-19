//! CPU reference math for the TAA, FXAA and SMAA resolve kernels.
//!
//! These functions intentionally mirror the GPU kernels closely enough for
//! unit tests and application-side image-filter validation.

use glam::Vec3;

use super::{EdgeFilterSettings, TaaSettings};

/// Quarter of a unit (the YCoCg weights).
const QUARTER: f32 = 0.25;
/// Half of a unit (the YCoCg weights).
const HALF: f32 = 0.5;
/// Variance inflation of the temporal neighbourhood clamp.
const CLAMP_GAMMA: f32 = 1.25;

/// Reversible YCoCg encoding of an RGB colour.
///
/// The chroma axes are decorrelated from the luma, so clamping the
/// history in this space keeps its brightness while rejecting the colour
/// drift that would show as ghosting.
pub fn ycocg_encode(color: Vec3) -> Vec3 {
    let (r, g, b) = (color.x, color.y, color.z);
    Vec3 {
        x: QUARTER * r + HALF * g + QUARTER * b,
        y: HALF * r - HALF * b,
        z: -QUARTER * r + HALF * g - QUARTER * b,
    }
}

/// Inverse of [`ycocg_encode`].
pub fn ycocg_decode(color: Vec3) -> Vec3 {
    let (y, co, cg) = (color.x, color.y, color.z);
    Vec3 {
        x: y + co - cg,
        y: y + cg,
        z: y - co - cg,
    }
}

/// Colour bounds of the current-frame neighbourhood, in YCoCg.
///
/// The bounds are the intersection of the neighbourhood's own minimum and
/// maximum with the mean ± `CLAMP_GAMMA` standard deviations: a flat
/// neighbourhood keeps its exact range, a noisy one is not widened by a
/// single outlier. The samples are encoded first, so the bounds live in
/// the same space the history is clamped in.
pub fn neighbourhood_bounds(samples: &[Vec3]) -> (Vec3, Vec3) {
    let (lower, upper, _) = neighbourhood_bounds_with_validity(samples);
    (lower, upper)
}

/// Computes the encoded neighborhood bounds plus a weight that rejects
/// history entirely when there are no current-frame samples.
pub(super) fn neighbourhood_bounds_with_validity(samples: &[Vec3]) -> (Vec3, Vec3, f32) {
    if samples.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO, 0.0);
    }
    let encode = |sample: Vec3| {
        let (r, g, b) = (sample.x, sample.y, sample.z);
        Vec3 {
            x: QUARTER * r + HALF * g + QUARTER * b,
            y: HALF * r - HALF * b,
            z: -QUARTER * r + HALF * g - QUARTER * b,
        }
    };
    let mut min = encode(samples[0]);
    let mut max = min;
    let mut sum = Vec3::ZERO;
    let mut sum_squares = Vec3::ZERO;
    for sample in samples {
        let encoded = encode(*sample);
        min = min.min(encoded);
        max = max.max(encoded);
        sum += encoded;
        sum_squares += encoded * encoded;
    }
    let count = samples.len() as f32;
    let mean = sum / count;
    let variance = (sum_squares / count - mean * mean).max(Vec3::ZERO);
    let sigma = variance.sqrt();
    let lower = mean - CLAMP_GAMMA * sigma;
    let upper = mean + CLAMP_GAMMA * sigma;
    (min.max(lower), max.min(upper), 1.0)
}

/// Clamps the history color in YCoCg, then restores RGB.
pub(super) fn clamp_history(history: Vec3, lower: Vec3, upper: Vec3) -> Vec3 {
    ycocg_decode(ycocg_encode(history).clamp(lower, upper))
}

/// Blends a clamped history sample; `validity` is zero for an empty neighborhood.
pub(super) fn blend_history(current: Vec3, history: Vec3, feedback: f32, validity: f32) -> Vec3 {
    if validity == 0.0 {
        return current;
    }
    current + (history - current) * feedback
}

/// Temporal resolve of one pixel: clamps `history` into the colour range
/// of the current neighbourhood and blends it with `current`.
///
/// The clamp is what keeps a moving object from dragging its old colour
/// across the screen: a history sample that no longer fits the current
/// neighbourhood is pulled back to it, so the ghost fades out instead of
/// smearing.
pub fn taa_resolve(
    current: Vec3,
    history: Vec3,
    neighbourhood: &[Vec3],
    settings: TaaSettings,
) -> Vec3 {
    let (lower, upper, validity) = neighbourhood_bounds_with_validity(neighbourhood);
    let clamped = clamp_history(history, lower, upper);
    blend_history(current, clamped, settings.feedback, validity)
}

/// Rec. 601 luma weights.
const LUMA_RED: f32 = 0.299;
/// Rec. 601 luma weights.
const LUMA_GREEN: f32 = 0.587;
/// Rec. 601 luma weights.
const LUMA_BLUE: f32 = 0.114;

/// Luma (Rec. 601) of a colour — the brightness the edge filters work on.
pub fn fxaa_luma(color: Vec3) -> f32 {
    LUMA_RED * color.x + LUMA_GREEN * color.y + LUMA_BLUE * color.z
}

/// Luma gradient of a pixel along the horizontal and the vertical axis.
///
/// The gradients are signed differences of the neighbouring luma values;
/// their magnitudes say how much the image changes across each axis.
pub fn edge_gradient(
    luma_north: f32,
    luma_south: f32,
    luma_east: f32,
    luma_west: f32,
) -> (f32, f32) {
    (luma_east - luma_west, luma_south - luma_north)
}

/// The axis (0 = horizontal, 1 = vertical) with the stronger gradient.
///
/// The edge filter blends across the dominant gradient, i.e. along the
/// axis the image changes fastest — that is the direction in which the
/// aliased edge sits.
pub fn edge_axis(gradient_x: f32, gradient_y: f32) -> u32 {
    if gradient_x.abs() > gradient_y.abs() {
        0
    } else {
        1
    }
}

/// FXAA blend of one pixel.
///
/// `tap_a` and `tap_b` are the two neighbours along the dominant
/// gradient axis, `range` the luma range across that axis. A flat pixel
/// (`range` below the threshold) is returned unchanged; a strong edge is
/// blended towards the average of the two taps.
pub fn fxaa_blend(
    center: Vec3,
    tap_a: Vec3,
    tap_b: Vec3,
    range: f32,
    settings: EdgeFilterSettings,
) -> Vec3 {
    if range.partial_cmp(&settings.threshold) != Some(std::cmp::Ordering::Greater) {
        return center;
    }
    let blurred = (center + center + tap_a + tap_b) * QUARTER;
    let weight = ((range - settings.threshold) / range).clamp(0.0, 1.0);
    center.lerp(blurred, weight)
}

/// SMAA blend of one pixel.
///
/// The two samples straddling the edge are weighted by their luma
/// similarity to the centre: the sample that matches the centre keeps
/// its weight, the other one fades — the morphological part of SMAA, with
/// the similarity replacing the search textures of the reference
/// implementation.
pub fn smaa_blend(
    center: Vec3,
    tap_a: Vec3,
    tap_b: Vec3,
    range: f32,
    settings: EdgeFilterSettings,
) -> Vec3 {
    if range.partial_cmp(&settings.threshold) != Some(std::cmp::Ordering::Greater) {
        return center;
    }
    let luma_center = fxaa_luma(center);
    let weight_a = 1.0 / (1.0 + (fxaa_luma(tap_a) - luma_center).abs());
    let weight_b = 1.0 / (1.0 + (fxaa_luma(tap_b) - luma_center).abs());
    let total = weight_a + weight_b;
    let blended = (tap_a * weight_a + tap_b * weight_b) / total;
    let weight = ((range - settings.threshold) / range).clamp(0.0, 1.0);
    center.lerp(blended, weight)
}
