//! Anti-aliasing: mode selection, temporal jitter and the CPU reference
//! of the three resolve kernels (TAA, FXAA, SMAA).
//!
//! Four techniques are selectable through [`AntiAliasMode`]: hardware MSAA
//! (the historical default, 4x multisampled geometry layers), temporal
//! anti-aliasing ([`AntiAliasMode::Taa`]), FXAA and SMAA (post-process
//! passes over the composited image).
//!
//! The temporal mode jitters the projection by a sub-pixel amount every
//! frame ([`JitterSequence`], Halton base 2/3) and resolves the frame
//! against a history buffer: the current sample is reprojected into the
//! previous frame (through the depth buffer), the history is clamped to
//! the colour range of the current neighbourhood and the two are blended.
//! The helpers below are the CPU reference of that resolve — the GPU
//! kernels in `aa_taa_generated` and `aa_edge_generated` mirror them, and the unit
//! tests pin the behaviour the resolve must show (a static scene
//! converges, a moving one does not ghost).
//!
//! FXAA and SMAA are single-pass edge filters over the composited image:
//! FXAA blends along the dominant luma gradient, SMAA weights the two
//! samples straddling the edge by their luma similarity to the centre.

use glam::{Mat4, Vec2, Vec3};
use ornis_core::units::Clamped01;

/// Which anti-aliasing technique the frame renders with.
///
/// The mode is a plan-level decision: [`crate::frame_exec::RenderFrame3D`]
/// wires the matching passes, and the renderer jitters the projection in
/// the temporal mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AntiAliasMode {
    /// No anti-aliasing at all: geometry edges alias (the cheapest frame).
    Off,
    /// Hardware multisampling (4x) — the historical default.
    #[default]
    Msaa,
    /// Temporal anti-aliasing: jittered projection plus a history resolve.
    Taa,
    /// FXAA: a post-process edge filter over the composited image.
    Fxaa,
    /// SMAA: a post-process morphological edge filter over the same image.
    Smaa,
}

impl AntiAliasMode {
    /// `true` for the temporal mode (jittered projection, history buffer).
    pub fn is_temporal(self) -> bool {
        matches!(self, Self::Taa)
    }

    /// `true` for the post-process modes (one fullscreen pass after the
    /// composite).
    pub fn is_post_process(self) -> bool {
        matches!(self, Self::Fxaa | Self::Smaa)
    }

    /// `true` when the mode runs an anti-aliasing pass of its own (every
    /// mode except the plain hardware MSAA and the disabled one).
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off | Self::Msaa)
    }

    /// `true` when the mode needs a persistent history buffer (only the
    /// temporal mode accumulates across frames).
    pub fn needs_history(self) -> bool {
        self.is_temporal()
    }

    /// The selector the edge-filter shader reads: 0 = FXAA, 1 = SMAA.
    ///
    /// The two post-process filters share one shader; the selector picks
    /// the blend (see [`crate::shaders::aa_edge_generated`]).
    pub fn shader_mode(self) -> u32 {
        match self {
            Self::Smaa => 1,
            _ => 0,
        }
    }
}

/// Whether a pixel's previous vertex position is reliable for temporal reprojection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionValidity {
    /// Reproject into history using the motion vector and previous depth.
    Valid,
    /// Keep the current color and do not sample history for this pixel.
    Invalid,
}

impl MotionValidity {
    /// `u32` encoding consumed by the g-buffer shader.
    pub(crate) const fn shader_flag(self) -> u32 {
        match self {
            Self::Valid => 1,
            Self::Invalid => 0,
        }
    }
}

/// The `index`-th value of the Halton sequence in `base`, in `[0, 1)`.
///
/// `index` is 1-based in the classic formulation; index `0` returns `0`.
/// The radical-inverse construction (`f = 1/base`, `r += f * (i % base)`)
/// is the standard one — the sequence fills the interval evenly, which is
/// what the temporal mode needs from a jitter pattern.
pub fn halton(index: u32, base: u32) -> f32 {
    let base = base.max(2);
    let mut fraction = 1.0f32;
    let mut result = 0.0f32;
    let mut rest = index;
    while rest > 0 {
        fraction /= base as f32;
        result += fraction * (rest % base) as f32;
        rest /= base;
    }
    result
}

/// Sub-pixel jitter sequence of the temporal mode.
///
/// The offsets live in `[-0.5, 0.5]` pixels (x right, y down) and cycle
/// over [`JitterSequence::PERIOD`] frames, so the same world point lands
/// on a different sub-pixel position every frame — the temporal
/// supersampling the resolve accumulates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitterSequence {
    index: u32,
}

impl JitterSequence {
    /// Cycle length of the sequence (the Halton period of bases 2 and 3).
    pub const PERIOD: u32 = 16;

    /// Half a pixel: the offsets are centred on the pixel centre.
    const HALF_PIXEL: f32 = 0.5;

    /// A fresh sequence, positioned at the first offset.
    pub const fn new() -> Self {
        Self { index: 0 }
    }

    /// The current index in the cycle.
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The jitter offset of the current index, in pixels (x right, y down).
    pub fn offset(self) -> Vec2 {
        Self::offset_at(self.index)
    }

    /// The jitter offset of `index`, in pixels (x right, y down).
    pub fn offset_at(index: u32) -> Vec2 {
        let x = halton(index + 1, 2) - Self::HALF_PIXEL;
        let y = halton(index + 1, 3) - Self::HALF_PIXEL;
        Vec2::new(x, y)
    }

    /// The current offset, then advance to the next index of the cycle.
    pub fn advance(&mut self) -> Vec2 {
        let offset = self.offset();
        self.index = (self.index + 1) % Self::PERIOD;
        offset
    }
}

impl Default for JitterSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// Applies a sub-pixel jitter to a view-projection matrix.
///
/// `offset` is measured in pixels (x right, y down): the rendered image
/// shifts by exactly that amount, so the world point under a given pixel
/// changes every frame. The offset is applied in clip space (it scales
/// with the clip `w`), which keeps the shift constant in screen space for
/// every depth — the property the temporal resolve relies on.
pub fn jittered_projection(view_proj: Mat4, offset: Vec2, extent: (u32, u32)) -> Mat4 {
    let width = extent.0.max(1) as f32;
    let height = extent.1.max(1) as f32;
    // NDC offset: x to the right, y upwards (the pixel y axis points down).
    let ndc_x = offset.x * 2.0 / width;
    let ndc_y = -offset.y * 2.0 / height;
    // Add the desired NDC offset times clip `w` to each clip-space x/y
    // row. In column-major form this left-multiplies by a clip translation,
    // keeping the pixel shift constant at every depth.
    let mut columns = view_proj.to_cols_array_2d();
    for column in &mut columns {
        column[0] += ndc_x * column[3];
        column[1] += ndc_y * column[3];
    }
    Mat4::from_cols_array_2d(&columns)
}

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
    Vec3::new(
        QUARTER * r + HALF * g + QUARTER * b,
        HALF * r - HALF * b,
        -QUARTER * r + HALF * g - QUARTER * b,
    )
}

/// Inverse of [`ycocg_encode`].
pub fn ycocg_decode(color: Vec3) -> Vec3 {
    let (y, co, cg) = (color.x, color.y, color.z);
    Vec3::new(y + co - cg, y + cg, y - co - cg)
}

/// Colour bounds of the current-frame neighbourhood, in YCoCg.
///
/// The bounds are the intersection of the neighbourhood's own minimum and
/// maximum with the mean ± `CLAMP_GAMMA` standard deviations: a flat
/// neighbourhood keeps its exact range, a noisy one is not widened by a
/// single outlier. The samples are encoded first, so the bounds live in
/// the same space the history is clamped in.
pub fn neighbourhood_bounds(samples: &[Vec3]) -> (Vec3, Vec3) {
    let Some(first) = samples.first() else {
        return (Vec3::ZERO, Vec3::ZERO);
    };
    let mut min = ycocg_encode(*first);
    let mut max = min;
    let mut sum = Vec3::ZERO;
    let mut sum_squares = Vec3::ZERO;
    for sample in samples {
        let encoded = ycocg_encode(*sample);
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
    (min.max(lower), max.min(upper))
}

/// Settings of the temporal resolve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaaSettings {
    feedback: Clamped01,
    clamp_radius: u32,
}

impl TaaSettings {
    /// Default history weight (nine tenths of the blend is the history).
    pub const DEFAULT_FEEDBACK: f32 = 0.9;
    /// Default neighbourhood radius in pixels (a 3×3 tap window).
    pub const DEFAULT_CLAMP_RADIUS: u32 = 1;
    /// Maximum neighbourhood radius (bounds the shader to 25 taps).
    pub const MAX_CLAMP_RADIUS: u32 = 2;

    /// Checked constructor: `None` for a non-finite feedback.
    ///
    /// The clamp radius is clamped to `1..=MAX_CLAMP_RADIUS` (the centre
    /// tap alone would clamp the history to one colour and freeze the image;
    /// the upper bound keeps the shader's dynamic loop cost finite).
    pub fn new(feedback: f32, clamp_radius: u32) -> Option<Self> {
        if !feedback.is_finite() {
            return None;
        }
        Some(Self {
            feedback: Clamped01::new(feedback),
            clamp_radius: clamp_radius.clamp(1, Self::MAX_CLAMP_RADIUS),
        })
    }

    /// Weight of the history in the blend (`0` = current frame only,
    /// `1` = history only).
    pub fn feedback(self) -> f32 {
        self.feedback.get()
    }

    /// Neighbourhood radius in pixels.
    pub fn clamp_radius(self) -> u32 {
        self.clamp_radius
    }
}

impl Default for TaaSettings {
    fn default() -> Self {
        Self {
            feedback: Clamped01::new(Self::DEFAULT_FEEDBACK),
            clamp_radius: Self::DEFAULT_CLAMP_RADIUS,
        }
    }
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
    if neighbourhood.is_empty() {
        return current;
    }
    let (lower, upper) = neighbourhood_bounds(neighbourhood);
    let clamped = ycocg_decode(ycocg_encode(history).clamp(lower, upper));
    current.lerp(clamped, settings.feedback())
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

/// Settings of the post-process edge filters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeFilterSettings {
    threshold: f32,
}

impl EdgeFilterSettings {
    /// Default luma threshold: pixels flatter than this are left alone.
    pub const DEFAULT_THRESHOLD: f32 = 0.05;

    /// Checked constructor: `None` for a non-finite threshold.
    pub fn new(threshold: f32) -> Option<Self> {
        if !threshold.is_finite() {
            return None;
        }
        Some(Self { threshold })
    }

    /// Luma range below which a pixel is not filtered.
    pub fn threshold(self) -> f32 {
        self.threshold
    }
}

impl Default for EdgeFilterSettings {
    fn default() -> Self {
        Self {
            threshold: Self::DEFAULT_THRESHOLD,
        }
    }
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

/// Opt-in anti-aliasing wiring for the frame plan: the mode plus the
/// settings of its kernel.
///
/// The plan wires the matching pass (the temporal resolve or the edge
/// filter); the renderer runs in the same mode, so the projection jitters
/// exactly when the resolve expects it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AaWiring {
    mode: AntiAliasMode,
    taa: TaaSettings,
    edge: EdgeFilterSettings,
}

impl AaWiring {
    /// The wiring for a mode with the default settings.
    pub fn new(mode: AntiAliasMode) -> Self {
        Self {
            mode,
            taa: TaaSettings::default(),
            edge: EdgeFilterSettings::default(),
        }
    }

    /// The wiring for a mode with explicit settings.
    pub fn with_settings(mode: AntiAliasMode, taa: TaaSettings, edge: EdgeFilterSettings) -> Self {
        Self { mode, taa, edge }
    }

    /// The selected mode.
    pub fn mode(&self) -> AntiAliasMode {
        self.mode
    }

    /// The settings of the temporal resolve.
    pub fn taa_settings(&self) -> TaaSettings {
        self.taa
    }

    /// The settings of the edge filters.
    pub fn edge_settings(&self) -> EdgeFilterSettings {
        self.edge
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Luma of a mid grey.
    const GREY_LUMA: f32 = 0.5;

    #[test]
    fn mode_defaults_to_msaa_and_classifies_the_techniques() {
        assert_eq!(AntiAliasMode::default(), AntiAliasMode::Msaa);
        assert!(AntiAliasMode::Taa.is_temporal());
        assert!(!AntiAliasMode::Msaa.is_temporal());
        assert!(AntiAliasMode::Fxaa.is_post_process());
        assert!(AntiAliasMode::Smaa.is_post_process());
        assert!(!AntiAliasMode::Taa.is_post_process());
        assert!(AntiAliasMode::Taa.needs_history());
        assert!(!AntiAliasMode::Fxaa.needs_history());
        assert!(!AntiAliasMode::Off.is_enabled());
        assert!(!AntiAliasMode::Msaa.is_enabled());
        assert!(AntiAliasMode::Taa.is_enabled());
    }

    #[test]
    fn halton_fills_the_interval_evenly() {
        // The classic first values of the base-2 sequence.
        let base_two: Vec<f32> = (1..=4).map(|i| halton(i, 2)).collect();
        assert!((base_two[0] - 0.5).abs() < 1.0e-6);
        assert!((base_two[1] - 0.25).abs() < 1.0e-6);
        assert!((base_two[2] - 0.75).abs() < 1.0e-6);
        assert!((base_two[3] - 0.125).abs() < 1.0e-6);
        // Base 3 starts at one third.
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1.0e-6);
        // Index zero is the neutral point and a base below two is clamped.
        assert_eq!(halton(0, 2), 0.0);
        assert_eq!(halton(1, 1), halton(1, 2));
        // Every value of a cycle stays inside the unit interval.
        for index in 1..=JitterSequence::PERIOD {
            for base in [2u32, 3] {
                let value = halton(index, base);
                assert!((0.0..1.0).contains(&value), "halton {value}");
            }
        }
    }

    #[test]
    fn jitter_offsets_stay_inside_half_a_pixel() {
        let mut sequence = JitterSequence::new();
        for _ in 0..JitterSequence::PERIOD {
            let offset = sequence.advance();
            assert!(offset.x.abs() <= 0.5, "x {offset}");
            assert!(offset.y.abs() <= 0.5, "y {offset}");
        }
        // The cycle wraps back to the first offset.
        assert_eq!(sequence.index(), 0);
        assert_eq!(JitterSequence::offset_at(0), JitterSequence::new().offset());
    }

    /// Pixel position of a world point under a view-projection matrix.
    fn project_pixel(view_proj: Mat4, point: glam::Vec3, extent: (u32, u32)) -> Vec2 {
        let clip = view_proj * point.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        Vec2::new(
            (ndc.x * 0.5 + 0.5) * extent.0 as f32,
            (0.5 - ndc.y * 0.5) * extent.1 as f32,
        )
    }

    #[test]
    fn jittered_projection_shifts_the_image_by_the_offset() {
        let extent = (64u32, 32u32);
        // Minimal perspective matrix: clip.w = -view.z, so the test
        // verifies that the jitter stays a constant pixel shift at depth.
        let projection = Mat4::from_cols_array_2d(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -1.0],
            [0.0, 0.0, -0.2, 0.0],
        ]);
        let point = glam::Vec3::new(0.3, -0.2, -5.0);
        let base = project_pixel(projection, point, extent);
        for offset in [
            Vec2::new(0.25, 0.0),
            Vec2::new(-0.5, 0.5),
            Vec2::new(0.0, -0.25),
        ] {
            let jittered = jittered_projection(projection, offset, extent);
            let moved = project_pixel(jittered, point, extent);
            assert!(
                (moved.x - base.x - offset.x).abs() < 1.0e-3,
                "x drift {moved:?} vs {base:?} offset {offset:?}"
            );
            assert!(
                (moved.y - base.y - offset.y).abs() < 1.0e-3,
                "y drift {moved:?} vs {base:?} offset {offset:?}"
            );
        }
        // A zero offset is the exact identity.
        let identity = jittered_projection(projection, Vec2::ZERO, extent);
        assert_eq!(identity, projection);
    }

    #[test]
    fn ycocg_round_trips() {
        for color in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.2, 0.7, 0.4),
            Vec3::new(-0.5, 1.5, 0.25),
        ] {
            let decoded = ycocg_decode(ycocg_encode(color));
            assert!(
                (decoded - color).length() < 1.0e-5,
                "{color:?} -> {decoded:?}"
            );
        }
    }

    #[test]
    fn neighbourhood_bounds_cover_the_taps() {
        let samples = [Vec3::new(0.2, 0.2, 0.2), Vec3::new(0.8, 0.8, 0.8)];
        let (lower, upper) = neighbourhood_bounds(&samples);
        assert!(lower.x <= 0.2 && upper.x >= 0.8, "{lower:?} {upper:?}");
        // An empty neighbourhood has no range at all.
        let (lower, upper) = neighbourhood_bounds(&[]);
        assert_eq!(lower, Vec3::ZERO);
        assert_eq!(upper, Vec3::ZERO);
    }

    #[test]
    fn temporal_resolve_keeps_the_history_inside_the_neighbourhood() {
        let settings = TaaSettings::default();
        let current = Vec3::new(0.5, 0.5, 0.5);
        let neighbourhood = [Vec3::splat(0.3), current, Vec3::splat(0.7)];
        // A history inside the range survives the clamp unchanged.
        let history = Vec3::new(0.4, 0.4, 0.4);
        let resolved = taa_resolve(current, history, &neighbourhood, settings);
        assert!(
            resolved.x < current.x && resolved.x > history.x,
            "the blend must sit between the current pixel and the history: {resolved:?}"
        );
        // A history outside the range is pulled back to it: no ghosting.
        let stale = Vec3::new(1.0, 0.0, 0.0);
        let clamped = taa_resolve(current, stale, &neighbourhood, settings);
        assert!((clamped - current).length() < 1.0e-5, "{clamped:?}");
        // An empty neighbourhood keeps the current pixel.
        assert_eq!(taa_resolve(current, stale, &[], settings), current);
    }

    #[test]
    fn temporal_resolve_without_history_is_the_identity() {
        let settings = TaaSettings::new(0.0, 1);
        let current = Vec3::new(0.25, 0.5, 0.75);
        let neighbourhood = [current, current];
        let resolved = taa_resolve(current, Vec3::ZERO, &neighbourhood, settings.unwrap());
        assert_eq!(resolved, current);
    }

    /// A one-pixel checkerboard: the worst case for a single sample.
    fn checkerboard(x: i32, y: i32) -> f32 {
        if (x + y) % 2 == 0 { 0.0 } else { 1.0 }
    }

    #[test]
    fn temporal_accumulation_converges_for_a_static_scene() {
        // The golden of the temporal mode: a static high-frequency scene
        // must converge to its true average instead of flickering between
        // the two aliased samples.
        let settings = TaaSettings::default();
        let (x, y) = (7i32, 5i32);
        let mut history = Vec3::ZERO;
        let mut last = Vec3::ZERO;
        for frame in 0..32u32 {
            let jitter = JitterSequence::offset_at(frame);
            // The jittered sample of the checkerboard.
            let sample = checkerboard(
                x + if jitter.x < 0.0 { -1 } else { 0 },
                y + if jitter.y < 0.0 { -1 } else { 0 },
            );
            let current = Vec3::new(sample, sample, sample);
            // The neighbourhood of the current frame (the 3×3 window).
            let mut neighbourhood = Vec::new();
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let value = checkerboard(x + dx, y + dy);
                    neighbourhood.push(Vec3::new(value, value, value));
                }
            }
            history = taa_resolve(current, history, &neighbourhood, settings);
            if frame >= 24 {
                last = history;
            }
        }
        // The converged value is the average of the two aliased samples.
        assert!(
            (last.x - 0.5).abs() < 0.05,
            "the temporal average must converge to 0.5, got {last:?}"
        );
    }

    #[test]
    fn temporal_resolve_drops_a_stale_history_under_motion() {
        // The ghosting case: the object moved, so the history belongs to a
        // surface that is no longer there. The clamp must reject it.
        let settings = TaaSettings::default();
        let current = Vec3::new(0.0, 0.0, 0.0);
        let neighbourhood = [current; 9];
        let stale = Vec3::new(1.0, 1.0, 1.0);
        let resolved = taa_resolve(current, stale, &neighbourhood, settings);
        assert!(
            resolved.x < 0.01,
            "the stale history must not bleed into the pixel: {resolved:?}"
        );
    }

    #[test]
    fn edge_filters_leave_flat_pixels_alone() {
        let settings = EdgeFilterSettings::default();
        let center = Vec3::new(GREY_LUMA, GREY_LUMA, GREY_LUMA);
        // A flat neighbourhood has no gradient: the pixel is untouched.
        let blended = fxaa_blend(center, center, center, 0.0, settings);
        assert_eq!(blended, center);
        let smaa = smaa_blend(center, center, center, 0.0, settings);
        assert_eq!(smaa, center);
    }

    #[test]
    fn edge_filters_blend_across_the_edge() {
        let settings = EdgeFilterSettings::default();
        let dark = Vec3::new(0.0, 0.0, 0.0);
        let bright = Vec3::new(1.0, 1.0, 1.0);
        // The centre sits on the edge: the two taps are the dark and the
        // bright side, so the filter has to pull the centre towards the
        // average of both.
        let blended = fxaa_blend(bright, dark, bright, 1.0, settings);
        assert!(blended.x < 1.0 && blended.x > 0.0, "{blended:?}");
        let smaa = smaa_blend(bright, dark, bright, 1.0, settings);
        // The matching tap wins: the blend stays above the plain average
        // of the two taps, which the similarity weighting pulls towards
        // the bright side.
        let plain_average = 0.5;
        assert!(smaa.x > plain_average, "smaa {smaa:?}");
    }

    #[test]
    fn edge_gradient_picks_the_dominant_axis() {
        let (gradient_x, gradient_y) = edge_gradient(0.0, 0.0, 1.0, 0.0);
        assert_eq!(edge_axis(gradient_x, gradient_y), 0);
        let (gradient_x, gradient_y) = edge_gradient(0.0, 1.0, 0.0, 0.0);
        assert_eq!(edge_axis(gradient_x, gradient_y), 1);
    }

    #[test]
    fn settings_reject_degenerate_input() {
        assert!(TaaSettings::new(f32::NAN, 1).is_none());
        assert!(TaaSettings::new(0.5, 0).is_some());
        assert_eq!(TaaSettings::new(0.5, 0).unwrap().clamp_radius(), 1);
        assert_eq!(
            TaaSettings::new(0.5, u32::MAX).unwrap().clamp_radius(),
            TaaSettings::MAX_CLAMP_RADIUS
        );
        assert_eq!(TaaSettings::new(2.0, 1).unwrap().feedback(), 1.0);
        assert!(EdgeFilterSettings::new(f32::NAN).is_none());
        assert_eq!(
            EdgeFilterSettings::default().threshold(),
            EdgeFilterSettings::DEFAULT_THRESHOLD
        );
    }
}
