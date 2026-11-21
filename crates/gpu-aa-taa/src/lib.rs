//! Temporal anti-aliasing history resolve on the stable Rust → LLVM IR → WGSL
//! path.
//!
//! The pixel is reprojected into the previous frame through the motion vector,
//! the history sample is clamped into the YCoCg range of the current
//! neighbourhood, and the two are blended. A pixel with no surface behind it,
//! an invalid motion word, or a reprojection that leaves the screen keeps the
//! current colour — the history is not read there. The pass writes two colour
//! attachments: the presented image and the history the next frame reads.
//!
//! `xtask build-shaders --package ornis-gpu-aa-taa --entry all` writes the
//! single-sample artifact; `--multisampled-textures 0:2` the one whose depth
//! binding is multisampled.

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_quad::standard_quad;
use ornis_gpu_shader::DepthTexture2d;
use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::Sampler;
use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::Texture2dU32;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::abs;
use ornis_gpu_shader::clamp;
use ornis_gpu_shader::fragment_input_lane;
use ornis_gpu_shader::fragment_store_at;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::mix;
use ornis_gpu_shader::sqrt;

/// Sampler binding of the source layer.
///
/// The `#[gpu(...)]` attribute needs a literal, so the same number appears
/// there; this constant is what the sample calls name.
const SRC_SAMPLER: u32 = 1;

/// Field of the history feedback in [`AaParams`].
const FEEDBACK: usize = 0;
/// Field of the neighbourhood radius in [`AaParams`].
const CLAMP_RADIUS: usize = 2;
/// Field of the history-valid flag in [`AaParams`].
const HISTORY_VALID: usize = 3;

/// Depth the renderer clears the g-buffer to: nothing is behind such a pixel.
const CLEAR_DEPTH: f32 = 1.0;
/// Packed word that marks an invalid motion vector.
const INVALID_MOTION: u32 = 65535;
/// Divisor of the packed previous-depth word (`R16Uint`, 65534 levels).
const PREVIOUS_DEPTH_SCALE: f32 = 65534.0;
/// Depth difference below which the history is trusted.
const DEPTH_GAP_LIMIT: f32 = 0.002;
/// Width of the neighbourhood clamp: mean ± this many standard deviations.
const SIGMA_SCALE: f32 = 1.25;
/// Flag value that marks a valid history buffer.
const HISTORY_VALID_FLAG: f32 = 0.5;

/// YCoCg encode weights: luma, orange chroma, green chroma.
const YCOCG_LUMA: f32 = 0.25;
/// See [`YCOCG_LUMA`].
const YCOCG_CHROMA: f32 = 0.5;

/// Sentinel that grows into the neighbourhood minimum on the first tap.
const BOUND_START: f32 = f32::MAX;
/// Sentinel that shrinks into the neighbourhood maximum on the first tap.
const BOUND_START_NEGATIVE: f32 = f32::MIN;
/// Lane of the history's packed depth, stored in the alpha channel.
const HISTORY_DEPTH_LANE: u32 = 3;

/// `AaParams`: feedback, threshold, clamp radius, history flag, mode, pad.
pub struct AaParams;

impl GpuStruct for AaParams {
    const ID: u32 = 0;
    const FIELD_TAGS: &'static [u8] = &[
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_U32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
    ];
}

/// YCoCg luma of one colour.
#[inline(always)]
fn ycocg_y(r: f32, g: f32, b: f32) -> f32 {
    YCOCG_LUMA * r + YCOCG_CHROMA * g + YCOCG_LUMA * b
}

/// YCoCg orange chroma of one colour.
#[inline(always)]
fn ycocg_co(r: f32, b: f32) -> f32 {
    YCOCG_CHROMA * r - YCOCG_CHROMA * b
}

/// YCoCg green chroma of one colour.
#[inline(always)]
fn ycocg_cg(r: f32, g: f32, b: f32) -> f32 {
    -YCOCG_LUMA * r + YCOCG_CHROMA * g - YCOCG_LUMA * b
}

/// TAA vertex stage: the shared fullscreen quad.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "vs_main")]`.
#[gpu_entry(vertex)]
pub fn vs_main() {
    standard_quad();
}

/// Bounds of the neighbourhood clamp, in YCoCg.
struct Bounds {
    /// Lower luma bound.
    low_y: f32,
    /// Lower orange-chroma bound.
    low_co: f32,
    /// Lower green-chroma bound.
    low_cg: f32,
    /// Upper luma bound.
    high_y: f32,
    /// Upper orange-chroma bound.
    high_co: f32,
    /// Upper green-chroma bound.
    high_cg: f32,
}

/// Write the resolved colour and its depth to both targets.
#[inline(always)]
fn store_outputs(r: f32, g: f32, b: f32, depth: f32) {
    fragment_store_at::<0>(r, g, b, 1.0);
    fragment_store_at::<1>(r, g, b, depth);
}

/// Clamp bounds of the neighbourhood around `(u, v)`.
///
/// The history is clamped into `mean ± SIGMA_SCALE σ` of the current
/// neighbourhood, itself clamped to the observed min and max: a stale sample
/// cannot drag its colour along, while a single loud texel cannot widen the
/// range it is allowed to move in.
impl Bounds {
    /// Bounds that grow towards the neighbourhood minimum and maximum.
    #[inline(always)]
    fn sentinel() -> Self {
        Self {
            low_y: BOUND_START,
            low_co: BOUND_START,
            low_cg: BOUND_START,
            high_y: BOUND_START_NEGATIVE,
            high_co: BOUND_START_NEGATIVE,
            high_cg: BOUND_START_NEGATIVE,
        }
    }

    /// The sentinel bounds narrowed to `mean ± SIGMA_SCALE σ`.
    #[inline(always)]
    fn narrowed(
        self,
        mean_y: f32,
        mean_co: f32,
        mean_cg: f32,
        sigma_y: f32,
        sigma_co: f32,
        sigma_cg: f32,
    ) -> Self {
        Self {
            low_y: self.low_y.max(mean_y - SIGMA_SCALE * sigma_y),
            low_co: self.low_co.max(mean_co - SIGMA_SCALE * sigma_co),
            low_cg: self.low_cg.max(mean_cg - SIGMA_SCALE * sigma_cg),
            high_y: self.high_y.min(mean_y + SIGMA_SCALE * sigma_y),
            high_co: self.high_co.min(mean_co + SIGMA_SCALE * sigma_co),
            high_cg: self.high_cg.min(mean_cg + SIGMA_SCALE * sigma_cg),
        }
    }
}

// qual:allow(srp, max_parameters=7) reason: "the neighbourhood needs the sampled layer, its sampler, the centre, the extent and the radius; each is a scalar or a zero-sized binding"
#[inline(always)]
fn neighbourhood_bounds(
    src_tex: Texture2d<0, 0>,
    src_sampler: Sampler<0, 1>,
    u: f32,
    v: f32,
    width: f32,
    height: f32,
    radius: i32,
) -> Bounds {
    let mut bounds = Bounds::sentinel();
    let mut sum_y = 0.0f32;
    let mut sum_co = 0.0f32;
    let mut sum_cg = 0.0f32;
    let mut sum_sq_y = 0.0f32;
    let mut sum_sq_co = 0.0f32;
    let mut sum_sq_cg = 0.0f32;
    let mut count = 0.0f32;
    let mut dy = -radius;
    while dy <= radius {
        let mut dx = -radius;
        while dx <= radius {
            let tap_u = clamp(u + dx as f32 / width, 0.0, 1.0);
            let tap_v = clamp(v + dy as f32 / height, 0.0, 1.0);
            let tap_r = src_tex.sample_lane::<0, { SRC_SAMPLER }, 0>(src_sampler, tap_u, tap_v);
            let tap_g = src_tex.sample_lane::<0, { SRC_SAMPLER }, 1>(src_sampler, tap_u, tap_v);
            let tap_b = src_tex.sample_lane::<0, { SRC_SAMPLER }, 2>(src_sampler, tap_u, tap_v);
            let y = ycocg_y(tap_r, tap_g, tap_b);
            let co = ycocg_co(tap_r, tap_b);
            let cg = ycocg_cg(tap_r, tap_g, tap_b);
            bounds.low_y = bounds.low_y.min(y);
            bounds.low_co = bounds.low_co.min(co);
            bounds.low_cg = bounds.low_cg.min(cg);
            bounds.high_y = bounds.high_y.max(y);
            bounds.high_co = bounds.high_co.max(co);
            bounds.high_cg = bounds.high_cg.max(cg);
            sum_y += y;
            sum_co += co;
            sum_cg += cg;
            sum_sq_y += y * y;
            sum_sq_co += co * co;
            sum_sq_cg += cg * cg;
            count += 1.0;
            dx += 1;
        }
        dy += 1;
    }
    let mean_y = sum_y / count;
    let mean_co = sum_co / count;
    let mean_cg = sum_cg / count;
    let sigma_y = sqrt((sum_sq_y / count - mean_y * mean_y).max(0.0));
    let sigma_co = sqrt((sum_sq_co / count - mean_co * mean_co).max(0.0));
    let sigma_cg = sqrt((sum_sq_cg / count - mean_cg * mean_cg).max(0.0));
    bounds.narrowed(mean_y, mean_co, mean_cg, sigma_y, sigma_co, sigma_cg)
}

/// TAA fragment stage: reproject, clamp the history, blend.
///
/// Writes `@location(0)` (presented image) and `@location(1)` (history).
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "fs_main")]`.
// qual:allow(srp, max_parameters=7) reason: "the pass reads five layers, one sampler and one uniform; the SDK binds each on its own parameter"
#[gpu_entry(fragment)]
pub fn fs_main(
    #[gpu(texture_2d, group = 0, binding = 0)] src_tex: Texture2d,
    #[gpu(sampler, group = 0, binding = 1)] src_sampler: Sampler,
    #[gpu(depth_texture_2d, group = 0, binding = 2)] depth_tex: DepthTexture2d,
    #[gpu(texture_2d, group = 0, binding = 3)] motion_tex: Texture2d,
    #[gpu(texture_2d_u32, group = 0, binding = 4)] previous_depth_tex: Texture2dU32,
    #[gpu(texture_2d, group = 0, binding = 5)] history_tex: Texture2d,
    #[gpu(uniform, group = 0, binding = 6)] aa_params: Uniform<AaParams>,
) {
    let u = fragment_input_lane::<0, 2, 0>();
    let v = fragment_input_lane::<0, 2, 1>();
    // Texel of this pixel: `uv == 1.0` is one past the last texel.
    let width = src_tex.width() as f32;
    let height = src_tex.height() as f32;
    let texel_x = (u * width) as u32;
    let texel_y = (v * height) as u32;
    let current_r = src_tex.sample_lane::<0, { SRC_SAMPLER }, 0>(src_sampler, u, v);
    let current_g = src_tex.sample_lane::<0, { SRC_SAMPLER }, 1>(src_sampler, u, v);
    let current_b = src_tex.sample_lane::<0, { SRC_SAMPLER }, 2>(src_sampler, u, v);
    // Nothing is behind a pixel that still holds the cleared depth.
    let depth = depth_tex.load(texel_x, texel_y);
    if depth >= CLEAR_DEPTH {
        store_outputs(current_r, current_g, current_b, 1.0);
        return;
    }
    // The g-buffer stores velocity in RG16F and packs previous NDC depth plus
    // validity into an R16Uint word; 65535 marks invalid motion.
    let motion_x = motion_tex.load::<0>(texel_x, texel_y);
    let motion_y = motion_tex.load::<1>(texel_x, texel_y);
    let packed_previous_depth = previous_depth_tex.load(texel_x, texel_y);
    let prev_u = u - motion_x;
    let prev_v = v - motion_y;
    // Clamped UVs are always inside the unit square, so this is an on-screen
    // test, not a range construction.
    // qual:allow(complexity, magic_numbers) reason: "0.0/1.0 are the UV bounds of the previous frame, not tunables"
    let off_screen = !(0.0..=1.0).contains(&prev_u) || !(0.0..=1.0).contains(&prev_v);
    if packed_previous_depth == INVALID_MOTION || off_screen {
        store_outputs(current_r, current_g, current_b, depth);
        return;
    }
    let previous_depth = packed_previous_depth as f32 / PREVIOUS_DEPTH_SCALE;
    let radius = aa_params.field_f32::<CLAMP_RADIUS>() as i32;
    let bounds = neighbourhood_bounds(src_tex, src_sampler, u, v, width, height, radius);
    let history_valid = aa_params.field_f32::<HISTORY_VALID>();
    let mut result_r = current_r;
    let mut result_g = current_g;
    let mut result_b = current_b;
    if history_valid > HISTORY_VALID_FLAG {
        let history_r =
            history_tex.sample_lane::<0, { SRC_SAMPLER }, 0>(src_sampler, prev_u, prev_v);
        let history_g =
            history_tex.sample_lane::<0, { SRC_SAMPLER }, 1>(src_sampler, prev_u, prev_v);
        let history_b =
            history_tex.sample_lane::<0, { SRC_SAMPLER }, 2>(src_sampler, prev_u, prev_v);
        let history_alpha = history_tex.sample_lane::<0, { SRC_SAMPLER }, { HISTORY_DEPTH_LANE }>(
            src_sampler,
            prev_u,
            prev_v,
        );
        // A depth mismatch means the previous surface was occluded or this is
        // a disocclusion: rejecting it keeps moving objects from leaving long
        // history trails.
        if abs(history_alpha - previous_depth) < DEPTH_GAP_LIMIT {
            let clamped_y = clamp(
                ycocg_y(history_r, history_g, history_b),
                bounds.low_y,
                bounds.high_y,
            );
            let clamped_co = clamp(
                ycocg_co(history_r, history_b),
                bounds.low_co,
                bounds.high_co,
            );
            let clamped_cg = clamp(
                ycocg_cg(history_r, history_g, history_b),
                bounds.low_cg,
                bounds.high_cg,
            );
            // YCoCg decode of the clamped sample.
            let feedback = aa_params.field_f32::<FEEDBACK>();
            result_r = mix(current_r, clamped_y + clamped_co - clamped_cg, feedback);
            result_g = mix(current_g, clamped_y + clamped_cg, feedback);
            result_b = mix(current_b, clamped_y - clamped_co - clamped_cg, feedback);
        }
    }
    store_outputs(result_r, result_g, result_b, depth);
}
