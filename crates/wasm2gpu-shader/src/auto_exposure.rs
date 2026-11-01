//! Scalarized auto-exposure guest: HDR log-luminance, histogram, and EV adaptation.
//!
//! The three compute exports preserve the production pass results without
//! workgroup synchronization: one lane writes each pixel's log luminance,
//! one lane owns each histogram bin and scans that scratch array, and one
//! lane reduces the histogram. HDR texels arrive in a storage-buffer mirror;
//! this keeps the experiment inside the current `wasm2gpu` ABI, which has no
//! `textureLoad`, atomics, workgroup memory, or barriers.

#[cfg(target_arch = "wasm32")]
use ornis_gpu_abi::storage_at_f32;
#[cfg(target_arch = "wasm32")]
use ornis_gpu_abi::storage_at_u32;
#[cfg(target_arch = "wasm32")]
use ornis_gpu_abi::storage_store_f32;
#[cfg(target_arch = "wasm32")]
use ornis_gpu_abi::storage_store_u32;
#[cfg(target_arch = "wasm32")]
use ornis_gpu_abi::uniform_array_f32;

mod math;

#[cfg(target_arch = "wasm32")]
use math::clamp_f32;
#[cfg(target_arch = "wasm32")]
use math::exp2_approx;
#[cfg(target_arch = "wasm32")]
use math::log2_approx;
#[cfg(target_arch = "wasm32")]
use math::log_luminance;
#[cfg(target_arch = "wasm32")]
use math::log_luminance_is_in_bin;
#[cfg(target_arch = "wasm32")]
use math::selected_channel;

/// Number of production histogram bins; bin zero is the below-window bucket.
#[cfg(target_arch = "wasm32")]
const HISTOGRAM_BINS: u32 = 256;
/// Eight packed RGBA lanes per texel: deferred then forward.
#[cfg(target_arch = "wasm32")]
const TEXEL_LANES: u32 = 8;
/// Storage binding with the packed deferred/forward HDR texels.
#[cfg(target_arch = "wasm32")]
const INPUT_BINDING: u32 = 0;
/// Storage binding with one log-luminance value per texel.
#[cfg(target_arch = "wasm32")]
const LOG_LUMINANCE_BINDING: u32 = 1;
/// Storage binding with the 256 production-style `u32` histogram counts.
#[cfg(target_arch = "wasm32")]
const HISTOGRAM_BINDING: u32 = 2;
/// Storage binding with the previous and adapted exposure multipliers.
#[cfg(target_arch = "wasm32")]
const EXPOSURE_BINDING: u32 = 6;
/// Storage binding holding pixel count and composite layer mode.
#[cfg(target_arch = "wasm32")]
const FRAME_BINDING: u32 = 3;
/// Uniform binding with bin-edge values, minimum log luminance, and inverse range.
#[cfg(target_arch = "wasm32")]
const HISTOGRAM_PARAMS_BINDING: u32 = 4;
/// Uniform binding with bin centres and adaptation settings.
#[cfg(target_arch = "wasm32")]
const ADAPT_PARAMS_BINDING: u32 = 5;
/// Histogram parameters contain edge values 0..=256, then minimum and inverse range.
#[cfg(target_arch = "wasm32")]
const HISTOGRAM_PARAMS_LEN: u32 = 259;
/// Adaptation parameters contain 256 centres followed by nine scalar settings.
#[cfg(target_arch = "wasm32")]
const ADAPT_PARAMS_LEN: u32 = 265;
#[cfg(target_arch = "wasm32")]
const MIN_LOG_LUMINANCE_INDEX: u32 = 257;
#[cfg(target_arch = "wasm32")]
const INV_LOG_RANGE_INDEX: u32 = 258;
#[cfg(target_arch = "wasm32")]
const ADAPT_LOW_CLIP_INDEX: u32 = 256;
#[cfg(target_arch = "wasm32")]
const ADAPT_HIGH_CLIP_INDEX: u32 = 257;
#[cfg(target_arch = "wasm32")]
const ADAPT_TARGET_EV_INDEX: u32 = 258;
#[cfg(target_arch = "wasm32")]
const ADAPT_MIN_EV_INDEX: u32 = 259;
#[cfg(target_arch = "wasm32")]
const ADAPT_MAX_EV_INDEX: u32 = 260;
#[cfg(target_arch = "wasm32")]
const ADAPT_BRIGHTEN_SPEED_INDEX: u32 = 261;
#[cfg(target_arch = "wasm32")]
const ADAPT_DARKEN_SPEED_INDEX: u32 = 262;
#[cfg(target_arch = "wasm32")]
const ADAPT_TRANSITION_DISTANCE_INDEX: u32 = 263;
#[cfg(target_arch = "wasm32")]
const ADAPT_DT_INDEX: u32 = 264;
#[cfg(target_arch = "wasm32")]
const LUMINANCE_FLOOR: f32 = 1e-6;

/// Wasm entry for per-pixel HDR log-luminance; writes `log_luminance[id]`.
///
/// Binding 0 stores eight packed `f32` lanes per pixel (deferred RGBA then
/// forward RGBA); binding 1 stores the resulting scalar log luminance. Frame
/// binding 3 contains pixel count at word 0 and layer mode at word 1.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn ae_log_luma(id: u32) {
    let pixel_count = storage_at_u32::<0, FRAME_BINDING>(0);
    if id < pixel_count {
        let mode = storage_at_u32::<0, FRAME_BINDING>(1);
        let base = id * TEXEL_LANES;
        let deferred_r = storage_at_f32::<0, INPUT_BINDING>(base);
        let deferred_g = storage_at_f32::<0, INPUT_BINDING>(base + 1);
        let deferred_b = storage_at_f32::<0, INPUT_BINDING>(base + 2);
        let forward_r = storage_at_f32::<0, INPUT_BINDING>(base + 4);
        let forward_g = storage_at_f32::<0, INPUT_BINDING>(base + 5);
        let forward_b = storage_at_f32::<0, INPUT_BINDING>(base + 6);
        let forward_alpha = storage_at_f32::<0, INPUT_BINDING>(base + 7);
        let red = selected_channel(mode, deferred_r, forward_r, forward_alpha);
        let green = selected_channel(mode, deferred_g, forward_g, forward_alpha);
        let blue = selected_channel(mode, deferred_b, forward_b, forward_alpha);
        storage_store_f32::<0, LOG_LUMINANCE_BINDING>(id, log_luminance(red, green, blue));
    }
}

/// Wasm entry for a histogram bin; each lane scans scratch and writes one count.
///
/// The uniform table stores float bin thresholds `0..=256` at indices
/// `0..=256`, then `min_log_lum` and `inv_log_range` at 257 and 258. This
/// interval test mirrors the production `u32(t * 254 + 1)` mapping without a
/// float-to-integer cast or cross-lane atomic update.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn ae_histogram(bin: u32) {
    if bin < HISTOGRAM_BINS {
        let pixel_count = storage_at_u32::<0, FRAME_BINDING>(0);
        let min_log_lum = uniform_array_f32::<0, HISTOGRAM_PARAMS_BINDING, HISTOGRAM_PARAMS_LEN>(
            MIN_LOG_LUMINANCE_INDEX,
        );
        let inv_log_range = uniform_array_f32::<0, HISTOGRAM_PARAMS_BINDING, HISTOGRAM_PARAMS_LEN>(
            INV_LOG_RANGE_INDEX,
        );
        let mut count = 0_u32;
        let mut pixel = 0_u32;
        while pixel < pixel_count {
            let log_lum = storage_at_f32::<0, LOG_LUMINANCE_BINDING>(pixel);
            if log_luminance_is_in_bin(log_lum, bin, min_log_lum, inv_log_range) {
                count += 1;
            }
            pixel += 1;
        }
        storage_store_u32::<0, HISTOGRAM_BINDING>(bin, count);
    }
}

/// Wasm entry for clipped-average reduction, smoothing, and histogram clear.
///
/// Binding 2 holds `u32` histogram bins; binding 6 holds previous exposure
/// at slot 0 and the adapted multiplier at slot 1. Only invocation zero works.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn ae_adapt(id: u32) {
    if id == 0 {
        adapt_histogram();
    }
}

#[cfg(target_arch = "wasm32")]
fn adapt_histogram() {
    let mut total = 0_u32;
    let mut bin = 1_u32;
    while bin < HISTOGRAM_BINS {
        total += storage_at_u32::<0, HISTOGRAM_BINDING>(bin);
        bin += 1;
    }

    let previous = storage_at_f32::<0, EXPOSURE_BINDING>(0);
    let mut adapted = previous;
    if 0 < total {
        adapted = adapted_exposure(total, previous);
    }
    clear_histogram();
    storage_store_f32::<0, EXPOSURE_BINDING>(0, adapted);
    storage_store_f32::<0, EXPOSURE_BINDING>(1, adapted);
}

#[cfg(target_arch = "wasm32")]
fn adapted_exposure(total: u32, previous: f32) -> f32 {
    let low_clip = adapt_param(ADAPT_LOW_CLIP_INDEX);
    let high_clip = adapt_param(ADAPT_HIGH_CLIP_INDEX);
    let total_f32 = total as f32;
    let low = clamp_f32(total_f32 * low_clip, 0.0, total_f32);
    let high = clamp_f32(total_f32 * high_clip, 0.0, total_f32);
    if !(low < high) {
        return previous;
    }
    let average_log_lum = clipped_log_average(low, high);
    let minimum_ev = adapt_param(ADAPT_MIN_EV_INDEX);
    let maximum_ev = adapt_param(ADAPT_MAX_EV_INDEX);
    let safe_previous = if previous < LUMINANCE_FLOOR {
        LUMINANCE_FLOOR
    } else {
        previous
    };
    let previous_ev = clamp_f32(log2_approx(safe_previous), minimum_ev, maximum_ev);
    let target_ev = clamp_f32(
        adapt_param(ADAPT_TARGET_EV_INDEX) - average_log_lum,
        minimum_ev,
        maximum_ev,
    );
    exp2_approx(smooth_ev(previous_ev, target_ev))
}

#[cfg(target_arch = "wasm32")]
fn clipped_log_average(low: f32, high: f32) -> f32 {
    let mut cumulative = 0_u32;
    let mut sum = 0.0_f32;
    let mut sum_weight = 0.0_f32;
    let mut bin = 1_u32;
    while bin < HISTOGRAM_BINS {
        let before = cumulative;
        cumulative += storage_at_u32::<0, HISTOGRAM_BINDING>(bin);
        let weight = clamp_f32(cumulative as f32, low, high)
            - clamp_f32(before as f32, low, high);
        sum += weight * adapt_param(bin);
        sum_weight += weight;
        bin += 1;
    }
    sum / sum_weight
}

#[cfg(target_arch = "wasm32")]
fn smooth_ev(previous_ev: f32, target_ev: f32) -> f32 {
    let delta = target_ev - previous_ev;
    let absolute_delta = if delta < 0.0 {
        previous_ev - target_ev
    } else {
        delta
    };
    let speed = if 0.0 < delta {
        adapt_param(ADAPT_BRIGHTEN_SPEED_INDEX)
    } else {
        adapt_param(ADAPT_DARKEN_SPEED_INDEX)
    };
    let max_step = speed * adapt_param(ADAPT_DT_INDEX);
    let transition_distance = adapt_param(ADAPT_TRANSITION_DISTANCE_INDEX);
    let linear_step = absolute_delta * max_step / transition_distance;
    let first_clamp = if linear_step < max_step {
        linear_step
    } else {
        max_step
    };
    let step = if absolute_delta < first_clamp {
        absolute_delta
    } else {
        first_clamp
    };
    if delta < 0.0 {
        previous_ev - step
    } else {
        previous_ev + step
    }
}

#[cfg(target_arch = "wasm32")]
fn clear_histogram() {
    let mut bin = 0_u32;
    while bin < HISTOGRAM_BINS {
        storage_store_u32::<0, HISTOGRAM_BINDING>(bin, 0);
        bin += 1;
    }
}

#[cfg(target_arch = "wasm32")]
fn adapt_param(index: u32) -> f32 {
    uniform_array_f32::<0, ADAPT_PARAMS_BINDING, ADAPT_PARAMS_LEN>(index)
}
