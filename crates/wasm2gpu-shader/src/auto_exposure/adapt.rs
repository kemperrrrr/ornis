//! Clipped-histogram reduction, EV smoothing, and histogram reset.

use ornis_gpu_abi::storage_at_f32;
use ornis_gpu_abi::storage_at_u32;
use ornis_gpu_abi::storage_store_f32;
use ornis_gpu_abi::storage_store_u32;
use ornis_gpu_abi::uniform_array_f32;

use super::math::clamp_f32;
use super::math::exp2_approx;
use super::math::log2_approx;
use super::ADAPT_BRIGHTEN_SPEED_INDEX;
use super::ADAPT_DARKEN_SPEED_INDEX;
use super::ADAPT_DT_INDEX;
use super::ADAPT_HIGH_CLIP_INDEX;
use super::ADAPT_LOW_CLIP_INDEX;
use super::ADAPT_MAX_EV_INDEX;
use super::ADAPT_MIN_EV_INDEX;
use super::ADAPT_PARAMS_BINDING;
use super::ADAPT_PARAMS_LEN;
use super::ADAPT_TARGET_EV_INDEX;
use super::ADAPT_TRANSITION_DISTANCE_INDEX;
use super::EXPOSURE_BINDING;
use super::HISTOGRAM_BINDING;
use super::HISTOGRAM_BINS;
use super::LUMINANCE_FLOOR;

/// Wasm entry for clipped-average reduction, smoothing, and histogram clear.
///
/// Binding 2 holds `u32` histogram bins; binding 6 holds previous exposure
/// at slot 0 and the adapted multiplier at slot 1. Only invocation zero works.
#[unsafe(no_mangle)]
pub extern "C" fn ae_adapt(id: u32) {
    if id == 0 {
        adapt_histogram();
    }
}

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

fn clipped_log_average(low: f32, high: f32) -> f32 {
    let mut cumulative = 0_u32;
    let mut sum = 0.0_f32;
    let mut sum_weight = 0.0_f32;
    let mut bin = 1_u32;
    while bin < HISTOGRAM_BINS {
        let before = cumulative;
        cumulative += storage_at_u32::<0, HISTOGRAM_BINDING>(bin);
        let weight = clamp_f32(cumulative as f32, low, high) - clamp_f32(before as f32, low, high);
        sum += weight * adapt_param(bin);
        sum_weight += weight;
        bin += 1;
    }
    sum / sum_weight
}

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

fn clear_histogram() {
    let mut bin = 0_u32;
    while bin < HISTOGRAM_BINS {
        storage_store_u32::<0, HISTOGRAM_BINDING>(bin, 0);
        bin += 1;
    }
}

fn adapt_param(index: u32) -> f32 {
    uniform_array_f32::<0, ADAPT_PARAMS_BINDING, ADAPT_PARAMS_LEN>(index)
}
