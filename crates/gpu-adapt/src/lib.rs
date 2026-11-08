//! Histogram reduction for the stable Rust to WGSL path.
//!
//! One thread snapshots the 256-bin histogram, clears it for the next
//! frame, and writes the smoothed exposure. Bindings match the host
//! pass: histogram storage `0`, `AdaptParams` uniform `1`, exposure
//! storage `2`. The kernel is a safe `fn` on the shader SDK.

#![no_std]
#![allow(clippy::all)]

use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::ReadWrite;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::TYPE_TAG_F32;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::WorkgroupU32;
use ornis_gpu_shader::abs;
use ornis_gpu_shader::clamp;
use ornis_gpu_shader::exp2;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::log2;
use ornis_gpu_shader::sign;

/// Host `AdaptParams`: twelve `f32`s, the last one pad.
pub struct AdaptParams;

impl GpuStruct for AdaptParams {
    /// The adapt uniform is struct 0.
    const ID: u32 = 0;
    /// `min_log_lum` through `dt`, then the pad word.
    const FIELD_TAGS: &'static [u8] = &[TYPE_TAG_F32; 12];
}

const BINS: u32 = 256;
const FLOOR: f32 = 1.0e-6;
const MIN_TRANSITION: f32 = 1.0e-3;

#[inline(always)]
// qual:allow(complexity, magic_numbers) reason: "bin centre is (index - 0.5) / (bins - 2), matching the host reduction"
fn bin_log_luminance(params: Uniform<AdaptParams, 0, 1>, index: u32) -> f32 {
    let t = clamp((index as f32 - 0.5) / (BINS as f32 - 2.0), 0.0, 1.0);
    params.field_f32::<0>() + t * params.field_f32::<1>()
}

#[inline(always)]
fn snapshot(histogram: Storage<u32, 0, 0, ReadWrite>) -> u32 {
    let mut total = 0u32;
    let mut index = 0u32;
    while index < BINS {
        let count = histogram.load(index);
        WorkgroupU32::store(index, count);
        histogram.store(index, 0);
        if index > 0 {
            total += count;
        }
        index += 1;
    }
    total
}

#[inline(always)]
// qual:allow(complexity, magic_numbers) reason: "the reduction starts at bin 1; bin 0 is the below-range reserve"
fn reduce(total: u32, params: Uniform<AdaptParams, 0, 1>, exposure: Storage<f32, 0, 2, ReadWrite>) {
    let low = (total as f32) * params.field_f32::<2>();
    let high = (total as f32) * params.field_f32::<3>();
    let mut cumulative = 0u32;
    let mut sum = 0.0f32;
    let mut sum_weight = 0.0f32;
    let mut index = 1u32;
    while index < BINS {
        let count = WorkgroupU32::load(index);
        let before = cumulative;
        cumulative += count;
        let weight = clamp(cumulative as f32, low, high) - clamp(before as f32, low, high);
        sum += weight * bin_log_luminance(params, index);
        sum_weight += weight;
        index += 1;
    }
    if sum_weight > 0.0 {
        exposure.store(0, exp2(adapted_ev(params, exposure, sum / sum_weight)));
    }
}

#[inline(always)]
// qual:allow(complexity, magic_numbers) reason: "AdaptParams fields 4..10 are the EV window and the step"
fn adapted_ev(
    params: Uniform<AdaptParams, 0, 1>,
    exposure: Storage<f32, 0, 2, ReadWrite>,
    avg_log_lum: f32,
) -> f32 {
    let previous = exposure.load(0);
    let min_ev = params.field_f32::<5>();
    let max_ev = params.field_f32::<6>();
    let prev_ev = clamp(log2(f32::max(previous, FLOOR)), min_ev, max_ev);
    let target_ev = clamp(params.field_f32::<4>() - avg_log_lum, min_ev, max_ev);
    let delta = target_ev - prev_ev;
    // Both speeds are constant field indices. Selecting the index would be a
    // dynamic uniform access, which the translator rejects.
    let brighten = params.field_f32::<7>();
    let darken = params.field_f32::<8>();
    let speed = if delta > 0.0 { brighten } else { darken };
    let max_step = speed * params.field_f32::<10>();
    let distance = f32::max(params.field_f32::<9>(), MIN_TRANSITION);
    let step = f32::min(max_step, abs(delta) * max_step / distance);
    let clamped = f32::min(step, abs(delta));
    let moved = prev_ev + sign(delta) * clamped;
    if delta == 0.0 { prev_ev } else { moved }
}

#[gpu_entry(compute(threads(1)))]
pub fn cs_main(
    #[gpu(storage_buffer, group = 0, binding = 0)] histogram: Storage<u32, ReadWrite>,
    #[gpu(uniform, group = 0, binding = 1)] params: Uniform<AdaptParams>,
    #[gpu(storage_buffer, group = 0, binding = 2)] exposure: Storage<f32, ReadWrite>,
) {
    let total = snapshot(histogram);
    if total != 0 {
        reduce(total, params, exposure);
    }
}
