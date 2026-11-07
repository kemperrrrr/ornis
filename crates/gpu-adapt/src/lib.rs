//! Histogram reduction for the stable Rust to WGSL path.
//!
//! One thread snapshots the 256-bin histogram, clears it for the next
//! frame, and writes the smoothed exposure. Bindings match the host
//! pass: histogram storage `0`, `AdaptParams` uniform `1`, exposure
//! storage `2`. The translator only has scalar imports, so the exposure
//! block is the `f32` at index 0.

#![no_std]
#![allow(clippy::all)]
#![allow(unsafe_op_in_unsafe_fn)]

unsafe extern "C" {
    fn storage_read_u32(group: u32, binding: u32, index: u32) -> u32;
    fn storage_write_u32(group: u32, binding: u32, index: u32, value: u32);
    fn storage_read_f32(group: u32, binding: u32, index: u32) -> f32;
    fn storage_write_f32(group: u32, binding: u32, index: u32, value: f32);
    fn workgroup_load_u32(index: u32) -> u32;
    fn workgroup_store_u32(index: u32, value: u32);
    fn uniform_field_f32(group: u32, binding: u32, sid: u32, field: u32) -> f32;
    fn gpu_log2(value: f32) -> f32;
    fn gpu_exp2(value: f32) -> f32;
    fn gpu_clamp(value: f32, low: f32, high: f32) -> f32;
    fn gpu_abs(value: f32) -> f32;
    fn gpu_sign(value: f32) -> f32;
}

#[used]
#[cfg_attr(target_arch = "wasm32", unsafe(link_section = "ornis_gpu.shader_meta"))]
#[unsafe(export_name = "ornis_gpu_shader_meta")]
static ORNIS_GPU_SHADER_META: [u8; 52] = *b"entry=cs_main;workgroup=1,1,1;uniform=0:111111111111";

const BINS: u32 = 256;
const FLOOR: f32 = 1.0e-6;
const MIN_TRANSITION: f32 = 1.0e-3;

#[inline(always)]
// qual:allow(magic_numbers) reason: "AdaptParams field order matches the host uniform"
unsafe fn param(field: u32) -> f32 {
    uniform_field_f32(0, 1, 0, field)
}

#[inline(always)]
// qual:allow(magic_numbers) reason: "bin centre is (index - 0.5) / (bins - 2), matching adapt_wgsl"
unsafe fn bin_log_luminance(index: u32) -> f32 {
    let t = gpu_clamp((index as f32 - 0.5) / (BINS as f32 - 2.0), 0.0, 1.0);
    param(0) + t * param(1)
}

#[inline(always)]
unsafe fn snapshot() -> u32 {
    let mut total = 0u32;
    let mut index = 0u32;
    while index < BINS {
        let count = storage_read_u32(0, 0, index);
        workgroup_store_u32(index, count);
        storage_write_u32(0, 0, index, 0);
        if index > 0 {
            total += count;
        }
        index += 1;
    }
    total
}

#[inline(always)]
// qual:allow(magic_numbers) reason: "the reduction starts at bin 1; bin 0 is the below-range reserve"
unsafe fn reduce(total: u32) {
    let low = (total as f32) * param(2);
    let high = (total as f32) * param(3);
    let mut cumulative = 0u32;
    let mut sum = 0.0f32;
    let mut sum_weight = 0.0f32;
    let mut index = 1u32;
    while index < BINS {
        let count = workgroup_load_u32(index);
        let before = cumulative;
        cumulative += count;
        let weight = gpu_clamp(cumulative as f32, low, high) - gpu_clamp(before as f32, low, high);
        sum += weight * bin_log_luminance(index);
        sum_weight += weight;
        index += 1;
    }
    if sum_weight > 0.0 {
        storage_write_f32(0, 2, 0, gpu_exp2(adapted_ev(sum / sum_weight)));
    }
}

#[inline(always)]
// qual:allow(magic_numbers) reason: "AdaptParams fields 4..10 are the EV window and the step"
unsafe fn adapted_ev(avg_log_lum: f32) -> f32 {
    let exposure = storage_read_f32(0, 2, 0);
    let min_ev = param(5);
    let max_ev = param(6);
    let prev_ev = gpu_clamp(gpu_log2(f32::max(exposure, FLOOR)), min_ev, max_ev);
    let target_ev = gpu_clamp(param(4) - avg_log_lum, min_ev, max_ev);
    let delta = target_ev - prev_ev;
    // Both fields stay constant indices. A `select` of the index becomes a
    // dynamic uniform access, which the translator rejects.
    let brighten = param(7);
    let darken = param(8);
    let speed = if delta > 0.0 { brighten } else { darken };
    let max_step = speed * param(10);
    let distance = f32::max(param(9), MIN_TRANSITION);
    let step = f32::min(max_step, gpu_abs(delta) * max_step / distance);
    let clamped = f32::min(step, gpu_abs(delta));
    let moved = prev_ev + gpu_sign(delta) * clamped;
    if delta == 0.0 { prev_ev } else { moved }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cs_main() {
    // SAFETY: these calls are the translator's GPU imports, not host memory.
    unsafe {
        let total = snapshot();
        if total != 0 {
            reduce(total);
        }
    }
}
