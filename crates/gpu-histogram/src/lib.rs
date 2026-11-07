//! Log-luminance histogram for the stable Rust to WGSL path.
//!
//! One workgroup meters the same deferred/forward mix the tone-mapper
//! composites. Bindings match the host pass: deferred texture `0`, forward
//! texture `1`, atomic histogram `2`, `HistogramParams` uniform `3`.
//! The translator only has scalar imports, so each texel is three lanes
//! and the 256-bin flush is a stride of the 8×8 workgroup.

#![no_std]
#![allow(clippy::all)]
#![allow(unsafe_op_in_unsafe_fn)]

unsafe extern "C" {
    fn global_invocation_lane(axis: u32) -> u32;
    fn local_invocation_index() -> u32;
    fn texture_load_2d_f32x4(group: u32, binding: u32, x: u32, y: u32, lane: u32) -> f32;
    fn texture_width_2d_f32x4(group: u32, binding: u32) -> i32;
    fn texture_height_2d_f32x4(group: u32, binding: u32) -> i32;
    fn workgroup_atomic_store_u32(index: u32, value: u32);
    fn workgroup_atomic_load_u32(index: u32) -> u32;
    fn workgroup_atomic_add_u32(index: u32, value: u32) -> u32;
    fn workgroup_barrier();
    fn storage_atomic_add_u32(group: u32, binding: u32, index: u32, value: u32) -> u32;
    fn gpu_dot3(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32;
    fn gpu_log2(value: f32) -> f32;
    fn gpu_abs(value: f32) -> f32;
    fn gpu_clamp(value: f32, low: f32, high: f32) -> f32;
    fn uniform_field_f32(group: u32, binding: u32, sid: u32, field: u32) -> f32;
    fn uniform_field_u32(group: u32, binding: u32, sid: u32, field: u32) -> u32;
}

#[used]
#[cfg_attr(target_arch = "wasm32", unsafe(link_section = "ornis_gpu.shader_meta"))]
#[unsafe(export_name = "ornis_gpu_shader_meta")]
static ORNIS_GPU_SHADER_META: [u8; 44] = *b"entry=cs_main;workgroup=8,8,1;uniform=0:1112";

const BINS: u32 = 256;
const THREADS: u32 = 64;
const FLOOR: f32 = 1.0e-6;
const INF_GUARD: f32 = 3.4e38;
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

#[inline(always)]
// qual:allow(magic_numbers) reason: "Rec.709 luma and the host non-finite floor"
unsafe fn log_luminance(red: f32, green: f32, blue: f32) -> f32 {
    let luma = gpu_dot3(red, green, blue, LUMA_R, LUMA_G, LUMA_B);
    let non_finite = luma != luma || gpu_abs(luma) > INF_GUARD;
    let safe = if non_finite {
        FLOOR
    } else {
        f32::max(luma, FLOOR)
    };
    gpu_log2(safe)
}

#[inline(always)]
// qual:allow(magic_numbers) reason: "HistogramParams field order: min_log, inv_range, bin_count"
unsafe fn histogram_bin(log_lum: f32) -> u32 {
    let min_log = uniform_field_f32(0, 3, 0, 0);
    let inv_range = uniform_field_f32(0, 3, 0, 1);
    let bin_count = uniform_field_f32(0, 3, 0, 2);
    if log_lum < min_log {
        0
    } else {
        let t = gpu_clamp((log_lum - min_log) * inv_range, 0.0, 1.0);
        (t * (bin_count - 2.0) + 1.0) as u32
    }
}

#[inline(always)]
// qual:allow(magic_numbers) reason: "mode 0 deferred, 1 forward premultiplied, 2 additive mix"
unsafe fn mixed_lane(deferred: f32, forward: f32, alpha: f32, mode: u32) -> f32 {
    let premultiplied = forward * alpha;
    // Two single comparisons, not an if/else chain: rustc turns the chain
    // into `switch`, which the translator does not lower.
    let keep_deferred = if mode == 1 { 0.0 } else { 1.0 };
    let add_forward = if mode == 0 { 0.0 } else { 1.0 };
    deferred * keep_deferred + premultiplied * add_forward
}

#[inline(always)]
unsafe fn clear_shared(lid: u32) {
    let mut slot = lid;
    while slot < BINS {
        workgroup_atomic_store_u32(slot, 0);
        slot += THREADS;
    }
}

#[inline(always)]
unsafe fn flush_shared(lid: u32) {
    let mut slot = lid;
    while slot < BINS {
        let count = workgroup_atomic_load_u32(slot);
        if count > 0 {
            let _added = storage_atomic_add_u32(0, 2, slot, count);
        }
        slot += THREADS;
    }
}

#[inline(always)]
// qual:allow(magic_numbers, complexity) reason: "one texel of the host histogram; bindings are the pass layout"
unsafe fn meter_texel(x: u32, y: u32) {
    let width = texture_width_2d_f32x4(0, 0) as u32;
    let height = texture_height_2d_f32x4(0, 0) as u32;
    if x < width && y < height {
        let deferred_r = texture_load_2d_f32x4(0, 0, x, y, 0);
        let deferred_g = texture_load_2d_f32x4(0, 0, x, y, 1);
        let deferred_b = texture_load_2d_f32x4(0, 0, x, y, 2);
        let forward_r = texture_load_2d_f32x4(0, 1, x, y, 0);
        let forward_g = texture_load_2d_f32x4(0, 1, x, y, 1);
        let forward_b = texture_load_2d_f32x4(0, 1, x, y, 2);
        let forward_a = texture_load_2d_f32x4(0, 1, x, y, 3);
        let mode = uniform_field_u32(0, 3, 0, 3);
        let red = mixed_lane(deferred_r, forward_r, forward_a, mode);
        let green = mixed_lane(deferred_g, forward_g, forward_a, mode);
        let blue = mixed_lane(deferred_b, forward_b, forward_a, mode);
        let bin = histogram_bin(log_luminance(red, green, blue));
        let _previous = workgroup_atomic_add_u32(bin, 1);
    }
}

#[unsafe(no_mangle)]
// qual:allow(magic_numbers) reason: "workgroup axes 0 and 1 are the texel coordinate"
pub unsafe extern "C" fn cs_main() {
    // SAFETY: these calls are the translator's GPU imports, not host memory.
    unsafe {
        let lid = local_invocation_index();
        clear_shared(lid);
        workgroup_barrier();
        meter_texel(global_invocation_lane(0), global_invocation_lane(1));
        workgroup_barrier();
        flush_shared(lid);
    }
}
