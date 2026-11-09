//! Log-luminance histogram for the stable Rust to WGSL path.
//!
//! One workgroup meters the same deferred/forward mix the tone-mapper
//! composites. Bindings match the host pass: deferred texture `0`, forward
//! texture `1`, atomic histogram `2`, `HistogramParams` uniform `3`.
//! The kernel is a safe `fn` on the shader SDK. It does not declare
//! `extern "C"`: texture loads, workgroup atomics, and the barrier are SDK
//! methods, and those methods inline to the translator's imports.

#![no_std]

use ornis_gpu_shader::AtomicU32;
use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::ReadWrite;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::TYPE_TAG_F32;
use ornis_gpu_shader::TYPE_TAG_U32;
use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::WorkgroupAtomicU32;
use ornis_gpu_shader::abs;
use ornis_gpu_shader::clamp;
use ornis_gpu_shader::dot3;
use ornis_gpu_shader::global_invocation_lane;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::local_invocation_index;
use ornis_gpu_shader::log2;
use ornis_gpu_shader::workgroup_barrier;

/// Host `HistogramParams`: `min_log_lum`, `inv_log_range`, `bin_count`, `mode`.
pub struct HistogramParams;

impl GpuStruct for HistogramParams {
    /// The histogram uniform is struct 0.
    const ID: u32 = 0;
    /// Three `f32`s and the `mode` `u32`.
    const FIELD_TAGS: &'static [u8] = &[TYPE_TAG_F32, TYPE_TAG_F32, TYPE_TAG_F32, TYPE_TAG_U32];
}

const BINS: u32 = 256;
const THREADS: u32 = 64;
const FLOOR: f32 = 1.0e-6;
const INF_GUARD: f32 = 3.4e38;
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

#[inline(always)]
// NaN self-comparison, not `is_nan()`: the translator lowers this to `fcmp`
// and the shader SDK has no `is_nan` yet.
#[allow(clippy::eq_op)]
fn log_luminance(red: f32, green: f32, blue: f32) -> f32 {
    let luma = dot3(red, green, blue, LUMA_R, LUMA_G, LUMA_B);
    let non_finite = luma != luma || abs(luma) > INF_GUARD;
    let safe = if non_finite {
        FLOOR
    } else {
        f32::max(luma, FLOOR)
    };
    log2(safe)
}

#[inline(always)]
fn histogram_bin(params: Uniform<HistogramParams, 0, 3>, log_lum: f32) -> u32 {
    let min_log = params.field_f32::<0>();
    let inv_range = params.field_f32::<1>();
    let bin_count = params.field_f32::<2>();
    if log_lum < min_log {
        0
    } else {
        let t = clamp((log_lum - min_log) * inv_range, 0.0, 1.0);
        (t * (bin_count - 2.0) + 1.0) as u32
    }
}

#[inline(always)]
fn mixed_lane(deferred: f32, forward: f32, alpha: f32, mode: u32) -> f32 {
    let premultiplied = forward * alpha;
    // Select the layer, never multiply-mask it: `NaN * 0.0` is NaN, so a
    // non-finite texel in the unselected HDR target would poison the chosen
    // one (master's shader only read the selected layer). Two single
    // comparisons, not an if/else chain: rustc turns the chain into
    // `switch`, which the translator does not lower.
    let mut mixed = deferred;
    if mode == 1 {
        mixed = premultiplied;
    }
    if mode == 2 {
        mixed = deferred + premultiplied;
    }
    mixed
}

#[inline(always)]
fn clear_shared(lid: u32) {
    let mut slot = lid;
    while slot < BINS {
        WorkgroupAtomicU32::store(slot, 0);
        slot += THREADS;
    }
}

#[inline(always)]
fn flush_shared(lid: u32, histogram: Storage<AtomicU32, 0, 2, ReadWrite>) {
    let mut slot = lid;
    while slot < BINS {
        let count = WorkgroupAtomicU32::load(slot);
        if count > 0 {
            let _added = histogram.atomic_add(slot, count);
        }
        slot += THREADS;
    }
}

#[inline(always)]
// qual:allow(complexity, magic_numbers) reason: "one texel of the host histogram; bindings are the pass layout"
fn meter_texel(
    x: u32,
    y: u32,
    deferred: Texture2d<0, 0>,
    forward: Texture2d<0, 1>,
    params: Uniform<HistogramParams, 0, 3>,
) {
    let width = deferred.width() as u32;
    let height = deferred.height() as u32;
    if x < width && y < height {
        let deferred_r = deferred.load::<0>(x, y);
        let deferred_g = deferred.load::<1>(x, y);
        let deferred_b = deferred.load::<2>(x, y);
        let forward_r = forward.load::<0>(x, y);
        let forward_g = forward.load::<1>(x, y);
        let forward_b = forward.load::<2>(x, y);
        let forward_a = forward.load::<3>(x, y);
        let mode = params.field_u32::<3>();
        let red = mixed_lane(deferred_r, forward_r, forward_a, mode);
        let green = mixed_lane(deferred_g, forward_g, forward_a, mode);
        let blue = mixed_lane(deferred_b, forward_b, forward_a, mode);
        let bin = histogram_bin(params, log_luminance(red, green, blue));
        let _previous = WorkgroupAtomicU32::add(bin, 1);
    }
}

#[gpu_entry(compute(threads(8, 8)))]
pub fn cs_main(
    #[gpu(texture_2d, group = 0, binding = 0)] deferred: Texture2d,
    #[gpu(texture_2d, group = 0, binding = 1)] forward: Texture2d,
    #[gpu(storage_buffer, group = 0, binding = 2)] histogram: Storage<AtomicU32, ReadWrite>,
    #[gpu(uniform, group = 0, binding = 3)] params: Uniform<HistogramParams>,
) {
    let lid = local_invocation_index();
    clear_shared(lid);
    workgroup_barrier();
    meter_texel(
        global_invocation_lane::<0>(),
        global_invocation_lane::<1>(),
        deferred,
        forward,
        params,
    );
    workgroup_barrier();
    flush_shared(lid, histogram);
}
