//! Production-style floating-point histogram bin assignment.

use ornis_gpu_abi::storage_at_f32;
use ornis_gpu_abi::storage_at_u32;
use ornis_gpu_abi::storage_store_u32;
use ornis_gpu_abi::uniform_array_f32;

use super::FRAME_BINDING;
use super::HISTOGRAM_BINDING;
use super::HISTOGRAM_BINS;
use super::HISTOGRAM_PARAMS_BINDING;
use super::HISTOGRAM_PARAMS_LEN;
use super::INV_LOG_RANGE_INDEX;
use super::LOG_LUMINANCE_BINDING;
use super::MIN_LOG_LUMINANCE_INDEX;
use super::math::log_luminance_is_in_bin;

/// Wasm entry for a histogram bin; each lane scans scratch and writes one count.
///
/// The uniform table stores float bin thresholds `0..=256` at indices
/// `0..=256`, then `min_log_lum` and `inv_log_range` at 257 and 258. This
/// interval test mirrors the production `u32(t * 254 + 1)` mapping without a
/// float-to-integer cast or cross-lane atomic update.
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
