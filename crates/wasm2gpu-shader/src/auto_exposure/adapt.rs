//! Histogram reduction orchestration and reset.

mod exposure;

use ornis_gpu_abi::storage_at_f32;
use ornis_gpu_abi::storage_at_u32;
use ornis_gpu_abi::storage_store_f32;
use ornis_gpu_abi::storage_store_u32;

use super::EXPOSURE_BINDING;
use super::HISTOGRAM_BINDING;
use super::HISTOGRAM_BINS;

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
    let adapted = if 0 < total {
        exposure::adapted_exposure(total, previous)
    } else {
        previous
    };
    clear_histogram();
    storage_store_f32::<0, EXPOSURE_BINDING>(0, adapted);
    storage_store_f32::<0, EXPOSURE_BINDING>(1, adapted);
}

fn clear_histogram() {
    let mut bin = 0_u32;
    while bin < HISTOGRAM_BINS {
        storage_store_u32::<0, HISTOGRAM_BINDING>(bin, 0);
        bin += 1;
    }
}
