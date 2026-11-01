//! Build the guest once, then lift and translate each compute export.

use std::time::Duration;
use std::time::Instant;

use ornis_wasm2gpu_testkit::compile_wasm;
use ornis_wasm2gpu_testkit::compute_lift::lowered_kernel_from;

/// Translated compute entry points and the nested guest build duration.
pub(crate) struct Lowered {
    /// WGSL for the per-pixel log-luminance entry.
    pub(crate) log_luma: String,
    /// WGSL for the per-bin histogram entry.
    pub(crate) histogram: String,
    /// WGSL for the clipped-average and adaptation entry.
    pub(crate) adapt: String,
    /// Time spent building the guest and translating all three entries.
    pub(crate) elapsed: Duration,
}

/// Compiles the Rust guest once and lowers its per-pixel, histogram, and
/// reduction exports through the shared compute lifter.
pub(crate) fn lower_guest() -> Lowered {
    let started = Instant::now();
    let wasm = compile_wasm();
    let log_luma = lowered_kernel_from(&wasm, "ae_log_luma", "storage_0_1_");
    let histogram = lowered_kernel_from(&wasm, "ae_histogram", "storage_0_2_");
    let adapt = lowered_kernel_from(&wasm, "ae_adapt", "storage_0_2_");
    assert!(
        adapt.contains("storage_0_6_"),
        "guest exposure binding missing: {adapt}"
    );
    Lowered {
        log_luma,
        histogram,
        adapt,
        elapsed: started.elapsed(),
    }
}
