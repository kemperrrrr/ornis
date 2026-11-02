//! Build the guest once, then lift and translate the `skin` compute export.

use std::time::Duration;
use std::time::Instant;

use ornis_wasm2gpu_testkit::compile_wasm;
use ornis_wasm2gpu_testkit::compute_lift::lowered_kernel_from;

/// Translated skinning entry point and the nested guest build duration.
pub(crate) struct Lowered {
    /// WGSL of the `skin` compute entry (`cs_main` after the lift).
    pub(crate) wgsl: String,
    /// Time spent building the guest and translating the entry.
    pub(crate) elapsed: Duration,
}

/// Compiles the Rust guest and lowers its `skin` export through the shared
/// compute lifter, pinning the output storage binding of the guest ABI.
pub(crate) fn lower_guest() -> Lowered {
    let started = Instant::now();
    let wasm = compile_wasm();
    let wgsl = lowered_kernel_from(&wasm, "skin", "storage_0_4_");
    Lowered {
        wgsl,
        elapsed: started.elapsed(),
    }
}
