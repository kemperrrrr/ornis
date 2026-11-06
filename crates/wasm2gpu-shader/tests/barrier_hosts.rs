//! Host-side translate of the real compute guests.
//!
//! No GPU adapter. This checks that barrier uniformity accepts the kernels
//! the renderer lowers: skinning, auto-exposure, and cull.

use ornis_wasm2gpu::translate;
use ornis_wasm2gpu_testkit::compile_wasm;
use ornis_wasm2gpu_testkit::compute_lift;

#[test]
fn skin_auto_exposure_and_cull_pass_barrier_uniformity() {
    let wasm = compile_wasm();
    for export in ["skin", "ae_log_luma", "ae_histogram", "ae_adapt", "cull"] {
        let module = compute_lift::lift(&wasm, export);
        translate(&module).unwrap_or_else(|err| panic!("{export} barrier check: {err}"));
    }
}
