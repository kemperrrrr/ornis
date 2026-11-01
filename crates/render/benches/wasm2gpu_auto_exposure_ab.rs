//! A/B of the production auto-exposure passes and the wasm2gpu guest.
//!
//! Release build, five warmups, then thirty submissions per shader and size.
//! Reproduce with:
//!
//! ```text
//! cargo bench -p ornis-render --bench wasm2gpu_auto_exposure_ab --features test-util
//! ```
//!
//! The report labels software adapters; those timings are CPU Vulkan costs.

fn main() {
    print!(
        "{}",
        ornis_render::auto_exposure::wasm2gpu::run_auto_exposure_ab()
    );
}
