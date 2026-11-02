//! A/B of the CPU skin (`ornis_animation::skin_vertices`) and the wasm2gpu guest.
//!
//! Release build, five warmups, then thirty runs per scene. Reproduce with:
//!
//! ```text
//! cargo bench -p ornis-render --bench wasm2gpu_skin_ab --features test-util
//! ```
//!
//! The report names the adapter; software-adapter timings are CPU Vulkan costs.

fn main() {
    print!("{}", ornis_render::skinning::wasm2gpu::run_skin_ab());
}
