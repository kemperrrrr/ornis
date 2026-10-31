//! A/B of the wasm2gpu `cull` kernel against [`crate::CULL_COMPUTE_WGSL`].
//!
//! Release build, warmup, then thirty dispatches per shader per scene.
//! Reproduce with:
//!
//! ```text
//! cargo bench -p ornis-render --bench wasm2gpu_cull_ab --features test-util
//! ```
//!
//! The header names the adapter. A software device (lavapipe / llvmpipe)
//! is a CPU Vulkan implementation; its times are not discrete-GPU costs.

fn main() {
    print!("{}", ornis_render::gpu_culling::shader_ab::run_cull_ab());
}
