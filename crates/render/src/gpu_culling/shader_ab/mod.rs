//! Measurements for the wasm2gpu `cull` kernel against the shipped
//! [`super::CULL_COMPUTE_WGSL`] pass (issue #201).
//!
//! The frame path does not include this module. Unit tests cover the
//! scene packing, the quartile summary, and the naga IR walk. The GPU
//! timings live behind the `test-util` feature and run from the
//! `wasm2gpu_cull_ab` bench.

mod ir;
mod scene;
mod stats;

#[cfg(feature = "test-util")]
mod gpu;

#[cfg(feature = "test-util")]
pub use gpu::run_cull_ab;
