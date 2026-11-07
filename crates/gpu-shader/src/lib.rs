//! Stable `no_std` shader SDK for the llvm2gpu path.
//!
//! The crate follows rust-gpu's user-facing shape where stable Rust
//! allows it: `#[gpu_entry(compute(threads(64)))]` declares the entry and
//! its metadata, and typed resource wrappers (`Storage`/`Uniform`) will
//! replace the raw `ornis_gpu` imports as the SDK grows. Today the entry
//! attribute records entry/workgroup/uniform metadata in the shader
//! crate; the typed wrappers land next.

#![no_std]
#![warn(missing_docs)]

pub use ornis_gpu_shader_macros::gpu_entry;
