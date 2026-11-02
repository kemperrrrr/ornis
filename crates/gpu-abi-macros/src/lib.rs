//! Entry-point attribute macros for the toy wasm2gpu GPU ABI.
//!
//! `#[shader::compute]` / `#[shader::vertex]` / `#[shader::fragment]` declare
//! wasm2gpu entry points and record their stage and workgroup size in the
//! `ornis_gpu.entries` wasm custom section. `ornis-gpu-abi` re-exports them as
//! `ornis_gpu_abi::shader`. This crate sits in the foundation layer and
//! depends only on `syn`, `quote`, and `proc-macro2`.
#![warn(missing_docs)]

use proc_macro::TokenStream;

mod shader_entry;
mod shader_entry_sig;

/// Declare a compute entry point for the wasm2gpu path.
///
/// Adds `#[unsafe(no_mangle)] pub extern "C"` to the annotated function and
/// records its stage and `workgroup_size` in the `ornis_gpu.entries` custom
/// section. The function takes exactly one `u32` invocation id and returns
/// `()`. The translator reads the workgroup size from that section.
#[proc_macro_attribute]
pub fn compute(attr: TokenStream, item: TokenStream) -> TokenStream {
    shader_entry::compute(attr, item)
}

/// Declare a vertex entry point for the wasm2gpu path.
///
/// Adds the export wrapper and entry metadata. Parameters and the return
/// value must be `f32`, `i32`, `u32`, or `()`.
#[proc_macro_attribute]
pub fn vertex(attr: TokenStream, item: TokenStream) -> TokenStream {
    shader_entry::vertex(attr, item)
}

/// Declare a fragment entry point for the wasm2gpu path.
///
/// Adds the export wrapper and entry metadata. Parameters and the return
/// value must be `f32`, `i32`, `u32`, or `()`.
#[proc_macro_attribute]
pub fn fragment(attr: TokenStream, item: TokenStream) -> TokenStream {
    shader_entry::fragment(attr, item)
}
