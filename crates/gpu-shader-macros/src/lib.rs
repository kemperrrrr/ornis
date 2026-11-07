//! Stable shader entry attribute for the llvm2gpu path.
//!
//! [`gpu_entry`] keeps the annotated function, strips its resource parameters,
//! and emits a `#[used]` metadata record in a known LLVM global, so the offline
//! builder reads the entry name, workgroup size, and uniform field tags from
//! the shader crate itself instead of CLI flags.
//!
//! The attribute is the stable stand-in for rust-gpu's `#[spirv(…)]`: the same
//! `compute(threads(N))` shape, `global_invocation_id` as the one plain
//! argument, and resources bound by their buffer attributes.

mod entry;
mod param;
mod tags;

use proc_macro::TokenStream;

/// Declare a stable GPU entry point.
///
/// Syntax follows rust-gpu's `#[spirv(compute(threads(N)))]`:
///
/// ```ignore
/// #[gpu_entry(compute(threads(64)))]
/// #[unsafe(no_mangle)]
/// pub extern "C" fn cs_main(
///     id: u32,
///     #[gpu(uniform, group = 0, binding = 0)] params: Uniform<CullParams>,
///     #[gpu(storage_buffer, group = 0, binding = 1, read_only)] src: Storage<f32>,
/// ) {
///     // ...
/// }
/// ```
///
/// Each annotated parameter is removed from the exported signature and bound at
/// the head of the body to `Type::__bind()`, so the kernel body keeps using the
/// name it wrote while the translator sees a plain `cs_main(u32)`.
///
/// The metadata global is named `ornis_gpu_shader_meta` and carries the record
/// format of `ornis_gpu_shader::meta`: `entry=…;workgroup=x,y,z` plus one
/// `uniform=<id>:<tags>` entry per uniform parameter, with the tags taken from
/// the parameter's [`GpuStruct`](ornis_gpu_shader::GpuStruct) type, not from the
/// attribute. A kernel therefore cannot publish tags that disagree with the
/// fields it reads.
#[proc_macro_attribute]
pub fn gpu_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    match entry::expand(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}
