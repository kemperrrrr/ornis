//! Stable shader entry attribute for the llvm2gpu path.
//!
//! [`gpu_entry`] keeps the annotated function, strips its resource parameters,
//! and emits a `#[used]` metadata record in a known LLVM global, so the offline
//! builder reads the entry name, workgroup size, and uniform field tags from
//! the shader crate itself instead of CLI flags.
//!
//! The attribute is the stable stand-in for rust-gpu's `#[spirv(…)]`:
//! `compute(threads(N))`, `compute(threads(X, Y))`, `compute(threads(X, Y, Z))`,
//! `vertex`, or `fragment`. Resources are bound by their attributes. A kernel
//! does not declare `extern "C"`; a missing operation is added to
//! `ornis-gpu-shader`.

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
/// pub fn cs_main(
///     id: u32,
///     #[gpu(uniform, group = 0, binding = 0)] params: Uniform<CullParams>,
///     #[gpu(storage_buffer, group = 0, binding = 1, read_only)] src: Storage<f32>,
/// ) {
///     // ...
/// }
/// ```
///
/// Each annotated parameter stays in the signature, with the group, binding, and
/// access mode written into its type. The type is zero-sized, so rustc drops the
/// argument from the compiled entry: the translator sees the same
/// `cs_main(i32)` it always did, and the body uses the parameter as an ordinary
/// argument.
///
/// The attribute also attaches `#[unsafe(export_name = "<fn name>")]`, so a
/// kernel is a plain safe `pub fn` — no `extern "C"` and no
/// `#[unsafe(no_mangle)]`.
///
/// A raster entry is named for its stage: `#[gpu_entry(vertex)] fn vs_main()`
/// and `#[gpu_entry(fragment)] fn fs_main()` compile to `vs_main()` and
/// `fs_main()`. Both read builtins and `@location` values through
/// `ornis_gpu_shader` functions (`vertex_index`, `vertex_input_lane::<LOC,
/// LANE>`, `fragment_input_lane::<LOC, LANE>`) and write outputs the same way
/// (`vertex_position`, `vertex_store_{2,4}::<LOC>`, `fragment_store_at::<LOC>`),
/// so the compiled signature carries no interface type the translator would
/// have to lay out from Rust.
///
/// The metadata global is named `ornis_gpu_shader_meta` and carries the record
/// format of `ornis_gpu_shader::meta`: `entry=…;workgroup=x,y,z` plus one
/// `uniform=<id>:<tags>` entry per uniform parameter, with the tags taken from
/// the parameter's `GpuStruct` type, not from the attribute. A kernel therefore
/// cannot publish tags that disagree with the fields it reads.
#[proc_macro_attribute]
pub fn gpu_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    match entry::expand(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}
