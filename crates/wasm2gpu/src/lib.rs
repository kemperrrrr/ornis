//! Offline translator from the toy `ornis_gpu` wasm ABI to WGSL and SPIR-V.
//!
//! This is not the production shader DSL. Vertex and fragment entries may use
//! structured `block` / `loop` / `if` / `br` / `br_if`. `br_table` lowers to
//! a WGSL `switch`. A direct `call` of a defined function becomes a naga
//! function specialized by immediate group and binding; the offset passes as
//! an argument or a result. The same constant arguments reuse one function.
//! An `f32` argument is not specialized. Recursion, `call_indirect`, and a
//! specialization past the budget are rejected. The budget counts distinct
//! calls, not instructions in a body that does not call. A `br_table` that continues
//! a loop is a `continue`. A `br_table` aimed at a region this translator
//! did not wrap in a naga loop is rejected.
//! `cs_main`, when exported, is a compute entry with `@workgroup_size` and
//! `global_invocation_id`. `raw_read` and `texture_sample_2d` lower as before.
//! Storage imports lower to an access chain on one array global: scalars,
//! vectors, and simple structs use the host `GpuSafeLayout` stride. Uniform
//! bindings use that same provenance and the same `place_fields` layout: a
//! struct field index is constant, and a fixed-size array index may be
//! dynamic where WGSL allows it. Group and binding stay immediate operands.
//! A join that names two bindings is rejected. Linear memory is rejected.
//! See `docs/rendering/wasm-segmented-memory.md`.

#![warn(missing_docs)]

mod ai;
mod call;
mod cfg;
mod cx;
mod decode;
mod emit;
mod env;
mod error;
mod flow;
mod globals;
mod ir;
mod lower;
mod ops;
mod parse;
mod ret;
mod storage;
mod struct_members;
mod structure;
mod ty;
mod uniform;
mod uniform_global;

pub use emit::Translated;
pub use error::TranslateError;

/// Parse `wasm` and lower the toy vertex and fragment entries.
///
/// A `cs_main` export is lowered as well. Its SPIR-V words are
/// [`Translated::compute_spirv`].
///
/// The result has already been validated by naga, reprinted as WGSL, parsed
/// back, and round-tripped through SPIR-V. Bindless `binding_array` types are
/// rejected.
// qual:api — public translator entry; in-tree callers are the integration tests.
pub fn translate(wasm: &[u8]) -> Result<Translated, TranslateError> {
    let module = parse::parse(wasm)?;
    let lowered = lower::lower(&module)?;
    emit::emit(&lowered)
}
