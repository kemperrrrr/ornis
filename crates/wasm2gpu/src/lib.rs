//! Offline translator from the toy `ornis_gpu` wasm ABI to WGSL and SPIR-V.
//!
//! This is not the production shader DSL. Vertex and fragment entries may use
//! structured `block` / `loop` / `if` / `br` / `br_if`. `raw_read` and
//! `texture_sample_2d` lower as before. Storage imports lower to an access
//! chain on one array global: scalars, vectors, and simple structs use the
//! host `GpuSafeLayout` stride. Uniform bindings use that same provenance
//! and the same `place_fields` layout: a struct field index is constant, and
//! a fixed-size array index may be dynamic where WGSL allows it. Group and
//! binding stay immediate operands. A join that names two bindings is
//! rejected. `br_table` and linear memory are rejected.
//! See `docs/rendering/wasm-segmented-memory.md`.

#![warn(missing_docs)]

mod ai;
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
mod structure;
mod ty;
mod uniform;
mod uniform_global;

pub use emit::Translated;
pub use error::TranslateError;

/// Parse `wasm` and lower the toy vertex and fragment entries.
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
