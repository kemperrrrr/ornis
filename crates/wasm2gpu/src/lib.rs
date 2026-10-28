//! Offline translator from the toy `ornis_gpu` wasm ABI to WGSL and SPIR-V.
//!
//! This is not the production shader DSL. It accepts one straight-line vertex
//! function and one straight-line fragment function, lowers `raw_read` and
//! `texture_sample_2d` imports to naga, and rejects linear-memory loads and
//! non-constant bindings. See `docs/rendering/wasm-segmented-memory.md`.

#![warn(missing_docs)]

mod cx;
mod emit;
mod error;
mod globals;
mod ir;
mod lower;
mod ops;
mod parse;
mod ret;

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
