//! Offline translator from LLVM IR (stable `rustc`) to WGSL and SPIR-V.
//!
//! The translator covers the spike-B subset of root `3.1.2`: a single
//! scalar compute function (`cs_main(i32) -> f32`) built from `SROA`
//! scalarized arithmetic, comparisons, `select`, and the `sqrt`/`sin`
//! intrinsics. The scalar result is stored to `storage_0_0` at the
//! invocation id, exactly like the toy `compute` kernel `wasm2gpu`
//! already tests. Control flow (`phi`/`br`/`loop`) and memory
//! (`load`/`store`/`getelementptr`) are rejected here; they are
//! phase-2 work for the slice → resource+index restoration.

#![warn(missing_docs)]

mod emit;
mod error;
mod lower;
mod parse;

pub use emit::Translated;
pub use error::TranslateError;

/// Parse textual LLVM IR and lower the `cs_main` compute entry.
///
/// Requires stable `rustc` with the `wasm32-unknown-unknown` target,
/// `opt-level=3`, `panic=abort`, and a `#[no_mangle] cs_main(i32) -> f32`
/// export compiled with `--emit=llvm-ir`.
pub fn translate(llvm_ir: &str) -> Result<Translated, TranslateError> {
    let func = parse::parse(llvm_ir)?;
    let module = lower::lower(&func)?;
    emit::emit(&module)
}
