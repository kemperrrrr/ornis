//! Offline translator from LLVM IR (stable `rustc`) to WGSL and SPIR-V.
//!
//! The translator covers the spike-B subset of root `3.1.2`: a scalar
//! compute entry (`cs_main`) built from `SROA`-scalarized arithmetic,
//! comparisons, `select`, integer bit/arithmetic ops, the `sqrt`/`sin`
//! family, the `ornis_gpu` storage/uniform ABI, structured `if`/`loop`
//! control flow with `phi` values, and uniform struct fields with
//! metadata from `ornis_gpu.types`. A scalar `-> f32` result is stored
//! to `storage_0_0` at the invocation id, exactly like the toy `compute`
//! kernel `wasm2gpu` already tests. Linear memory (`load`/`store`/
//! `getelementptr`), irreducible CFGs, nested loops, and nested uniform
//! structs are rejected so an unknown shape never becomes a wrong shader.

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
/// `panic=abort`, and a `#[no_mangle] cs_main(i32) -> f32` export compiled
/// with `--emit=llvm-ir`. Prefer `opt-level=s` for dynamic loops: `3` may
/// unroll them into a CFG shape this translator rejects.
///
/// Uniform struct fields need layout metadata; use
/// [`translate_with_types`] for kernels that call `uniform_field_*`.
pub fn translate(llvm_ir: &str) -> Result<Translated, TranslateError> {
    translate_with_types(llvm_ir, &[])
}

/// Translate with `ornis_gpu.types` field-tag payloads, one slice per
/// struct id (`SID`).
///
/// This is the same metadata the `gpu_struct!` macro exposes as
/// `FIELD_TAGS` and `wasm2gpu` reads from its custom section. A call to
/// `uniform_field_f32`/`uniform_field_u32` requires the `SID` to be present
/// here; otherwise it is rejected as an unsupported type.
pub fn translate_with_types(
    llvm_ir: &str,
    structs: &[&[u8]],
) -> Result<Translated, TranslateError> {
    let func = parse::parse(llvm_ir)?;
    let module = lower::lower(&func, structs)?;
    emit::emit(&module)
}
