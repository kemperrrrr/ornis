//! Offline translator from LLVM IR (stable `rustc`) to WGSL and SPIR-V.
//!
//! The translator covers the compute subset of `3.1.2`: a scalar compute entry
//! (`cs_main`) built from `SROA`-scalarized arithmetic, comparisons, `select`,
//! integer bit/arithmetic ops, the `sqrt`/`sin` family, the `ornis_gpu`
//! storage/uniform ABI, structured `if`/`loop` control flow with `phi` values,
//! and uniform struct fields with metadata from `ornis_gpu.types`. A scalar
//! `-> f32` result is stored to `storage_0_0` at the invocation id, exactly like
//! the toy `compute` kernel `wasm2gpu` already tests. Linear memory
//! (`load`/`store`/`getelementptr`), irreducible CFGs, nested loops, and nested
//! uniform structs are rejected so an unknown shape never becomes a wrong
//! shader.
//!
//! Kernels written against the typed SDK (`ornis-gpu-shader`) lower to the same
//! IR as the raw `extern "C"` ones, so nothing here knows about `Storage` or
//! `Uniform`: the wrapper methods inline into direct `ornis_gpu` import calls,
//! and the entry, workgroup, and field tags come from the record
//! [`ShaderMeta`] describes.

#![warn(missing_docs)]

mod emit;
mod error;
mod lower;
pub mod meta;
mod parse;

pub use emit::Translated;
pub use error::TranslateError;
pub use meta::{ShaderMeta, extract as extract_meta, parse as parse_meta};

/// Translate using the metadata emitted by `#[gpu_entry]`.
///
/// Falls back to the scalar toy defaults when the shader has no metadata
/// global (older kernels).
///
/// # Errors
///
/// [`TranslateError::Unsupported`] when the entry is not `cs_main` or the body
/// leaves the translated subset, and [`TranslateError::UnsupportedType`] when
/// the metadata names a uniform struct the translator cannot lay out.
pub fn translate_with_meta(llvm_ir: &str) -> Result<Translated, TranslateError> {
    let func = parse::parse(llvm_ir)?;
    let (meta, workgroup) = match meta::extract(llvm_ir)? {
        Some(meta) => {
            if meta.entry != ornis_gpu_abi::symbols::CS_MAIN
                && meta.entry != ornis_gpu_abi::symbols::FS_MAIN
            {
                return Err(TranslateError::Unsupported);
            }
            let workgroup = meta.workgroup;
            (Some(meta), workgroup)
        }
        None => (None, ornis_gpu_abi::symbols::CS_WORKGROUP),
    };
    let structs = match &meta {
        Some(meta) => meta.table(),
        None => Vec::new(),
    };
    let refs: Vec<&[u8]> = structs.iter().map(Vec::as_slice).collect();
    let module = lower::lower(&func, &refs, workgroup)?;
    emit::emit(&module)
}

/// Parse textual LLVM IR and lower the `cs_main` compute entry.
///
/// Requires stable `rustc` with the `wasm32-unknown-unknown` target,
/// `panic=abort`, and a `#[no_mangle] cs_main(i32) -> f32` export compiled
/// with `--emit=llvm-ir`. Prefer `opt-level=s` for dynamic loops: `3` may
/// unroll them into a CFG shape this translator rejects.
///
/// Uniform struct fields need layout metadata; use
/// [`translate_with_types`] for kernels that call `uniform_field_*`.
///
/// # Errors
///
/// Same as [`translate_with_types`].
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
///
/// # Errors
///
/// [`TranslateError::Parse`] for IR outside the accepted grammar,
/// [`TranslateError::Unsupported`] for constructs the lowered subset does not
/// cover, and [`TranslateError::UnsupportedType`] for a uniform field the
/// metadata does not declare.
pub fn translate_with_types(
    llvm_ir: &str,
    structs: &[&[u8]],
) -> Result<Translated, TranslateError> {
    translate_with_types_and_workgroup(llvm_ir, structs, ornis_gpu_abi::symbols::CS_WORKGROUP)
}

/// Translate with uniform metadata and an explicit `@workgroup_size(x,y,z)`.
///
/// Production kernels choose their own workgroup (culling uses 64); the
/// scalar toy entry keeps [`CS_WORKGROUP`](ornis_gpu_abi::symbols::CS_WORKGROUP).
///
/// # Errors
///
/// Same as [`translate_with_types`].
pub fn translate_with_types_and_workgroup(
    llvm_ir: &str,
    structs: &[&[u8]],
    workgroup: [u32; 3],
) -> Result<Translated, TranslateError> {
    let func = parse::parse(llvm_ir)?;
    let module = lower::lower(&func, structs, workgroup)?;
    emit::emit(&module)
}
