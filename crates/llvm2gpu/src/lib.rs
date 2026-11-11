//! Offline translator from LLVM IR (stable `rustc`) to WGSL and SPIR-V.
//!
//! The translator covers the compute and raster subset: `cs_main`, `vs_main`,
//! and `fs_main` built from `SROA`-scalarized arithmetic, comparisons,
//! `select`, signed and unsigned integer bit/arithmetic ops, the
//! `sqrt`/`sin` family, the `ornis_gpu` storage/uniform/texture ABI,
//! structured `if`/`loop` control flow with `phi` values (nested loops and
//! early returns included), nested uniform struct fields with metadata from
//! `ornis_gpu.types`, and workgroup memory/atomics. A scalar `-> f32` result is
//! stored to `storage_0_0` at the invocation id, exactly like the toy
//! `compute` kernel `wasm2gpu` already tests. `TextureSampling::Multisampled`
//! re-declares the same kernel's textures as multisampled images with
//! sample-0 loads, which is the 4x AO variant.
//!
//! A raster entry has no parameters: builtins and `@location` values are
//! `ornis_gpu` imports (`vertex_index`, `instance_index`,
//! `vertex_input_lane`, `fragment_input_lane`) and outputs are stores
//! (`vertex_position`, `vertex_store_*`, `fragment_store_location`). The lowerer
//! reads the interface back out of those calls, so the declared `vecN` width of
//! each location is the highest lane the body touches. A vertex entry returns
//! a struct with `@builtin(position)`; a fragment entry returns the bare
//! `@location(0) vec4<f32>` when that is its only target.
//!
//! Values keep their LLVM bit pattern: `i32` lives as WGSL `u32`, and signed
//! operators bitcast around the operation, so `ashr`/`sdiv`/`srem`/`sitofp`/
//! `icmp s*`/`llvm.smin`/`llvm.smax` keep two's-complement semantics. `fcmp`
//! unordered predicates (`u*`, `uno`) stay true for NaN while `ord`/`one` are
//! ordered. Integer division/remainder use WGSL's runtime contract where LLVM
//! would be undefined: a zero divisor or `i32::MIN / -1` returns the numerator
//! for division and zero for remainder (not Rust's panicking operators).
//! A workgroup barrier is accepted only in straight-line code: the
//! lowerer does not prove branch or trip-count uniformity, so a barrier inside
//! an `if`/`loop` or after an arm that returned is rejected instead of
//! emitting WGSL Tint refuses (or native UB).
//!
//! Linear memory (`load`/`store`/`getelementptr`), irreducible CFGs, and
//! multi-exit loops are rejected so an unknown shape never becomes a wrong
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

/// How the translator declares texture bindings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextureSampling {
    /// `texture_2d<f32>` / `texture_depth_2d`; `textureLoad` reads mip 0.
    #[default]
    SingleSample,
    /// `texture_multisampled_2d<f32>` / `texture_depth_multisampled_2d`;
    /// `textureLoad` reads sample 0. This is the 4x MSAA build of the same
    /// kernel: the body is unchanged, only the declared texture type is.
    Multisampled,
}

impl TextureSampling {
    /// Whether bindings lower to multisampled image types.
    fn multisampled(self) -> bool {
        self == Self::Multisampled
    }
}

/// Translate using the metadata emitted by `#[gpu_entry]`.
///
/// Falls back to the scalar toy defaults when the shader has no metadata
/// global (older kernels).
///
/// # Errors
///
/// [`TranslateError::Unsupported`] when the entry is not `cs_main`, `vs_main`,
/// or `fs_main`, or the body leaves the translated subset, and
/// [`TranslateError::UnsupportedType`] when the metadata names a uniform
/// struct the translator cannot lay out.
pub fn translate_with_meta(llvm_ir: &str) -> Result<Translated, TranslateError> {
    translate_with_meta_and_sampling(llvm_ir, TextureSampling::SingleSample)
}

/// Translate using `#[gpu_entry]` metadata and an explicit texture sampling.
///
/// [`TextureSampling::Multisampled`] turns every texture binding into its
/// multisampled counterpart and reads sample 0, which is how the 4x AO pass
/// binds its g-buffer.
///
/// # Errors
///
/// Same as [`translate_with_meta`].
pub fn translate_with_meta_and_sampling(
    llvm_ir: &str,
    sampling: TextureSampling,
) -> Result<Translated, TranslateError> {
    let func = parse::parse(llvm_ir)?;
    let (meta, workgroup) = match meta::extract(llvm_ir)? {
        Some(meta) => {
            if meta.entry != ornis_gpu_abi::symbols::CS_MAIN
                && meta.entry != ornis_gpu_abi::symbols::VS_MAIN
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
    let module = lower::lower(&func, &refs, workgroup, sampling.multisampled())?;
    emit::emit(&module)
}

/// Parse textual LLVM IR and lower the entry point it defines.
///
/// Requires stable `rustc` with the `wasm32-unknown-unknown` target,
/// `panic=abort`, and an exported `cs_main`, `vs_main`, or `fs_main` compiled
/// with `--emit=llvm-ir`. `xtask build-shaders` builds the checked-in
/// production artifacts at `opt-level=1`, which keeps dynamic loops
/// structured; `3` may unroll them into a CFG shape this translator rejects.
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
    let module = lower::lower(&func, structs, workgroup, false)?;
    emit::emit(&module)
}
