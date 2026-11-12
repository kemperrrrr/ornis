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
pub use meta::{
    ShaderMeta, extract as extract_meta, extract_entry as extract_meta_for, parse as parse_meta,
};

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
/// Every entry point the module defines goes into one WGSL module, in the
/// order the IR declares them: a pass ships `vs_main` and `fs_main` together,
/// sharing the bindings both stages read. A shader without a metadata record
/// (older kernels) falls back to the scalar toy defaults.
///
/// # Errors
///
/// [`TranslateError::Unsupported`] when an entry is not `cs_main`, `vs_main`,
/// or `fs_main`, or a body leaves the translated subset, and
/// [`TranslateError::UnsupportedType`] when the metadata names a uniform
/// struct the translator cannot lay out, or two entries declare the same
/// struct id with different tags.
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
    translate_entries(llvm_ir, None, sampling)
}

/// Translate one named entry: the single-entry form `xtask build-shaders
/// --entry …` uses, so the other entries stay out of the emitted module.
///
/// # Errors
///
/// Same as [`translate_with_meta`], plus [`TranslateError::Unsupported`] when
/// the module defines no entry called `entry`.
pub fn translate_entry_with_sampling(
    llvm_ir: &str,
    entry: &str,
    sampling: TextureSampling,
) -> Result<Translated, TranslateError> {
    translate_entries(llvm_ir, Some(entry), sampling)
}

/// Translate every entry, or only `entry` when one is named.
fn translate_entries(
    llvm_ir: &str,
    only: Option<&str>,
    sampling: TextureSampling,
) -> Result<Translated, TranslateError> {
    // Every define is parsed: a kernel may call a function of its own crate,
    // and those helpers are lowered into the same module.
    let module = parse::parse_module(llvm_ir)?;
    let funcs: Vec<&parse::LlvmFunc> = match only {
        Some(entry) => vec![
            module
                .iter()
                .find(|func| func.name == entry)
                .ok_or(TranslateError::Unsupported)?,
        ],
        None => module
            .iter()
            .filter(|func| parse::is_entry(&func.name))
            .collect(),
    };
    if funcs.is_empty() {
        return Err(TranslateError::Unsupported);
    }
    let metas = meta::extract(llvm_ir)?;
    let structs = merge_tables(&metas)?;
    let refs: Vec<&[u8]> = structs.iter().map(Vec::as_slice).collect();
    let mut lowering = lower::Lowering::new(&refs, sampling.multisampled());
    lowering.add_helpers(&module)?;
    for func in &funcs {
        let workgroup = metas
            .iter()
            .find(|meta| meta.entry == func.name)
            .map_or(ornis_gpu_abi::symbols::CS_WORKGROUP, |meta| meta.workgroup);
        lowering.push(func, workgroup)?;
    }
    emit::emit(&lowering.finish())
}

/// Merge the uniform-struct tables of every entry record.
///
/// Struct ids are assigned by the kernel author and shared per crate, so two
/// entries that use the same id must agree on it. Disagreeing tags would make
/// one entry read the other's layout, so that is an error rather than a
/// last-one-wins merge.
fn merge_tables(metas: &[meta::ShaderMeta]) -> Result<Vec<Vec<u8>>, TranslateError> {
    let mut merged: Vec<Vec<u8>> = Vec::new();
    for meta in metas {
        for (id, tags) in meta.structs.iter() {
            let index = *id as usize;
            if merged.len() <= index {
                merged.resize(index + 1, Vec::new());
            }
            let slot = &mut merged[index];
            if slot.is_empty() {
                *slot = tags.clone();
            } else if slot != tags {
                return Err(TranslateError::UnsupportedType(format!(
                    "uniform struct {id} has different fields in two entries"
                )));
            }
        }
    }
    Ok(merged)
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
    let module = parse::parse_module(llvm_ir)?;
    let mut lowering = lower::Lowering::new(structs, false);
    lowering.add_helpers(&module)?;
    for func in module.iter().filter(|func| parse::is_entry(&func.name)) {
        lowering.push(func, workgroup)?;
    }
    emit::emit(&lowering.finish())
}
