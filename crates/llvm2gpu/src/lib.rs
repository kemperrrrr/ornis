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

/// Entry metadata emitted by `#[gpu_entry]` into the shader crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderMeta {
    /// Entry function name (`cs_main` today).
    pub entry: String,
    /// `@workgroup_size(x, y, z)`.
    pub workgroup: [u32; 3],
    /// `ornis_gpu.types` field tags of the entry's uniform struct.
    pub types: Vec<u8>,
}

/// Extract `ornis_gpu_shader_meta` from textual LLVM IR.
///
/// The attribute writes a `#[used]` global named
/// `ornis_gpu_shader_meta` with `entry=…;workgroup=x,y,z;types=…`.
/// `None` means the shader predates the attribute; callers fall back to
/// their own options.
pub fn extract_meta(llvm_ir: &str) -> Option<ShaderMeta> {
    let line = llvm_ir
        .lines()
        .find(|line| line.contains("@ornis_gpu_shader_meta"))?;
    let start = line.find("c\"")? + 2;
    let rest = &line[start..];
    let mut end = None;
    let mut index = 0;
    while index < rest.len() {
        match rest.as_bytes()[index] {
            b'\\' => index += 3,
            b'"' => {
                end = Some(index);
                break;
            }
            _ => index += 1,
        }
    }
    let mut text = String::new();
    let mut chars = rest[..end?].chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let high = chars.next()?;
            let low = chars.next()?;
            let byte = u8::from_str_radix(&format!("{high}{low}"), 16).ok()?;
            text.push(byte as char);
        } else {
            text.push(ch);
        }
    }
    let mut entry = None;
    let mut workgroup = None;
    let mut types = Vec::new();
    for part in text.split(';') {
        let (key, value) = part.split_once('=')?;
        match key {
            "entry" => entry = Some(value.to_string()),
            "workgroup" => {
                let mut sizes = value.split(',');
                let x = sizes.next()?.parse().ok()?;
                let y = sizes.next()?.parse().ok()?;
                let z = sizes.next()?.parse().ok()?;
                workgroup = Some([x, y, z]);
            }
            "types" => {
                types = value
                    .split(',')
                    .map(|tag| tag.parse::<u8>().ok())
                    .collect::<Option<Vec<_>>>()?;
            }
            _ => {}
        }
    }
    Some(ShaderMeta {
        entry: entry?,
        workgroup: workgroup?,
        types,
    })
}

/// Translate using the metadata emitted by `#[gpu_entry]`.
///
/// Falls back to the scalar toy defaults when the shader has no metadata
/// global (older kernels).
pub fn translate_with_meta(llvm_ir: &str) -> Result<Translated, TranslateError> {
    let func = parse::parse(llvm_ir)?;
    let (types, workgroup) = match extract_meta(llvm_ir) {
        Some(meta) => {
            if meta.entry != ornis_gpu_abi::symbols::CS_MAIN {
                return Err(TranslateError::Unsupported);
            }
            (Some(meta.types), meta.workgroup)
        }
        None => (None, ornis_gpu_abi::symbols::CS_WORKGROUP),
    };
    let structs: Vec<&[u8]> = types.as_deref().into_iter().collect();
    let module = lower::lower(&func, &structs, workgroup)?;
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
    translate_with_types_and_workgroup(llvm_ir, structs, ornis_gpu_abi::symbols::CS_WORKGROUP)
}

/// Translate with uniform metadata and an explicit `@workgroup_size(x,y,z)`.
///
/// Production kernels choose their own workgroup (culling uses 64); the
/// scalar toy entry keeps [`CS_WORKGROUP`](ornis_gpu_abi::symbols::CS_WORKGROUP).
pub fn translate_with_types_and_workgroup(
    llvm_ir: &str,
    structs: &[&[u8]],
    workgroup: [u32; 3],
) -> Result<Translated, TranslateError> {
    let func = parse::parse(llvm_ir)?;
    let module = lower::lower(&func, structs, workgroup)?;
    emit::emit(&module)
}
