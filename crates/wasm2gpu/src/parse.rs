//! Wasm bytes to a structured [`crate::ir::Module`].

use wasmparser::Parser;
use wasmparser::Payload;
use wasmparser::ValType;

use crate::decode;
use crate::error::TranslateError;
use crate::ir::Func;
use crate::ir::Import;
use crate::ir::Kind;
use crate::ir::MAX_PARAMS;
use crate::ir::Module;
use crate::ir::Sig;
use crate::structure;
use crate::structure::Flat;
use crate::ty::StructDef;

pub(crate) struct Raw {
    types: Vec<Sig>,
    imports: Vec<Import>,
    func_types: Vec<u32>,
    locals: Vec<Vec<Kind>>,
    bodies: Vec<Vec<Flat>>,
    exports: Vec<(String, u32)>,
    structs: Vec<StructDef>,
    saw_types: bool,
}

/// Parse `bytes`, rejecting linear memory and anything but the toy operators.
pub fn parse(bytes: &[u8]) -> Result<Module, TranslateError> {
    let mut raw = Raw {
        types: Vec::new(),
        imports: Vec::new(),
        func_types: Vec::new(),
        locals: Vec::new(),
        bodies: Vec::new(),
        exports: Vec::new(),
        structs: Vec::new(),
        saw_types: false,
    };
    for payload in Parser::new(0).parse_all(bytes) {
        feed(
            &mut raw,
            payload.map_err(|err| TranslateError::Wasm(err.to_string()))?,
        )?;
    }
    assemble(raw)
}

fn feed(raw: &mut Raw, payload: Payload<'_>) -> Result<(), TranslateError> {
    match payload {
        Payload::TypeSection(section) => read_types(raw, section),
        Payload::ImportSection(section) => read_imports(raw, section),
        Payload::FunctionSection(section) => read_funcs(raw, section),
        Payload::CodeSectionEntry(body) => read_body(raw, body),
        Payload::ExportSection(section) => read_exports(raw, section),
        Payload::CustomSection(section) => read_custom(raw, &section),
        Payload::MemorySection(_) | Payload::DataSection(_) | Payload::DataCountSection { .. } => {
            Err(TranslateError::LinearMemory)
        }
        Payload::TableSection(_) | Payload::ElementSection(_) => Err(TranslateError::IndirectCall),
        Payload::CodeSectionStart { .. } | Payload::End(_) | Payload::Version { .. } => Ok(()),
        _ => Err(TranslateError::Unsupported),
    }
}

pub(crate) fn read_custom(
    raw: &mut Raw,
    section: &wasmparser::CustomSectionReader<'_>,
) -> Result<(), TranslateError> {
    if section.name() != ornis_gpu_abi::symbols::TYPES_SECTION {
        return Ok(());
    }
    if raw.saw_types {
        return Err(TranslateError::UnsupportedType(
            "duplicate ornis_gpu.types".to_string(),
        ));
    }
    raw.saw_types = true;
    raw.structs = crate::ty::parse_types(section.data())?;
    Ok(())
}

fn read_types(
    raw: &mut Raw,
    section: wasmparser::SectionLimited<'_, wasmparser::RecGroup>,
) -> Result<(), TranslateError> {
    for group in section {
        let group = group.map_err(|err| TranslateError::Wasm(err.to_string()))?;
        for ty in group.types() {
            raw.types.push(func_sig(ty)?);
        }
    }
    Ok(())
}

fn func_sig(ty: &wasmparser::SubType) -> Result<Sig, TranslateError> {
    let wasmparser::CompositeInnerType::Func(func) = &ty.composite_type.inner else {
        return Err(TranslateError::Unsupported);
    };
    let (param_kind, params) = kinds(func.params())?;
    let (result_kind, results) = kinds(func.results())?;
    Ok(Sig {
        params,
        results,
        param_kind,
        result_kind,
    })
}

fn kinds(vals: &[ValType]) -> Result<([Kind; MAX_PARAMS], u32), TranslateError> {
    let mut out = [Kind::F32; MAX_PARAMS];
    let mut index = 0usize;
    while index < vals.len() {
        if index >= MAX_PARAMS {
            return Err(TranslateError::Unsupported);
        }
        out[index] = decode::one_kind(&vals[index])?;
        index += 1;
    }
    Ok((out, index as u32))
}

fn read_imports(
    raw: &mut Raw,
    section: wasmparser::SectionLimited<'_, wasmparser::Imports<'_>>,
) -> Result<(), TranslateError> {
    for group in section {
        let group = group.map_err(|err| TranslateError::Wasm(err.to_string()))?;
        let (module, name, ty) = match group {
            wasmparser::Imports::Single(_, import) => (import.module, import.name, import.ty),
            wasmparser::Imports::Compact1 { module, items } => {
                for item in items {
                    let item = item.map_err(|err| TranslateError::Wasm(err.to_string()))?;
                    record_import(raw, module, item.name, item.ty)?;
                }
                continue;
            }
            wasmparser::Imports::Compact2 { module, ty, names } => {
                for name in names {
                    let name = name.map_err(|err| TranslateError::Wasm(err.to_string()))?;
                    record_import(raw, module, name, ty)?;
                }
                continue;
            }
        };
        record_import(raw, module, name, ty)?;
    }
    Ok(())
}

fn record_import(
    raw: &mut Raw,
    module: &str,
    name: &str,
    ty: wasmparser::TypeRef,
) -> Result<(), TranslateError> {
    let index = import_func(module, name, ty)?;
    let sig = *raw
        .types
        .get(index as usize)
        .ok_or(TranslateError::Unsupported)?;
    raw.imports.push(Import {
        name: name.to_string(),
        sig,
    });
    Ok(())
}

fn import_func(module: &str, name: &str, ty: wasmparser::TypeRef) -> Result<u32, TranslateError> {
    let known =
        module == ornis_gpu_abi::symbols::MODULE && ornis_gpu_abi::symbols::is_toy_import(name);
    if !known {
        return Err(TranslateError::Unsupported);
    }
    match ty {
        wasmparser::TypeRef::Func(index) | wasmparser::TypeRef::FuncExact(index) => Ok(index),
        _ => Err(TranslateError::Unsupported),
    }
}

fn read_funcs(
    raw: &mut Raw,
    section: wasmparser::SectionLimited<'_, u32>,
) -> Result<(), TranslateError> {
    for index in section {
        raw.func_types
            .push(index.map_err(|err| TranslateError::Wasm(err.to_string()))?);
    }
    Ok(())
}

pub(crate) fn read_body(
    raw: &mut Raw,
    body: wasmparser::FunctionBody<'_>,
) -> Result<(), TranslateError> {
    let func_slot = raw.bodies.len();
    let sig = func_sig_at(raw, func_slot)?;
    let locals = read_locals(&body, &sig)?;
    let reader = body
        .get_operators_reader()
        .map_err(|err| TranslateError::Wasm(err.to_string()))?;
    let mut flats = Vec::new();
    for op in reader {
        let op = op.map_err(|err| TranslateError::Wasm(err.to_string()))?;
        if decode::memory(&op) {
            return Err(TranslateError::LinearMemory);
        }
        flats.push(decode::op(&raw.types, &locals, &op)?);
    }
    raw.locals.push(locals);
    raw.bodies.push(flats);
    Ok(())
}

fn func_sig_at(raw: &Raw, func_slot: usize) -> Result<Sig, TranslateError> {
    let ty = *raw
        .func_types
        .get(func_slot)
        .ok_or(TranslateError::Unsupported)?;
    raw.types
        .get(ty as usize)
        .copied()
        .ok_or(TranslateError::Unsupported)
}

pub(crate) fn read_locals(
    body: &wasmparser::FunctionBody<'_>,
    sig: &Sig,
) -> Result<Vec<Kind>, TranslateError> {
    let mut locals = Vec::new();
    let mut index = 0u32;
    while index < sig.params {
        locals.push(sig.param_kind[index as usize]);
        index += 1;
    }
    let mut reader = body
        .get_locals_reader()
        .map_err(|err| TranslateError::Wasm(err.to_string()))?;
    let count = reader.get_count();
    let mut group = 0u32;
    while group < count {
        let (n, val) = reader
            .read()
            .map_err(|err| TranslateError::Wasm(err.to_string()))?;
        let kind = decode::one_kind(&val)?;
        let mut item = 0u32;
        while item < n {
            locals.push(kind);
            item += 1;
        }
        group += 1;
    }
    Ok(locals)
}

fn read_exports(
    raw: &mut Raw,
    section: wasmparser::SectionLimited<'_, wasmparser::Export<'_>>,
) -> Result<(), TranslateError> {
    for export in section {
        let export = export.map_err(|err| TranslateError::Wasm(err.to_string()))?;
        let wasmparser::ExternalKind::Func = export.kind else {
            return Err(TranslateError::Unsupported);
        };
        raw.exports.push((export.name.to_string(), export.index));
    }
    Ok(())
}

pub(crate) fn assemble(raw: Raw) -> Result<Module, TranslateError> {
    if raw.bodies.len() != raw.func_types.len() {
        return Err(TranslateError::Unsupported);
    }
    let mut funcs = Vec::new();
    let mut index = 0usize;
    while index < raw.bodies.len() {
        let sig = func_sig_at(&raw, index)?;
        let func_index = raw.imports.len() + index;
        let body = structure::build(raw.bodies[index].clone())?;
        funcs.push(Func {
            export: export_name(&raw.exports, func_index as u32),
            sig,
            locals: raw.locals[index].clone(),
            body,
        });
        index += 1;
    }
    Ok(Module {
        imports: raw.imports,
        structs: raw.structs,
        funcs,
    })
}

fn export_name(exports: &[(String, u32)], func: u32) -> String {
    let mut index = 0usize;
    while index < exports.len() {
        if exports[index].1 == func {
            return exports[index].0.clone();
        }
        index += 1;
    }
    String::new()
}
