//! Wasm bytes to the straight-line [`crate::ir::Module`].

use wasmparser::Operator;
use wasmparser::Parser;
use wasmparser::Payload;
use wasmparser::ValType;

use crate::error::TranslateError;
use crate::ir::Func;
use crate::ir::Import;
use crate::ir::Kind;
use crate::ir::MAX_PARAMS;
use crate::ir::Module;
use crate::ir::Op;
use crate::ir::Sig;

struct Raw {
    types: Vec<Sig>,
    imports: Vec<Import>,
    func_types: Vec<u32>,
    bodies: Vec<Vec<Op>>,
    exports: Vec<(String, u32)>,
}

/// Parse `bytes`, rejecting linear memory and anything but the toy operators.
pub fn parse(bytes: &[u8]) -> Result<Module, TranslateError> {
    let mut raw = Raw {
        types: Vec::new(),
        imports: Vec::new(),
        func_types: Vec::new(),
        bodies: Vec::new(),
        exports: Vec::new(),
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
        Payload::MemorySection(_) | Payload::DataSection(_) | Payload::DataCountSection { .. } => {
            Err(TranslateError::LinearMemory)
        }
        Payload::CodeSectionStart { .. }
        | Payload::End(_)
        | Payload::Version { .. }
        | Payload::CustomSection(_) => Ok(()),
        _ => Err(TranslateError::Unsupported),
    }
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
    let (param_kind, params) = param_kinds(func.params())?;
    Ok(Sig {
        params,
        results: count_vals(func.results())?,
        param_kind,
    })
}

fn param_kinds(vals: &[ValType]) -> Result<([Kind; MAX_PARAMS], u32), TranslateError> {
    let mut kinds = [Kind::F32; MAX_PARAMS];
    let mut index = 0usize;
    while index < vals.len() {
        if index >= MAX_PARAMS {
            return Err(TranslateError::Unsupported);
        }
        kinds[index] = one_kind(&vals[index])?;
        index += 1;
    }
    Ok((kinds, index as u32))
}

fn one_kind(val: &ValType) -> Result<Kind, TranslateError> {
    match val {
        ValType::I32 => Ok(Kind::I32),
        ValType::F32 => Ok(Kind::F32),
        _ => Err(TranslateError::Unsupported),
    }
}

fn count_vals(vals: &[ValType]) -> Result<u32, TranslateError> {
    let mut count = 0u32;
    for val in vals {
        match val {
            ValType::I32 | ValType::F32 => count += 1,
            _ => return Err(TranslateError::Unsupported),
        }
    }
    Ok(count)
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

fn read_body(raw: &mut Raw, body: wasmparser::FunctionBody<'_>) -> Result<(), TranslateError> {
    let func_slot = raw.bodies.len();
    let reader = body
        .get_operators_reader()
        .map_err(|err| TranslateError::Wasm(err.to_string()))?;
    let mut ops = Vec::new();
    for op in reader {
        let op = op.map_err(|err| TranslateError::Wasm(err.to_string()))?;
        if let Some(decoded) = decode_op(raw, func_slot, &op)? {
            ops.push(decoded);
        }
    }
    raw.bodies.push(ops);
    Ok(())
}

fn local_op(raw: &Raw, func_slot: usize, local_index: u32) -> Op {
    let kind = param_kind(raw, func_slot, local_index);
    if kind == Kind::I32 {
        Op::LocalI32(local_index)
    } else {
        Op::Local(local_index)
    }
}

fn param_kind(raw: &Raw, func_slot: usize, local_index: u32) -> Kind {
    let Some(ty) = raw.func_types.get(func_slot) else {
        return Kind::F32;
    };
    let Some(sig) = raw.types.get(*ty as usize) else {
        return Kind::F32;
    };
    let slot = local_index as usize;
    if slot < sig.params as usize {
        sig.param_kind[slot]
    } else {
        Kind::F32
    }
}

fn decode_op(raw: &Raw, func_slot: usize, op: &Operator<'_>) -> Result<Option<Op>, TranslateError> {
    if is_memory(op) {
        return Err(TranslateError::LinearMemory);
    }
    Ok(match op {
        Operator::I32Const { value } => Some(Op::I32(
            u32::try_from(*value).map_err(|_| TranslateError::Unsupported)?,
        )),
        Operator::F32Const { value } => Some(Op::F32(value.bits())),
        Operator::LocalGet { local_index } => Some(local_op(raw, func_slot, *local_index)),
        Operator::Call { function_index } => Some(Op::Call(*function_index)),
        Operator::F32Mul => Some(Op::Mul),
        Operator::F32Add => Some(Op::Add),
        Operator::End => None,
        _ => return Err(TranslateError::Unsupported),
    })
}

fn is_memory(op: &Operator<'_>) -> bool {
    matches!(
        op,
        Operator::I32Load { .. }
            | Operator::I64Load { .. }
            | Operator::F32Load { .. }
            | Operator::F64Load { .. }
            | Operator::I32Load8S { .. }
            | Operator::I32Load8U { .. }
            | Operator::I32Load16S { .. }
            | Operator::I32Load16U { .. }
            | Operator::I32Store { .. }
            | Operator::I64Store { .. }
            | Operator::F32Store { .. }
            | Operator::F64Store { .. }
            | Operator::I32Store8 { .. }
            | Operator::I32Store16 { .. }
            | Operator::MemorySize { .. }
            | Operator::MemoryGrow { .. }
    )
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

fn assemble(raw: Raw) -> Result<Module, TranslateError> {
    if raw.bodies.len() != raw.func_types.len() {
        return Err(TranslateError::Unsupported);
    }
    let mut funcs = Vec::new();
    let mut index = 0usize;
    while index < raw.bodies.len() {
        let ty = *raw
            .func_types
            .get(index)
            .ok_or(TranslateError::Unsupported)?;
        let sig = *raw
            .types
            .get(ty as usize)
            .ok_or(TranslateError::Unsupported)?;
        let func_index = raw.imports.len() + index;
        funcs.push(Func {
            export: export_name(&raw.exports, func_index as u32),
            sig,
            ops: raw.bodies[index].clone(),
        });
        index += 1;
    }
    Ok(Module {
        imports: raw.imports,
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
