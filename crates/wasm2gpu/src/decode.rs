//! Wasm operators to flat instructions, before label matching.

use wasmparser::BlockType;
use wasmparser::MemArg;
use wasmparser::Operator;

use crate::error::TranslateError;
use crate::ir::BlockTy;
use crate::ir::Kind;
use crate::ir::Op;
use crate::ir::Sig;
use crate::structure::Flat;

/// Decode one operator. Memory ops are rejected by the caller.
pub(super) fn op(
    types: &[Sig],
    locals: &[Kind],
    op: &Operator<'_>,
) -> Result<Flat, TranslateError> {
    if let Some(flat) = control(types, op)? {
        return Ok(flat);
    }
    stack_op(locals, op).map(Flat::Op)
}

pub(super) fn memory(op: &Operator<'_>) -> bool {
    matches!(
        op,
        Operator::I64Load { .. }
            | Operator::F64Load { .. }
            | Operator::I32Load8S { .. }
            | Operator::I32Load8U { .. }
            | Operator::I32Load16S { .. }
            | Operator::I32Load16U { .. }
            | Operator::I64Store { .. }
            | Operator::F64Store { .. }
            | Operator::I32Store8 { .. }
            | Operator::I32Store16 { .. }
            | Operator::MemorySize { .. }
            | Operator::MemoryGrow { .. }
    )
}

fn control(types: &[Sig], op: &Operator<'_>) -> Result<Option<Flat>, TranslateError> {
    let flat = match op {
        Operator::Block { blockty } => Flat::Block(block_ty(types, blockty)?),
        Operator::Loop { blockty } => Flat::Loop(block_ty(types, blockty)?),
        Operator::If { blockty } => Flat::If(block_ty(types, blockty)?),
        Operator::Else => Flat::Else,
        Operator::End => Flat::End,
        Operator::Br { relative_depth } => Flat::Br(*relative_depth),
        Operator::BrIf { relative_depth } => Flat::BrIf(*relative_depth),
        Operator::Return => Flat::Return,
        Operator::BrTable { targets } => return br_table(targets),
        _ => return Ok(None),
    };
    Ok(Some(flat))
}

fn stack_op(locals: &[Kind], op: &Operator<'_>) -> Result<Op, TranslateError> {
    match op {
        Operator::I32Const { value } => i32_const(*value),
        Operator::F32Const { value } => Ok(Op::F32(value.bits())),
        Operator::LocalGet { local_index } => local_get(locals, *local_index),
        Operator::LocalSet { local_index } => Ok(Op::LocalSet(*local_index)),
        Operator::LocalTee { local_index } => Ok(Op::LocalTee(*local_index)),
        Operator::Drop => Ok(Op::Drop),
        Operator::Call { function_index } => Ok(Op::Call(*function_index)),
        Operator::CallIndirect { .. } => Err(TranslateError::IndirectCall),
        Operator::F32Mul => Ok(Op::Mul),
        Operator::F32Add => Ok(Op::Add),
        Operator::I32Add => Ok(Op::I32Add),
        Operator::I32Mul => Ok(Op::I32Mul),
        Operator::I32Sub => Ok(Op::I32Sub),
        Operator::I32Eqz => Ok(Op::I32Eqz),
        Operator::I32Eq => Ok(Op::I32Eq),
        Operator::I32Ne => Ok(Op::I32Ne),
        Operator::I32LtU => Ok(Op::I32LtU),
        Operator::I32GtU => Ok(Op::I32GtU),
        Operator::I32GeU => Ok(Op::I32GeU),
        Operator::I32Shl => Ok(Op::I32Shl),
        Operator::I32ShrU => Ok(Op::I32ShrU),
        Operator::I32Load { memarg } => Ok(Op::LoadI32(mem_offset(memarg)?)),
        Operator::F32Load { memarg } => Ok(Op::LoadF32(mem_offset(memarg)?)),
        Operator::I32Store { memarg } => Ok(Op::StoreI32(mem_offset(memarg)?)),
        Operator::F32Store { memarg } => Ok(Op::StoreF32(mem_offset(memarg)?)),
        Operator::F32Lt => Ok(Op::F32Lt),
        Operator::Select => Ok(Op::Select),
        _ => Err(TranslateError::Unsupported),
    }
}

fn br_table(targets: &wasmparser::BrTable<'_>) -> Result<Option<Flat>, TranslateError> {
    let mut labels = Vec::new();
    for target in targets.targets() {
        labels.push(target.map_err(|err| TranslateError::Wasm(err.to_string()))?);
    }
    Ok(Some(Flat::BrTable {
        targets: labels,
        default: targets.default(),
    }))
}

fn i32_const(value: i32) -> Result<Op, TranslateError> {
    let bits = u32::try_from(value).map_err(|_| TranslateError::Unsupported)?;
    Ok(Op::I32(bits))
}

/// A wasm memory offset wider than `u32` can never address wasm32 memory.
fn mem_offset(memarg: &MemArg) -> Result<u32, TranslateError> {
    u32::try_from(memarg.offset).map_err(|_| TranslateError::Unsupported)
}

fn local_get(locals: &[Kind], index: u32) -> Result<Op, TranslateError> {
    let kind = locals
        .get(index as usize)
        .ok_or(TranslateError::Unsupported)?;
    if *kind == Kind::I32 {
        Ok(Op::LocalI32(index))
    } else {
        Ok(Op::Local(index))
    }
}

fn block_ty(types: &[Sig], block: &BlockType) -> Result<BlockTy, TranslateError> {
    match block {
        BlockType::Empty => Ok(BlockTy {
            params: Vec::new(),
            results: Vec::new(),
        }),
        BlockType::Type(val) => Ok(BlockTy {
            params: Vec::new(),
            results: vec![one_kind(val)?],
        }),
        BlockType::FuncType(index) => func_block(types, *index),
    }
}

fn func_block(types: &[Sig], index: u32) -> Result<BlockTy, TranslateError> {
    let sig = types
        .get(index as usize)
        .ok_or(TranslateError::Unsupported)?;
    Ok(BlockTy {
        params: prefix(&sig.param_kind, sig.params),
        results: prefix(&sig.result_kind, sig.results),
    })
}

fn prefix(kinds: &[Kind], n: u32) -> Vec<Kind> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < n as usize && index < kinds.len() {
        out.push(kinds[index]);
        index += 1;
    }
    out
}

pub(super) fn one_kind(val: &wasmparser::ValType) -> Result<Kind, TranslateError> {
    match val {
        wasmparser::ValType::I32 => Ok(Kind::I32),
        wasmparser::ValType::F32 => Ok(Kind::F32),
        _ => Err(TranslateError::Unsupported),
    }
}
