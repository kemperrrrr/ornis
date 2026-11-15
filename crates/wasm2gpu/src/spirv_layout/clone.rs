//! Explicit-layout helpers for the SPIR-V legalizer.
//!
//! See the parent module. These functions clone or retarget types so a
//! Workgroup variable does not share an explicitly laid out aggregate with a
//! storage buffer.

use std::collections::HashMap;
use std::collections::HashSet;

use super::ARRAY_END;
use super::DEC_ARRAY_STRIDE;
use super::DEC_BLOCK;
use super::DEC_BUFFER_BLOCK;
use super::DEC_MATRIX_STRIDE;
use super::DEC_OFFSET;
use super::HEADER_LEN;
use super::MEMBER_DEC_AT;
use super::Module;
use super::OP_DECORATE;
use super::OP_MEMBER_DECORATE;
use super::OP_TYPE_ARRAY;
use super::OP_TYPE_FUNCTION;
use super::OP_TYPE_RUNTIME_ARRAY;
use super::OP_TYPE_STRUCT;
use super::OP_TYPE_VOID;
use crate::error::TranslateError;

macro_rules! opcode {
    ($inst:expr) => {
        $inst.first().copied().unwrap_or(0) & super::OP_MASK
    };
}

pub(super) fn cloning_ids(
    insts: &[Vec<u32>],
    forbidden: &HashSet<u32>,
    allowed: &HashSet<u32>,
) -> HashSet<u32> {
    let mut cloning = HashSet::new();
    loop {
        let added = add_clones(insts, forbidden, allowed, &mut cloning);
        if added == 0 {
            break;
        }
    }
    cloning
}

fn add_clones(
    insts: &[Vec<u32>],
    forbidden: &HashSet<u32>,
    allowed: &HashSet<u32>,
    cloning: &mut HashSet<u32>,
) -> usize {
    let mut added = 0;
    for id in forbidden.intersection(allowed).copied() {
        if cloning.contains(&id) || !should_clone(insts, id, cloning) {
            continue;
        }
        cloning.insert(id);
        added += 1;
    }
    added
}

fn should_clone(insts: &[Vec<u32>], id: u32, cloning: &HashSet<u32>) -> bool {
    let Some(inst) = super::find_type(insts, id) else {
        return false;
    };
    let op = opcode!(inst);
    if op != OP_TYPE_ARRAY && op != OP_TYPE_RUNTIME_ARRAY && op != OP_TYPE_STRUCT {
        return false;
    }
    decorated(insts, id) || member_is_cloned(inst, cloning)
}

fn decorated(insts: &[Vec<u32>], id: u32) -> bool {
    let mut index = 0;
    while index < insts.len() {
        if layout_target(&insts[index]) == Some(id) {
            return true;
        }
        index += 1;
    }
    false
}

fn member_is_cloned(inst: &[u32], cloning: &HashSet<u32>) -> bool {
    let end = operand_end(opcode!(inst), inst.len());
    let mut index = 2;
    while index < end {
        if cloning.contains(&inst[index]) {
            return true;
        }
        index += 1;
    }
    false
}

fn operand_end(op: u32, len: usize) -> usize {
    if op == OP_TYPE_STRUCT {
        return len;
    }
    ARRAY_END.min(len)
}

pub(super) fn layout_target(inst: &[u32]) -> Option<u32> {
    let at = decoration_index(opcode!(inst))?;
    let dec = *inst.get(at)?;
    is_layout_dec(dec).then(|| inst.get(1).copied()).flatten()
}

fn decoration_index(op: u32) -> Option<usize> {
    if op == OP_DECORATE {
        return Some(2);
    }
    if op == OP_MEMBER_DECORATE {
        return Some(MEMBER_DEC_AT);
    }
    None
}

fn is_layout_dec(dec: u32) -> bool {
    dec == DEC_BLOCK
        || dec == DEC_BUFFER_BLOCK
        || dec == DEC_ARRAY_STRIDE
        || dec == DEC_MATRIX_STRIDE
        || dec == DEC_OFFSET
}

pub(super) fn clone_types(
    mut module: Module,
    cloning: &HashSet<u32>,
) -> Result<(Module, HashMap<u32, u32>), TranslateError> {
    let mut done = HashMap::new();
    let mut guard = cloning.len() + 1;
    while done.len() < cloning.len() {
        if guard == 0 {
            return Err(TranslateError::Naga("spirv layout: cyclic type".into()));
        }
        guard -= 1;
        let made = clone_ready(&mut module, cloning, &mut done)?;
        if made == 0 {
            return Err(TranslateError::Naga("spirv layout: cyclic type".into()));
        }
    }
    Ok((module, done))
}

fn clone_ready(
    module: &mut Module,
    cloning: &HashSet<u32>,
    done: &mut HashMap<u32, u32>,
) -> Result<usize, TranslateError> {
    let pending = pending_ids(cloning, done);
    let mut made = 0;
    for id in pending {
        if !deps_ready(&module.insts, id, cloning, done) {
            continue;
        }
        clone_one(module, id, done)?;
        made += 1;
    }
    Ok(made)
}

fn pending_ids(cloning: &HashSet<u32>, done: &HashMap<u32, u32>) -> Vec<u32> {
    let mut ids: Vec<u32> = cloning
        .iter()
        .copied()
        .filter(|id| !done.contains_key(id))
        .collect();
    ids.sort_unstable();
    ids
}

fn deps_ready(
    insts: &[Vec<u32>],
    id: u32,
    cloning: &HashSet<u32>,
    done: &HashMap<u32, u32>,
) -> bool {
    match super::find_type(insts, id) {
        Some(inst) => !member_waits(inst, cloning, done),
        None => true,
    }
}

fn member_waits(inst: &[u32], cloning: &HashSet<u32>, done: &HashMap<u32, u32>) -> bool {
    let end = operand_end(opcode!(inst), inst.len());
    let mut index = 2;
    while index < end {
        let member = inst[index];
        if cloning.contains(&member) && !done.contains_key(&member) {
            return true;
        }
        index += 1;
    }
    false
}

fn clone_one(
    module: &mut Module,
    id: u32,
    done: &mut HashMap<u32, u32>,
) -> Result<(), TranslateError> {
    let (index, words) = staged_clone(module, id, done)?;
    module.insts.insert(index + 1, words);
    Ok(())
}

fn staged_clone(
    module: &mut Module,
    id: u32,
    done: &mut HashMap<u32, u32>,
) -> Result<(usize, Vec<u32>), TranslateError> {
    let mut words = super::find_type(&module.insts, id)
        .map(<[u32]>::to_vec)
        .ok_or_else(|| TranslateError::Naga("spirv layout: missing type".into()))?;
    let new_id = alloc_id(&mut module.header);
    words[1] = new_id;
    rewrite_operands(&mut words, done);
    let index = type_index(&module.insts, id)
        .ok_or_else(|| TranslateError::Naga("spirv layout: missing type".into()))?;
    done.insert(id, new_id);
    Ok((index, words))
}

pub(super) fn rewrite_operands(words: &mut [u32], done: &HashMap<u32, u32>) {
    let op = opcode!(words);
    let end = if op == OP_TYPE_STRUCT {
        words.len()
    } else if op == OP_TYPE_ARRAY || op == OP_TYPE_RUNTIME_ARRAY {
        ARRAY_END.min(words.len())
    } else {
        return;
    };
    let mut index = 2;
    while index < end {
        if let Some(next) = done.get(&words[index]) {
            words[index] = *next;
        }
        index += 1;
    }
}

pub(super) fn alloc_id(header: &mut [u32; HEADER_LEN]) -> u32 {
    if header[3] == 0 {
        header[3] = 1;
    }
    let id = header[3];
    header[3] = id + 1;
    id
}

fn type_index(insts: &[Vec<u32>], id: u32) -> Option<usize> {
    let mut index = 0;
    while index < insts.len() {
        let inst = &insts[index];
        let op = opcode!(inst);
        let is_type = (OP_TYPE_VOID..=OP_TYPE_FUNCTION).contains(&op);
        if is_type && inst.get(1) == Some(&id) {
            return Some(index);
        }
        index += 1;
    }
    None
}
