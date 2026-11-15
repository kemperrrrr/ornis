//! Explicit-layout helpers for the SPIR-V legalizer.
//!
//! See the parent module. These functions clone or retarget types so a
//! Workgroup variable does not share an explicitly laid out aggregate with a
//! storage buffer.

use std::collections::HashMap;
use std::collections::HashSet;

use super::CLASS_WORDS;
use super::CONSTANT_OPS;
use super::HEADER_LEN;
use super::Module;
use super::OP_LOAD;
use super::OP_STORE;
use super::OP_TYPE_ARRAY;
use super::OP_TYPE_FUNCTION;
use super::OP_TYPE_POINTER;
use super::OP_TYPE_RUNTIME_ARRAY;
use super::OP_TYPE_STRUCT;
use super::OP_TYPE_VOID;
use super::OP_VARIABLE;
use super::Pass;
use super::TYPED_VALUE_OPS;
use super::WORD_CLASS;
use super::WORD_INIT;
use super::blocked;
use super::find_type;

macro_rules! opcode {
    ($inst:expr) => {
        $inst.first().copied().unwrap_or(0) & super::OP_MASK
    };
}

pub(super) fn retarget_members(
    mut module: Module,
    forbidden: &HashSet<u32>,
    cloning: &HashSet<u32>,
    clones: &HashMap<u32, u32>,
) -> Module {
    let mut index = 0;
    while index < module.insts.len() {
        let retarget = member_retarget(&module.insts[index], forbidden, cloning);
        if retarget {
            super::clone::rewrite_operands(&mut module.insts[index], clones);
        }
        index += 1;
    }
    module
}

fn member_retarget(inst: &[u32], forbidden: &HashSet<u32>, cloning: &HashSet<u32>) -> bool {
    let op = opcode!(inst);
    let composite = op == OP_TYPE_ARRAY || op == OP_TYPE_RUNTIME_ARRAY || op == OP_TYPE_STRUCT;
    if !composite || inst.len() < 2 {
        return false;
    }
    forbidden.contains(&inst[1]) && !cloning.contains(&inst[1])
}

pub(super) fn retarget_pointers(
    mut module: Module,
    pass: &Pass,
    clones: &HashMap<u32, u32>,
) -> Module {
    let mut index = 0;
    while index < module.insts.len() {
        let inst = &mut module.insts[index];
        if opcode!(inst) == OP_TYPE_POINTER
            && inst.len() > WORD_CLASS
            && blocked(inst[2], pass)
            && let Some(next) = clones.get(&inst[3])
        {
            inst[3] = *next;
        }
        index += 1;
    }
    module
}

pub(super) fn retarget_values(
    mut module: Module,
    pass: &Pass,
    clones: &HashMap<u32, u32>,
) -> Module {
    retarget_constants(&mut module, pass, clones);
    retarget_loads(&mut module.insts, pass, clones);
    module
}

fn retarget_constants(module: &mut Module, pass: &Pass, clones: &HashMap<u32, u32>) {
    let mut index = 0;
    while index < module.insts.len() {
        let op = opcode!(&module.insts[index]);
        let typed = CONSTANT_OPS.binary_search(&op).is_ok();
        let result_ty = module.insts[index].get(1).copied().unwrap_or(0);
        if typed && clones.contains_key(&result_ty) {
            let clone_ty = clones[&result_ty];
            let result_id = module.insts[index].get(2).copied().unwrap_or(0);
            if split_constant(module, index, result_id, clone_ty, pass) {
                index += 1;
            }
        }
        index += 1;
    }
}

fn split_constant(
    module: &mut Module,
    index: usize,
    result_id: u32,
    clone_ty: u32,
    pass: &Pass,
) -> bool {
    let uses = value_uses(&module.insts, result_id);
    let mut forbidden_uses = Vec::new();
    let mut kept = 0;
    let mut use_index = 0;
    while use_index < uses.len() {
        let (at, word) = uses[use_index];
        if use_is_forbidden(&module.insts, at, word, pass) {
            forbidden_uses.push((at, word));
        } else {
            kept += 1;
        }
        use_index += 1;
    }
    if forbidden_uses.is_empty() {
        return false;
    }
    if kept == 0 {
        module.insts[index][1] = clone_ty;
        return false;
    }
    let new_id = super::clone::alloc_id(&mut module.header);
    let mut cloned = module.insts[index].clone();
    cloned[1] = clone_ty;
    cloned[2] = new_id;
    let mut replace = 0;
    while replace < forbidden_uses.len() {
        let (at, word) = forbidden_uses[replace];
        module.insts[at][word] = new_id;
        replace += 1;
    }
    module.insts.insert(index + 1, cloned);
    true
}

fn value_uses(insts: &[Vec<u32>], id: u32) -> Vec<(usize, usize)> {
    let mut uses = Vec::new();
    let mut index = 0;
    while index < insts.len() {
        let skip = result_slot(&insts[index]);
        let mut word = 1;
        while word < insts[index].len() {
            if Some(word) != skip && insts[index][word] == id {
                uses.push((index, word));
            }
            word += 1;
        }
        index += 1;
    }
    uses
}

fn result_slot(inst: &[u32]) -> Option<usize> {
    let op = opcode!(inst);
    if TYPED_VALUE_OPS.binary_search(&op).is_ok() {
        return Some(2);
    }
    if (OP_TYPE_VOID..=OP_TYPE_FUNCTION).contains(&op) {
        return Some(1);
    }
    None
}

fn use_is_forbidden(insts: &[Vec<u32>], index: usize, word: usize, pass: &Pass) -> bool {
    let inst = &insts[index];
    let op = opcode!(inst);
    if op == OP_STORE && word == 2 && inst.len() > 2 {
        return pointer_blocked(insts, inst[1], pass);
    }
    if op == OP_VARIABLE && word == WORD_INIT && inst.len() > WORD_INIT {
        return blocked(inst[3], pass);
    }
    false
}

fn pointer_blocked(insts: &[Vec<u32>], pointer_id: u32, pass: &Pass) -> bool {
    let Some(pointer_ty) = value_type(insts, pointer_id) else {
        return false;
    };
    let Some(inst) = find_type(insts, pointer_ty) else {
        return false;
    };
    if opcode!(inst) != OP_TYPE_POINTER || inst.len() < CLASS_WORDS {
        return false;
    }
    blocked(inst[2], pass)
}

fn value_type(insts: &[Vec<u32>], id: u32) -> Option<u32> {
    let mut index = 0;
    while index < insts.len() {
        let inst = &insts[index];
        let op = opcode!(inst);
        if TYPED_VALUE_OPS.binary_search(&op).is_ok() && inst.get(2) == Some(&id) {
            return inst.get(1).copied();
        }
        index += 1;
    }
    None
}

fn retarget_loads(insts: &mut [Vec<u32>], pass: &Pass, clones: &HashMap<u32, u32>) {
    let mut index = 0;
    while index < insts.len() {
        let op = opcode!(&insts[index]);
        if op == OP_LOAD && insts[index].len() > WORD_CLASS {
            let result_ty = insts[index][1];
            let pointer = insts[index][3];
            if let Some(next) = clones.get(&result_ty)
                && pointer_blocked(insts, pointer, pass)
            {
                insts[index][1] = *next;
            }
        }
        index += 1;
    }
}

pub(super) fn strip(
    mut module: Module,
    forbidden: &HashSet<u32>,
    cloning: &HashSet<u32>,
) -> Module {
    module
        .insts
        .retain(|inst| keep_inst(inst, forbidden, cloning));
    module
}

fn keep_inst(inst: &[u32], forbidden: &HashSet<u32>, cloning: &HashSet<u32>) -> bool {
    let Some(target) = super::clone::layout_target(inst) else {
        return true;
    };
    !(forbidden.contains(&target) && !cloning.contains(&target))
}

pub(super) fn pack(module: Module) -> Vec<u32> {
    let mut out = Vec::with_capacity(HEADER_LEN + module.insts.iter().map(Vec::len).sum::<usize>());
    out.extend_from_slice(&module.header);
    let mut index = 0;
    while index < module.insts.len() {
        out.extend_from_slice(&module.insts[index]);
        index += 1;
    }
    out
}
