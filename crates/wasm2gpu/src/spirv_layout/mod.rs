//! Drop explicit-layout decorations from SPIR-V types that cannot carry them.
//!
//! Naga decorates every array with `ArrayStride` and every struct member with
//! `Offset`. Vulkan's VUID-10684 rejects those decorations on a variable whose
//! storage class is not explicitly laid out: `Workgroup` always (unless the
//! module enables `WorkgroupMemoryExplicitLayoutKHR`), `UniformConstant`
//! always, and `Function` or `Private` when the module is SPIR-V 1.5 or
//! newer. `StorageBuffer`, `Uniform`, and `PushConstant` keep the decorations.
//! A type used from both sides is cloned; the copy referenced by the
//! non-explicit variable has no layout decorations.

use std::collections::HashSet;

use crate::error::TranslateError;

pub(super) const MAGIC: u32 = 0x0723_0203;
pub(super) const HEADER_LEN: usize = 5;
pub(super) const OP_MASK: u32 = 0xFFFF;
pub(super) const WORD_SHIFT: u32 = 16;

pub(super) const OP_CAPABILITY: u32 = 17;
pub(super) const OP_TYPE_VOID: u32 = 19;
pub(super) const OP_TYPE_FUNCTION: u32 = 33;
pub(super) const OP_TYPE_ARRAY: u32 = 28;
pub(super) const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
pub(super) const OP_TYPE_STRUCT: u32 = 30;
pub(super) const OP_TYPE_POINTER: u32 = 32;
pub(super) const OP_VARIABLE: u32 = 59;
pub(super) const OP_LOAD: u32 = 61;
pub(super) const OP_STORE: u32 = 62;
pub(super) const OP_DECORATE: u32 = 71;
pub(super) const OP_MEMBER_DECORATE: u32 = 72;

pub(super) const SC_UNIFORM_CONSTANT: u32 = 0;
pub(super) const SC_WORKGROUP: u32 = 4;
pub(super) const SC_PRIVATE: u32 = 6;
pub(super) const SC_FUNCTION: u32 = 7;

pub(super) const DEC_BLOCK: u32 = 2;
pub(super) const DEC_BUFFER_BLOCK: u32 = 3;
pub(super) const DEC_ARRAY_STRIDE: u32 = 6;
pub(super) const DEC_MATRIX_STRIDE: u32 = 7;
pub(super) const DEC_OFFSET: u32 = 35;

const CAP_WORKGROUP_EXPLICIT: u32 = 4428;
pub(super) const SPIRV_1_4: u32 = 0x0001_0400;
pub(super) const WORD_CLASS: usize = 3;
pub(super) const WORD_INIT: usize = 4;
pub(super) const POINTER_WORDS: usize = 4;
pub(super) const CLASS_WORDS: usize = 3;
pub(super) const ARRAY_END: usize = 3;
pub(super) const MEMBER_DEC_AT: usize = 3;

/// Opcodes that carry a result type in word 1 and a result id in word 2.
pub(super) const TYPED_VALUE_OPS: &[u32] = &[
    1, 43, 44, 46, 50, 51, 55, 57, 59, 61, 65, 66, 67, 70, 80, 81, 83, 124, 169, 245,
];

/// Constants whose result type may need to follow a cloned aggregate.
pub(super) const CONSTANT_OPS: &[u32] = &[43, 44, 46, 50, 51];

pub(super) struct Module {
    header: [u32; 5],
    insts: Vec<Vec<u32>>,
}

pub(super) struct Pass {
    version: u32,
    workgroup_explicit: bool,
}

struct Var {
    class: u32,
    pointer_ty: u32,
}

macro_rules! opcode {
    ($inst:expr) => {
        $inst.first().copied().unwrap_or(0) & OP_MASK
    };
}

mod clone;
mod retarget;

/// Rewrite `words` so non-explicit storage classes do not use layout decorations.
pub(super) fn legalize(words: Vec<u32>) -> Result<Vec<u32>, TranslateError> {
    let module = parse(words)?;
    let pass = read_pass(&module);
    let vars = variables(&module.insts);
    let forbidden = forbidden_ids(&module.insts, &vars, &pass);
    let allowed = allowed_ids(&module.insts, &vars, &pass);
    let cloning = clone::cloning_ids(&module.insts, &forbidden, &allowed);
    let (module, clones) = clone::clone_types(module, &cloning)?;
    let module = retarget::retarget_members(module, &forbidden, &cloning, &clones);
    let module = retarget::retarget_pointers(module, &pass, &clones);
    let module = retarget::retarget_values(module, &pass, &clones);
    let module = retarget::strip(module, &forbidden, &cloning);
    Ok(retarget::pack(module))
}

fn parse(words: Vec<u32>) -> Result<Module, TranslateError> {
    if words.len() < HEADER_LEN || words[0] != MAGIC {
        return Err(TranslateError::Naga(
            "spirv layout: truncated module".into(),
        ));
    }
    let mut header = [0; HEADER_LEN];
    header.copy_from_slice(&words[..HEADER_LEN]);
    let mut insts = Vec::new();
    let mut cursor = HEADER_LEN;
    while cursor < words.len() {
        let count = (words[cursor] >> WORD_SHIFT) as usize;
        if count == 0 || cursor + count > words.len() {
            return Err(TranslateError::Naga("spirv layout: bad instruction".into()));
        }
        insts.push(words[cursor..cursor + count].to_vec());
        cursor += count;
    }
    Ok(Module { header, insts })
}

fn read_pass(module: &Module) -> Pass {
    let mut workgroup_explicit = false;
    let mut index = 0;
    while index < module.insts.len() {
        let inst = &module.insts[index];
        if opcode!(inst) == OP_CAPABILITY && inst.get(1) == Some(&CAP_WORKGROUP_EXPLICIT) {
            workgroup_explicit = true;
        }
        index += 1;
    }
    Pass {
        version: module.header[1],
        workgroup_explicit,
    }
}

fn variables(insts: &[Vec<u32>]) -> Vec<Var> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < insts.len() {
        let inst = &insts[index];
        if opcode!(inst) == OP_VARIABLE && inst.len() > WORD_CLASS {
            out.push(Var {
                pointer_ty: inst[1],
                class: inst[3],
            });
        }
        index += 1;
    }
    out
}

fn forbidden_ids(insts: &[Vec<u32>], vars: &[Var], pass: &Pass) -> HashSet<u32> {
    let mut roots = Vec::new();
    let mut index = 0;
    while index < vars.len() {
        if blocked(vars[index].class, pass) {
            roots.push(vars[index].pointer_ty);
        }
        index += 1;
    }
    walk(insts, &roots, pass, true)
}

fn allowed_ids(insts: &[Vec<u32>], vars: &[Var], pass: &Pass) -> HashSet<u32> {
    let mut roots = Vec::new();
    let mut index = 0;
    while index < vars.len() {
        if !blocked(vars[index].class, pass)
            && let Some(pointee) = pointer_pointee(insts, vars[index].pointer_ty)
        {
            roots.push(pointee);
        }
        index += 1;
    }
    walk(insts, &roots, pass, false)
}

pub(super) fn blocked(class: u32, pass: &Pass) -> bool {
    if class == SC_WORKGROUP {
        return !pass.workgroup_explicit;
    }
    if class == SC_UNIFORM_CONSTANT {
        return true;
    }
    if class == SC_PRIVATE || class == SC_FUNCTION {
        return pass.version > SPIRV_1_4;
    }
    false
}

fn pointer_pointee(insts: &[Vec<u32>], pointer_ty: u32) -> Option<u32> {
    let inst = find_type(insts, pointer_ty)?;
    if opcode!(inst) != OP_TYPE_POINTER || inst.len() < POINTER_WORDS {
        return None;
    }
    Some(inst[3])
}

fn walk(insts: &[Vec<u32>], roots: &[u32], pass: &Pass, into_pointers: bool) -> HashSet<u32> {
    let mut seen = HashSet::new();
    let mut stack = roots.to_vec();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        push_children(insts, id, pass, into_pointers, &mut stack);
    }
    seen
}

fn push_children(
    insts: &[Vec<u32>],
    id: u32,
    pass: &Pass,
    into_pointers: bool,
    stack: &mut Vec<u32>,
) {
    let Some(inst) = find_type(insts, id) else {
        return;
    };
    let op = opcode!(inst);
    if op == OP_TYPE_ARRAY || op == OP_TYPE_RUNTIME_ARRAY {
        if let Some(elem) = inst.get(2) {
            stack.push(*elem);
        }
        return;
    }
    if op == OP_TYPE_STRUCT {
        let mut member = 2;
        while member < inst.len() {
            stack.push(inst[member]);
            member += 1;
        }
        return;
    }
    if op != OP_TYPE_POINTER || !into_pointers || inst.len() < POINTER_WORDS {
        return;
    }
    if blocked(inst[2], pass) {
        stack.push(inst[3]);
    }
}

pub(super) fn find_type(insts: &[Vec<u32>], id: u32) -> Option<&[u32]> {
    let mut index = 0;
    while index < insts.len() {
        let inst = &insts[index];
        let op = opcode!(inst);
        let is_type = (OP_TYPE_VOID..=OP_TYPE_FUNCTION).contains(&op);
        if is_type && inst.get(1) == Some(&id) {
            return Some(inst.as_slice());
        }
        index += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::DEC_ARRAY_STRIDE;
    use super::DEC_BLOCK;
    use super::DEC_OFFSET;
    use super::MAGIC;
    use super::OP_DECORATE;
    use super::OP_MEMBER_DECORATE;
    use super::OP_STORE;
    use super::OP_TYPE_ARRAY;
    use super::OP_TYPE_POINTER;
    use super::OP_TYPE_STRUCT;
    use super::OP_VARIABLE;
    use super::SC_FUNCTION;
    use super::SC_PRIVATE;
    use super::SC_WORKGROUP;
    use super::SPIRV_1_4;
    use super::WORD_SHIFT;
    use super::legalize;

    const OP_TYPE_INT: u32 = 21;
    const OP_CONSTANT: u32 = 43;
    const OP_CONSTANT_NULL: u32 = 46;
    const SC_STORAGE_BUFFER: u32 = 12;

    const SPIRV_1_0: u32 = 0x0001_0000;
    const SPIRV_1_5: u32 = 0x0001_0500;
    const STRIDE: u32 = 4;
    const BOUND: u32 = 32;
    const UINT: u32 = 1;
    const LEN: u32 = 2;
    const ARR: u32 = 3;
    const STRUCT: u32 = 4;
    const PTR_WG: u32 = 5;
    const PTR_SSBO: u32 = 6;
    const VAR_WG: u32 = 7;
    const VAR_SSBO: u32 = 8;
    const NULL: u32 = 9;

    fn insn(op: u32, operands: &[u32]) -> Vec<u32> {
        let mut words = Vec::with_capacity(operands.len() + 1);
        let count = (operands.len() as u32) + 1;
        words.push((count << WORD_SHIFT) | op);
        words.extend_from_slice(operands);
        words
    }

    fn module(version: u32, body: Vec<Vec<u32>>) -> Vec<u32> {
        let mut words = vec![MAGIC, version, 0, BOUND, 0];
        for inst in body {
            words.extend(inst);
        }
        words
    }

    fn body(words: Vec<u32>) -> Vec<Vec<u32>> {
        super::parse(words).expect("parse fixture").insts
    }

    fn op_of(inst: &[u32]) -> u32 {
        inst.first().copied().unwrap_or(0) & super::OP_MASK
    }

    fn decorate_targets(insts: &[Vec<u32>], decoration: u32) -> Vec<u32> {
        let mut targets = Vec::new();
        for inst in insts {
            if op_of(inst) == OP_DECORATE && inst.get(2) == Some(&decoration) {
                targets.push(inst[1]);
            }
        }
        targets
    }

    fn pointer_pointee(insts: &[Vec<u32>], pointer_id: u32) -> u32 {
        for inst in insts {
            if op_of(inst) == OP_TYPE_POINTER && inst[1] == pointer_id {
                return inst[3];
            }
        }
        panic!("missing pointer {pointer_id}");
    }

    fn type_int() -> Vec<u32> {
        insn(OP_TYPE_INT, &[UINT, STRIDE, 0])
    }

    fn array_len() -> Vec<u32> {
        insn(OP_CONSTANT, &[UINT, LEN, STRIDE])
    }

    #[test]
    fn workgroup_array_loses_array_stride() {
        let words = module(
            SPIRV_1_0,
            vec![
                insn(OP_DECORATE, &[ARR, DEC_ARRAY_STRIDE, STRIDE]),
                type_int(),
                array_len(),
                insn(OP_TYPE_ARRAY, &[ARR, UINT, LEN]),
                insn(OP_TYPE_POINTER, &[PTR_WG, SC_WORKGROUP, ARR]),
                insn(OP_VARIABLE, &[PTR_WG, VAR_WG, SC_WORKGROUP]),
            ],
        );
        let legal = body(legalize(words).expect("legalize"));
        assert!(decorate_targets(&legal, DEC_ARRAY_STRIDE).is_empty());
        assert!(pointer_pointee(&legal, PTR_WG) == ARR);
    }

    #[test]
    fn storage_buffer_keeps_array_stride() {
        let words = module(
            SPIRV_1_0,
            vec![
                insn(OP_DECORATE, &[ARR, DEC_ARRAY_STRIDE, STRIDE]),
                insn(OP_DECORATE, &[STRUCT, DEC_BLOCK]),
                insn(OP_MEMBER_DECORATE, &[STRUCT, 0, DEC_OFFSET, 0]),
                type_int(),
                array_len(),
                insn(OP_TYPE_ARRAY, &[ARR, UINT, LEN]),
                insn(OP_TYPE_STRUCT, &[STRUCT, ARR]),
                insn(OP_TYPE_POINTER, &[PTR_SSBO, SC_STORAGE_BUFFER, STRUCT]),
                insn(OP_VARIABLE, &[PTR_SSBO, VAR_SSBO, SC_STORAGE_BUFFER]),
            ],
        );
        let legal = legalize(words.clone()).expect("legalize");
        assert!(legal == words);
    }

    #[test]
    fn shared_array_is_cloned_for_workgroup() {
        let words = module(
            SPIRV_1_0,
            vec![
                insn(OP_DECORATE, &[ARR, DEC_ARRAY_STRIDE, STRIDE]),
                insn(OP_DECORATE, &[STRUCT, DEC_BLOCK]),
                insn(OP_MEMBER_DECORATE, &[STRUCT, 0, DEC_OFFSET, 0]),
                type_int(),
                array_len(),
                insn(OP_TYPE_ARRAY, &[ARR, UINT, LEN]),
                insn(OP_TYPE_STRUCT, &[STRUCT, ARR]),
                insn(OP_TYPE_POINTER, &[PTR_WG, SC_WORKGROUP, ARR]),
                insn(OP_TYPE_POINTER, &[PTR_SSBO, SC_STORAGE_BUFFER, STRUCT]),
                insn(OP_VARIABLE, &[PTR_WG, VAR_WG, SC_WORKGROUP]),
                insn(OP_VARIABLE, &[PTR_SSBO, VAR_SSBO, SC_STORAGE_BUFFER]),
                insn(OP_CONSTANT_NULL, &[ARR, NULL]),
                insn(OP_STORE, &[VAR_WG, NULL]),
            ],
        );
        let legal = body(legalize(words).expect("legalize"));
        let clone = pointer_pointee(&legal, PTR_WG);
        assert!(clone != ARR);
        assert!(decorate_targets(&legal, DEC_ARRAY_STRIDE) == vec![ARR]);
        assert!(struct_member(&legal) == ARR);
        assert!(null_type(&legal) == clone);
    }

    fn struct_member(insts: &[Vec<u32>]) -> u32 {
        for inst in insts {
            if op_of(inst) == OP_TYPE_STRUCT && inst[1] == STRUCT {
                return inst[2];
            }
        }
        panic!("missing struct");
    }

    fn null_type(insts: &[Vec<u32>]) -> u32 {
        for inst in insts {
            if op_of(inst) == OP_CONSTANT_NULL {
                return inst[1];
            }
        }
        panic!("missing null");
    }

    #[test]
    fn function_array_keeps_stride_before_spirv_1_5() {
        let words = function_array(SPIRV_1_4);
        let legal = legalize(words.clone()).expect("legalize");
        assert!(legal == words);
    }

    #[test]
    fn function_array_loses_stride_at_spirv_1_5() {
        let legal = body(legalize(function_array(SPIRV_1_5)).expect("legalize"));
        assert!(decorate_targets(&legal, DEC_ARRAY_STRIDE).is_empty());
    }

    #[test]
    fn private_array_loses_stride_at_spirv_1_5() {
        let words = module(
            SPIRV_1_5,
            vec![
                insn(OP_DECORATE, &[ARR, DEC_ARRAY_STRIDE, STRIDE]),
                type_int(),
                array_len(),
                insn(OP_TYPE_ARRAY, &[ARR, UINT, LEN]),
                insn(OP_TYPE_POINTER, &[PTR_WG, SC_PRIVATE, ARR]),
                insn(OP_VARIABLE, &[PTR_WG, VAR_WG, SC_PRIVATE]),
            ],
        );
        let legal = body(legalize(words).expect("legalize"));
        assert!(decorate_targets(&legal, DEC_ARRAY_STRIDE).is_empty());
    }

    fn function_array(version: u32) -> Vec<u32> {
        module(
            version,
            vec![
                insn(OP_DECORATE, &[ARR, DEC_ARRAY_STRIDE, STRIDE]),
                type_int(),
                array_len(),
                insn(OP_TYPE_ARRAY, &[ARR, UINT, LEN]),
                insn(OP_TYPE_POINTER, &[PTR_WG, SC_FUNCTION, ARR]),
                insn(OP_VARIABLE, &[PTR_WG, VAR_WG, SC_FUNCTION]),
            ],
        )
    }
}
