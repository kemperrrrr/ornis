//! Workgroup aggregates must validate under spirv-val from spirv-tools 2026.1.
//!
//! That validator rejects explicit layout decorations (`ArrayStride`, `Offset`)
//! on types used by a `Workgroup` variable. Storage buffers still require them.
//! The gpu CI job sets `ORNIS_REQUIRE_SPIRV_VAL`, so a missing `spirv-val`
//! fails here instead of skipping.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;

const OP_MASK: u32 = 0xFFFF;
const WORD_SHIFT: u32 = 16;
const HEADER_LEN: usize = 5;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const SC_WORKGROUP: u32 = 4;
const SC_STORAGE_BUFFER: u32 = 12;
const DEC_BLOCK: u32 = 2;
const DEC_BUFFER_BLOCK: u32 = 3;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_MATRIX_STRIDE: u32 = 7;
const DEC_OFFSET: u32 = 35;

#[test]
fn workgroup_array_passes_spirv_val_without_layout_decorations() {
    let translated = translate(&wasm()).expect("translate workgroup fixture");
    let words = &translated.compute_spirv;
    assert!(!words.is_empty());
    let workgroup = storage_types(words, SC_WORKGROUP);
    assert!(
        !workgroup.is_empty(),
        "fixture did not emit a Workgroup variable"
    );
    assert!(
        layout_targets(words, &workgroup).is_empty(),
        "layout decoration targets a Workgroup type: {}",
        layout_targets(words, &workgroup)
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let storage = storage_types(words, SC_STORAGE_BUFFER);
    assert!(
        layout_targets(words, &storage).contains(&DEC_ARRAY_STRIDE),
        "storage buffer lost ArrayStride"
    );
    super::compute_abi::validate_compute_spirv(words);
}

fn wasm() -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
            (import "{module}" "{store}" (func $store (param i32 i32)))
            (import "{module}" "{load}" (func $load (param i32) (result i32)))
            (import "{module}" "{atomic}" (func $atomic (param i32 i32) (result i32)))
            (import "{module}" "{write}" (func $write (param i32 i32 i32 i32)))
            (func (export "{vs}") (param f32 f32 f32 f32)
                (result f32 f32 f32 f32 f32 f32)
                local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)
            (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
                f32.const 0 f32.const 0 f32.const 0 f32.const 1)
            (func (export "{cs}") (param i32)
                i32.const 0 i32.const 7 call $store
                i32.const 0 i32.const 1 i32.const 0
                i32.const 0 call $load
                call $write
                i32.const 1 i32.const 5 call $atomic
                drop))"#,
        module = symbols::MODULE,
        store = symbols::WORKGROUP_STORE_U32,
        load = symbols::WORKGROUP_LOAD_U32,
        atomic = symbols::WORKGROUP_ATOMIC_ADD_U32,
        write = symbols::STORAGE_WRITE_U32,
        vs = symbols::VS_MAIN,
        fs = symbols::FS_MAIN,
        cs = symbols::CS_MAIN,
    ))
    .expect("valid workgroup layout fixture")
}

fn storage_types(words: &[u32], class: u32) -> Vec<u32> {
    let insts = instructions(words);
    let mut roots = Vec::new();
    for inst in &insts {
        if opcode(inst) == OP_VARIABLE && inst.get(3) == Some(&class) {
            roots.push(inst[1]);
        }
    }
    let mut seen = Vec::new();
    let mut stack = roots;
    while let Some(id) = stack.pop() {
        if seen.contains(&id) {
            continue;
        }
        seen.push(id);
        push_members(&insts, id, &mut stack);
    }
    seen
}

fn push_members(insts: &[&[u32]], id: u32, stack: &mut Vec<u32>) {
    let Some(inst) = insts.iter().find(|inst| result_id(inst) == Some(id)) else {
        return;
    };
    match opcode(inst) {
        OP_TYPE_ARRAY | OP_TYPE_RUNTIME_ARRAY => stack.push(inst[2]),
        OP_TYPE_STRUCT => stack.extend(inst.iter().skip(2).copied()),
        OP_TYPE_POINTER => stack.push(inst[3]),
        _ => {}
    }
}

fn layout_targets(words: &[u32], types: &[u32]) -> Vec<u32> {
    let mut found = Vec::new();
    for inst in instructions(words) {
        let Some(target) = decoration_target(inst) else {
            continue;
        };
        if types.contains(&target) {
            let dec = decoration(inst);
            if is_layout(dec) {
                found.push(dec);
            }
        }
    }
    found
}

fn decoration_target(inst: &[u32]) -> Option<u32> {
    let op = opcode(inst);
    if op == OP_DECORATE || op == OP_MEMBER_DECORATE {
        return inst.get(1).copied();
    }
    None
}

fn decoration(inst: &[u32]) -> u32 {
    if opcode(inst) == OP_MEMBER_DECORATE {
        return inst[3];
    }
    inst[2]
}

fn is_layout(dec: u32) -> bool {
    matches!(
        dec,
        DEC_BLOCK | DEC_BUFFER_BLOCK | DEC_ARRAY_STRIDE | DEC_MATRIX_STRIDE | DEC_OFFSET
    )
}

fn instructions(words: &[u32]) -> Vec<&[u32]> {
    let mut out = Vec::new();
    let mut cursor = HEADER_LEN;
    while cursor < words.len() {
        let count = (words[cursor] >> WORD_SHIFT) as usize;
        out.push(&words[cursor..cursor + count]);
        cursor += count;
    }
    out
}

fn opcode(inst: &[u32]) -> u32 {
    inst[0] & OP_MASK
}

fn result_id(inst: &[u32]) -> Option<u32> {
    let op = opcode(inst);
    let typed = op == OP_TYPE_ARRAY
        || op == OP_TYPE_RUNTIME_ARRAY
        || op == OP_TYPE_STRUCT
        || op == OP_TYPE_POINTER
        || op == OP_VARIABLE;
    typed.then(|| if op == OP_VARIABLE { inst[2] } else { inst[1] })
}
