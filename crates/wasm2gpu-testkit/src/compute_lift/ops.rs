//! Wasm operator → WAT lowering for the test lifter. Kept separate from
//! section parsing and module wrapping so each part stays independently
//! readable and below the module-size ratchet.

use wasmparser::Operator;
use wasmparser::ValType;

/// IEEE-754 mantissa mask used to print a NaN payload in WAT.
const F32_NAN_PAYLOAD_MASK: u32 = 0x007f_ffff;

// qual:allow(complexity, error_handling) reason: "Unreadable wasm ops fail this test."
pub(super) fn ops_of(
    imports: &[String],
    body: &wasmparser::FunctionBody<'_>,
) -> Result<String, String> {
    let reader = body.get_operators_reader().expect("ops");
    let mut ops = String::new();
    let mut depth = 0u32;
    for op in reader {
        let op = op.expect("op");
        if region_end(&op, &mut depth) {
            continue;
        }
        let text = op_wat(imports, &op)?;
        if text.is_empty() {
            continue;
        }
        ops.push_str(&text);
        ops.push(' ');
    }
    Ok(ops)
}

/// The function's own `end` is depth 0. Block `end`s stay in the text.
fn region_end(op: &Operator<'_>, depth: &mut u32) -> bool {
    if opens_region(op) {
        *depth += 1;
        return false;
    }
    if !matches!(op, Operator::End) {
        return false;
    }
    if *depth == 0 {
        return true;
    }
    *depth -= 1;
    false
}

fn opens_region(op: &Operator<'_>) -> bool {
    matches!(
        op,
        Operator::Block { .. } | Operator::Loop { .. } | Operator::If { .. }
    )
}

fn op_wat(imports: &[String], op: &Operator<'_>) -> Result<String, String> {
    match op {
        Operator::I32Const { value } => Ok(format!("i32.const {value}")),
        Operator::F32Const { value } => Ok(format!("f32.const {}", f32_text(value.bits()))),
        Operator::LocalGet { local_index } => Ok(format!("local.get {local_index}")),
        Operator::LocalSet { local_index } => Ok(format!("local.set {local_index}")),
        Operator::LocalTee { local_index } => Ok(format!("local.tee {local_index}")),
        Operator::Call { function_index } => Ok(call_wat(imports, *function_index)),
        Operator::Block { .. } => Ok("block".to_string()),
        Operator::Loop { .. } => Ok("loop".to_string()),
        Operator::If { .. } => Ok("if".to_string()),
        Operator::Else => Ok("else".to_string()),
        Operator::End => Ok("end".to_string()),
        Operator::Br { relative_depth } => Ok(format!("br {relative_depth}")),
        Operator::BrIf { relative_depth } => Ok(format!("br_if {relative_depth}")),
        Operator::Return => Ok("return".to_string()),
        Operator::BrTable { targets } => Ok(br_table(targets)),
        Operator::Drop => Ok("drop".to_string()),
        Operator::I32Add => Ok("i32.add".to_string()),
        Operator::I32Sub => Ok("i32.sub".to_string()),
        Operator::I32Mul => Ok("i32.mul".to_string()),
        Operator::I32Shl => Ok("i32.shl".to_string()),
        Operator::I32ShrU => Ok("i32.shr_u".to_string()),
        Operator::I32Eqz => Ok("i32.eqz".to_string()),
        Operator::I32Eq => Ok("i32.eq".to_string()),
        Operator::I32Ne => Ok("i32.ne".to_string()),
        Operator::I32LtU => Ok("i32.lt_u".to_string()),
        Operator::I32GtU => Ok("i32.gt_u".to_string()),
        Operator::I32GeU => Ok("i32.ge_u".to_string()),
        Operator::I32LeU => Ok("i32.le_u".to_string()),
        Operator::I32And => Ok("i32.and".to_string()),
        Operator::I32Or => Ok("i32.or".to_string()),
        Operator::I32Xor => Ok("i32.xor".to_string()),
        Operator::I32ShrS => Ok("i32.shr_s".to_string()),
        Operator::F32Add => Ok("f32.add".to_string()),
        Operator::F32Mul => Ok("f32.mul".to_string()),
        Operator::F32Lt => Ok("f32.lt".to_string()),
        Operator::Select => Ok("select".to_string()),
        Operator::I32Load { memarg } => Ok(mem_op("i32.load", memarg)),
        Operator::I32Store { memarg } => Ok(mem_op("i32.store", memarg)),
        Operator::F32Load { memarg } => Ok(mem_op("f32.load", memarg)),
        Operator::F32Store { memarg } => Ok(mem_op("f32.store", memarg)),
        Operator::Nop | Operator::Unreachable => Ok(String::new()),
        other => Err(format!("{other:?}")),
    }
}

fn f32_text(bits: u32) -> String {
    let value = f32::from_bits(bits);
    if value.is_nan() {
        let payload = bits & F32_NAN_PAYLOAD_MASK;
        let sign = if value.is_sign_negative() { "-" } else { "" };
        return format!("{sign}nan:0x{payload:x}");
    }
    // `wast` reads `0x3f800000` as the integer 1065353216, not as bits.
    // Debug is the shortest decimal that parses back to the same value.
    format!("{value:?}")
}

/// A memory op with its immediate. `wast` defaults `align` to the natural
/// alignment, so only a nonzero `offset` has to come back.
fn mem_op(name: &str, memarg: &wasmparser::MemArg) -> String {
    if memarg.offset == 0 {
        name.to_string()
    } else {
        format!("{name} offset={}", memarg.offset)
    }
}

fn call_wat(imports: &[String], index: u32) -> String {
    if let Some(name) = imports.get(index as usize) {
        return format!("call ${name}");
    }
    format!("call $f{index}")
}

fn br_table(targets: &wasmparser::BrTable<'_>) -> String {
    let mut text = String::from("br_table");
    for target in targets.targets() {
        let target = target.expect("br_table label");
        text.push_str(&format!(" {target}"));
    }
    text.push_str(&format!(" {}", targets.default()));
    text
}

fn val_list(vals: &[ValType]) -> Result<String, String> {
    let mut text = String::new();
    for val in vals {
        text.push_str(val_name(*val)?);
        text.push(' ');
    }
    Ok(text)
}

fn val_name(val: ValType) -> Result<&'static str, String> {
    match val {
        ValType::I32 => Ok("i32"),
        ValType::F32 => Ok("f32"),
        other => Err(format!("{other:?}")),
    }
}
