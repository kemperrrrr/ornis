//! Lift a compute export (`compute` or `cull`) into a toy module with
//! `cs_main`.
//!
//! `rustc` emits a memory section for the cdylib. The translator rejects
//! that, so this lifter keeps only the functions the entry reaches: arm
//! calls, guards, and loops. Vertex and fragment stubs exist because the
//! translator still requires those entries.

use ornis_gpu_abi::symbols;
use wasmparser::Operator;
use wasmparser::Parser;
use wasmparser::Payload;
use wasmparser::TypeRef;
use wasmparser::ValType;

pub(super) fn lift(wasm: &[u8], export: &str) -> Vec<u8> {
    let wat = lift_wat(wasm, export);
    wat::parse_str(&wat).unwrap_or_else(|err| panic!("compute wat: {err}\n{wat}"))
}

/// The toy-module text `lift` parses: exposed so a translate failure can
/// print exactly what the translator saw.
pub(super) fn lift_wat(wasm: &[u8], export: &str) -> String {
    let module = read(wasm, export);
    wrap(&module)
}

struct Func {
    params: Result<String, String>,
    /// Locals and operators. `Err` is an op this lifter does not keep.
    /// Only a function `compute` reaches turns that into a panic.
    body: Result<(String, String), String>,
}

struct Read {
    imports: Vec<String>,
    bodies: Vec<Func>,
    compute: u32,
}

fn read(wasm: &[u8], export: &str) -> Read {
    let mut imports = Vec::new();
    let mut types = Vec::new();
    let mut func_types = Vec::new();
    let mut bodies = Vec::new();
    let mut compute = None;
    for payload in Parser::new(0).parse_all(wasm) {
        let payload = payload.unwrap_or_else(|err| panic!("wasm: {err}"));
        match payload {
            Payload::TypeSection(reader) => read_types(&mut types, reader),
            Payload::ImportSection(reader) => read_imports(&mut imports, reader),
            Payload::FunctionSection(reader) => {
                for index in reader {
                    func_types.push(index.expect("func type"));
                }
            }
            Payload::ExportSection(reader) => {
                for item in reader {
                    let item = item.expect("export");
                    if item.name == export {
                        compute = Some(item.index);
                    }
                }
            }
            Payload::CodeSectionEntry(body) => {
                let slot = bodies.len();
                let ty = func_types[slot] as usize;
                bodies.push(func_of(&imports, &types[ty], &body));
            }
            _ => {}
        }
    }
    Read {
        imports,
        bodies,
        compute: compute.unwrap_or_else(|| panic!("missing {export} export")),
    }
}

fn read_types(
    types: &mut Vec<Result<(String, String), String>>,
    reader: wasmparser::SectionLimited<'_, wasmparser::RecGroup>,
) {
    for group in reader {
        let group = group.expect("type");
        for ty in group.types() {
            let wasmparser::CompositeInnerType::Func(func) = &ty.composite_type.inner else {
                continue;
            };
            types.push(func_ty(func.params(), func.results()));
        }
    }
}

fn func_ty(params: &[ValType], results: &[ValType]) -> Result<(String, String), String> {
    Ok((val_list(params)?, val_list(results)?))
}

fn read_imports(imports: &mut Vec<String>, reader: wasmparser::ImportSectionReader<'_>) {
    for item in reader {
        let item = item.expect("import");
        let wasmparser::Imports::Single(_, import) = item else {
            panic!("compact imports are not the compute module");
        };
        if matches!(import.ty, TypeRef::Func(_)) {
            imports.push(import.name.to_string());
        }
    }
}

fn func_of(
    imports: &[String],
    ty: &Result<(String, String), String>,
    body: &wasmparser::FunctionBody<'_>,
) -> Func {
    let params = match ty {
        Ok((params, _)) => Ok(params.clone()),
        Err(err) => Err(err.clone()),
    };
    Func {
        params,
        body: lift_body(imports, body),
    }
}

fn lift_body(
    imports: &[String],
    body: &wasmparser::FunctionBody<'_>,
) -> Result<(String, String), String> {
    Ok((locals_of(body)?, ops_of(imports, body)?))
}

fn locals_of(body: &wasmparser::FunctionBody<'_>) -> Result<String, String> {
    let mut text = String::new();
    let reader = body.get_locals_reader().expect("locals");
    for local in reader {
        let (count, ty) = local.expect("local");
        let name = val_name(ty)?;
        let mut index = 0u32;
        while index < count {
            text.push_str(&format!("(local {name})"));
            index += 1;
        }
    }
    Ok(text)
}

fn ops_of(imports: &[String], body: &wasmparser::FunctionBody<'_>) -> Result<String, String> {
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
        let payload = bits & 0x007f_ffff;
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

fn wrap(module: &Read) -> String {
    let mut imports = String::new();
    for name in &module.imports {
        imports.push_str(&format!(
            "(import \"{m}\" \"{name}\" (func ${name} {sig}))\n",
            m = symbols::MODULE,
            sig = import_sig(name),
        ));
    }
    let mut funcs = String::new();
    let base = module.imports.len() as u32;
    let mut index = 0u32;
    while (index as usize) < module.bodies.len() {
        let wasm_index = base + index;
        if !reaches(module, wasm_index) {
            index += 1;
            continue;
        }
        let func = &module.bodies[index as usize];
        let (locals, ops) = func
            .body
            .as_ref()
            .unwrap_or_else(|op| panic!("compute reaches an op the lifter will not keep: {op}"));
        let export = if wasm_index == module.compute {
            format!("(export \"{}\")", symbols::CS_MAIN)
        } else {
            String::new()
        };
        let params = param_list(func.params.as_deref().unwrap_or(""));
        funcs.push_str(&format!(
            "(func $f{wasm_index} {export} {params} {locals} {ops})\n",
        ));
        index += 1;
    }
    format!(
        r#"(module
            {imports}
            {funcs}
            (func (export "{vs}") (param f32 f32 f32 f32) (result f32 f32 f32 f32 f32 f32)
                local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)
            (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
                f32.const 0 f32.const 0 f32.const 0 f32.const 1))"#,
        vs = symbols::VS_MAIN,
        fs = symbols::FS_MAIN,
    )
}

fn param_list(params: &str) -> String {
    let params = params.trim();
    if params.is_empty() {
        String::new()
    } else {
        format!("(param {params})")
    }
}

fn reaches(module: &Read, wasm_index: u32) -> bool {
    let base = module.imports.len() as u32;
    let mut seen = vec![false; module.bodies.len()];
    let mut stack = vec![module.compute];
    while let Some(index) = stack.pop() {
        if index < base {
            continue;
        }
        let slot = (index - base) as usize;
        if slot >= module.bodies.len() || seen[slot] {
            continue;
        }
        seen[slot] = true;
        if let Ok((_, ops)) = &module.bodies[slot].body {
            push_calls(&mut stack, ops);
        }
    }
    seen.get((wasm_index - base) as usize) == Some(&true)
}

fn push_calls(stack: &mut Vec<u32>, ops: &str) {
    for word in ops.split_whitespace() {
        let Some(rest) = word.strip_prefix("$f") else {
            continue;
        };
        let end = rest
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(rest.len());
        if let Ok(index) = rest[..end].parse() {
            stack.push(index);
        }
    }
}

fn import_sig(name: &str) -> &'static str {
    match name {
        symbols::STORAGE_WRITE_F32 => "(param i32 i32 i32 f32)",
        symbols::STORAGE_READ_F32 => "(param i32 i32 i32) (result f32)",
        symbols::UNIFORM_FIELD_F32 => "(param i32 i32 i32 i32) (result f32)",
        symbols::UNIFORM_FIELD_U32 => "(param i32 i32 i32 i32) (result i32)",
        symbols::UNIFORM_READ_F32 => "(param i32 i32 i32 i32) (result f32)",
        symbols::STORAGE_READ_U32 => "(param i32 i32 i32) (result i32)",
        symbols::STORAGE_WRITE_U32 => "(param i32 i32 i32 i32)",
        other => panic!("compute imports {other}, which the lifter does not know"),
    }
}
