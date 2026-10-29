//! Rebuild the toy vertex/fragment module from the scalar `shade` export.
//!
//! `rustc` still emits a memory section and runtime globals for a `cdylib`.
//! The translator rejects those. This lifter keeps only `shade`'s body, which
//! must be scalar imports, and wraps it as `fs_main`.

use ornis_gpu_abi::symbols;
use wasmparser::Operator;
use wasmparser::Parser;
use wasmparser::Payload;
use wasmparser::TypeRef;
use wasmparser::ValType;

pub(super) fn lift(wasm: &[u8]) -> Vec<u8> {
    let module = read(wasm);
    let wat = wrap(&module);
    wat::parse_str(&wat).unwrap_or_else(|err| panic!("lifted wat: {err}\n{wat}"))
}

struct FuncBody {
    locals: String,
    ops: String,
}

struct Read {
    imports: Vec<String>,
    shade: Option<u32>,
    bodies: Vec<FuncBody>,
}

fn read(wasm: &[u8]) -> Read {
    let mut out = Read {
        imports: Vec::new(),
        shade: None,
        bodies: Vec::new(),
    };
    for payload in Parser::new(0).parse_all(wasm) {
        let payload = payload.unwrap_or_else(|err| panic!("wasm: {err}"));
        match payload {
            Payload::ImportSection(reader) => read_imports(&mut out, reader),
            Payload::ExportSection(reader) => read_export(&mut out, reader),
            Payload::CodeSectionEntry(body) => out.bodies.push(body_of(&out.imports, &body)),
            _ => {}
        }
    }
    out
}

fn read_imports(out: &mut Read, reader: wasmparser::ImportSectionReader<'_>) {
    for item in reader {
        let item = item.expect("import");
        let wasmparser::Imports::Single(_, import) = item else {
            panic!("compact imports are not the shade module");
        };
        if !matches!(import.ty, TypeRef::Func(_)) {
            continue;
        }
        if import.module != symbols::MODULE {
            panic!("unexpected import module {}", import.module);
        }
        let _ = import_sig(import.name);
        out.imports.push(import.name.to_string());
    }
}

fn read_export(out: &mut Read, reader: wasmparser::ExportSectionReader<'_>) {
    for export in reader {
        let export = export.expect("export");
        if export.name == "shade" && matches!(export.kind, wasmparser::ExternalKind::Func) {
            out.shade = Some(export.index);
        }
    }
}

fn body_of(imports: &[String], body: &wasmparser::FunctionBody<'_>) -> FuncBody {
    FuncBody {
        locals: locals_of(body),
        ops: ops_of(imports, body),
    }
}

fn locals_of(body: &wasmparser::FunctionBody<'_>) -> String {
    let mut text = String::new();
    let reader = body.get_locals_reader().expect("locals");
    for local in reader {
        let (count, ty) = local.expect("local");
        let name = match ty {
            ValType::I32 => "i32",
            ValType::F32 => "f32",
            other => panic!("shade local {other:?} is not a scalar"),
        };
        let mut index = 0u32;
        while index < count {
            text.push_str(&format!("(local {name}) "));
            index += 1;
        }
    }
    text
}

fn ops_of(imports: &[String], body: &wasmparser::FunctionBody<'_>) -> String {
    let reader = body.get_operators_reader().expect("ops");
    let mut ops = Vec::new();
    for op in reader {
        let op = op.expect("op");
        if matches!(op, Operator::End) {
            continue;
        }
        ops.push(op_wat(imports, &op));
    }
    ops.join(" ")
}

fn op_wat(imports: &[String], op: &Operator<'_>) -> String {
    match op {
        Operator::I32Const { value } => format!("i32.const {value}"),
        Operator::F32Const { value } => format!("f32.const 0x{:08x}", value.bits()),
        Operator::LocalGet { local_index } => format!("local.get {local_index}"),
        Operator::LocalSet { local_index } => format!("local.set {local_index}"),
        Operator::LocalTee { local_index } => format!("local.tee {local_index}"),
        Operator::Call { function_index } => {
            let name = imports
                .get(*function_index as usize)
                .unwrap_or_else(|| panic!("shade calls a non-import function {function_index}"));
            format!("call ${name}")
        }
        Operator::F32Mul => "f32.mul".to_string(),
        Operator::F32Add => "f32.add".to_string(),
        Operator::Drop => "drop".to_string(),
        Operator::Nop => String::new(),
        other => panic!("shade uses an op the lifter will not keep: {other:?}"),
    }
}

fn import_sig(name: &str) -> &'static str {
    match name {
        symbols::UNIFORM_FIELD_F32 => "(param i32 i32 i32 i32) (result f32)",
        symbols::UNIFORM_FIELD_U32 => "(param i32 i32 i32 i32) (result i32)",
        symbols::UNIFORM_READ_F32 => "(param i32 i32 i32 i32) (result f32)",
        symbols::STORAGE_READ_F32 => "(param i32 i32 i32) (result f32)",
        symbols::STORAGE_WRITE_F32 => "(param i32 i32 i32 f32)",
        other => panic!("shade imports {other}, which the lifter does not know"),
    }
}

fn wrap(module: &Read) -> String {
    let index = module.shade.expect("missing shade export") as usize;
    let body = module
        .bodies
        .get(index - module.imports.len())
        .expect("shade body");
    let mut imports = String::new();
    for name in &module.imports {
        imports.push_str(&format!(
            "(import \"{m}\" \"{name}\" (func ${name} {sig}))\n",
            m = symbols::MODULE,
            sig = import_sig(name),
        ));
    }
    format!(
        r#"(module
            (@custom "{types}" "\4f\47\54\31\01\00\04\00\02\01\02\02")
            {imports}
            (func (export "{vs}") (param f32 f32 f32 f32) (result f32 f32 f32 f32 f32 f32)
                local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)
            (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
                {locals}
                {ops}
                f32.const 0 f32.const 0 f32.const 1))"#,
        types = symbols::TYPES_SECTION,
        vs = symbols::VS_MAIN,
        fs = symbols::FS_MAIN,
        locals = body.locals,
        ops = body.ops,
    )
}
