//! Rebuild the toy vertex/fragment module from the scalar `shade` export.
//!
//! `rustc` still emits a memory section and runtime globals for a `cdylib`.
//! The translator rejects those. This lifter keeps only `shade`'s body, which
//! must be scalar imports, and wraps it as `fs_main`.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu_shader::ShadeParams;
use ornis_wasm2gpu_shader::UniformF32Lane;
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
            Payload::CodeSectionEntry(body) => {
                // Only `shade` is rewritten. `compute` uses `br_table`, which
                // this lifter does not keep; its body is lifted separately.
                let wasm_index = (out.imports.len() + out.bodies.len()) as u32;
                let parsed = if out.shade == Some(wasm_index) {
                    body_of(&out.imports, &body)
                } else {
                    FuncBody {
                        locals: String::new(),
                        ops: String::new(),
                    }
                };
                out.bodies.push(parsed);
            }
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
        symbols::STORAGE_READ_U32 => "(param i32 i32 i32) (result i32)",
        symbols::STORAGE_WRITE_U32 => "(param i32 i32 i32 i32)",
        other => panic!("shade imports {other}, which the lifter does not know"),
    }
}

const fn shade_offsets(offsets: [u32; 4]) -> bool {
    offsets[0] == 0 && offsets[1] == 4 && offsets[2] == 8 && offsets[3] == 12
}

const fn shade_tags(tags: [u8; 4]) -> bool {
    tags[0] == ornis_gpu_abi::TYPE_TAG_U32
        && tags[1] == ornis_gpu_abi::TYPE_TAG_F32
        && tags[2] == ornis_gpu_abi::TYPE_TAG_U32
        && tags[3] == ornis_gpu_abi::TYPE_TAG_U32
}

fn types_body() -> String {
    // The section is ShadeParams' field tags. UniformF32Lane is the host
    // stride of the f32 array, not a wasm struct, so it stays out of the
    // section. A field-type change fails this assert before the bytes drift.
    const _: () = {
        assert!(shade_offsets(ShadeParams::FIELD_OFFSETS));
        assert!(shade_tags(ShadeParams::FIELD_TAGS));
        assert!(
            <ShadeParams as ornis_gpu_abi::GpuSafeLayout>::SIZE
                == ornis_gpu_abi::UNIFORM_STRUCT_ALIGN
        );
        assert!(shade_offsets(UniformF32Lane::FIELD_OFFSETS));
        assert!(
            <UniformF32Lane as ornis_gpu_abi::GpuSafeLayout>::SIZE
                == ornis_gpu_abi::UNIFORM_STRUCT_ALIGN
        );
        assert!(
            <UniformF32Lane as ornis_gpu_abi::GpuSafeLayout>::UNIFORM_STRIDE
                == ornis_gpu_abi::UNIFORM_STRUCT_ALIGN
        );
    };
    let bytes = ornis_gpu_abi::encode_types_section(&[&ShadeParams::FIELD_TAGS])
        .expect("shade types section");
    let mut body = String::new();
    for byte in &bytes {
        body.push_str(&format!("\\{byte:02x}"));
    }
    body
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
            (@custom "{types}" "{section}")
            {imports}
            (func (export "{vs}") (param f32 f32 f32 f32) (result f32 f32 f32 f32 f32 f32)
                local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)
            (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
                {locals}
                {ops}
                f32.const 0 f32.const 0 f32.const 1))"#,
        types = symbols::TYPES_SECTION,
        section = types_body(),
        vs = symbols::VS_MAIN,
        fs = symbols::FS_MAIN,
        locals = body.locals,
        ops = body.ops,
    )
}
