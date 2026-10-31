//! Lift a compute export (`compute`, `cull`, or the fog `vs_main`) into
//! a toy module with `cs_main`.
//!
//! `rustc` emits a memory section for the cdylib. The translator rejects
//! that, so this lifter keeps only the functions the entry reaches: arm
//! calls, guards, and loops. Vertex and fragment stubs exist because the
//! translator still requires those entries.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;
use wasmparser::Parser;
use wasmparser::Payload;
use wasmparser::TypeRef;
use wasmparser::ValType;

mod ops;

/// Lift `export` into a standalone binary module (no memory section,
/// no globals) that the translator accepts. The integration test's
/// `compute` case is its consumer.
// qual:test_helper
// qual:allow(complexity, error_handling) reason: "Invalid fixture WAT fails the integration test."
pub fn lift(wasm: &[u8], export: &str) -> Vec<u8> {
    let wat = lift_wat(wasm, export);
    wat::parse_str(&wat).unwrap_or_else(|err| panic!("compute wat: {err}\n{wat}"))
}

/// Lift `export` from the guest and translate it, pinning what every
/// lowered kernel must keep: the compute entry, the id builtin, the
/// named storage binding, and a non-empty SPIR-V twin. Shared by the
/// `cull` A/B and the fog vertex A/B so the sequence exists once.
// qual:allow(complexity, error_handling) reason: "Translation failure invalidates the shader test."
pub fn lowered_kernel(export: &str, storage_binding: &str) -> String {
    let wasm = crate::compile_wasm();
    let wat = lift_wat(&wasm, export);
    let module = wat::parse_str(&wat).unwrap_or_else(|err| panic!("{export} wat: {err}\n{wat}"));
    let lowered = match translate(&module) {
        Ok(lowered) => lowered,
        Err(err) => panic!("{export} translate {err:?}\nWAT:\n{wat}"),
    };
    let wgsl = &lowered.wgsl;
    assert!(wgsl.contains("@compute"), "{wgsl}");
    assert!(wgsl.contains("@builtin(global_invocation_id)"), "{wgsl}");
    assert!(wgsl.contains(storage_binding), "{wgsl}");
    assert!(!lowered.compute_spirv.is_empty());
    wgsl.clone()
}

/// The toy-module text `lift` parses: exposed so a translate failure can
/// print exactly what the translator saw.
pub fn lift_wat(wasm: &[u8], export: &str) -> String {
    let module = read(wasm, export);
    wrap(&module)
}

struct Func {
    params: Result<String, String>,
    /// Declared results. A returning helper without them would validate a
    /// stack the callee never agreed to and the translator rejects it.
    results: Result<String, String>,
    /// Locals and operators. `Err` is an op this lifter does not keep.
    /// Only a function `compute` reaches turns that into a panic.
    body: Result<(String, String), String>,
}

struct Read {
    imports: Vec<String>,
    bodies: Vec<Func>,
    compute: u32,
}

// qual:allow(complexity, error_handling) reason: "Malformed fixture wasm fails the test."
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

// qual:allow(complexity, error_handling) reason: "Invalid fixture type sections fail the test."
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
    Ok((ops::val_list(params)?, ops::val_list(results)?))
}

// qual:allow(complexity, error_handling) reason: "Unexpected imports invalidate the test fixture."
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
    let (params, results) = match ty {
        Ok((params, results)) => (Ok(params.clone()), Ok(results.clone())),
        Err(err) => (Err(err.clone()), Err(err.clone())),
    };
    Func {
        params,
        results,
        body: lift_body(imports, body),
    }
}

fn lift_body(
    imports: &[String],
    body: &wasmparser::FunctionBody<'_>,
) -> Result<(String, String), String> {
    Ok((locals_of(body)?, ops::ops_of(imports, body)?))
}

// qual:allow(complexity, error_handling) reason: "Unsupported fixture locals fail this test."
fn locals_of(body: &wasmparser::FunctionBody<'_>) -> Result<String, String> {
    let mut text = String::new();
    let reader = body.get_locals_reader().expect("locals");
    for local in reader {
        let (count, ty) = local.expect("local");
        let name = ops::val_name(ty)?;
        let mut index = 0u32;
        while index < count {
            text.push_str(&format!("(local {name})"));
            index += 1;
        }
    }
    Ok(text)
}

// qual:allow(complexity, error_handling) reason: "Unsupported reachable ops fail this test."
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
        let results = result_list(func.results.as_deref().unwrap_or(""));
        funcs.push_str(&format!(
            "(func $f{wasm_index} {export} {params} {results} {locals} {ops})\n",
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

fn result_list(results: &str) -> String {
    let results = results.trim();
    if results.is_empty() {
        String::new()
    } else {
        format!("(result {results})")
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

// qual:allow(complexity, error_handling) reason: "Unknown ABI imports invalidate the fixture."
fn import_sig(name: &str) -> &'static str {
    match name {
        symbols::STORAGE_BASE => "(param i32 i32) (result i32)",
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
