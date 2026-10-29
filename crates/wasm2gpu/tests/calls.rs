//! Direct calls, `br_table` is covered next door, and the compute entry.
//!
//! Each `translate` runs naga's validator, reprints WGSL, and round-trips
//! SPIR-V. A constant group or binding is specialized into the callee.
//! A returned immediate can be the next import's binding.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::TranslateError;
use ornis_wasm2gpu::translate;

fn vertex() -> String {
    format!(
        r#"(func (export "{vs}") (param f32 f32 f32 f32) (result f32 f32 f32 f32 f32 f32)
            local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)"#,
        vs = symbols::VS_MAIN,
    )
}

fn fragment(body: &str) -> String {
    format!(
        r#"(func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32) {body})"#,
        fs = symbols::FS_MAIN,
    )
}

fn wasm(imports: &str, funcs: &str) -> Vec<u8> {
    wat::parse_str(format!("(module {imports} {funcs} {vs})", vs = vertex(),)).expect("wat")
}

fn import(name: &str, sig: &str) -> String {
    format!(
        "(import \"{m}\" \"{name}\" (func ${name} {sig}))",
        m = symbols::MODULE,
    )
}

fn wgsl(imports: &str, funcs: &str) -> String {
    translate(&wasm(imports, funcs)).expect("translate").wgsl
}

fn err(imports: &str, funcs: &str) -> TranslateError {
    translate(&wasm(imports, funcs)).expect_err("reject")
}

fn read_f32() -> String {
    import(
        symbols::STORAGE_READ_F32,
        "(param i32 i32 i32) (result f32)",
    )
}

fn rgba_tail() -> &'static str {
    "f32.const 0 f32.const 0 f32.const 1"
}

#[test]
fn call_specializes_constant_provenance() {
    let load = format!(
        r#"(func $load (param i32 i32 i32) (result f32)
            local.get 0 local.get 1 local.get 2 call ${read})"#,
        read = symbols::STORAGE_READ_F32,
    );
    let fs = fragment(&format!(
        "i32.const 0 i32.const 1 i32.const 2 call $load {tail}",
        tail = rgba_tail(),
    ));
    let got = wgsl(&read_f32(), &format!("{load} {fs}"));
    assert!(got.contains("fn fn0_"), "{got}");
    assert!(got.contains("storage_0_1_"), "{got}");
    assert!(got.contains("2u"), "{got}");
}

#[test]
fn returned_immediate_is_a_binding() {
    let slot = "(func $slot (result i32) i32.const 1)";
    let fs = fragment(&format!(
        "i32.const 0 call $slot i32.const 0 call ${read} {tail}",
        read = symbols::STORAGE_READ_F32,
        tail = rgba_tail(),
    ));
    let got = wgsl(&read_f32(), &format!("{slot} {fs}"));
    assert!(got.contains("storage_0_1_"), "{got}");
}

#[test]
fn offset_passes_through_an_argument_and_a_result() {
    let plus = r#"(func $plus (param i32) (result i32)
        local.get 0 i32.const 1 i32.add)"#;
    let fs = format!(
        r#"(func (export "{fs}") (param f32 f32 i32) (result f32 f32 f32 f32)
            i32.const 0 i32.const 1 local.get 2 call $plus
            call ${read}
            {tail})"#,
        fs = symbols::FS_MAIN,
        read = symbols::STORAGE_READ_F32,
        tail = rgba_tail(),
    );
    let got = wgsl(&read_f32(), &format!("{plus} {fs}"));
    assert!(got.contains("fn fn0_"), "{got}");
    assert!(got.contains("storage_0_1_"), "{got}");
}

#[test]
fn recursion_is_rejected() {
    let rec = "(func $rec (param i32) (result i32) local.get 0 call $rec)";
    let fs = fragment("i32.const 0 call $rec drop f32.const 0 f32.const 0 f32.const 0 f32.const 1");
    assert!(matches!(
        err("", &format!("{rec} {fs}")),
        TranslateError::Recursion
    ));
}

#[test]
fn mutual_recursion_is_rejected() {
    let funcs = "(func $a (result i32) call $b) (func $b (result i32) call $a)";
    let fs = fragment("call $a drop f32.const 0 f32.const 0 f32.const 0 f32.const 1");
    assert!(matches!(
        err("", &format!("{funcs} {fs}")),
        TranslateError::Recursion
    ));
}

#[test]
fn indirect_call_is_rejected() {
    let table = "(table 1 funcref)";
    let fs = fragment("f32.const 0 f32.const 0 f32.const 0 f32.const 1");
    assert!(matches!(err(table, &fs), TranslateError::IndirectCall));
}

#[test]
fn compute_entry_has_workgroup_and_invocation_id() {
    let body = format!(
        r#"(func (export "{cs}") (param i32)
            i32.const 0 i32.const 1 local.get 0 f32.const 1 call ${write})
           (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
            f32.const 0 f32.const 0 f32.const 0 f32.const 1)"#,
        cs = symbols::CS_MAIN,
        fs = symbols::FS_MAIN,
        write = symbols::STORAGE_WRITE_F32,
    );
    let imports = import(symbols::STORAGE_WRITE_F32, "(param i32 i32 i32 f32)");
    let translated = translate(&wasm(&imports, &body)).expect("compute");
    let size = symbols::CS_WORKGROUP;
    assert!(
        translated.wgsl.contains(&format!(
            "@workgroup_size({}, {}, {})",
            size[0], size[1], size[2]
        )),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("@builtin(global_invocation_id)"),
        "{}",
        translated.wgsl
    );
    assert!(translated.wgsl.contains("@compute"), "{}", translated.wgsl);
    assert!(
        translated.wgsl.contains("storage_0_1_"),
        "{}",
        translated.wgsl
    );
    assert!(!translated.compute_spirv.is_empty());
    naga::front::wgsl::parse_str(&translated.wgsl).expect("wgsl");
}
