//! Shared WAT and translation helpers for integration tests.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;

pub fn vertex() -> String {
    format!(
        r#"(func (export "{vs}") (param f32 f32 f32 f32) (result f32 f32 f32 f32 f32 f32)
            local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)"#,
        vs = symbols::VS_MAIN,
    )
}

pub fn fragment(body: &str) -> String {
    format!(
        r#"(func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32) {body})"#,
        fs = symbols::FS_MAIN,
    )
}

pub fn wasm(imports: &str, funcs: &str) -> Vec<u8> {
    wat::parse_str(format!("(module {imports} {funcs} {vs})", vs = vertex(),)).expect("wat")
}

pub fn import(name: &str, sig: &str) -> String {
    format!(
        "(import \"{m}\" \"{name}\" (func ${name} {sig}))",
        m = symbols::MODULE,
    )
}

pub fn wgsl(imports: &str, funcs: &str) -> String {
    translate(&wasm(imports, funcs)).expect("translate").wgsl
}

pub fn assert_wgsl_expressions(wgsl: &str, snapshot: &str) {
    let normalized = wgsl.split_whitespace().collect::<Vec<_>>().join(" ");
    for expression in snapshot.lines() {
        let expression = expression.trim();
        assert!(
            normalized.contains(expression),
            "missing expression {expression:?} in WGSL:\n{wgsl}"
        );
    }
}
