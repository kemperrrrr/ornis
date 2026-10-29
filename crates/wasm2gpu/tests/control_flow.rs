//! Control-flow lowering: multi-level `br`, joins, and struct layout.
//!
//! A side-effect-free `if` with no results is dropped in `pure_if`: the
//! condition is consumed and neither arm is emitted.

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

fn wasm(custom: &str, imports: &str, fs: &str) -> Vec<u8> {
    wat::parse_str(format!(
        "(module {custom} {imports} {vs} (func (export \"{name}\") {fs}))",
        vs = vertex(),
        name = symbols::FS_MAIN,
    ))
    .expect("wat")
}

fn import(name: &str, sig: &str) -> String {
    format!(
        "(import \"{m}\" \"{name}\" (func ${name} {sig}))",
        m = symbols::MODULE,
    )
}

fn wgsl(custom: &str, imports: &str, fs: &str) -> String {
    translate(&wasm(custom, imports, fs))
        .expect("translate")
        .wgsl
}

fn err(custom: &str, imports: &str, fs: &str) -> TranslateError {
    translate(&wasm(custom, imports, fs)).expect_err("reject")
}

fn write_f32() -> String {
    import(symbols::STORAGE_WRITE_F32, "(param i32 i32 i32 f32)")
}

fn rgba() -> &'static str {
    "f32.const 0 f32.const 0 f32.const 0 f32.const 1"
}

fn write_call() -> String {
    format!(
        "i32.const 0 i32.const 1 i32.const 0 f32.const 1 call ${write}",
        write = symbols::STORAGE_WRITE_F32,
    )
}

fn contains(got: &str, snapshot: &str) {
    assert!(got.contains(snapshot.trim()), "{got}");
}

#[test]
fn br_from_if_arm_skips_later_write() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (block
           (if (local.get 2)
             (then
               (block (br 2))
               {write})
             (else)))
         {ret}",
        write = write_call(),
        ret = rgba(),
    );
    contains(
        &wgsl("", &write_f32(), &fs),
        include_str!("snapshots/if_arm_leave.wgsl"),
    );
}

#[test]
fn br_from_if_arm_continues_outer_loop() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (loop
           (if (local.get 2)
             (then
               (block (br 2))
               {write})
             (else))
           br 0)
         {ret}",
        write = write_call(),
        ret = rgba(),
    );
    contains(
        &wgsl("", &write_f32(), &fs),
        include_str!("snapshots/if_arm_again.wgsl"),
    );
}

#[test]
fn br_through_three_blocks_skips_the_middle_write() {
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         (block
           (block
             (block (br 2))
             {write}))
         {ret}",
        write = write_call(),
        ret = rgba(),
    );
    contains(
        &wgsl("", &write_f32(), &fs),
        include_str!("snapshots/br_three.wgsl"),
    );
}

#[test]
fn loop_br_if_to_outer_block_skips_the_write() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (block
           (loop
             local.get 2
             br_if 1
             {write}
             br 0))
         {ret}",
        write = write_call(),
        ret = rgba(),
    );
    contains(
        &wgsl("", &write_f32(), &fs),
        include_str!("snapshots/loop_exit.wgsl"),
    );
}

#[test]
fn loop_back_edge_joins_provenance() {
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         (local i32)
         i32.const 0
         local.set 2
         (loop
           i32.const 0
           local.get 2
           i32.const 0
           call ${read}
           drop
           i32.const 1
           local.set 2
           br 0)
         {ret}",
        read = symbols::STORAGE_READ,
        ret = rgba(),
    );
    let err = err(
        "",
        &import(
            symbols::STORAGE_READ,
            "(param i32 i32 i32) (result f32 f32 f32 f32)",
        ),
        &fs,
    );
    assert!(matches!(err, TranslateError::AmbiguousProvenance));
}

#[test]
fn branch_to_the_function_is_an_error() {
    let fs = "(param f32 f32) (result f32 f32 f32 f32)
         (block
           f32.const 0 f32.const 0 f32.const 0 f32.const 1
           br 1)";
    let err = err("", "", fs);
    assert!(matches!(err, TranslateError::FunctionBranch));
    assert!(
        err.to_string()
            .contains("branch to the function is not lowered")
    );
}

#[test]
fn pure_if_without_results_is_dropped() {
    let fs = "(param f32 f32 i32) (result f32 f32 f32 f32)
         (if (local.get 2) (then i32.const 1) (else i32.const 2))
         f32.const 0 f32.const 0 f32.const 0 f32.const 1";
    let got = wgsl("", "", fs);
    assert!(!got.contains("if ("), "{got}");
    assert!(!got.contains("select("), "{got}");
}

#[test]
fn unset_local_is_constant_zero() {
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         (local i32)
         local.get 2
         i32.const 1
         i32.const 0
         call ${read}
         f32.const 0 f32.const 0 f32.const 0",
        read = symbols::STORAGE_READ_F32,
    );
    let got = wgsl(
        "",
        &import(
            symbols::STORAGE_READ_F32,
            "(param i32 i32 i32) (result f32)",
        ),
        &fs,
    );
    assert!(got.contains("storage_0_1_"), "{got}");
    assert!(got.contains("array<f32>"), "{got}");
}

#[test]
fn two_vec3_and_integer_fields_lower() {
    let custom = format!(
        "(@custom \"{name}\" \"\\4f\\47\\54\\31\\01\\00\\04\\00\\05\\05\\02\\03\")",
        name = symbols::TYPES_SECTION,
    );
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         i32.const 0 i32.const 1 i32.const 0 i32.const 0 call ${read}
         drop drop drop drop drop drop drop drop
         {ret}",
        read = symbols::STORAGE_READ_STRUCT,
        ret = rgba(),
    );
    let got = wgsl(
        &custom,
        &import(
            symbols::STORAGE_READ_STRUCT,
            "(param i32 i32 i32 i32) (result f32 f32 f32 f32 f32 f32 i32 i32)",
        ),
        &fs,
    );
    contains(&got, include_str!("snapshots/struct_two_vec3.wgsl"));
    assert!(got.contains("array<S0_>"), "{got}");
}

#[test]
fn struct_id_out_of_range_is_rejected() {
    let custom = format!(
        "(@custom \"{name}\" \"\\4f\\47\\54\\31\\01\\00\\01\\00\\01\")",
        name = symbols::TYPES_SECTION,
    );
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         i32.const 0 i32.const 1 i32.const 0 i32.const 1 call ${read}
         drop
         {ret}",
        read = symbols::STORAGE_READ_STRUCT,
        ret = rgba(),
    );
    let err = err(
        &custom,
        &import(
            symbols::STORAGE_READ_STRUCT,
            "(param i32 i32 i32 i32) (result f32)",
        ),
        &fs,
    );
    match err {
        TranslateError::UnsupportedType(msg) => {
            assert!(msg.contains("struct type id is out of range"), "{msg}");
        }
        other => panic!("unexpected {other}"),
    }
}
