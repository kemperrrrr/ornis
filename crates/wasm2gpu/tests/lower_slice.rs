//! Storage scalars, vectors, structs, and structured control flow.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::TranslateError;
use ornis_wasm2gpu::translate;

fn vs() -> String {
    format!(
        r#"(func (export "{vs}") (param f32 f32 f32 f32) (result f32 f32 f32 f32 f32 f32)
            local.get 0
            local.get 1
            f32.const 0
            f32.const 1
            local.get 2
            local.get 3)"#,
        vs = symbols::VS_MAIN,
    )
}

fn module(custom: &str, imports: &str, fs: &str) -> Vec<u8> {
    wat::parse_str(format!(
        "(module {custom} {imports} {vs} (func (export \"{name}\") {fs}))",
        vs = vs(),
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
    translate(&module(custom, imports, fs))
        .expect("translate")
        .wgsl
}

fn f32_pad() -> &'static str {
    "(param f32 f32) (result f32 f32 f32 f32) f32.const 0 f32.const 0 f32.const 0 f32.const 1"
}

#[test]
fn storage_f32_is_an_array_of_scalars() {
    let got = wgsl(
        "",
        &import(
            symbols::STORAGE_READ_F32,
            "(param i32 i32 i32) (result f32)",
        ),
        &format!(
            "(param f32 f32) (result f32 f32 f32 f32)
             i32.const 0 i32.const 1 i32.const 0 call ${read}
             f32.const 0 f32.const 0 f32.const 0",
            read = symbols::STORAGE_READ_F32,
        ),
    );
    assert!(
        got.contains(include_str!("snapshots/storage_f32.wgsl").trim()),
        "{got}"
    );
}

#[test]
fn storage_u32_and_i32_keep_their_scalar_types() {
    let got = wgsl(
        "",
        &import(
            symbols::STORAGE_READ_U32,
            "(param i32 i32 i32) (result i32)",
        ),
        &format!(
            "(param f32 f32) (result f32 f32 f32 f32)
             i32.const 0 i32.const 1 i32.const 0 call ${read} drop
             f32.const 0 f32.const 0 f32.const 0 f32.const 1",
            read = symbols::STORAGE_READ_U32,
        ),
    );
    assert!(got.contains("array<u32>"), "{got}");
    let got = wgsl(
        "",
        &import(
            symbols::STORAGE_READ_I32,
            "(param i32 i32 i32) (result i32)",
        ),
        &format!(
            "(param f32 f32) (result f32 f32 f32 f32)
             i32.const 0 i32.const 1 i32.const 0 call ${read} drop
             f32.const 0 f32.const 0 f32.const 0 f32.const 1",
            read = symbols::STORAGE_READ_I32,
        ),
    );
    assert!(got.contains("array<i32>"), "{got}");
}

#[test]
fn storage_vec2_and_vec3_flatten_to_scalar_lanes() {
    let got = wgsl(
        "",
        &import(
            symbols::STORAGE_READ_VEC2,
            "(param i32 i32 i32) (result f32 f32)",
        ),
        &format!(
            "(param f32 f32) (result f32 f32 f32 f32)
             i32.const 0 i32.const 1 i32.const 0 call ${read}
             f32.const 0 f32.const 1",
            read = symbols::STORAGE_READ_VEC2,
        ),
    );
    assert!(got.contains("array<vec2<f32>>"), "{got}");
    let got = wgsl(
        "",
        &import(
            symbols::STORAGE_READ_VEC3,
            "(param i32 i32 i32) (result f32 f32 f32)",
        ),
        &format!(
            "(param f32 f32) (result f32 f32 f32 f32)
             i32.const 0 i32.const 1 i32.const 0 call ${read}
             f32.const 1",
            read = symbols::STORAGE_READ_VEC3,
        ),
    );
    assert!(got.contains("array<vec3<f32>>"), "{got}");
}

#[test]
fn struct_vec3_then_f32_matches_host_alignment() {
    let custom = format!(
        "(@custom \"{name}\" \"\\4f\\47\\54\\31\\01\\00\\02\\00\\05\\01\")",
        name = symbols::TYPES_SECTION,
    );
    let got = wgsl(
        &custom,
        &import(
            symbols::STORAGE_READ_STRUCT,
            "(param i32 i32 i32 i32) (result f32 f32 f32 f32)",
        ),
        &format!(
            "(param f32 f32) (result f32 f32 f32 f32)
             i32.const 0 i32.const 1 i32.const 0 i32.const 0 call ${read}",
            read = symbols::STORAGE_READ_STRUCT,
        ),
    );
    assert!(
        got.contains(include_str!("snapshots/struct_tint.wgsl").trim()),
        "{got}"
    );
    assert!(got.contains("array<S0_>"), "{got}");
}

#[test]
fn unknown_struct_field_is_rejected() {
    let custom = format!(
        "(@custom \"{name}\" \"\\4f\\47\\54\\31\\01\\00\\01\\00\\00\")",
        name = symbols::TYPES_SECTION,
    );
    let err = translate(&module(
        &custom,
        &import(
            symbols::STORAGE_READ_F32,
            "(param i32 i32 i32) (result f32)",
        ),
        f32_pad(),
    ))
    .expect_err("tag");
    assert!(matches!(err, TranslateError::UnsupportedType(_)));
}

#[test]
fn one_binding_cannot_be_two_storage_types() {
    let imports = format!(
        "{a} {b}",
        a = import(
            symbols::STORAGE_READ,
            "(param i32 i32 i32) (result f32 f32 f32 f32)"
        ),
        b = import(
            symbols::STORAGE_READ_F32,
            "(param i32 i32 i32) (result f32)"
        ),
    );
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         i32.const 0 i32.const 1 i32.const 0 call ${vec4}
         drop
         i32.const 0 i32.const 1 i32.const 0 call ${scalar}
         f32.const 0 f32.const 0 f32.const 0",
        vec4 = symbols::STORAGE_READ,
        scalar = symbols::STORAGE_READ_F32,
    );
    let err = translate(&module("", &imports, &fs)).expect_err("clash");
    assert!(matches!(err, TranslateError::UnsupportedType(_)));
}

#[test]
fn mismatched_provenance_at_a_join_is_rejected() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         i32.const 0
         (if (result i32) (local.get 2) (then i32.const 0) (else i32.const 1))
         i32.const 0
         call ${read}",
        read = symbols::STORAGE_READ,
    );
    let err = translate(&module(
        "",
        &import(
            symbols::STORAGE_READ,
            "(param i32 i32 i32) (result f32 f32 f32 f32)",
        ),
        &fs,
    ))
    .expect_err("join");
    assert!(matches!(err, TranslateError::AmbiguousProvenance));
    assert!(err.to_string().contains("ambiguous provenance at a join"));
}

#[test]
fn pure_if_index_lowers_to_select() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         i32.const 0
         i32.const 1
         (if (result i32) (local.get 2) (then i32.const 0) (else i32.const 1))
         call ${read}",
        read = symbols::STORAGE_READ,
    );
    let got = wgsl(
        "",
        &import(
            symbols::STORAGE_READ,
            "(param i32 i32 i32) (result f32 f32 f32 f32)",
        ),
        &fs,
    );
    assert!(
        got.contains(include_str!("snapshots/select_index.wgsl").trim()),
        "{got}"
    );
}

#[test]
fn side_effect_if_lowers_to_if() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (if (local.get 2)
           (then
             i32.const 0 i32.const 1 i32.const 0
             f32.const 1
             call ${write})
           (else))
         f32.const 0 f32.const 0 f32.const 0 f32.const 1",
        write = symbols::STORAGE_WRITE_F32,
    );
    let got = wgsl(
        "",
        &import(symbols::STORAGE_WRITE_F32, "(param i32 i32 i32 f32)"),
        &fs,
    );
    assert!(got.contains("if ("), "{got}");
    assert!(got.contains("array<f32>"), "{got}");
}

#[test]
fn loop_with_a_dynamic_index_lowers_to_loop() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (local i32)
         i32.const 0
         local.set 3
         (loop
           i32.const 0
           i32.const 1
           local.get 3
           call ${read}
           drop
           local.get 3
           i32.const 1
           i32.add
           local.set 3
           local.get 3
           local.get 2
           i32.lt_u
           br_if 0)
         f32.const 0 f32.const 0 f32.const 0 f32.const 1",
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
    assert!(got.contains("loop"), "{got}");
    assert!(got.contains("storage_0_1_"), "{got}");
    assert!(got.contains("array<f32>"), "{got}");
}

#[test]
fn block_br_if_breaks_the_lowered_loop() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (block
           local.get 2
           br_if 0
           i32.const 0 i32.const 1 i32.const 0 call ${read}
           drop)
         f32.const 0 f32.const 0 f32.const 0 f32.const 1",
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
    assert!(got.contains("loop"), "{got}");
    assert!(got.contains("break"), "{got}");
}

#[test]
fn br_to_an_outer_block_sets_a_flag() {
    let fs = "(param f32 f32 i32) (result f32 f32 f32 f32)
         (block
           (loop
             local.get 2
             br_if 1
             br 0))
         f32.const 0 f32.const 0 f32.const 0 f32.const 1";
    let got = wgsl("", "", fs);
    assert!(got.contains("loop"), "{got}");
    assert!(got.contains("continue"), "{got}");
    assert!(got.contains("break"), "{got}");
}
