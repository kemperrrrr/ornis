//! Linear memory through storage provenance: `storage_base`, pointer
//! arithmetic, and scalar `i32` / `f32` loads and stores.
//!
//! The address must be a pointer the analysis built; a plain integer is
//! rejected, and a misaligned constant offset never reaches a load.

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

fn wasm(imports: &str, fs: &str) -> Vec<u8> {
    wat::parse_str(format!("(module {imports} {vs} {fs})", vs = vertex())).expect("wat")
}

fn import(name: &str, sig: &str) -> String {
    format!(
        "(import \"{m}\" \"{name}\" (func ${name} {sig}))",
        m = symbols::MODULE,
    )
}

fn fs_func(header: &str, body: &str) -> String {
    format!(
        r#"(func (export "{fs}") {header} {body})"#,
        fs = symbols::FS_MAIN,
    )
}

/// Address of binding `(0, 1)`: group and binding, then `storage_base`.
fn base() -> String {
    format!(
        "i32.const 0 i32.const 1 call ${b}",
        b = symbols::STORAGE_BASE,
    )
}

fn rgba() -> &'static str {
    "f32.const 0 f32.const 0 f32.const 0 f32.const 1"
}

fn wgsl(imports: &str, fs: &str) -> String {
    translate(&wasm(imports, fs)).expect("translate").wgsl
}

fn err(imports: &str, fs: &str) -> TranslateError {
    translate(&wasm(imports, fs)).expect_err("reject")
}

fn storage_base_import() -> String {
    import(symbols::STORAGE_BASE, "(param i32 i32) (result i32)")
}

#[test]
fn constant_offset_folds_to_one_element() {
    let imports = storage_base_import();
    let fs = fs_func(
        "(param f32 f32 i32) (result f32 f32 f32 f32)",
        &format!(
            "{base} i32.const 4 i32.add i32.load f32.const 0 f32.const 0 f32.const 1",
            base = base(),
        ),
    );
    let got = wgsl(&imports, &fs);
    assert!(got.contains("storage_0_1_[1"), "{got}");
    assert!(!got.contains(">>"), "{got}");
}

#[test]
fn dynamic_offset_becomes_a_shifted_index() {
    let imports = storage_base_import();
    let fs = fs_func(
        "(param f32 f32 i32) (result f32 f32 f32 f32)",
        &format!(
            "{base} local.get 2 i32.const 4 i32.mul i32.add i32.load \
             {base} i32.add i32.load drop {rgba}",
            base = base(),
            rgba = rgba(),
        ),
    );
    let got = wgsl(&imports, &fs);
    assert!(got.contains("storage_0_1_"), "{got}");
    assert!(got.contains(">>"), "{got}");
}

#[test]
fn shift_offset_keeps_provenance() {
    let imports = storage_base_import();
    let fs = fs_func(
        "(param f32 f32 i32) (result f32 f32 f32 f32)",
        &format!(
            "{base} local.get 2 i32.const 2 i32.shl i32.add i32.load \
             {base} i32.add i32.load drop {rgba}",
            base = base(),
            rgba = rgba(),
        ),
    );
    let got = wgsl(&imports, &fs);
    assert!(got.contains("storage_0_1_"), "{got}");
    assert!(got.contains(">>"), "{got}");
}

#[test]
fn store_then_load_round_trips_through_the_pointer() {
    let imports = storage_base_import();
    let fs = fs_func(
        "(param f32 f32 i32) (result f32 f32 f32 f32)",
        &format!(
            "{base} i32.const 0 i32.add i32.const 123 i32.store \
             {base} i32.const 0 i32.add i32.load drop {rgba}",
            base = base(),
            rgba = rgba(),
        ),
    );
    let got = wgsl(&imports, &fs);
    assert!(got.contains("storage_0_1_"), "{got}");
}

#[test]
fn float_store_and_load_use_one_binding() {
    let imports = storage_base_import();
    let fs = fs_func(
        "(param f32 f32 i32) (result f32 f32 f32 f32)",
        &format!(
            "{base} i32.const 4 i32.add f32.const 2.5 f32.store \
             {base} i32.const 4 i32.add f32.load \
             f32.const 0 f32.const 0 f32.const 1",
            base = base(),
        ),
    );
    let got = wgsl(&imports, &fs);
    assert!(got.contains("storage_0_1_"), "{got}");
}

#[test]
fn pointer_survives_a_local_round_trip() {
    let imports = storage_base_import();
    let fs = fs_func(
        "(param f32 f32 i32) (result f32 f32 f32 f32) (local i32)",
        &format!(
            "{base} local.set 3 \
             local.get 3 i32.const 4 i32.add i32.const 7 i32.store {rgba}",
            base = base(),
            rgba = rgba(),
        ),
    );
    let got = wgsl(&imports, &fs);
    assert!(got.contains("storage_0_1_"), "{got}");
}

#[test]
fn import_and_pointer_share_one_binding() {
    let imports = format!(
        "{} {}",
        storage_base_import(),
        import(
            symbols::STORAGE_READ_U32,
            "(param i32 i32 i32) (result i32)"
        ),
    );
    let fs = fs_func(
        "(param f32 f32 i32) (result f32 f32 f32 f32)",
        &format!(
            "i32.const 0 i32.const 1 local.get 2 call ${read} drop \
             {base} i32.const 4 i32.add i32.load drop {rgba}",
            read = symbols::STORAGE_READ_U32,
            base = base(),
            rgba = rgba(),
        ),
    );
    let got = wgsl(&imports, &fs);
    assert!(got.contains("storage_0_1_"), "{got}");
}

#[test]
fn plain_integer_address_is_rejected() {
    let imports = storage_base_import();
    let fs = fs_func(
        "(param f32 f32 i32) (result f32 f32 f32 f32)",
        &format!("local.get 2 i32.load drop {rgba}", rgba = rgba()),
    );
    let got = err(&imports, &fs);
    assert!(matches!(got, TranslateError::LinearMemory), "{got:?}");
}

#[test]
fn misaligned_constant_offset_is_rejected() {
    let imports = storage_base_import();
    let fs = fs_func(
        "(param f32 f32 i32) (result f32 f32 f32 f32)",
        &format!(
            "{base} i32.const 2 i32.add i32.load drop {rgba}",
            base = base(),
            rgba = rgba(),
        ),
    );
    match err(&imports, &fs) {
        TranslateError::UnsupportedType(msg) => {
            assert!(msg.contains("4-byte aligned"), "{msg}");
        }
        other => panic!("expected an alignment rejection, got {other:?}"),
    }
}

#[test]
fn pointer_result_across_a_call_is_rejected() {
    let imports = format!(
        "{}\n(func $addr (result i32) {base})",
        storage_base_import(),
        base = base(),
    );
    let fs = fs_func(
        "(param f32 f32 i32) (result f32 f32 f32 f32)",
        &format!("call $addr i32.load drop {rgba}", rgba = rgba()),
    );
    let got = err(&imports, &fs);
    assert!(matches!(got, TranslateError::Unsupported), "{got:?}");
}

#[test]
fn mixed_element_types_on_one_binding_are_rejected() {
    // The fs comes first so this window cannot mirror the other tests here.
    let fs = fs_func(
        "(param f32 f32 i32) (result f32 f32 f32 f32)",
        &format!(
            "{base} i32.const 0 i32.add i32.load drop \
             {base} i32.const 4 i32.add f32.load drop {rgba}",
            base = base(),
            rgba = rgba(),
        ),
    );
    let imports = storage_base_import();
    match err(&imports, &fs) {
        TranslateError::UnsupportedType(msg) => {
            assert!(msg.contains("two storage types"), "{msg}");
        }
        other => panic!("expected a binding type rejection, got {other:?}"),
    }
}
