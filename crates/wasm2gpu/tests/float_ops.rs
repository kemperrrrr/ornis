//! Scalar `f32` arithmetic and comparisons accepted by the wasm lifter.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;

fn wasm() -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
            (import "{module}" "{store_f32}" (func $store_f32 (param i32 i32 i32 f32)))
            (import "{module}" "{store_u32}" (func $store_u32 (param i32 i32 i32 i32)))
            (func (export "{vs}") (param f32 f32 f32 f32)
                (result f32 f32 f32 f32 f32 f32)
                local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)
            (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
                f32.const 0 f32.const 0 f32.const 0 f32.const 1)
            (func (export "{cs}") (param i32)
                i32.const 0 i32.const 0 i32.const 0
                f32.const 7 f32.const 2 f32.add call $store_f32
                i32.const 0 i32.const 0 i32.const 1
                f32.const 7 f32.const 2 f32.sub call $store_f32
                i32.const 0 i32.const 0 i32.const 2
                f32.const 7 f32.const 2 f32.mul call $store_f32
                i32.const 0 i32.const 0 i32.const 3
                f32.const 9 f32.const 2 f32.div call $store_f32
                i32.const 0 i32.const 0 i32.const 4
                f32.const -3.25 f32.abs call $store_f32
                i32.const 0 i32.const 0 i32.const 5
                f32.const 3.25 f32.neg call $store_f32
                i32.const 0 i32.const 0 i32.const 6
                f32.const inf call $store_f32
                i32.const 0 i32.const 0 i32.const 7
                f32.const nan call $store_f32
                i32.const 0 i32.const 1 i32.const 0
                f32.const 1 f32.const 1 f32.eq call $store_u32
                i32.const 0 i32.const 1 i32.const 1
                f32.const 1 f32.const 2 f32.ne call $store_u32
                i32.const 0 i32.const 1 i32.const 2
                f32.const 1 f32.const 2 f32.lt call $store_u32
                i32.const 0 i32.const 1 i32.const 3
                f32.const 1 f32.const 2 f32.le call $store_u32
                i32.const 0 i32.const 1 i32.const 4
                f32.const 2 f32.const 1 f32.gt call $store_u32
                i32.const 0 i32.const 1 i32.const 5
                f32.const 2 f32.const 1 f32.ge call $store_u32
                i32.const 0 i32.const 0 i32.const 8
                local.get 0 f32.convert_i32_u call $store_f32))"#,
        module = symbols::MODULE,
        store_f32 = symbols::STORAGE_WRITE_F32,
        store_u32 = symbols::STORAGE_WRITE_U32,
        vs = symbols::VS_MAIN,
        fs = symbols::FS_MAIN,
        cs = symbols::CS_MAIN,
    ))
    .expect("valid float-operator wasm")
}

#[test]
fn scalar_float_ops_and_non_finite_constants_lower_to_wgsl() {
    let translated = translate(&wasm()).expect("translate float operators");
    for operator in [
        " + ", " - ", " * ", " / ", " == ", " != ", " < ", " <= ", " > ", " >= ",
    ] {
        assert!(
            translated.wgsl.contains(operator),
            "missing {operator:?}: {}",
            translated.wgsl
        );
    }
    assert!(
        translated.wgsl.contains("abs("),
        "missing absolute value: {}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("sqrt("),
        "missing square root: {}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("f32("),
        "missing unsigned i32 conversion: {}",
        translated.wgsl
    );
    assert!(
        !translated.wgsl.contains("bitcast<f32>"),
        "non-finite values must be emitted as runtime operations: {}",
        translated.wgsl
    );
    assert!(!translated.compute_spirv.is_empty());
}
