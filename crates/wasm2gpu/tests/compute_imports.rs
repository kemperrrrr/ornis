//! Translator snapshots for vector math and compute synchronization imports.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;

mod common;

use common::{assert_vector_lanes_in_order, assert_wgsl_expressions, fragment, import, wasm, wgsl};

#[test]
fn atomics_workgroup_memory_and_barriers_lower_to_wgsl() {
    let imports = format!(
        "{} {} {} {} {} {}",
        import(
            symbols::STORAGE_ATOMIC_ADD_U32,
            "(param i32 i32 i32 i32) (result i32)"
        ),
        import(symbols::WORKGROUP_LOAD_U32, "(param i32) (result i32)"),
        import(symbols::WORKGROUP_STORE_U32, "(param i32 i32)"),
        import(
            symbols::WORKGROUP_ATOMIC_ADD_U32,
            "(param i32 i32) (result i32)"
        ),
        import(symbols::WORKGROUP_BARRIER, ""),
        import(symbols::STORAGE_BARRIER, ""),
    );
    let compute = format!(
        r#"(func (export "{cs}") (param i32)
            i32.const 0 i32.const 2 i32.const 1 i32.const 4
            call ${storage_atomic} drop
            i32.const 3 i32.const 9 call ${workgroup_store}
            call ${workgroup_barrier}
            i32.const 3 i32.const 3 call ${workgroup_load}
            call ${workgroup_atomic} drop
            call ${storage_barrier})
           (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
            f32.const 0 f32.const 0 f32.const 0 f32.const 1)"#,
        cs = symbols::CS_MAIN,
        fs = symbols::FS_MAIN,
        storage_atomic = symbols::STORAGE_ATOMIC_ADD_U32,
        workgroup_store = symbols::WORKGROUP_STORE_U32,
        workgroup_barrier = symbols::WORKGROUP_BARRIER,
        workgroup_load = symbols::WORKGROUP_LOAD_U32,
        workgroup_atomic = symbols::WORKGROUP_ATOMIC_ADD_U32,
        storage_barrier = symbols::STORAGE_BARRIER,
    );
    let translated = translate(&wasm(&imports, &compute)).expect("atomics and barriers");
    assert_wgsl_expressions(
        &translated.wgsl,
        include_str!("snapshots/atomics_workgroup.wgsl"),
    );
    assert!(!translated.compute_spirv.is_empty());
}

#[test]
fn scalar_min_max_clamp_lower_to_wgsl_builtins() {
    let cases = [
        "f32.const 2 f32.const 1 f32.min f32.const 0 f32.const 0 f32.const 1",
        "f32.const 2 f32.const 1 f32.max f32.const 0 f32.const 0 f32.const 1",
        "f32.const 2 f32.const 0 f32.max f32.const 1 f32.min f32.const 0 f32.const 0 f32.const 1",
    ];
    let snapshots = include_str!("snapshots/math_scalar.wgsl").lines();
    for (body, snapshot) in cases.into_iter().zip(snapshots) {
        let got = wgsl("", &fragment(body));
        assert_wgsl_expressions(&got, snapshot);
    }
}

#[test]
fn vector_math_imports_lower_to_wgsl_builtins() {
    let cases = [
        (
            symbols::MATH_DOT3,
            "(param f32 f32 f32 f32 f32 f32) (result f32)",
            "f32.const 1 f32.const 2 f32.const 3 f32.const 4 f32.const 5 f32.const 6 call $op f32.const 0 f32.const 0 f32.const 1",
        ),
        (
            symbols::MATH_CROSS3,
            "(param f32 f32 f32 f32 f32 f32) (result f32 f32 f32)",
            "f32.const 1 f32.const 0 f32.const 0 f32.const 0 f32.const 1 f32.const 0 call $op f32.const 1",
        ),
        (
            symbols::MATH_LENGTH3,
            "(param f32 f32 f32) (result f32)",
            "f32.const 0 f32.const 3 f32.const 4 call $op f32.const 0 f32.const 0 f32.const 1",
        ),
        (
            symbols::MATH_NORMALIZE3,
            "(param f32 f32 f32) (result f32 f32 f32)",
            "f32.const 0 f32.const 3 f32.const 4 call $op f32.const 1",
        ),
    ];
    let snapshots = include_str!("snapshots/math_vec3.wgsl").lines();
    for ((name, signature, body), snapshot) in cases.into_iter().zip(snapshots) {
        let imports = import(name, signature);
        let fs = fragment(&body.replace("$op", &format!("${name}")));
        let got = wgsl(&imports, &fs);
        assert_wgsl_expressions(&got, snapshot);
        if name == symbols::MATH_CROSS3 || name == symbols::MATH_NORMALIZE3 {
            assert_vector_lanes_in_order(&got, snapshot);
        }
    }
}

#[test]
fn texture_sample_explicit_lod_lowers_to_wgsl() {
    let name = symbols::TEXTURE_SAMPLE_2D_LOD;
    let imports = import(
        name,
        "(param i32 i32 i32 i32 f32 f32 f32) (result f32 f32 f32 f32)",
    );
    let body = format!(
        "i32.const 1 i32.const 2 i32.const 3 i32.const 4 \
         f32.const 0.25 f32.const 0.75 f32.const 2 call ${name}",
    );
    let got = wgsl(&imports, &fragment(&body));
    assert_wgsl_expressions(&got, include_str!("snapshots/texture_sample_lod.wgsl"));
}

#[test]
fn new_compute_imports_reject_wrong_signatures() {
    let cases = [
        (
            symbols::MATH_DOT3,
            "(param i32 f32 f32 f32 f32 f32) (result f32)",
            "i32.const 0 f32.const 0 f32.const 0 f32.const 0 f32.const 0 f32.const 0",
        ),
        (
            symbols::STORAGE_ATOMIC_ADD_U32,
            "(param i32 i32 i32 i32) (result f32)",
            "i32.const 0 i32.const 1 i32.const 0 i32.const 1",
        ),
        (
            symbols::WORKGROUP_LOAD_U32,
            "(param i32) (result f32)",
            "i32.const 0",
        ),
        (
            symbols::WORKGROUP_STORE_U32,
            "(param i32 i32) (result i32)",
            "i32.const 0 i32.const 1",
        ),
        (
            symbols::WORKGROUP_ATOMIC_ADD_U32,
            "(param i32 f32) (result i32)",
            "i32.const 0 f32.const 1",
        ),
        (symbols::WORKGROUP_BARRIER, "(param i32)", "i32.const 0"),
        (symbols::STORAGE_BARRIER, "(result i32)", ""),
    ];
    for (name, signature, arguments) in cases {
        let import = import(name, signature);
        let discard = if name == symbols::WORKGROUP_BARRIER {
            ""
        } else {
            "drop"
        };
        let fs = fragment(&format!(
            "{arguments} call ${name} {discard} f32.const 0 f32.const 0 f32.const 0 f32.const 1"
        ));
        let error = translate(&wasm(&import, &fs)).expect_err("reject invalid signature");
        assert!(
            matches!(error, TranslateError::UnsupportedType(_)),
            "{name} accepted {signature}: {error:?}"
        );
    }
}

#[test]
fn storage_binding_cannot_be_both_atomic_and_ordinary() {
    let atomic = import(
        symbols::STORAGE_ATOMIC_ADD_U32,
        "(param i32 i32 i32 i32) (result i32)",
    );
    let read = import(
        symbols::STORAGE_READ_F32,
        "(param i32 i32 i32) (result f32)",
    );
    let tail = "f32.const 0 f32.const 0 f32.const 0 f32.const 1";
    for operations in [
        format!(
            "i32.const 0 i32.const 1 i32.const 0 i32.const 1 call ${} drop \
             i32.const 0 i32.const 1 i32.const 0 call ${} drop {tail}",
            symbols::STORAGE_ATOMIC_ADD_U32,
            symbols::STORAGE_READ_F32,
        ),
        format!(
            "i32.const 0 i32.const 1 i32.const 0 call ${} drop \
             i32.const 0 i32.const 1 i32.const 0 i32.const 1 call ${} drop {tail}",
            symbols::STORAGE_READ_F32,
            symbols::STORAGE_ATOMIC_ADD_U32,
        ),
    ] {
        let error = translate(&wasm(&format!("{atomic} {read}"), &fragment(&operations)))
            .expect_err("reject mixed atomic and ordinary storage");
        assert!(
            matches!(error, TranslateError::UnsupportedType(message)
                if message.contains("atomic and ordinary storage")),
            "{error:?}"
        );
    }
}

#[test]
fn new_texture_imports_reject_wrong_lane_signatures() {
    let cases = [
        (
            symbols::TEXTURE_LOAD_2D,
            "(param f32 i32 i32 i32) (result f32 f32 f32 f32)",
            "f32.const 0 i32.const 1 i32.const 0 i32.const 0",
        ),
        (
            symbols::TEXTURE_SAMPLE_2D_LOD,
            "(param i32 i32 i32 i32 f32 f32 i32) (result f32 f32 f32 f32)",
            "i32.const 0 i32.const 1 i32.const 0 i32.const 1 \
             f32.const 0.5 f32.const 0.5 i32.const 0",
        ),
    ];
    for (name, signature, arguments) in cases {
        let import = import(name, signature);
        let fs = fragment(&format!(
            "{arguments} call ${name} drop drop drop drop \
             f32.const 0 f32.const 0 f32.const 0 f32.const 1"
        ));
        let error = translate(&wasm(&import, &fs)).expect_err("reject invalid texture signature");
        assert!(
            matches!(error, TranslateError::UnsupportedType(_)),
            "{name} accepted {signature}: {error:?}"
        );
    }
}
