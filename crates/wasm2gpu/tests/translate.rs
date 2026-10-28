//! One vertex+fragment pass: uniform `raw_read` and `texture_sample_2d`.

use ornis_gpu_abi::FieldLayout;
use ornis_gpu_abi::GpuSafeLayout;
use ornis_gpu_abi::GpuVec3;
use ornis_gpu_abi::place_fields;
use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::TranslateError;
use ornis_wasm2gpu::translate;

fn pass_wat() -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
            (import "{module}" "{read}" (func $read (param i32 i32 i32) (result f32 f32 f32 f32)))
            (import "{module}" "{sample}" (func $sample (param i32 i32 i32 i32 f32 f32) (result f32 f32 f32 f32)))
            (func (export "{vs}") (param f32 f32 f32 f32) (result f32 f32 f32 f32 f32 f32)
                local.get 0
                local.get 1
                f32.const 0
                f32.const 1
                local.get 2
                local.get 3)
            (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
                i32.const 1
                i32.const 0
                i32.const 1
                i32.const 1
                local.get 0
                local.get 1
                call $sample
                i32.const 0
                i32.const 0
                i32.const 0
                call $read
                f32.mul))
        "#,
        module = symbols::MODULE,
        read = symbols::RAW_READ,
        sample = symbols::TEXTURE_SAMPLE_2D,
        vs = symbols::VS_MAIN,
        fs = symbols::FS_MAIN,
    ))
    .expect("wat")
}

#[test]
fn uniform_and_texture_round_trip() {
    let translated = translate(&pass_wat()).expect("translate");
    assert!(translated.wgsl.contains("var<uniform>"));
    assert!(translated.wgsl.contains("@group(0)"));
    assert!(translated.wgsl.contains("@binding(0)"));
    assert!(translated.wgsl.contains("@group(1)"));
    assert!(translated.wgsl.contains("texture_2d<f32>"));
    assert!(translated.wgsl.contains(symbols::VS_MAIN));
    assert!(translated.wgsl.contains(symbols::FS_MAIN));
    assert!(!translated.wgsl.contains("binding_array"));
    assert!(!translated.vertex_spirv.is_empty());
    assert!(!translated.fragment_spirv.is_empty());
}

#[test]
fn linear_memory_is_rejected() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "vs_main") (param f32 f32 f32 f32) (result f32 f32 f32 f32 f32 f32)
                i32.const 0
                i32.load
                drop
                local.get 0
                local.get 1
                f32.const 0
                f32.const 1
                local.get 2
                local.get 3)
            (func (export "fs_main") (param f32 f32) (result f32 f32 f32 f32)
                f32.const 0
                f32.const 0
                f32.const 0
                f32.const 1))
        "#,
    )
    .expect("wat");
    let err = translate(&wasm).expect_err("memory");
    assert!(matches!(err, TranslateError::LinearMemory));
}

#[test]
fn dynamic_binding_is_rejected() {
    let wasm = wat::parse_str(format!(
        r#"(module
            (import "{module}" "{read}" (func $read (param i32 i32 i32) (result f32 f32 f32 f32)))
            (func (export "{vs}") (param f32 f32 f32 f32) (result f32 f32 f32 f32 f32 f32)
                local.get 0
                local.get 1
                f32.const 0
                f32.const 1
                local.get 2
                local.get 3)
            (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
                (local i32)
                i32.const 0
                local.get 2
                i32.const 0
                call $read))
        "#,
        module = symbols::MODULE,
        read = symbols::RAW_READ,
        vs = symbols::VS_MAIN,
        fs = symbols::FS_MAIN,
    ))
    .expect("wat");
    let err = translate(&wasm).expect_err("dynamic");
    assert!(matches!(err, TranslateError::DynamicBinding));
}

#[test]
fn naga_accepts_vec3_then_f32_at_place_fields_offsets() {
    let mut offsets = [0u32; 2];
    let span = place_fields(
        &[
            FieldLayout {
                size: GpuVec3::SIZE,
                align: GpuVec3::ALIGN,
            },
            FieldLayout {
                size: f32::SIZE,
                align: f32::ALIGN,
            },
        ],
        true,
        &mut offsets,
    );
    let mut module = naga::Module::default();
    let f32_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Float,
                width: 4,
            }),
        },
        naga::Span::default(),
    );
    let vec3 = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Vector {
                size: naga::VectorSize::Tri,
                scalar: naga::Scalar {
                    kind: naga::ScalarKind::Float,
                    width: 4,
                },
            },
        },
        naga::Span::default(),
    );
    module.types.insert(
        naga::Type {
            name: Some("Frame".to_string()),
            inner: naga::TypeInner::Struct {
                members: vec![
                    naga::StructMember {
                        name: Some("color".to_string()),
                        ty: vec3,
                        binding: None,
                        offset: offsets[0],
                    },
                    naga::StructMember {
                        name: Some("intensity".to_string()),
                        ty: f32_ty,
                        binding: None,
                        offset: offsets[1],
                    },
                ],
                span,
            },
        },
        naga::Span::default(),
    );
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("layout");
    assert!(offsets[1] == GpuVec3::SIZE);
}
