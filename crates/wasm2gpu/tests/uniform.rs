//! Uniform struct fields, fixed arrays, and the host `gpu_struct!` layout.
//!
//! Offsets come from naga's layout of the lowered WGSL, not from a text
//! snapshot. `br_table` is covered in `control_flow`.

use ornis_gpu_abi::GpuSafeLayout;
use ornis_gpu_abi::GpuVec3;
use ornis_gpu_abi::gpu_struct;
use ornis_gpu_abi::symbols;

#[path = "support/uniform.rs"]
mod support;
use support::*;

gpu_struct! {
    struct FrameTint {
        color: GpuVec3,
        intensity: f32,
    }
}

gpu_struct! {
    struct TwoVec {
        a: GpuVec3,
        pad: [u8; 4],
        b: GpuVec3,
    }
}

fn tint_custom() -> String {
    types("\\4f\\47\\54\\31\\01\\00\\02\\00\\05\\01")
}

fn two_vec_custom() -> String {
    types("\\4f\\47\\54\\31\\01\\00\\02\\00\\05\\05")
}

#[test]
fn uniform_struct_offsets_match_gpu_struct() {
    let got = wgsl(
        &tint_custom(),
        &import(
            symbols::UNIFORM_READ_STRUCT,
            "(param i32 i32 i32) (result f32 f32 f32 f32)",
        ),
        &format!(
            "(param f32 f32) (result f32 f32 f32 f32)
             i32.const 0 i32.const 0 i32.const 0 call ${read}",
            read = symbols::UNIFORM_READ_STRUCT,
        ),
    );
    let (offsets, span) = struct_layout(&got, 2);
    assert!(offsets[0] == FrameTint::FIELD_OFFSETS[0]);
    assert!(offsets[1] == FrameTint::FIELD_OFFSETS[1]);
    assert!(span == FrameTint::SIZE);
}

#[test]
fn uniform_two_vec3_offsets_match_the_host_pad() {
    let got = wgsl(
        &two_vec_custom(),
        &import(
            symbols::UNIFORM_READ_STRUCT,
            "(param i32 i32 i32) (result f32 f32 f32 f32 f32 f32)",
        ),
        &format!(
            "(param f32 f32) (result f32 f32 f32 f32)
             i32.const 0 i32.const 0 i32.const 0 call ${read}
             drop drop",
            read = symbols::UNIFORM_READ_STRUCT,
        ),
    );
    let (offsets, span) = struct_layout(&got, 2);
    assert!(offsets[0] == TwoVec::FIELD_OFFSETS[0]);
    assert!(offsets[1] == TwoVec::FIELD_OFFSETS[2]);
    assert!(span == TwoVec::SIZE);
}

#[test]
fn uniform_f32_array_uses_uniform_stride_and_a_dynamic_index() {
    let got = wgsl(
        "",
        &import(
            symbols::UNIFORM_READ_F32,
            "(param i32 i32 i32 i32) (result f32)",
        ),
        &format!(
            "(param f32 f32 i32) (result f32 f32 f32 f32)
             i32.const 0 i32.const 1 i32.const 4 local.get 2 call ${read}
             f32.const 0 f32.const 0 f32.const 1",
            read = symbols::UNIFORM_READ_F32,
        ),
    );
    let module = naga::front::wgsl::parse_str(&got).expect("parse");
    let mut found = false;
    for (_, global) in module.global_variables.iter() {
        if global.space != naga::AddressSpace::Uniform {
            continue;
        }
        let naga::TypeInner::Struct { members, .. } = &module.types[global.ty].inner else {
            continue;
        };
        let Some(member) = members.first() else {
            continue;
        };
        let naga::TypeInner::Array { stride, size, .. } = &module.types[member.ty].inner else {
            continue;
        };
        assert!(member.offset == 0);
        assert!(*stride == f32::UNIFORM_STRIDE);
        assert!(matches!(size, naga::ArraySize::Constant(n) if n.get() == 4));
        found = true;
    }
    assert!(found, "{got}");
    assert!(got.contains('['), "{got}");
}

#[test]
fn dynamic_uniform_field_index_is_rejected() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         i32.const 0 i32.const 0 i32.const 0 local.get 2 call ${read}
         f32.const 0 f32.const 0 f32.const 1",
        read = symbols::UNIFORM_FIELD_F32,
    );
    let msg = type_msg(err(
        &tint_custom(),
        &import(
            symbols::UNIFORM_FIELD_F32,
            "(param i32 i32 i32 i32) (result f32)",
        ),
        &fs,
    ));
    assert!(
        msg.contains("uniform field index must be a constant"),
        "{msg}"
    );
}

#[test]
fn dynamic_uniform_length_is_rejected() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         i32.const 0 i32.const 1 local.get 2 i32.const 0 call ${read}
         f32.const 0 f32.const 0 f32.const 1",
        read = symbols::UNIFORM_READ_F32,
    );
    let msg = type_msg(err(
        "",
        &import(
            symbols::UNIFORM_READ_F32,
            "(param i32 i32 i32 i32) (result f32)",
        ),
        &fs,
    ));
    assert!(
        msg.contains("uniform array length must be a constant"),
        "{msg}"
    );
}

#[test]
fn zero_uniform_length_is_rejected() {
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         i32.const 0 i32.const 1 i32.const 0 i32.const 0 call ${read}
         f32.const 0 f32.const 0 f32.const 1",
        read = symbols::UNIFORM_READ_F32,
    );
    let msg = type_msg(err(
        "",
        &import(
            symbols::UNIFORM_READ_F32,
            "(param i32 i32 i32 i32) (result f32)",
        ),
        &fs,
    ));
    assert!(
        msg.contains("uniform array length must be positive"),
        "{msg}"
    );
}

#[test]
fn one_binding_cannot_be_two_uniform_types() {
    let imports = format!(
        "{a} {b}",
        a = import(
            symbols::UNIFORM_READ_F32,
            "(param i32 i32 i32 i32) (result f32)",
        ),
        b = import(
            symbols::UNIFORM_READ_U32,
            "(param i32 i32 i32 i32) (result i32)",
        ),
    );
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         i32.const 0 i32.const 1 i32.const 4 i32.const 0 call ${f}
         drop
         i32.const 0 i32.const 1 i32.const 4 i32.const 0 call ${u}
         drop
         {ret}",
        f = symbols::UNIFORM_READ_F32,
        u = symbols::UNIFORM_READ_U32,
        ret = rgba(),
    );
    let msg = type_msg(err("", &imports, &fs));
    assert!(msg.contains("two uniform types"), "{msg}");
}

#[test]
fn one_binding_cannot_be_uniform_and_storage() {
    let imports = format!(
        "{u} {s}",
        u = import(
            symbols::UNIFORM_READ_F32,
            "(param i32 i32 i32 i32) (result f32)",
        ),
        s = import(
            symbols::STORAGE_READ_F32,
            "(param i32 i32 i32) (result f32)"
        ),
    );
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         i32.const 0 i32.const 1 i32.const 4 i32.const 0 call ${u}
         drop
         i32.const 0 i32.const 1 i32.const 0 call ${s}
         f32.const 0 f32.const 0 f32.const 1",
        u = symbols::UNIFORM_READ_F32,
        s = symbols::STORAGE_READ_F32,
    );
    let msg = type_msg(err("", &imports, &fs));
    assert!(msg.contains("both uniform and storage"), "{msg}");
}

#[test]
fn uniform_binding_over_64kib_is_rejected() {
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         i32.const 0 i32.const 1 i32.const 4097 i32.const 0 call ${read}
         f32.const 0 f32.const 0 f32.const 1",
        read = symbols::UNIFORM_READ_F32,
    );
    let msg = type_msg(err(
        "",
        &import(
            symbols::UNIFORM_READ_F32,
            "(param i32 i32 i32 i32) (result f32)",
        ),
        &fs,
    ));
    assert!(msg.contains("64KiB"), "{msg}");
}
