//! Tests for the lowering passes: translation, stage IO and uniformity.

use super::flow::uniform_callee;
use super::helpers::{derivative_axis, unary_math};
use super::memory::local_label;
use super::texture::{SampleMip, SampleShape, compare_import, sample_import};
use super::{Lowering, VEC2_LANES, VEC3_LANES};

/// The uniform list answers per symbol, never by prefix.
#[test]
pub(crate) fn uniformity_is_decided_per_symbol_not_by_prefix() {
    // Every entry is reachable both ways: bare test-IR name and SDK
    // spelling (`import_uniform_read_f32` from a mangled call).
    assert!(uniform_callee(ornis_gpu_abi::symbols::UNIFORM_READ_F32));
    // `imported()` strips the shim's own `import_` prefix, so the SDK
    // spelling reaches the list as the ABI symbol.
    assert!(uniform_callee(
        "_RNvNtCs7qzl3DdH8fB_16ornis_gpu_shader3abi23import_uniform_read_f32"
    ));
    // A helper whose name only *sounds* uniform is per-invocation: the
    // prefix granted it before, which was the finding.
    assert!(!uniform_callee("uniform_helper"));
    // The per-invocation surface is absent from the list.
    for name in [
        "global_invocation_lane",
        "local_invocation_index",
        "storage_read_f32",
        "texture_load_2d_f32",
        "texture_sample_2d_f32x4",
        "workgroup_load_u32",
        "vertex_index",
        "instance_index",
        "fragment_position_lane",
    ] {
        assert!(!uniform_callee(name), "{name} must not be uniform");
    }
    // Dimension queries are.
    assert!(uniform_callee("texture_width_2d"));
    assert!(uniform_callee("storage_len_f32"));
}

/// Lower a fragment entry written as LLVM IR and return the module.
///
/// Every sample-path test does the same four things — parse, pick `fs_main`
/// out of the module, lower it, finish — so they go through here instead of
/// repeating the setup.
/// `(offsets, span)` of uniform struct `name` in `module`.
pub(crate) fn uniform_members(module: &naga::Module, name: &str) -> (Vec<u32>, u32) {
    let ty = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .map(|(handle, _)| handle)
        .expect("uniform struct type");
    let naga::TypeInner::Struct { members, span } = &module.types[ty].inner else {
        panic!("expected a struct");
    };
    (members.iter().map(|member| member.offset).collect(), *span)
}

/// Lower `cs_main` with the given uniform-struct tag table.
pub(crate) fn lower_compute(ir: &str, structs: &[&[u8]]) -> naga::Module {
    let func = crate::parse::parse_module(ir)
        .expect("parse")
        .funcs
        .into_iter()
        .find(|func| func.name == ornis_gpu_abi::symbols::CS_MAIN)
        .expect("entry");
    let mut lowering = Lowering::new(structs, crate::TextureSampling::SingleSample);
    lowering
        .push(&func, ornis_gpu_abi::symbols::CS_WORKGROUP)
        .expect("lower");
    lowering.finish()
}

pub(crate) fn lower_fragment(ir: &str) -> naga::Module {
    let func = crate::parse::parse_module(ir)
        .expect("parse")
        .funcs
        .into_iter()
        .find(|func| func.name == "fs_main")
        .expect("entry");
    let mut lowering = Lowering::new(&[], crate::TextureSampling::SingleSample);
    lowering
        .push(&func, ornis_gpu_abi::symbols::CS_WORKGROUP)
        .expect("lower");
    lowering.finish()
}

#[test]
pub(crate) fn unary_math_maps_the_sdk_symbols() {
    use ornis_gpu_abi::symbols::{GPU_EXP, GPU_LN, GPU_SQRT, GPU_TAN};
    assert_eq!(unary_math(GPU_SQRT), Some(naga::MathFunction::Sqrt));
    assert_eq!(unary_math(GPU_TAN), Some(naga::MathFunction::Tan));
    assert_eq!(unary_math(GPU_LN), Some(naga::MathFunction::Log));
    assert_eq!(unary_math(GPU_EXP), Some(naga::MathFunction::Exp));
    // `gpu_exp2` starts with `gpu_exp`: the generic spelling has to win,
    // otherwise every `exp2` in a shader becomes an `exp`.
    assert_eq!(unary_math("gpu_exp2"), Some(naga::MathFunction::Exp2));
    assert_eq!(unary_math("gpu_log2"), Some(naga::MathFunction::Log2));
    assert_eq!(unary_math("gpu_floor"), Some(naga::MathFunction::Floor));
    assert_eq!(unary_math("gpu_ceil"), Some(naga::MathFunction::Ceil));
    assert_eq!(unary_math("llvm.sqrt.f32"), Some(naga::MathFunction::Sqrt));
    assert_eq!(unary_math("llvm.fabs.f32"), Some(naga::MathFunction::Abs));
}

#[test]
pub(crate) fn an_inline_counter_is_dropped_from_a_local_name() {
    assert_eq!(local_label("lo.sroa.0.0.i1522"), "lo.sroa.0.0");
    assert_eq!(local_label("k.sroa.0.0.i1520"), "k.sroa.0.0");
    assert_eq!(local_label("lo.sroa.9.01521"), "lo.sroa.9");
    assert_eq!(local_label("sum_y.sroa.0.0.i101"), "sum_y.sroa.0.0");
    assert_eq!(local_label("_0.sroa.0.0.i11"), "_0.sroa.0.0");
    assert_eq!(local_label("lo.sroa.9.01519"), "lo.sroa.9");
    // A counter in the middle of the name goes with the trailing ones: it
    // numbers the same function-wide stream.
    assert_eq!(local_label("15.i156.sroa.0.0"), "15.sroa.0.0");
    assert_eq!(local_label("15.i77.sroa.0.0"), "15.sroa.0.0");
    // A counter can also hide behind the bare inlining marker.
    assert_eq!(local_label("0..sroa.0.0.0.i554.i"), "0..sroa.0.0.0");
    assert_eq!(local_label("0..sroa.0.0.0.i267.i"), "0..sroa.0.0.0");
    // Member indices are one or two digits and stay.
    assert_eq!(local_label("lo.sroa.9.0"), "lo.sroa.9.0");
    assert_eq!(local_label("lo.sroa.12.lcssa"), "lo.sroa.12.lcssa");
    assert_eq!(local_label("lo.sroa.0.0"), "lo.sroa.0.0");
    assert_eq!(local_label("west.blue"), "west.blue");
    // A segment that only looks like a counter keeps its place.
    assert_eq!(local_label("lo.sroa.0.quad.x"), "lo.sroa.0.quad.x");
    assert_eq!(local_label("value.inline"), "value.inline");
}

#[test]
pub(crate) fn sample_imports_name_their_shape_and_mip_source() {
    use SampleMip::{Auto, Operand, Zero};
    use SampleShape::{Cube, Plane};
    assert_eq!(
        sample_import("texture_sample_level_cube_f32x4"),
        Some((Cube, Operand))
    );
    assert_eq!(
        sample_import("texture_sample_cube_f32x4"),
        Some((Cube, Auto))
    );
    assert_eq!(
        sample_import("texture_sample_2d_f32x4"),
        Some((Plane, Auto))
    );
    assert_eq!(
        sample_import("texture_sample_level_2d_f32x4"),
        Some((Plane, Zero))
    );
    assert_eq!(sample_import("texture_load_2d_f32x4"), None);
    assert_eq!(Cube.coordinate_lanes(), VEC3_LANES as usize);
    assert_eq!(Plane.coordinate_lanes(), VEC2_LANES as usize);
}

#[test]
pub(crate) fn cube_sample_lowers_to_a_cube_image() {
    // Bindings 0:0 (cube), 0:1 (sampler), level and lane as literals: the
    // shape has to come from the import name alone.
    let ir = format!(
        r#"
declare float @{sample}(i32, i32, i32, i32, float, float, float, float, i32)
declare void @fragment_store_f32x4(float, float, float, float)
define void @fs_main() {{
start:
  %v = call float @{sample}(i32 0, i32 0, i32 0, i32 1, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, i32 0)
  call void @fragment_store_f32x4(float %v, float %v, float %v, float %v)
  ret void
}}
"#,
        sample = ornis_gpu_abi::symbols::TEXTURE_SAMPLE_LEVEL_CUBE_F32X4
    );
    let module = lower_fragment(&ir);
    let cube = module
        .types
        .iter()
        .find(|(_, ty)| {
            matches!(
                &ty.inner,
                naga::TypeInner::Image {
                    dim: naga::ImageDimension::Cube,
                    ..
                }
            )
        })
        .map(|(handle, _)| handle)
        .expect("a cube image type");
    let consumed = module.entry_points.iter().any(|entry| {
            entry.function.expressions.iter().any(|(_, expr)| {
                matches!(
                    expr,
                    naga::Expression::ImageSample { level: naga::SampleLevel::Exact(_), .. }
                )
            }) && entry.function.expressions.iter().any(|(_, expr)| {
                matches!(expr, naga::Expression::GlobalVariable(var) if module.global_variables[*var].ty == cube)
            })
        });
    assert!(
        consumed,
        "the sample reads the cube image at an exact level"
    );
}

#[test]
pub(crate) fn compare_imports_name_their_shape() {
    use ornis_gpu_abi::symbols::{
        TEXTURE_SAMPLE_COMPARE_2D_ARRAY_F32, TEXTURE_SAMPLE_COMPARE_CUBE_ARRAY_F32,
    };
    assert_eq!(
        compare_import(TEXTURE_SAMPLE_COMPARE_2D_ARRAY_F32),
        Some(SampleShape::Plane)
    );
    assert_eq!(
        compare_import(TEXTURE_SAMPLE_COMPARE_CUBE_ARRAY_F32),
        Some(SampleShape::Cube)
    );
    assert_eq!(compare_import("texture_sample_2d_f32x4"), None);
    // A compare import must not be mistaken for a plain sample: it has no
    // level and returns a scalar visibility factor.
    assert_eq!(sample_import(TEXTURE_SAMPLE_COMPARE_2D_ARRAY_F32), None);
    assert_eq!(sample_import(TEXTURE_SAMPLE_COMPARE_CUBE_ARRAY_F32), None);
}

#[test]
pub(crate) fn depth_compare_samples_validate() {
    use ornis_gpu_abi::symbols::{
        TEXTURE_SAMPLE_COMPARE_2D_ARRAY_F32, TEXTURE_SAMPLE_COMPARE_CUBE_ARRAY_F32,
    };
    let ir = format!(
        r#"
declare float @{plane}(i32, i32, i32, i32, float, float, i32, float)
declare float @{cube}(i32, i32, i32, i32, float, float, float, i32, float)
declare void @fragment_store_f32x4(float, float, float, float)
define void @fs_main() {{
start:
  %a = call float @{plane}(i32 0, i32 0, i32 0, i32 1, float 5.000000e-01, float 5.000000e-01, i32 0, float 9.900000e-01)
  %b = call float @{cube}(i32 0, i32 2, i32 0, i32 1, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00, i32 1, float 9.900000e-01)
  %s = fadd float %a, %b
  call void @fragment_store_f32x4(float %s, float %s, float %s, float %s)
  ret void
}}
"#,
        plane = TEXTURE_SAMPLE_COMPARE_2D_ARRAY_F32,
        cube = TEXTURE_SAMPLE_COMPARE_CUBE_ARRAY_F32
    );
    let module = lower_fragment(&ir);
    // Both samples compare, so both carry a depth reference and read an
    // arrayed depth image through a comparison sampler.
    let compared = module
        .entry_points
        .iter()
        .flat_map(|entry| entry.function.expressions.iter())
        .any(|(_, expr)| {
            matches!(
                expr,
                naga::Expression::ImageSample {
                    depth_ref: Some(_),
                    array_index: Some(_),
                    ..
                }
            )
        });
    assert!(
        compared,
        "a compare sample carries depth_ref and array_index"
    );
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("depth compare samples validate");
}

#[test]
pub(crate) fn cube_and_auto_level_samples_validate() {
    // A cube at an explicit level (the prefiltered IBL chain) and a 2D at
    // the level the derivatives pick (the BRDF LUT). naga is the only place
    // the "implicit level is fragment-only" rule is enforced, so the module
    // goes through the real validator.
    use ornis_gpu_abi::symbols::{TEXTURE_SAMPLE_2D_F32X4, TEXTURE_SAMPLE_LEVEL_CUBE_F32X4};
    let ir = format!(
        r#"
declare float @{cube}(i32, i32, i32, i32, float, float, float, float, i32)
declare float @{plane}(i32, i32, i32, i32, float, float, i32)
declare void @fragment_store_f32x4(float, float, float, float)
define void @fs_main() {{
start:
  %a = call float @{cube}(i32 0, i32 0, i32 0, i32 1, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, i32 0)
  %b = call float @{plane}(i32 0, i32 2, i32 0, i32 3, float 5.000000e-01, float 2.500000e-01, i32 1)
  %s = fadd float %a, %b
  call void @fragment_store_f32x4(float %s, float %s, float %s, float %s)
  ret void
}}
"#,
        cube = TEXTURE_SAMPLE_LEVEL_CUBE_F32X4,
        plane = TEXTURE_SAMPLE_2D_F32X4
    );
    let module = lower_fragment(&ir);
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("cube and auto-level samples validate");
}

#[test]
pub(crate) fn derivative_imports_lower_to_naga_derivatives() {
    use ornis_gpu_abi::symbols::{GPU_DPDX, GPU_DPDY};
    assert_eq!(derivative_axis(GPU_DPDX), Some(naga::DerivativeAxis::X));
    assert_eq!(derivative_axis(GPU_DPDY), Some(naga::DerivativeAxis::Y));
    assert_eq!(derivative_axis("gpu_sqrt"), None);
    // A derivative needs the fragment stage: `dpdx` exists only there, so
    // the check runs on a fragment entry writing the derivative out.
    let ir = format!(
        r#"
declare float @{dx}(float)
declare void @fragment_store_f32x4(float, float, float, float)
define void @fs_main() {{
start:
  %v = call float @{dx}(float 1.000000e+00)
  call void @fragment_store_f32x4(float %v, float %v, float %v, float %v)
  ret void
}}
"#,
        dx = GPU_DPDX
    );
    let module = lower_fragment(&ir);
    let has_derivative = module.entry_points.iter().any(|entry| {
        entry
            .function
            .expressions
            .iter()
            .any(|(_, expr)| matches!(expr, naga::Expression::Derivative { .. }))
    });
    assert!(
        has_derivative,
        "dpdx lowers to naga::Expression::Derivative"
    );
}

#[test]
pub(crate) fn vec3_uniform_struct_uses_wgsl_offsets() {
    use ornis_gpu_abi::{TYPE_TAG_F32, TYPE_TAG_U32, TYPE_TAG_VEC3};
    let ir = r#"
declare float @uniform_field_f32(i32, i32, i32, i32)
declare void @storage_write_f32(i32, i32, i32, float)
define void @cs_main(i32 %id) {
start:
  %v = call float @uniform_field_f32(i32 0, i32 0, i32 0, i32 1)
  call void @storage_write_f32(i32 0, i32 2, i32 %id, float %v)
  ret void
}
"#;
    // `vec3` is 12 bytes but aligns to 16; the scalar after it sits at
    // byte 12 and the struct rounds up to 32 (uniform alignment).
    let structs: &[&[u8]] = &[&[TYPE_TAG_VEC3, TYPE_TAG_F32, TYPE_TAG_U32]];
    let module = lower_compute(ir, structs);
    let (offsets, span) = uniform_members(&module, "UniformStruct0");
    assert_eq!(span, 32);
    assert_eq!(offsets, [0, 12, 16]);
}

/// Reading `csm.vps[cascade]` — an element of a fixed-size uniform array by
/// a runtime index, then one of its `vec4` columns and one lane.
#[test]
pub(crate) fn a_uniform_array_element_reads_by_a_runtime_index() {
    use ornis_gpu_abi::symbols::UNIFORM_ARRAY_VEC4_LANE;
    use ornis_gpu_abi::{TYPE_TAG_ARRAY, TYPE_TAG_NESTED, TYPE_TAG_VEC4};
    let ir = format!(
        r#"
declare float @{name}(i32, i32, i32, i32, i32, i32, i32)
declare void @storage_write_f32(i32, i32, i32, float)
define void @cs_main(i32 %id) {{
start:
  %v = call float @{name}(i32 0, i32 0, i32 0, i32 0, i32 %id, i32 2, i32 1)
  call void @storage_write_f32(i32 0, i32 2, i32 %id, float %v)
  ret void
}}
"#,
        name = UNIFORM_ARRAY_VEC4_LANE
    );
    // Struct 0 is `Csm`, struct 1 is one `CsmVp`: four `vec4` columns, the
    // same bytes a `mat4x4<f32>` occupies.
    let vp: &[u8] = &[TYPE_TAG_VEC4, TYPE_TAG_VEC4, TYPE_TAG_VEC4, TYPE_TAG_VEC4];
    let csm: &[u8] = &[TYPE_TAG_ARRAY, TYPE_TAG_NESTED + 1, 4, TYPE_TAG_VEC4];
    let structs: &[&[u8]] = &[csm, vp];
    let module = lower_compute(&ir, structs);
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("a dynamic uniform array index validates");
    let indexed = module.entry_points.iter().any(|entry| {
        entry
            .function
            .expressions
            .iter()
            .any(|(_, expr)| matches!(expr, naga::Expression::Access { .. }))
    });
    assert!(
        indexed,
        "the element read is a dynamic naga::Expression::Access"
    );
}

/// The same read one struct deeper: `Lighting.csm.vps[cascade].cols[lane]`.
#[test]
pub(crate) fn a_nested_uniform_array_element_reads_through_two_fields() {
    use ornis_gpu_abi::symbols::UNIFORM_NESTED_ARRAY_VEC4_LANE;
    use ornis_gpu_abi::{TYPE_TAG_ARRAY, TYPE_TAG_F32, TYPE_TAG_NESTED, TYPE_TAG_VEC4};
    let ir = format!(
        r#"
declare float @{name}(i32, i32, i32, i32, i32, i32, i32, i32)
declare void @storage_write_f32(i32, i32, i32, float)
define void @cs_main(i32 %id) {{
start:
  %v = call float @{name}(i32 0, i32 0, i32 1, i32 1, i32 0, i32 %id, i32 3, i32 0)
  call void @storage_write_f32(i32 0, i32 2, i32 %id, float %v)
  ret void
}}
"#,
        name = UNIFORM_NESTED_ARRAY_VEC4_LANE
    );
    // Struct 0 is `Csm` (`vps: array<CsmVp, 4>` then a `vec4`), struct 1 is
    // `Lighting` (`0: vec4`, `1: Csm`, `2: f32`), struct 2 is `CsmVp`.
    let csm: &[u8] = &[TYPE_TAG_ARRAY, TYPE_TAG_NESTED + 2, 4, TYPE_TAG_VEC4];
    let lighting: &[u8] = &[TYPE_TAG_VEC4, TYPE_TAG_NESTED, TYPE_TAG_F32];
    let vp: &[u8] = &[TYPE_TAG_VEC4, TYPE_TAG_VEC4, TYPE_TAG_VEC4, TYPE_TAG_VEC4];
    let structs: &[&[u8]] = &[csm, lighting, vp];
    let module = lower_compute(&ir, structs);
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("a nested dynamic uniform array index validates");
    let indexed = module.entry_points.iter().any(|entry| {
        entry
            .function
            .expressions
            .iter()
            .any(|(_, expr)| matches!(expr, naga::Expression::Access { .. }))
    });
    assert!(indexed, "the element read keeps its runtime index");
}

/// The layout of `struct Lighting { ambient: vec4, lights: array<Light, 8>,
/// count: u32 }` where `Light` is `{ vec4 }` and therefore 16 bytes: the
/// array of eight spans 128 bytes, so the scalar after it lands at 144 and
/// the struct rounds up to 160.
#[test]
pub(crate) fn a_uniform_struct_with_an_array_field_places_the_fields_after_it() {
    use ornis_gpu_abi::{TYPE_TAG_ARRAY, TYPE_TAG_NESTED, TYPE_TAG_U32, TYPE_TAG_VEC4};
    let ir = r#"
declare i32 @uniform_field_u32(i32, i32, i32, i32)
declare void @storage_write_u32(i32, i32, i32, i32)
define void @cs_main(i32 %id) {
start:
  %v = call i32 @uniform_field_u32(i32 0, i32 0, i32 0, i32 2)
  call void @storage_write_u32(i32 0, i32 2, i32 %id, i32 %v)
  ret void
}
"#;
    // Struct 0 is `Lighting`, struct 1 is the 16-byte `Light` record.
    let light: &[u8] = &[TYPE_TAG_VEC4];
    let lighting: &[u8] = &[
        TYPE_TAG_VEC4,
        TYPE_TAG_ARRAY,
        TYPE_TAG_NESTED + 1,
        8,
        TYPE_TAG_U32,
    ];
    let structs: &[&[u8]] = &[lighting, light];
    let module = lower_compute(ir, structs);
    let (offsets, span) = uniform_members(&module, "UniformStruct0");
    assert_eq!(span, 160);
    assert_eq!(offsets, [0, 16, 144]);
    let array_ty = module
        .types
        .iter()
        .filter(|(_, ty)| matches!(ty.inner, naga::TypeInner::Struct { .. }))
        .find_map(|(_, ty)| match &ty.inner {
            naga::TypeInner::Struct { members, .. } if members.len() == 3 => Some(members[1].ty),
            _ => None,
        })
        .expect("the array field");
    let naga::TypeInner::Array { stride, size, .. } = &module.types[array_ty].inner else {
        panic!("the second field is an array");
    };
    assert_eq!(*stride, 16);
    assert_eq!(
        *size,
        naga::ArraySize::Constant(8.try_into().expect("eight"))
    );
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("a uniform struct with an array validates");
}
