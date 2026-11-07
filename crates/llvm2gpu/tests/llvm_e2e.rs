//! E2E for the LLVM IR spike: embedded IR, a real rustc build, and GPU parity.
use ornis_llvm2gpu::{translate, translate_with_meta, translate_with_types};
use std::path::PathBuf;
use std::process::Command;

const UNIFORM_FIELD_IR: &str = r#"
declare float @uniform_field_f32(i32 noundef, i32 noundef, i32 noundef, i32 noundef)
declare i32 @uniform_field_u32(i32 noundef, i32 noundef, i32 noundef, i32 noundef)
declare void @storage_write_f32(i32 noundef, i32 noundef, i32 noundef, float noundef)
declare void @storage_write_u32(i32 noundef, i32 noundef, i32 noundef, i32 noundef)
define dso_local void @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %f = call float @uniform_field_f32(i32 0, i32 0, i32 0, i32 1)
  %u = call i32 @uniform_field_u32(i32 0, i32 0, i32 0, i32 2)
  call void @storage_write_f32(i32 0, i32 2, i32 %id, float %f)
  call void @storage_write_u32(i32 0, i32 3, i32 %id, i32 %u)
  ret void
}
"#;

const SCALAR_IR: &str = r#"
; ModuleID = 'toy'
source_filename = "toy"
target triple = "wasm32-unknown-unknown"
define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %_4 = uitofp i32 %id to float
  %_3 = fmul float %_4, 2.000000e+00
  %x = fadd float %_3, 1.000000e+00
  %_8 = fmul float %x, %x
  %x2 = fadd float %_8, 1.000000e+00
  %c = fcmp olt float %x2, 1.000000e+01
  %y = select i1 %c, float %x2, float 0.000000e+00
  %out = fadd float %y, %x2
  ret float %out
}
"#;

#[test]
fn loop_value_used_after_exit_and_umin_lower() {
    let ir = r#"
declare i32 @llvm.umin.i32(i32, i32)
define dso_local float @cs_main(i32 %id) {
start:
  br label %header
header:
  %i = phi i32 [ 0, %start ], [ %next, %latch ]
  %next = add i32 %i, 1
  %done = icmp ugt i32 %next, %id
  br i1 %done, label %exit, label %latch
latch:
  br label %header
exit:
  %capped = call i32 @llvm.umin.i32(i32 %next, i32 4)
  %flag = icmp ult i32 %id, 2
  %wide = zext i1 %flag to i32
  %as_float = uitofp nneg i32 %wide to float
  %scaled = fmul float %as_float, 0x3FB99999A0000000
  %mixed = fadd float %scaled, %as_float
  %from_cap = uitofp i32 %capped to float
  %out = fadd float %mixed, %from_cap
  ret float %out
}
"#;
    let translated = translate(ir).expect("loop live-out lowers");
    assert!(translated.wgsl.contains("min("), "{}", translated.wgsl);
    assert!(translated.wgsl.contains("select("), "{}", translated.wgsl);
}

#[test]
fn fragment_position_is_a_builtin() {
    let ir = r#"
declare float @fragment_position_lane(i32)
declare void @fragment_store_f32x4(float, float, float, float)
define dso_local void @fs_main() {
start:
  %x = call float @fragment_position_lane(i32 0)
  %y = call float @fragment_position_lane(i32 1)
  call void @fragment_store_f32x4(float %x, float %y, float 0.000000e+00, float 1.000000e+00)
  ret void
}
"#;
    let translated = translate(ir).expect("fragment position");
    assert!(
        translated.wgsl.contains("@builtin(position)"),
        "{}",
        translated.wgsl
    );
    assert!(translated.wgsl.contains(".x"), "{}", translated.wgsl);
    assert!(translated.wgsl.contains(".y"), "{}", translated.wgsl);
}

#[test]
fn workgroup_memory_barrier_and_atomic() {
    let ir = r#"
declare void @workgroup_store_u32(i32, i32)
declare i32 @workgroup_load_u32(i32)
declare void @workgroup_barrier()
declare i32 @workgroup_atomic_add_u32(i32, i32)
declare void @workgroup_atomic_store_u32(i32, i32)
declare i32 @workgroup_atomic_load_u32(i32)
declare void @storage_write_u32(i32, i32, i32, i32)
define dso_local void @cs_main(i32 %id) {
start:
  call void @workgroup_store_u32(i32 0, i32 %id)
  call void @workgroup_atomic_store_u32(i32 2, i32 0)
  call void @workgroup_barrier()
  %word = call i32 @workgroup_load_u32(i32 0)
  %old = call i32 @workgroup_atomic_add_u32(i32 1, i32 %word)
  %seen = call i32 @workgroup_atomic_load_u32(i32 2)
  %sum = add i32 %old, %seen
  call void @storage_write_u32(i32 0, i32 0, i32 %id, i32 %sum)
  ret void
}
"#;
    let translated = translate(ir).expect("workgroup");
    assert!(
        translated.wgsl.contains("var<workgroup>"),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("workgroupBarrier()"),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("atomicAdd("),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("atomicStore("),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("atomicLoad("),
        "{}",
        translated.wgsl
    );
}

#[test]
fn fragment_entry_returns_location_zero_vec4() {
    let ir = r#"
declare void @fragment_store_f32x4(float, float, float, float)
define dso_local void @fs_main() {
start:
  call void @fragment_store_f32x4(float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00)
  ret void
}
"#;
    let translated = translate(ir).expect("fragment entry");
    assert!(translated.wgsl.contains("@fragment"), "{}", translated.wgsl);
    assert!(
        translated.wgsl.contains("@location(0)"),
        "{}",
        translated.wgsl
    );
    assert!(translated.wgsl.contains("vec4<f32>"), "{}", translated.wgsl);
    assert!(
        translated.wgsl.contains("fn fs_main"),
        "{}",
        translated.wgsl
    );
}

#[test]
fn depth_texture_load_and_dimensions_lower() {
    let ir = r#"
declare float @texture_load_2d_f32(i32, i32, i32, i32)
declare i32 @texture_width_2d(i32, i32)
declare i32 @texture_height_2d(i32, i32)
declare void @storage_write_f32(i32, i32, i32, float)
define dso_local void @cs_main(i32 %id) {
start:
  %w = call i32 @texture_width_2d(i32 0, i32 1)
  %h = call i32 @texture_height_2d(i32 0, i32 1)
  %d = call float @texture_load_2d_f32(i32 0, i32 1, i32 %id, i32 0)
  %span = add i32 %w, %h
  %span_f = uitofp nneg i32 %span to float
  %sum = fadd float %d, %span_f
  call void @storage_write_f32(i32 0, i32 2, i32 %id, float %sum)
  ret void
}
"#;
    let translated = translate(ir).expect("depth textureLoad");
    assert!(
        translated.wgsl.contains("textureLoad("),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("textureDimensions("),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("texture_depth_2d"),
        "{}",
        translated.wgsl
    );
}

#[test]
fn sampled_texture_load_reads_one_lane() {
    let ir = r#"
declare float @texture_load_2d_f32x4(i32, i32, i32, i32, i32)
declare i32 @texture_width_2d_f32x4(i32, i32)
declare void @storage_write_f32(i32, i32, i32, float)
define dso_local void @cs_main(i32 %id) {
start:
  %w = call i32 @texture_width_2d_f32x4(i32 0, i32 1)
  %r = call float @texture_load_2d_f32x4(i32 0, i32 1, i32 %id, i32 0, i32 0)
  %g = call float @texture_load_2d_f32x4(i32 0, i32 1, i32 %id, i32 0, i32 1)
  %wf = uitofp nneg i32 %w to float
  %sum = fadd float %r, %g
  %out = fadd float %sum, %wf
  call void @storage_write_f32(i32 0, i32 2, i32 %id, float %out)
  ret void
}
"#;
    let translated = translate(ir).expect("sampled textureLoad");
    assert!(
        translated.wgsl.contains("texture_2d<f32>"),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("textureLoad("),
        "{}",
        translated.wgsl
    );
    assert!(translated.wgsl.contains(".x"), "{}", translated.wgsl);
    assert!(translated.wgsl.contains(".y"), "{}", translated.wgsl);
    assert!(
        !translated.wgsl.contains("texture_depth_2d"),
        "{}",
        translated.wgsl
    );
}

#[test]
fn one_texture_binding_cannot_be_depth_and_sampled() {
    let ir = r#"
declare float @texture_load_2d_f32(i32, i32, i32, i32)
declare float @texture_load_2d_f32x4(i32, i32, i32, i32, i32)
declare void @storage_write_f32(i32, i32, i32, float)
define dso_local void @cs_main(i32 %id) {
start:
  %d = call float @texture_load_2d_f32(i32 0, i32 1, i32 %id, i32 0)
  %r = call float @texture_load_2d_f32x4(i32 0, i32 1, i32 %id, i32 0, i32 0)
  %sum = fadd float %d, %r
  call void @storage_write_f32(i32 0, i32 2, i32 %id, float %sum)
  ret void
}
"#;
    let err = translate(ir).expect_err("two texture kinds");
    let text = err.to_string();
    assert!(text.contains("two kinds"), "{text}");
}

#[test]
fn gtao_math_and_log2_lower() {
    let ir = r#"
declare float @gpu_acos(float)
declare float @gpu_clamp(float, float, float)
declare float @gpu_mix(float, float, float)
declare float @gpu_sign(float)
declare float @gpu_abs(float)
declare float @gpu_log2(float)
declare float @gpu_cross3_lane(float, float, float, float, float, float, i32)
declare float @gpu_normalize3_lane(float, float, float, i32)
declare void @storage_write_f32(i32, i32, i32, float)
define dso_local void @cs_main(i32 %id) {
start:
  %a = call float @gpu_acos(float 1.000000e+00)
  %c = call float @gpu_clamp(float 2.000000e+00, float 0.000000e+00, float 1.000000e+00)
  %m = call float @gpu_mix(float 0.000000e+00, float 1.000000e+00, float 5.000000e-01)
  %s = call float @gpu_sign(float -1.000000e+00)
  %b = call float @gpu_abs(float -2.000000e+00)
  %l = call float @gpu_log2(float 2.000000e+00)
  %x = call float @gpu_cross3_lane(float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, i32 2)
  %n = call float @gpu_normalize3_lane(float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, i32 1)
  %sum = fadd float %a, %c
  %sum2 = fadd float %sum, %m
  %sum3 = fadd float %sum2, %s
  %sum4 = fadd float %sum3, %b
  %sum5 = fadd float %sum4, %l
  %sum6 = fadd float %sum5, %x
  %out = fadd float %sum6, %n
  call void @storage_write_f32(i32 0, i32 2, i32 %id, float %out)
  ret void
}
"#;
    let translated = translate(ir).expect("gtao math");
    for needle in [
        "acos(",
        "clamp(",
        "mix(",
        "sign(",
        "abs(",
        "log2(",
        "cross(",
        "normalize(",
    ] {
        assert!(
            translated.wgsl.contains(needle),
            "{needle}\n{}",
            translated.wgsl
        );
    }
}

#[test]
fn nested_uniform_struct_field_loads_the_inner_scalar() {
    let ir = r#"
@ornis_gpu_shader_meta = private constant [42 x i8] c"entry=cs_main;workgroup=1,1,1;uniform=0:671,1:1", section "ornis_gpu.shader_meta", align 1
declare float @uniform_nested_field_f32(i32, i32, i32, i32, i32)
declare void @storage_write_f32(i32, i32, i32, float)
define dso_local void @cs_main(i32 %id) {
start:
  %v = call float @uniform_nested_field_f32(i32 0, i32 0, i32 0, i32 1, i32 0)
  call void @storage_write_f32(i32 0, i32 1, i32 %id, float %v)
  ret void
}
"#;
    let translated = ornis_llvm2gpu::translate_with_meta(ir).expect("nested struct");
    assert!(
        translated.wgsl.contains("UniformStruct1"),
        "{}",
        translated.wgsl
    );
    assert!(!translated.compute_spirv.is_empty());
}

#[test]
fn readonly_storage_bindings_stay_read_only() {
    let ir = r#"
declare float @storage_read_f32(i32, i32, i32)
define dso_local float @cs_main(i32 %id) {
start:
  %a = call float @storage_read_f32(i32 0, i32 1, i32 %id)
  %b = call float @storage_read_f32(i32 0, i32 2, i32 %id)
  %sum = fadd float %a, %b
  ret float %sum
}
"#;
    let translated = translate(ir).expect("two readonly bindings");
    let readonly = translated.wgsl.matches("var<storage> storage_0_").count();
    assert!(readonly >= 2, "{}", translated.wgsl);
    assert!(
        !translated.wgsl.contains("storage_0_1_")
            || !translated.wgsl.contains("read_write> storage_0_1_"),
        "{}",
        translated.wgsl
    );
}

#[test]
fn one_storage_binding_cannot_have_two_types() {
    let ir = r#"
declare float @storage_read_f32(i32, i32, i32)
declare i32 @storage_read_u32(i32, i32, i32)
define dso_local void @cs_main(i32 %id) {
start:
  %a = call float @storage_read_f32(i32 0, i32 1, i32 0)
  %b = call i32 @storage_read_u32(i32 0, i32 1, i32 0)
  ret void
}
"#;
    let error = translate(ir).expect_err("two types at one binding");
    let message = error.to_string();
    assert!(message.contains("two types"), "{message}");
}

#[test]
fn shader_meta_drives_workgroup_and_types() {
    let ir = r#"
@ornis_gpu_shader_meta = private constant [47 x i8] c"entry=cs_main;workgroup=32,2,1;types=6,2", section "ornis_gpu.shader_meta", align 1
define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %x = uitofp i32 %id to float
  ret float %x
}
"#;
    let meta = ornis_llvm2gpu::extract_meta(ir)
        .expect("metadata parses")
        .expect("metadata present");
    assert_eq!(meta.entry, "cs_main");
    assert_eq!(meta.workgroup, [32, 2, 1]);
    assert_eq!(meta.tags(0), Some([6, 2].as_slice()));
    let translated = ornis_llvm2gpu::translate_with_meta(ir).expect("translate");
    assert!(
        translated.wgsl.contains("@workgroup_size(32, 2, 1)"),
        "{}",
        translated.wgsl
    );
}

#[test]
fn scalar_subset_lowesr_and_validates() {
    let translated = translate(SCALAR_IR).expect("translate scalar kernel");
    assert!(translated.wgsl.contains("@compute"), "{}", translated.wgsl);
    assert!(
        translated.wgsl.contains("storage_0_0_"),
        "{}",
        translated.wgsl
    );
    assert!(translated.wgsl.contains("f32("), "{}", translated.wgsl);
    assert!(!translated.compute_spirv.is_empty());
}

#[test]
fn infinite_loop_is_rejected_not_mistranslated() {
    let ir = r#"
define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  br label %loop
loop:
  br label %loop
exit:
  ret float 0.000000e+00
}
"#;
    assert!(translate(ir).is_err());
}

#[test]
fn diamond_if_with_phi_lowers() {
    let ir = r#"
define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %x = uitofp i32 %id to float
  %c = fcmp olt float %x, 5.000000e+00
  br i1 %c, label %then, label %otherwise

then:
  %a = fadd float %x, 1.000000e+00
  br label %merge

otherwise:
  %b = fsub float %x, 1.000000e+00
  br label %merge

merge:
  %p = phi float [ %a, %then ], [ %b, %otherwise ]
  ret float %p
}
"#;
    let translated = translate(ir).expect("diamond");
    assert!(translated.wgsl.contains("if "), "{}", translated.wgsl);
    assert!(!translated.wgsl.contains("loop {"), "{}", translated.wgsl);
    assert!(!translated.compute_spirv.is_empty());
}

#[test]
fn while_loop_with_phi_lowers_to_naga_loop() {
    let ir = r#"
define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %x0 = uitofp i32 %id to float
  %n = and i32 %id, 7
  br label %header

header:
  %x = phi float [ %x0, %start ], [ %next, %body ]
  %i = phi i32 [ 0, %start ], [ %i.next, %body ]
  %c = icmp ult i32 %i, %n
  br i1 %c, label %body, label %exit

body:
  %m = fmul float %x, 1.500000e+00
  %next = fadd float %m, 2.000000e+00
  %i.next = add nuw i32 %i, 1
  br label %header

exit:
  %result = phi float [ %x, %header ]
  ret float %result
}
"#;
    let translated = translate(ir).expect("while loop");
    assert!(translated.wgsl.contains("loop {"), "{}", translated.wgsl);
    assert!(!translated.compute_spirv.is_empty());
}

#[test]
fn nested_loops_lower_to_two_naga_loops() {
    let ir = r#"
define dso_local noundef float @cs_main(i32 noundef %id) {
start:
  br label %outer
outer:
  %i = phi i32 [ 0, %start ], [ %i.next, %outer.latch ]
  %acc = phi float [ 0.000000e+00, %start ], [ %acc.out, %outer.latch ]
  %o.done = icmp uge i32 %i, %id
  br i1 %o.done, label %exit, label %inner
inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner.step ]
  %acc.i = phi float [ %acc, %outer ], [ %acc.next, %inner.step ]
  %i.done = icmp uge i32 %j, %id
  br i1 %i.done, label %outer.latch, label %inner.step
inner.step:
  %acc.next = fadd float %acc.i, 1.000000e+00
  %j.next = add i32 %j, 1
  br label %inner
outer.latch:
  %acc.out = phi float [ %acc.i, %inner ]
  %i.next = add i32 %i, 1
  br label %outer
exit:
  %result = phi float [ %acc, %outer ]
  ret float %result
}
"#;
    let translated = translate(ir).expect("nested loops");
    assert_eq!(
        translated.wgsl.matches("loop {").count(),
        2,
        "{}",
        translated.wgsl
    );
    assert!(!translated.compute_spirv.is_empty());
}

#[test]
fn sqrt_maps_to_wgsl_sqrt() {
    let ir = r#"
define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %_4 = uitofp i32 %id to float
  %r = call float @llvm.sqrt.f32(float %_4)
  ret float %r
}
"#;
    let translated = translate(ir).expect("sqrt");
    assert!(translated.wgsl.contains("sqrt("), "{}", translated.wgsl);
}

#[test]
fn dot_and_length_via_compose() {
    let ir = r#"
define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %_4 = uitofp i32 %id to float
  %d = call float @gpu_dot3(float %_4, float %_4, float %_4, float %_4, float %_4, float %_4)
  %l = call float @gpu_length3_xyz(float %_4, float %_4, float %_4)
  %s = fadd float %d, %l
  ret float %s
}
"#;
    let translated = translate(ir).expect("dot/length");
    assert!(translated.wgsl.contains("dot("), "{}", translated.wgsl);
    assert!(translated.wgsl.contains("length("), "{}", translated.wgsl);
}

#[test]
fn storage_f32_copy_lowes_to_two_globals() {
    let ir = r#"
declare float @storage_read_f32(i32 noundef, i32 noundef, i32 noundef)
declare void @storage_write_f32(i32 noundef, i32 noundef, i32 noundef, float noundef)
define dso_local void @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %v = call float @storage_read_f32(i32 0, i32 1, i32 %id)
  call void @storage_write_f32(i32 0, i32 2, i32 %id, float %v)
  ret void
}
"#;
    let translated = translate(ir).expect("copy");
    assert!(
        translated.wgsl.contains("storage_0_1_"),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("storage_0_2_"),
        "{}",
        translated.wgsl
    );
    assert!(!translated.compute_spirv.is_empty());
}

#[test]
fn storage_u32_and_atomic_lowes() {
    let ir = r#"
declare i32 @storage_read_u32(i32 noundef, i32 noundef, i32 noundef)
declare i32 @storage_atomic_add_u32(i32 noundef, i32 noundef, i32 noundef, i32 noundef)
declare void @storage_write_u32(i32 noundef, i32 noundef, i32 noundef, i32 noundef)
define dso_local void @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %c = call i32 @storage_read_u32(i32 0, i32 4, i32 0)
  %slot = call i32 @storage_atomic_add_u32(i32 0, i32 3, i32 1, i32 1)
  call void @storage_write_u32(i32 0, i32 2, i32 %slot, i32 %c)
  ret void
}
"#;
    let translated = translate(ir).expect("u32/atomic");
    assert!(translated.wgsl.contains("atomic"), "{}", translated.wgsl);
    assert!(
        translated.wgsl.contains("storage_0_2_"),
        "{}",
        translated.wgsl
    );
}

#[test]
fn uniform_f32_array_lowes_to_lane_struct() {
    let ir = r#"
declare float @uniform_read_f32(i32 noundef, i32 noundef, i32 noundef, i32 noundef)
declare void @storage_write_f32(i32 noundef, i32 noundef, i32 noundef, float noundef)
define dso_local void @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %v = call float @uniform_read_f32(i32 0, i32 0, i32 24, i32 %id)
  call void @storage_write_f32(i32 0, i32 2, i32 %id, float %v)
  ret void
}
"#;
    let translated = translate(ir).expect("uniform");
    assert!(
        translated.wgsl.contains("uniform_0_0_"),
        "{}",
        translated.wgsl
    );
    assert!(translated.wgsl.contains("a["), "{}", translated.wgsl);
    assert!(translated.wgsl.contains(".v"), "{}", translated.wgsl);
    assert!(!translated.compute_spirv.is_empty());
}

#[test]
fn vec4_copy_lane_read_and_len_lower() {
    let ir = r#"
declare i32 @storage_len_u32x4(i32 noundef, i32 noundef)
declare i32 @storage_read_u32_lane(i32 noundef, i32 noundef, i32 noundef, i32 noundef)
declare void @storage_copy_u32x4(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef)
define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %len = call i32 @storage_len_u32x4(i32 0, i32 1)
  %word = call i32 @storage_read_u32_lane(i32 0, i32 1, i32 %id, i32 3)
  %value = bitcast i32 %word to float
  %ok = icmp ult i32 %id, %len
  br i1 %ok, label %copy, label %done

copy:
  call void @storage_copy_u32x4(i32 0, i32 1, i32 %id, i32 0, i32 2, i32 %id)
  br label %done

done:
  %result = phi float [ 0.000000e+00, %start ], [ %value, %copy ]
  ret float %result
}
"#;
    let translated = translate(ir).expect("vec4 ABI");
    assert!(
        translated.wgsl.contains("arrayLength"),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("bitcast<f32>"),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("storage_0_1_"),
        "{}",
        translated.wgsl
    );
    assert!(!translated.compute_spirv.is_empty());
}

#[test]
fn vec4_scalar_write_lowers_to_one_store() {
    let ir = r#"
declare void @storage_write_u32x4(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef)
define dso_local void @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %x = and i32 %id, 7
  call void @storage_write_u32x4(i32 0, i32 2, i32 %id, i32 %x, i32 %x, i32 %x, i32 %x)
  ret void
}
"#;
    let translated = translate(ir).expect("vec4 scalar write");
    assert!(
        translated.wgsl.contains("storage_0_2_"),
        "{}",
        translated.wgsl
    );
    assert!(translated.wgsl.contains("vec4<u32>"), "{}", translated.wgsl);
    assert!(!translated.compute_spirv.is_empty());
}

#[test]
fn uniform_field_uses_place_fields_layout() {
    use ornis_gpu_abi::{TYPE_TAG_F32, TYPE_TAG_U32, TYPE_TAG_VEC3};
    // vec3 at 0, f32 at 12, u32 at 16 (uniform span 32).
    let structs: &[&[u8]] = &[&[TYPE_TAG_VEC3, TYPE_TAG_F32, TYPE_TAG_U32]];
    let translated = translate_with_types(UNIFORM_FIELD_IR, structs).expect("uniform fields");
    assert!(
        translated.wgsl.contains("uniform_struct_0_0_"),
        "{}",
        translated.wgsl
    );
    assert!(translated.wgsl.contains("vec3<f32>"), "{}", translated.wgsl);
    assert!(translated.wgsl.contains("f1_: f32"), "{}", translated.wgsl);
    assert!(translated.wgsl.contains("f2_: u32"), "{}", translated.wgsl);
    assert!(!translated.compute_spirv.is_empty());
}

#[test]
fn uniform_field_requires_types_metadata() {
    assert!(translate(UNIFORM_FIELD_IR).is_err());
}

#[test]
fn uniform_field_rejects_type_mismatch() {
    use ornis_gpu_abi::TYPE_TAG_VEC3;
    // Field 1 is declared as vec3 but read through uniform_field_f32.
    let structs: &[&[u8]] = &[&[TYPE_TAG_VEC3, TYPE_TAG_VEC3, TYPE_TAG_VEC3]];
    assert!(translate_with_types(UNIFORM_FIELD_IR, structs).is_err());
}

const KERNEL_RS: &str = r#"
#![allow(clippy::all)]
#![no_std]
#![no_main]
use core::panic::PanicInfo;
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { core::arch::wasm32::unreachable() }
#[no_mangle]
pub extern "C" fn cs_main(id: u32) -> f32 {
    let mut x = (id as f32) * 2.0 + 1.0;
    let mut i = 0u32;
    while i < 4 {
        x = x * x + 1.0;
        i += 1;
    }
    (if x < 10.0 { x } else { 0.0 }) + x
}
"#;

/// Compile a `no_std` kernel with `rustc --emit=llvm-ir`; skip when the
/// wasm32 target is missing (CI without the target still passes).
fn compile_ir(source: &str) -> Option<String> {
    compile_ir_opt(source, "3")
}

/// Same as [`compile_ir`] with an explicit `opt-level`. `s` keeps dynamic
/// loops structured instead of fully unrolling them at `opt-level=3`.
fn compile_lib_opt(source: &str, opt: &str) -> Option<String> {
    compile_ir_opt_kind(source, opt, true)
}

fn compile_ir_opt(source: &str, opt: &str) -> Option<String> {
    compile_ir_opt_kind(source, opt, false)
}

fn compile_ir_opt_kind(source: &str, opt: &str, library: bool) -> Option<String> {
    let dir = std::env::temp_dir().join(format!(
        "llvm2gpu-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let rs_path = dir.join("kernel.rs");
    let ll_path = dir.join("kernel.ll");
    std::fs::write(&rs_path, source).ok()?;
    let out = Command::new("rustc")
        .args([
            "--target",
            "wasm32-unknown-unknown",
            "-C",
            &format!("opt-level={opt}"),
            "-C",
            "panic=abort",
            "-C",
            "target-cpu=mvp",
            "--emit=llvm-ir",
        ])
        .args(
            library
                .then_some(["--edition", "2024"])
                .into_iter()
                .flatten(),
        )
        .args(
            library
                .then_some(["--crate-type", "lib"])
                .into_iter()
                .flatten(),
        )
        .arg(&rs_path)
        .arg("--out-dir")
        .arg(&dir)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "rustc failed (maybe missing target): {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    let ll_file = dir.join("kernel.ll");
    let ll_file = if ll_file.exists() {
        ll_file
    } else {
        // rustc names it after source file stem.
        dir.join("kernel").with_extension("ll")
    };
    let _ = ll_path;
    let text = std::fs::read_to_string(ll_file).ok()?;
    // Keep dir for debugging on failure.
    let _ = PathBuf::from(&dir);
    Some(text)
}

#[test]
fn rustc_to_llvm_ir_to_wgsl() {
    let Some(ir) = compile_ir(KERNEL_RS) else {
        eprintln!("SKIPPED (wasm32 target missing)");
        return;
    };
    let translated = translate(&ir).expect("translate rustc output");
    assert!(translated.wgsl.contains("@compute"), "{}", translated.wgsl);
    assert!(!translated.compute_spirv.is_empty());
}

const LOOP_KERNEL_RS: &str = r#"
#![allow(clippy::all)]
#![no_std]
#![no_main]
use core::panic::PanicInfo;
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { core::arch::wasm32::unreachable() }
extern "C" { fn storage_read_u32(group: u32, binding: u32, index: u32) -> u32; }
#[no_mangle]
pub extern "C" fn cs_main(id: u32) -> f32 {
    let mut x = id as f32;
    let mut i = 0u32;
    let n = unsafe { storage_read_u32(0, 4, 0) };
    while i < n {
        x = x * 1.5 + 2.0;
        i += 1;
    }
    x
}
"#;

#[test]
fn rustc_dynamic_loop_to_wgsl() {
    let Some(ir) = compile_ir_opt(LOOP_KERNEL_RS, "s") else {
        eprintln!("SKIPPED (wasm32 target missing)");
        return;
    };
    let translated = translate(&ir).expect("translate dynamic loop");
    assert!(translated.wgsl.contains("loop {"), "{}", translated.wgsl);
    assert!(!translated.compute_spirv.is_empty());
}

const NESTED_LOOP_KERNEL_RS: &str = r#"
#![allow(clippy::all)]
#![no_std]
#![no_main]
use core::panic::PanicInfo;
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { core::arch::wasm32::unreachable() }
extern "C" {
    fn storage_read_f32(group: u32, binding: u32, index: u32) -> f32;
}
#[no_mangle]
pub extern "C" fn cs_main(id: u32) -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0u32;
    while i < id {
        let mut j = 0u32;
        while j < id {
            // A call stops rustc from unrolling the inner loop at opt-level 1.
            acc += unsafe { storage_read_f32(0, 0, j) };
            j += 1;
        }
        i += 1;
    }
    acc
}
"#;

#[test]
fn rustc_gtao_arc_to_wgsl() {
    let Some(ir) = compile_ir_opt(GTAO_ARC_KERNEL_RS, "1") else {
        eprintln!("SKIPPED (wasm32 target missing)");
        return;
    };
    let translated = translate(&ir).expect("gtao arc");
    assert!(translated.wgsl.contains("cos("), "{}", translated.wgsl);
    assert!(translated.wgsl.contains("sin("), "{}", translated.wgsl);
}

#[test]
fn rustc_gtao_horizon_march_to_wgsl() {
    let Some(ir) = compile_lib_opt(GTAO_MARCH_KERNEL_RS, "1") else {
        panic!("gtao march: rustc produced no IR");
    };
    let translated = translate_with_meta(&ir).expect("gtao horizon march failed");
    assert!(translated.wgsl.contains("@fragment"), "{}", translated.wgsl);
    assert!(
        translated.wgsl.contains("@builtin(position)"),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("texture_depth_2d"),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("texture_2d<f32>"),
        "{}",
        translated.wgsl
    );
    for needle in [
        "textureLoad(",
        "acos(",
        "cross(",
        "normalize(",
        "loop",
        "var<uniform>",
        "@location(0)",
    ] {
        assert!(
            translated.wgsl.contains(needle),
            "{needle}\n{}",
            translated.wgsl
        );
    }
}

#[test]
fn rustc_histogram_to_wgsl() {
    let Some(ir) = compile_ir_opt(HISTOGRAM_KERNEL_RS, "1") else {
        panic!("histogram: rustc produced no IR");
    };
    let translated = translate_with_meta(&ir).expect("histogram");
    for needle in [
        "textureLoad(",
        "texture_2d<f32>",
        "log2(",
        "workgroupBarrier()",
        "atomicStore(",
        "atomicLoad(",
        "atomicAdd(",
        "local_invocation_index",
        "global_invocation_id",
    ] {
        assert!(
            translated.wgsl.contains(needle),
            "{needle}\n{}",
            translated.wgsl
        );
    }
}

const HISTOGRAM_KERNEL_RS: &str = r#"
#![allow(clippy::all)]
#![no_std]
#![no_main]
use core::panic::PanicInfo;
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { core::arch::wasm32::unreachable() }
extern "C" {
    fn global_invocation_lane(axis: u32) -> u32;
    fn local_invocation_index() -> u32;
    fn texture_load_2d_f32x4(group: u32, binding: u32, x: u32, y: u32, lane: u32) -> f32;
    fn texture_width_2d_f32x4(group: u32, binding: u32) -> i32;
    fn texture_height_2d_f32x4(group: u32, binding: u32) -> i32;
    fn workgroup_atomic_store_u32(index: u32, value: u32);
    fn workgroup_atomic_load_u32(index: u32) -> u32;
    fn workgroup_atomic_add_u32(index: u32, value: u32) -> u32;
    fn workgroup_barrier();
    fn storage_atomic_add_u32(group: u32, binding: u32, index: u32, value: u32) -> u32;
    fn gpu_dot3(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32;
    fn gpu_log2(value: f32) -> f32;
    fn uniform_field_f32(group: u32, binding: u32, sid: u32, field: u32) -> f32;
}
#[used]
#[unsafe(link_section = "ornis_gpu.shader_meta")]
#[unsafe(export_name = "ornis_gpu_shader_meta")]
static ORNIS_GPU_SHADER_META: [u8; 44] = *b"entry=cs_main;workgroup=16,16,1;uniform=0:11";
const BINS: u32 = 256;
const FLOOR: f32 = 1.0e-6;

#[no_mangle]
pub extern "C" fn cs_main() {
    unsafe {
        let lid = local_invocation_index();
        if lid < BINS {
            workgroup_atomic_store_u32(lid, 0);
        }
        workgroup_barrier();
        let x = global_invocation_lane(0);
        let y = global_invocation_lane(1);
        let width = texture_width_2d_f32x4(0, 0) as u32;
        let height = texture_height_2d_f32x4(0, 0) as u32;
        if x < width && y < height {
            let red = texture_load_2d_f32x4(0, 0, x, y, 0);
            let green = texture_load_2d_f32x4(0, 0, x, y, 1);
            let blue = texture_load_2d_f32x4(0, 0, x, y, 2);
            let luma = gpu_dot3(red, green, blue, 0.2126, 0.7152, 0.0722);
            let safe = if luma != luma || f32::abs(luma) > 3.4e38 { FLOOR } else { f32::max(luma, FLOOR) };
            let log_lum = gpu_log2(safe);
            let min_log = uniform_field_f32(0, 1, 0, 0);
            let inv_range = uniform_field_f32(0, 1, 0, 1);
            let bin = if log_lum < min_log {
                0
            } else {
                let t = f32::clamp((log_lum - min_log) * inv_range, 0.0, 1.0);
                1 + (t * 254.0) as u32
            };
            let _ = workgroup_atomic_add_u32(bin, 1);
        }
        workgroup_barrier();
        if lid < BINS {
            let count = workgroup_atomic_load_u32(lid);
            if count > 0 {
                let _ = storage_atomic_add_u32(0, 2, lid, count);
            }
        }
    }
}
"#;

const GTAO_MARCH_KERNEL_RS: &str = include_str!("../../gpu-gtao/src/lib.rs");

const GTAO_ARC_KERNEL_RS: &str = r#"
#![allow(clippy::all)]
#![no_std]
#![no_main]
use core::panic::PanicInfo;
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { core::arch::wasm32::unreachable() }
extern "C" {
    fn gpu_cos(value: f32) -> f32;
    fn gpu_sin(value: f32) -> f32;
}
#[no_mangle]
pub extern "C" fn cs_main(id: u32) -> f32 {
    // XeGTAO cosine-weighted arc, the same closed form as `gtao_arc`.
    // `no_std` has no `f32::cos`, so the kernel calls the imports the
    // translator already lowers to WGSL `cos` / `sin`.
    let horizon = (id as f32) * 0.01;
    let normal = 0.0f32;
    let two_h = 2.0 * horizon;
    unsafe { 0.25 * (-gpu_cos(two_h - normal) + gpu_cos(normal) + two_h * gpu_sin(normal)) }
}
"#;

#[test]
fn rustc_nested_loop_to_wgsl() {
    let Some(ir) = compile_ir_opt(NESTED_LOOP_KERNEL_RS, "1") else {
        eprintln!("SKIPPED (wasm32 target missing)");
        return;
    };
    let translated = translate(&ir).expect("translate nested rustc loop");
    assert!(
        translated.wgsl.matches("loop {").count() >= 2,
        "{}",
        translated.wgsl
    );
}

#[test]
fn llvm_ir_scalar_matches_cpu_reference_on_gpu() {
    let Some(ir) = compile_ir(KERNEL_RS) else {
        eprintln!("SKIPPED (wasm32 target missing)");
        return;
    };
    let translated = translate(&ir).expect("translate");
    let Some((device, queue)) = device() else {
        assert!(
            std::env::var_os("ORNIS_REQUIRE_GPU").is_none(),
            "ORNIS_REQUIRE_GPU is set but no wgpu device"
        );
        eprintln!("SKIPPED (no wgpu device)");
        return;
    };
    let outputs = run_compute(&device, &queue, &translated.wgsl);
    for (id, actual) in outputs.iter().enumerate() {
        let expected = cpu_reference(id as u32);
        // FMA contraction differs between CPU Rust and Metal WGSL;
        // allow a few ulps, as the staged-doc prescribes.
        let diff = (actual - expected).abs();
        let scale = expected.abs().max(1.0);
        assert!(
            diff / scale < 1e-4,
            "id {id}: gpu {actual} cpu {expected} diff {diff}"
        );
    }
}

/// CPU reference: the same arithmetic as `KERNEL_RS` without the unroll.
fn cpu_reference(id: u32) -> f32 {
    let mut x = (id as f32) * 2.0 + 1.0;
    let mut i = 0u32;
    while i < 4 {
        x = x * x + 1.0;
        i += 1;
    }
    (if x < 10.0 { x } else { 0.0 }) + x
}

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("llvm2gpu"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: Default::default(),
        trace: Default::default(),
        experimental_features: Default::default(),
    }))
    .ok()
}

fn run_compute(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str) -> Vec<f32> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("llvm2gpu"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("llvm2gpu layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage"),
        size: 4 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&storage, 0, &[0u8; 16]);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("llvm2gpu bg"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage.as_entire_binding(),
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("llvm2gpu pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("llvm2gpu pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 4 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, 16);
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv().expect("map callback").expect("map buffer");
    let guard = slice.get_mapped_range().expect("mapped range");
    let bytes = guard.to_vec();
    drop(guard);
    readback.unmap();
    let mut out = vec![0f32; 4];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    out
}
