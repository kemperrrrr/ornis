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

/// Signed `i32` path: `sub 0, id` then `ashr`/`sdiv`/`smin`/`sitofp`/`icmp slt`
/// must keep the two's-complement sign instead of reinterpreting as `u32`.
const SIGNED_IR: &str = r#"
declare i32 @llvm.smin.i32(i32, i32)
define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %neg = sub i32 0, %id
  %shift = ashr i32 %neg, 1
  %div = sdiv i32 %neg, 2
  %low = call i32 @llvm.smin.i32(i32 %neg, i32 0)
  %cmp = icmp slt i32 %neg, 0
  %flag = select i1 %cmp, float 1.000000e+00, float 0.000000e+00
  %shift_f = sitofp i32 %shift to float
  %div_f = sitofp i32 %div to float
  %low_f = sitofp i32 %low to float
  %a = fmul float %shift_f, 1.000000e+01
  %b = fmul float %div_f, 1.000000e+02
  %c = fmul float %low_f, 1.000000e+03
  %d = fadd float %a, %b
  %e = fadd float %d, %c
  %out = fadd float %e, %flag
  ret float %out
}
"#;

fn signed_reference(id: u32) -> f32 {
    let neg = 0i32.wrapping_sub(id as i32);
    let shift = neg >> 1;
    let div = neg / 2;
    let low = neg.min(0);
    let flag = f32::from(u8::from(neg < 0));
    shift as f32 * 10.0 + div as f32 * 100.0 + low as f32 * 1000.0 + flag
}

#[test]
fn signed_integer_ops_match_the_cpu_on_gpu() {
    let translated = translate(SIGNED_IR).expect("signed ops lower");
    assert!(
        translated.wgsl.contains("bitcast<i32>"),
        "signed ops must bitcast to i32:\n{}",
        translated.wgsl
    );
    let Some(outputs) = gpu_outputs(&translated) else {
        return;
    };
    for (id, actual) in outputs.iter().enumerate() {
        let expected = signed_reference(id as u32);
        assert!(
            (actual - expected).abs() < 1e-3,
            "id {id}: gpu {actual} cpu {expected}"
        );
    }
}

/// Build runtime operands for the four-lane harness. Exceptional integer
/// divisions are legal WGSL at runtime but rejected as constant expressions.
fn integer_div_rem_ir(op: &str, numerators: [i32; 4], divisors: [i32; 4]) -> String {
    let mut ir = String::from("define float @cs_main(i32 %id) {\nstart:\n");
    ir.push_str(&format!(
        "  %a0 = add i32 %id, {}\n  %b0 = add i32 %id, {}\n",
        numerators[0], divisors[0]
    ));
    // Override every lane, including lane 0, so operands remain runtime SSA
    // values without relying on arithmetic to synthesize zero or INT_MIN.
    let mut a = String::from("%a0");
    let mut b = String::from("%b0");
    for lane in 0..4 {
        ir.push_str(&format!(
            "  %lane{lane} = icmp eq i32 %id, {lane}\n  %a{lane}_selected = select i1 %lane{lane}, i32 {}, i32 {a}\n  %b{lane}_selected = select i1 %lane{lane}, i32 {}, i32 {b}\n",
            numerators[lane], divisors[lane]
        ));
        a = format!("%a{lane}_selected");
        b = format!("%b{lane}_selected");
    }
    let cast = if op.starts_with('s') {
        "sitofp"
    } else {
        "uitofp"
    };
    ir.push_str(&format!(
        "  %result = {op} i32 {a}, {b}\n  %out = {cast} i32 %result to float\n  ret float %out\n}}\n"
    ));
    ir
}

/// LLVM leaves zero divisors and signed division overflow undefined. The
/// translator deliberately emits WGSL operators: quotient is the numerator
/// for zero/overflow, remainder is zero, not Rust's panicking `/` or `%`.
#[test]
fn signed_div_rem_corner_cases_match_wgsl_on_gpu() {
    for (numerators, divisors) in [
        ([i32::MIN, -7, 0, 7], [-1, 0, 0, 0]),
        ([i32::MIN, -7, 7, -7], [1, 2, -2, -2]),
    ] {
        for op in ["sdiv", "srem"] {
            let translated = translate(&integer_div_rem_ir(op, numerators, divisors))
                .expect("signed corner cases lower to WGSL");
            assert!(
                translated.wgsl.contains("bitcast<i32>"),
                "{}",
                translated.wgsl
            );
            let operator = if op == "sdiv" { " / " } else { " % " };
            assert!(translated.wgsl.contains(operator), "{}", translated.wgsl);
            let Some(outputs) = gpu_outputs(&translated) else {
                return;
            };
            for (lane, actual) in outputs.iter().enumerate() {
                let (a, b) = (numerators[lane], divisors[lane]);
                let expected = if op == "sdiv" {
                    a.checked_div(b).unwrap_or(a)
                } else {
                    a.checked_rem(b).unwrap_or(0)
                } as f32;
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "lane {lane}: {op} {a}, {b}: GPU {actual}, WGSL {expected}"
                );
            }
        }
    }
}

#[test]
fn unsigned_div_rem_by_zero_matches_wgsl_on_gpu() {
    let numerators = [0, 1, 7, 65535];
    for op in ["udiv", "urem"] {
        let translated = translate(&integer_div_rem_ir(op, numerators, [0; 4]))
            .expect("unsigned zero divisors lower to WGSL");
        let Some(outputs) = gpu_outputs(&translated) else {
            return;
        };
        for (lane, actual) in outputs.iter().enumerate() {
            let expected = if op == "udiv" { numerators[lane] } else { 0 } as f32;
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "lane {lane}: {op} by zero: GPU {actual}, WGSL {expected}"
            );
        }
    }
}

/// Encode nine comparison results into one bit mask (bits 0..8).
fn compare_mask_ir(operands: &str) -> String {
    let mut ir = String::from(
        "define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {\nstart:\n",
    );
    ir.push_str(operands);
    const PREDS: [&str; 9] = [
        "ueq", "une", "ult", "ule", "ugt", "uge", "one", "ord", "uno",
    ];
    let mut accum: Option<String> = None;
    for (bit, pred) in PREDS.iter().enumerate() {
        let name = format!("%{pred}");
        ir.push_str(&format!(
            "  {name} = fcmp {pred} float %a, %b\n  %{pred}_i = zext i1 {name} to i32\n"
        ));
        accum = Some(match accum {
            None => format!("%{pred}_i"),
            Some(prev) => {
                let shifted = format!("%{pred}_s");
                ir.push_str(&format!(
                    "  {shifted} = shl i32 %{pred}_i, {bit}\n  %{pred}_or = or i32 {prev}, {shifted}\n"
                ));
                format!("%{pred}_or")
            }
        });
    }
    let mask = accum.expect("at least one predicate");
    ir.push_str(&format!(
        "  %mask_f = uitofp i32 {mask} to float\n  ret float %mask_f\n}}\n"
    ));
    ir
}

/// `fcmp` masks follow IEEE: on NaN every unordered predicate is true, so
/// bits 0,1,2,3,4,5,8 set (1+2+4+8+16+32+256 = 319) while `one`/`ord` stay
/// false; on finite `1 < 2` only `une`, `ult`, `ule`, `one`, `ord` hold
/// (2+4+8+64+128 = 206).
#[test]
fn float_compare_masks_follow_ieee_on_gpu() {
    // NaN is derived at runtime: a literal `0f / 0f` is constant-folded into
    // a rejected `nan` literal by naga's WGSL front end.
    let nan_operands = "  %x = uitofp i32 %id to float\n  %zero = fsub float %x, %x\n  %a = fdiv float %zero, %zero\n  %b = fadd float 1.000000e+00, %zero\n";
    let finite_operands = "  %a = fadd float 1.000000e+00, 0.000000e+00\n  %b = fadd float 2.000000e+00, 0.000000e+00\n";
    for (label, operands, expected) in [
        ("NaN", nan_operands, 319.0),
        ("ordered", finite_operands, 206.0),
    ] {
        let translated = translate(&compare_mask_ir(operands)).expect("fcmp comparisons lower");
        // naga lowers float `!=` to `OpFOrdNotEqual` (false for NaN), so the
        // mask must be built from ordered comparisons on every backend.
        assert!(
            !translated.wgsl.contains(" != "),
            "{label} compares must not use float `!=`:\n{}",
            translated.wgsl
        );
        let Some(outputs) = gpu_outputs(&translated) else {
            return;
        };
        for (id, actual) in outputs.iter().enumerate() {
            assert!(
                (actual - expected).abs() < 1e-3,
                "id {id}: {label} mask {actual}"
            );
        }
    }
}

#[test]
fn barrier_inside_a_branch_is_rejected() {
    let ir = r#"
declare void @workgroup_barrier()
define dso_local void @cs_main(i32 noundef %id) {
start:
  %lt = icmp ult i32 %id, 3
  br i1 %lt, label %inside, label %outside
inside:
  call void @workgroup_barrier()
  br label %merge
outside:
  br label %merge
merge:
  ret void
}
"#;
    assert!(
        translate(ir).is_err(),
        "a barrier only some invocations reach must be rejected"
    );
}

#[test]
fn barrier_after_a_nonuniform_return_is_rejected() {
    for (true_label, false_label) in [("early", "merge"), ("merge", "early")] {
        let ir = format!(
            r#"
declare void @workgroup_barrier()
define void @cs_main(i32 %id) {{
start:
  %lt = icmp ult i32 %id, 3
  br i1 %lt, label %{true_label}, label %{false_label}
early:
  ret void
merge:
  call void @workgroup_barrier()
  ret void
}}
"#
        );
        let without_barrier = ir.replace("  call void @workgroup_barrier()", "");
        translate(&without_barrier).expect("the early-return CFG itself is supported");
        assert!(
            matches!(
                translate(&ir),
                Err(ornis_llvm2gpu::TranslateError::Unsupported)
            ),
            "a top-level barrier skipped by a returning arm must be rejected"
        );
    }
}

#[test]
fn barrier_inside_a_loop_is_rejected() {
    let ir = r#"
declare void @workgroup_barrier()
define dso_local void @cs_main(i32 noundef %id) {
start:
  br label %header
header:
  %i = phi i32 [ 0, %start ], [ %next, %latch ]
  %next = add i32 %i, 1
  call void @workgroup_barrier()
  %done = icmp ugt i32 %next, %id
  br i1 %done, label %exit, label %latch
latch:
  br label %header
exit:
  ret void
}
"#;
    assert!(
        translate(ir).is_err(),
        "a barrier in a loop is rejected until trip-count uniformity is proved"
    );
}

#[test]
fn barrier_after_a_reconverged_diamond_is_allowed() {
    let ir = r#"
declare void @workgroup_barrier()
define dso_local void @cs_main(i32 noundef %id) {
start:
  %lt = icmp ult i32 %id, 3
  br i1 %lt, label %left, label %right
left:
  br label %merge
right:
  br label %merge
merge:
  call void @workgroup_barrier()
  ret void
}
"#;
    let translated = translate(ir).expect("a top-level barrier after reconvergence lowers");
    assert!(
        translated.wgsl.contains("workgroupBarrier()"),
        "{}",
        translated.wgsl
    );
}

/// Stage interface of the first entry point of `wgsl`.
///
/// The assertions read the parsed module, not the printed text: naga's writer
/// renames identifiers (a trailing digit grows an underscore), so matching
/// strings would pin the writer instead of the interface.
fn stage_interface(wgsl: &str) -> StageInterface {
    use naga::TypeInner;
    let module = naga::front::wgsl::parse_str(wgsl).expect("emitted WGSL parses");
    let entry = module.entry_points.first().expect("one entry point");
    let args = entry
        .function
        .arguments
        .iter()
        .map(|argument| {
            let lanes = match &module.types[argument.ty].inner {
                TypeInner::Vector { size, .. } => *size as u32,
                _ => 1,
            };
            (argument.binding.clone().expect("bound argument"), lanes)
        })
        .collect();
    StageInterface {
        stage: entry.stage,
        inputs: args,
        outputs: result_members(&module, &entry.function),
    }
}

/// Result members of `function`: binding, lane count, and byte offset.
fn result_members(
    module: &naga::Module,
    function: &naga::Function,
) -> Vec<(naga::Binding, (u32, u32))> {
    use naga::TypeInner;
    let Some(result) = &function.result else {
        return Vec::new();
    };
    let TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
        return Vec::new();
    };
    members
        .iter()
        .map(|member| {
            let binding = member.binding.clone().expect("bound output");
            let lanes = match &module.types[member.ty].inner {
                TypeInner::Vector { size, .. } => *size as u32,
                _ => 1,
            };
            (binding, (lanes, member.offset))
        })
        .collect()
}

/// Stage interface of one entry point.
struct StageInterface {
    /// Stage of the entry point.
    stage: naga::ShaderStage,
    /// Entry arguments: binding and declared lane count.
    inputs: Vec<(naga::Binding, u32)>,
    /// Result members: binding, lane count, and byte offset.
    outputs: Vec<(naga::Binding, (u32, u32))>,
}

/// Whether `binding` is `@location(location)`.
fn at_location(binding: &naga::Binding, location: u32) -> bool {
    matches!(binding, naga::Binding::Location { location: at, .. } if *at == location)
}

/// Whether `binding` is the named builtin.
fn at_builtin(binding: &naga::Binding, built_in: naga::BuiltIn) -> bool {
    matches!(binding, naga::Binding::BuiltIn(found) if *found == built_in)
}

#[test]
fn vertex_entry_declares_position_builtins_and_locations() {
    let ir = r#"
declare i32 @vertex_index()
declare i32 @instance_index()
declare float @vertex_input_lane(i32, i32, i32)
declare void @vertex_position(float, float, float, float)
declare void @vertex_store_f32x2(i32, float, float)
define dso_local void @vs_main() {
start:
  %vi = call i32 @vertex_index()
  %ii = call i32 @instance_index()
  %ux = call float @vertex_input_lane(i32 0, i32 2, i32 0)
  %uy = call float @vertex_input_lane(i32 0, i32 2, i32 1)
  %vif = uitofp i32 %vi to float
  %iif = uitofp i32 %ii to float
  %x = fadd float %vif, %iif
  call void @vertex_position(float %x, float %ux, float %uy, float 1.000000e+00)
  call void @vertex_store_f32x2(i32 0, float %ux, float %uy)
  ret void
}
"#;
    let translated = translate(ir).expect("vertex entry");
    let wgsl = &translated.wgsl;
    assert!(wgsl.contains("@vertex"), "{wgsl}");
    let interface = stage_interface(wgsl);
    assert_eq!(interface.stage, naga::ShaderStage::Vertex);
    let args = &interface.inputs;
    let outputs = &interface.outputs;
    assert_eq!(args.len(), 3, "{wgsl}");
    assert!(at_builtin(&args[0].0, naga::BuiltIn::VertexIndex), "{wgsl}");
    assert!(
        at_builtin(&args[1].0, naga::BuiltIn::InstanceIndex),
        "{wgsl}"
    );
    assert!(at_location(&args[2].0, 0), "{wgsl}");
    assert_eq!(args[2].1, 2, "{wgsl}");
    // A vertex result must carry the clip position, and the two `vec2` lanes
    // the body wrote become one located `vec2` member.
    assert_eq!(outputs.len(), 2, "{wgsl}");
    assert!(
        at_builtin(&outputs[0].0, naga::BuiltIn::Position { invariant: false }),
        "{wgsl}"
    );
    assert!(at_location(&outputs[1].0, 0), "{wgsl}");
    assert_eq!(outputs[1].1.0, 2, "{wgsl}");
}

#[test]
fn fragment_entry_infers_interface_widths_from_the_lanes_used() {
    let ir = r#"
declare float @fragment_input_lane(i32, i32, i32)
declare void @fragment_store_location(i32, float, float, float, float)
define dso_local void @fs_main() {
start:
  %a = call float @fragment_input_lane(i32 0, i32 3, i32 0)
  %b = call float @fragment_input_lane(i32 0, i32 3, i32 1)
  %c = call float @fragment_input_lane(i32 0, i32 3, i32 2)
  call void @fragment_store_location(i32 0, float %a, float %b, float %c, float 1.000000e+00)
  call void @fragment_store_location(i32 1, float %b, float %a, float %c, float 1.000000e+00)
  ret void
}
"#;
    let translated = translate(ir).expect("fragment entry");
    let wgsl = &translated.wgsl;
    // Two targets become an output struct instead of the bare location-0 vec4.
    assert!(wgsl.contains("-> FragmentOut"), "{wgsl}");
    let interface = stage_interface(wgsl);
    assert_eq!(interface.stage, naga::ShaderStage::Fragment);
    let args = &interface.inputs;
    let outputs = &interface.outputs;
    // The fragment's `@builtin(position)` is always declared, then the inputs.
    assert_eq!(args.len(), 2, "{wgsl}");
    assert!(at_builtin(
        &args[0].0,
        naga::BuiltIn::Position { invariant: false }
    ));
    assert!(at_location(&args[1].0, 0), "{wgsl}");
    assert_eq!(args[1].1, 3, "{wgsl}");
    assert_eq!(outputs.len(), 2, "{wgsl}");
    assert!(at_location(&outputs[0].0, 0), "{wgsl}");
    assert!(at_location(&outputs[1].0, 1), "{wgsl}");
    assert_eq!(outputs[0].1.0, 4, "{wgsl}");
    assert!(outputs[1].1.1 >= outputs[0].1.1 + 16, "{wgsl}");
}

#[test]
fn one_location_cannot_have_two_widths() {
    let ir = r#"
declare float @vertex_input_lane(i32, i32, i32)
declare void @vertex_position(float, float, float, float)
define dso_local void @vs_main() {
start:
  %ux = call float @vertex_input_lane(i32 0, i32 2, i32 0)
  %uz = call float @vertex_input_lane(i32 0, i32 2)
  call void @vertex_position(float %ux, float %uz, float 0.000000e+00, float 1.000000e+00)
  ret void
}
"#;
    // Location 0 is declared two-wide and three-wide by the same body; a shader
    // that disagrees with itself is rejected instead of padded.
    assert!(translate(ir).is_err());
}

#[test]
fn vertex_entry_rejects_fragment_io() {
    let ir = r#"
declare void @fragment_store_f32x4(float, float, float, float)
define dso_local void @vs_main() {
start:
  call void @fragment_store_f32x4(float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00)
  ret void
}
"#;
    assert!(translate(ir).is_err());
}

#[test]
fn vertex_entry_without_position_is_rejected() {
    let ir = r#"
declare void @vertex_store_f32x2(i32, float, float)
define dso_local void @vs_main() {
start:
  call void @vertex_store_f32x2(i32 0, float 1.000000e+00, float 0.000000e+00)
  ret void
}
"#;
    // A vertex result must carry `@builtin(position)`; guessing it would emit
    // a module naga refuses.
    assert!(translate(ir).is_err());
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
        .pop()
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
fn dot2_lowers_to_vec2_dot() {
    let ir = r#"
define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %_4 = uitofp i32 %id to float
  %d = call float @gpu_dot2_xy(float %_4, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00)
  ret float %d
}
"#;
    let translated = translate(ir).expect("dot2");
    assert!(
        translated.wgsl.contains("dot(vec2"),
        "vec2 dot must stay a builtin, not a scalar sum: {}",
        translated.wgsl
    );
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
/// Compile a workspace shader crate the way `xtask build-shaders` does.
///
/// The kernel depends on `ornis-gpu-shader`, so a lone `rustc` of its source
/// cannot see the SDK. The IR goes to an explicit, unique path: the emit path
/// is part of cargo's fingerprint, so rustc always runs and always writes it,
/// and the test never has to guess which `.ll` in `deps` is current.
fn compile_package(package: &str) -> String {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace root");
    let ll = std::env::temp_dir().join(format!("llvm2gpu-e2e-{package}-{}.ll", std::process::id()));
    let ll = unique_ir_path(&ll);
    cargo_rustc_llvm_ir(root, package, &ll);
    assert!(
        ll.exists(),
        "cargo produced no LLVM IR for {package} at {}",
        ll.display()
    );
    let text = std::fs::read_to_string(&ll)
        .unwrap_or_else(|error| panic!("read {}: {error}", ll.display()));
    let _ = std::fs::remove_file(&ll);
    text
}

/// Adds a per-run suffix so cargo cannot reuse an earlier build of this crate.
fn unique_ir_path(base: &std::path::Path) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    let stem = base
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("kernel");
    base.with_file_name(format!("{stem}-{nanos}.ll"))
}

/// Runs `cargo rustc` and asks rustc to emit LLVM IR at `ll`.
fn cargo_rustc_llvm_ir(root: &std::path::Path, package: &str, ll: &std::path::Path) {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .current_dir(root)
        .env("CARGO_BUILD_RUSTC_WRAPPER", "/usr/bin/env")
        .args([
            "rustc",
            "-p",
            package,
            "--target",
            "wasm32-unknown-unknown",
            "--release",
            "--",
            "-C",
            "opt-level=1",
            "-C",
            "panic=abort",
            "-C",
            "target-cpu=mvp",
            "-C",
            "codegen-units=1",
        ])
        .arg(format!("--emit=llvm-ir={}", ll.display()))
        .status()
        .unwrap_or_else(|error| panic!("spawn cargo rustc -p {package}: {error}"));
    assert!(status.success(), "cargo rustc -p {package} failed");
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

/// One crate, two entry points: a vertex and a fragment stage of one pass.
const RASTER_PASS_RS: &str = r#"
#![no_std]
#![no_main]
use core::panic::PanicInfo;
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { core::arch::wasm32::unreachable() }
extern "C" {
    fn vertex_index() -> u32;
    fn vertex_input_lane(location: u32, lanes: u32, lane: u32) -> f32;
    fn vertex_position(x: f32, y: f32, z: f32, w: f32);
    fn vertex_store_f32x2(location: u32, x: f32, y: f32);
    fn fragment_input_lane(location: u32, lanes: u32, lane: u32) -> f32;
    fn fragment_store_f32x4(r: f32, g: f32, b: f32, a: f32);
}
#[no_mangle]
pub extern "C" fn vs_main() {
    let index = unsafe { vertex_index() };
    let x = unsafe { vertex_input_lane(0, 2, 0) };
    let y = unsafe { vertex_input_lane(0, 2, 1) };
    unsafe {
        vertex_position(index as f32, x, y, 1.0);
        vertex_store_f32x2(0, x, y);
    }
}
#[no_mangle]
pub extern "C" fn fs_main() {
    let u = unsafe { fragment_input_lane(0, 2, 0) };
    let v = unsafe { fragment_input_lane(0, 2, 1) };
    unsafe { fragment_store_f32x4(u, v, 0.0, 1.0) };
}
"#;

/// Both entries of one crate lower into one module and share its bindings.
#[test]
fn rustc_two_raster_entries_lower_into_one_module() {
    let Some(ir) = compile_ir(RASTER_PASS_RS) else {
        eprintln!("SKIPPED (wasm32 target missing)");
        return;
    };
    let translated = translate(&ir).expect("translate both entries");
    // rustc emits the two entries in its own order; both have to be there.
    let mut entries = translated.entries.clone();
    entries.sort();
    assert_eq!(entries, ["fs_main", "vs_main"]);
    assert!(translated.wgsl.contains("@vertex"), "{}", translated.wgsl);
    assert!(
        translated.wgsl.contains("fn fs_main"),
        "{}",
        translated.wgsl
    );
    // One translation per entry still works on the same IR.
    let only = ornis_llvm2gpu::translate_entry_with_sampling(
        &ir,
        "fs_main",
        ornis_llvm2gpu::TextureSampling::SingleSample,
    )
    .expect("fragment only");
    assert_eq!(only.entries, ["fs_main"]);
    assert!(!only.wgsl.contains("@vertex"), "{}", only.wgsl);
    assert!(
        ornis_llvm2gpu::translate_entry_with_sampling(
            &ir,
            "cs_main",
            ornis_llvm2gpu::TextureSampling::SingleSample,
        )
        .is_err(),
        "a missing entry is an error, not an empty module"
    );
}

/// A `switch`-terminated block: rustc emits one for `match` on integers.
const SWITCH_IR: &str = r#"
define dso_local float @cs_main(i32 %id) {
start:
  switch i32 %id, label %other [ i32 0, label %zero i32 1, label %one i32 2, label %zero ]
zero:
  br label %merge
one:
  br label %merge
other:
  br label %merge
merge:
  %value = phi float [ 1.000000e+00, %zero ], [ 2.000000e+00, %one ], [ 3.000000e+00, %other ]
  ret float %value
}
"#;

#[test]
fn switch_lowers_to_a_naga_switch() {
    let translated = translate(SWITCH_IR).expect("switch entry");
    let wgsl = &translated.wgsl;
    assert!(wgsl.contains("switch "), "{wgsl}");
    assert!(wgsl.contains("default"), "{wgsl}");
    // The two values that share a target share one body: naga prints them as
    // one case label list.
    assert!(
        wgsl.contains("case 0u, 2u") || wgsl.contains("case 0, 2"),
        "{wgsl}"
    );
    assert!(wgsl.contains("case 1"), "{wgsl}");
}

#[test]
fn switch_lowers_through_rustc_match() {
    const MATCH_RS: &str = r#"
#![no_std]
#![no_main]
use core::panic::PanicInfo;
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { core::arch::wasm32::unreachable() }
#[no_mangle]
pub extern "C" fn cs_main(id: u32) -> f32 {
    // Sparse values keep rustc on a `switch` instead of a lookup table.
    match id {
        0 => 1.0,
        10 => 2.0,
        20 | 30 => 4.0,
        _ => 0.5,
    }
}
"#;
    let Some(ir) = compile_ir_opt(MATCH_RS, "s") else {
        eprintln!("SKIPPED (wasm32 target missing)");
        return;
    };
    let translated = translate(&ir).expect("match lowers");
    assert!(translated.wgsl.contains("switch "), "{}", translated.wgsl);
    assert!(translated.wgsl.contains("20u, 30u"), "{}", translated.wgsl);
}

const LOOP_KERNEL_RS: &str = r#"
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
    let ir = compile_package("ornis-gpu-gtao");
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
        "dot(vec2",
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
    let ir = compile_package("ornis-gpu-histogram");
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

#[test]
fn shader_kernels_do_not_declare_extern_imports() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for relative in [
        "../gpu-histogram/src/lib.rs",
        "../gpu-adapt/src/lib.rs",
        "../gpu-gtao/src/lib.rs",
    ] {
        let source = std::fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            !source.contains("extern \"C\" {"),
            "{relative} declares an import; add the operation to ornis-gpu-shader instead"
        );
    }
}

#[test]
fn rustc_adapt_to_wgsl() {
    let ir = compile_package("ornis-gpu-adapt");
    let translated = translate_with_meta(&ir).expect("adapt");
    for needle in [
        "exp2(",
        "log2(",
        "array<u32, 256>",
        "var<workgroup>",
        "@workgroup_size(1, 1, 1)",
        "loop {",
    ] {
        assert!(
            translated.wgsl.contains(needle),
            "{needle}\n{}",
            translated.wgsl
        );
    }
}

const GTAO_ARC_KERNEL_RS: &str = r#"
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

/// Screen quad from `vertex_index` alone: positions and UVs are `select`s.
const QUAD_VERTEX_IR: &str = r#"
declare i32 @vertex_index()
declare void @vertex_position(float, float, float, float)
declare void @vertex_store_f32x2(i32, float, float)
define dso_local void @vs_main() {
start:
  %vi = call i32 @vertex_index()
  %right = icmp ugt i32 %vi, 1
  %is_one = icmp eq i32 %vi, 1
  %is_three = icmp eq i32 %vi, 3
  %top_or_bottom = select i1 %is_one, float 1.000000e+00, float -1.000000e+00
  %y = select i1 %is_three, float 1.000000e+00, float %top_or_bottom
  %x = select i1 %right, float 1.000000e+00, float -1.000000e+00
  %u = select i1 %right, float 1.000000e+00, float 0.000000e+00
  %odd_v = select i1 %is_one, float 0.000000e+00, float 1.000000e+00
  %v = select i1 %is_three, float 0.000000e+00, float %odd_v
  call void @vertex_position(float %x, float %y, float 0.000000e+00, float 1.000000e+00)
  call void @vertex_store_f32x2(i32 0, float %u, float %v)
  ret void
}
"#;

/// Fragment stage that writes the interpolated UV into the color target.
const UV_FRAGMENT_IR: &str = r#"
declare float @fragment_input_lane(i32, i32, i32)
declare void @fragment_store_location(i32, float, float, float, float)
define dso_local void @fs_main() {
start:
  %u = call float @fragment_input_lane(i32 0, i32 2, i32 0)
  %v = call float @fragment_input_lane(i32 0, i32 2, i32 1)
  call void @fragment_store_location(i32 0, float %u, float %v, float 0.000000e+00, float 1.000000e+00)
  ret void
}
"#;

/// Both raster stages translated from IR, then drawn as one pipeline.
///
/// The vertex module alone is not proof: the location the vertex stage writes
/// and the one the fragment stage reads have to agree, and the clip position
/// has to land where `@builtin(vertex_index)` said it would. Reading the UV
/// back from the color target pins all three at once.
#[test]
fn translated_vertex_and_fragment_stages_draw_the_screen_uv() {
    let vs = translate(QUAD_VERTEX_IR).expect("vertex stage");
    let fs = translate(UV_FRAGMENT_IR).expect("fragment stage");
    // Two independent modules: the vertex one declares no globals, so the two
    // WGSL texts concatenate into one module with both entry points.
    let wgsl = format!("{}\n{}", vs.wgsl, fs.wgsl);
    let Some((device, queue)) = device() else {
        assert!(
            std::env::var_os("ORNIS_REQUIRE_GPU").is_none(),
            "ORNIS_REQUIRE_GPU is set but no wgpu device"
        );
        eprintln!("SKIPPED (no wgpu device)");
        return;
    };
    const SIZE: u32 = 8;
    let pixels = render_quad(&device, &queue, &wgsl, SIZE);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let index = ((y * SIZE + x) * 4) as usize;
            let expected_u = ((x as f32 + 0.5) / SIZE as f32 * 255.0).round() as u8;
            let expected_v = ((y as f32 + 0.5) / SIZE as f32 * 255.0).round() as u8;
            assert_eq!(
                (pixels[index], pixels[index + 1], pixels[index + 2]),
                (expected_u, expected_v, 0),
                "pixel ({x}, {y})"
            );
        }
    }
}

/// Draw a `TriangleStrip` quad from `wgsl` into an `Rgba8Unorm` target.
fn render_quad(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str, size: u32) -> Vec<u8> {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("llvm2gpu raster"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("llvm2gpu raster"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("llvm2gpu target"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("quad"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..4, 0..1);
    }
    read_back(device, queue, encoder, &target, size)
}

/// Copy the rendered target into host memory, one unpadded RGBA row per line.
fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    target: &wgpu::Texture,
    size: u32,
) -> Vec<u8> {
    /// `copy_texture_to_buffer` row alignment.
    const ROW_ALIGN: u32 = 256;
    let unpadded = size * 4;
    let padded = unpadded.div_ceil(ROW_ALIGN) * ROW_ALIGN;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(padded * size),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let data = map_bytes(device, &readback);
    let mut pixels = vec![0u8; (unpadded * size) as usize];
    for row in 0..size as usize {
        pixels[row * unpadded as usize..][..unpadded as usize]
            .copy_from_slice(&data[row * padded as usize..][..unpadded as usize]);
    }
    pixels
}

/// Map `buffer` for reading and return its contents.
fn map_bytes(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv().expect("map callback").expect("map buffer");
    let data = slice.get_mapped_range().expect("mapped range");
    data.to_vec()
}

#[test]
fn llvm_ir_scalar_matches_cpu_reference_on_gpu() {
    let Some(ir) = compile_ir(KERNEL_RS) else {
        eprintln!("SKIPPED (wasm32 target missing)");
        return;
    };
    let translated = translate(&ir).expect("translate");
    let Some(outputs) = gpu_outputs(&translated) else {
        return;
    };
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

/// Run a translated compute entry and return the four `storage_0_0` lanes.
///
/// `None` when the host has no adapter, unless `ORNIS_REQUIRE_GPU` turns the
/// skip into a failure (the CI GPU shard sets it).
fn gpu_outputs(translated: &ornis_llvm2gpu::Translated) -> Option<Vec<f32>> {
    let Some((device, queue)) = device() else {
        assert!(
            std::env::var_os("ORNIS_REQUIRE_GPU").is_none(),
            "ORNIS_REQUIRE_GPU is set but no wgpu device"
        );
        eprintln!("SKIPPED (no wgpu device)");
        return None;
    };
    Some(run_compute(&device, &queue, &translated.wgsl))
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
    let bytes = map_bytes(device, &readback);
    let mut out = vec![0f32; 4];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    out
}
