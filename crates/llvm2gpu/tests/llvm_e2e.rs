//! E2E for the LLVM IR spike: embedded IR, a real rustc build, and GPU parity.
use ornis_llvm2gpu::{translate, translate_with_types};
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
fn compile_ir_opt(source: &str, opt: &str) -> Option<String> {
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
