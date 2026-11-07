//! E2E for the LLVM IR spike: embedded IR, a real rustc build, and GPU parity.
use ornis_llvm2gpu::translate;
use std::path::PathBuf;
use std::process::Command;

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
fn loops_are_rejected_not_mistranslated() {
    let phi_ir = r#"
define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {
start:
  %x = phi float [ 0.000000e+00, %start ]
  br label %start
  ret float %x
}
"#;
    assert!(translate(phi_ir).is_err());
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
            "opt-level=3",
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
