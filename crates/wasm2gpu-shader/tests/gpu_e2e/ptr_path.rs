//! Real `rustc` output through the `GpuRef` pointer path: `addr` builds
//! `storage_base + offset`, `load` / `store` are one scalar each.
//!
//! Unlike `linear_memory.rs` (hand-written WAT), this export is the wasm32
//! build of `ornis-wasm2gpu-shader`: provenance crosses rustc's locals and
//! `i32.add`, and the GPU buffer shows what the translated chain wrote.

use std::time::Instant;

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;

/// Source values: four lanes the kernel doubles and offsets by one.
const SRC: [f32; 4] = [1.5, -2.25, 0.0, 4.0];

#[test]
fn gpu_ref_pointer_path_writes_storage() {
    let wasm = super::compile_wasm();
    let wat = super::compute_lift::lift_wat(&wasm, "ptr_copy");
    let module = match wat::parse_str(&wat) {
        Ok(module) => module,
        Err(err) => panic!("ptr wat: {err}\n{wat}"),
    };
    let lowered = match translate(&module) {
        Ok(lowered) => lowered,
        Err(err) => panic!("ptr translate {err:?}\nWAT:\n{wat}"),
    };
    let wgsl = &lowered.wgsl;
    let dynamic = wgsl.contains(">>");
    assert!(wgsl.contains("storage_0_1_"), "{wgsl}");
    assert!(wgsl.contains("storage_0_2_"), "{wgsl}");
    assert!(!lowered.compute_spirv.is_empty());
    assert!(dynamic, "runtime index lost its shift: {wgsl}");
    match super::device() {
        Some((device, queue)) => {
            let values = dispatch(&device, &queue, wgsl);
            let mut slot = 0usize;
            while slot < SRC.len() {
                let want = SRC[slot] + SRC[slot] + 1.0;
                assert_eq!(values[slot].to_bits(), want.to_bits(), "slot {slot}: {values:?}");
                slot += 1;
            }
        }
        None if std::env::var_os("ORNIS_REQUIRE_GPU").is_some() => {
            panic!("ORNIS_REQUIRE_GPU is set and no wgpu adapter was found");
        }
        None => eprintln!("gpu_ref_pointer_path_writes_storage: no wgpu adapter — skipped"),
    }
}

/// Lowered `ptr_copy`: storage `(0, 1)` in, storage `(0, 2)` out, one
/// workgroup of `symbols::CS_WORKGROUP` lanes.
fn dispatch(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str) -> [f32; 4] {
    let src = super::storage_buffer(device, "ptr-src", 16);
    let out = super::storage_buffer(device, "ptr-out", 16);
    queue.write_buffer(&src, 0, &src_bytes());
    queue.write_buffer(&out, 0, &[0u8; 16]);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wasm2gpu ptr"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wasm2gpu ptr"),
        entries: &[storage_slot(1), storage_slot(2)],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wasm2gpu ptr"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wasm2gpu ptr"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(symbols::CS_MAIN),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wasm2gpu ptr"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 1,
                resource: src.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wasm2gpu ptr"),
    });
    let start = Instant::now();
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wasm2gpu ptr"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
    let values = super::map_f32s(device, queue, &out);
    let elapsed = start.elapsed();
    eprintln!("wasm2gpu GpuRef pointer path {values:?} in {elapsed:?}");
    values
}

fn storage_slot(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn src_bytes() -> [u8; 16] {
    let mut bytes = [0u8; 16];
    let mut slot = 0usize;
    while slot < SRC.len() {
        super::put_f32(&mut bytes, (slot as u32) * 4, SRC[slot]);
        slot += 1;
    }
    bytes
}
