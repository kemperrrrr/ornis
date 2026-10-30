//! Dispatch the lowered `cs_main` and read the storage buffer back.
//!
//! The workgroup is four invocations wide. Each one writes
//! [`compute_value`] at its `global_invocation_id.x`. The host compares
//! those four floats. The line this test prints is how the CI log shows
//! that the GPU half ran.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;
use ornis_wasm2gpu_shader::compute_value;

const LANES: usize = symbols::CS_WORKGROUP[0] as usize;

#[test]
fn compute_kernel_writes_storage() {
    let wasm = super::compile_wasm();
    let lowered = translate(&super::compute_lift::lift(&wasm, "compute")).expect("wasm2gpu");
    let wgsl = &lowered.wgsl;
    assert!(wgsl.contains("switch "), "{wgsl}");
    assert!(wgsl.contains("case 0u"), "{wgsl}");
    assert!(wgsl.contains("default"), "{wgsl}");
    let size = symbols::CS_WORKGROUP;
    assert!(
        wgsl.contains(&format!(
            "@workgroup_size({}, {}, {})",
            size[0], size[1], size[2]
        )),
        "{wgsl}"
    );
    assert!(wgsl.contains("@builtin(global_invocation_id)"), "{wgsl}");
    assert!(wgsl.contains("@compute"), "{wgsl}");
    assert!(!lowered.compute_spirv.is_empty());
    match super::device() {
        Some((device, queue)) => {
            let got = dispatch(&device, &queue, wgsl);
            let mut index = 0usize;
            while index < LANES {
                let expect = compute_value(index as u32);
                assert!(
                    got[index].to_bits() == expect.to_bits(),
                    "lane {index}: got {got:?}, expected {expect}"
                );
                index += 1;
            }
            eprintln!("wasm2gpu compute GPU readback {got:?}");
        }
        None if std::env::var_os("ORNIS_REQUIRE_GPU").is_some() => {
            panic!("ORNIS_REQUIRE_GPU is set and no wgpu adapter was found");
        }
        None => eprintln!("compute_kernel_writes_storage: no wgpu adapter — skipped"),
    }
}

fn dispatch(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str) -> [f32; LANES] {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wasm2gpu compute"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wasm2gpu compute"),
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
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wasm2gpu compute"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wasm2gpu compute"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(symbols::CS_MAIN),
        compilation_options: Default::default(),
        cache: None,
    });
    let bytes = (LANES * 4) as u64;
    let storage = super::storage_buffer(device, "compute-out", bytes);
    queue.write_buffer(&storage, 0, &vec![0u8; bytes as usize]);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wasm2gpu compute"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wasm2gpu compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
    super::map_f32s(device, queue, &storage)
}
