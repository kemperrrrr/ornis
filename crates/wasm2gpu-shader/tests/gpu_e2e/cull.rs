//! Dispatch the lowered `cull` kernel and A/B it against a hand twin of
//! the production `sphere_visible` test.
//!
//! The kernel flags four spheres against six planes. Sphere 3 sits past
//! the guarded count and keeps the sentinel, which is how the test sees
//! that `id >= count` does no work. Both shaders run on the same bytes:
//! the outputs must match bitwise, and the line this test prints shows
//! both GPU times from the CI log.

use std::time::Duration;
use std::time::Instant;

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;
use ornis_wasm2gpu_shader::COUNT_BINDING;
use ornis_wasm2gpu_shader::FLAG_BINDING;
use ornis_wasm2gpu_shader::PLANE_BINDING;
use ornis_wasm2gpu_shader::PLANE_COUNT;
use ornis_wasm2gpu_shader::PLANE_LEN;
use ornis_wasm2gpu_shader::SPHERE_BINDING;

const LANES: usize = symbols::CS_WORKGROUP[0] as usize;
/// Sphere 3 is past `count` and must keep this value.
const SENTINEL: u32 = 7;
const RUNS: u32 = 64;

/// Sphere centers and radii: four lanes each, packed storage order.
const SPHERES: [[f32; 4]; 4] = [
    [0.5, 0.0, 0.0, 0.25],
    [-1.0, 0.0, 0.0, 0.25],
    [2.0, 0.0, 0.0, 0.1],
    [9.0, 9.0, 9.0, 1.0],
];

/// Visible, culled, visible, untouched (past `count`).
const EXPECTED: [u32; 4] = [1, 0, 1, SENTINEL];

/// The hand twin runs the same guard, the same `dot + w + radius < 0.0`
/// order, and the same workgroup, so bitwise agreement is the contract.
const TWIN_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> planes: array<f32, 24>;
@group(0) @binding(1) var<storage, read_write> spheres: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> sphere_count: array<u32, 4>;

@compute @workgroup_size(4, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id < sphere_count[0]) {
        let center = spheres[id];
        var visible = 1u;
        var plane = 0u;
        while (plane < 6u) {
            let at = plane * 4u;
            let px = planes[at];
            let py = planes[at + 1u];
            let pz = planes[at + 2u];
            let pw = planes[at + 3u];
            if (px * center.x + py * center.y + pz * center.z + pw + center.w < 0.0) {
                visible = 0u;
            }
            plane += 1u;
        }
        flags[id] = visible;
    }
}
"#;

#[test]
fn cull_kernel_matches_the_hand_twin() {
    assert_eq!(LANES, SPHERES.len());
    let wasm = super::compile_wasm();
    let lowered = translate(&super::compute_lift::lift(&wasm, "cull")).expect("wasm2gpu");
    let wgsl = &lowered.wgsl;
    assert!(wgsl.contains("@compute"), "{wgsl}");
    assert!(wgsl.contains("@builtin(global_invocation_id)"), "{wgsl}");
    assert!(
        wgsl.contains(&format!("storage_0_{FLAG_BINDING}_")),
        "{wgsl}"
    );
    assert!(!lowered.compute_spirv.is_empty());
    match super::device() {
        Some((device, queue)) => {
            let (flags, wasm_time) = run(&device, &queue, wgsl);
            assert_eq!(flags, EXPECTED, "wasm2gpu kernel output");
            let (twin, twin_time) = run(&device, &queue, TWIN_WGSL);
            assert_eq!(twin, EXPECTED, "hand twin output");
            assert_eq!(flags, twin, "bitwise A/B");
            eprintln!("cull {flags:?}: wasm {wasm_time:?} twin {twin_time:?} x{RUNS}");
        }
        None if std::env::var_os("ORNIS_REQUIRE_GPU").is_some() => {
            panic!("ORNIS_REQUIRE_GPU is set and no wgpu adapter was found");
        }
        None => eprintln!("cull_kernel_matches_the_hand_twin: no wgpu adapter — skipped"),
    }
}

fn run(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str) -> ([u32; LANES], Duration) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wasm2gpu cull"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = bind_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wasm2gpu cull"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wasm2gpu cull"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(symbols::CS_MAIN),
        compilation_options: Default::default(),
        cache: None,
    });
    let (flags, bind) = upload(device, queue, &bgl);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wasm2gpu cull"),
    });
    let start = Instant::now();
    for _ in 0..RUNS {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wasm2gpu cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let time = start.elapsed();
    (map_u32s(device, queue, &flags), time)
}

fn bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = wgpu::BufferBindingType::Storage { read_only: false };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wasm2gpu cull"),
        entries: &[
            buffer_entry(PLANE_BINDING, wgpu::BufferBindingType::Uniform),
            buffer_entry(SPHERE_BINDING, storage),
            buffer_entry(FLAG_BINDING, storage),
            buffer_entry(COUNT_BINDING, storage),
        ],
    })
}

fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Upload the inputs and return the flags buffer: the bind group keeps
/// the other buffers alive for as long as it lives.
fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let planes = super::gpu_buffer(device, "cull-planes", 384, uniform);
    let spheres = super::storage_buffer(device, "cull-spheres", 64);
    let flags = super::storage_buffer(device, "cull-flags", 16);
    let count = super::storage_buffer(device, "cull-count", 16);
    queue.write_buffer(&planes, 0, &pack_planes());
    queue.write_buffer(&spheres, 0, &pack_spheres());
    queue.write_buffer(&flags, 0, &[0u8; 16]);
    queue.write_buffer(&count, 0, &pack_count());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wasm2gpu cull"),
        layout,
        entries: &[
            super::entry(PLANE_BINDING, &planes),
            super::entry(SPHERE_BINDING, &spheres),
            super::entry(FLAG_BINDING, &flags),
            super::entry(COUNT_BINDING, &count),
        ],
    });
    (flags, bind)
}

/// Plane components in uniform slots: one `f32` per 16-byte stride.
/// Plane 0 points along `+x` through the origin; the rest pass everything.
fn pack_planes() -> Vec<u8> {
    let mut bytes = vec![0u8; (PLANE_LEN * 16) as usize];
    super::put_f32(&mut bytes, 0, 1.0);
    let mut plane = 1u32;
    while plane < PLANE_COUNT {
        super::put_f32(&mut bytes, (plane * 4 + 3) * 16, 1.0);
        plane += 1;
    }
    bytes
}

fn pack_spheres() -> Vec<u8> {
    let mut bytes = vec![0u8; SPHERES.len() * 16];
    let mut sphere = 0usize;
    while sphere < SPHERES.len() {
        let mut lane = 0usize;
        while lane < 4 {
            let at = (sphere * 4 + lane) as u32 * 4;
            super::put_f32(&mut bytes, at, SPHERES[sphere][lane]);
            lane += 1;
        }
        sphere += 1;
    }
    bytes
}

fn pack_count() -> Vec<u8> {
    let mut bytes = vec![0u8; 16];
    super::put_u32(&mut bytes, 0, SPHERES.len() as u32 - 1);
    bytes
}

fn map_u32s(device: &wgpu::Device, queue: &wgpu::Queue, storage: &wgpu::Buffer) -> [u32; LANES] {
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cull-flags-read"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cull-flags-read"),
    });
    encoder.copy_buffer_to_buffer(storage, 0, &read, 0, 16);
    queue.submit([encoder.finish()]);
    let bytes = super::map_bytes(device, &read);
    let mut values = [0u32; LANES];
    for (slot, word) in bytes.chunks_exact(4).take(LANES).enumerate() {
        values[slot] = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
    }
    values
}
