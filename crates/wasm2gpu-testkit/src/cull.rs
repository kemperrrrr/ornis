//! The `cull` guest's GPU inputs and dispatcher: sphere/plane tables,
//! expected flags, packing, and the timed `run` loop.
//!
//! Shared by the hand-twin test in `wasm2gpu-shader` and by `render`'s
//! production A/B (`cull_prod`): the tables are plain bytes, the kernel
//! comes from the guest, and neither side needs a `foundation → domain`
//! import.

use std::time::Duration;
use std::time::Instant;

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu_shader::COUNT_BINDING;
use ornis_wasm2gpu_shader::FLAG_BINDING;
use ornis_wasm2gpu_shader::PLANE_BINDING;
use ornis_wasm2gpu_shader::PLANE_COUNT;
use ornis_wasm2gpu_shader::PLANE_LEN;
use ornis_wasm2gpu_shader::SPHERE_BINDING;

use crate::entry;
use crate::gpu_buffer;
use crate::map_bytes;
use crate::put_f32;
use crate::put_u32;
use crate::storage_buffer;

/// Workgroup size of the guest's `cs_main`: one sphere lane per thread.
pub const LANES: usize = symbols::CS_WORKGROUP[0] as usize;
/// Workgroups the test dispatches: ids run `0..LANES * WORKGROUPS`.
pub const WORKGROUPS: u32 = 2;
/// An untouched slot must keep this value.
pub const SENTINEL: u32 = 7;
/// Repeats for the timing line: the writes are idempotent, the elapsed
/// time across all of them is what the test prints.
pub const RUNS: u32 = 64;

/// Sphere centers and radii: four lanes each, packed storage order.
/// Slots 6 and 7 sit past [`ACTIVE_COUNT`] and are never read.
pub const SPHERES: [[f32; 4]; 8] = [
    // A clear pass against plane 0 (+x).
    [0.5, 0.0, 0.0, 0.25],
    // A clear cut by plane 0.
    [-1.0, 0.0, 0.0, 0.25],
    // A clear pass.
    [2.0, 0.0, 0.0, 0.1],
    // Boundary: `dot + w + radius == 0.0`, and the production
    // `dot + w < -radius` is `false` on the same value — visible.
    [-0.25, 0.0, 0.0, 0.25],
    // NaN radius: IEEE `<` fails open in the kernel and in the WGSL twin.
    [0.0, 0.0, 0.0, f32::NAN],
    // A clear cut by plane 3 (+y), not by plane 0.
    [0.0, -0.5, 0.0, 0.25],
    [9.0, 9.0, 9.0, 1.0],
    [9.0, 9.0, 9.0, 1.0],
];

/// Slots that do work: `id < count`; the rest must keep [`SENTINEL`].
pub const ACTIVE_COUNT: usize = 6;

/// Flag slots one dispatch covers: every sphere lane.
pub const SLOTS: usize = SPHERES.len();

/// Visible, culled, visible, boundary pass, NaN pass, cut by plane 3,
/// then two untouched slots past [`ACTIVE_COUNT`].
pub const EXPECTED: [u32; SPHERES.len()] = [1, 0, 1, 1, 1, 0, SENTINEL, SENTINEL];

/// The six clip planes in uniform slots: plane 0 is `+x`, plane 3 `+y`,
/// the rest pass every sphere the test feeds.
pub const PLANES: [[f32; 4]; PLANE_COUNT as usize] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Lift the `cull` guest export through [`crate::compute_lift`]'s shared
/// pinning helper: the compute entry, the id builtin, the flag binding,
/// and a non-empty SPIR-V twin. The hand-twin test and `render`'s
/// `cull_prod` A/B both pin the same lowered WGSL through it.
pub fn lowered_kernel() -> String {
    crate::compute_lift::lowered_kernel("cull", &format!("storage_0_{FLAG_BINDING}_"))
}

/// Dispatch `wgsl` at `cs_main` for [`RUNS`] repeats and read the flags
/// back; the repeats feed the timing line (idempotent writes).
pub fn run(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str) -> ([u32; SLOTS], Duration) {
    let bgl = bind_layout(device);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wasm2gpu cull"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
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
        pass.dispatch_workgroups(WORKGROUPS, 1, 1);
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

/// One compute-visible buffer binding at `binding`.
pub fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
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
    let planes = gpu_buffer(device, "cull-planes", 384, uniform);
    let spheres = storage_buffer(device, "cull-spheres", 128);
    let flags = storage_buffer(device, "cull-flags", 32);
    let count = storage_buffer(device, "cull-count", 16);
    queue.write_buffer(&planes, 0, &pack_planes());
    queue.write_buffer(&spheres, 0, &pack_spheres());
    queue.write_buffer(&flags, 0, &pack_flags());
    queue.write_buffer(&count, 0, &pack_count());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wasm2gpu cull"),
        layout,
        entries: &[
            entry(PLANE_BINDING, &planes),
            entry(SPHERE_BINDING, &spheres),
            entry(FLAG_BINDING, &flags),
            entry(COUNT_BINDING, &count),
        ],
    });
    (flags, bind)
}

/// Plane components in uniform slots: one `f32` per 16-byte stride.
/// Plane 0 points along `+x`, plane 3 along `+y`; the rest pass everything.
fn pack_planes() -> Vec<u8> {
    let mut bytes = vec![0u8; (PLANE_LEN * 16) as usize];
    let mut plane = 0u32;
    while plane < PLANE_COUNT {
        let mut lane = 0u32;
        while lane < 4 {
            put_f32(
                &mut bytes,
                (plane * 4 + lane) * 16,
                PLANES[plane as usize][lane as usize],
            );
            lane += 1;
        }
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
            put_f32(&mut bytes, at, SPHERES[sphere][lane]);
            lane += 1;
        }
        sphere += 1;
    }
    bytes
}

/// Every flag starts at [`SENTINEL`]: an untouched slot keeps it, which
/// is how the test sees that the count guard wrote nothing.
fn pack_flags() -> [u8; 32] {
    let mut bytes = [0u8; 32];
    let mut slot = 0usize;
    while slot < SLOTS {
        put_u32(&mut bytes, (slot as u32) * 4, SENTINEL);
        slot += 1;
    }
    bytes
}

fn pack_count() -> Vec<u8> {
    let mut bytes = vec![0u8; 16];
    put_u32(&mut bytes, 0, ACTIVE_COUNT as u32);
    bytes
}

fn map_u32s(device: &wgpu::Device, queue: &wgpu::Queue, storage: &wgpu::Buffer) -> [u32; SLOTS] {
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cull-flags-read"),
        size: 32,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cull-flags-read"),
    });
    encoder.copy_buffer_to_buffer(storage, 0, &read, 0, 32);
    queue.submit([encoder.finish()]);
    let bytes = map_bytes(device, &read);
    let mut values = [0u32; SLOTS];
    for (slot, word) in bytes.chunks_exact(4).take(SLOTS).enumerate() {
        values[slot] = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
    }
    values
}
