//! A/B the lowered `cull` kernel against the shipped production WGSL.
//!
//! `CULL_COMPUTE_WGSL` (`render/src/gpu_culling.rs`) compacts
//! `PerObject` copies through `atomicAdd` and a `count` guard. This test
//! feeds it the same frustum and the same eight spheres the wasm2gpu
//! kernel reads — encoded as model matrices, so `instance_sphere` reads
//! the center from column 3 and the radius from the column lengths — and
//! compares the visible object sets the two shaders produce. Compaction
//! order is scheduler-dependent, so the contract is a sorted set of
//! `material_index` values plus the atomic count, not byte order.

use super::cull::ACTIVE_COUNT;
use super::cull::EXPECTED;
use super::cull::PLANES;
use super::cull::SPHERES;
use super::cull::buffer_entry;
use super::cull::lowered_kernel;
use super::cull::run;

/// The shipped cull kernel WGSL, bound by full path: this
/// foundation-layer test may not `use` items from `render` (the layer
/// gate flags the import edge), and the A/B needs the production shader.
const CULL_COMPUTE_WGSL: &str = ornis_render::gpu_culling::CULL_COMPUTE_WGSL;
/// Its params size, likewise bound by path instead of `use`.
const CULL_PARAMS_BYTE_SIZE: u64 = ornis_render::gpu_culling::CULL_PARAMS_BYTE_SIZE;
/// Word index of the atomic count inside the production `args` buffer.
const INSTANCE_COUNT_WORD: usize = ornis_render::gpu_culling::INSTANCE_COUNT_WORD;

/// Bytes one `PerObject` occupies in the production buffers: three
/// `mat4x4` (192) plus `material_index`, `motion_valid`, and two pads.
const OBJECT_BYTES: usize = 208;
/// Byte offset of `material_index` inside an object: the three matrices.
const MATERIAL_OFFSET: usize = 192;
/// `args` holds `array<atomic<u32>, 5>`; word 1 is the compacted count.
const ARGS_WORDS: usize = 5;

/// The translated kernel and the production kernel must cull the same
/// objects from the same bytes — the review's "A/B against the real
/// shader" instead of a CPU re-implementation of the formula.
#[test]
fn cull_kernel_matches_the_production_wgsl() {
    let wgsl = lowered_kernel();
    let expected: Vec<u32> = (0..ACTIVE_COUNT as u32)
        .filter(|slot| EXPECTED[*slot as usize] == 1)
        .collect();
    match super::device() {
        Some((device, queue)) => {
            let (flags, _) = run(&device, &queue, &wgsl);
            let wasm_visible: Vec<u32> = flags
                .iter()
                .enumerate()
                .filter(|(_, flag)| **flag == 1)
                .map(|(slot, _)| slot as u32)
                .collect();
            let (prod_visible, prod_count) = run_production(&device, &queue);
            assert_eq!(wasm_visible, expected, "wasm set vs EXPECTED");
            assert_eq!(prod_visible, expected, "production set vs EXPECTED");
            assert_eq!(prod_count as usize, expected.len(), "args count");
            eprintln!("cull A/B: production {prod_visible:?} == wasm2gpu {wasm_visible:?}");
        }
        None if std::env::var_os("ORNIS_REQUIRE_GPU").is_some() => {
            panic!("ORNIS_REQUIRE_GPU is set and no wgpu adapter was found");
        }
        None => eprintln!("cull_kernel_matches_the_production_wgsl: no wgpu adapter — skipped"),
    }
}

/// Dispatch the shipped kernel once and read back the compacted set:
/// `material_index` of every entry below the atomic count, sorted.
fn run_production(device: &wgpu::Device, queue: &wgpu::Queue) -> (Vec<u32>, u32) {
    let bgl = production_layout(device);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production cull"),
        source: wgpu::ShaderSource::Wgsl(CULL_COMPUTE_WGSL.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("production cull"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("production cull"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cull_instances"),
        compilation_options: Default::default(),
        cache: None,
    });
    let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let params = super::gpu_buffer(device, "prod-cull-params", CULL_PARAMS_BYTE_SIZE, uniform);
    let src = super::storage_buffer(
        device,
        "prod-cull-src",
        (SPHERES.len() * OBJECT_BYTES) as u64,
    );
    let dst = super::storage_buffer(
        device,
        "prod-cull-dst",
        (ACTIVE_COUNT * OBJECT_BYTES) as u64,
    );
    let args = super::storage_buffer(device, "prod-cull-args", (ARGS_WORDS * 4) as u64);
    queue.write_buffer(&params, 0, &pack_prod_params());
    queue.write_buffer(&src, 0, &pack_prod_objects());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("production cull"),
        layout: &bgl,
        entries: &[
            super::entry(0, &params),
            super::entry(1, &src),
            super::entry(2, &dst),
            super::entry(3, &args),
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("production cull"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("production cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        // One workgroup covers `count` lanes; repeats would accumulate
        // through `atomicAdd`, so the A/B dispatches exactly once.
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
    let count = read_u32(device, queue, &args, (INSTANCE_COUNT_WORD * 4) as u64);
    if count == 0 {
        return (Vec::new(), count);
    }
    let bytes = read_bytes(device, queue, &dst, count as u64 * OBJECT_BYTES as u64);
    let mut visible = Vec::with_capacity(count as usize);
    let mut entry = 0u32;
    while entry < count {
        let at = entry as usize * OBJECT_BYTES + MATERIAL_OFFSET;
        visible.push(u32::from_le_bytes([
            bytes[at],
            bytes[at + 1],
            bytes[at + 2],
            bytes[at + 3],
        ]));
        entry += 1;
    }
    visible.sort_unstable();
    (visible, count)
}

/// The production module's four bindings: the params uniform, the
/// read-only object source, the compacted destination, and the atomic
/// args word.
fn production_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mutable = wgpu::BufferBindingType::Storage { read_only: false };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("production cull"),
        entries: &[
            buffer_entry(0, wgpu::BufferBindingType::Uniform),
            buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            buffer_entry(2, mutable),
            buffer_entry(3, mutable),
        ],
    })
}

/// `CullParams`: the six planes as real `vec4`s (stride 16), then the
/// count word at its struct offset.
fn pack_prod_params() -> Vec<u8> {
    let mut bytes = vec![0u8; CULL_PARAMS_BYTE_SIZE as usize];
    let mut plane = 0usize;
    while plane < PLANES.len() {
        let base = plane as u32 * 16;
        let mut lane = 0u32;
        while lane < 4 {
            super::put_f32(&mut bytes, base + lane * 4, PLANES[plane][lane as usize]);
            lane += 1;
        }
        plane += 1;
    }
    super::put_u32(&mut bytes, (PLANES.len() * 16) as u32, ACTIVE_COUNT as u32);
    bytes
}

/// `PerObject[]` with each sphere on its model matrix: the radius on
/// column diagonals (so `length` of a column reads it back) and the
/// center as the translation column. `material_index` carries the slot
/// id the comparison keys on.
fn pack_prod_objects() -> Vec<u8> {
    let mut bytes = vec![0u8; SPHERES.len() * OBJECT_BYTES];
    let mut object = 0usize;
    while object < SPHERES.len() {
        let sphere = SPHERES[object];
        let base = (object * OBJECT_BYTES) as u32;
        super::put_f32(&mut bytes, base, sphere[3]);
        super::put_f32(&mut bytes, base + 20, sphere[3]);
        super::put_f32(&mut bytes, base + 40, sphere[3]);
        super::put_f32(&mut bytes, base + 48, sphere[0]);
        super::put_f32(&mut bytes, base + 52, sphere[1]);
        super::put_f32(&mut bytes, base + 56, sphere[2]);
        super::put_f32(&mut bytes, base + 60, 1.0);
        super::put_u32(&mut bytes, base + MATERIAL_OFFSET as u32, object as u32);
        object += 1;
    }
    bytes
}

fn read_u32(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, at: u64) -> u32 {
    let bytes = read_bytes(device, queue, src, src.size());
    let start = at as usize;
    u32::from_le_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ])
}

fn read_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size: u64,
) -> Vec<u8> {
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prod-cull-read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("prod-cull-read"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &read, 0, size);
    queue.submit([encoder.finish()]);
    super::map_bytes(device, &read)
}
