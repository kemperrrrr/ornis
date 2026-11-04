//! Same-input parity for the wasm2gpu cull-compaction guest and shipped WGSL.

use ornis_wasm2gpu_testkit::compute_lift::lowered_kernel;
use ornis_wasm2gpu_testkit::cull::buffer_entry;
use ornis_wasm2gpu_testkit::symbols;
use ornis_wasm2gpu_testkit::{device, entry, gpu_buffer, map_bytes, put_f32, put_u32};

const COUNT_ELEMENTS: usize = 3;
const UNIFORM_SLOT_BYTES: usize = 16;
const PLANE_LANES: usize = 4;
const U32_BYTES: usize = std::mem::size_of::<u32>();
const RECORD_WORDS: usize = 52;
const RECORD_BYTES: usize = RECORD_WORDS * U32_BYTES;
const WORKGROUP_SIZE: u32 = 64;
const INDEX_COUNT: u32 = 36;
const ARGS_WORDS: usize = 5;
const PLANE_BINDING: u32 = 0;
const SOURCE_FLOAT_BINDING: u32 = 1;
const DESTINATION_BINDING: u32 = 2;
const ARGS_BINDING: u32 = 3;
const SOURCE_WORDS_BINDING: u32 = 4;
const COUNTS_BINDING: u32 = 5;

struct CompactBuffers {
    source: wgpu::Buffer,
    source_words: wgpu::Buffer,
    destination: wgpu::Buffer,
    args: wgpu::Buffer,
    planes: wgpu::Buffer,
    counts: wgpu::Buffer,
}

#[test]
fn cull_compact_matches_the_production_wgsl() {
    let Some((device, queue)) = device() else {
        assert!(
            std::env::var_os("ORNIS_REQUIRE_GPU").is_none(),
            "ORNIS_REQUIRE_GPU is set and no wgpu adapter was found"
        );
        eprintln!("cull_compact_matches_the_production_wgsl: no wgpu adapter — skipped");
        return;
    };
    assert_eq!(RECORD_BYTES, super::OBJECT_BYTES);

    let objects = super::pack_prod_objects();
    let (production_visible, production_count) = super::run_production(&device, &queue, &objects);
    let (compact_visible, compact_count) = run_compact(&device, &queue, &objects);
    let expected: Vec<u32> = (0..super::ACTIVE_COUNT as u32)
        .filter(|slot| super::EXPECTED[*slot as usize] == 1)
        .collect();

    assert_eq!(production_visible, expected, "production WGSL vs expected");
    assert_eq!(compact_visible, expected, "wasm2gpu compaction vs expected");
    assert_eq!(compact_visible, production_visible, "same-input GPU parity");
    assert_eq!(production_count as usize, expected.len());
    assert_eq!(compact_count, production_count);
}

fn run_compact(device: &wgpu::Device, queue: &wgpu::Queue, objects: &[u8]) -> (Vec<u32>, u32) {
    let buffers = create_compact_buffers(device, queue, objects);
    let layout = compact_layout(device);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wasm2gpu cull compact bindings"),
        layout: &layout,
        entries: &[
            entry(PLANE_BINDING, &buffers.planes),
            entry(SOURCE_FLOAT_BINDING, &buffers.source),
            entry(DESTINATION_BINDING, &buffers.destination),
            entry(ARGS_BINDING, &buffers.args),
            entry(SOURCE_WORDS_BINDING, &buffers.source_words),
            entry(COUNTS_BINDING, &buffers.counts),
        ],
    });
    let pipeline = compact_pipeline(device, &layout);
    let source_size = (super::SPHERES.len() * super::OBJECT_BYTES) as u64;
    let readback_size = source_size + (ARGS_WORDS * U32_BYTES) as u64;
    let readback = gpu_buffer(
        device,
        "compact-readback",
        readback_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wasm2gpu cull compact"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wasm2gpu cull compact"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            super::ACTIVE_COUNT.div_ceil(WORKGROUP_SIZE as usize) as u32,
            1,
            1,
        );
    }
    encoder.copy_buffer_to_buffer(&buffers.destination, 0, &readback, 0, source_size);
    encoder.copy_buffer_to_buffer(
        &buffers.args,
        0,
        &readback,
        source_size,
        (ARGS_WORDS * std::mem::size_of::<u32>()) as u64,
    );
    queue.submit([encoder.finish()]);
    let bytes = map_bytes(device, &readback);
    let args_offset = source_size as usize;
    let count = read_u32(&bytes, args_offset + U32_BYTES);
    let mut visible = (0..count as usize)
        .map(|slot| read_u32(&bytes, slot * RECORD_BYTES + super::MATERIAL_OFFSET))
        .collect::<Vec<_>>();
    visible.sort_unstable();
    (visible, count)
}

fn create_compact_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    objects: &[u8],
) -> CompactBuffers {
    let source_size = (super::SPHERES.len() * super::OBJECT_BYTES) as u64;
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let source = gpu_buffer(device, "compact-source-f32", source_size, storage);
    let source_words = gpu_buffer(device, "compact-source-u32", source_size, storage);
    let destination = gpu_buffer(
        device,
        "compact-destination",
        source_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let args_size = (ARGS_WORDS * U32_BYTES) as u64;
    let args = gpu_buffer(
        device,
        "compact-indirect-args",
        args_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let plane_bytes = super::PLANES.len() * PLANE_LANES * UNIFORM_SLOT_BYTES;
    let planes = gpu_buffer(
        device,
        "compact-planes",
        plane_bytes as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let count_bytes = COUNT_ELEMENTS * UNIFORM_SLOT_BYTES;
    let counts = gpu_buffer(
        device,
        "compact-counts",
        count_bytes as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let mut initial_args = vec![0; args_size as usize];
    put_u32(&mut initial_args, 0, INDEX_COUNT);
    queue.write_buffer(&source, 0, objects);
    queue.write_buffer(&source_words, 0, objects);
    queue.write_buffer(&destination, 0, &vec![0; source_size as usize]);
    queue.write_buffer(&args, 0, &initial_args);
    queue.write_buffer(&planes, 0, &pack_planes());
    queue.write_buffer(&counts, 0, &pack_counts());
    CompactBuffers {
        source,
        source_words,
        destination,
        args,
        planes,
        counts,
    }
}

fn compact_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wasm2gpu cull compact layout"),
        entries: &[
            buffer_entry(PLANE_BINDING, wgpu::BufferBindingType::Uniform),
            buffer_entry(
                SOURCE_FLOAT_BINDING,
                wgpu::BufferBindingType::Storage { read_only: false },
            ),
            buffer_entry(
                DESTINATION_BINDING,
                wgpu::BufferBindingType::Storage { read_only: false },
            ),
            buffer_entry(
                ARGS_BINDING,
                wgpu::BufferBindingType::Storage { read_only: false },
            ),
            buffer_entry(
                SOURCE_WORDS_BINDING,
                wgpu::BufferBindingType::Storage { read_only: false },
            ),
            buffer_entry(COUNTS_BINDING, wgpu::BufferBindingType::Uniform),
        ],
    })
}

// qual:allow(dry, duplicate) reason: "This parity harness independently builds the same lowered guest entry as its e2e test, using a production-record input layout."
fn compact_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let wgsl = lowered_kernel("cull_compact", "storage_0_2_");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wasm2gpu cull compact"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wasm2gpu cull compact"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wasm2gpu cull compact"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(symbols::CS_MAIN),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn pack_planes() -> Vec<u8> {
    let mut bytes = vec![0; super::PLANES.len() * PLANE_LANES * UNIFORM_SLOT_BYTES];
    for (plane_index, plane) in super::PLANES.iter().enumerate() {
        for (lane, value) in plane.iter().enumerate() {
            let offset = (plane_index * PLANE_LANES + lane) * UNIFORM_SLOT_BYTES;
            put_f32(&mut bytes, offset as u32, *value);
        }
    }
    bytes
}

fn pack_counts() -> Vec<u8> {
    let values = [
        super::ACTIVE_COUNT as f32,
        super::SPHERES.len() as f32,
        super::ACTIVE_COUNT as f32,
    ];
    let mut bytes = vec![0; COUNT_ELEMENTS * UNIFORM_SLOT_BYTES];
    for (index, value) in values.into_iter().enumerate() {
        put_f32(&mut bytes, (index * UNIFORM_SLOT_BYTES) as u32, value);
    }
    bytes
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + U32_BYTES]
            .try_into()
            .expect("u32 word"),
    )
}
