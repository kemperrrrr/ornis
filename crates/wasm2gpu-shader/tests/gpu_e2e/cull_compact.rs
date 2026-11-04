//! GPU parity test for the wasm2gpu frustum-cull-and-compact kernel.

use ornis_wasm2gpu_testkit::compute_lift::lowered_kernel;
use ornis_wasm2gpu_testkit::device;
use ornis_wasm2gpu_testkit::entry;
use ornis_wasm2gpu_testkit::gpu_buffer;
use ornis_wasm2gpu_testkit::map_bytes;
use ornis_wasm2gpu_testkit::symbols;

const RECORD_WORDS: usize = 52;
const RECORD_BYTES: usize = RECORD_WORDS * std::mem::size_of::<u32>();
const U32_BYTES: usize = std::mem::size_of::<u32>();
const MATERIAL_WORD: usize = 48;
const MOTION_VALID_WORD: usize = 49;
const MATERIAL_BASE: u32 = 10;
const MATERIAL_NEXT: u32 = 11;
const MATERIAL_LAST_VISIBLE: u32 = 12;
const MATERIAL_INACTIVE: u32 = 13;
const MATERIAL_NAN_RADIUS: u32 = 14;
const MATERIAL_NAN_AXIS: u32 = 15;
const ACTIVE_COUNT: u32 = 5;
const INDEX_COUNT: u32 = 36;
const SOURCE_COUNT: usize = 65;
const COUNT_UNIFORM_ELEMENTS: usize = 3;
const EXPECTED_WORKGROUP_SIZE: u32 = 64;
const DESTINATION_CANARY: u32 = 0xdead_beef;
const PLANES: usize = 6;
const LANES: usize = 4;
const UNIFORM_LANE_BYTES: usize = 16;
const PLANES_BYTES: usize = PLANES * LANES * UNIFORM_LANE_BYTES;
const COUNT_UNIFORM_BYTES: usize = COUNT_UNIFORM_ELEMENTS * UNIFORM_LANE_BYTES;
const ARGS_WORDS: usize = 5;
const MODEL_X_AXIS_WORD: usize = 0;
const MODEL_Y_AXIS_WORD: usize = 5;
const MODEL_Z_AXIS_WORD: usize = 10;
const MODEL_CENTER_X_WORD: usize = 12;
const MODEL_CENTER_W_WORD: usize = 15;
const NORMAL_MATRIX_START_WORD: usize = 16;
const EXTRA_PAD_0_WORD: usize = 50;
const EXTRA_PAD_1_WORD: usize = 51;
const VISIBLE_CENTER_NEAR: f32 = 0.5;
const VISIBLE_CENTER_FAR: f32 = 2.0;
const CULLED_CENTER: f32 = -1.0;
const VISIBLE_RADIUS: f32 = 0.25;
const CULL_TEST_BOUNDARY_CENTER: f32 = -VISIBLE_RADIUS;
const FAR_VISIBLE_RADIUS: f32 = 0.1;
const OPEN_PLANE_DISTANCE: f32 = 100.0;

struct CullCompactGpu {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    destination: wgpu::Buffer,
    args: wgpu::Buffer,
}

#[derive(Clone, Copy)]
struct CullCounts {
    active: u32,
    source: u32,
    destination: u32,
}

struct CullCompactInputs<'a> {
    source: &'a wgpu::Buffer,
    source_words: &'a wgpu::Buffer,
    args: &'a wgpu::Buffer,
    planes: &'a wgpu::Buffer,
    count: &'a wgpu::Buffer,
}

#[test]
fn cull_compact_preserves_visible_records_and_updates_indirect_count() {
    let Some((device, queue)) = device() else {
        assert!(
            !super::gpu_required(),
            "ORNIS_REQUIRE_GPU is set but cull-compaction test has no wgpu device"
        );
        eprintln!("SKIPPED (wasm2gpu cull compaction): no wgpu device");
        return;
    };

    let records = test_records();
    let counts = CullCounts {
        active: ACTIVE_COUNT,
        source: SOURCE_COUNT as u32,
        destination: SOURCE_COUNT as u32,
    };
    let gpu = create_gpu_case(&device, &queue, &records, counts);
    let bytes = dispatch_and_readback(&device, &queue, &gpu);
    assert_compacted_output(&bytes, &records);
}

#[test]
fn cull_compact_guards_source_and_destination_capacity() {
    let Some((device, queue)) = device() else {
        assert!(
            !super::gpu_required(),
            "ORNIS_REQUIRE_GPU is set but cull-compaction capacity test has no wgpu device"
        );
        eprintln!("SKIPPED (wasm2gpu cull compaction capacity): no wgpu device");
        return;
    };

    let mut records = test_records();
    records[1] = object_record(MATERIAL_NEXT, 1, VISIBLE_CENTER_FAR, FAR_VISIBLE_RADIUS);
    records[2] = object_record(
        MATERIAL_LAST_VISIBLE,
        1,
        VISIBLE_CENTER_NEAR,
        VISIBLE_RADIUS,
    );
    records[SOURCE_COUNT - 1] =
        object_record(MATERIAL_INACTIVE, 1, VISIBLE_CENTER_FAR, FAR_VISIBLE_RADIUS);
    let counts = CullCounts {
        active: SOURCE_COUNT as u32,
        source: 2,
        destination: 1,
    };
    let gpu = create_gpu_case(&device, &queue, &records, counts);
    let bytes = dispatch_and_readback(&device, &queue, &gpu);
    assert_capacity_guards(&bytes, &records);
}

fn test_records() -> [[u32; RECORD_WORDS]; SOURCE_COUNT] {
    let mut records =
        [object_record(MATERIAL_INACTIVE, 0, CULLED_CENTER, VISIBLE_RADIUS); SOURCE_COUNT];
    records[0] = object_record(MATERIAL_BASE, 1, VISIBLE_CENTER_NEAR, VISIBLE_RADIUS);
    records[1] = object_record(MATERIAL_NEXT, 0, CULLED_CENTER, VISIBLE_RADIUS);
    records[2] = object_record(
        MATERIAL_LAST_VISIBLE,
        1,
        CULL_TEST_BOUNDARY_CENTER,
        VISIBLE_RADIUS,
    );
    records[3] = object_record(MATERIAL_NAN_RADIUS, 1, CULLED_CENTER, f32::NAN);
    records[4] = object_record_with_scales(
        MATERIAL_NAN_AXIS,
        1,
        CULLED_CENTER,
        [f32::NAN, VISIBLE_RADIUS, VISIBLE_RADIUS],
    );
    records
}

fn create_gpu_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    records: &[[u32; RECORD_WORDS]; SOURCE_COUNT],
    counts: CullCounts,
) -> CullCompactGpu {
    let source = gpu_buffer(
        device,
        "wasm2gpu cull compact source floats",
        (SOURCE_COUNT * RECORD_BYTES) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    // The current lowering declares storage imports as read_write. Keep
    // the u32 copy view separate from the f32 culling view to avoid binding
    // the same writable buffer range twice.
    let source_words = gpu_buffer(
        device,
        "wasm2gpu cull compact source words",
        (SOURCE_COUNT * RECORD_BYTES) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let destination = gpu_buffer(
        device,
        "wasm2gpu cull compact destination",
        (SOURCE_COUNT * RECORD_BYTES) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let args = gpu_buffer(
        device,
        "wasm2gpu cull compact args",
        (ARGS_WORDS * U32_BYTES) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let planes = gpu_buffer(
        device,
        "wasm2gpu cull compact planes",
        PLANES_BYTES as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let count = gpu_buffer(
        device,
        "wasm2gpu cull compact count",
        COUNT_UNIFORM_BYTES as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let inputs = CullCompactInputs {
        source: &source,
        source_words: &source_words,
        args: &args,
        planes: &planes,
        count: &count,
    };
    queue.write_buffer(
        &destination,
        0,
        &words_to_bytes(&vec![DESTINATION_CANARY; SOURCE_COUNT * RECORD_WORDS]),
    );
    upload_inputs(queue, records, &inputs, counts);

    let layout = create_bind_group_layout(device);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wasm2gpu cull compact bindings"),
        layout: &layout,
        entries: &[
            entry(0, &planes),
            entry(1, &source),
            entry(2, &destination),
            entry(3, &args),
            entry(4, &source_words),
            entry(5, &count),
        ],
    });
    let pipeline = create_pipeline(device, &layout);

    CullCompactGpu {
        pipeline,
        bind_group,
        destination,
        args,
    }
}

fn upload_inputs(
    queue: &wgpu::Queue,
    records: &[[u32; RECORD_WORDS]; SOURCE_COUNT],
    inputs: &CullCompactInputs<'_>,
    counts: CullCounts,
) {
    let source_bytes = records_to_bytes(records);
    queue.write_buffer(inputs.source, 0, &source_bytes);
    queue.write_buffer(inputs.source_words, 0, &source_bytes);
    queue.write_buffer(inputs.args, 0, &words_to_bytes(&[INDEX_COUNT, 0, 0, 0, 0]));
    queue.write_buffer(inputs.planes, 0, &packed_planes());
    queue.write_buffer(inputs.count, 0, &packed_counts(counts));
}

fn packed_counts(counts: CullCounts) -> Vec<u8> {
    let values = [counts.active, counts.source, counts.destination];
    let mut bytes = vec![0; COUNT_UNIFORM_BYTES];
    for (index, value) in values.into_iter().enumerate() {
        let offset = index * UNIFORM_LANE_BYTES;
        bytes[offset..offset + U32_BYTES].copy_from_slice(&(value as f32).to_le_bytes());
    }
    bytes
}

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wasm2gpu cull compact layout"),
        entries: &[
            buffer_layout_entry(0, wgpu::BufferBindingType::Uniform),
            buffer_layout_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout_entry(5, wgpu::BufferBindingType::Uniform),
        ],
    })
}

// qual:allow(dry, fragment) reason: "The compute pipeline repeats the guest shader module, pipeline layout, and entry-point wiring required by this standalone GPU test."
// qual:allow(dry, duplicate) reason: "The standalone shader test and render parity test both construct lowered compute pipelines with separate GPU contexts."
fn create_pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
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

fn dispatch_and_readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &CullCompactGpu,
) -> Vec<u8> {
    let workgroup_size = EXPECTED_WORKGROUP_SIZE as usize;
    let readback = gpu_buffer(
        device,
        "wasm2gpu cull compact readback",
        (SOURCE_COUNT * RECORD_BYTES + ARGS_WORDS * U32_BYTES) as u64,
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
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        pass.dispatch_workgroups(SOURCE_COUNT.div_ceil(workgroup_size) as u32, 1, 1);
    }
    copy_outputs(&mut encoder, gpu, &readback);
    queue.submit([encoder.finish()]);
    map_bytes(device, &readback)
}

fn copy_outputs(encoder: &mut wgpu::CommandEncoder, gpu: &CullCompactGpu, readback: &wgpu::Buffer) {
    let args_offset = SOURCE_COUNT * RECORD_BYTES;
    encoder.copy_buffer_to_buffer(
        &gpu.destination,
        0,
        readback,
        0,
        (SOURCE_COUNT * RECORD_BYTES) as u64,
    );
    encoder.copy_buffer_to_buffer(
        &gpu.args,
        0,
        readback,
        args_offset as u64,
        (ARGS_WORDS * U32_BYTES) as u64,
    );
}

fn assert_compacted_output(bytes: &[u8], records: &[[u32; RECORD_WORDS]; SOURCE_COUNT]) {
    let args_offset = SOURCE_COUNT * RECORD_BYTES;
    let instance_count = read_u32(bytes, args_offset + U32_BYTES);
    assert_eq!(instance_count, 3, "indirect args visible count");
    assert_eq!(read_u32(bytes, args_offset), INDEX_COUNT);

    let actual = compacted_records(bytes);
    let expected = vec![
        records[0].to_vec(),
        records[2].to_vec(),
        records[3].to_vec(),
    ];
    assert_eq!(actual, expected, "compacted records must be bit-identical");
    assert_eq!(actual[0][MOTION_VALID_WORD], 1);
    assert_canary_slots(bytes, instance_count as usize);
}

fn assert_capacity_guards(bytes: &[u8], records: &[[u32; RECORD_WORDS]; SOURCE_COUNT]) {
    let args_offset = SOURCE_COUNT * RECORD_BYTES;
    let instance_count = read_u32(bytes, args_offset + U32_BYTES);
    assert_eq!(instance_count, 2, "count is bounded by source, not dst");
    assert_eq!(read_u32(bytes, args_offset), INDEX_COUNT);

    let first = read_record(bytes, 0);
    assert!(
        first.as_slice() == records[0].as_slice() || first.as_slice() == records[1].as_slice(),
        "only an in-range visible source may be compacted"
    );
    assert_canary_slots(bytes, 1);
}

fn assert_canary_slots(bytes: &[u8], start_slot: usize) {
    for slot in start_slot..SOURCE_COUNT {
        for word in 0..RECORD_WORDS {
            let offset = slot * RECORD_BYTES + word * U32_BYTES;
            assert_eq!(
                read_u32(bytes, offset),
                DESTINATION_CANARY,
                "destination slot {slot} word {word} must stay within capacity"
            );
        }
    }
}

fn compacted_records(bytes: &[u8]) -> Vec<Vec<u32>> {
    let args_offset = SOURCE_COUNT * RECORD_BYTES;
    let count = read_u32(bytes, args_offset + U32_BYTES) as usize;
    let mut records = (0..count)
        .map(|slot| read_record(bytes, slot * RECORD_BYTES))
        .collect::<Vec<_>>();
    records.sort_by_key(|record| record[MATERIAL_WORD]);
    records
}

fn read_record(bytes: &[u8], start: usize) -> Vec<u32> {
    (0..RECORD_WORDS)
        .map(|word| read_u32(bytes, start + word * U32_BYTES))
        .collect()
}

fn buffer_layout_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
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

fn object_record(
    material: u32,
    motion_valid: u32,
    center_x: f32,
    scale: f32,
) -> [u32; RECORD_WORDS] {
    object_record_with_scales(material, motion_valid, center_x, [scale; 3])
}

fn object_record_with_scales(
    material: u32,
    motion_valid: u32,
    center_x: f32,
    scales: [f32; 3],
) -> [u32; RECORD_WORDS] {
    let mut words = [0_u32; RECORD_WORDS];
    words[MODEL_X_AXIS_WORD] = scales[0].to_bits();
    words[MODEL_Y_AXIS_WORD] = scales[1].to_bits();
    words[MODEL_Z_AXIS_WORD] = scales[2].to_bits();
    words[MODEL_CENTER_X_WORD] = center_x.to_bits();
    words[MODEL_CENTER_W_WORD] = 1.0_f32.to_bits();
    for (index, word) in words[NORMAL_MATRIX_START_WORD..MATERIAL_WORD]
        .iter_mut()
        .enumerate()
    {
        *word = 0x3f00_0000 + index as u32;
    }
    words[MATERIAL_WORD] = material;
    words[MOTION_VALID_WORD] = motion_valid;
    words[EXTRA_PAD_0_WORD] = 0x1234_5678;
    words[EXTRA_PAD_1_WORD] = 0x9abc_def0;
    words
}

fn frustum_planes() -> [[f32; LANES]; PLANES] {
    std::array::from_fn(|plane| {
        if plane == 0 {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.0, 0.0, 0.0, OPEN_PLANE_DISTANCE]
        }
    })
}

fn packed_planes() -> Vec<u8> {
    let mut bytes = vec![0_u8; PLANES_BYTES];
    for (plane, values) in frustum_planes().into_iter().enumerate() {
        for (lane, value) in values.into_iter().enumerate() {
            let start = (plane * LANES + lane) * UNIFORM_LANE_BYTES;
            bytes[start..start + U32_BYTES].copy_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn records_to_bytes(records: &[[u32; RECORD_WORDS]; SOURCE_COUNT]) -> Vec<u8> {
    let words = records
        .iter()
        .flat_map(|record| record.iter().copied())
        .collect::<Vec<_>>();
    words_to_bytes(&words)
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + U32_BYTES]
            .try_into()
            .expect("u32 lane"),
    )
}
