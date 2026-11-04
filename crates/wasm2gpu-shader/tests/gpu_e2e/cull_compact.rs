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
const ACTIVE_COUNT: u32 = 3;
const INDEX_COUNT: u32 = 36;
const SOURCE_COUNT: usize = 4;
const PLANES: usize = 6;
const LANES: usize = 4;
const UNIFORM_LANE_BYTES: usize = 16;
const PLANES_BYTES: usize = PLANES * LANES * UNIFORM_LANE_BYTES;
const CULL_UNIFORM_BYTES: usize = PLANES_BYTES + UNIFORM_LANE_BYTES;
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
const FAR_VISIBLE_RADIUS: f32 = 0.1;
const OPEN_PLANE_DISTANCE: f32 = 100.0;

struct CullCompactGpu {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    destination: wgpu::Buffer,
    args: wgpu::Buffer,
}

struct CullCompactInputs<'a> {
    source: &'a wgpu::Buffer,
    source_words: &'a wgpu::Buffer,
    args: &'a wgpu::Buffer,
    planes: &'a wgpu::Buffer,
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
    let gpu = create_gpu_case(&device, &queue, &records);
    let bytes = dispatch_and_readback(&device, &queue, &gpu);
    assert_compacted_output(&bytes, &records);
}

fn test_records() -> [[u32; RECORD_WORDS]; SOURCE_COUNT] {
    [
        object_record(MATERIAL_BASE, 1, VISIBLE_CENTER_NEAR, VISIBLE_RADIUS),
        object_record(MATERIAL_NEXT, 0, CULLED_CENTER, VISIBLE_RADIUS),
        object_record(
            MATERIAL_LAST_VISIBLE,
            1,
            VISIBLE_CENTER_FAR,
            FAR_VISIBLE_RADIUS,
        ),
        object_record(MATERIAL_INACTIVE, 0, 0.0, VISIBLE_RADIUS),
    ]
}

fn create_gpu_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    records: &[[u32; RECORD_WORDS]; SOURCE_COUNT],
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
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
        CULL_UNIFORM_BYTES as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let inputs = CullCompactInputs {
        source: &source,
        source_words: &source_words,
        args: &args,
        planes: &planes,
    };
    upload_inputs(queue, records, &inputs);

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
) {
    let source_record_words = records
        .iter()
        .flat_map(|record| record.iter().copied())
        .collect::<Vec<_>>();
    let source_bytes = words_to_bytes(&source_record_words);
    queue.write_buffer(inputs.source, 0, &source_bytes);
    queue.write_buffer(inputs.source_words, 0, &source_bytes);
    queue.write_buffer(inputs.args, 0, &words_to_bytes(&[INDEX_COUNT, 0, 0, 0, 0]));
    queue.write_buffer(inputs.planes, 0, &packed_planes(ACTIVE_COUNT));
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
        ],
    })
}

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
    let workgroup_size = symbols::CS_WORKGROUP[0] as usize;
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
        pass.dispatch_workgroups(
            (ACTIVE_COUNT as usize).div_ceil(workgroup_size) as u32,
            1,
            1,
        );
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
    assert_eq!(instance_count, 2, "indirect args visible count");
    assert_eq!(read_u32(bytes, args_offset), INDEX_COUNT);

    let mut actual = (0..instance_count as usize)
        .map(|slot| read_record(bytes, slot * RECORD_BYTES))
        .collect::<Vec<_>>();
    actual.sort_by_key(|record| record[MATERIAL_WORD]);
    let expected = vec![records[0].to_vec(), records[2].to_vec()];
    assert_eq!(actual, expected, "compacted records must be bit-identical");
    assert_eq!(actual[0][MOTION_VALID_WORD], 1);
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
    let mut words = [0_u32; RECORD_WORDS];
    words[MODEL_X_AXIS_WORD] = scale.to_bits();
    words[MODEL_Y_AXIS_WORD] = scale.to_bits();
    words[MODEL_Z_AXIS_WORD] = scale.to_bits();
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

fn packed_planes(count: u32) -> Vec<u8> {
    let mut bytes = vec![0_u8; CULL_UNIFORM_BYTES];
    for plane in 0..PLANES {
        let values: [f32; LANES] = if plane == 0 {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.0, 0.0, 0.0, OPEN_PLANE_DISTANCE]
        };
        for (lane, value) in values.into_iter().enumerate() {
            let start = (plane * LANES + lane) * UNIFORM_LANE_BYTES;
            bytes[start..start + U32_BYTES].copy_from_slice(&value.to_le_bytes());
        }
    }
    bytes[PLANES_BYTES..PLANES_BYTES + U32_BYTES].copy_from_slice(&(count as f32).to_le_bytes());
    bytes
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
