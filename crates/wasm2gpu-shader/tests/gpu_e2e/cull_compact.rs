//! GPU parity test for the wasm2gpu frustum-cull-and-compact kernel.

use ornis_wasm2gpu_testkit::compute_lift::lowered_kernel;
use ornis_wasm2gpu_testkit::device;
use ornis_wasm2gpu_testkit::entry;
use ornis_wasm2gpu_testkit::gpu_buffer;
use ornis_wasm2gpu_testkit::map_bytes;
use ornis_wasm2gpu_testkit::symbols;

const RECORD_WORDS: usize = 52;
const RECORD_BYTES: usize = RECORD_WORDS * std::mem::size_of::<u32>();
const MATERIAL_WORD: usize = 48;
const MOTION_VALID_WORD: usize = 49;
const ACTIVE_COUNT: u32 = 3;
const INDEX_COUNT: u32 = 36;
const SOURCE_COUNT: usize = 4;
const PLANES: usize = 6;
const LANES: usize = 4;
const UNIFORM_LANE_BYTES: usize = 16;
const PLANES_BYTES: usize = PLANES * LANES * UNIFORM_LANE_BYTES;
const ARGS_WORDS: usize = 5;

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

    let workgroup_size = symbols::CS_WORKGROUP[0] as usize;
    let records = [
        object_record(10, 1, 0.5, 0.25),
        object_record(11, 0, -1.0, 0.25),
        object_record(12, 1, 2.0, 0.1),
        object_record(13, 0, 0.0, 0.25),
    ];
    let source = gpu_buffer(
        &device,
        "wasm2gpu cull compact source floats",
        (SOURCE_COUNT * RECORD_BYTES) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    // The current lowering declares storage imports as read_write. Keep
    // the u32 copy view separate from the f32 culling view to avoid binding
    // the same writable buffer range twice.
    let source_words = gpu_buffer(
        &device,
        "wasm2gpu cull compact source words",
        (SOURCE_COUNT * RECORD_BYTES) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let destination = gpu_buffer(
        &device,
        "wasm2gpu cull compact destination",
        (SOURCE_COUNT * RECORD_BYTES) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let args = gpu_buffer(
        &device,
        "wasm2gpu cull compact args",
        (ARGS_WORDS * std::mem::size_of::<u32>()) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let planes = gpu_buffer(
        &device,
        "wasm2gpu cull compact planes",
        PLANES_BYTES as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let count = gpu_buffer(
        &device,
        "wasm2gpu cull compact count",
        std::mem::size_of::<u32>() as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let source_record_words = records
        .iter()
        .flat_map(|record| record.iter().copied())
        .collect::<Vec<_>>();
    let source_bytes = words_to_bytes(&source_record_words);
    queue.write_buffer(&source, 0, &source_bytes);
    queue.write_buffer(&source_words, 0, &source_bytes);
    queue.write_buffer(&args, 0, &words_to_bytes(&[INDEX_COUNT, 0, 0, 0, 0]));
    queue.write_buffer(&planes, 0, &packed_planes());
    queue.write_buffer(&count, 0, &ACTIVE_COUNT.to_le_bytes());

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wasm2gpu cull compact layout"),
        entries: &[
            buffer_layout_entry(0, wgpu::BufferBindingType::Uniform),
            buffer_layout_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_layout_entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });
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
    let wgsl = lowered_kernel("cull_compact", "storage_0_2_");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wasm2gpu cull compact"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wasm2gpu cull compact"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wasm2gpu cull compact"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(symbols::CS_MAIN),
        compilation_options: Default::default(),
        cache: None,
    });
    let readback = gpu_buffer(
        &device,
        "wasm2gpu cull compact readback",
        ((SOURCE_COUNT * RECORD_BYTES) + ARGS_WORDS * 4) as u64,
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
        pass.dispatch_workgroups((ACTIVE_COUNT as usize).div_ceil(workgroup_size) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &destination,
        0,
        &readback,
        0,
        (SOURCE_COUNT * RECORD_BYTES) as u64,
    );
    encoder.copy_buffer_to_buffer(
        &args,
        0,
        &readback,
        (SOURCE_COUNT * RECORD_BYTES) as u64,
        (ARGS_WORDS * 4) as u64,
    );
    queue.submit([encoder.finish()]);

    let bytes = map_bytes(&device, &readback);
    let args_offset = SOURCE_COUNT * RECORD_BYTES;
    let instance_count = read_u32(&bytes, args_offset + 4);
    assert_eq!(instance_count, 2, "indirect args visible count");
    assert_eq!(read_u32(&bytes, args_offset), INDEX_COUNT);

    let mut actual = Vec::new();
    for slot in 0..instance_count as usize {
        let start = slot * RECORD_BYTES;
        let words = (0..RECORD_WORDS)
            .map(|word| read_u32(&bytes, start + word * 4))
            .collect::<Vec<_>>();
        actual.push(words);
    }
    actual.sort_by_key(|record| record[MATERIAL_WORD]);
    let expected = vec![records[0].to_vec(), records[2].to_vec()];
    assert_eq!(actual, expected, "compacted records must be bit-identical");
    assert_eq!(actual[0][MOTION_VALID_WORD], 1);
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

fn object_record(material: u32, motion_valid: u32, center_x: f32, scale: f32) -> [u32; RECORD_WORDS] {
    let mut words = [0_u32; RECORD_WORDS];
    words[0] = scale.to_bits();
    words[5] = scale.to_bits();
    words[10] = scale.to_bits();
    words[12] = center_x.to_bits();
    words[15] = 1.0_f32.to_bits();
    for (index, word) in words[16..48].iter_mut().enumerate() {
        *word = 0x3f00_0000 + index as u32;
    }
    words[MATERIAL_WORD] = material;
    words[MOTION_VALID_WORD] = motion_valid;
    words[50] = 0x1234_5678;
    words[51] = 0x9abc_def0;
    words
}

fn packed_planes() -> Vec<u8> {
    let mut bytes = vec![0_u8; PLANES_BYTES];
    for plane in 0..PLANES {
        let values = if plane == 0 {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.0, 0.0, 0.0, 100.0]
        };
        for (lane, value) in values.into_iter().enumerate() {
            let start = (plane * LANES + lane) * UNIFORM_LANE_BYTES;
            bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 lane"))
}
