//! A/B of the shipped cull/compaction WGSL against the llvm2gpu kernel.
//!
//! Four scenes (few, many, all visible, late reject), one dispatch per
//! sample, GPU timestamps when the adapter exposes them. Reproduce:
//!
//! ```text
//! ORNIS_REQUIRE_GPU=1 cargo bench -p ornis-render \
//!     --bench llvm2gpu_cull_ab --features test-util
//! ```

use std::time::Instant;

use ornis_render::gpu_culling::{
    CULL_COMPUTE_WGSL, CULL_PARAMS_BYTE_SIZE, CULL_WORKGROUP_SIZE, INSTANCE_COUNT_WORD,
    LEGACY_CULL_COMPUTE_WGSL,
};
use ornis_render::renderer::{MaterialIdx, PerObjectGpu};
use ornis_wasm2gpu_testkit::{entry, gpu_buffer, map_bytes, storage_buffer};

/// Record stride in the shared `PerObject` bytes.
const OBJECT_BYTES: usize = std::mem::size_of::<PerObjectGpu>();
/// Words in the indirect-args buffer.
const ARGS_WORDS: usize = 5;
/// Measured dispatches per shader and scene.
const RUNS: usize = 100;
/// Warmup dispatches per shader and scene.
const WARMUP: usize = 5;
/// Six normalized planes: identity cube around the origin.
const PLANES: [[f32; 4]; 6] = [
    [-1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

struct Scene {
    name: &'static str,
    records: usize,
    visible: usize,
    late: bool,
}

fn main() {
    let Some(device) = device() else {
        eprintln!("llvm2gpu_cull_ab: no wgpu adapter — skipped");
        if std::env::var_os("ORNIS_REQUIRE_GPU").is_some() {
            std::process::exit(1);
        }
        return;
    };
    let bgl = layout(&device.device);
    let legacy = pipeline(
        &device.device,
        &bgl,
        "legacy cull",
        LEGACY_CULL_COMPUTE_WGSL,
        "cull_instances",
    );
    let llvm2gpu = pipeline(
        &device.device,
        &bgl,
        "llvm2gpu cull",
        CULL_COMPUTE_WGSL,
        "cs_main",
    );
    let scenes = [
        Scene {
            name: "few",
            records: 256,
            visible: 64,
            late: false,
        },
        Scene {
            name: "many",
            records: 8192,
            visible: 64,
            late: false,
        },
        Scene {
            name: "all_visible",
            records: 8192,
            visible: 8192,
            late: false,
        },
        Scene {
            name: "late_reject",
            records: 8192,
            visible: 0,
            late: true,
        },
    ];
    println!(
        "adapter: {} ({:?}, timestamps: {})\n",
        device.name, device.backend, device.timestamps
    );
    println!(
        "{:<12} {:>6} {:>7} {:>10} {:>10} {:>8}",
        "scene", "n", "visible", "legacy us", "llvm2gpu", "delta"
    );
    for scene in &scenes {
        let (prod, llvm) = run_scene(&device, &bgl, &legacy, &llvm2gpu, scene);
        let delta = (llvm / prod - 1.0) * 100.0;
        println!(
            "{:<12} {:>6} {:>7} {:>10.2} {:>10.2} {:>7.1}%",
            scene.name, scene.records, scene.visible, prod, llvm, delta
        );
    }
}

struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    name: String,
    backend: wgpu::Backend,
    timestamps: bool,
    period_ns: f32,
}

fn device() -> Option<Device> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let info = adapter.get_info();
    let features = wgpu::Features::TIMESTAMP_QUERY & adapter.features();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("llvm2gpu cull ab"),
        required_features: features,
        required_limits: wgpu::Limits::default(),
        memory_hints: Default::default(),
        trace: Default::default(),
        experimental_features: Default::default(),
    }))
    .ok()?;
    Some(Device {
        timestamps: device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        period_ns: queue.get_timestamp_period(),
        name: info.name,
        backend: info.backend,
        device,
        queue,
    })
}

fn pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    label: &str,
    wgsl: &str,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mutable = wgpu::BufferBindingType::Storage { read_only: false };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("llvm2gpu cull ab"),
        entries: &[
            buffer_entry(0, wgpu::BufferBindingType::Uniform),
            buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            buffer_entry(2, mutable),
            buffer_entry(3, mutable),
            buffer_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
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

fn run_scene(
    device: &Device,
    bgl: &wgpu::BindGroupLayout,
    production: &wgpu::ComputePipeline,
    llvm2gpu: &wgpu::ComputePipeline,
    scene: &Scene,
) -> (f64, f64) {
    let records = scene_records(scene);
    let params = pack_params(scene.records as u32);
    let params_buffer = gpu_buffer(
        &device.device,
        "cull-params",
        CULL_PARAMS_BYTE_SIZE,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    device.queue.write_buffer(&params_buffer, 0, &params);
    let src = storage_buffer(
        &device.device,
        "cull-src",
        (records.len() * OBJECT_BYTES) as u64,
    );
    device
        .queue
        .write_buffer(&src, 0, bytemuck::cast_slice(&records));
    let dst = storage_buffer(
        &device.device,
        "cull-dst",
        (records.len() * OBJECT_BYTES) as u64,
    );
    let args = storage_buffer(
        &device.device,
        "cull-args",
        (ARGS_WORDS * std::mem::size_of::<u32>()) as u64,
    );
    let bind = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cull ab"),
        layout: bgl,
        entries: &[
            entry(0, &params_buffer),
            entry(1, &src),
            entry(2, &dst),
            entry(3, &args),
            entry(4, &src),
        ],
    });
    let groups = (records.len() as u32).div_ceil(CULL_WORKGROUP_SIZE);
    measure_pair(
        device,
        production,
        llvm2gpu,
        &bind,
        &args,
        groups,
        scene.visible,
    )
}

fn measure_pair(
    device: &Device,
    legacy: &wgpu::ComputePipeline,
    llvm2gpu: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    args: &wgpu::Buffer,
    groups: u32,
    visible: usize,
) -> (f64, f64) {
    for warmup in 0..WARMUP {
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cull warmup"),
            });
        let (first, second) = if warmup % 2 == 0 {
            (legacy, llvm2gpu)
        } else {
            (llvm2gpu, legacy)
        };
        encode_run(&mut encoder, first, bind, args, groups, None, 0);
        encode_run(&mut encoder, second, bind, args, groups, None, 0);
        device.queue.submit([encoder.finish()]);
    }
    let _ = device.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let times = if device.timestamps {
        measure_pair_gpu(device, legacy, llvm2gpu, bind, args, groups)
    } else {
        (
            measure_wall(device, legacy, bind, args, groups),
            measure_wall(device, llvm2gpu, bind, args, groups),
        )
    };
    let count = read_count(device, args);
    assert_eq!(
        count as usize, visible,
        "scene count mismatch: gpu {count} expected {visible}"
    );
    times
}

fn encode_run(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    args: &wgpu::Buffer,
    groups: u32,
    stamps: Option<&wgpu::QuerySet>,
    stamp_index: u32,
) {
    encoder.clear_buffer(
        args,
        (INSTANCE_COUNT_WORD * std::mem::size_of::<u32>()) as u64,
        Some(std::mem::size_of::<u32>() as u64),
    );
    let timestamp_writes = stamps.map(|queries| wgpu::ComputePassTimestampWrites {
        query_set: queries,
        beginning_of_pass_write_index: Some(stamp_index),
        end_of_pass_write_index: Some(stamp_index + 1),
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("cull ab"),
        timestamp_writes,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Interleave one legacy and one llvm2gpu pass per sample so clock/thermal
/// drift hits both sides equally; the pass timestamps bracket only the
/// dispatch, with the args reset outside the window.
fn measure_pair_gpu(
    device: &Device,
    legacy: &wgpu::ComputePipeline,
    llvm2gpu: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    args: &wgpu::Buffer,
    groups: u32,
) -> (f64, f64) {
    let slots = 4 * RUNS;
    let queries = device.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("cull stamps"),
        ty: wgpu::QueryType::Timestamp,
        count: slots as u32,
    });
    let bytes = (slots * std::mem::size_of::<u64>()) as u64;
    let resolve = gpu_buffer(
        &device.device,
        "cull resolve",
        bytes,
        wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = gpu_buffer(
        &device.device,
        "cull stamps readback",
        bytes,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cull stamps"),
        });
    for run in 0..RUNS {
        let base = (4 * run) as u32;
        let (first, second) = if run % 2 == 0 {
            (legacy, llvm2gpu)
        } else {
            (llvm2gpu, legacy)
        };
        encode_run(
            &mut encoder,
            first,
            bind,
            args,
            groups,
            Some(&queries),
            base,
        );
        encode_run(
            &mut encoder,
            second,
            bind,
            args,
            groups,
            Some(&queries),
            base + 2,
        );
    }
    encoder.resolve_query_set(&queries, 0..slots as u32, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    let _ = device.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let bytes = map_bytes(&device.device, &readback);
    let words: Vec<u64> = bytes
        .chunks_exact(std::mem::size_of::<u64>())
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("timestamp word")))
        .collect();
    let mut legacy_samples = Vec::with_capacity(RUNS);
    let mut llvm_samples = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        let base = 4 * run;
        let (legacy_at, llvm_at) = if run % 2 == 0 {
            (base, base + 2)
        } else {
            (base + 2, base)
        };
        legacy_samples.push(delta_us(&words, legacy_at, device.period_ns));
        llvm_samples.push(delta_us(&words, llvm_at, device.period_ns));
    }
    (median(&mut legacy_samples), median(&mut llvm_samples))
}

fn delta_us(words: &[u64], at: usize, period_ns: f32) -> f64 {
    (words[at + 1].saturating_sub(words[at])) as f64 * period_ns as f64 / 1000.0
}

fn median(samples: &mut Vec<f64>) -> f64 {
    samples.retain(|sample| *sample > 0.0);
    samples.sort_by(f64::total_cmp);
    samples.get(samples.len() / 2).copied().unwrap_or_default()
}

fn measure_wall(
    device: &Device,
    pipeline: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    args: &wgpu::Buffer,
    groups: u32,
) -> f64 {
    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cull wall"),
        });
    for _ in 0..RUNS {
        encode_run(&mut encoder, pipeline, bind, args, groups, None, 0);
    }
    let started = Instant::now();
    device.queue.submit([encoder.finish()]);
    let _ = device.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    started.elapsed().as_nanos() as f64 / RUNS as f64 / 1000.0
}

fn read_count(device: &Device, args: &wgpu::Buffer) -> u32 {
    let readback = gpu_buffer(
        &device.device,
        "cull args readback",
        std::mem::size_of::<u32>() as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cull args"),
        });
    encoder.copy_buffer_to_buffer(
        args,
        (INSTANCE_COUNT_WORD * std::mem::size_of::<u32>()) as u64,
        &readback,
        0,
        std::mem::size_of::<u32>() as u64,
    );
    device.queue.submit([encoder.finish()]);
    let bytes = map_bytes(&device.device, &readback);
    u32::from_le_bytes(bytes[..4].try_into().expect("count word"))
}

fn scene_records(scene: &Scene) -> Vec<PerObjectGpu> {
    (0..scene.records)
        .map(|index| {
            let mut record = PerObjectGpu::default();
            record.model = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            if index < scene.visible {
                record.model[3] = [0.0, 0.0, 0.0, 1.0];
            } else if scene.late {
                record.model[3] = [0.0, 0.0, -1000.0, 1.0];
            } else {
                record.model[3] = [1000.0, 0.0, 0.0, 1.0];
            }
            record.material_index = MaterialIdx::from_raw(index as u32);
            record
        })
        .collect()
}

fn pack_params(count: u32) -> Vec<u8> {
    let mut bytes = vec![0u8; CULL_PARAMS_BYTE_SIZE as usize];
    for (index, plane) in PLANES.iter().enumerate() {
        let base = index * 16;
        for (lane, value) in plane.iter().enumerate() {
            bytes[base + lane * 4..base + lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    let count_at = PLANES.len() * 16;
    bytes[count_at..count_at + 4].copy_from_slice(&count.to_le_bytes());
    bytes
}
