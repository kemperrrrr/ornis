//! Parity of the generated llvm2gpu cull kernel with the legacy WGSL.
//!
//! [`CULL_COMPUTE_WGSL`] is the production artifact generated at build
//! time from `crates/gpu-shaders`; [`LEGACY_CULL_COMPUTE_WGSL`] is the
//! hand-written oracle kept test-only until the gates pass. Both read the
//! same frustum and the same eight spheres encoded as `PerObject` model
//! matrices, so `instance_sphere` reads the center from column 3 and the
//! radius from the column lengths. Compaction order is scheduler
//! dependent, so the contract is a sorted set of `material_index` values
//! plus the atomic count, not byte order.

use ornis_render::gpu_culling::{
    CULL_COMPUTE_WGSL, CULL_PARAMS_BYTE_SIZE, INSTANCE_COUNT_WORD, LEGACY_CULL_COMPUTE_WGSL,
};
use ornis_render::renderer::PerObjectGpu;
use ornis_wasm2gpu_testkit::cull::{ACTIVE_COUNT, EXPECTED, PLANES, SPHERES, buffer_entry};
use ornis_wasm2gpu_testkit::{
    device, entry, gpu_buffer, map_bytes, put_f32, put_u32, storage_buffer,
};

/// Production object-record stride, derived from the Rust ABI type.
const OBJECT_BYTES: usize = std::mem::size_of::<PerObjectGpu>();
/// Byte offset of `material_index` in the Rust ABI type.
const MATERIAL_OFFSET: usize = std::mem::offset_of!(PerObjectGpu, material_index);
/// Byte offset of the model matrix in the Rust ABI type.
const MODEL_OFFSET: u32 = std::mem::offset_of!(PerObjectGpu, model) as u32;
/// Bytes in one column of the model matrix.
const MODEL_COLUMN_BYTES: u32 = std::mem::size_of::<[f32; 4]>() as u32;
/// Bytes in one scalar lane.
const F32_BYTES: u32 = std::mem::size_of::<f32>() as u32;
/// Model-matrix diagonal offsets used to encode the sphere radius.
const RADIUS_X_OFFSET: u32 = MODEL_OFFSET;
const RADIUS_Y_OFFSET: u32 = MODEL_OFFSET + MODEL_COLUMN_BYTES + F32_BYTES;
const RADIUS_Z_OFFSET: u32 = MODEL_OFFSET + 2 * MODEL_COLUMN_BYTES + 2 * F32_BYTES;
/// Translation column of the model matrix.
const TRANSLATION_OFFSET: u32 = MODEL_OFFSET + 3 * MODEL_COLUMN_BYTES;
/// `args` holds `array<atomic<u32>, 5>`; word 1 is the compacted count.
const ARGS_WORDS: usize = 5;

#[derive(Clone, Copy)]
enum CullOutput {
    /// Dense `PerObject` records (legacy oracle).
    Record,
    /// Dense `u32` source indices (llvm2gpu production).
    Index,
}

/// Dispatch the legacy oracle once and read back the compacted set:
/// `material_index` of every entry below the atomic count, sorted.
fn run_legacy(device: &wgpu::Device, queue: &wgpu::Queue) -> (Vec<u32>, u32) {
    run_cull(
        device,
        queue,
        "legacy cull",
        LEGACY_CULL_COMPUTE_WGSL,
        "cull_instances",
        CullOutput::Record,
    )
}

/// Dispatch any cull WGSL with the shipped compaction bindings and read
/// back the sorted `material_index` set plus the atomic count.
fn run_cull(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    wgsl: &str,
    entry_point: &str,
    output: CullOutput,
) -> (Vec<u32>, u32) {
    let bgl = production_layout(device);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let params = gpu_buffer(device, "prod-cull-params", CULL_PARAMS_BYTE_SIZE, uniform);
    let src = storage_buffer(
        device,
        "prod-cull-src",
        (SPHERES.len() * OBJECT_BYTES) as u64,
    );
    let dst = storage_buffer(
        device,
        "prod-cull-dst",
        (ACTIVE_COUNT * OBJECT_BYTES) as u64,
    );
    let args = storage_buffer(device, "prod-cull-args", (ARGS_WORDS * 4) as u64);
    queue.write_buffer(&params, 0, &pack_prod_params());
    queue.write_buffer(&src, 0, &pack_prod_objects());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("production cull"),
        layout: &bgl,
        entries: &[
            entry(0, &params),
            entry(1, &src),
            entry(2, &dst),
            entry(3, &args),
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
    let mut visible = Vec::with_capacity(count as usize);
    match output {
        CullOutput::Record => {
            let bytes = read_bytes(device, queue, &dst, count as u64 * OBJECT_BYTES as u64);
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
        }
        CullOutput::Index => {
            let bytes = read_bytes(device, queue, &dst, count as u64 * 4);
            let source = pack_prod_objects();
            let mut entry = 0u32;
            while entry < count {
                let word = entry as usize * 4;
                let index = u32::from_le_bytes([
                    bytes[word],
                    bytes[word + 1],
                    bytes[word + 2],
                    bytes[word + 3],
                ]) as usize;
                let at = index * OBJECT_BYTES + MATERIAL_OFFSET;
                visible.push(u32::from_le_bytes([
                    source[at],
                    source[at + 1],
                    source[at + 2],
                    source[at + 3],
                ]));
                entry += 1;
            }
        }
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
            put_f32(&mut bytes, base + lane * 4, PLANES[plane][lane as usize]);
            lane += 1;
        }
        plane += 1;
    }
    put_u32(&mut bytes, (PLANES.len() * 16) as u32, ACTIVE_COUNT as u32);
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
        put_f32(&mut bytes, base + RADIUS_X_OFFSET, sphere[3]);
        put_f32(&mut bytes, base + RADIUS_Y_OFFSET, sphere[3]);
        put_f32(&mut bytes, base + RADIUS_Z_OFFSET, sphere[3]);
        put_f32(&mut bytes, base + TRANSLATION_OFFSET, sphere[0]);
        put_f32(&mut bytes, base + TRANSLATION_OFFSET + F32_BYTES, sphere[1]);
        put_f32(
            &mut bytes,
            base + TRANSLATION_OFFSET + 2 * F32_BYTES,
            sphere[2],
        );
        put_f32(&mut bytes, base + TRANSLATION_OFFSET + 3 * F32_BYTES, 1.0);
        put_u32(&mut bytes, base + MATERIAL_OFFSET as u32, object as u32);
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
    map_bytes(device, &read)
}

#[test]
fn llvm2gpu_cull_matches_legacy_oracle() {
    let nan_radius_slot = SPHERES
        .iter()
        .position(|sphere| sphere[3].is_nan())
        .expect("test scene has a NaN radius");
    assert_eq!(
        EXPECTED[nan_radius_slot], 1,
        "a NaN radius must fail open and remain visible"
    );
    let expected: Vec<u32> = (0..ACTIVE_COUNT as u32)
        .filter(|slot| EXPECTED[*slot as usize] == 1)
        .collect();
    match device() {
        Some((device, queue)) => {
            let (legacy_visible, legacy_count) = run_legacy(&device, &queue);
            let (llvm_visible, llvm_count) = run_cull(
                &device,
                &queue,
                "llvm2gpu cull",
                CULL_COMPUTE_WGSL,
                "cs_main",
                CullOutput::Index,
            );
            assert_eq!(llvm_count, legacy_count, "atomic count");
            assert_eq!(llvm_visible, legacy_visible, "compacted set");
            assert_eq!(llvm_visible, expected, "expected set");
            let nan_slot = nan_radius_slot as u32;
            assert!(
                llvm_visible.contains(&nan_slot),
                "llvm2gpu NaN radius fails open"
            );
            eprintln!(
                "llvm2gpu cull parity: legacy {legacy_visible:?} == llvm2gpu {llvm_visible:?}"
            );
        }
        None if std::env::var_os("ORNIS_REQUIRE_GPU").is_some() => {
            panic!("ORNIS_REQUIRE_GPU is set and no wgpu adapter was found");
        }
        None => eprintln!("llvm2gpu_cull_matches_legacy_oracle: no wgpu adapter — skipped"),
    }
}
