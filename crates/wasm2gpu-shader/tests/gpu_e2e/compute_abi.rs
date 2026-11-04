//! Execute the ABI imports after lowering a compact Wasm fixture.
//!
//! The same import sequence is host-shadow tested separately; this explicit
//! Wasm avoids rustc's stack-pointer globals so this test isolates import ABI
//! lowering and GPU execution.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;
use ornis_wasm2gpu_testkit::record_compute_pass;
use ornis_wasm2gpu_testkit::storage_buffer;
use std::fs;
use std::process::Command;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

static SPIRV_TEMP_FILE_ID: AtomicU64 = AtomicU64::new(0);

const FLOAT_OUTPUT_BITS: [u32; 8] = [
    0,
    0x3f80_0000,
    0,
    0x3f80_0000,
    0,
    0x3f80_0000,
    0,
    0x3f80_0000,
];
const WORKGROUP_INITIAL_VALUE: u32 = 7;
const U32_OUTPUT: [u32; 4] = [
    INITIAL_ATOMIC_VALUE,
    WORKGROUP_INITIAL_VALUE,
    0,
    WORKGROUP_ATOMIC_DELTA,
];
const ATOMIC_OUTPUT: [u32; 1] = [15];
const FLOAT_OUTPUT_BYTES: u64 = 32;
const U32_OUTPUT_BYTES: u64 = 16;
const ATOMIC_OUTPUT_BYTES: u64 = 4;
const INITIAL_ATOMIC_VALUE: u32 = 10;
const WORKGROUP_ATOMIC_DELTA: u32 = 5;
const DISPATCH_GROUPS: [u32; 3] = [1, 1, 1];
const TEXTURE_BINDING: u32 = 1;
const SAMPLER_BINDING: u32 = 2;
const FLOAT_OUTPUT_BINDING: u32 = 3;
const ATOMIC_OUTPUT_BINDING: u32 = 4;
const U32_OUTPUT_BINDING: u32 = 5;
const TEXTURE_WIDTH: u32 = 2;
const TEXTURE_HEIGHT: u32 = 1;
const TEXTURE_LAYERS: u32 = 1;
const TEXTURE_MIP_LEVELS: u32 = 1;
const TEXTURE_SAMPLE_COUNT: u32 = 1;

fn wasm() -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
            (import "{module}" "{texture_load}" (func $texture_load
                (param i32 i32 i32 i32) (result f32 f32 f32 f32)))
            (import "{module}" "{texture_sample_lod}" (func $texture_sample_lod
                (param i32 i32 i32 i32 f32 f32 f32) (result f32 f32 f32 f32)))
            (import "{module}" "{storage_write_f32}" (func $storage_write_f32
                (param i32 i32 i32 f32)))
            (import "{module}" "{storage_write_u32}" (func $storage_write_u32
                (param i32 i32 i32 i32)))
            (import "{module}" "{storage_atomic_add}" (func $storage_atomic_add
                (param i32 i32 i32 i32) (result i32)))
            (import "{module}" "{workgroup_store}" (func $workgroup_store
                (param i32 i32)))
            (import "{module}" "{workgroup_load}" (func $workgroup_load
                (param i32) (result i32)))
            (import "{module}" "{workgroup_atomic_add}" (func $workgroup_atomic_add
                (param i32 i32) (result i32)))
            (import "{module}" "{workgroup_barrier}" (func $workgroup_barrier))
            (import "{module}" "{storage_barrier}" (func $storage_barrier))
            (func (export "{vs}") (param f32 f32 f32 f32)
                (result f32 f32 f32 f32 f32 f32)
                local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)
            (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
                f32.const 0 f32.const 0 f32.const 0 f32.const 1)
            (func (export "{cs}") (param i32)
                (local $storage_old i32) (local $workgroup_loaded i32)
                (local $workgroup_old i32)
                ;; Lane zero performs the writes; every lane reaches each barrier.
                local.get 0 i32.eqz
                if
                    i32.const 0 i32.const 1 i32.const 1 i32.const 0 call $texture_load drop
                    i32.const 0 i32.const 1 i32.const 0 i32.const 2
                    f32.const 0.75 f32.const 0.5 f32.const 0 call $texture_sample_lod drop

                    i32.const 0 i32.const 3 i32.const 0 f32.const 0 call $storage_write_f32
                    i32.const 0 i32.const 3 i32.const 1 f32.const 1 call $storage_write_f32
                    i32.const 0 i32.const 3 i32.const 2 f32.const 0 call $storage_write_f32
                    i32.const 0 i32.const 3 i32.const 3 f32.const 1 call $storage_write_f32
                    i32.const 0 i32.const 3 i32.const 4 f32.const 0 call $storage_write_f32
                    i32.const 0 i32.const 3 i32.const 5 f32.const 1 call $storage_write_f32
                    i32.const 0 i32.const 3 i32.const 6 f32.const 0 call $storage_write_f32
                    i32.const 0 i32.const 3 i32.const 7 f32.const 1 call $storage_write_f32

                    i32.const 0 i32.const 4 i32.const 0 i32.const 5
                    call $storage_atomic_add local.set $storage_old
                    i32.const 0 i32.const 5 i32.const 0 local.get $storage_old
                    call $storage_write_u32
                    i32.const 0 i32.const 7 call $workgroup_store
                end
                call $workgroup_barrier
                local.get 0 i32.eqz
                if
                    i32.const 0 call $workgroup_load local.set $workgroup_loaded
                    i32.const 0 i32.const 5 i32.const 1 local.get $workgroup_loaded
                    call $storage_write_u32
                    i32.const 1 i32.const 5 call $workgroup_atomic_add
                    local.set $workgroup_old
                    i32.const 0 i32.const 5 i32.const 2 local.get $workgroup_old
                    call $storage_write_u32
                    i32.const 1 i32.const 7 call $workgroup_atomic_add
                    local.set $workgroup_old
                    i32.const 0 i32.const 5 i32.const 3 local.get $workgroup_old
                    call $storage_write_u32
                end
                call $storage_barrier))"#,
        module = symbols::MODULE,
        texture_load = symbols::TEXTURE_LOAD_2D,
        texture_sample_lod = symbols::TEXTURE_SAMPLE_2D_LOD,
        storage_write_f32 = symbols::STORAGE_WRITE_F32,
        storage_write_u32 = symbols::STORAGE_WRITE_U32,
        storage_atomic_add = symbols::STORAGE_ATOMIC_ADD_U32,
        workgroup_store = symbols::WORKGROUP_STORE_U32,
        workgroup_load = symbols::WORKGROUP_LOAD_U32,
        workgroup_atomic_add = symbols::WORKGROUP_ATOMIC_ADD_U32,
        workgroup_barrier = symbols::WORKGROUP_BARRIER,
        storage_barrier = symbols::STORAGE_BARRIER,
        vs = symbols::VS_MAIN,
        fs = symbols::FS_MAIN,
        cs = symbols::CS_MAIN,
    ))
    .expect("valid compute ABI fixture")
}

#[test]
fn wasm_compute_abi_imports_execute_on_gpu() {
    let translated = translate(&wasm()).unwrap_or_else(|error| {
        eprintln!("error: compute ABI translation failed: {error:?}");
        panic!("translate ABI imports: {error:?}");
    });
    assert!(!translated.compute_spirv.is_empty());
    validate_compute_spirv(&translated.compute_spirv);
    let Some((device, queue)) = super::device() else {
        let required = env_flag_enabled("ORNIS_REQUIRE_GPU");
        assert!(
            !required,
            "ORNIS_REQUIRE_GPU is set but no adapter was found"
        );
        eprintln!("SKIPPED (wasm compute ABI imports): no wgpu adapter");
        return;
    };

    let (floats, workgroup, atomic) = dispatch(&device, &queue, &translated.wgsl);
    assert_eq!(floats.as_slice(), FLOAT_OUTPUT_BITS.as_slice());
    assert_eq!(workgroup.as_slice(), U32_OUTPUT.as_slice());
    assert_eq!(atomic.as_slice(), ATOMIC_OUTPUT.as_slice());
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| !value.is_empty() && value != "0")
}

fn validate_compute_spirv(words: &[u32]) {
    let required = env_flag_enabled("ORNIS_REQUIRE_SPIRV_VAL");
    let probe = Command::new("spirv-val").arg("--version").output();
    let Ok(probe) = probe else {
        assert!(
            !required,
            "ORNIS_REQUIRE_SPIRV_VAL is enabled but spirv-val is unavailable"
        );
        eprintln!("SKIPPED (compute SPIR-V validation): spirv-val is unavailable");
        return;
    };
    assert!(probe.status.success(), "spirv-val --version failed");

    let file_id = SPIRV_TEMP_FILE_ID.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "ornis-wasm2gpu-compute-{}-{file_id}.spv",
        std::process::id()
    ));
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(&path, bytes).expect("write compute SPIR-V for validation");
    let result = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3"])
        .arg(&path)
        .output()
        .expect("run spirv-val");
    let _ = fs::remove_file(&path);
    assert!(
        result.status.success(),
        "spirv-val rejected compute SPIR-V:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wasm2gpu compute ABI probe"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wasm2gpu compute ABI probe"),
        layout: None,
        module: &shader,
        entry_point: Some(symbols::CS_MAIN),
        compilation_options: Default::default(),
        cache: None,
    });
    let floats = storage_buffer(device, "abi-float-output", FLOAT_OUTPUT_BYTES);
    let workgroup = storage_buffer(device, "abi-u32-output", U32_OUTPUT_BYTES);
    let atomic = storage_buffer(device, "abi-atomic-output", ATOMIC_OUTPUT_BYTES);
    queue.write_buffer(&atomic, 0, &INITIAL_ATOMIC_VALUE.to_le_bytes());
    let (bind_group, _texture, _view, _sampler) =
        bind_probe(device, &pipeline, [&floats, &atomic, &workgroup]);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wasm2gpu compute ABI probe"),
    });
    record_compute_pass(&mut encoder, &pipeline, &bind_group, DISPATCH_GROUPS);
    queue.submit([encoder.finish()]);
    (
        super::read_words(device, queue, &floats, FLOAT_OUTPUT_BYTES),
        super::read_words(device, queue, &workgroup, U32_OUTPUT_BYTES),
        super::read_words(device, queue, &atomic, ATOMIC_OUTPUT_BYTES),
    )
}

fn bind_probe(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    buffers: [&wgpu::Buffer; 3],
) -> (
    wgpu::BindGroup,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Sampler,
) {
    let [floats, atomic, workgroup] = buffers;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("wasm2gpu compute ABI texture"),
        size: wgpu::Extent3d {
            width: TEXTURE_WIDTH,
            height: TEXTURE_HEIGHT,
            depth_or_array_layers: TEXTURE_LAYERS,
        },
        mip_level_count: TEXTURE_MIP_LEVELS,
        sample_count: TEXTURE_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wasm2gpu compute ABI probe"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: TEXTURE_BINDING,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: SAMPLER_BINDING,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            super::entry(FLOAT_OUTPUT_BINDING, floats),
            super::entry(ATOMIC_OUTPUT_BINDING, atomic),
            super::entry(U32_OUTPUT_BINDING, workgroup),
        ],
    });
    (bind_group, texture, view, sampler)
}
