//! Execute the ABI imports after lowering a compact Wasm fixture.
//!
//! The matching Rust shader is host-shadow tested separately; this explicit
//! Wasm avoids rustc's stack-pointer globals so this test isolates import ABI
//! lowering and GPU execution.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;
use ornis_wasm2gpu_testkit::gpu_buffer;
use ornis_wasm2gpu_testkit::map_bytes;
use ornis_wasm2gpu_testkit::record_compute_pass;
use ornis_wasm2gpu_testkit::storage_buffer;

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
const TEXTURE_TEXELS: [u8; 8] = [255, 0, 0, 255, 0, 255, 0, 255];
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
const TEXTURE_ROW_BYTES: u32 = 8;
const TEXTURE_ROWS_PER_IMAGE: u32 = 1;
const GPU_WORD_BYTES: usize = 4;
const SPIRV_VALIDATION_TARGET: &str = "vulkan1.3";

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
                (local $load_x f32) (local $load_y f32)
                (local $load_z f32) (local $load_w f32)
                (local $sample_x f32) (local $sample_y f32)
                (local $sample_z f32) (local $sample_w f32)
                (local $storage_old i32) (local $workgroup_loaded i32)
                (local $workgroup_old i32)
                local.get 0 if return end

                i32.const 0 i32.const 1 i32.const 1 i32.const 0 call $texture_load
                local.set $load_w local.set $load_z
                local.set $load_y local.set $load_x
                i32.const 0 i32.const 3 i32.const 0 local.get $load_x call $storage_write_f32
                i32.const 0 i32.const 3 i32.const 1 local.get $load_y call $storage_write_f32
                i32.const 0 i32.const 3 i32.const 2 local.get $load_z call $storage_write_f32
                i32.const 0 i32.const 3 i32.const 3 local.get $load_w call $storage_write_f32

                i32.const 0 i32.const 1 i32.const 0 i32.const 2
                f32.const 0.75 f32.const 0.5 f32.const 0 call $texture_sample_lod
                local.set $sample_w local.set $sample_z
                local.set $sample_y local.set $sample_x
                i32.const 0 i32.const 3 i32.const 4 local.get $sample_x call $storage_write_f32
                i32.const 0 i32.const 3 i32.const 5 local.get $sample_y call $storage_write_f32
                i32.const 0 i32.const 3 i32.const 6 local.get $sample_z call $storage_write_f32
                i32.const 0 i32.const 3 i32.const 7 local.get $sample_w call $storage_write_f32

                i32.const 0 i32.const 4 i32.const 0 i32.const 5
                call $storage_atomic_add local.set $storage_old
                i32.const 0 i32.const 5 i32.const 0 local.get $storage_old
                call $storage_write_u32

                i32.const 0 i32.const 7 call $workgroup_store
                call $workgroup_barrier
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
    let translated = translate(&wasm()).expect("translate ABI imports");
    assert!(!translated.compute_spirv.is_empty());
    validate_spirv(&translated.compute_spirv);
    let Some((device, queue)) = super::device() else {
        let required = std::env::var_os("ORNIS_REQUIRE_GPU").is_some();
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
        bind_probe(device, queue, &pipeline, [&floats, &atomic, &workgroup]);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wasm2gpu compute ABI probe"),
    });
    record_compute_pass(&mut encoder, &pipeline, &bind_group, DISPATCH_GROUPS);
    queue.submit([encoder.finish()]);
    (
        read_words(device, queue, &floats, FLOAT_OUTPUT_BYTES),
        read_words(device, queue, &workgroup, U32_OUTPUT_BYTES),
        read_words(device, queue, &atomic, ATOMIC_OUTPUT_BYTES),
    )
}

fn bind_probe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: [&wgpu::Buffer; 3],
) -> (
    wgpu::BindGroup,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Sampler,
) {
    let [floats, atomic, workgroup] = buffers;
    let (texture, view) = texture(device, queue);
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

fn texture(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::TextureView) {
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
    queue.write_texture(
        texture.as_image_copy(),
        &TEXTURE_TEXELS,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(TEXTURE_ROW_BYTES),
            rows_per_image: Some(TEXTURE_ROWS_PER_IMAGE),
        },
        wgpu::Extent3d {
            width: TEXTURE_WIDTH,
            height: TEXTURE_HEIGHT,
            depth_or_array_layers: TEXTURE_LAYERS,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn read_words(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Vec<u32> {
    map_bytes(device, queue, buffer, size)
        .chunks_exact(GPU_WORD_BYTES)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte word")))
        .collect()
}

fn validate_spirv(words: &[u32]) {
    let module = rspirv::dr::load_words(words).expect("parse compute ABI SPIR-V");
    let mut validator = spirv_tools::val::create(SPIRV_VALIDATION_TARGET)
        .expect("create SPIR-V validator");
    validator
        .validate(&module)
        .expect("validate compute ABI SPIR-V");
}
