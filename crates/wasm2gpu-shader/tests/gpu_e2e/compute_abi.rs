//! Execute the new ABI imports after a real wasm32 build and lowering pass.

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
const U32_OUTPUT: [u32; 2] = [10, 7];
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

#[test]
fn wasm_compute_abi_imports_execute_on_gpu() {
    let wasm = super::compile_wasm();
    let module = super::compute_lift::lift(&wasm, "abi_probe");
    let translated = translate(&module).expect("translate ABI probe");
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
    assert_eq!(&workgroup[..2], U32_OUTPUT.as_slice());
    assert_eq!(workgroup[3].wrapping_sub(workgroup[2]), WORKGROUP_ATOMIC_DELTA);
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
        bind_probe(device, queue, &pipeline, &floats, &atomic, &workgroup);
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
    floats: &wgpu::Buffer,
    atomic: &wgpu::Buffer,
    workgroup: &wgpu::Buffer,
) -> (
    wgpu::BindGroup,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Sampler,
) {
    let (texture, view) = texture(device, queue);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
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
    let size = wgpu::Extent3d {
        width: TEXTURE_WIDTH,
        height: TEXTURE_HEIGHT,
        depth_or_array_layers: TEXTURE_LAYERS,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("wasm2gpu ABI probe texture"),
        size,
        mip_level_count: TEXTURE_MIP_LEVELS,
        sample_count: TEXTURE_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &TEXTURE_TEXELS,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(TEXTURE_ROW_BYTES),
            rows_per_image: Some(TEXTURE_ROWS_PER_IMAGE),
        },
        size,
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
    let staging = gpu_buffer(
        device,
        "wasm2gpu ABI readback",
        size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    map_bytes(device, &staging)
        .chunks_exact(GPU_WORD_BYTES)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte word")))
        .collect()
}

fn validate_spirv(words: &[u32]) {
    let mut bytes = Vec::with_capacity(words.len() * GPU_WORD_BYTES);
    for word in words {
        bytes.extend(word.to_le_bytes());
    }
    let path = std::env::temp_dir().join(format!("ornis-abi-probe-{}.spv", std::process::id()));
    std::fs::write(&path, bytes).expect("write compute SPIR-V");
    let result = std::process::Command::new("spirv-val")
        .args(["--target-env", SPIRV_VALIDATION_TARGET])
        .arg(&path)
        .output();
    let _ = std::fs::remove_file(&path);
    match result {
        Ok(output) => assert!(
            output.status.success(),
            "spirv-val rejected workgroup-atomic output:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            assert!(
                std::env::var_os("ORNIS_REQUIRE_SPIRV_VAL").is_none(),
                "ORNIS_REQUIRE_SPIRV_VAL is set but spirv-val is not installed"
            );
            eprintln!("SKIPPED (SPIR-V validation): spirv-val is not installed");
        }
        Err(error) => panic!("failed to run spirv-val: {error}"),
    }
}
