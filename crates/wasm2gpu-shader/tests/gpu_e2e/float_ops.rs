//! Executable value coverage for the scalar float operators lowered to WGSL.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;

use super::device;
use super::entry;
use super::gpu_buffer;
use super::map_bytes;
use super::record_compute_pass;
use super::storage_buffer;

const FLOAT_EXPECTED: [f32; 7] = [9.0, 5.0, 14.0, 4.5, 3.25, -3.25, 4_294_967_296.0];
const COMPARISON_EXPECTED: [u32; 12] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];

fn comparisons() -> String {
    let tests = [
        (0, 1.0, 1.0, "eq"),
        (1, 1.0, 2.0, "eq"),
        (2, 1.0, 2.0, "ne"),
        (3, 2.0, 2.0, "ne"),
        (4, 1.0, 2.0, "lt"),
        (5, 2.0, 1.0, "lt"),
        (6, 1.0, 1.0, "le"),
        (7, 2.0, 1.0, "le"),
        (8, 2.0, 1.0, "gt"),
        (9, 1.0, 2.0, "gt"),
        (10, 2.0, 2.0, "ge"),
        (11, 1.0, 2.0, "ge"),
    ];
    let mut body = String::new();
    for (index, left, right, op) in tests {
        body.push_str(&format!(
            concat!(
                "i32.const 0 i32.const 1 i32.const {index} ",
                "f32.const {left} f32.const {right} f32.{op} call $store_u32\n"
            ),
            index = index,
            left = left,
            right = right,
            op = op,
        ));
    }
    body
}

fn wasm() -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
            (import "{module}" "{store_f32}" (func $store_f32 (param i32 i32 i32 f32)))
            (import "{module}" "{store_u32}" (func $store_u32 (param i32 i32 i32 i32)))
            (func (export "{vs}") (param f32 f32 f32 f32)
                (result f32 f32 f32 f32 f32 f32)
                local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)
            (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
                f32.const 0 f32.const 0 f32.const 0 f32.const 1)
            (func (export "{cs}") (param i32)
                ;; Run one lane so output writes have no data race.
                local.get 0 if return end
                i32.const 0 i32.const 0 i32.const 0
                f32.const 7 f32.const 2 f32.add call $store_f32
                i32.const 0 i32.const 0 i32.const 1
                f32.const 7 f32.const 2 f32.sub call $store_f32
                i32.const 0 i32.const 0 i32.const 2
                f32.const 7 f32.const 2 f32.mul call $store_f32
                i32.const 0 i32.const 0 i32.const 3
                f32.const 9 f32.const 2 f32.div call $store_f32
                i32.const 0 i32.const 0 i32.const 4
                f32.const -3.25 f32.abs call $store_f32
                i32.const 0 i32.const 0 i32.const 5
                f32.const 3.25 f32.neg call $store_f32
                i32.const 0 i32.const 0 i32.const 6
                i32.const -1 f32.convert_i32_u call $store_f32
                {comparisons}))"#,
        module = symbols::MODULE,
        store_f32 = symbols::STORAGE_WRITE_F32,
        store_u32 = symbols::STORAGE_WRITE_U32,
        vs = symbols::VS_MAIN,
        fs = symbols::FS_MAIN,
        cs = symbols::CS_MAIN,
        comparisons = comparisons(),
    ))
    .expect("valid scalar-float wasm")
}

#[test]
fn lowered_scalar_float_ops_match_expected_gpu_values() {
    // Keep runtime assertions finite; non-finite constants are validated by
    // the translator test, while their exact GPU results vary by backend.
    let translated = translate(&wasm()).expect("translate scalar float operators");
    let Some((device, queue)) = device() else {
        let required =
            std::env::var("ORNIS_REQUIRE_GPU").is_ok_and(|value| !value.is_empty() && value != "0");
        assert!(
            !required,
            "ORNIS_REQUIRE_GPU is set but scalar-float test has no wgpu device"
        );
        eprintln!("SKIPPED (wasm2gpu scalar floats): no wgpu device");
        return;
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wasm2gpu scalar float operators"),
        source: wgpu::ShaderSource::Wgsl(translated.wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wasm2gpu scalar float operators"),
        layout: None,
        module: &shader,
        entry_point: Some(symbols::CS_MAIN),
        compilation_options: Default::default(),
        cache: None,
    });
    let float_bytes = (FLOAT_EXPECTED.len() * std::mem::size_of::<f32>()) as u64;
    let comparison_bytes = (COMPARISON_EXPECTED.len() * std::mem::size_of::<u32>()) as u64;
    let float_output = storage_buffer(&device, "scalar-float-output", float_bytes);
    let comparison_output = storage_buffer(&device, "float-comparison-output", comparison_bytes);
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wasm2gpu scalar float operators"),
        layout: &layout,
        entries: &[entry(0, &float_output), entry(1, &comparison_output)],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wasm2gpu scalar float operators"),
    });
    record_compute_pass(&mut encoder, &pipeline, &bind_group, [1, 1, 1]);
    queue.submit([encoder.finish()]);

    let floats: Vec<f32> = readback_words(&device, &queue, &float_output, float_bytes)
        .into_iter()
        .map(f32::from_bits)
        .collect();
    assert_eq!(floats.as_slice(), FLOAT_EXPECTED.as_slice());
    let comparisons = readback_words(&device, &queue, &comparison_output, comparison_bytes);
    assert_eq!(comparisons.as_slice(), COMPARISON_EXPECTED.as_slice());
}

fn readback_words(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Vec<u32> {
    readback(device, queue, buffer, size)
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte word")))
        .collect()
}

fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Vec<u8> {
    let staging = gpu_buffer(
        device,
        "wasm2gpu scalar-float readback",
        size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wasm2gpu scalar-float readback"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    map_bytes(device, &staging)
}
