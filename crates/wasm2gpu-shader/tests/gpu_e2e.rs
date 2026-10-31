//! Rust shader → wasm32 → wasm2gpu → wgpu, compared with the host `shade`.
//!
//! The gpu CI job runs this with `ORNIS_REQUIRE_GPU=1` on lavapipe. Without
//! that variable and without an adapter the GPU half is skipped; the wasm
//! compile and the lowerer still run.

#[path = "gpu_e2e/lift.rs"]
mod lift;

#[path = "gpu_e2e/compute.rs"]
mod compute;

#[path = "gpu_e2e/cull.rs"]
mod cull;

#[path = "gpu_e2e/ptr_path.rs"]
mod ptr_path;

use ornis_gpu_abi::HostMemory;
use ornis_gpu_abi::with_host;
use ornis_wasm2gpu::translate;
use ornis_wasm2gpu_shader::STORAGE_LEN;
use ornis_wasm2gpu_shader::ShadeParams;
use ornis_wasm2gpu_shader::shade;
use ornis_wasm2gpu_testkit::{
    compile_wasm, compute_lift, device, entry, gpu_buffer, map_bytes, put_f32, put_u32,
    storage_buffer,
};

const INDEX: u32 = 1;
const INTENSITY: f32 = 0.5;
const SRC: f32 = 0.25;
const EXTRA: f32 = 0.25;

#[test]
fn lowered_shader_matches_the_cpu_reference() {
    let wasm = compile_wasm();
    let lowered = translate(&lift::lift(&wasm)).expect("wasm2gpu");
    let expected = cpu_reference();
    match device() {
        Some((device, queue)) => {
            let (stored, pixel) = run_gpu(&device, &queue, &lowered.wgsl);
            assert!(stored[INDEX as usize].to_bits() == expected.to_bits());
            assert!(stored[0].to_bits() == 0f32.to_bits());
            assert!(
                pixel[3] == 255,
                "fragment did not cover the pixel: {pixel:?}"
            );
            assert!(pixel[0] > 0, "fragment wrote the clear color: {pixel:?}");
        }
        None if std::env::var_os("ORNIS_REQUIRE_GPU").is_some() => {
            panic!("ORNIS_REQUIRE_GPU is set and no wgpu adapter was found");
        }
        None => eprintln!("lowered_shader_matches_the_cpu_reference: no wgpu adapter — skipped"),
    }
}

fn cpu_reference() -> f32 {
    let mut mem = HostMemory::new();
    mem.bind_uniform::<0, 0, ShadeParams>();
    mem.write_uniform::<0, 0, ShadeParams>(&params());
    mem.bind_storage::<0, 1, [f32; STORAGE_LEN as usize]>();
    mem.write_storage::<0, 1, [f32; STORAGE_LEN as usize]>(&src_elems());
    mem.bind_uniform::<0, 2, [ornis_wasm2gpu_shader::UniformF32Lane; STORAGE_LEN as usize]>();
    mem.write_uniform::<0, 2, [ornis_wasm2gpu_shader::UniformF32Lane; STORAGE_LEN as usize]>(
        &lanes(),
    );
    mem.bind_storage::<0, 3, [f32; STORAGE_LEN as usize]>();
    mem.write_storage::<0, 3, [f32; STORAGE_LEN as usize]>(&[0.0; STORAGE_LEN as usize]);
    with_host(&mut mem, || shade(0.0, 0.0))
}

fn params() -> ShadeParams {
    ShadeParams {
        index: INDEX,
        intensity: INTENSITY,
        pad0: 0,
        pad1: 0,
    }
}

fn src_elems() -> [f32; STORAGE_LEN as usize] {
    let mut values = [0.0; STORAGE_LEN as usize];
    values[INDEX as usize] = SRC;
    values
}

fn lanes() -> [ornis_wasm2gpu_shader::UniformF32Lane; STORAGE_LEN as usize] {
    let mut values = [ornis_wasm2gpu_shader::UniformF32Lane {
        value: 0.0,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }; STORAGE_LEN as usize];
    values[INDEX as usize].value = EXTRA;
    values
}

fn run_gpu(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str) -> ([f32; 4], [u8; 4]) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wasm2gpu shade"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = bind_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wasm2gpu"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = render_pipeline(device, &pipeline_layout, &shader);
    let (buffers, bind) = upload(device, queue, &layout);
    draw(device, queue, &pipeline, &bind, &buffers);
    let stored = map_f32s(device, queue, &buffers.out);
    let pixel = map_pixel(device, &buffers.color_read);
    (stored, pixel)
}

struct GpuBuffers {
    out: wgpu::Buffer,
    color: wgpu::Texture,
    color_read: wgpu::Buffer,
}

fn bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = wgpu::BufferBindingType::Storage { read_only: false };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wasm2gpu"),
        entries: &[
            buffer_entry(0, wgpu::BufferBindingType::Uniform),
            buffer_entry(1, storage),
            buffer_entry(2, wgpu::BufferBindingType::Uniform),
            buffer_entry(3, storage),
        ],
    })
}

fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("wasm2gpu"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    vertex_attr(0, 0),
                    vertex_attr(4, 1),
                    vertex_attr(8, 2),
                    vertex_attr(12, 3),
                ],
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn vertex_attr(offset: u64, location: u32) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset,
        shader_location: location,
    }
}

fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> (GpuBuffers, wgpu::BindGroup) {
    let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let params = gpu_buffer(device, "params", 16, uniform);
    let src = storage_buffer(device, "src", 16);
    let extra = gpu_buffer(device, "extra", 64, uniform);
    let out = storage_buffer(device, "out", 16);
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("color"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_read = gpu_buffer(
        device,
        "color-read",
        256,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    queue.write_buffer(&params, 0, &pack_params());
    queue.write_buffer(&src, 0, &pack_src());
    queue.write_buffer(&extra, 0, &pack_lanes());
    queue.write_buffer(&out, 0, &[0u8; 16]);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wasm2gpu"),
        layout,
        entries: &[
            entry(0, &params),
            entry(1, &src),
            entry(2, &extra),
            entry(3, &out),
        ],
    });
    (
        GpuBuffers {
            out,
            color,
            color_read,
        },
        bind,
    )
}

fn pack_params() -> [u8; 16] {
    let mut bytes = [0u8; 16];
    put_u32(&mut bytes, ShadeParams::FIELD_OFFSETS[0], INDEX);
    put_f32(&mut bytes, ShadeParams::FIELD_OFFSETS[1], INTENSITY);
    bytes
}

fn pack_src() -> [u8; 16] {
    let mut bytes = [0u8; 16];
    put_f32(&mut bytes, INDEX * 4, SRC);
    bytes
}

fn pack_lanes() -> [u8; 64] {
    let mut bytes = [0u8; 64];
    put_f32(
        &mut bytes,
        INDEX * <f32 as ornis_gpu_abi::GpuSafeLayout>::UNIFORM_STRIDE,
        EXTRA,
    );
    bytes
}

fn draw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind: &wgpu::BindGroup,
    buffers: &GpuBuffers,
) {
    let verts = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("verts"),
        size: 48,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let xyuv: [f32; 12] = [
        -1.0, -1.0, 0.0, 0.0, 3.0, -1.0, 0.0, 0.0, -1.0, 3.0, 0.0, 0.0,
    ];
    let mut bytes = [0u8; 48];
    let mut index = 0usize;
    while index < xyuv.len() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&xyuv[index].to_le_bytes());
        index += 1;
    }
    queue.write_buffer(&verts, 0, &bytes);
    let view = buffers
        .color
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wasm2gpu"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.set_vertex_buffer(0, verts.slice(..));
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        buffers.color.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffers.color_read,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
}

fn map_f32s(device: &wgpu::Device, queue: &wgpu::Queue, storage: &wgpu::Buffer) -> [f32; 4] {
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out-read"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(storage, 0, &read, 0, 16);
    queue.submit([encoder.finish()]);
    let bytes = map_bytes(device, &read);
    let mut values = [0f32; 4];
    let mut index = 0usize;
    while index < values.len() {
        let mut chunk = [0u8; 4];
        chunk.copy_from_slice(&bytes[index * 4..index * 4 + 4]);
        values[index] = f32::from_le_bytes(chunk);
        index += 1;
    }
    values
}

fn map_pixel(device: &wgpu::Device, read: &wgpu::Buffer) -> [u8; 4] {
    let bytes = map_bytes(device, read);
    [bytes[0], bytes[1], bytes[2], bytes[3]]
}
