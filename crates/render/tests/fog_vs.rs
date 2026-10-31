//! Second live shader, fog vertex stage, through wasm2gpu.
//!
//! The production fog `vs_main` — fullscreen-quad corner/uv fetch by
//! `vertex_index`, `ornis_render::shaders::fog_generated::
//! wgsl_vertex_source()` — runs VERBATIM in pipeline A. The wasm guest
//! carries the same corner and uv inputs (shared by both sides), is
//! lifted to a compute kernel and writes `{ pos, uv }` per vertex into
//! storage. Pipeline B re-emits those outputs through a thin adapter
//! vertex stage; one shared fragment stage stores the rasterized
//! `position.xy` + `uv` per pixel. The readbacks must be bitwise
//! identical, so the interpolated field is the comparison channel: any
//! divergence between the lifted kernel and the current shader lands
//! in the bits.

use std::borrow::Cow;

use wasm2gpu_testkit::symbols;

use wasm2gpu_testkit::compute_lift::lowered_kernel;
use wasm2gpu_testkit::{device, entry, map_bytes, storage_buffer};

/// Side of the drawn quad in pixels: one fragment per texel of capture.
const SIZE: u32 = 8;

/// Vertices in the fullscreen strip.
const VERTICES: u32 = 4;

/// Bytes of one kernel vertex output (32-byte WGSL struct stride).
const STRIDE: u64 = 32;

/// Shared fragment stage: park what the rasterizer hands it in storage.
/// The `@location(0)` return satisfies wgpu's "a render pipeline needs
/// a color target" rule — nothing reads the attachment back.
const CAPTURE: &str = "
struct QuadOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
@fragment
fn fs_main(input: QuadOut) -> @location(0) vec4<f32> {
    let idx = u32(input.clip_position.y) * 8u + u32(input.clip_position.x);
    pixels[idx] = vec4<f32>(
        input.clip_position.x,
        input.clip_position.y,
        input.uv.x,
        input.uv.y,
    );
    return vec4<f32>(input.uv.x, input.uv.y, 0.0, 1.0);
}
";

/// Pipeline B's vertex stage: re-emit the kernel's per-vertex outputs.
const ADAPTER: &str = "
struct QuadOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
struct KernelVert {
    pos: vec4<f32>,
    uv: vec2<f32>,
}
@group(1) @binding(0) var<storage, read> kernel_out: array<KernelVert>;
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> QuadOut {
    let k = kernel_out[vertex_index];
    return QuadOut(k.pos, k.uv);
}
";

/// Lift the `vs_main` guest export through the shared pinning helper.
fn kernel_wgsl() -> String {
    lowered_kernel("vs_main", "storage_0_0_")
}

/// Render pipeline over two vertex modules (production or adapter) and
/// the shared storage-capturing fragment stage; no color outputs.
fn capture_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    vs: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    let fs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fog-vs capture"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CAPTURE)),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("fog-vs"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: vs,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Copy `words` u32 words out of a GPU buffer and return them.
fn read_words(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    words: u32,
) -> Vec<u32> {
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fog-vs read"),
        size: u64::from(words) * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(src, 0, &read, 0, u64::from(words) * 4);
    queue.submit([encoder.finish()]);
    let bytes = map_bytes(device, &read);
    let mut out = Vec::with_capacity(words as usize);
    let mut index = 0usize;
    while index < words as usize {
        let mut chunk = [0u8; 4];
        chunk.copy_from_slice(&bytes[index * 4..index * 4 + 4]);
        out.push(u32::from_le_bytes(chunk));
        index += 1;
    }
    out
}

/// One storage layout entry at the given visibility.
fn layout_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// One pipeline layout over the given bind-group layouts.
fn pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    first: &wgpu::BindGroupLayout,
    second: Option<&wgpu::BindGroupLayout>,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(first), second],
        immediate_size: 0,
    })
}

/// One bind group binding a single buffer at binding 0.
fn bind_group_for(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[entry(0, buffer)],
    })
}

/// The lifted kernel, the production fog stage, and the adapter module.
fn shader_modules(
    device: &wgpu::Device,
    wgsl: String,
) -> (wgpu::ShaderModule, wgpu::ShaderModule, wgpu::ShaderModule) {
    let kernel = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fog-vs kernel"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl)),
    });
    let production = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fog-vs production"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(
            ornis_render::shaders::fog_generated::wgsl_vertex_source(),
        )),
    });
    let adapter = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fog-vs adapter"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ADAPTER)),
    });
    (kernel, production, adapter)
}

/// The pixel-capture, adapter-out, and kernel-storage group layouts.
fn group_layouts(
    device: &wgpu::Device,
) -> (
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
) {
    let storage = wgpu::BufferBindingType::Storage { read_only: false };
    let pixels = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog-vs pixels"),
        entries: &[layout_entry(0, storage, wgpu::ShaderStages::FRAGMENT)],
    });
    let out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog-vs kernel out"),
        entries: &[layout_entry(
            0,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::ShaderStages::VERTEX,
        )],
    });
    let kernel = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog-vs kernel"),
        entries: &[layout_entry(0, storage, wgpu::ShaderStages::COMPUTE)],
    });
    (pixels, out, kernel)
}

/// Production and adapter render pipelines plus the kernel compute one.
fn build_pipelines(
    device: &wgpu::Device,
    shaders: &(wgpu::ShaderModule, wgpu::ShaderModule, wgpu::ShaderModule),
    layouts: &(
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ),
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::ComputePipeline,
) {
    let (kernel, production, adapter) = shaders;
    let (pixels_layout, out_layout, kernel_layout) = layouts;
    let layout_a = pipeline_layout(device, "fog-vs a", pixels_layout, None);
    let layout_b = pipeline_layout(device, "fog-vs b", pixels_layout, Some(out_layout));
    let pipeline_a = capture_pipeline(device, &layout_a, production);
    let pipeline_b = capture_pipeline(device, &layout_b, adapter);
    let kernel_pl = pipeline_layout(device, "fog-vs kernel", kernel_layout, None);
    let compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fog-vs kernel"),
        layout: Some(&kernel_pl),
        module: kernel,
        entry_point: Some(symbols::CS_MAIN),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline_a, pipeline_b, compute)
}

/// Kernel-output and both pixel-capture buffers, zero-initialized.
fn frame_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let out_bytes = STRIDE * u64::from(VERTICES);
    let pixels_bytes = u64::from(SIZE * SIZE) * 16;
    let kernel_out = storage_buffer(device, "fog-vs kernel out", out_bytes);
    let pixels_a = storage_buffer(device, "fog-vs pixels a", pixels_bytes);
    let pixels_b = storage_buffer(device, "fog-vs pixels b", pixels_bytes);
    queue.write_buffer(&kernel_out, 0, &vec![0u8; out_bytes as usize]);
    queue.write_buffer(&pixels_a, 0, &vec![0u8; pixels_bytes as usize]);
    queue.write_buffer(&pixels_b, 0, &vec![0u8; pixels_bytes as usize]);
    (kernel_out, pixels_a, pixels_b)
}

/// The four bind groups: kernel out, pixel captures, adapter out.
fn frame_groups(
    device: &wgpu::Device,
    layouts: &(
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ),
    buffers: &(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer),
) -> (
    wgpu::BindGroup,
    wgpu::BindGroup,
    wgpu::BindGroup,
    wgpu::BindGroup,
) {
    let (pixels_layout, out_layout, kernel_layout) = layouts;
    let (kernel_out, pixels_a, pixels_b) = buffers;
    let group_kernel = bind_group_for(device, "fog-vs kernel", kernel_layout, kernel_out);
    let group_pixels_a = bind_group_for(device, "fog-vs pixels a", pixels_layout, pixels_a);
    let group_pixels_b = bind_group_for(device, "fog-vs pixels b", pixels_layout, pixels_b);
    let group_out = bind_group_for(device, "fog-vs adapter out", out_layout, kernel_out);
    (group_kernel, group_pixels_a, group_pixels_b, group_out)
}

/// The color target both pipelines rasterize into.
fn render_target(device: &wgpu::Device) -> wgpu::TextureView {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fog-vs target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    target.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Dispatch the lifted kernel, then draw both pipelines over the same
/// viewport in one command buffer.
fn encode_and_submit(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &wgpu::TextureView,
    pipelines: &(
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::ComputePipeline,
    ),
    groups: &(
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
    ),
) {
    let (pipeline_a, pipeline_b, kernel_pipeline) = pipelines;
    let (group_kernel, group_pixels_a, group_pixels_b, group_out) = groups;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fog-vs kernel"),
            timestamp_writes: None,
        });
        pass.set_pipeline(kernel_pipeline);
        pass.set_bind_group(0, group_kernel, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    let draw = wgpu::RenderPassDescriptor {
        label: Some("fog-vs draw"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
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
    };
    {
        let mut pass = encoder.begin_render_pass(&draw);
        pass.set_pipeline(pipeline_a);
        pass.set_bind_group(0, group_pixels_a, &[]);
        pass.draw(0..VERTICES, 0..1);
    }
    {
        let mut pass = encoder.begin_render_pass(&draw);
        pass.set_pipeline(pipeline_b);
        pass.set_bind_group(0, group_pixels_b, &[]);
        pass.set_bind_group(1, group_out, &[]);
        pass.draw(0..VERTICES, 0..1);
    }
    queue.submit([encoder.finish()]);
}

#[test]
fn fog_vertex_kernel_matches_the_production_stage() {
    assert_eq!(SIZE, 8, "CAPTURE's pixel index hardcodes `* 8u` for the SIZE grid");
    let Some((device, queue)) = device() else {
        if std::env::var_os("ORNIS_REQUIRE_GPU").is_none() {
            eprintln!("no adapter; skipping (set ORNIS_REQUIRE_GPU to fail instead)");
            return;
        }
        panic!("ORNIS_REQUIRE_GPU is set but no adapter is available");
    };

    let wgsl = kernel_wgsl();
    let shaders = shader_modules(&device, wgsl);
    let layouts = group_layouts(&device);
    let pipelines = build_pipelines(&device, &shaders, &layouts);
    let buffers = frame_buffers(&device, &queue);
    let groups = frame_groups(&device, &layouts, &buffers);
    let view = render_target(&device);
    encode_and_submit(&device, &queue, &view, &pipelines, &groups);
    let (_kernel_out, pixels_a, pixels_b) = buffers;

    let words = SIZE * SIZE * 4;
    let production_pixels = read_words(&device, &queue, &pixels_a, words);
    let kernel_pixels = read_words(&device, &queue, &pixels_b, words);
    assert!(
        kernel_pixels.iter().any(|word| *word != 0),
        "the lifted kernel wrote nothing"
    );
    let mut index = 0usize;
    while index < production_pixels.len() {
        assert!(
            production_pixels[index] == kernel_pixels[index],
            "pixel word {index} differs: production {:08x} vs kernel {:08x}",
            production_pixels[index],
            kernel_pixels[index],
        );
        index += 1;
    }
}
