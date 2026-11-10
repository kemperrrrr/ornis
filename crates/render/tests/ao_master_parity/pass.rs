//! Draw one AO fragment shader over the synthetic g-buffer and read RGBA8.

use super::scene::{self, H, W};
use crate::common::sized::{FrameSpec, render_and_readback};

const VERTEX: &str = r#"
struct QuadVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> QuadVertexOutput {
    var pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    var uv = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );
    return QuadVertexOutput(vec4<f32>(pos[vertex_index], 0.0, 1.0), uv[vertex_index]);
}
"#;

const BLIT: &str = r#"
struct QuadVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var depth_src: texture_2d<f32>;
@group(0) @binding(1) var normal_src: texture_2d<f32>;

struct FsOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_blit(input: QuadVertexOutput) -> FsOut {
    let dims = vec2<f32>(textureDimensions(depth_src));
    let x = min(u32(input.uv.x * dims.x), u32(dims.x) - 1u);
    let y = min(u32(input.uv.y * dims.y), u32(dims.y) - 1u);
    let coord = vec2<i32>(i32(x), i32(y));
    return FsOut(textureLoad(depth_src, coord, 0).r, textureLoad(normal_src, coord, 0));
}
"#;

struct Gpu<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
}

/// G-buffer views the raw kernels load. Owned textures stay with the caller.
pub struct GBuffer {
    /// Single-sample or 4x depth.
    pub depth: wgpu::TextureView,
    /// Single-sample or 4x octahedral normal.
    pub normal: wgpu::TextureView,
}

/// Uploaded single-sample depth and +Z normals.
pub struct Sources {
    _depth: wgpu::Texture,
    _normal: wgpu::Texture,
    depth_view: wgpu::TextureView,
    normal_view: wgpu::TextureView,
}

impl Sources {
    fn upload(gpu: &Gpu<'_>, depth: &[f32]) -> Self {
        let depth_tex = texture(
            gpu.device,
            "ao depth src",
            wgpu::TextureFormat::R32Float,
            1,
            false,
        );
        let normal_tex = texture(
            gpu.device,
            "ao normal src",
            wgpu::TextureFormat::Rgba16Float,
            1,
            false,
        );
        write_f32(gpu.queue, &depth_tex, depth);
        write_normals(gpu.queue, &normal_tex);
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let normal_view = normal_tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _depth: depth_tex,
            _normal: normal_tex,
            depth_view,
            normal_view,
        }
    }
}

/// RGBA8 AO of `fragment` over `shot`'s depth. `multisampled` blits the
/// g-buffer to 4x and runs the multisampled kernel.
pub fn draw_raw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shot: &scene::Shot,
    fragment: &str,
    multisampled: bool,
) -> Vec<u8> {
    let gpu = Gpu { device, queue };
    let sources = Sources::upload(&gpu, &shot.depth);
    let samples = if multisampled { 4 } else { 1 };
    let gbuffer = blit_depth(&gpu, &sources, samples);
    let camera = uniform(device, "ao camera", &scene::camera_bytes(shot));
    let ao = uniform(device, "ao params", &scene::ao_bytes(shot));
    draw_color(
        &gpu,
        fragment,
        &[
            view_entry(0, &gbuffer.depth),
            view_entry(1, &gbuffer.normal),
            buffer_entry(2, &camera),
            buffer_entry(3, &ao),
        ],
    )
}

/// RGBA8 blurred factor. `ao_rgba` is the raw factor the sampler reads.
pub fn draw_blur(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shot: &scene::Shot,
    fragment: &str,
    ao_rgba: &[u8],
) -> Vec<u8> {
    let gpu = Gpu { device, queue };
    let sources = Sources::upload(&gpu, &shot.depth);
    let gbuffer = blit_depth(&gpu, &sources, 1);
    let ao_tex = texture(device, "ao raw", wgpu::TextureFormat::Rgba8Unorm, 1, false);
    write_rgba(queue, &ao_tex, ao_rgba);
    let ao_view = ao_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ao blur"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let camera = uniform(device, "ao camera", &scene::camera_bytes(shot));
    let ao = uniform(device, "ao params", &scene::ao_bytes(shot));
    draw_color(
        &gpu,
        fragment,
        &[
            view_entry(0, &ao_view),
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            view_entry(2, &gbuffer.depth),
            buffer_entry(3, &ao),
            buffer_entry(4, &camera),
        ],
    )
}

fn draw_color(gpu: &Gpu<'_>, fragment: &str, entries: &[wgpu::BindGroupEntry<'_>]) -> Vec<u8> {
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let vs = module(gpu.device, "ao vs", VERTEX);
    let fs = module(gpu.device, "ao fs", fragment);
    let pipeline = color_pipeline(gpu.device, &vs, &fs, 1, None);
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ao bind"),
        layout: &pipeline.get_bind_group_layout(0),
        entries,
    });
    let pixels = render_and_readback(gpu.device, gpu.queue, &frame(), |encoder, view| {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ao pass"),
            color_attachments: &[Some(color_attachment(view))],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..4, 0..1);
    });
    assert_no_error(scope, "ao draw");
    pixels
}

fn blit_depth(gpu: &Gpu<'_>, sources: &Sources, samples: u32) -> GBuffer {
    let depth = texture(
        gpu.device,
        "ao depth",
        wgpu::TextureFormat::Depth32Float,
        samples,
        true,
    );
    let normal = texture(
        gpu.device,
        "ao normal",
        wgpu::TextureFormat::Rgba16Float,
        samples,
        true,
    );
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    let normal_view = normal.create_view(&wgpu::TextureViewDescriptor::default());
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let vs = module(gpu.device, "blit vs", VERTEX);
    let fs = module(gpu.device, "blit fs", BLIT);
    let pipeline = color_pipeline(
        gpu.device,
        &vs,
        &fs,
        samples,
        Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
    );
    let layout = pipeline.get_bind_group_layout(0);
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit bind"),
        layout: &layout,
        entries: &[
            view_entry(0, &sources.depth_view),
            view_entry(1, &sources.normal_view),
        ],
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit 4x"),
            color_attachments: &[Some(color_attachment(&normal_view))],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..4, 0..1);
    }
    gpu.queue.submit([encoder.finish()]);
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("blit poll");
    assert_no_error(scope, "msaa blit");
    GBuffer {
        depth: depth_view,
        normal: normal_view,
    }
}

fn color_pipeline(
    device: &wgpu::Device,
    vs: &wgpu::ShaderModule,
    fs: &wgpu::ShaderModule,
    samples: u32,
    depth: Option<wgpu::DepthStencilState>,
) -> wgpu::RenderPipeline {
    let entry = if depth.is_some() {
        "fs_blit"
    } else {
        "fs_main"
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ao pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: vs,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: fs,
            entry_point: Some(entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: if depth.is_some() {
                    wgpu::TextureFormat::Rgba16Float
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                },
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: depth,
        multisample: wgpu::MultisampleState {
            count: samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    })
}

fn color_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: wgpu::StoreOp::Store,
        },
    }
}

fn frame() -> FrameSpec {
    FrameSpec {
        size: (W, H),
        format: wgpu::TextureFormat::Rgba8Unorm,
    }
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    samples: u32,
    render_target: bool,
) -> wgpu::Texture {
    let usage = if render_target {
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT
    } else {
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
    };
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn uniform(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("uniform map");
        mapped.copy_from_slice(bytes);
    }
    buffer.unmap();
    buffer
}

fn write_f32(queue: &wgpu::Queue, texture: &wgpu::Texture, pixels: &[f32]) {
    let mut bytes = Vec::with_capacity(pixels.len() * 4);
    for value in pixels {
        bytes.extend(value.to_le_bytes());
    }
    write_bytes(queue, texture, &bytes, W * 4);
}

fn write_normals(queue: &wgpu::Queue, texture: &wgpu::Texture) {
    let texel = [0u8, 0, 0, 0, 0, 0, 0x00, 0x3C];
    let mut bytes = Vec::with_capacity((W * H) as usize * 8);
    for _ in 0..(W * H) {
        bytes.extend(texel);
    }
    write_bytes(queue, texture, &bytes, W * 8);
}

fn write_rgba(queue: &wgpu::Queue, texture: &wgpu::Texture, pixels: &[u8]) {
    write_bytes(queue, texture, pixels, W * 4);
}

fn write_bytes(queue: &wgpu::Queue, texture: &wgpu::Texture, bytes: &[u8], row: u32) {
    queue.write_texture(
        texture.as_image_copy(),
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
}

fn module(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn assert_no_error(scope: wgpu::ErrorScopeGuard, what: &str) {
    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("{what}: {error}");
    }
}
