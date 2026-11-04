//! GPU-visible results for texture-load and explicit-LOD sampling imports.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;
use ornis_wasm2gpu_testkit::{device, gpu_buffer, map_bytes};

const TEXTURE_WIDTH: u32 = 2;
const TEXTURE_HEIGHT: u32 = 1;
const TARGET_SIZE: u32 = 1;
const ROW_BYTES: u32 = 256;
const EXPECTED_GREEN: [u8; 4] = [0, 255, 0, 255];
const EXPECTED_BLUE: [u8; 4] = [0, 0, 255, 255];
const INPUT_MIP_SIZE: wgpu::Extent3d = wgpu::Extent3d {
    width: 1,
    height: 1,
    depth_or_array_layers: 1,
};
const SAMPLER_LOD_MIN: f32 = 0.0;
const SAMPLER_LOD_MAX: f32 = 1.0;
const SAMPLER_ANISOTROPY_OFF: u16 = 1;

#[test]
fn texture_load_and_sample_lod_return_the_expected_texel_on_gpu() {
    let Some((device, queue)) = device() else {
        assert!(
            !super::gpu_required(),
            "ORNIS_REQUIRE_GPU is set but texture-fetch test has no wgpu device"
        );
        eprintln!("SKIPPED (wasm2gpu texture fetch): no wgpu device");
        return;
    };

    let cases = [
        (
            format!(
                "i32.const 0 i32.const 1 i32.const 1 i32.const 0 call ${}",
                symbols::TEXTURE_LOAD_2D
            ),
            false,
            "textureLoad",
            EXPECTED_GREEN,
        ),
        (
            format!(
                "i32.const 0 i32.const 1 i32.const 0 i32.const 2 \
                 f32.const 0.75 f32.const 0.5 f32.const 1 call ${}",
                symbols::TEXTURE_SAMPLE_2D_LOD
            ),
            true,
            "textureSampleLevel",
            EXPECTED_BLUE,
        ),
    ];

    for (body, needs_sampler, label, expected) in cases {
        let translated = translate(&wasm(&body)).unwrap_or_else(|error| {
            panic!("translate {label} fixture: {error:?}");
        });
        assert_eq!(
            render(&device, &queue, &translated.wgsl, needs_sampler),
            expected,
            "{label} returned the wrong texel"
        );
    }
}

fn import(name: &str, signature: &str) -> String {
    format!(
        "(import \"{}\" \"{}\" (func ${} {}))",
        symbols::MODULE,
        name,
        name,
        signature
    )
}

fn wasm(fragment_body: &str) -> Vec<u8> {
    let imports = format!(
        "{} {}",
        import(
            symbols::TEXTURE_LOAD_2D,
            "(param i32 i32 i32 i32) (result f32 f32 f32 f32)"
        ),
        import(
            symbols::TEXTURE_SAMPLE_2D_LOD,
            "(param i32 i32 i32 i32 f32 f32 f32) (result f32 f32 f32 f32)"
        ),
    );
    wat::parse_str(format!(
        r#"(module {imports}
            (func (export "{vs}") (param f32 f32 f32 f32)
                (result f32 f32 f32 f32 f32 f32)
                local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)
            (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
                {fragment_body}))"#,
        vs = symbols::VS_MAIN,
        fs = symbols::FS_MAIN,
    ))
    .expect("valid texture-fetch Wasm fixture")
}

fn render(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str, needs_sampler: bool) -> [u8; 4] {
    let pipeline = render_pipeline(device, wgsl);
    let (_texture, texture_view, sampler) = input_texture(device, queue);
    let bind_group = texture_bind_group(device, &pipeline, &texture_view, &sampler, needs_sampler);
    let vertices = fullscreen_triangle(device, queue);
    let target = output_target(device);
    draw_and_copy(device, queue, &pipeline, &bind_group, &vertices, &target);
    let bytes = map_bytes(device, &target.readback);
    [bytes[0], bytes[1], bytes[2], bytes[3]]
}

fn render_pipeline(device: &wgpu::Device, wgsl: &str) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wasm2gpu texture fetch"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("wasm2gpu texture fetch"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(symbols::VS_MAIN),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    super::vertex_attr(0, 0),
                    super::vertex_attr(4, 1),
                    super::vertex_attr(8, 2),
                    super::vertex_attr(12, 3),
                ],
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(symbols::FS_MAIN),
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

fn input_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let size = wgpu::Extent3d {
        width: TEXTURE_WIDTH,
        height: TEXTURE_HEIGHT,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("known texture-fetch input"),
        size,
        mip_level_count: 2,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_texture_mip(
        queue,
        &texture,
        0,
        &[255, 0, 0, 255, 0, 255, 0, 255],
        size,
        8,
    );
    // A distinct second mip makes an LOD-1 selection observable in readback.
    write_texture_mip(queue, &texture, 1, &EXPECTED_BLUE, INPUT_MIP_SIZE, 4);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("known texture-fetch sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        lod_min_clamp: SAMPLER_LOD_MIN,
        lod_max_clamp: SAMPLER_LOD_MAX,
        compare: None,
        anisotropy_clamp: SAMPLER_ANISOTROPY_OFF,
        border_color: None,
    });
    (texture, view, sampler)
}

fn write_texture_mip(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    data: &[u8],
    size: wgpu::Extent3d,
    bytes_per_row: u32,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(1),
        },
        size,
    );
}

fn texture_bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    needs_sampler: bool,
) -> wgpu::BindGroup {
    let mut entries = vec![texture_bind_group_entry(
        1,
        wgpu::BindingResource::TextureView(view),
    )];
    if needs_sampler {
        entries.push(texture_bind_group_entry(
            2,
            wgpu::BindingResource::Sampler(sampler),
        ));
    }
    let layout = pipeline.get_bind_group_layout(0);
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wasm2gpu texture fetch"),
        layout: &layout,
        entries: &entries,
    })
}

fn texture_bind_group_entry(
    binding: u32,
    resource: wgpu::BindingResource<'_>,
) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn fullscreen_triangle(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    let vertices = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("texture fetch fullscreen triangle"),
        size: 48,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let vertex_data: [f32; 12] = [
        -1.0, -1.0, 0.0, 0.0, 3.0, -1.0, 0.0, 0.0, -1.0, 3.0, 0.0, 0.0,
    ];
    let bytes = vertex_data
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    queue.write_buffer(&vertices, 0, &bytes);
    vertices
}

struct RenderTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
}

fn output_target(device: &wgpu::Device) -> RenderTarget {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("texture fetch color output"),
        size: wgpu::Extent3d {
            width: TARGET_SIZE,
            height: TARGET_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let readback = gpu_buffer(
        device,
        "texture fetch pixel readback",
        u64::from(ROW_BYTES),
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    RenderTarget {
        texture: target,
        view,
        readback,
    }
}

fn draw_and_copy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    vertices: &wgpu::Buffer,
    target: &RenderTarget,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wasm2gpu texture fetch"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wasm2gpu texture fetch"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
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
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        target.texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &target.readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: TARGET_SIZE,
            height: TARGET_SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
}
