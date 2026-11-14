//! Temporal-resolve parity: the translated shader against the raw WGSL.
//!
//! The resolve reads five textures (colour, depth, motion, packed previous
//! depth, history) and writes two colour targets, so the gate renders it twice
//! and compares both attachments.

#[allow(dead_code)]
mod common;

use common::parity::{depth_ramp, gradient, map_readback, mismatched_pixels, upload_rgba8};

/// The raw-WGSL temporal resolve this pass used before the migration.
const DSL_WGSL: &str = include_str!("data/aa_taa_dsl.wgsl");

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Float format of the motion and history layers.
///
/// `16`-bit float is what the pass binds in production *and* the only float
/// format a filterable sampler accepts; a `32`-bit layer would need a
/// non-filtering sampler the shader does not declare.
const FLOAT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Integer format of the packed previous-depth layer.
const PACKED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const SIZE: (u32, u32) = (64, 64);
/// `R16Uint` levels the pack divides by.
const PACKED_SCALE: f32 = 65534.0;

#[test]
fn translated_taa_matches_the_raw_wgsl_pixels() {
    let color = gradient(SIZE.0, SIZE.1, |x, y| {
        [
            ((x * 4) as u8).wrapping_add(if x % 7 == 0 { 60 } else { 0 }),
            (y * 4) as u8,
            ((x + y) * 2) as u8,
            255,
        ]
    });
    // Motion: a small shift, so the reprojection stays on screen.
    let motion: Vec<f32> = (0..SIZE.0 * SIZE.1)
        .flat_map(|index| {
            let x = index % SIZE.0;
            [(x as f32 / 16.0) * 0.004, 0.0, 0.0, 1.0]
        })
        .collect();
    // History: the colour layer shifted, with a depth word matching the ramp.
    let history: Vec<f32> = (0..SIZE.0 * SIZE.1)
        .flat_map(|index| {
            let y = index / SIZE.0;
            let depth = 0.2 + 0.7 * (y as f32 / (SIZE.1 - 1) as f32);
            [0.6, 0.3, 0.1, depth]
        })
        .collect();
    let packed: Vec<u32> = (0..SIZE.0 * SIZE.1)
        .map(|index: u32| {
            let y = index / SIZE.0;
            let depth = 0.2 + 0.7 * (y as f32 / (SIZE.1 - 1) as f32);
            (depth * PACKED_SCALE) as u32
        })
        .collect();
    let Some((device, queue)) = common::parity::device_pair() else {
        return;
    };
    let color_view = upload_rgba8(&device, &queue, "taa color", SIZE, &color);
    let depth_view = depth_ramp(&device, &queue, SIZE);
    let motion_view = upload_f32(&device, &queue, "taa motion", FLOAT_FORMAT, &motion);
    let history_view = upload_f32(&device, &queue, "taa history", FLOAT_FORMAT, &history);
    let packed_view = upload_u32(&device, &queue, &packed);
    let sampler = common::parity::linear_sampler(&device);
    let translated = ornis_render::shaders::aa_taa_generated::wgsl_source();
    for (feedback, radius, history_valid) in
        [(0.9f32, 1.0f32, 1.0f32), (0.5, 2.0, 1.0), (0.9, 1.0, 0.0)]
    {
        let params = create_params(&device, &queue, feedback, radius, history_valid);
        let inputs = Inputs {
            color: &color_view,
            depth: &depth_view,
            motion: &motion_view,
            packed: &packed_view,
            history: &history_view,
            sampler: &sampler,
            params: &params,
        };
        let reference = render(&device, &queue, DSL_WGSL, &inputs);
        let new = render(&device, &queue, &translated, &inputs);
        for (target, (a, b)) in reference.iter().zip(new.iter()).enumerate() {
            let mismatches = mismatched_pixels(a, b);
            assert_eq!(
                mismatches, 0,
                "target {target} (feedback {feedback}, radius {radius}, valid {history_valid}): \
                 {mismatches} pixels differ"
            );
        }
    }
}

/// Everything the resolve reads.
struct Inputs<'a> {
    color: &'a wgpu::TextureView,
    depth: &'a wgpu::TextureView,
    motion: &'a wgpu::TextureView,
    packed: &'a wgpu::TextureView,
    history: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    params: &'a wgpu::Buffer,
}

/// `AaParams` in the host layout: feedback, threshold, clamp radius, history
/// flag, mode, pad.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AaParamsBuffer {
    feedback: f32,
    threshold: f32,
    clamp_radius: f32,
    history_valid: f32,
    mode: u32,
    pad: [f32; 3],
}

/// Upload one `AaParams`.
fn create_params(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    feedback: f32,
    radius: f32,
    history_valid: f32,
) -> wgpu::Buffer {
    let params = AaParamsBuffer {
        feedback,
        threshold: 0.05,
        clamp_radius: radius,
        history_valid,
        mode: 0,
        pad: [0.0; 3],
    };
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("taa params"),
        size: std::mem::size_of::<AaParamsBuffer>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&params));
    buffer
}

/// Upload one float layer.
fn upload_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    format: wgpu::TextureFormat,
    pixels: &[f32],
) -> wgpu::TextureView {
    let bytes: Vec<u8> = pixels.iter().flat_map(|value| half(*value)).collect();
    upload_texels(
        device,
        queue,
        &Layer {
            label,
            format,
            bytes: &bytes,
            texel: 8,
        },
    )
}

/// Upload one packed-depth layer.
fn upload_u32(device: &wgpu::Device, queue: &wgpu::Queue, values: &[u32]) -> wgpu::TextureView {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    upload_texels(
        device,
        queue,
        &Layer {
            label: "taa packed depth",
            format: PACKED_FORMAT,
            bytes: &bytes,
            texel: 4,
        },
    )
}

/// IEEE-754 binary16 encoding of `value` (round to nearest even).
fn half(value: f32) -> [u8; 2] {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = bits & 0x007F_FFFF;
    let encoded = if exponent <= 0 {
        sign
    } else if exponent >= 0x1F {
        sign | 0x7C00
    } else {
        sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
    };
    encoded.to_le_bytes()
}

/// One uploaded layer: its label, format, bytes, and texel size.
struct Layer<'a> {
    /// Texture label.
    label: &'a str,
    /// Texel format.
    format: wgpu::TextureFormat,
    /// Raw texels, row-major.
    bytes: &'a [u8],
    /// Bytes per texel.
    texel: u32,
}

/// Upload raw texels and return the view.
fn upload_texels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layer: &Layer<'_>,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(layer.label),
        size: wgpu::Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: layer.format,
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
        layer.bytes,
        wgpu::TexelCopyBufferLayout {
            bytes_per_row: Some(SIZE.0 * layer.texel),
            ..Default::default()
        },
        wgpu::Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            ..Default::default()
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// The pipeline both shaders run through: one vertex entry, two targets.
fn pipeline(device: &wgpu::Device, shader: &wgpu::ShaderModule) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("taa parity"),
        layout: None,
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[color_target(), color_target()],
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

/// One colour attachment of the resolve.
fn color_target() -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState {
        format: FORMAT,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })
}

/// Every texture and uniform the resolve reads, bound once.
fn bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    inputs: &Inputs<'_>,
) -> wgpu::BindGroup {
    let entries = [
        (0, wgpu::BindingResource::TextureView(inputs.color)),
        (1, wgpu::BindingResource::Sampler(inputs.sampler)),
        (2, wgpu::BindingResource::TextureView(inputs.depth)),
        (3, wgpu::BindingResource::TextureView(inputs.motion)),
        (4, wgpu::BindingResource::TextureView(inputs.packed)),
        (5, wgpu::BindingResource::TextureView(inputs.history)),
        (6, inputs.params.as_entire_binding()),
    ];
    let entries: Vec<wgpu::BindGroupEntry<'_>> = entries
        .into_iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("taa parity"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

/// Render both colour attachments and return their pixels.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    inputs: &Inputs<'_>,
) -> [Vec<u8>; 2] {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("taa parity"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = pipeline(device, &shader);
    let bind = bind_group(device, &pipeline, inputs);
    let targets: Vec<wgpu::Texture> = (0..2)
        .map(|_| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("taa parity target"),
                size: wgpu::Extent3d {
                    width: SIZE.0,
                    height: SIZE.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        })
        .collect();
    let views: Vec<wgpu::TextureView> = targets
        .iter()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
        .collect();
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let attachments: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = views
            .iter()
            .map(|view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })
            })
            .collect();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("taa parity"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..4, 0..1);
    }
    read_targets(device, queue, encoder, &targets)
}
/// Copy both colour targets into one buffer and split the pixels.
fn read_targets(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    targets: &[wgpu::Texture],
) -> [Vec<u8>; 2] {
    const ROW_ALIGN: u32 = 256;
    let unpadded = SIZE.0 * 4;
    let padded = unpadded.div_ceil(ROW_ALIGN) * ROW_ALIGN;
    let plane = padded * SIZE.1;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("taa parity readback"),
        size: u64::from(plane * 2),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    for (index, target) in targets.iter().enumerate() {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: u64::from(plane * index as u32),
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(SIZE.1),
                },
            },
            wgpu::Extent3d {
                width: SIZE.0,
                height: SIZE.1,
                depth_or_array_layers: 1,
            },
        );
    }
    queue.submit([encoder.finish()]);
    let data = map_readback(device, &readback);
    let mut planes = [Vec::new(), Vec::new()];
    for (index, plane_pixels) in planes.iter_mut().enumerate() {
        plane_pixels.resize((unpadded * SIZE.1) as usize, 0);
        let base = plane as usize * index;
        for row in 0..SIZE.1 as usize {
            plane_pixels[row * unpadded as usize..][..unpadded as usize]
                .copy_from_slice(&data[base + row * padded as usize..][..unpadded as usize]);
        }
    }
    readback.unmap();
    planes
}
