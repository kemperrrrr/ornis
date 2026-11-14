//! Shared setup for the translated-shader pixel-parity gates.
//!
//! Both gates render the frozen DSL shader and the kernel-crate shader into one
//! target with identical inputs, so the setup lives here once: one image
//! generator, one uploader. The gates differ only in which textures, uniforms
//! and modes they feed.

#![allow(dead_code)]

/// Linear clamp sampler, the filter every gate binds.
pub fn linear_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("parity sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// A headless device and queue, or `None` when the host has no adapter.
///
/// One line per gate: the gates differ in what they draw, not in how they open
/// the device.
pub fn device_pair() -> Option<(wgpu::Device, wgpu::Queue)> {
    crate::common::adapter_or_skip().map(|(_, device, queue)| (device, queue))
}

/// One `width`×`height` RGBA8 image from a per-texel function.
pub fn gradient(width: u32, height: u32, fill: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.extend_from_slice(&fill(x, y));
        }
    }
    pixels
}

/// Upload one RGBA8 image and return its sampled view.
pub fn upload_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    size: (u32, u32),
    pixels: &[u8],
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
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
        pixels,
        wgpu::TexelCopyBufferLayout {
            bytes_per_row: Some(size.0 * 4),
            ..Default::default()
        },
        wgpu::Extent3d {
            width: size.0,
            height: size.1,
            ..Default::default()
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Depth-only WGSL that fills a target with a near-to-far ramp.
///
/// A depth texture cannot be uploaded (`write_texture` refuses depth formats),
/// so the gates that read depth render it: `frag_depth` runs from [`NEAR`] at
/// the top of the frame to [`FAR`] at the bottom.
pub const DEPTH_RAMP: &str = r#"
struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) t: f32,
}

@vertex
fn vs_depth(@builtin(vertex_index) index: u32) -> Varyings {
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let corner = corners[index];
    var out: Varyings;
    out.position = vec4<f32>(corner, 0.0, 1.0);
    out.t = corner.y * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_depth(input: Varyings) -> @builtin(frag_depth) f32 {
    return mix(NEAR_DEPTH, FAR_DEPTH, input.t);
}
"#;

/// Depth of the top of the ramp.
pub const NEAR_DEPTH: f32 = 0.2;
/// Depth of the bottom of the ramp.
pub const FAR_DEPTH: f32 = 0.9;

/// Render a `size`-square `Depth32Float` texture and return its sampled view.
pub fn depth_ramp(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: (u32, u32),
) -> wgpu::TextureView {
    let formatted = DEPTH_RAMP
        .replace("NEAR_DEPTH", &format!("{NEAR_DEPTH:?}"))
        .replace("FAR_DEPTH", &format!("{FAR_DEPTH:?}"));
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("depth ramp"),
        source: wgpu::ShaderSource::Wgsl(formatted.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("depth ramp"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_depth"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_depth"),
            targets: &[],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth ramp"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("depth ramp"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &view,
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
        pass.draw(0..4, 0..1);
    }
    queue.submit([encoder.finish()]);
    view
}

/// Map `buffer` for reading and return its bytes.
///
/// The no-callback form [`sized`](super::sized) and `ibl_split_sum` use: polling
/// until the map completes keeps the readback to one shape across the crate.
pub fn map_readback(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll readback");
    // Dropping the mapped range is what unmaps the buffer in wgpu 30.
    slice.get_mapped_range().expect("mapped range").to_vec()
}

/// Count the pixels of two read-backs that differ by more than one 8-bit step.
pub fn mismatched_pixels(reference: &[u8], candidate: &[u8]) -> usize {
    assert_eq!(reference.len(), candidate.len(), "read-back size");
    reference
        .as_chunks::<4>()
        .0
        .iter()
        .zip(candidate.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a.iter().zip(b.iter()).any(|(&a, &b)| a.abs_diff(b) > 1))
        .count()
}
