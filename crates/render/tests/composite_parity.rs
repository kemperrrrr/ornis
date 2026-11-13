//! Composite pass parity: the translated shader against the DSL one it replaces.
//!
//! The pass used to be assembled from `#[stage]` sources at runtime; its WGSL is
//! frozen in `data/composite_dsl.wgsl` so the Rust kernel
//! (`crates/gpu-composite`) can be compared with it pixel for pixel. Both
//! shaders read the same two textures through the same samplers, so a pixel
//! difference is a semantics difference, not a sampling one.

#[allow(dead_code)]
mod common;

use common::sized::{FrameSpec, render_and_readback};

/// The DSL shader this pass was assembled from before the migration.
const DSL_WGSL: &str = include_str!("data/composite_dsl.wgsl");

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: (u32, u32) = (64, 64);

#[test]
fn translated_composite_matches_the_dsl_pixels() {
    let Some((_, device, queue)) = common::adapter_or_skip() else {
        return;
    };
    let translated = ornis_render::shaders::composite_generated::wgsl_source();
    // A scene texel and a UI texel per channel, including both sides of the
    // sRGB cutoff and the alpha ramp the pass mixes by.
    let pbr = gradient(|x, y| [(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8, 255]);
    let ui = gradient(|x, y| {
        // Below the linear cutoff (0.04045 → 10), at it, and well above it.
        let level = [0u8, 8, 10, 11, 128, 255][(x + y) as usize % 6];
        [level, level / 2, 255 - level, (x * 4) as u8]
    });
    let dsl = render(&device, &queue, DSL_WGSL, &pbr, &ui);
    let new = render(&device, &queue, &translated, &pbr, &ui);
    assert_eq!(dsl.len(), new.len(), "read-back size");
    let mismatches = dsl
        .as_chunks::<4>()
        .0
        .iter()
        .zip(new.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a.iter().zip(b.iter()).any(|(&a, &b)| a.abs_diff(b) > 1))
        .count();
    assert_eq!(
        mismatches, 0,
        "{mismatches} pixels differ from the DSL pass"
    );
}

/// One `SIZE`×`SIZE` RGBA8 image from a per-texel function.
fn gradient(fill: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((SIZE.0 * SIZE.1 * 4) as usize);
    for y in 0..SIZE.1 {
        for x in 0..SIZE.0 {
            pixels.extend_from_slice(&fill(x, y));
        }
    }
    pixels
}

/// Draw the fullscreen quad through `wgsl` and return the read-back pixels.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    pbr: &[u8],
    ui: &[u8],
) -> Vec<u8> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("composite parity"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("composite parity"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: FORMAT,
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
    });
    let pbr_view = upload(device, queue, "pbr", pbr);
    let ui_view = upload(device, queue, "ui", ui);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("composite parity"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite parity"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&pbr_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&ui_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    render_and_readback(
        device,
        queue,
        &FrameSpec {
            size: SIZE,
            format: FORMAT,
        },
        |encoder, target| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..4, 0..1);
        },
    )
}

/// Upload one RGBA8 image and return its sampled view.
fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    pixels: &[u8],
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
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
            bytes_per_row: Some(SIZE.0 * 4),
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
