//! Composite pass parity: the translated shader against the DSL one it replaces.
//!
//! The pass used to be assembled from `#[stage]` sources at runtime; its WGSL is
//! frozen in `data/composite_dsl.wgsl` so the Rust kernel
//! (`crates/gpu-composite`) can be compared with it pixel for pixel. Both
//! shaders read the same two textures through the same samplers, so a pixel
//! difference is a semantics difference, not a sampling one.

#[allow(dead_code)]
mod common;

use common::parity::{gradient, linear_sampler, mismatched_pixels, upload_rgba8};
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
    let pbr = gradient(SIZE.0, SIZE.1, |x, y| {
        [(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8, 255]
    });
    let ui = gradient(SIZE.0, SIZE.1, |x, y| {
        // Below the linear cutoff (0.04045 → 10), at it, and well above it.
        let level = [0u8, 8, 10, 11, 128, 255][(x + y) as usize % 6];
        [level, level / 2, 255 - level, (x * 4) as u8]
    });
    let dsl = render(&device, &queue, DSL_WGSL, &pbr, &ui);
    let new = render(&device, &queue, &translated, &pbr, &ui);
    let mismatches = mismatched_pixels(&dsl, &new);
    assert_eq!(
        mismatches, 0,
        "{mismatches} pixels differ from the DSL pass"
    );
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
    let pbr_view = upload_rgba8(device, queue, "pbr", SIZE, pbr);
    let ui_view = upload_rgba8(device, queue, "ui", SIZE, ui);
    let sampler = linear_sampler(device);
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
