//! Edge-filter parity: the translated FXAA/SMAA shader against the raw WGSL.
//!
//! The filter reads one layer and a uniform; the mode word picks the blend, so
//! each mode is compared over an image with edges in both axes.

#[allow(dead_code)]
mod common;

use common::parity::{gradient, linear_sampler, mismatched_pixels, upload_rgba8};
use common::sized::{FrameSpec, render_and_readback};

/// The raw-WGSL edge filter this pass used before the migration.
const DSL_WGSL: &str = include_str!("data/aa_edge_dsl.wgsl");

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: (u32, u32) = (64, 64);

#[test]
fn translated_edge_filter_matches_the_raw_wgsl_pixels() {
    // Vertical bands and a horizontal edge, so both gradient axes matter.
    let source = gradient(SIZE.0, SIZE.1, edge_frame);
    let Some((device, queue)) = common::parity::device_pair() else {
        return;
    };
    let translated = ornis_render::shaders::aa_edge_generated::wgsl_source();
    let view = upload_rgba8(&device, &queue, "edge source", SIZE, &source);
    let sampler = linear_sampler(&device);
    for (mode, threshold) in [(0u32, 0.05f32), (1, 0.05), (1, 0.2)] {
        let params = create_params(&device, &queue, mode, threshold);
        let reference = render(&device, &queue, DSL_WGSL, &view, &sampler, &params);
        let new = render(&device, &queue, &translated, &view, &sampler, &params);
        let mismatches = mismatched_pixels(&reference, &new);
        assert_eq!(
            mismatches, 0,
            "mode {mode} threshold {threshold}: {mismatches} pixels differ"
        );
    }
}

/// One texel of the gate's input: vertical bands above a horizontal edge.
fn edge_frame(x: u32, y: u32) -> [u8; 4] {
    let band = if (x / 8).is_multiple_of(2) { 40 } else { 220 };
    if y > SIZE.1 / 2 {
        [band, band, 255 - band, 255]
    } else {
        [255 - band, band, band, 255]
    }
}

/// `AaParams`: feedback, threshold, clamp radius, history flag, mode, pad.
fn create_params(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mode: u32,
    threshold: f32,
) -> wgpu::Buffer {
    let mut words = [0u8; 32];
    words[0..4].copy_from_slice(&0.0f32.to_le_bytes());
    words[4..8].copy_from_slice(&threshold.to_le_bytes());
    words[8..12].copy_from_slice(&1.0f32.to_le_bytes());
    words[12..16].copy_from_slice(&0.0f32.to_le_bytes());
    words[16..20].copy_from_slice(&mode.to_le_bytes());
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aa params"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &words);
    buffer
}

/// The three bindings the filter reads: source, sampler, uniform.
fn bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    source: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("edge parity"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
        ],
    })
}

/// Draw the fullscreen quad through `wgsl` and return the read-back pixels.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    source: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    params: &wgpu::Buffer,
) -> Vec<u8> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("edge parity"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("edge parity"),
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
    let bind = bind_group(device, &pipeline, source, sampler, params);
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
