//! Bloom parity: the translated bright-pass against the DSL one it replaces.
//!
//! The pass samples one layer and scales it by the luminance knee, so the
//! inputs are an image and the bloom uniform. The DSL module is frozen under
//! `data/`.

#[allow(dead_code)]
mod common;

use common::parity::{gradient, linear_sampler, mismatched_pixels, upload_rgba8};
use common::sized::{FrameSpec, render_and_readback};

/// The DSL module this pass was assembled from (both stages in one module).
const DSL_WGSL: &str = include_str!("data/bloom_dsl.wgsl");

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: (u32, u32) = (64, 64);

#[test]
fn translated_bloom_matches_the_dsl_pixels() {
    let source = gradient(SIZE.0, SIZE.1, |x, y| {
        // A ramp across the bright-pass threshold, so both the dark and the
        // knee region are covered.
        [((x + y) * 2) as u8, (x * 4) as u8, (y * 4) as u8, 255]
    });
    let Some((device, queue)) = common::parity::device_pair() else {
        return;
    };
    let view = upload_rgba8(&device, &queue, "bloom source", SIZE, &source);
    let sampler = linear_sampler(&device);
    let translated = ornis_render::shaders::bloom_generated::wgsl_source();
    for threshold in [0.0f32, 0.2, 0.6] {
        let params = create_params(&device, &queue, threshold);
        let reference = render(&device, &queue, DSL_WGSL, &view, &sampler, &params);
        let new = render(&device, &queue, &translated, &view, &sampler, &params);
        let mismatches = mismatched_pixels(&reference, &new);
        assert_eq!(
            mismatches, 0,
            "threshold {threshold}: {mismatches} pixels differ from the DSL pass"
        );
    }
}

/// `BloomParams`: threshold, intensity, mode word, pad.
fn create_params(device: &wgpu::Device, queue: &wgpu::Queue, threshold: f32) -> wgpu::Buffer {
    let mut words = [0u8; 16];
    words[0..4].copy_from_slice(&threshold.to_le_bytes());
    words[4..8].copy_from_slice(&1.0f32.to_le_bytes());
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bloom params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &words);
    buffer
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
        label: Some("bloom parity"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bloom parity"),
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
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bloom parity"),
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
