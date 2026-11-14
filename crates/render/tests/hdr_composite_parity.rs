//! HDR composite parity: the translated shader against the DSL one it replaces.
//!
//! The pass used to be assembled from `#[stage]` sources at runtime (a separate
//! vertex and fragment module); both are frozen under `data/` so the Rust kernel
//! (`crates/gpu-hdr-composite`) is compared with them pixel for pixel, over the
//! three bloom modes.

#[allow(dead_code)]
mod common;

use common::parity::{gradient, linear_sampler, mismatched_pixels, upload_rgba8};
use common::sized::{FrameSpec, render_and_readback};

/// The DSL fragment module this pass was assembled from.
const DSL_FRAGMENT: &str = include_str!("data/hdr_composite_dsl.wgsl");
/// The DSL vertex module (a pass keeps its stages in separate modules there).
const DSL_VERTEX: &str = include_str!("data/hdr_composite_dsl_vertex.wgsl");

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: (u32, u32) = (64, 64);

#[test]
fn translated_hdr_composite_matches_the_dsl_pixels() {
    let Some((_, device, queue)) = common::adapter_or_skip() else {
        return;
    };
    let translated = ornis_render::shaders::hdr_composite_generated::wgsl_source();
    let dsl = dsl_stage_pair();
    let deferred = gradient(SIZE.0, SIZE.1, |x, y| {
        [(x * 3) as u8, (y * 3) as u8, ((x + y) * 2) as u8, 255]
    });
    let forward = gradient(SIZE.0, SIZE.1, |x, y| {
        [
            255 - (x * 3) as u8,
            (y * 2) as u8,
            (x * 2) as u8,
            (x * 4) as u8,
        ]
    });
    let bloom = gradient(SIZE.0, SIZE.1, |x, y| {
        [((x + y) * 2) as u8, 40, (x * 3) as u8, 255]
    });
    // `BloomParams` mode word picks the layer mix; intensity and exposure scale
    // what follows, so each mode exercises a different arithmetic path.
    for (mode, intensity, exposure) in [(0u32, 1.0f32, 1.0f32), (1, 0.5, 2.0), (2, 0.25, 0.5)] {
        let params = create_params(&device, &queue, intensity, mode, exposure);
        let reference = render(
            &device,
            &queue,
            &dsl,
            [&deferred, &forward, &bloom],
            &params,
        );
        let new = render(
            &device,
            &queue,
            &translated,
            [&deferred, &forward, &bloom],
            &params,
        );
        let mismatches = mismatched_pixels(&reference, &new);
        assert_eq!(mismatches, 0, "mode {mode}: {mismatches} pixels differ");
    }
}

/// The two DSL modules as one: the pass keeps its stages apart at runtime, and
/// both declare the quad constants, so the fragment copy is dropped.
fn dsl_stage_pair() -> String {
    let fragment: String = DSL_FRAGMENT
        .lines()
        .filter(|line| !line.starts_with("const quad") && !line.starts_with("const uvs"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{DSL_VERTEX}\n{fragment}")
}

/// The two uniforms the fragment reads, packed like the host does.
fn create_params(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    intensity: f32,
    mode: u32,
    exposure: f32,
) -> [wgpu::Buffer; 2] {
    // Two separate uniform buffers, like the pass binds them: a shared buffer
    // would need 256-byte offsets between the two bindings.
    let mut bloom = [0u8; 16];
    bloom[0..4].copy_from_slice(&0.0f32.to_le_bytes());
    bloom[4..8].copy_from_slice(&intensity.to_le_bytes());
    bloom[8..12].copy_from_slice(&mode.to_le_bytes());
    let mut exposure_words = [0u8; 16];
    exposure_words[0..4].copy_from_slice(&exposure.to_le_bytes());
    let create = |label: &str, words: &[u8; 16]| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, words);
        buffer
    };
    [
        create("bloom params", &bloom),
        create("exposure", &exposure_words),
    ]
}

/// Draw the fullscreen quad through `wgsl` and return the read-back pixels.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    textures: [&[u8]; 3],
    params: &[wgpu::Buffer; 2],
) -> Vec<u8> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hdr composite parity"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("hdr composite parity"),
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
    let views: Vec<wgpu::TextureView> = textures
        .iter()
        .map(|pixels| upload_rgba8(device, queue, "hdr composite input", SIZE, pixels))
        .collect();
    let sampler = linear_sampler(device);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hdr composite parity"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&views[2]),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params[1].as_entire_binding(),
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
