//! Fog parity: the translated shader against the DSL one it replaces.
//!
//! The pass rebuilds a world position from the depth buffer and the camera's
//! inverse view-projection, so the inputs are a depth texture and a camera
//! uniform. The DSL modules are frozen under `data/`.

#[allow(dead_code)]
mod common;

use common::parity::{depth_ramp, gradient, mismatched_pixels, upload_rgba8};
use common::sized::{FrameSpec, render_and_readback};

/// The DSL fragment module this pass was assembled from.
const DSL_FRAGMENT: &str = include_str!("data/fog_dsl.wgsl");
/// The DSL vertex module (a pass keeps its stages in separate modules there).
const DSL_VERTEX: &str = include_str!("data/fog_dsl_vertex.wgsl");

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: (u32, u32) = (64, 64);
/// Inverse view-projection of a camera at `(0, 0, 5)` looking down -Z.
const INV_VIEW_PROJ: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 5.0, 1.0,
];

#[test]
fn translated_fog_matches_the_dsl_pixels() {
    let Some((device, queue)) = common::parity::device_pair() else {
        return;
    };
    let hdr = gradient(SIZE.0, SIZE.1, |x, y| {
        [(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8, 255]
    });
    let translated = ornis_render::shaders::fog_generated::wgsl_source();
    let dsl = dsl_stage_pair();
    // A near-to-far depth ramp, rendered because a depth texture cannot be
    // uploaded.
    let hdr_view = upload_rgba8(&device, &queue, "hdr", SIZE, &hdr);
    let depth_view = depth_ramp(&device, &queue, SIZE);
    for (color, density) in [
        ([1.0f32, 0.5, 0.25], 0.5f32),
        ([0.2, 0.6, 0.9], 2.0),
        ([0.0, 0.0, 0.0], 0.0),
    ] {
        let params = create_params(&device, &queue, color, density);
        let inputs = Inputs {
            hdr: &hdr_view,
            depth: &depth_view,
            params: &params,
        };
        let reference = render(&device, &queue, &dsl, &inputs);
        let new = render(&device, &queue, &translated, &inputs);
        let mismatches = mismatched_pixels(&reference, &new);
        assert_eq!(
            mismatches, 0,
            "{mismatches} pixels differ from the DSL pass"
        );
    }
}

/// The two DSL modules as one: both declare the quad constants and the varying
/// struct, so the fragment copies are dropped.
fn dsl_stage_pair() -> String {
    let fragment: String = DSL_FRAGMENT
        .lines()
        .filter(|line| !line.starts_with("const quad") && !line.starts_with("const uvs"))
        .collect::<Vec<_>>()
        .join("\n");
    remove_struct(&fragment, "QuadVertexOutput")
        .map(|fragment| format!("{DSL_VERTEX}\n{fragment}"))
        .unwrap_or_default()
}

/// Drop one `struct <name> { … };` declaration from `source`.
fn remove_struct(source: &str, name: &str) -> Option<String> {
    let header = format!("struct {name} {{");
    let start = source.find(&header)?;
    let rest = &source[start..];
    let end = rest.find('}')? + 1;
    Some(format!("{}{}", &source[..start], &rest[end..]))
}

/// Camera uniform (`13 × vec4`) and fog params (`colour + density`) in the
/// layout both shaders declare.
fn create_params(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color: [f32; 3],
    density: f32,
) -> [wgpu::Buffer; 2] {
    // Thirteen `vec4` slots: view-projection (0..4), inverse (4..8), eye
    // position (8), previous view-projection (9..13).
    let mut camera = [0u8; 13 * 16];
    for (index, value) in INV_VIEW_PROJ.iter().enumerate() {
        // One matrix element per float: field `4 + index / 4`, lane `index % 4`.
        let offset = (4 + index / 4) * 16 + (index % 4) * 4;
        camera[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    // Eye position, field 8.
    camera[8 * 16..8 * 16 + 4].copy_from_slice(&0.0f32.to_le_bytes());
    camera[8 * 16 + 4..8 * 16 + 8].copy_from_slice(&0.0f32.to_le_bytes());
    camera[8 * 16 + 8..8 * 16 + 12].copy_from_slice(&5.0f32.to_le_bytes());
    camera[8 * 16 + 12..8 * 16 + 16].copy_from_slice(&1.0f32.to_le_bytes());
    let mut fog = [0u8; 16];
    fog[0..4].copy_from_slice(&color[0].to_le_bytes());
    fog[4..8].copy_from_slice(&color[1].to_le_bytes());
    fog[8..12].copy_from_slice(&color[2].to_le_bytes());
    fog[12..16].copy_from_slice(&density.to_le_bytes());
    let create = |label: &str, words: &[u8]| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: words.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, words);
        buffer
    };
    [create("camera", &camera), create("fog params", &fog)]
}

/// Everything the fragment reads, bound once per rendering.
struct Inputs<'a> {
    /// HDR layer.
    hdr: &'a wgpu::TextureView,
    /// G-buffer depth.
    depth: &'a wgpu::TextureView,
    /// Camera and fog uniforms.
    params: &'a [wgpu::Buffer; 2],
}

/// Draw the fullscreen quad through `wgsl` and return the read-back pixels.
fn render(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str, inputs: &Inputs<'_>) -> Vec<u8> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fog parity"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("fog parity"),
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
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("fog parity"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fog parity"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(inputs.hdr),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(inputs.depth),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: inputs.params[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: inputs.params[1].as_entire_binding(),
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
