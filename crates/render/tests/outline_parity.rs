//! Outline parity: the translated pass against the DSL one it replaces.
//!
//! The pass has no textures — it reads the camera, the per-object record and the
//! two compact lists, and writes a clip-space position plus a flat colour. So
//! the gate draws the same triangle twice, once through the frozen DSL modules
//! (`data/outline_vertex_dsl.wgsl`, `data/outline_fragment_dsl.wgsl`) and once
//! through `crates/gpu-outline`, into identical targets, and compares every
//! pixel.
//!
//! Culling and the depth state are the pipeline's business and unchanged by the
//! migration, so both sides draw with culling off and no depth attachment: what
//! is compared here is the arithmetic — the object-space expansion, the
//! column-major `view_proj · model · p` and the flag pick.

#[allow(dead_code)]
mod common;

use common::parity::{device_pair, mismatched_pixels};
use common::sized::{FrameSpec, render_and_readback};

/// The frozen DSL vertex module.
const DSL_VERTEX: &str = include_str!("data/outline_vertex_dsl.wgsl");
/// The frozen DSL fragment module.
const DSL_FRAGMENT: &str = include_str!("data/outline_fragment_dsl.wgsl");

/// Read-back format (what both pipelines are built for).
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Offscreen extent.
const SIZE: (u32, u32) = (64, 64);
/// Floats per vertex: position, normal, uv, tangent.
const VERTEX_FLOATS: usize = 11;

/// One vertex of the parity triangle.
///
/// The DSL module declares all four attributes, the translated one only the
/// position and the normal; the buffer carries all four so one layout serves
/// both.
fn vertex(position: [f32; 3], normal: [f32; 3]) -> Vec<f32> {
    let mut out = Vec::with_capacity(VERTEX_FLOATS);
    out.extend_from_slice(&position);
    out.extend_from_slice(&normal);
    out.extend_from_slice(&[0.0, 0.0]);
    out.extend_from_slice(&normal);
    out
}

/// The parity mesh: one quad whose normals are not axis-aligned, so the
/// object-space expansion has to be a real `p + n · width` and not a `z += k`.
fn triangle() -> Vec<f32> {
    let mut data = Vec::new();
    for (position, normal) in [
        ([-0.5, -0.5, 0.0], [0.6, 0.8, 0.0]),
        ([0.5, -0.5, 0.0], [0.6, 0.8, 0.0]),
        ([0.0, 0.5, 0.0], [0.6, 0.8, 0.0]),
        ([0.6, -0.4, 0.1], [0.0, 0.0, 1.0]),
        ([0.2, 0.4, -0.1], [0.0, 1.0, 0.0]),
        ([-0.3, 0.3, 0.2], [1.0, 0.0, 0.0]),
    ] {
        data.extend_from_slice(&vertex(position, normal));
    }
    data
}

/// Everything both stages read, uploaded once per flag value.
struct Inputs {
    /// Vertex buffer, three vertices per triangle.
    vertices: wgpu::Buffer,
    /// View-projection (identity) plus the twelve unused camera slots.
    camera: wgpu::Buffer,
    /// One per-object record with an identity model matrix.
    objects: wgpu::Buffer,
    /// Compact scene-index list with a single entry.
    indices: wgpu::Buffer,
    /// Flags parallel to the index list.
    flags: wgpu::Buffer,
}

/// Build the buffers for one flag value (`0` none, `1` hover, `2` selected).
fn inputs(device: &wgpu::Device, queue: &wgpu::Queue, flag: u32) -> Inputs {
    let create = |label: &str, words: &[u8], usage: wgpu::BufferUsages| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: words.len() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, words);
        buffer
    };
    let vertices = triangle();
    let vertex_bytes: Vec<u8> = vertices.iter().flat_map(|f| f.to_le_bytes()).collect();
    // Thirteen `vec4` slots; slot 0..4 is `view_proj`, and an identity there
    // makes the clip position the expanded object-space position.
    let mut camera = [0u8; 13 * 16];
    for (slot, lane) in [(0usize, 0usize), (1, 1), (2, 2), (3, 3)] {
        let offset = slot * 16 + lane * 4;
        camera[offset..offset + 4].copy_from_slice(&1.0f32.to_le_bytes());
    }
    // `PerObject`: model (16 words, identity), the rest zero.
    let mut objects = [0u8; 52 * 4];
    for (word, lane) in [(0usize, 0usize), (4, 1), (8, 2), (12, 3)] {
        objects[(word + lane) * 4..(word + lane + 1) * 4].copy_from_slice(&1.0f32.to_le_bytes());
    }
    Inputs {
        vertices: create(
            "outline vertices",
            &vertex_bytes,
            wgpu::BufferUsages::VERTEX,
        ),
        camera: create("outline camera", &camera, wgpu::BufferUsages::UNIFORM),
        objects: create("outline objects", &objects, wgpu::BufferUsages::STORAGE),
        indices: create(
            "outline indices",
            &0u32.to_le_bytes(),
            wgpu::BufferUsages::STORAGE,
        ),
        flags: create(
            "outline flags",
            &flag.to_le_bytes(),
            wgpu::BufferUsages::STORAGE,
        ),
    }
}

/// The vertex layout both modules are built against.
///
/// The offsets follow the attribute order (`VERTEX_FLOATS` floats per vertex),
/// so the macro derives them instead of four hand-written literals.
fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32x3
    ];
    wgpu::VertexBufferLayout {
        array_stride: (VERTEX_FLOATS * 4) as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }
}

/// Draw the parity mesh through `vertex_src`/`fragment_src` and read it back.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vertex_src: &str,
    fragment_src: &str,
    inputs: &Inputs,
) -> Vec<u8> {
    let shader = |label: &str, source: &str| {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    };
    let vertex_module = shader("outline parity vertex", vertex_src);
    let fragment_module = shader("outline parity fragment", fragment_src);
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("outline parity"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &vertex_module,
            entry_point: Some("vs_main"),
            buffers: &[Some(vertex_layout())],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment_module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            // The hull's winding is not what this gate compares; the pipeline
            // state that culls it is unchanged by the migration.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("outline parity"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inputs.camera.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: inputs.objects.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: inputs.indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: inputs.flags.as_entire_binding(),
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
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, inputs.vertices.slice(..));
            pass.draw(0..6, 0..1);
        },
    )
}

#[test]
fn translated_outline_matches_the_dsl_pixels() {
    let Some((device, queue)) = device_pair() else {
        return;
    };
    let translated_vertex = ornis_render::shaders::outline_generated::wgsl_vertex_source();
    let translated_fragment = ornis_render::shaders::outline_generated::wgsl_fragment_source();
    // Every flag the host can upload: `0` and `1` both paint the hover colour,
    // `2` the selected one, so the pick is compared in both directions.
    let mut references = Vec::new();
    for flag in [0u32, 1, 2] {
        let inputs = inputs(&device, &queue, flag);
        let reference = render(&device, &queue, DSL_VERTEX, DSL_FRAGMENT, &inputs);
        let new = render(
            &device,
            &queue,
            &translated_vertex,
            &translated_fragment,
            &inputs,
        );
        let mismatches = mismatched_pixels(&reference, &new);
        assert_eq!(mismatches, 0, "flag {flag}: {mismatches} pixels differ");
        // A gate over two blank images proves nothing: the hull has to paint
        // pixels on both sides.
        let painted = painted_pixels(&reference);
        assert!(
            painted > (SIZE.0 * SIZE.1 / 20) as usize,
            "flag {flag} painted only {painted} pixels"
        );
        references.push(reference);
    }
    // Flags `0` and `1` are the same hover colour, `2` is the selected one: if
    // the comparison cannot tell them apart, the flag path is not exercised.
    assert_eq!(
        mismatched_pixels(&references[0], &references[1]),
        0,
        "flags 0 and 1 must paint the hover colour"
    );
    assert!(
        mismatched_pixels(&references[1], &references[2]) > 0,
        "the selected colour must differ from the hover one"
    );
}

/// Pixels the hull painted, counted by a non-black colour.
fn painted_pixels(image: &[u8]) -> usize {
    image
        .chunks_exact(4)
        .filter(|texel| texel[0] | texel[1] | texel[2] > 0)
        .count()
}
