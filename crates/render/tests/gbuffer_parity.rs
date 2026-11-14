//! G-buffer parity: the translated pass against the frozen DSL version.
//!
//! The pass writes seven attachments — five lighting targets plus the motion
//! vector and the quantized previous depth — so the gate renders both shaders
//! with one instance, one material record and one camera, and compares every
//! attachment: integer targets exactly, float targets with a tolerance smaller
//! than one 16-bit step.

#[allow(dead_code)]
mod common;

use common::parity::map_readback;
use glam::{Mat4, Vec3};

/// The g-buffer vertex stage this pass used before the migration.
const DSL_VERTEX: &str = include_str!("data/gbuffer_vertex_dsl.wgsl");
/// The g-buffer fragment stage this pass used before the migration.
const DSL_FRAGMENT: &str = include_str!("data/gbuffer_fragment_dsl.wgsl");

const SIZE: (u32, u32) = (64, 64);

/// Attachment formats: albedo, encoded normal, material id, world xy, material
/// parameters, motion, previous depth.
///
const FORMATS: [wgpu::TextureFormat; 7] = [
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rg16Float,
    wgpu::TextureFormat::R16Uint,
    wgpu::TextureFormat::Rg16Float,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rg16Float,
    wgpu::TextureFormat::R16Uint,
];

/// Bytes per texel of each attachment format (32 bytes per sample in total,
/// the device limit).
const TEXEL_BYTES: [u32; 7] = [4, 4, 2, 4, 8, 4, 2];

/// Words per `PerObjectGpu` record.
const OBJECT_BYTES: usize = 208;
/// Words per `OpenPBRMaterial` record.
const MATERIAL_BYTES: usize = 320;
/// Bytes of `CameraUniform`: two matrices, the eye, the previous matrix.
const CAMERA_BYTES: usize = 208;

/// Half-float tolerance: one step of a `Rg16Float` target near unity.
const FLOAT_TOLERANCE: f32 = 2.0e-3;

#[test]
fn translated_gbuffer_matches_the_dsl_pixels() {
    let Some((device, queue)) = common::parity::device_pair() else {
        return;
    };
    let translated_vertex = ornis_render::shaders::gbuffer_generated::wgsl_vertex_source();
    let translated_fragment = ornis_render::shaders::gbuffer_generated::wgsl_source();
    let inputs = Inputs::new(&device, &queue);
    let reference = render(&device, &queue, DSL_VERTEX, DSL_FRAGMENT, &inputs);
    let new = render(
        &device,
        &queue,
        &translated_vertex,
        &translated_fragment,
        &inputs,
    );
    for (target, (expected, actual)) in reference.iter().zip(new.iter()).enumerate() {
        let mismatches = mismatch_count(target, expected, actual);
        assert_eq!(
            mismatches, 0,
            "attachment {target} ({:?}): {mismatches} texels differ",
            FORMATS[target]
        );
        assert!(
            expected.iter().any(|byte| *byte != 0),
            "attachment {target} is empty — the gate would pass vacuously"
        );
    }
}

/// The camera, the object record, the material record and the mesh.
struct Inputs {
    /// Group 0 binding 0.
    camera: wgpu::Buffer,
    /// Group 0 binding 1: one `PerObjectGpu`.
    objects: wgpu::Buffer,
    /// Group 0 binding 2: one `OpenPBRMaterial`.
    materials: wgpu::Buffer,
    /// Positions, normals, uvs and tangents of one triangle.
    vertices: wgpu::Buffer,
}

impl Inputs {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            camera: uniform(device, queue, "gbuffer camera", &camera_bytes()),
            objects: uniform(device, queue, "gbuffer objects", &object_bytes()),
            materials: uniform(device, queue, "gbuffer materials", &material_bytes()),
            vertices: vertex_buffer(device, queue),
        }
    }
}

/// One frame of camera data: the current and the previous view-projection.
fn camera_bytes() -> Vec<u8> {
    let projection =
        glam::camera::rh::proj::directx::perspective(60f32.to_radians(), 1.0, 0.1, 100.0);
    let view = glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y);
    let view_proj = projection * view;
    let previous_view_proj = projection
        * glam::camera::rh::view::look_at_mat4(Vec3::new(0.02, 0.0, 3.0), Vec3::ZERO, Vec3::Y);
    let mut bytes = Vec::with_capacity(CAMERA_BYTES);
    for matrix in [view_proj, view_proj.inverse()] {
        bytes.extend_from_slice(bytemuck::bytes_of(&matrix.to_cols_array()));
    }
    bytes.extend_from_slice(bytemuck::bytes_of(&[0.0f32, 0.0, 3.0, 1.0]));
    bytes.extend_from_slice(bytemuck::bytes_of(&previous_view_proj.to_cols_array()));
    bytes
}

/// One object record: a non-uniformly scaled, rotated instance that moved
/// since the previous frame, so the motion vector and the inverse-transpose
/// normal both carry a value the gate can see.
fn object_bytes() -> Vec<u8> {
    let model = Mat4::from_scale_rotation_translation(
        Vec3::new(2.0, 1.0, 1.0),
        glam::Quat::from_rotation_y(0.6),
        Vec3::new(0.0, 0.0, -3.0),
    );
    let previous = Mat4::from_translation(Vec3::new(-0.05, 0.0, 0.0)) * model;
    let mut bytes = Vec::with_capacity(OBJECT_BYTES);
    bytes.extend_from_slice(bytemuck::bytes_of(&model.to_cols_array()));
    bytes.extend_from_slice(bytemuck::bytes_of(
        &model.inverse().transpose().to_cols_array(),
    ));
    bytes.extend_from_slice(bytemuck::bytes_of(&previous.to_cols_array()));
    // material_index, motion_valid, then two words of padding. The index is
    // not zero so the id attachment carries a value the gate can compare.
    bytes.extend_from_slice(bytemuck::bytes_of(&[1u32, 1, 0, 0]));
    bytes
}

/// Two `OpenPBRMaterial` records of twenty `vec4` slots each, in host order.
///
/// The object reads the second one, so a kernel that ignored the material
/// index would produce the first record's albedo and fail the gate.
fn material_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(2 * MATERIAL_BYTES);
    for (index, color) in [[0.9, 0.1, 0.1, 1.0], [0.8, 0.4, 0.2, 0.9]]
        .into_iter()
        .enumerate()
    {
        let mut slots = [[0.0f32; 4]; 20];
        slots[0] = [0.0, 0.0, 0.6 - 0.1 * index as f32, 0.0];
        slots[1] = color;
        slots[2] = [0.5, 0.35, 1.5, 0.0];
        slots[3] = [1.0, 1.0, 1.0, 1.0];
        slots[12] = [0.3, 0.1, 0.0, 0.0];
        for slot in slots {
            bytes.extend_from_slice(bytemuck::bytes_of(&slot));
        }
    }
    bytes
}

/// One interleaved vertex in object space.
struct Vertex {
    /// Object-space position.
    position: [f32; 3],
    /// Object-space normal.
    normal: [f32; 3],
    /// Texture coordinates.
    uv: [f32; 2],
    /// Object-space tangent.
    tangent: [f32; 3],
}

/// A vertex with the surface facing the camera.
const fn vertex(position: [f32; 3], uv: [f32; 2]) -> Vertex {
    Vertex {
        position,
        normal: [0.0, 0.0, 1.0],
        uv,
        tangent: [1.0, 0.0, 0.0],
    }
}

/// One triangle in object space.
const TRIANGLE: [Vertex; 3] = [
    vertex([-0.6, -0.6, 0.0], [0.0, 0.0]),
    vertex([0.6, -0.6, 0.0], [1.0, 0.0]),
    vertex([0.0, 0.6, 0.0], [0.5, 1.0]),
];

/// Pack [`TRIANGLE`] into one interleaved vertex buffer.
fn vertex_buffer(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    let mut bytes = Vec::new();
    for vertex in TRIANGLE {
        bytes.extend_from_slice(bytemuck::bytes_of(&vertex.position));
        bytes.extend_from_slice(bytemuck::bytes_of(&vertex.normal));
        bytes.extend_from_slice(bytemuck::bytes_of(&vertex.uv));
        bytes.extend_from_slice(bytemuck::bytes_of(&vertex.tangent));
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gbuffer vertices"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

/// A storage or uniform buffer holding `bytes`.
fn uniform(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

/// One vertex attribute of the interleaved mesh.
const fn attribute(
    format: wgpu::VertexFormat,
    offset: u64,
    shader_location: u32,
) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        format,
        offset,
        shader_location,
    }
}

/// Vertex layout of the interleaved mesh: 12, 12, 8 and 12 bytes per stride.
fn vertex_layout() -> [Option<wgpu::VertexBufferLayout<'static>>; 1] {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
        attribute(wgpu::VertexFormat::Float32x3, 0, 0),
        attribute(wgpu::VertexFormat::Float32x3, 12, 1),
        attribute(wgpu::VertexFormat::Float32x2, 24, 2),
        attribute(wgpu::VertexFormat::Float32x3, 32, 3),
    ];
    [Some(wgpu::VertexBufferLayout {
        array_stride: 44,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    })]
}

/// Render one frame with `vertex`/`fragment` and return every attachment.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vertex: &str,
    fragment: &str,
    inputs: &Inputs,
) -> Vec<Vec<u8>> {
    let vs_module = shader_module(device, "gbuffer parity vertex", vertex);
    let fs_module = shader_module(device, "gbuffer parity fragment", fragment);
    let targets: Vec<Option<wgpu::ColorTargetState>> = FORMATS
        .iter()
        .map(|format| {
            Some(wgpu::ColorTargetState {
                format: *format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        })
        .collect();
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gbuffer parity"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: Some("vs_main"),
            buffers: &vertex_layout(),
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: Some("fs_main"),
            targets: &targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gbuffer parity"),
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
                resource: inputs.materials.as_entire_binding(),
            },
        ],
    });
    let attachments = targets_textures(device);
    let views: Vec<wgpu::TextureView> = attachments
        .iter()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
        .collect();
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let color: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = views
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
            label: Some("gbuffer parity"),
            color_attachments: &color,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_vertex_buffer(0, inputs.vertices.slice(..));
        pass.draw(0..3, 0..1);
    }
    read_targets(device, queue, encoder, &attachments)
}

/// A shader module from WGSL source.
fn shader_module(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// The seven render targets, sized like the frame.
fn targets_textures(device: &wgpu::Device) -> Vec<wgpu::Texture> {
    FORMATS
        .iter()
        .map(|format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gbuffer parity target"),
                size: wgpu::Extent3d {
                    width: SIZE.0,
                    height: SIZE.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: *format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        })
        .collect()
}

/// Copy every attachment into one buffer and split it into planes.
fn read_targets(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    targets: &[wgpu::Texture],
) -> Vec<Vec<u8>> {
    const ROW_ALIGN: u32 = 256;
    let mut planes = Vec::new();
    let mut total = 0u32;
    for texel in TEXEL_BYTES {
        let unpadded = SIZE.0 * texel;
        let padded = unpadded.div_ceil(ROW_ALIGN) * ROW_ALIGN;
        planes.push((total, padded, unpadded));
        total += padded * SIZE.1;
    }
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gbuffer parity readback"),
        size: u64::from(total),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    for (index, target) in targets.iter().enumerate() {
        let (offset, padded, _) = planes[index];
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
                    offset: u64::from(offset),
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
    planes
        .iter()
        .map(|(offset, padded, unpadded)| {
            let mut pixels = vec![0u8; (*unpadded * SIZE.1) as usize];
            for row in 0..SIZE.1 as usize {
                let start = *offset as usize + row * *padded as usize;
                pixels[row * *unpadded as usize..][..*unpadded as usize]
                    .copy_from_slice(&data[start..][..*unpadded as usize]);
            }
            pixels
        })
        .collect()
}

/// Count the texels of two read-backs that differ.
fn mismatch_count(target: usize, expected: &[u8], actual: &[u8]) -> usize {
    assert_eq!(expected.len(), actual.len(), "read-back size");
    match FORMATS[target] {
        wgpu::TextureFormat::R16Uint => expected
            .chunks_exact(2)
            .zip(actual.chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count(),
        wgpu::TextureFormat::Rgba8Unorm => common::parity::mismatched_pixels(expected, actual),
        _ => float_mismatches(expected, actual, TEXEL_BYTES[target] as usize),
    }
}

/// Count the texels whose float lanes differ by more than the tolerance.
fn float_mismatches(expected: &[u8], actual: &[u8], texel: usize) -> usize {
    expected
        .chunks_exact(texel)
        .zip(actual.chunks_exact(texel))
        .filter(|(a, b)| {
            a.chunks_exact(2)
                .zip(b.chunks_exact(2))
                .any(|(x, y)| (half_to_f32(x) - half_to_f32(y)).abs() > FLOAT_TOLERANCE)
        })
        .count()
}

/// Decode one little-endian IEEE-754 binary16.
fn half_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let exponent = i32::from((bits >> 10) & 0x1F);
    let fraction = f32::from(bits & 0x03FF) / 1024.0;
    let magnitude = match exponent {
        0 => fraction * 2.0f32.powi(-14),
        0x1F if fraction == 0.0 => f32::INFINITY,
        0x1F => f32::NAN,
        _ => (1.0 + fraction) * 2.0f32.powi(exponent - 15),
    };
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}
