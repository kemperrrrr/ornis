//! Pixel gate for the skinned vertex stage: the translated module
//! (`ornis-gpu-skinning`) against the frozen DSL assembly.
//!
//! Both stages draw the same triangle with the same two joint palettes and the
//! same object record, and the fragment stage — one per module, because the
//! varying field names differ — writes what the vertex stage produced. A
//! non-identity palette, a non-uniform model matrix and four influence weights
//! that are none of them one-hot keep every channel load-bearing: an identity
//! skin would pass even if the blend were dropped entirely.

#[allow(dead_code)]
mod common;

use common::parity::{half_to_f32, map_readback};
use glam::{Mat4, Vec3};

/// The frozen DSL assembly of `vs_main_skinned`.
const DSL_VERTEX: &str = include_str!("data/skinning_dsl.wgsl");

/// Bytes of one `Rgba16Float` texel: four half floats.
const TEXEL_BYTES: usize = 8;
/// Lanes of the packed `vec4` target.
const PACKED_LANES: usize = 4;
/// Bytes of one half float.
const HALF_BYTES: usize = 2;
/// Row alignment the copy requires: `copy_texture_to_buffer` wants 256 bytes.
const ROW_ALIGN: usize = 256;
/// Frames per second… no: field of view of the test camera, in degrees.
const FOV_DEGREES: f32 = 60.0;
/// Near and far planes of the test camera.
const NEAR_PLANE: f32 = 0.1;
const FAR_PLANE: f32 = 100.0;
/// Distance from the origin to the test camera.
const EYE_DISTANCE: f32 = 4.0;
/// Non-uniform scale of the object: enough that the inverse transpose is not
/// the model matrix.
const OBJECT_SCALE: [f32; 3] = [1.0, 1.25, 0.75];
/// Modelling rotation of the current frame, and the previous frame's (which
/// differs so the motion frame is not the current one).
const OBJECT_YAW: f32 = 0.3;
const PREVIOUS_YAW: f32 = 0.2;
/// Modelling translation of the object, and the previous frame's.
const OBJECT_SHIFT: [f32; 3] = [0.1, -0.2, 0.05];
const PREVIOUS_SHIFT: [f32; 3] = [0.0, -0.2, 0.05];
/// Material index the object record carries.
const OBJECT_MATERIAL: u32 = 3;
/// Joints the fixture actually uses; the rest of the palette is identity.
const USED_JOINTS: usize = 4;
/// Yaw step between two fixture joints.
const JOINT_YAW_STEP: f32 = 0.15;
/// Translation step between two fixture joints.
const JOINT_SHIFT_STEP: f32 = 0.1;
/// Constant `y`/`z` translation of every fixture joint.
const JOINT_SHIFT_YZ: [f32; 2] = [0.05, 0.0];

/// Rendering target size; the triangle covers most of it.
const SIZE: (u32, u32) = (64, 64);

/// Per-channel tolerance: both stages run the same arithmetic in the same
/// order, so the only difference a driver may introduce is FMA fusion.
const TOLERANCE: f32 = 2.0e-3;

/// Words of one `PerObjectGpu` record.
const OBJECT_BYTES: usize = 208;
/// Words of one `SkinJoint`.
const JOINT_BYTES: usize = 128;
/// Joints in the staged palette.
const PALETTE_SLOTS: usize = 128;
/// Words of one `CameraUniform`.
const CAMERA_BYTES: usize = 208;

/// One skinned vertex: the interleaved layout `SkinnedVertex` describes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkinnedVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    tangent: [f32; 3],
    joints: [u32; 4],
    weights: [f32; 4],
}

/// Three vertices in the bind pose, each weighting the joints differently so
/// no two vertices blend the same way.
const VERTICES: [SkinnedVertex; 3] = [
    SkinnedVertex {
        position: [-1.5, -1.5, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        tangent: [1.0, 0.0, 0.0],
        joints: [0, 1, 2, 3],
        weights: [0.4, 0.3, 0.2, 0.1],
    },
    SkinnedVertex {
        position: [1.5, -1.5, 0.25],
        normal: [0.0, 0.6, 0.8],
        uv: [1.0, 0.0],
        tangent: [0.0, 1.0, 0.0],
        joints: [1, 0, 3, 2],
        weights: [0.5, 0.25, 0.125, 0.125],
    },
    SkinnedVertex {
        position: [0.0, 1.5, -0.25],
        normal: [0.0, -0.8, 0.6],
        uv: [0.5, 1.0],
        tangent: [0.0, 0.0, 1.0],
        joints: [2, 3, 0, 1],
        weights: [0.35, 0.35, 0.2, 0.1],
    },
];

#[test]
fn translated_skinning_matches_the_dsl_pixels() {
    let Some((device, queue)) = common::parity::device_pair() else {
        return;
    };
    let translated = ornis_render::skinning::wgsl_vertex_source_skinned();
    let entry = ornis_render::skinning::skinned_entry_point();
    let (reference, lit) = render(&device, &queue, DSL_VERTEX, entry, true);
    let candidate = render(&device, &queue, &translated, entry, true).0;
    compare(&reference, &candidate, lit);
    // Every channel of the packed varying must be non-zero somewhere: a pass
    // that dropped the palette would leave the marker lanes at zero and the
    // gate would compare two equally wrong frames.
    assert_channel_spread(&reference);
    // The motion-valid flag is the one branch in the stage; dropping it is a
    // silent history bug, so it gets its own frame.
    let (no_history_ref, lit) = render(&device, &queue, DSL_VERTEX, entry, false);
    let no_history = render(&device, &queue, &translated, entry, false).0;
    compare(&no_history_ref, &no_history, lit);
    assert_ne!(
        reference, no_history_ref,
        "the motion-valid flag changed nothing — the previous position is not exercised"
    );
}

/// Assert the two frames agree, and that enough texels are non-zero for the
/// comparison to mean anything.
fn compare(reference: &[u8], candidate: &[u8], lit: usize) {
    assert_eq!(reference.len(), candidate.len());
    let mut differing = 0usize;
    // The packed target is `vec4(o0_ + o1_, material_marker, previous_z)`, so
    // every 8-byte texel is four half floats.
    for (a, b) in reference
        .chunks_exact(TEXEL_BYTES)
        .zip(candidate.chunks_exact(TEXEL_BYTES))
    {
        for lane in 0..PACKED_LANES {
            let x = half_to_f32(&a[lane * HALF_BYTES..lane * HALF_BYTES + HALF_BYTES]);
            let y = half_to_f32(&b[lane * HALF_BYTES..lane * HALF_BYTES + HALF_BYTES]);
            if (x - y).abs() > TOLERANCE {
                differing += 1;
            }
        }
    }
    assert_eq!(differing, 0, "{differing} lanes differ");
    assert!(
        lit > (SIZE.0 * SIZE.1 / 2) as usize,
        "only {lit} texels carry a value — the gate would pass vacuously"
    );
}

/// Every lane of the packed varying is non-zero somewhere in the frame.
fn assert_channel_spread(reference: &[u8]) {
    let mut seen = [false; PACKED_LANES];
    for texel in reference.chunks_exact(TEXEL_BYTES) {
        for (lane, seen) in seen.iter_mut().enumerate() {
            if half_to_f32(&texel[lane * HALF_BYTES..lane * HALF_BYTES + HALF_BYTES]) != 0.0 {
                *seen = true;
            }
        }
    }
    assert_eq!(
        seen, [true; PACKED_LANES],
        "a packed lane is zero everywhere — it carries no input"
    );
}

/// One frame of camera data: the view-projection plus the previous frame's.
fn camera_bytes() -> Vec<u8> {
    let projection = glam::camera::rh::proj::directx::perspective(
        FOV_DEGREES.to_radians(),
        1.0,
        NEAR_PLANE,
        FAR_PLANE,
    );
    let view = glam::camera::rh::view::look_at_mat4(
        Vec3::new(0.0, 0.0, EYE_DISTANCE),
        Vec3::ZERO,
        Vec3::Y,
    );
    let view_proj = projection * view;
    let mut bytes = Vec::with_capacity(CAMERA_BYTES);
    for matrix in [view_proj, view_proj.inverse(), view_proj] {
        bytes.extend_from_slice(bytemuck::bytes_of(&matrix.to_cols_array()));
    }
    bytes.extend_from_slice(bytemuck::cast_slice(&[0.0f32, 0.0, 4.0, 1.0]));
    // The remaining slots (ibl weight, exposure, …) are not read here.
    bytes.resize(CAMERA_BYTES, 0);
    bytes
}

/// The per-object record: a modelling matrix with a non-uniform scale (so the
/// inverse transpose is not the model), its predecessor, the material index and
/// the `motion_valid` flag.
fn object_bytes(motion_valid: bool) -> Vec<u8> {
    let model = Mat4::from_scale_rotation_translation(
        Vec3::from(OBJECT_SCALE),
        glam::Quat::from_rotation_z(OBJECT_YAW),
        Vec3::from(OBJECT_SHIFT),
    );
    let previous = Mat4::from_scale_rotation_translation(
        Vec3::from(OBJECT_SCALE),
        glam::Quat::from_rotation_z(PREVIOUS_YAW),
        Vec3::from(PREVIOUS_SHIFT),
    );
    let mut bytes = Vec::with_capacity(OBJECT_BYTES);
    for matrix in [model, model.inverse().transpose(), previous] {
        bytes.extend_from_slice(bytemuck::bytes_of(&matrix.to_cols_array()));
    }
    bytes.extend_from_slice(bytemuck::bytes_of(&[OBJECT_MATERIAL]));
    bytes.extend_from_slice(bytemuck::bytes_of(&[u32::from(motion_valid)]));
    bytes.resize(OBJECT_BYTES, 0);
    bytes
}

/// The palette: four joints that each rotate and translate differently, so a
/// swapped index or a dropped weight changes the frame.
fn palette_bytes() -> Vec<u8> {
    let mut joints: Vec<u8> = Vec::with_capacity(PALETTE_SLOTS * JOINT_BYTES);
    for joint in 0..PALETTE_SLOTS {
        let angle = JOINT_YAW_STEP * joint as f32;
        let matrix = if joint < USED_JOINTS {
            Mat4::from_rotation_translation(
                glam::Quat::from_rotation_y(angle),
                Vec3::new(
                    JOINT_SHIFT_STEP * joint as f32,
                    JOINT_SHIFT_YZ[0],
                    JOINT_SHIFT_YZ[1],
                ),
            )
        } else {
            Mat4::IDENTITY
        };
        joints.extend_from_slice(bytemuck::bytes_of(&matrix.to_cols_array()));
        joints.extend_from_slice(bytemuck::bytes_of(
            &matrix.inverse().transpose().to_cols_array(),
        ));
    }
    joints
}

/// Storage buffer of `bytes`.
fn storage(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

/// One bind-group entry: the skinned stage binds four resources at fixed slots.
fn bind_entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

/// Draw one triangle with the vertex module `vertex` and read the target back.
///
/// The fragment stage packs the three varying families the skinned stage
/// writes — position, normal, and the motion frame with the material index —
/// into one `vec4` so a single frame gates all of them.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vertex: &str,
    entry: &str,
    motion_valid: bool,
) -> (Vec<u8>, usize) {
    let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("skinning parity vertex"),
        source: wgpu::ShaderSource::Wgsl(vertex.into()),
    });
    let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("skinning parity fragment"),
        source: wgpu::ShaderSource::Wgsl(FRAGMENT.into()),
    });
    let camera = storage(device, queue, "camera", &camera_bytes());
    let objects = storage(device, queue, "objects", &object_bytes(motion_valid));
    let palette = storage(device, queue, "palette", &palette_bytes());
    let previous = storage(device, queue, "previous palette", &palette_bytes());
    let vertices: Vec<SkinnedVertex> = VERTICES.to_vec();
    let vertex_buffer = storage(
        device,
        queue,
        "skinned vertices",
        bytemuck::cast_slice(&vertices),
    );
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("skinning parity"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: Some(entry),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SkinnedVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![
                    0 => Float32x3,
                    1 => Float32x3,
                    2 => Float32x2,
                    3 => Float32x3,
                    4 => Uint32x4,
                    5 => Float32x4,
                ],
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
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
        label: Some("skinning parity"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            bind_entry(0, camera.as_entire_binding()),
            bind_entry(1, objects.as_entire_binding()),
            bind_entry(3, palette.as_entire_binding()),
            bind_entry(4, previous.as_entire_binding()),
        ],
    });
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("skinning parity target"),
        size: wgpu::Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("skinning parity"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
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
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.draw(0..3, 0..1);
    }
    let bytes = read_target(device, queue, encoder, &target);
    let lit = bytes
        .chunks_exact(8)
        .filter(|texel| texel.iter().any(|byte| *byte != 0))
        .count();
    (bytes, lit)
}

/// The fragment stage: unpack the varying families the skinned stage writes.
///
/// Both modules declare the same locations (the DSL one names them, the
/// translated one emits `oN_`), so one fragment stage serves both; it packs
/// position, normal, the material index and the previous position into one
/// target so a single frame gates all of them. The previous position's `w` is
/// the motion-valid marker, so it rides the same lane as `z`: that is the only
/// way the flag's branch shows up in the frame.
const FRAGMENT: &str = "\
struct In {
    @location(0) p: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(4) @interpolate(flat) m: u32,
    @location(5) q: vec4<f32>,
};
@fragment
fn fs_main(input: In) -> @location(0) vec4<f32> {
    let packed = input.p + input.n;
    return vec4<f32>(packed.x, packed.y, f32(input.m) + input.q.y, input.q.z + input.q.w);
}
";

/// Copy the colour target into a buffer and unpad its rows.
fn read_target(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    target: &wgpu::Texture,
) -> Vec<u8> {
    let unpadded = SIZE.0 * TEXEL_BYTES as u32;
    let padded = unpadded.div_ceil(ROW_ALIGN as u32) * ROW_ALIGN as u32;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skinning parity readback"),
        size: (padded * SIZE.1) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
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
    queue.submit(Some(encoder.finish()));
    let bytes = map_readback(device, &buffer);
    let mut out = Vec::with_capacity((unpadded * SIZE.1) as usize);
    for row in 0..SIZE.1 {
        let start = (row * padded) as usize;
        out.extend_from_slice(&bytes[start..start + unpadded as usize]);
    }
    out
}
