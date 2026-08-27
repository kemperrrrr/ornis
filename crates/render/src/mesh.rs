//! GPU mesh representation and procedural primitive generation.

use wgpu::util::DeviceExt;

/// Vertex + index buffers uploaded to the device, ready to draw.
pub struct Mesh {
    /// Interleaved [`Vertex`] data.
    pub vertex_buffer: wgpu::Buffer,
    /// Triangle index list (`u32`).
    pub index_buffer: wgpu::Buffer,
    /// Number of indices to draw.
    pub num_indices: u32,
    /// Number of vertices in `vertex_buffer`.
    pub vertex_count: u32,
}

/// GPU vertex layout shared by every mesh (must match the WGSL inputs).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// Shading normal (unit length for generated primitives).
    pub normal: [f32; 3],
    /// Texture coordinates in [0, 1].
    pub uv: [f32; 2],
    /// Surface tangent for normal mapping / anisotropy.
    pub tangent: [f32; 3],
}

impl Vertex {
    /// wgpu vertex buffer layout matching this struct's memory layout.
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<[f32; 2]>())
                        as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Generate a UV sphere with positions, normals, UVs and tangents, uploading
/// it to `device`. `sectors`/`stacks` are clamped to at least 3/2 so degenerate
/// arguments still produce valid geometry.
pub fn create_sphere(device: &wgpu::Device, radius: f32, sectors: u32, stacks: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let sector_count = sectors.max(3);
    let stack_count = stacks.max(2);
    let sector_step = 2.0 * std::f32::consts::PI / sector_count as f32;
    let stack_step = std::f32::consts::PI / stack_count as f32;

    for i in 0..=stack_count {
        let stack_angle = std::f32::consts::PI / 2.0 - i as f32 * stack_step;
        let xy = radius * stack_angle.cos();
        let z = radius * stack_angle.sin();

        for j in 0..=sector_count {
            let sector_angle = j as f32 * sector_step;
            let x = xy * sector_angle.cos();
            let y = xy * sector_angle.sin();

            let nx = x / radius;
            let ny = y / radius;
            let nz = z / radius;

            let tx = -sector_angle.sin();
            let ty = sector_angle.cos();
            let tz = 0.0;

            let u = j as f32 / sector_count as f32;
            let v = i as f32 / stack_count as f32;

            vertices.push(Vertex {
                position: [x, y, z],
                normal: [nx, ny, nz],
                uv: [u, v],
                tangent: [tx, ty, tz],
            });
        }
    }

    for i in 0..stack_count {
        let k1 = i * (sector_count + 1);
        let k2 = k1 + sector_count + 1;
        for j in 0..sector_count {
            if i != 0 {
                indices.push(k1 + j);
                indices.push(k2 + j);
                indices.push(k1 + j + 1);
            }
            if i != stack_count - 1 {
                indices.push(k1 + j + 1);
                indices.push(k2 + j);
                indices.push(k2 + j + 1);
            }
        }
    }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sphere vertex buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sphere index buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    Mesh {
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
        vertex_count: vertices.len() as u32,
    }
}
