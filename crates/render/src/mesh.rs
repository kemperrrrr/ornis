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

/// GPU skinned-vertex layout: bind-pose [`Vertex`] attributes (locations
/// 0–3, same offsets) plus joint influences — joints as `vec4<u32>` at
/// location 4, canonicalized weights as `vec4<f32>` at location 5.
///
/// Matches the skinned vertex stage input (`SkinnedVertexInput` in
/// [`crate::skinning`]: locations 0–3 mirror the classic attributes, 4–5
/// carry the influences): one interleaved buffer, so a skinned [`Mesh`]
/// reuses the same upload/draw shape as a classic one (only the pipeline,
/// bind group and buffer contents differ).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    /// Bind-pose position.
    pub position: [f32; 3],
    /// Bind-pose shading normal.
    pub normal: [f32; 3],
    /// Texture coordinates in [0, 1] (passthrough).
    pub uv: [f32; 2],
    /// Bind-pose surface tangent.
    pub tangent: [f32; 3],
    /// Influencing joints (top-4, `< joint count` — validated at staging).
    pub joints: [u32; 4],
    /// Influence weights (canonicalized at staging, see
    /// [`ornis_animation::canonical_staged_weights`]).
    pub weights: [f32; 4],
}

impl SkinnedVertex {
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
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Uint32x4,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<Vertex>() + std::mem::size_of::<[u32; 4]>())
                        as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
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

/// CPU-side geometry of an axis-aligned box: 24 vertices (4 per face)
/// and 36 indices, with analytic per-face normals, a planar UV map per
/// face and unit tangents orthogonal to the normals (same contract as
/// [`create_sphere`).
pub fn box_data(size: [f32; 3]) -> (Vec<Vertex>, Vec<u32>) {
    // Unit corners, scaled by `size` below (mirrors
    // `MeshData::unit_box`, whose winding is CCW from outside).
    let p = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    #[rustfmt::skip]
    let faces: [[usize; 4]; 6] = [
        [0, 1, 2, 3], // -z
        [5, 4, 7, 6], // +z
        [4, 0, 3, 7], // -x
        [1, 5, 6, 2], // +x
        [4, 5, 1, 0], // -y
        [3, 2, 6, 7], // +y
    ];
    let normals: [[f32; 3]; 6] = [
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (face, normal) in faces.iter().zip(normals) {
        let base = vertices.len() as u32;
        let tangent = ortho_tangent(normal);
        for (k, &vi) in face.iter().enumerate() {
            vertices.push(Vertex {
                position: [p[vi][0] * size[0], p[vi][1] * size[1], p[vi][2] * size[2]],
                normal,
                uv: [f32::from(k == 1 || k == 2), f32::from(k >= 2)],
                tangent,
            });
        }
        // Mirrored vs the `MeshData::unit_box` corner order: the stored
        // per-face normals are the true outward normals, and the engine
        // convention is CCW seen from outside (`front_face: Ccw`), so
        // each quad is emitted as (0,2,1)/(0,3,2). NOTE: this differs
        // from `MeshData::unit_box`, whose index winding yields the
        // negated face normals — see the report; the editor crate is
        // outside this change.
        indices.extend([base, base + 2, base + 1, base, base + 3, base + 2]);
    }
    (vertices, indices)
}

/// CPU-side geometry of a flat quad in the local XZ plane (`+Y` face
/// normal): 4 vertices and 6 indices, UVs spanning [0, 1] and the `+X`
/// tangent (unit, orthogonal to the normal — same contract as
/// [`create_sphere` on the equator).
pub fn plane_data(size: [f32; 2]) -> (Vec<Vertex>, Vec<u32>) {
    let (hx, hz) = (size[0] * 0.5, size[1] * 0.5);
    let vertices = vec![
        Vertex {
            position: [-hx, 0.0, -hz],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [hx, 0.0, -hz],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [hx, 0.0, hz],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [-hx, 0.0, hz],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];
    // CCW seen from +Y (verified by the winding test below).
    (vertices, vec![0, 2, 1, 0, 3, 2])
}

/// CPU-side geometry of a right circular cylinder around `+Y`:
/// `radial_segments` is clamped to at least 3. Layout is side quads
/// (`2 * (n + 1)` vertices, analytic radial normals and tangents) plus
/// two triangle fans (`n + 2` vertices each: center + duplicated-seam
/// ring, `∓Y` normals, `+X` tangents). Vertex count is `4 * n + 6`,
/// index count `12 * n`.
pub fn cylinder_data(radius: f32, height: f32, radial_segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let n = radial_segments.max(3);
    let step = 2.0 * std::f32::consts::PI / n as f32;
    let half = height * 0.5;
    let mut vertices = Vec::with_capacity((4 * n + 6) as usize);
    let mut indices = Vec::with_capacity((12 * n) as usize);

    // Side: bottom/top ring pair per step (seam duplicated for UVs).
    for j in 0..=n {
        let a = j as f32 * step;
        let (ca, sa) = (a.cos(), a.sin());
        let normal = [ca, 0.0, sa];
        let tangent = [-sa, 0.0, ca];
        let u = j as f32 / n as f32;
        vertices.push(Vertex {
            position: [radius * ca, -half, radius * sa],
            normal,
            uv: [u, 0.0],
            tangent,
        });
        vertices.push(Vertex {
            position: [radius * ca, half, radius * sa],
            normal,
            uv: [u, 1.0],
            tangent,
        });
    }
    for j in 0..n {
        let (b0, t0) = (2 * j, 2 * j + 1);
        let (b1, t1) = (2 * j + 2, 2 * j + 3);
        indices.extend([b0, t0, t1, b0, t1, b1]);
    }

    // Caps: center + ring per cap (seam duplicated for the planar UV map).
    for (y, normal) in [(-half, [0.0, -1.0, 0.0]), (half, [0.0, 1.0, 0.0])] {
        let base = vertices.len() as u32;
        vertices.push(Vertex {
            position: [0.0, y, 0.0],
            normal,
            uv: [0.5, 0.5],
            tangent: [1.0, 0.0, 0.0],
        });
        for j in 0..=n {
            let a = j as f32 * step;
            let (ca, sa) = (a.cos(), a.sin());
            vertices.push(Vertex {
                position: [radius * ca, y, radius * sa],
                normal,
                uv: [ca * 0.5 + 0.5, sa * 0.5 + 0.5],
                tangent: [1.0, 0.0, 0.0],
            });
        }
        // Bottom fan is (center, ring_j, ring_j+1) for `-Y`; the top fan
        // is mirrored for `+Y` (verified by the winding test below).
        for j in 0..n {
            if y < 0.0 {
                indices.extend([base, base + 1 + j, base + 2 + j]);
            } else {
                indices.extend([base, base + 2 + j, base + 1 + j]);
            }
        }
    }
    (vertices, indices)
}

/// Generate an axis-aligned box with positions, normals, UVs and tangents
/// (see [`box_data`]), uploading it to `device`.
pub fn create_box(device: &wgpu::Device, size: [f32; 3]) -> Mesh {
    let (vertices, indices) = box_data(size);
    upload_vertices(
        device,
        "box vertex buffer",
        "box index buffer",
        &vertices,
        &indices,
    )
}

/// Generate a flat quad in the local XZ plane (see [`plane_data`]),
/// uploading it to `device`.
pub fn create_plane(device: &wgpu::Device, size: [f32; 2]) -> Mesh {
    let (vertices, indices) = plane_data(size);
    upload_vertices(
        device,
        "plane vertex buffer",
        "plane index buffer",
        &vertices,
        &indices,
    )
}

/// Generate a right circular cylinder around `+Y` (see [`cylinder_data`]),
/// uploading it to `device`. `radial_segments` is clamped to at least 3.
pub fn create_cylinder(
    device: &wgpu::Device,
    radius: f32,
    height: f32,
    radial_segments: u32,
) -> Mesh {
    let (vertices, indices) = cylinder_data(radius, height, radial_segments);
    upload_vertices(
        device,
        "cylinder vertex buffer",
        "cylinder index buffer",
        &vertices,
        &indices,
    )
}

/// Move CPU-side geometry into GPU buffers, exactly like [`create_sphere`].
fn upload_vertices(
    device: &wgpu::Device,
    vertex_label: &str,
    index_label: &str,
    vertices: &[Vertex],
    indices: &[u32],
) -> Mesh {
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(vertex_label),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(index_label),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Mesh {
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
        vertex_count: vertices.len() as u32,
    }
}

/// Any unit vector orthogonal to `normal` (tangent fallback for analytic
/// per-face normals); degenerate input falls back to `+X` so the vertex
/// stays finite.
fn ortho_tangent(normal: [f32; 3]) -> [f32; 3] {
    let n = glam::Vec3::from_array(normal);
    if n.length_squared() <= f32::EPSILON {
        return [1.0, 0.0, 0.0];
    }
    n.normalize().any_orthonormal_vector().to_array()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_orthogonal_unit_tangents(vertices: &[Vertex]) {
        for vertex in vertices {
            let n = glam::Vec3::from_array(vertex.normal);
            let t = glam::Vec3::from_array(vertex.tangent);
            assert!((n.length() - 1.0).abs() < 1e-6, "normal unit: {n:?}");
            assert!((t.length() - 1.0).abs() < 1e-6, "tangent unit: {t:?}");
            assert!(n.dot(t).abs() < 1e-6, "tangent orthogonal: {t:?}");
        }
    }

    /// Area-weighted face normal of one triangle (unnormalized ×2, sign =
    /// winding).
    fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> glam::Vec3 {
        let ab = glam::Vec3::from_array(b) - glam::Vec3::from_array(a);
        let ac = glam::Vec3::from_array(c) - glam::Vec3::from_array(a);
        ab.cross(ac)
    }

    #[test]
    fn box_has_24_vertices_and_36_indices() {
        let (vertices, indices) = box_data([2.0, 4.0, 6.0]);
        assert_eq!(vertices.len(), 24);
        assert_eq!(indices.len(), 36);
        assert_orthogonal_unit_tangents(&vertices);
        // Positions span exactly ±size/2 per axis.
        for (axis, extent) in [2.0, 4.0, 6.0].iter().enumerate() {
            let min = vertices
                .iter()
                .map(|v| v.position[axis])
                .fold(f32::INFINITY, f32::min);
            let max = vertices
                .iter()
                .map(|v| v.position[axis])
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(min, -extent * 0.5, "axis {axis} min");
            assert_eq!(max, extent * 0.5, "axis {axis} max");
        }
        // Every triangle winds CCW from outside (face normal agrees with
        // the stored vertex normal).
        let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
        for tri in indices.chunks_exact(3) {
            let n = face_normal(
                positions[tri[0] as usize],
                positions[tri[1] as usize],
                positions[tri[2] as usize],
            );
            let stored = glam::Vec3::from_array(vertices[tri[0] as usize].normal);
            assert!(n.dot(stored) > 0.0, "box winding: {tri:?}");
        }
    }

    #[test]
    fn plane_has_4_vertices_and_faces_up() {
        let (vertices, indices) = plane_data([3.0, 5.0]);
        assert_eq!(vertices.len(), 4);
        assert_eq!(indices, vec![0, 2, 1, 0, 3, 2]);
        assert_orthogonal_unit_tangents(&vertices);
        for vertex in &vertices {
            assert_eq!(vertex.normal, [0.0, 1.0, 0.0]);
            assert_eq!(vertex.tangent, [1.0, 0.0, 0.0]);
        }
        let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
        for tri in indices.chunks_exact(3) {
            let n = face_normal(
                positions[tri[0] as usize],
                positions[tri[1] as usize],
                positions[tri[2] as usize],
            )
            .normalize();
            assert!(
                (n - glam::Vec3::Y).length() < 1e-6,
                "plane winding faces +Y: {tri:?}"
            );
        }
    }

    #[test]
    fn cylinder_counts_clamp_and_cap_normals() {
        // Degenerate segment counts clamp to 3: 4*3+6 vertices, 12*3 indices.
        let (degenerate, degenerate_indices) = cylinder_data(1.0, 2.0, 0);
        assert_eq!(degenerate.len(), 18);
        assert_eq!(degenerate_indices.len(), 36);
        let (vertices, indices) = cylinder_data(1.5, 7.0, 8);
        assert_eq!(vertices.len(), 4 * 8 + 6);
        assert_eq!(indices.len(), 12 * 8);
        assert_orthogonal_unit_tangents(&vertices);
        // Side vertices: radial unit normals in XZ; caps: ∓Y.
        let side_count = 2 * (8 + 1);
        for vertex in &vertices[..side_count as usize] {
            let n = glam::Vec3::from_array(vertex.normal);
            assert!(n.y.abs() < 1e-6, "side normal horizontal: {n:?}");
            assert!((n.length() - 1.0).abs() < 1e-6);
        }
        for vertex in &vertices[side_count as usize..] {
            let n = glam::Vec3::from_array(vertex.normal);
            assert!(
                n == glam::Vec3::NEG_Y || n == glam::Vec3::Y,
                "cap normal vertical: {n:?}"
            );
        }
        // Winding: every triangle's face normal agrees with its stored normal.
        let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
        for tri in indices.chunks_exact(3) {
            let n = face_normal(
                positions[tri[0] as usize],
                positions[tri[1] as usize],
                positions[tri[2] as usize],
            );
            let stored = glam::Vec3::from_array(vertices[tri[0] as usize].normal);
            assert!(n.dot(stored) > 0.0, "cylinder winding: {tri:?}");
        }
    }

    #[test]
    fn skinned_vertex_shares_the_classic_prefix() {
        // 44-byte classic prefix + 16-byte joints + 16-byte weights, no
        // padding: the skinned stage reads locations 0–3 exactly like the
        // classic input, then the influences at 4–5.
        assert_eq!(std::mem::size_of::<Vertex>(), 44);
        assert_eq!(std::mem::size_of::<SkinnedVertex>(), 76);
        assert_eq!(std::mem::offset_of!(SkinnedVertex, position), 0);
        assert_eq!(std::mem::offset_of!(SkinnedVertex, normal), 12);
        assert_eq!(std::mem::offset_of!(SkinnedVertex, uv), 24);
        assert_eq!(std::mem::offset_of!(SkinnedVertex, tangent), 32);
        assert_eq!(std::mem::offset_of!(SkinnedVertex, joints), 44);
        assert_eq!(std::mem::offset_of!(SkinnedVertex, weights), 60);
        let desc = SkinnedVertex::desc();
        assert_eq!(desc.array_stride, 76);
        let attrs = desc.attributes;
        assert_eq!(attrs.len(), 6);
        let expected = [
            (0u64, 0u32, wgpu::VertexFormat::Float32x3),
            (12, 1, wgpu::VertexFormat::Float32x3),
            (24, 2, wgpu::VertexFormat::Float32x2),
            (32, 3, wgpu::VertexFormat::Float32x3),
            (44, 4, wgpu::VertexFormat::Uint32x4),
            (60, 5, wgpu::VertexFormat::Float32x4),
        ];
        for (attr, (offset, location, format)) in attrs.iter().zip(expected) {
            assert_eq!(attr.offset, offset, "location {location} offset");
            assert_eq!(attr.shader_location, location);
            assert_eq!(attr.format, format);
        }
    }
}
