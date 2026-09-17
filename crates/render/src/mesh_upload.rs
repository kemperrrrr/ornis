//! Upload [`ornis_mesh_editor::MeshData`] to the GPU as a [`Mesh`].
//!
//! Adapter direction is one-way: `ornis-render` depends on
//! `ornis-mesh-editor`, never the reverse. The pure conversion
//! ([`to_vertices`]) is `wgpu`-free so it stays unit-testable; the thin
//! [`upload_mesh_data`] wrapper only moves the converted arrays into GPU
//! buffers, mirroring [`crate::mesh::create_sphere`].

use wgpu::util::DeviceExt;

use crate::mesh::{Mesh, Vertex};

/// Upload of [`ornis_mesh_editor::MeshData`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadError {
    /// [`ornis_mesh_editor::MeshData`] failed [`ornis_mesh_editor::MeshData::validate`].
    InvalidMesh(ornis_mesh_editor::MeshError),
}

impl std::fmt::Display for UploadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMesh(inner) => write!(f, "invalid mesh data: {inner}"),
        }
    }
}

impl std::error::Error for UploadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidMesh(inner) => Some(inner),
        }
    }
}

/// Convert [`ornis_mesh_editor::MeshData`] into GPU-ready vertices + indices.
///
/// Positions, normals and uvs are copied verbatim; the tangent is an
/// orthogonal fallback derived from the normal
/// (`glam::Vec3::any_orthonormal_vector`): this is NOT a MikkTSpace tangent,
/// so normal mapping on uploaded meshes is approximate (correct silhouette
/// and lighting, slightly skewed tangent-space detail). Meshes needing
/// exact normal-mapped detail should carry precomputed tangents instead.
///
/// # Errors
///
/// Returns [`UploadError::InvalidMesh`] when `data` fails validation.
pub fn to_vertices(
    data: &ornis_mesh_editor::MeshData,
) -> Result<(Vec<Vertex>, Vec<u32>), UploadError> {
    data.validate().map_err(UploadError::InvalidMesh)?;
    let vertices = data
        .positions
        .iter()
        .zip(&data.normals)
        .zip(&data.uvs)
        .map(|((&position, &normal), &uv)| Vertex {
            position,
            normal,
            uv,
            tangent: fallback_tangent(normal),
        })
        .collect();
    Ok((vertices, data.indices.clone()))
}

/// Upload validated [`ornis_mesh_editor::MeshData`] to `device` as a [`Mesh`].
///
/// Buffers are created with [`wgpu::BufferUsages::VERTEX`] /
/// [`wgpu::BufferUsages::INDEX`] via `create_buffer_init`, exactly like
/// [`crate::mesh::create_sphere`].
///
/// # Errors
///
/// Returns [`UploadError::InvalidMesh`] when `data` fails validation.
pub fn upload_mesh_data(
    device: &wgpu::Device,
    data: &ornis_mesh_editor::MeshData,
) -> Result<Mesh, UploadError> {
    let (vertices, indices) = to_vertices(data)?;
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh upload vertex buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh upload index buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Ok(Mesh {
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
        vertex_count: vertices.len() as u32,
    })
}

/// Any unit vector orthogonal to `normal` (tangent fallback, see [`to_vertices`]).
fn fallback_tangent(normal: [f32; 3]) -> [f32; 3] {
    let n = glam::Vec3::from_array(normal);
    // Degenerate (zero-length) normals have no orthogonal direction;
    // fall back to +X so the vertex stays finite.
    if n.length_squared() <= f32::EPSILON {
        return [1.0, 0.0, 0.0];
    }
    n.normalize().any_orthonormal_vector().to_array()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_box_converts_to_24_vertices_and_36_indices() {
        let data = ornis_mesh_editor::MeshData::unit_box();
        let (vertices, indices) = to_vertices(&data).expect("unit box valid");
        assert_eq!(vertices.len(), 24);
        assert_eq!(indices.len(), 36);
        assert_eq!(indices, data.indices);
        for (vertex, (&position, &normal, &uv)) in vertices.iter().zip(
            data.positions
                .iter()
                .zip(&data.normals)
                .zip(&data.uvs)
                .map(|((p, n), u)| (p, n, u)),
        ) {
            assert_eq!(vertex.position, position);
            assert_eq!(vertex.normal, normal);
            assert_eq!(vertex.uv, uv);
        }
    }

    #[test]
    fn fallback_tangents_are_unit_and_orthogonal_to_normals() {
        let data = ornis_mesh_editor::MeshData::unit_box();
        let (vertices, _) = to_vertices(&data).expect("unit box valid");
        for vertex in &vertices {
            let n = glam::Vec3::from_array(vertex.normal);
            let t = glam::Vec3::from_array(vertex.tangent);
            assert!((t.length() - 1.0).abs() < 1e-6, "tangent unit: {t:?}");
            assert!(n.dot(t).abs() < 1e-6, "tangent orthogonal: {t:?}");
        }
    }

    #[test]
    fn invalid_mesh_is_rejected() {
        let data = ornis_mesh_editor::MeshData {
            positions: vec![[0.0; 3]],
            normals: vec![[0.0, 1.0, 0.0]],
            uvs: vec![[0.0; 2]],
            indices: vec![0, 0, 7],
        };
        assert!(matches!(
            to_vertices(&data),
            Err(UploadError::InvalidMesh(_))
        ));
    }
}
