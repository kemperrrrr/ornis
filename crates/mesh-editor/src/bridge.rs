//! Bridge between [`crate::MeshData`] and the `manifold-rust` kernel.
//!
//! Positions cross as `f32` → `f64`; only the first three properties
//! travel (xyz) — normals/uvs are editor-side attributes recomputed after
//! topological ops. Unstitched duplicate vertices are welded with
//! `merge()` before import so coincident corners do not read as holes.

use manifold_rust::{manifold::Manifold, types::Error as ManifoldError};

use crate::mesh_data::TRIANGLE_VERTS;

/// Spatial components in a position.
const VEC3_COMPONENTS: usize = 3;

/// Kernel-side failure of a boolean operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BridgeError {
    /// Input tripped the engine-side shape check.
    #[error("mesh data failed validation")]
    InvalidInput,
    /// Kernel reported a non-manifold result or construction failure.
    #[error("manifold kernel reported an error")]
    KernelFailed,
}

/// Convert validated [`crate::MeshData`] into a kernel [`Manifold`].
///
/// Welds duplicate vertices before import.
///
/// # Errors
///
/// Returns [`BridgeError::InvalidInput`] when validation fails.
pub fn to_manifold(mesh: &crate::MeshData) -> Result<Manifold, BridgeError> {
    mesh.validate().map_err(|_| BridgeError::InvalidInput)?;
    let mut gl = manifold_rust::types::MeshGL {
        num_prop: TRIANGLE_VERTS as u32,
        vert_properties: mesh
            .positions
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        tri_verts: mesh.indices.clone(),
        ..Default::default()
    };
    gl.merge();
    Ok(Manifold::from_mesh_gl(&gl))
}

/// Convert a kernel [`Manifold`] back into [`crate::MeshData`].
///
/// Positions come from the kernel; normals are an analytic placeholder
/// (unit +Y) and uvs are zeroed — the editor recomputes shading attributes
/// per affected region after the boolean lands.
pub fn from_manifold(manifold: &Manifold) -> crate::MeshData {
    let gl = manifold.get_mesh_gl(-1);
    let positions: Vec<[f32; VEC3_COMPONENTS]> =
        (0..gl.num_vert()).map(|i| gl.get_vert_pos(i)).collect();
    let n = positions.len();
    crate::MeshData {
        positions,
        normals: vec![[0.0, 1.0, 0.0]; n],
        uvs: vec![[0.0; 2]; n],
        indices: gl.tri_verts.clone(),
    }
}

/// Combine two meshes with a CSG operation.
///
/// # Errors
///
/// Returns [`BridgeError`] when an input is invalid or the kernel reports
/// a non-manifold result.
pub fn boolean(
    base: &crate::MeshData,
    tool: &crate::MeshData,
    kind: crate::BooleanKind,
) -> Result<crate::MeshData, BridgeError> {
    let a = to_manifold(base)?;
    let b = to_manifold(tool)?;
    let out = match kind {
        crate::BooleanKind::Union => a.union(&b),
        crate::BooleanKind::Subtract => a.difference(&b),
        crate::BooleanKind::Intersect => a.intersection(&b),
    };
    if out.status() != ManifoldError::NoError {
        return Err(BridgeError::KernelFailed);
    }
    Ok(from_manifold(&out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_of_overlapping_boxes_is_nonempty() {
        let a = crate::MeshData::unit_box();
        let mut b = crate::MeshData::unit_box();
        for p in &mut b.positions {
            p[0] += 0.5;
        }
        let out = boolean(&a, &b, crate::BooleanKind::Union).expect("union works");
        assert!(out.triangle_count() > 0);
        assert!(out.vertex_count() > 0);
    }
}
