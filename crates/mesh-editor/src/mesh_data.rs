//! Canonical editable mesh shared by render, physics and WASM views.
//!
//! [`MeshData`] stores an indexed triangle soup in engine units (`f32`).
//! All attribute arrays run parallel to `positions`: `normals[i]` and
//! `uvs[i]` describe `positions[i]`. Triangle `t` owns indices
//! `indices[3*t..3*t+3]`, wound counter-clockwise seen from outside.
//! Manifold input additionally requires a closed, orientation-consistent
//! shell — [`MeshData::validate`] checks shapes the engine can check
//! cheaply (lengths, bounds); closedness is reported back by the kernel
//! via [`crate::BridgeError`].

/// Canonical editable triangle mesh (engine source of truth).
#[derive(Debug, Clone, Default)]
pub struct MeshData {
    /// Vertex positions in engine units.
    pub positions: Vec<[f32; 3]>,
    /// Unit-length shading normals, one per position.
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates in [0, 1], one per position.
    pub uvs: Vec<[f32; 2]>,
    /// Triangle index list (`u32`, triples, CCW from outside).
    pub indices: Vec<u32>,
}

/// Shape violation found by [`MeshData::validate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshError {
    /// `indices` length is not a multiple of 3.
    IndexCountNotMultipleOfThree,
    /// An index points past the end of `positions`.
    IndexOutOfBounds,
    /// `normals` length differs from `positions` length.
    NormalCountMismatch,
    /// `uvs` length differs from `positions` length.
    UvCountMismatch,
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexCountNotMultipleOfThree => write!(f, "indices not a multiple of 3"),
            Self::IndexOutOfBounds => write!(f, "index out of bounds"),
            Self::NormalCountMismatch => write!(f, "normals/positions length mismatch"),
            Self::UvCountMismatch => write!(f, "uvs/positions length mismatch"),
        }
    }
}

impl std::error::Error for MeshError {}

impl MeshData {
    /// Build a mesh from raw parts, checking invariants.
    ///
    /// # Errors
    ///
    /// Returns [`MeshError`] when index/attribute lengths are inconsistent.
    pub fn new(
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<u32>,
    ) -> Result<Self, MeshError> {
        let mesh = Self {
            positions,
            normals,
            uvs,
            indices,
        };
        mesh.validate()?;
        Ok(mesh)
    }

    /// Check index/attribute invariants without copying.
    ///
    /// # Errors
    ///
    /// Returns [`MeshError`] on the first violation found.
    pub fn validate(&self) -> Result<(), MeshError> {
        if !self.indices.len().is_multiple_of(3) {
            return Err(MeshError::IndexCountNotMultipleOfThree);
        }
        if self.normals.len() != self.positions.len() {
            return Err(MeshError::NormalCountMismatch);
        }
        if self.uvs.len() != self.positions.len() {
            return Err(MeshError::UvCountMismatch);
        }
        let n = self.positions.len() as u32;
        if self.indices.iter().any(|&i| i >= n) {
            return Err(MeshError::IndexOutOfBounds);
        }
        Ok(())
    }

    /// Number of vertices (`positions.len()`).
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangles (`indices.len() / 3`).
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Axis-aligned unit box centered at the origin (12 triangles).
    ///
    /// Normals are analytic (per-face), uvs are a placeholder planar map;
    /// both are recomputed by the editor after boolean operations anyway.
    pub fn unit_box() -> Self {
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
        let mut positions = Vec::with_capacity(24);
        let mut normals_out = Vec::with_capacity(24);
        let mut uvs = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);
        for (face, n) in faces.iter().zip(normals) {
            let base = positions.len() as u32;
            for (k, &vi) in face.iter().enumerate() {
                positions.push(p[vi]);
                normals_out.push(n);
                uvs.push([(k == 1 || k == 2) as u8 as f32, (k >= 2) as u8 as f32]);
            }
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Self {
            positions,
            normals: normals_out,
            uvs,
            indices,
        }
    }

    /// Build a mesh from a vertex soup: positions plus a triangle index
    /// list. Uvs are zeroed and normals are a `+Y` placeholder — call
    /// [`Self::with_computed_normals`] after loading to get shading
    /// normals. This is the load path for `MeshDesc::Custom`.
    ///
    /// # Errors
    ///
    /// Returns [`MeshError`] when index/attribute lengths are inconsistent.
    pub fn from_positions(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Result<Self, MeshError> {
        let n = positions.len();
        let mesh = Self {
            positions,
            normals: vec![[0.0, 1.0, 0.0]; n],
            uvs: vec![[0.0; 2]; n],
            indices,
        };
        mesh.validate()?;
        Ok(mesh)
    }

    /// Recompute smooth per-vertex normals with area-weighted face
    /// contributions. Degenerate triangles contribute nothing; vertices
    /// untouched by any triangle keep their current normal.
    pub fn with_computed_normals(mut self) -> Self {
        let mut acc = vec![[0.0f32; 3]; self.positions.len()];
        for tri in self.indices.chunks_exact(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            // `validate` guarantees in-bounds indices, but don't panic on
            // hand-built callers that skipped it.
            let (Some(&pa), Some(&pb), Some(&pc)) = (
                self.positions.get(a),
                self.positions.get(b),
                self.positions.get(c),
            ) else {
                continue;
            };
            let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            // Cross product = 2 × area-weighted face normal.
            let n = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            for idx in [a, b, c] {
                acc[idx][0] += n[0];
                acc[idx][1] += n[1];
                acc[idx][2] += n[2];
            }
        }
        for (normal, a) in self.normals.iter_mut().zip(acc) {
            let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            if len > 0.0 {
                *normal = [a[0] / len, a[1] / len, a[2] / len];
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_box_is_valid_with_12_triangles() {
        let mesh = MeshData::unit_box();
        mesh.validate().expect("unit box valid");
        assert_eq!(mesh.triangle_count(), 12);
        assert_eq!(mesh.vertex_count(), 24);
    }

    #[test]
    fn rejects_dangling_index() {
        let mesh = MeshData {
            positions: vec![[0.0; 3]],
            normals: vec![[0.0, 1.0, 0.0]],
            uvs: vec![[0.0; 2]],
            indices: vec![0, 0, 7],
        };
        assert_eq!(mesh.validate(), Err(MeshError::IndexOutOfBounds));
    }

    #[test]
    fn rejects_attribute_mismatch() {
        let mesh = MeshData {
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 1.0, 0.0]],
            uvs: vec![[0.0; 2]; 3],
            indices: vec![0, 1, 2],
        };
        assert_eq!(mesh.validate(), Err(MeshError::NormalCountMismatch));
    }

    #[test]
    fn from_positions_builds_valid_placeholder_mesh() {
        let mesh = MeshData::from_positions(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        )
        .expect("single triangle valid");
        mesh.validate().expect("from_positions output valid");
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.triangle_count(), 1);
        assert!(mesh.normals.iter().all(|&n| n == [0.0, 1.0, 0.0]));
        assert!(mesh.uvs.iter().all(|&uv| uv == [0.0; 2]));
    }

    #[test]
    fn from_positions_rejects_bad_indices() {
        assert_eq!(
            MeshData::from_positions(vec![[0.0; 3]], vec![0, 1]).unwrap_err(),
            MeshError::IndexCountNotMultipleOfThree
        );
        assert_eq!(
            MeshData::from_positions(vec![[0.0; 3]], vec![0, 0, 5]).unwrap_err(),
            MeshError::IndexOutOfBounds
        );
    }

    #[test]
    fn with_computed_normals_recovers_flat_normal() {
        // Winding (0,1,2): ab=(0,0,1), ac=(1,0,0), ab×ac=(0,1,0).
        let mesh = MeshData::from_positions(
            vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            vec![0, 1, 2],
        )
        .expect("valid")
        .with_computed_normals();
        mesh.validate().expect("normals stay length-matched");
        for n in &mesh.normals {
            assert!(
                n[0].abs() < 1e-6 && (n[1] - 1.0).abs() < 1e-6 && n[2].abs() < 1e-6,
                "expected +Y, got {n:?}"
            );
        }
    }

    #[test]
    fn with_computed_normals_averages_adjacent_faces() {
        // Two perpendicular unit triangles sharing edge 0-1: A in the
        // y=0 plane (normal +Y), B in the x=0 plane (normal -X). Shared
        // vertices must get the normalized (-1,1,0) sum.
        let mesh = MeshData::from_positions(
            vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![0, 1, 2, 0, 1, 3],
        )
        .expect("valid")
        .with_computed_normals();
        let s = std::f32::consts::FRAC_1_SQRT_2;
        for &i in &[0, 1] {
            let n = mesh.normals[i];
            assert!(
                (n[0] + s).abs() < 1e-5 && (n[1] - s).abs() < 1e-5 && n[2].abs() < 1e-5,
                "vertex {i}: expected [{}, {s}, 0], got {n:?}",
                -s,
            );
        }
        assert_eq!(mesh.normals[2], [0.0, 1.0, 0.0]);
        assert_eq!(mesh.normals[3], [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn with_computed_normals_keeps_placeholder_on_degenerate() {
        // Collinear triangle: zero area, no contribution.
        let mesh = MeshData::from_positions(
            vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            vec![0, 1, 2],
        )
        .expect("valid")
        .with_computed_normals();
        assert!(mesh.normals.iter().all(|&n| n == [0.0, 1.0, 0.0]));
    }
}
