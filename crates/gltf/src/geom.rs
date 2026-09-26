//! Pure geometry helpers: column-major `Mat4` TRS math plus the two
//! `custom_mesh_data` fallbacks (area-weighted normals, box-projection uvs).
//!
//! Everything here is `f32`-array math with no engine dependency so the
//! fallbacks stay byte-compatible with the upload path by construction
//! (same formulas, cited per function).

/// Column-major `4×4` matrix: `m[column][row]`, as in glTF.
pub(crate) type Mat4 = [[f32; 4]; 4];

/// Identity matrix.
pub(crate) const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Matrix product `a * b` (column-major): applies `b` first, then `a`.
pub(crate) fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k][row] * b[col][k];
            }
            out[col][row] = sum;
        }
    }
    out
}

/// Compose `T * R * S` from translation, unit `(x, y, z, w)` quaternion
/// and per-axis scale (the `Transform::matrix` convention).
pub(crate) fn mat_from_trs(t: [f32; 3], r: [f32; 4], s: [f32; 3]) -> Mat4 {
    let rot = quat_to_mat3(r);
    let mut out = IDENTITY;
    for col in 0..3 {
        for row in 0..3 {
            out[col][row] = rot[col][row] * s[col];
        }
    }
    out[3][0] = t[0];
    out[3][1] = t[1];
    out[3][2] = t[2];
    out
}

/// Splits a world matrix into `(translation, rotation, scale)`.
///
/// Rotation comes from the normalized basis via [`quat_from_mat3`]; a
/// mirrored basis (`det < 0`) bakes the flip into `scale.x` so the
/// quaternion stays a pure rotation. A degenerate basis (any axis near
/// zero) yields identity rotation with the measured scale kept — flattening
/// a zero scale into a fake rotation would be less honest.
pub(crate) fn decompose(mat: &Mat4) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let translation = [mat[3][0], mat[3][1], mat[3][2]];
    let mut cols = [
        [mat[0][0], mat[0][1], mat[0][2]],
        [mat[1][0], mat[1][1], mat[1][2]],
        [mat[2][0], mat[2][1], mat[2][2]],
    ];
    let mut scale = [0.0f32; 3];
    for axis in 0..3 {
        let (x, y, z) = (cols[axis][0], cols[axis][1], cols[axis][2]);
        scale[axis] = (x * x + y * y + z * z).sqrt();
    }
    if scale.iter().all(|s| *s > f32::EPSILON) {
        for (col, factor) in cols.iter_mut().zip(&scale) {
            for x in col.iter_mut() {
                *x /= *factor;
            }
        }
        if mat3_det(&cols) < 0.0 {
            scale[0] = -scale[0];
            for x in cols[0].iter_mut() {
                *x = -*x;
            }
        }
        (translation, quat_from_mat3(&cols), scale)
    } else {
        (translation, [0.0, 0.0, 0.0, 1.0], scale)
    }
}

/// Rotation matrix columns from a unit `(x, y, z, w)` quaternion.
fn quat_to_mat3(q: [f32; 4]) -> [[f32; 3]; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)],
        [2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)],
        [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)],
    ]
}

/// Unit `(x, y, z, w)` quaternion from orthonormal basis columns.
fn quat_from_mat3(m: &[[f32; 3]; 3]) -> [f32; 4] {
    // Columns are basis vectors; index as rows: m[col][row].
    let trace = m[0][0] + m[1][1] + m[2][2];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (m[1][2] - m[2][1]) / s,
            (m[2][0] - m[0][2]) / s,
            (m[0][1] - m[1][0]) / s,
            0.25 * s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] - m[2][1]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[2][0] - m[0][2]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[0][1] - m[1][0]) / s,
        ]
    }
}

/// Determinant of a `3×3` given as columns.
fn mat3_det(m: &[[f32; 3]; 3]) -> f32 {
    let (a, b, c) = (m[0], m[1], m[2]);
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

/// Area-weighted smooth normals; mirrors `MeshData::with_computed_normals`.
///
/// Face contributions accumulate unnormalized (`(b-a)×(c-a)`, i.e. twice the
/// area-weighted face normal); degenerate triangles add nothing and
/// vertices no triangle touches keep `+Y`. Out-of-range indices are skipped
/// (the importer rejects them; this stays panic-free regardless).
pub(crate) fn area_weighted_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![[0.0f32; 3]; positions.len()];
    for tri in indices
        .chunks_exact(3)
        .map(|c| crate::Triangle::from_raw([c[0], c[1], c[2]]))
    {
        let raw = tri.as_u32();
        let (Some(&pa), Some(&pb), Some(&pc)) = (
            positions.get(raw[0] as usize),
            positions.get(raw[1] as usize),
            positions.get(raw[2] as usize),
        ) else {
            continue;
        };
        let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for idx in raw {
            if let Some(slot) = acc.get_mut(idx as usize) {
                slot[0] += n[0];
                slot[1] += n[1];
                slot[2] += n[2];
            }
        }
    }
    acc.into_iter()
        .map(|a| {
            let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            if len > 0.0 {
                [a[0] / len, a[1] / len, a[2] / len]
            } else {
                [0.0, 1.0, 0.0]
            }
        })
        .collect()
}

/// Box-projection uvs; mirrors `apply_box_project_uvs` in `mesh_upload.rs`.
///
/// Each vertex maps planarly from its dominant normal axis over the soup
/// bounds into `[0, 1]`; degenerate spans collapse to `0.5` so output stays
/// finite on degenerate soups.
pub(crate) fn box_project_uvs(positions: &[[f32; 3]], normals: &[[f32; 3]]) -> Vec<[f32; 2]> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for position in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    let normalized = |value: f32, axis: usize| -> f32 {
        let span = max[axis] - min[axis];
        if span <= f32::EPSILON {
            0.5
        } else {
            ((value - min[axis]) / span).clamp(0.0, 1.0)
        }
    };
    positions
        .iter()
        .zip(normals.iter().chain(std::iter::repeat(&[0.0, 1.0, 0.0])))
        .map(|(position, normal)| {
            let (nx, ny, nz) = (normal[0].abs(), normal[1].abs(), normal[2].abs());
            if nx >= ny && nx >= nz {
                [normalized(position[2], 2), normalized(position[1], 1)]
            } else if ny >= nz {
                [normalized(position[0], 0), normalized(position[2], 2)]
            } else {
                [normalized(position[0], 0), normalized(position[1], 1)]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_composes_and_decomposes() {
        let mat = mat_from_trs([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]);
        assert_eq!(mat, IDENTITY);
        let (t, r, s) = decompose(&IDENTITY);
        assert_eq!(t, [0.0, 0.0, 0.0]);
        assert_eq!(r, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(s, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn trs_round_trip_preserves_components() {
        // 90° about Y: (x, y, z, w).
        let half = std::f32::consts::FRAC_PI_4;
        let rotation = [0.0, half.sin(), 0.0, half.cos()];
        let (t, s) = ([1.0, 2.0, 3.0], [2.0, 0.5, 4.0]);
        let mat = mat_from_trs(t, rotation, s);
        let (rt, rr, rs) = decompose(&mat);
        assert_eq!(rt, t);
        for (got, want) in rs.iter().zip(s) {
            assert!((got - want).abs() < 1e-6, "scale {rs:?} vs {s:?}");
        }
        // Quaternion sign is free; compare rotation matrices instead.
        let (got, want) = (quat_to_mat3(rr), quat_to_mat3(rotation));
        for (gc, wc) in got.iter().zip(want) {
            for (g, w) in gc.iter().zip(wc) {
                assert!((g - w).abs() < 1e-6, "rotation {got:?} vs {want:?}");
            }
        }
    }

    #[test]
    fn parent_child_matrices_compose() {
        let parent = mat_from_trs([5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]);
        let child = mat_from_trs([0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0]);
        let (t, _, s) = decompose(&mat_mul(&parent, &child));
        assert_eq!(t, [5.0, 3.0, 0.0]);
        assert_eq!(s, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn degenerate_scale_keeps_scale_with_identity_rotation() {
        let mut mat = IDENTITY;
        mat[0] = [0.0, 0.0, 0.0, 0.0];
        let (t, r, s) = decompose(&mat);
        assert_eq!(t, [0.0, 0.0, 0.0]);
        assert_eq!(r, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(s[0], 0.0);
    }

    #[test]
    fn flat_triangle_gets_face_normal() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let normals = area_weighted_normals(&positions, &[0, 1, 2]);
        // (b-a)×(c-a) = (1,0,0)×(0,0,1) = (0,-1,0).
        assert_eq!(normals, [[0.0, -1.0, 0.0]; 3]);
    }

    #[test]
    fn degenerate_triangle_keeps_up_normal() {
        let positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let normals = area_weighted_normals(&positions, &[0, 1, 2]);
        assert_eq!(normals, [[0.0, 1.0, 0.0]; 3]);
    }

    #[test]
    fn box_projection_maps_dominant_axis_in_unit_range() {
        // +Y normals → (x, z) over the [0, 2]×[0, 4] bounds.
        let positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 4.0]];
        let normals = [[0.0, 1.0, 0.0]; 2];
        let uvs = box_project_uvs(&positions, &normals);
        assert_eq!(uvs, [[0.0, 0.0], [1.0, 1.0]]);
    }

    #[test]
    fn box_projection_collapses_degenerate_span() {
        let positions = [[1.0, 2.0, 3.0]; 2];
        let normals = [[0.0, 0.0, 1.0]; 2];
        assert_eq!(box_project_uvs(&positions, &normals), [[0.5, 0.5]; 2]);
    }
}
