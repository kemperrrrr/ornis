//! Shading-normal recomputation and physics-view extraction.
//!
//! Normals are editor-side attributes: the kernel drops them, so they are
//! rebuilt area-weighted after every topological op, either for the whole
//! mesh or for a dirty region plus its one-ring seam.

use std::collections::HashSet;

/// Recompute smooth area-weighted normals in place.
///
/// With `faces == None` every normal is rebuilt. With `Some(faces)` only
/// vertices of those faces are rewritten, but accumulation runs over the
/// one-ring seam (all faces touching those vertices) so region borders
/// shade continuously with untouched geometry.
pub fn recompute_normals(mesh: &mut crate::MeshData, faces: Option<&[u32]>) {
    let tri_count = mesh.triangle_count();
    let Some(faces) = faces else {
        accumulate_all(mesh, 0..tri_count);
        return;
    };
    let mut verts = HashSet::new();
    for &f in faces {
        let f = f as usize;
        if f >= tri_count {
            continue;
        }
        for k in 0..3 {
            verts.insert(mesh.indices[3 * f + k]);
        }
    }
    if verts.is_empty() {
        return;
    }
    // Seam: every face touching a dirty vertex contributes.
    let mut seam = Vec::new();
    for f in 0..tri_count {
        if (0..3).any(|k| verts.contains(&mesh.indices[3 * f + k])) {
            seam.push(f);
        }
    }
    // Zero only dirty vertices, accumulate over the seam, normalize dirty.
    let zero = [0.0f32; 3];
    for &v in &verts {
        mesh.normals[v as usize] = zero;
    }
    accumulate_faces(mesh, &seam, Some(&verts));
    for &v in &verts {
        let v = v as usize;
        let n = mesh.normals[v];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > f32::EPSILON {
            mesh.normals[v] = [n[0] / len, n[1] / len, n[2] / len];
        } else {
            mesh.normals[v] = [0.0, 1.0, 0.0];
        }
    }
}

/// Extract vertex/triangle arrays for `TriMesh::from_indexed`-style import.
///
/// Returns positions as [`glam::Vec3`] and triangles as index triples.
pub fn to_physics_arrays(mesh: &crate::MeshData) -> (Vec<glam::Vec3>, Vec<[u32; 3]>) {
    let verts = mesh
        .positions
        .iter()
        .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
        .collect();
    let tris = mesh
        .indices
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    (verts, tris)
}

/// Budget-gated [`to_physics_arrays`]: over-budget meshes are denied BEFORE
/// any allocation, so a huge edit never hitches the frame building physics
/// arrays it cannot use. The caller keeps the old collider and retries
/// ([`RefitDecision::Defer`](crate::RefitDecision)) on a later frame or in
/// a background pass.
///
/// # Errors
///
/// Returns [`RefitDefer`](crate::RefitDefer) when `mesh.triangle_count()`
/// exceeds the budget. Never panics on size: the count check precedes the
/// conversion.
pub fn to_physics_arrays_gated(
    mesh: &crate::MeshData,
    budget: &crate::RefitBudget,
) -> Result<(Vec<glam::Vec3>, Vec<[u32; 3]>), crate::RefitDefer> {
    let tris = mesh.triangle_count();
    if budget.decide(tris) == crate::RefitDecision::Defer {
        return Err(crate::RefitDefer {
            tris,
            max_tris: budget.max_tris,
        });
    }
    Ok(to_physics_arrays(mesh))
}

/// Zero all normals, accumulate over `range`, normalize everything.
fn accumulate_all(mesh: &mut crate::MeshData, range: std::ops::Range<usize>) {
    for n in &mut mesh.normals {
        *n = [0.0; 3];
    }
    let faces: Vec<usize> = range.collect();
    accumulate_faces(mesh, &faces, None);
    for n in &mut mesh.normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > f32::EPSILON {
            *n = [n[0] / len, n[1] / len, n[2] / len];
        } else {
            *n = [0.0, 1.0, 0.0];
        }
    }
}

/// Add each face's area-weighted (unnormalized cross product) normal to its
/// corners; when `filter` is set, only those vertices accumulate.
fn accumulate_faces(mesh: &mut crate::MeshData, faces: &[usize], filter: Option<&HashSet<u32>>) {
    for &f in faces {
        let [a, b, c] = [
            mesh.indices[3 * f],
            mesh.indices[3 * f + 1],
            mesh.indices[3 * f + 2],
        ];
        let pa = mesh.positions[a as usize];
        let pb = mesh.positions[b as usize];
        let pc = mesh.positions[c as usize];
        let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        // Cross product magnitude equals 2x triangle area: planar regions
        // correctly outweigh slivers without an explicit area pass.
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for v in [a, b, c] {
            if filter.is_some_and(|set| !set.contains(&v)) {
                continue;
            }
            let dst = &mut mesh.normals[v as usize];
            dst[0] += n[0];
            dst[1] += n[1];
            dst[2] += n[2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_recompute_yields_unit_normals() {
        let mut mesh = crate::MeshData::unit_box();
        recompute_normals(&mut mesh, None);
        for n in &mesh.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "len {len}");
        }
    }

    #[test]
    fn regional_recompute_matches_full() {
        let mut full = crate::MeshData::unit_box();
        recompute_normals(&mut full, None);
        let mut part = crate::MeshData::unit_box();
        let faces: Vec<u32> = (0..part.triangle_count() as u32).collect();
        recompute_normals(&mut part, Some(&faces));
        assert_eq!(full.normals, part.normals);
    }

    #[test]
    fn gated_bridge_defers_over_budget_before_allocating() {
        let mesh = crate::MeshData::unit_box(); // 12 triangles.
        let tight = crate::RefitBudget::new(crate::Millis::new(2), 11);
        let denial = to_physics_arrays_gated(&mesh, &tight)
            .expect_err("12 tris over an 11-tri budget defers");
        assert_eq!(denial.tris, 12);
        assert_eq!(denial.max_tris, 11);
        assert_eq!(denial.decision(), crate::RefitDecision::Defer);
        let roomy = crate::RefitBudget::new(crate::Millis::new(2), 12);
        let (verts, tris) = to_physics_arrays_gated(&mesh, &roomy).expect("exact fit rebuilds");
        assert_eq!(verts.len(), mesh.vertex_count());
        assert_eq!(tris.len(), mesh.triangle_count());
    }

    #[test]
    fn physics_arrays_roundtrip_counts() {
        let mesh = crate::MeshData::unit_box();
        let (verts, tris) = to_physics_arrays(&mesh);
        assert_eq!(verts.len(), mesh.vertex_count());
        assert_eq!(tris.len(), mesh.triangle_count());
    }
}
