//! Upload [`ornis_mesh_editor::MeshData`] to the GPU as a [`Mesh`].
//!
//! Adapter direction is one-way: `ornis-render` depends on
//! `ornis-mesh-editor`, never the reverse. The pure conversion
//! ([`to_vertices`]) is `wgpu`-free so it stays unit-testable; the thin
//! [`upload_mesh_data`] wrapper only moves the converted arrays into GPU
//! buffers, mirroring [`crate::mesh::create_sphere`].

use wgpu::util::DeviceExt;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::mesh::{Mesh, Vertex};

/// Spatial components in a position / normal.
const VEC3_COMPONENTS: usize = 3;
/// UV fallback for a degenerate projected span.
const HALF: f32 = 0.5;

/// Upload of [`ornis_mesh_editor::MeshData`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum UploadError {
    /// [`ornis_mesh_editor::MeshData`] failed [`ornis_mesh_editor::MeshData::validate`].
    ///
    /// The index-out-of-range case surfaces here as
    /// [`ornis_mesh_editor::MeshError::IndexOutOfBounds`] (see
    /// [`Self::is_index_out_of_range`]); callers skip the entity, never
    /// a stub mesh.
    #[error("invalid mesh data: {0}")]
    InvalidMesh(#[source] ornis_mesh_editor::MeshError),
    /// Inline `MeshDesc::Custom` soup has no vertices or no indices —
    /// there is no honest GPU mesh for it (callers skip the entity,
    /// never a sphere stub).
    #[error("custom mesh has no vertices or indices")]
    EmptyMesh,
}

impl UploadError {
    /// True for the empty-soup case ([`Self::EmptyMesh`]).
    pub const fn is_empty_soup(&self) -> bool {
        matches!(self, Self::EmptyMesh)
    }

    /// True when an index points past the end of the positions
    /// ([`ornis_mesh_editor::MeshError::IndexOutOfBounds`] wrapped in
    /// [`Self::InvalidMesh`]).
    pub const fn is_index_out_of_range(&self) -> bool {
        matches!(
            self,
            Self::InvalidMesh(ornis_mesh_editor::MeshError::IndexOutOfBounds)
        )
    }
}

/// Deterministic identity of one inline `MeshDesc::Custom` soup.
///
/// Newtype over the raw `u64` digest so cache maps cannot be mixed up
/// with unrelated hashes (entity ids, material slots). Frame-local:
/// built with `DefaultHasher`, so equal soups hash equally inside one
/// process run, but values are not stable across runs and must never be
/// serialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoupHash(u64);

impl SoupHash {
    /// Raw digest value (map-key material only).
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Rebuild from a raw digest (round-trip for maps and tests).
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
}

/// Outcome of one [`SoupCache`] lookup: whether the soup conversion was
/// reused or computed fresh. The extraction copies the outcome into
/// [`crate::extraction::ExtractionStats`] (`custom_cache_hits` /
/// `custom_cache_misses`), so per-frame dedup stays observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadCache {
    /// Identical soup already converted this frame — stored vertices
    /// cloned, no normal/UV recompute.
    Hit,
    /// Distinct soup — converted once and stored for later hits.
    Miss,
}

/// One converted Custom soup: GPU-ready vertices plus the triangle index
/// list. Alias over the bare tuple so cached-conversion signatures stay
/// readable (and under the `type_complexity` gate).
pub type ConvertedSoup = (Vec<Vertex>, Vec<u32>);

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

/// Build editor mesh data from an inline `MeshDesc::Custom` soup.
///
/// Positions plus a triangle index list; normals are recomputed
/// area-weighted (`with_computed_normals`), so shading normals are never
/// transported — the same contract as the physics `to_physics_arrays`
/// import. Uvs carry no transport either, so they are rebuilt here with a
/// box projection (planar map from the dominant normal axis, normalized
/// over the soup bounds into `[0, 1]`). Pure and `wgpu`-free.
///
/// # Errors
///
/// Returns [`UploadError::EmptyMesh`] when either slice is empty,
/// [`UploadError::InvalidMesh`] when validation fails.
pub fn custom_mesh_data(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Result<ornis_mesh_editor::MeshData, UploadError> {
    if positions.is_empty() || indices.is_empty() {
        return Err(UploadError::EmptyMesh);
    }
    let mut mesh =
        ornis_mesh_editor::MeshData::from_positions(positions.to_vec(), indices.to_vec())
            .map(|mesh| mesh.with_computed_normals())
            .map_err(UploadError::InvalidMesh)?;
    apply_box_project_uvs(&mut mesh);
    Ok(mesh)
}

/// Convert an inline `MeshDesc::Custom` soup into GPU-ready vertices + indices.
///
/// Thin pure wrapper over [`custom_mesh_data`] + [`to_vertices`]: the
/// extraction's per-entity path calls this without touching `wgpu`; the
/// renderer then moves the arrays into buffers
/// (`renderer::upload_custom_mesh`).
///
/// Per-soup budget: time is `O(V + T)` over `V` vertices and `T`
/// triangles (one area-weighted normal pass, one box-projection UV pass,
/// one tangent pass — no kernel, no allocation beyond the outputs);
/// memory is 44 bytes per output [`Vertex`] plus 4 bytes per index, with
/// ~32 bytes per vertex of transient [`ornis_mesh_editor::MeshData`]
/// scratch (positions + normals + uvs) freed on return. Guidance: keep
/// single soups well under the `u32` index space (validation rejects
/// out-of-range indices with [`UploadError::is_index_out_of_range`]),
/// and share repeated geometry — [`SoupCache`] converts each distinct
/// soup once per frame, so `N` identical entities pay one conversion
/// plus `N-1` cache-hit clones.
///
/// # Errors
///
/// Same as [`custom_mesh_data`] (empty or invalid soup).
pub fn custom_vertices(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Result<(Vec<Vertex>, Vec<u32>), UploadError> {
    to_vertices(&custom_mesh_data(positions, indices)?)
}

/// Deterministic hash of an inline `MeshDesc::Custom` soup (`positions` +
/// `indices`; `f32` hashed by bits, so `0.0` and `-0.0` differ).
///
/// Cache key for [`custom_vertices_cached`] and [`SoupCache`]: identical
/// soups hash identically, so a frame with repeated geometry converts
/// each distinct soup once.
pub fn soup_hash(positions: &[[f32; 3]], indices: &[u32]) -> SoupHash {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    positions.len().hash(&mut hasher);
    for position in positions {
        for component in position {
            component.to_bits().hash(&mut hasher);
        }
    }
    indices.len().hash(&mut hasher);
    indices.hash(&mut hasher);
    SoupHash::from_raw(hasher.finish())
}

/// Per-frame Custom-soup conversion cache: dedups identical soups by
/// [`SoupHash`] and counts [`UploadCache`] hits/misses for
/// [`crate::extraction::ExtractionStats`].
///
/// Staging reuse: the map is reserved once per frame
/// ([`Self::with_capacity`]) and [`Self::clear_reuse`] drops entries
/// while keeping the allocation for the next frame, so a steady scene
/// pays no per-frame map growth. Converted vertex/index `Vec`s stay
/// owned by the map; lookups clone out of it (one allocation per
/// per-entity entry at the renderer, which keeps its own per-entity
/// GPU buffers).
#[derive(Debug, Default)]
pub struct SoupCache {
    /// Converted geometry by soup identity.
    map: HashMap<SoupHash, ConvertedSoup>,
    /// [`UploadCache::Hit`] count this frame.
    hits: u64,
    /// [`UploadCache::Miss`] count this frame (distinct soups converted).
    misses: u64,
}

impl SoupCache {
    /// Empty cache without reservation (one-off conversions, tests).
    pub fn new() -> Self {
        Self::default()
    }

    /// Empty cache reserving room for `custom_entities` distinct soups —
    /// the extraction passes its entity-lane length (capped), so a frame
    /// with `N` distinct soups grows the map at most once.
    pub fn with_capacity(custom_entities: usize) -> Self {
        Self {
            map: HashMap::with_capacity(custom_entities),
            hits: 0,
            misses: 0,
        }
    }

    /// Drop all entries but keep the map allocation (and reset the
    /// hit/miss counters) for the next frame's conversions.
    pub fn clear_reuse(&mut self) {
        self.map.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Convert `positions`/`indices` once per distinct soup, cloning the
    /// stored result on repeats (see [`custom_vertices_cached`]).
    ///
    /// # Errors
    ///
    /// Same as [`custom_vertices`] (empty or invalid soup); failures are
    /// never cached and never counted, so a retry re-attempts the
    /// conversion.
    pub fn get_or_convert(
        &mut self,
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Result<(ConvertedSoup, UploadCache), UploadError> {
        let (converted, outcome) =
            custom_vertices_cached_with_outcome(positions, indices, &mut self.map)?;
        match outcome {
            UploadCache::Hit => self.hits += 1,
            UploadCache::Miss => self.misses += 1,
        }
        Ok((converted, outcome))
    }

    /// Hits this frame (see [`UploadCache::Hit`]).
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Misses this frame — the number of distinct soups converted.
    pub const fn misses(&self) -> u64 {
        self.misses
    }

    /// Distinct soups currently stored.
    pub fn distinct_soups(&self) -> usize {
        self.map.len()
    }
}

/// Cached variant of [`custom_vertices`]: on a cache hit the stored
/// conversion is cloned without recomputing normals; on a miss the soup
/// is converted once and the result stored under [`soup_hash`].
///
/// The caller owns the map (per-frame in the extraction, see
/// [`SoupCache`]); the cache never outlives the converted data it clones
/// from. Hash collisions across distinct soups are not rechecked —
/// `u64` makes them negligible, and a collision would only reuse shading
/// for one soup, never panic.
///
/// # Errors
///
/// Same as [`custom_vertices`] (empty or invalid soup); failures are
/// never cached, so a retry re-attempts the conversion.
pub fn custom_vertices_cached(
    positions: &[[f32; 3]],
    indices: &[u32],
    cache: &mut HashMap<SoupHash, ConvertedSoup>,
) -> Result<ConvertedSoup, UploadError> {
    Ok(custom_vertices_cached_with_outcome(positions, indices, cache)?.0)
}

/// [`custom_vertices_cached`] plus the [`UploadCache`] outcome: [`Hit`](UploadCache::Hit)
/// when the stored conversion was reused, [`Miss`](UploadCache::Miss)
/// when the soup was converted and stored.
///
/// # Errors
///
/// Same as [`custom_vertices_cached`].
pub fn custom_vertices_cached_with_outcome(
    positions: &[[f32; 3]],
    indices: &[u32],
    cache: &mut HashMap<SoupHash, ConvertedSoup>,
) -> Result<(ConvertedSoup, UploadCache), UploadError> {
    let key = soup_hash(positions, indices);
    if let Some((vertices, soup_indices)) = cache.get(&key) {
        return Ok(((vertices.clone(), soup_indices.clone()), UploadCache::Hit));
    }
    let converted = custom_vertices(positions, indices)?;
    cache.insert(key, converted.clone());
    Ok((converted, UploadCache::Miss))
}

/// Rebuild per-vertex uvs with a box projection: each vertex is mapped
/// planarly from its dominant normal axis (`|nx|` → `(z, y)`,
/// `|ny|` → `(x, z)`, else `(x, y)`), normalized over the mesh bounds
/// into `[0, 1]`. Degenerate spans collapse to `0.5` so output stays
/// finite; a fully degenerate soup yields constant uvs.
fn apply_box_project_uvs(mesh: &mut ornis_mesh_editor::MeshData) {
    let mut min = [f32::INFINITY; VEC3_COMPONENTS];
    let mut max = [f32::NEG_INFINITY; VEC3_COMPONENTS];
    for position in &mesh.positions {
        for axis in 0..VEC3_COMPONENTS {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    let normalized = |value: f32, axis: usize| -> f32 {
        let span = max[axis] - min[axis];
        if span <= f32::EPSILON {
            HALF
        } else {
            ((value - min[axis]) / span).clamp(0.0, 1.0)
        }
    };
    for ((position, normal), uv) in mesh
        .positions
        .iter()
        .zip(&mesh.normals)
        .zip(mesh.uvs.iter_mut())
    {
        let (nx, ny, nz) = (normal[0].abs(), normal[1].abs(), normal[2].abs());
        *uv = if nx >= ny && nx >= nz {
            [normalized(position[2], 2), normalized(position[1], 1)]
        } else if ny >= nz {
            [normalized(position[0], 0), normalized(position[2], 2)]
        } else {
            [normalized(position[0], 0), normalized(position[1], 1)]
        };
    }
}

/// Any unit vector orthogonal to `normal` (tangent fallback, see [`to_vertices`]).
///
/// NOTE: this stays an orthogonal fallback, not MikkTSpace. A proper
/// tangent-space bake (MikkTSpace) is future work — it needs a new
/// dependency plus index-split vertices, far beyond this bridge — so
/// normal mapping on uploaded meshes remains approximate (correct
/// silhouette and lighting, slightly skewed tangent-space detail).
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

    #[test]
    fn custom_quad_converts_with_computed_normals() {
        // Planar quad in y=0 (winding gives +Y face normals).
        let positions = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let indices = [0, 1, 2, 0, 2, 3];
        let (vertices, out_indices) = custom_vertices(&positions, &indices).expect("quad valid");
        assert_eq!(vertices.len(), 4);
        assert_eq!(out_indices, indices);
        for vertex in &vertices {
            let n = glam::Vec3::from_array(vertex.normal);
            assert!(
                (n - glam::Vec3::Y).length() < 1e-6,
                "expected +Y, got {n:?}"
            );
            let t = glam::Vec3::from_array(vertex.tangent);
            assert!((t.length() - 1.0).abs() < 1e-6, "tangent unit: {t:?}");
            assert!(n.dot(t).abs() < 1e-6, "tangent orthogonal: {t:?}");
        }
    }

    #[test]
    fn custom_quad_gets_box_project_uvs_in_unit_range() {
        // Same planar quad as above: normals are +Y, so the box projection
        // maps (x, z) normalized over the [0, 1]² bounds.
        let positions = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let indices = [0, 1, 2, 0, 2, 3];
        let (vertices, _) = custom_vertices(&positions, &indices).expect("quad valid");
        let uvs: Vec<[f32; 2]> = vertices.iter().map(|vertex| vertex.uv).collect();
        assert_eq!(uvs, [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]);
        for uv in &uvs {
            assert!(
                (0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]),
                "uv in [0, 1]: {uv:?}"
            );
        }
    }

    #[test]
    fn fallback_tangent_is_unit_orthogonal_and_finite_on_degenerate() {
        for normal in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.3, -0.5, 0.8],
            [0.0, 0.0, 0.0],
        ] {
            let tangent = fallback_tangent(normal);
            let (n, t) = (
                glam::Vec3::from_array(normal),
                glam::Vec3::from_array(tangent),
            );
            assert!(t.is_finite(), "finite tangent for {normal:?}");
            assert!((t.length() - 1.0).abs() < 1e-6, "unit: {t:?}");
            if n.length_squared() > f32::EPSILON {
                assert!(n.normalize().dot(t).abs() < 1e-6, "orthogonal: {t:?}");
            }
        }
    }

    #[test]
    fn empty_or_invalid_custom_soup_is_rejected() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(matches!(
            custom_vertices(&[], &[0, 1, 2]),
            Err(UploadError::EmptyMesh)
        ));
        assert!(matches!(
            custom_vertices(&positions, &[]),
            Err(UploadError::EmptyMesh)
        ));
        assert!(matches!(
            custom_vertices(&positions, &[0, 1, 9]),
            Err(UploadError::InvalidMesh(_))
        ));
    }

    #[test]
    fn identical_quads_convert_once_and_failures_are_not_cached() {
        // Two identical quads → one cache entry (one conversion); a
        // different soup adds a second entry; failures never populate
        // the map, so a retry re-attempts the conversion.
        let quad = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let quad_indices = [0, 1, 2, 0, 2, 3];
        let mut cache = HashMap::new();
        let first = custom_vertices_cached(&quad, &quad_indices, &mut cache).expect("quad valid");
        let second = custom_vertices_cached(&quad, &quad_indices, &mut cache).expect("quad cached");
        assert_eq!(cache.len(), 1, "identical soups convert once");
        assert_eq!(first.0.len(), second.0.len());
        assert_eq!(first.1, second.1);
        for (a, b) in first.0.iter().zip(&second.0) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.normal, b.normal);
        }
        let other = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        custom_vertices_cached(&other, &[0, 1, 2], &mut cache).expect("triangle valid");
        assert_eq!(cache.len(), 2, "distinct soup converts separately");
        assert!(!cache.is_empty());
        let bad = [[0.0, 0.0, 0.0]];
        assert!(custom_vertices_cached(&bad, &[0, 0, 7], &mut cache).is_err());
        assert_eq!(cache.len(), 2, "failures are not cached");
    }

    #[test]
    fn soup_hash_is_a_stable_newtype_within_a_run() {
        // Same soup → equal `SoupHash`; any position/index change flips
        // it; the raw value round-trips through `from_raw`/`get`.
        let quad = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let indices = [0, 1, 2, 0, 2, 3];
        assert_eq!(soup_hash(&quad, &indices), soup_hash(&quad, &indices));
        let mut moved = quad;
        moved[0][0] = 0.5;
        assert_ne!(soup_hash(&quad, &indices), soup_hash(&moved, &indices));
        assert_ne!(
            soup_hash(&quad, &indices),
            soup_hash(&quad, &[0, 1, 2, 0, 2, 1])
        );
        let hash = soup_hash(&quad, &indices);
        assert_eq!(SoupHash::from_raw(hash.get()), hash);
    }

    #[test]
    fn cache_outcome_reports_hit_then_miss_counts() {
        // First sight of a soup is a `Miss`, the repeat is a `Hit`; the
        // `SoupCache` mirrors the outcomes in its hit/miss counters and
        // converts each distinct soup exactly once.
        let quad = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let quad_indices = [0, 1, 2, 0, 2, 3];
        let mut cache = SoupCache::with_capacity(4);
        let (_, first) = cache
            .get_or_convert(&quad, &quad_indices)
            .expect("quad valid");
        assert_eq!(first, UploadCache::Miss);
        let (again, second) = cache
            .get_or_convert(&quad, &quad_indices)
            .expect("quad cached");
        assert_eq!(second, UploadCache::Hit);
        assert_eq!(again.1, quad_indices);
        let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (_, third) = cache.get_or_convert(&tri, &[0, 1, 2]).expect("tri valid");
        assert_eq!(third, UploadCache::Miss);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 2);
        assert_eq!(cache.distinct_soups(), 2, "identical soups convert once");
        // Failures are neither cached nor counted.
        assert!(
            cache
                .get_or_convert(&[[0.0, 0.0, 0.0]], &[0, 0, 7])
                .is_err()
        );
        assert_eq!((cache.hits(), cache.misses()), (1, 2));
        // `clear_reuse` drops entries but keeps the reservation working.
        cache.clear_reuse();
        assert_eq!(cache.distinct_soups(), 0);
        assert_eq!((cache.hits(), cache.misses()), (0, 0));
    }

    #[test]
    fn upload_error_cases_are_first_class() {
        // Empty soup vs index-out-of-range classify through the
        // predicates without matching on the wrapped `MeshError`.
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let empty = custom_vertices(&[], &[0, 1, 2]).expect_err("empty positions");
        assert!(empty.is_empty_soup());
        assert!(!empty.is_index_out_of_range());
        let bad = custom_vertices(&positions, &[0, 1, 9]).expect_err("bad index");
        assert!(bad.is_index_out_of_range());
        assert!(!bad.is_empty_soup());
    }
}
