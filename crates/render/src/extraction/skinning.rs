//! Skinned custom-mesh entries and GPU joint-palette staging.

use super::TRIANGLE_VERTS;
use crate::mesh::SkinnedVertex;
use crate::mesh::Vertex;
use crate::renderer::InstanceData;
use crate::skinning::PaletteHandle;
use crate::skinning::SkinBindError;
use crate::skinning::SkinnedDraw;
use glam::Mat4;
use glam::Vec3;
use ornis_animation::JointPose;
use ornis_animation::Skeleton;
use ornis_animation::SkinnedMesh;
use ornis_animation::SkinningMode;
use ornis_animation::SkinningResources;
use ornis_animation::canonical_staged_weights;
use ornis_animation::skinning_matrices;
use ornis_core::SmartStore;

/// One valid `MeshDesc::Custom` entity: its CPU-side geometry plus the
/// instance pointing at the merged [`FrameUpload::materials`] table.
///
/// The renderer uploads `vertices`/`indices` per entity
/// (`renderer::upload_custom_mesh`) and draws the entry with `instance`.
#[derive(Clone, Debug)]
pub struct CustomMeshEntry {
    /// GPU-ready vertices: pre-skinned world-space rows for
    /// [`MeshPose::Skinned`] entries (classic soup path and CPU fallback),
    /// bind-pose rows for [`MeshPose::Bind`] entries (GPU blend — the
    /// skinned vertex stage reads these plus the influences in
    /// [`CustomMeshEntry::pose`]).
    pub vertices: Vec<Vertex>,
    /// Triangle index list (`u32`, triples, CCW from outside).
    pub indices: Vec<u32>,
    /// Model/normal matrices and the index into `FrameUpload::materials`.
    pub instance: InstanceData,
    /// How this entry is blended (phase D, `docs/animation-design.md` §2.3):
    /// [`SkinningMode::Cpu`] on the classic soup path below (bind-pose
    /// geometry transformed by `instance.model_matrix`, never pre-skinned)
    /// and on the CPU pre-skin fallback; [`SkinningMode::Gpu`] exactly when
    /// the entity carries the [`SkinnedMesh`] lane *and* its joint palette
    /// staged (see [`CustomMeshEntry::joint_palette`]).
    ///
    /// Freshness is the skin system's contract (`skel_skin_cpu` runs
    /// PostFrame before the frame upload); inconsistent lane arrays skip
    /// the entity with [`ExtractionStats::skipped_bad_skin`], never a stub
    /// (see [`extract_render_data_with_stats`]).
    pub skinning: SkinningMode,
    /// Staged GPU joint palette (phase D): final joint matrices
    /// (`model * inverse_bind`) as column-major arrays, one per joint —
    /// the upload bytes behind [`crate::skinning::joint_palette_bytes`].
    ///
    /// `Some` exactly when [`CustomMeshEntry::skinning`] is
    /// [`SkinningMode::Gpu`]; `None` on the classic path and on the CPU
    /// fallback (over-limit skeleton, stale pose, out-of-range joint
    /// indices — anything that would read out of bounds in the shader).
    pub joint_palette: Option<Vec<[[f32; 4]; 4]>>,
    /// Which buffer [`CustomMeshEntry::vertices`] holds: bind-pose rows
    /// (GPU blend) or pre-skinned world-space rows (CPU blend). Always in
    /// lockstep with [`CustomMeshEntry::skinning`] (`Gpu` ⟺ `Bind`,
    /// `Cpu` ⟺ `Skinned`); the draw decision reads both through
    /// [`CustomMeshEntry::draw`].
    pub pose: MeshPose,
}

impl CustomMeshEntry {
    /// Color-draw decision for this entry: [`SkinnedDraw::Gpu`] exactly
    /// when the entry blends on the GPU *and* a palette slot was uploaded
    /// for it, [`SkinnedDraw::Cpu`] otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`SkinBindError::MissingPalette`] when the entry claims the
    /// GPU path but `handle` is `None` (or no palette staged) — the caller
    /// falls back to [`SkinnedDraw::Cpu`], never a panic.
    pub fn draw(&self, handle: Option<PaletteHandle>) -> Result<SkinnedDraw, SkinBindError> {
        match self.skinning {
            SkinningMode::Cpu => Ok(SkinnedDraw::Cpu),
            SkinningMode::Gpu => match (self.joint_palette.is_some(), handle) {
                (true, Some(handle)) => Ok(SkinnedDraw::Gpu(handle)),
                _ => Err(SkinBindError::MissingPalette),
            },
        }
    }

    /// Depth-pre-pass decision for this entry: same rule as
    /// [`CustomMeshEntry::draw`], but the failure carries
    /// [`SkinBindError::ShadowWithoutPalette`] — skinned depth needs the
    /// palette, otherwise the shadow would come from bind-pose geometry.
    ///
    /// # Errors
    ///
    /// Returns [`SkinBindError::ShadowWithoutPalette`] when the entry
    /// claims the GPU path but `handle` is `None` (or no palette staged) —
    /// the caller falls back to [`SkinnedDraw::Cpu`], never a panic.
    pub fn shadow_draw(&self, handle: Option<PaletteHandle>) -> Result<SkinnedDraw, SkinBindError> {
        match self.skinning {
            SkinningMode::Cpu => Ok(SkinnedDraw::Cpu),
            SkinningMode::Gpu => match (self.joint_palette.is_some(), handle) {
                (true, Some(handle)) => Ok(SkinnedDraw::Gpu(handle)),
                _ => Err(SkinBindError::ShadowWithoutPalette),
            },
        }
    }

    /// Interleaved GPU rows for the skinned vertex stage: bind-pose
    /// vertices plus joint influences.
    ///
    /// `Some` exactly for [`MeshPose::Bind`] entries with agreeing lane
    /// lengths; `None` for CPU-blended entries (draw those with the
    /// classic upload) and for length defects (the caller keeps the entry
    /// on the CPU path, never a partial buffer).
    pub fn skinned_gpu_vertices(&self) -> Option<Vec<SkinnedVertex>> {
        let MeshPose::Bind(influences) = &self.pose else {
            return None;
        };
        if influences.joints.len() != self.vertices.len()
            || influences.weights.len() != self.vertices.len()
        {
            return None;
        }
        Some(
            self.vertices
                .iter()
                .zip(influences.joints.iter())
                .zip(influences.weights.iter())
                .map(|((vertex, &joints), &weights)| SkinnedVertex {
                    position: vertex.position,
                    normal: vertex.normal,
                    uv: vertex.uv,
                    tangent: vertex.tangent,
                    joints,
                    weights,
                })
                .collect(),
        )
    }
}
/// Which buffer a [`CustomMeshEntry`] holds: bind-pose rows for the vertex
/// stage to blend, or pre-skinned world-space rows for the classic draw.
///
/// `enum`, never `bool`: the GPU variant carries the per-vertex influences
/// the stage blends with, so a bind-pose buffer without influences is
/// unrepresentable.
#[derive(Clone, Debug)]
pub enum MeshPose {
    /// Bind-pose vertices: the skinned pipeline blends these with
    /// `influences` over the bound palette. Only for
    /// [`SkinningMode::Gpu`] entries.
    Bind(SkinInfluences),
    /// Pre-skinned world-space vertices: the classic pipeline draws these
    /// as-is. Classic soup path and CPU fallback.
    Skinned,
}

impl MeshPose {
    /// Whether this pose carries bind-pose rows for the GPU blend.
    pub const fn is_bind(&self) -> bool {
        matches!(self, Self::Bind(_))
    }
}

/// Per-vertex joint influences for the GPU blend of one
/// [`MeshPose::Bind`] entry.
///
/// Joints are widened `u32` lanes (`SkinnedMesh` stores `u16` — exact);
/// weights are canonicalized at staging (see
/// [`canonical_staged_weights`]) because the vertex stage multiplies raw
/// weights without normalizing. Lengths always agree with the entry's
/// vertex count.
#[derive(Clone, Debug)]
pub struct SkinInfluences {
    /// Influencing joints per vertex (top-4, `< joint count`).
    pub joints: Vec<[u32; 4]>,
    /// Canonicalized influence weights per vertex.
    pub weights: Vec<[f32; 4]>,
}
/// Validated vertex count of one [`SkinnedMesh`] lane set: every bind and
/// output array agrees on the length, and every index lands inside it.
///
/// [`None`] (bad skin) on empty binds, array length defects, or an empty /
/// malformed index list — the caller counts
/// [`ExtractionStats::skipped_bad_skin`]. Joint-index range against the
/// skeleton is the skin system's verdict (it owns that counter); the lane
/// buffers validated here are that system's output contract.
fn skin_vertex_count(mesh: &SkinnedMesh) -> Option<usize> {
    let count = mesh.joints.len();
    if count == 0
        || mesh.weights.len() != count
        || mesh.positions.len() != count
        || mesh.normals.len() != count
        || mesh.skinned_positions.len() != count
        || mesh.skinned_normals.len() != count
        || mesh.uvs.len() != count
    {
        return None;
    }
    if mesh.indices.is_empty()
        || !mesh.indices.len().is_multiple_of(TRIANGLE_VERTS)
        || mesh.indices.iter().any(|index| (*index as usize) >= count)
    {
        return None;
    }
    Some(count)
}

/// Builds the pre-skinned payload of one [`SkinnedMesh`] entity: world-space
/// output buffers as [`Vertex`] rows plus the passthrough bind indices.
///
/// [`None`] (bad skin) when [`skin_vertex_count`] rejects the lanes — the
/// caller counts [`ExtractionStats::skipped_bad_skin`].
pub(super) fn skinned_entry(mesh: &SkinnedMesh) -> Option<(Vec<Vertex>, Vec<u32>)> {
    skin_vertex_count(mesh)?;
    let vertices = mesh
        .skinned_positions
        .iter()
        .zip(mesh.skinned_normals.iter())
        .zip(mesh.uvs.iter())
        .map(|((&position, &normal), &uv)| Vertex {
            position,
            normal,
            uv,
            tangent: skinned_tangent(normal),
        })
        .collect();
    Some((vertices, mesh.indices.to_vec()))
}

/// Builds the bind-pose payload of one [`SkinnedMesh`] entity for the GPU
/// blend: bind-pose [`Vertex`] rows, the passthrough bind indices, and the
/// staged influences (widened joints, canonicalized weights).
///
/// [`None`] (bad skin) when [`skin_vertex_count`] rejects the lanes — the
/// caller counts [`ExtractionStats::skipped_bad_skin`]. Joint-index range
/// against the skeleton is gated upstream by [`gpu_joint_palette`] (only
/// staged palettes reach the GPU path); the widened lanes here carry the
/// values verbatim.
pub(super) fn bind_pose_entry(
    mesh: &SkinnedMesh,
) -> Option<(Vec<Vertex>, Vec<u32>, SkinInfluences)> {
    skin_vertex_count(mesh)?;
    let vertices = mesh
        .positions
        .iter()
        .zip(mesh.normals.iter())
        .zip(mesh.uvs.iter())
        .map(|((&position, &normal), &uv)| Vertex {
            position,
            normal,
            uv,
            tangent: skinned_tangent(normal),
        })
        .collect();
    let joints = mesh
        .joints
        .iter()
        .map(|lane| {
            [
                lane[0] as u32,
                lane[1] as u32,
                lane[2] as u32,
                lane[3] as u32,
            ]
        })
        .collect();
    let weights = mesh
        .weights
        .iter()
        .map(|&lane| canonical_staged_weights(lane))
        .collect();
    Some((
        vertices,
        mesh.indices.to_vec(),
        SkinInfluences { joints, weights },
    ))
}

/// Stages the GPU joint palette of one [`SkinnedMesh`] entity: final joint
/// matrices (`model * inverse_bind`) as column-major arrays.
///
/// [`None`] (CPU fallback — same pre-skinned vertices, no palette) when the
/// skeleton lanes are absent, the topology is invalid, the pose is stale,
/// or a joint index reaches past the palette (an out-of-bounds read in the
/// shader must never stage). Callers keep the entity on the CPU path; only
/// lane-buffer defects skip it (see [`skinned_entry`]).
pub(super) fn gpu_joint_palette(
    store: &SmartStore,
    mesh: &SkinnedMesh,
) -> Option<Vec<[[f32; 4]; 4]>> {
    let skeletons = store.read_lane::<Skeleton>()?;
    let poses = store.read_lane::<JointPose>()?;
    let skeleton = skeletons.get(mesh.skeleton)?;
    let count = skeleton.validate().ok()?;
    let pose = poses.get(mesh.skeleton)?;
    if pose.matrices.len() != count {
        return None;
    }
    let staged = SkinningResources::build(
        &skinning_matrices(&pose.matrices, &skeleton.inverse_bind),
        SkinningMode::Gpu,
    )
    .ok()?;
    if mesh
        .joints
        .iter()
        .flatten()
        .any(|index| (*index as usize) >= count)
    {
        return None;
    }
    Some(
        staged
            .palette_matrices()
            .iter()
            .map(Mat4::to_cols_array_2d)
            .collect(),
    )
}

/// Any unit vector orthogonal to a skinned normal (tangent fallback).
///
/// Same contract as `mesh_upload::fallback_tangent` (duplicated here: that
/// module is outside this track's file bounds, so the formula — not the
/// function — is shared): degenerate normals fall back to `+X` so the
/// vertex stays finite.
fn skinned_tangent(normal: [f32; 3]) -> [f32; 3] {
    let direction = Vec3::from_array(normal);
    if direction.length_squared() <= f32::EPSILON {
        return [1.0, 0.0, 0.0];
    }
    direction.normalize().any_orthonormal_vector().to_array()
}
#[cfg(test)]
mod tests {
    use super::super::extract_render_data;
    use super::super::extract_render_data_with_stats;
    use super::super::test_util::test_material;
    use super::*;
    use ornis_animation::JointPose;
    use ornis_animation::Skeleton;
    use ornis_animation::SkinnedMesh;
    use ornis_assets::scene::MeshDesc;
    use ornis_assets::scene::TransformDesc;
    use ornis_core::Engine;
    use ornis_core::Entity;

    /// Test helper: a single-joint skeleton root plus one skinned triangle
    /// entity (identity pose → CPU vertices equal the bind positions).
    /// Returns the mesh entity (for lane corruption) — the root is the
    /// mesh's `skeleton` link.
    fn push_skinned_triangle(engine: &mut Engine) -> Entity {
        {
            let store = engine.world_mut().store_mut().expect("store");
            store.register::<Skeleton>();
            store.register::<JointPose>();
            store.register::<SkinnedMesh>();
        }
        let root = {
            let store = engine.world_mut().store_mut().expect("store");
            let root = store.create_entity();
            store.insert(
                root,
                Skeleton::new(vec![None], vec![Mat4::IDENTITY], vec!["root".to_string()]),
            );
            store.insert(root, JointPose::identity(1));
            root
        };
        let positions = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        {
            let store = engine.world_mut().store_mut().expect("store");
            let mesh = store.create_entity();
            store.insert(mesh, TransformDesc::IDENTITY);
            store.insert(
                mesh,
                MeshDesc::Custom {
                    positions: positions.clone(),
                    indices: vec![0, 1, 2],
                },
            );
            store.insert(mesh, test_material());
            store.insert(
                mesh,
                SkinnedMesh::new(
                    root,
                    vec![[0, 0, 0, 0]; 3],
                    vec![[1.0, 0.0, 0.0, 0.0]; 3],
                    positions,
                    vec![[0.0, 0.0, 1.0]; 3],
                    vec![[0.0, 0.0]; 3],
                    vec![0, 1, 2],
                ),
            );
            mesh
        }
    }

    #[test]
    fn skinned_entry_stages_gpu_palette_with_bind_pose_vertices() {
        // Valid skin: GPU mode with the staged palette, while `vertices`
        // carry the bind pose (identity skin → bind positions) for the
        // skinned vertex stage; the influences ride `pose`. Parity: the
        // palette-blend of the bind data matches the CPU skin output
        // within the допуск.
        use ornis_animation::{CPU_GPU_TOLERANCE, blend_vertex_reference};
        let mut engine = Engine::new();
        push_skinned_triangle(&mut engine);

        let (upload, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(upload.custom_meshes.len(), 1);
        assert_eq!(stats.skipped_bad_skin, 0);
        let entry = &upload.custom_meshes[0];
        assert_eq!(entry.skinning, SkinningMode::Gpu);
        assert!(entry.pose.is_bind());
        assert_eq!(entry.instance.model_matrix, Mat4::IDENTITY);
        let palette = entry.joint_palette.as_ref().expect("palette staged");
        assert_eq!(palette.len(), 1);
        assert_eq!(Mat4::from_cols_array_2d(&palette[0]), Mat4::IDENTITY);
        assert_eq!(entry.vertices[0].position, [1.0, 0.0, 0.0]);
        // Bind pose rides the entry: joints widened, weights canonical.
        let MeshPose::Bind(influences) = &entry.pose else {
            panic!("gpu entry must carry bind influences");
        };
        assert_eq!(influences.joints.len(), 3);
        assert_eq!(influences.joints[0], [0, 0, 0, 0]);
        assert_eq!(influences.weights[0], [1.0, 0.0, 0.0, 0.0]);
        // Parity: palette-blend(bind) ≈ CPU-skinned positions.
        let bind = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let matrices = [Mat4::IDENTITY];
        for (index, vertex) in entry.vertices.iter().enumerate() {
            let (position, _) = blend_vertex_reference(
                &matrices,
                [0, 0, 0, 0],
                [1.0, 0.0, 0.0, 0.0],
                bind[index],
                [0.0, 0.0, 1.0],
            );
            let drift = (Vec3::from_array(position) - Vec3::from_array(vertex.position)).length();
            assert!(drift < CPU_GPU_TOLERANCE, "vertex {index} drifts {drift}");
        }
        // The interleaved GPU rows zip bind vertices with influences.
        let gpu_rows = entry.skinned_gpu_vertices().expect("bind rows");
        assert_eq!(gpu_rows.len(), 3);
        assert_eq!(gpu_rows[0].position, [1.0, 0.0, 0.0]);
        assert_eq!(gpu_rows[0].joints, [0, 0, 0, 0]);
        assert_eq!(gpu_rows[0].weights, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn missing_skeleton_falls_back_to_cpu_without_palette() {
        // No skeleton/pose lanes: the entity still extracts (valid CPU
        // buffers) but in CPU mode with no palette — never a GPU claim
        // the shader could read out of bounds with.
        let mut engine = Engine::new();
        push_skinned_triangle(&mut engine);
        // Destroy every skeleton root (same honesty as `RenderWorld` scene
        // replacement): the mesh lane outlives its skeleton.
        let roots: Vec<Entity> = {
            let store = engine.world().store().expect("store");
            store
                .read_lane::<Skeleton>()
                .map(|lane| lane.entities.clone())
                .unwrap_or_default()
        };
        for root in roots {
            let store = engine.world_mut().store_mut().expect("store");
            if store.is_alive(root) {
                store.destroy_entity(root);
            }
        }

        let (upload, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(upload.custom_meshes.len(), 1);
        assert_eq!(stats.skipped_bad_skin, 0);
        let entry = &upload.custom_meshes[0];
        assert_eq!(entry.skinning, SkinningMode::Cpu);
        assert!(entry.joint_palette.is_none());
        assert!(matches!(entry.pose, MeshPose::Skinned));
        assert!(entry.skinned_gpu_vertices().is_none());
        assert_eq!(entry.vertices[0].position, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn over_limit_skeleton_falls_back_to_cpu_without_palette() {
        // 129 joints: the palette cannot stage (typed overflow), so the
        // entry extracts in CPU mode with no palette — never truncated.
        use ornis_animation::JointLimit;
        let over = JointLimit::GPU.index() + 1;
        let mut engine = Engine::new();
        {
            let store = engine.world_mut().store_mut().expect("store");
            store.register::<Skeleton>();
            store.register::<JointPose>();
            store.register::<SkinnedMesh>();
            let root = store.create_entity();
            store.insert(
                root,
                Skeleton::new(
                    vec![None; over],
                    vec![Mat4::IDENTITY; over],
                    vec!["joint".to_string(); over],
                ),
            );
            store.insert(root, JointPose::identity(over));
            let mesh = store.create_entity();
            store.insert(mesh, TransformDesc::IDENTITY);
            store.insert(
                mesh,
                MeshDesc::Custom {
                    positions: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    indices: vec![0, 1, 2],
                },
            );
            store.insert(mesh, test_material());
            store.insert(
                mesh,
                SkinnedMesh::new(
                    root,
                    vec![[0, 0, 0, 0]; 3],
                    vec![[1.0, 0.0, 0.0, 0.0]; 3],
                    vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    vec![[0.0, 0.0, 1.0]; 3],
                    vec![[0.0, 0.0]; 3],
                    vec![0, 1, 2],
                ),
            );
        }

        let (upload, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(upload.custom_meshes.len(), 1);
        assert_eq!(stats.skipped_bad_skin, 0);
        let entry = &upload.custom_meshes[0];
        assert_eq!(entry.skinning, SkinningMode::Cpu);
        assert!(entry.joint_palette.is_none());
        assert!(matches!(entry.pose, MeshPose::Skinned));
        assert!(entry.skinned_gpu_vertices().is_none());
    }

    /// Test helper: a two-joint skeleton root with a rigid rotated pose
    /// plus one skinned triangle entity. The palette is a 90° Z-rotation
    /// with a translation on joint 1, so the GPU blend and the CPU skin
    /// must agree within the parity допуск (rigid joints are exact up to
    /// FMA ordering).
    fn push_rotated_two_joint_triangle(engine: &mut Engine) {
        {
            let store = engine.world_mut().store_mut().expect("store");
            store.register::<Skeleton>();
            store.register::<JointPose>();
            store.register::<SkinnedMesh>();
        }
        let root = {
            let store = engine.world_mut().store_mut().expect("store");
            let root = store.create_entity();
            store.insert(
                root,
                Skeleton::new(
                    vec![None, Some(ornis_animation::JointId::from_raw(0))],
                    vec![Mat4::IDENTITY; 2],
                    vec!["root".to_string(), "child".to_string()],
                ),
            );
            let rotated = Mat4::from_rotation_translation(
                glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                Vec3::X,
            );
            store.insert(
                root,
                JointPose {
                    matrices: vec![Mat4::IDENTITY, rotated],
                },
            );
            root
        };
        {
            let store = engine.world_mut().store_mut().expect("store");
            let mesh = store.create_entity();
            store.insert(mesh, TransformDesc::IDENTITY);
            store.insert(
                mesh,
                MeshDesc::Custom {
                    positions: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    indices: vec![0, 1, 2],
                },
            );
            store.insert(mesh, test_material());
            store.insert(
                mesh,
                SkinnedMesh::new(
                    root,
                    vec![[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
                    vec![
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.0, 0.0],
                    ],
                    vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    vec![[0.0, 0.0, 1.0]; 3],
                    vec![[0.0, 0.0]; 3],
                    vec![0, 1, 2],
                ),
            );
        }
    }

    #[test]
    fn gpu_bind_blend_matches_cpu_skin_within_tolerance() {
        // Cpu-draw vs Gpu-draw parity at the entry level: the staged bind
        // data blended through the GPU reference formula lands within
        // `CPU_GPU_TOLERANCE` of the CPU skin output for a rigid palette
        // (both sides use the per-joint inverse-transpose; positions differ
        // only by FMA ordering).
        use ornis_animation::{CPU_GPU_TOLERANCE, blend_vertex_reference, skin_vertices};
        let mut engine = Engine::new();
        push_rotated_two_joint_triangle(&mut engine);

        let (upload, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(upload.custom_meshes.len(), 1);
        assert_eq!(stats.skipped_bad_skin, 0);
        let entry = &upload.custom_meshes[0];
        assert_eq!(entry.skinning, SkinningMode::Gpu);
        let palette = entry
            .joint_palette
            .as_ref()
            .expect("palette staged")
            .iter()
            .map(Mat4::from_cols_array_2d)
            .collect::<Vec<_>>();
        assert_eq!(palette.len(), 2);
        let MeshPose::Bind(influences) = &entry.pose else {
            panic!("gpu entry must carry bind influences");
        };
        // CPU draw: the skin system's own blend over the same palette.
        let joints_u16 = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]];
        let weights = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ];
        let bind_positions = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let bind_normals = [[0.0, 0.0, 1.0]; 3];
        let (cpu_positions, cpu_normals) = skin_vertices(
            &palette,
            &joints_u16,
            &weights,
            &bind_positions,
            &bind_normals,
        );
        // GPU draw: the reference blend over the staged entry data.
        for index in 0..3 {
            let (gpu_position, gpu_normal) = blend_vertex_reference(
                &palette,
                influences.joints[index].map(|joint| joint as u16),
                influences.weights[index],
                entry.vertices[index].position,
                entry.vertices[index].normal,
            );
            let position_drift =
                (Vec3::from_array(gpu_position) - Vec3::from_array(cpu_positions[index])).length();
            let normal_drift =
                (Vec3::from_array(gpu_normal) - Vec3::from_array(cpu_normals[index])).length();
            assert!(
                position_drift < CPU_GPU_TOLERANCE,
                "vertex {index} position drifts {position_drift}"
            );
            assert!(
                normal_drift < CPU_GPU_TOLERANCE,
                "vertex {index} normal drifts {normal_drift}"
            );
        }
        // Spot check: joint-1-only vertex rides the rotation+translation.
        assert!(
            (Vec3::from_array(cpu_positions[0]) - Vec3::new(1.0, 1.0, 0.0)).length()
                < CPU_GPU_TOLERANCE
        );
    }

    #[test]
    fn draw_resolves_gpu_with_handle_and_falls_back_without_palette() {
        // Gpu entries resolve through the uploaded handle; a missing
        // palette (or slot) falls back to Cpu without panicking — the
        // color path reports `MissingPalette`, the shadow path
        // `ShadowWithoutPalette` (bind-pose shadows are never drawn).
        let mut engine = Engine::new();
        push_skinned_triangle(&mut engine);
        let upload = extract_render_data(engine.world().store().expect("store"));
        let entry = &upload.custom_meshes[0];
        assert_eq!(entry.skinning, SkinningMode::Gpu);
        let handle = PaletteHandle::from_raw(0);
        assert_eq!(entry.draw(Some(handle)), Ok(SkinnedDraw::Gpu(handle)));
        assert_eq!(
            entry.shadow_draw(Some(handle)),
            Ok(SkinnedDraw::Gpu(handle))
        );
        assert_eq!(entry.draw(None), Err(SkinBindError::MissingPalette));
        assert_eq!(
            entry.shadow_draw(None),
            Err(SkinBindError::ShadowWithoutPalette)
        );
        // Both failures degrade to the CPU draw, never a panic.
        assert_eq!(
            entry.draw(None).unwrap_or(SkinnedDraw::Cpu),
            SkinnedDraw::Cpu
        );
        assert_eq!(
            entry.shadow_draw(None).unwrap_or(SkinnedDraw::Cpu),
            SkinnedDraw::Cpu
        );
        // Cpu entries ignore handles entirely.
        let mut cpu_entry = entry.clone();
        cpu_entry.skinning = SkinningMode::Cpu;
        cpu_entry.joint_palette = None;
        cpu_entry.pose = MeshPose::Skinned;
        assert_eq!(cpu_entry.draw(Some(handle)), Ok(SkinnedDraw::Cpu));
        assert_eq!(cpu_entry.shadow_draw(Some(handle)), Ok(SkinnedDraw::Cpu));
        assert_eq!(cpu_entry.draw(None), Ok(SkinnedDraw::Cpu));
    }

    #[test]
    fn skinned_gpu_vertices_reject_cpu_pose_and_length_defects() {
        // Cpu entries have no GPU rows; a Bind pose whose lanes disagree
        // with the vertex count yields None instead of a partial buffer.
        let mut engine = Engine::new();
        push_skinned_triangle(&mut engine);
        let upload = extract_render_data(engine.world().store().expect("store"));
        let mut entry = upload.custom_meshes[0].clone();
        entry.pose = MeshPose::Skinned;
        assert!(entry.skinned_gpu_vertices().is_none());
        let mut broken = upload.custom_meshes[0].clone();
        let MeshPose::Bind(influences) = &mut broken.pose else {
            panic!("gpu entry must carry bind influences");
        };
        influences.joints.pop();
        assert!(broken.skinned_gpu_vertices().is_none());
    }
}
