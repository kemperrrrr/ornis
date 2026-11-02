//! Guest ABI packing: render skin entries and synthetic scenes into the
//! scalar storage lanes the `skin` kernel reads.

use glam::Mat4;
use glam::Quat;
use glam::Vec3;

#[cfg(test)]
use crate::extraction::CustomMeshEntry;
#[cfg(test)]
use crate::extraction::MeshPose;
use crate::skinning::SkinJoint;

/// `f32` lanes of one packed palette joint (matrix + inverse-transpose).
const JOINT_LANES: usize = 32;
/// `f32` lanes of one packed vertex (position, then normal).
const VERTEX_LANES: usize = 6;
/// Influence lanes per vertex.
const MAX_INFLUENCES: usize = 4;
/// Grid width of the synthetic bind-pose sheet.
const SYNTHETIC_GRID: usize = 64;
/// Per-joint rotation step (radians) of the synthetic palette.
const JOINT_ANGLE_STEP: f32 = 0.37;
/// Synthetic weights: a fixed non-trivial blend over four influences.
const SYNTHETIC_WEIGHTS: [f32; MAX_INFLUENCES] = [0.5, 0.25, 0.125, 0.125];
/// Axis cycle of the synthetic joint rotations (`x`, `y`, `z`, …).
const AXIS_CYCLE: usize = 3;
/// Non-uniform scale step of the synthetic joints (x grows per cycle step).
const JOINT_SCALE_STEP: f32 = 0.1;
/// Z scale shared by every synthetic joint.
const JOINT_SCALE_Z: f32 = 0.9;
/// Per-joint translation step of the synthetic palette.
const JOINT_TRANSLATION_STEP: f32 = 0.25;
/// Per-joint depth step of the synthetic palette.
const JOINT_DEPTH_STEP: f32 = -0.125;
/// Sheet spacing of the synthetic bind positions.
const SHEET_SPACING: f32 = 0.1;
/// Depth ripple period of the synthetic bind positions.
const SHEET_RIPPLE_PERIOD: usize = 7;
/// Depth ripple step of the synthetic bind positions.
const SHEET_RIPPLE_STEP: f32 = 0.05;
/// Third component of the unnormalized synthetic bind normals.
const NORMAL_Z: f32 = 3.0;

/// One mesh packed into the guest's scalar storage lanes.
///
/// The palette lanes are byte-identical to the production
/// [`SkinJoint`] staging (column-major skin matrix, then its
/// inverse-transpose), so a guest dispatch and the DSL vertex path blend
/// the same bytes.
pub(crate) struct SkinInputs {
    /// Packed palette lanes, [`JOINT_LANES`] per joint.
    pub(crate) palette: Vec<f32>,
    /// Packed bind vertices, [`VERTEX_LANES`] per vertex.
    pub(crate) bind: Vec<f32>,
    /// Joint indices, [`MAX_INFLUENCES`] per vertex.
    pub(crate) joints: Vec<u32>,
    /// Influence weights, [`MAX_INFLUENCES`] per vertex.
    pub(crate) weights: Vec<f32>,
    /// Number of packed vertices.
    pub(crate) vertex_count: u32,
    /// Number of packed joints.
    pub(crate) joint_count: u32,
}

impl SkinInputs {
    /// Packs one mesh from plain bind data.
    ///
    /// [`None`] on empty inputs, disagreeing lane lengths, or counts past
    /// `u32` — the caller keeps the mesh on the CPU path, never stages a
    /// partial guest input.
    pub(crate) fn pack(
        palette: &[Mat4],
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        joints: &[[u32; MAX_INFLUENCES]],
        weights: &[[f32; MAX_INFLUENCES]],
    ) -> Option<Self> {
        let vertex_count = positions.len();
        if vertex_count == 0
            || palette.is_empty()
            || normals.len() != vertex_count
            || joints.len() != vertex_count
            || weights.len() != vertex_count
        {
            return None;
        }
        let mut packed_palette = Vec::with_capacity(palette.len() * JOINT_LANES);
        for matrix in palette {
            let joint = SkinJoint::from_matrix(matrix.to_cols_array_2d());
            packed_palette.extend(joint.matrix.into_iter().flatten());
            packed_palette.extend(joint.normal_matrix.into_iter().flatten());
        }
        let mut bind = Vec::with_capacity(vertex_count * VERTEX_LANES);
        for (position, normal) in positions.iter().zip(normals) {
            bind.extend_from_slice(position);
            bind.extend_from_slice(normal);
        }
        Some(Self {
            palette: packed_palette,
            bind,
            joints: joints.iter().flatten().copied().collect(),
            weights: weights.iter().flatten().copied().collect(),
            vertex_count: u32::try_from(vertex_count).ok()?,
            joint_count: u32::try_from(palette.len()).ok()?,
        })
    }

    /// Packs one extracted GPU skin entry (bind pose plus staged palette).
    ///
    /// [`None`] for CPU entries and for lane defects — same verdict as
    /// [`CustomMeshEntry::skinned_gpu_vertices`].
    // qual:api — harness packing surface for production skinning entries.
    #[cfg(test)]
    pub(crate) fn from_entry(entry: &CustomMeshEntry) -> Option<Self> {
        let MeshPose::Bind(influences) = &entry.pose else {
            return None;
        };
        let palette: Vec<Mat4> = entry
            .joint_palette
            .as_ref()?
            .iter()
            .map(Mat4::from_cols_array_2d)
            .collect();
        let positions: Vec<_> = entry.vertices.iter().map(|v| v.position).collect();
        let normals: Vec<_> = entry.vertices.iter().map(|v| v.normal).collect();
        Self::pack(
            &palette,
            &positions,
            &normals,
            &influences.joints,
            &influences.weights,
        )
    }
}

/// Deterministic synthetic skinned mesh shared by the parity test and the
/// A/B bench: a bind-pose sheet over a palette of rotating, translating,
/// non-uniformly scaling joints (so normals exercise the inverse-transpose).
pub(crate) struct SyntheticSkin {
    /// Final joint matrices, one per joint.
    pub(crate) palette: Vec<Mat4>,
    /// Bind-pose positions.
    pub(crate) positions: Vec<[f32; 3]>,
    /// Bind-pose normals (unit length).
    pub(crate) normals: Vec<[f32; 3]>,
    /// Joint indices per vertex (CPU-path `u16` lanes).
    pub(crate) joints: Vec<[u16; MAX_INFLUENCES]>,
    /// Influence weights per vertex.
    pub(crate) weights: Vec<[f32; MAX_INFLUENCES]>,
}

impl SyntheticSkin {
    /// The CPU reference skin of this scene (`ornis_animation::skin_vertices`).
    pub(crate) fn cpu_skin(&self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        ornis_animation::skin_vertices(
            &self.palette,
            &self.joints,
            &self.weights,
            &self.positions,
            &self.normals,
        )
    }

    /// This scene packed for the guest kernel; `None` only on `u32` overflow.
    pub(crate) fn guest_inputs(&self) -> Option<SkinInputs> {
        let joints: Vec<[u32; MAX_INFLUENCES]> =
            self.joints.iter().map(|lane| lane.map(u32::from)).collect();
        SkinInputs::pack(
            &self.palette,
            &self.positions,
            &self.normals,
            &joints,
            &self.weights,
        )
    }
}

/// Builds one [`SyntheticSkin`] with `vertex_count` vertices over
/// `joint_count` joints.
pub(crate) fn synthetic_skin(vertex_count: usize, joint_count: usize) -> SyntheticSkin {
    let joints_per_vertex = joint_count.max(1);
    SyntheticSkin {
        palette: (0..joint_count)
            .map(|joint| {
                let angle = joint as f32 * JOINT_ANGLE_STEP;
                let rotation = match joint % AXIS_CYCLE {
                    0 => Quat::from_rotation_x(angle),
                    1 => Quat::from_rotation_y(angle),
                    _ => Quat::from_rotation_z(angle),
                };
                let scale = Vec3::new(
                    1.0 + JOINT_SCALE_STEP * (joint % AXIS_CYCLE) as f32,
                    1.0,
                    JOINT_SCALE_Z,
                );
                let translation = Vec3::new(
                    joint as f32 * JOINT_TRANSLATION_STEP,
                    0.0,
                    joint as f32 * JOINT_DEPTH_STEP,
                );
                Mat4::from_scale_rotation_translation(scale, rotation, translation)
            })
            .collect(),
        positions: (0..vertex_count)
            .map(|index| {
                [
                    sheet_column(index) * SHEET_SPACING,
                    sheet_row(index) * SHEET_SPACING,
                    (index % SHEET_RIPPLE_PERIOD) as f32 * SHEET_RIPPLE_STEP,
                ]
            })
            .collect(),
        normals: (0..vertex_count)
            .map(|index| {
                Vec3::new(sheet_column(index) + 1.0, sheet_row(index) + 2.0, NORMAL_Z)
                    .normalize()
                    .to_array()
            })
            .collect(),
        joints: (0..vertex_count)
            .map(|index| {
                let first = index % joints_per_vertex;
                std::array::from_fn(|slot| {
                    u16::try_from((first + slot) % joints_per_vertex).unwrap_or_default()
                })
            })
            .collect(),
        weights: vec![SYNTHETIC_WEIGHTS; vertex_count],
    }
}

/// Column of synthetic vertex `index` on the bind-pose sheet.
fn sheet_column(index: usize) -> f32 {
    (index % SYNTHETIC_GRID) as f32
}

/// Row of synthetic vertex `index` on the bind-pose sheet.
fn sheet_row(index: usize) -> f32 {
    (index / SYNTHETIC_GRID) as f32
}
