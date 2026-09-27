//! Phase D GPU-skinning types: how a joint palette reaches the vertex shader.
//!
//! The CPU path ([`crate::skin_vertices`]) blends bind vertices on the host
//! and publishes world-space buffers. The GPU path stages the same final
//! joint matrices (`model * inverse_bind`) as a storage palette and blends
//! in the vertex stage (see `ornis-render` `skinning`). This module is the
//! type-level contract between the two: [`SkinningMode`] (which path an
//! entry takes), [`JointCount`]/[`JointLimit`] (the 128-joint cap as types,
//! not a magic number), [`SkinningResources`] (the validated palette) and
//! [`SkinError`] (the fallible build verdict).
//!
//! CPU-vs-GPU parity is approximate, never bit-identical: both sides use
//! `f32` linear blend skinning over the same canonicalized weights, but the
//! GPU normal path uses the joint linear part while the CPU path uses the
//! inverse-transpose 3x3 (exact match for rigid/uniform-scale joints only),
//! and driver FMA fusion may move the last ulp. Callers assert
//! [`CPU_GPU_TOLERANCE`], not equality.

use glam::{Mat4, Vec3};

/// How one skinned entry is blended: on the host or in the vertex shader.
///
/// `Cpu` is the phase B path (pre-skinned world-space buffers, `IDENTITY`
/// instance matrices) and the fallback whenever the GPU palette cannot be
/// staged (over-limit skeleton, bad skin). `Gpu` means the joint palette is
/// staged alongside the bind data and the vertex stage blends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum SkinningMode {
    /// Host-side blend (classic transform path and CPU pre-skin fallback).
    #[default]
    Cpu,
    /// Vertex-shader blend over the staged joint palette.
    Gpu,
}

impl SkinningMode {
    /// Whether this mode blends in the vertex shader.
    pub const fn is_gpu(self) -> bool {
        matches!(self, Self::Gpu)
    }
}

/// Validated joint count of one skeleton: `0 < count <= [`JointLimit::GPU`]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JointCount(u32);

impl JointCount {
    /// Wraps a raw joint count without checking (validation lives in
    /// [`SkinningResources::build`]).
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw joint count.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Joint count as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for JointCount {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<JointCount> for u32 {
    fn from(h: JointCount) -> Self {
        h.0
    }
}

impl From<JointCount> for usize {
    fn from(h: JointCount) -> Self {
        h.0 as usize
    }
}

/// Joint-capacity limit of one palette: the uniform/storage bound of the
/// GPU path (design §2.1). Skeletons beyond the limit are rejected, never
/// silently truncated — the entry falls back to [`SkinningMode::Cpu`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JointLimit(u32);

impl JointLimit {
    /// GPU palette capacity: 128 joints (64 bytes each, 8 KiB per palette).
    pub const GPU: Self = Self(128);

    /// Raw limit.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Limit as `usize` for comparisons against slice lengths.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<JointLimit> for u32 {
    fn from(h: JointLimit) -> Self {
        h.0
    }
}

impl From<JointLimit> for usize {
    fn from(h: JointLimit) -> Self {
        h.0 as usize
    }
}

/// Why a joint palette cannot be staged for the GPU path.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum SkinError {
    /// No joint matrices: nothing to stage.
    #[error("bad skin: empty joint palette")]
    EmptyPalette,
    /// More joints than [`JointLimit::GPU`]: the entry must take the
    /// [`SkinningMode::Cpu`] fallback, never a truncated palette.
    #[error("palette overflow: {count} joints exceed the limit of {limit}")]
    PaletteOverflow {
        /// Rejected joint count.
        count: u32,
        /// Limit that rejected it.
        limit: u32,
    },
    /// The skin claim is inconsistent (stale pose length, bad indices):
    /// the entity keeps its previous buffers and counts as bad skin.
    #[error("bad skin: {reason}")]
    BadSkin {
        /// What failed validation (static wording, no payload).
        reason: &'static str,
    },
}

/// CPU-vs-GPU parity tolerance in engine units: the documented допуск for
/// comparing [`crate::skin_vertices`] output against the vertex-shader
/// blend (see the module docs for why bit-identity is not promised).
pub const CPU_GPU_TOLERANCE: f32 = 1e-4;

/// Validated joint palette plus the mode it was built for.
///
/// Built by [`SkinningResources::build`] from final joint matrices
/// (`model * inverse_bind`, see [`crate::skinning_matrices`]): empty and
/// over-limit inputs fail with [`SkinError`] so the caller can fall back
/// to [`SkinningMode::Cpu`] instead of staging a truncated palette.
#[derive(Debug, Clone, PartialEq)]
pub struct SkinningResources {
    /// Which path this palette was built for (always [`SkinningMode::Gpu`]
    /// from [`SkinningResources::build`]; CPU entries carry no palette).
    mode: SkinningMode,
    /// Validated joint count (`matrices.len()`).
    count: JointCount,
    /// Final joint matrices, one per joint.
    palette: Vec<Mat4>,
}

impl SkinningResources {
    /// Validates `matrices` against [`JointLimit::GPU`] and stages them
    /// for `requested`.
    ///
    /// # Errors
    ///
    /// Returns [`SkinError::EmptyPalette`] on zero matrices and
    /// [`SkinError::PaletteOverflow`] when the count exceeds
    /// [`JointLimit::GPU`] — the caller falls back to
    /// [`SkinningMode::Cpu`], never a truncated palette.
    ///
    /// # Examples
    ///
    /// ```
    /// # use glam::Mat4;
    /// # use ornis_animation::{SkinError, SkinningMode, SkinningResources};
    /// let palette =
    ///     SkinningResources::build(&[Mat4::IDENTITY; 2], SkinningMode::Gpu).expect("fits");
    /// assert_eq!(palette.joint_count().get(), 2);
    /// assert!(matches!(
    ///     SkinningResources::build(&[Mat4::IDENTITY; 129], SkinningMode::Gpu),
    ///     Err(SkinError::PaletteOverflow { .. })
    /// ));
    /// ```
    pub fn build(matrices: &[Mat4], requested: SkinningMode) -> Result<Self, SkinError> {
        if matrices.is_empty() {
            return Err(SkinError::EmptyPalette);
        }
        let limit = JointLimit::GPU;
        if matrices.len() > limit.index() {
            return Err(SkinError::PaletteOverflow {
                count: matrices.len().min(u32::MAX as usize) as u32,
                limit: limit.get(),
            });
        }
        Ok(Self {
            mode: requested,
            count: JointCount::from_raw(matrices.len() as u32),
            palette: matrices.to_vec(),
        })
    }

    /// Which path this palette was built for.
    pub const fn mode(&self) -> SkinningMode {
        self.mode
    }

    /// Validated joint count (`palette.len()`).
    pub const fn joint_count(&self) -> JointCount {
        self.count
    }

    /// Final joint matrices, one per joint.
    pub fn palette_matrices(&self) -> &[Mat4] {
        &self.palette
    }
}

/// Reference blend of one vertex with the GPU-path formula: weighted
/// `palette * vec4(position, 1)` for positions, weighted joint-linear-part
/// transform for normals (renormalized unless the blend collapses).
///
/// This is the CPU mirror the vertex stage is pinned against: same weight
/// canonicalization as [`crate::skin_vertices`] (finite positive sums
/// normalize, otherwise full weight on joint 0) and out-of-range indices
/// read as identity. It deliberately differs from [`crate::skin_vertices`]
/// on normals (inverse-transpose there, linear part here): the two agree
/// exactly for rigid/uniform-scale joints and drift within
/// [`CPU_GPU_TOLERANCE`] otherwise — the documented parity допуск.
pub fn blend_vertex_reference(
    palette: &[Mat4],
    joints: [u16; 4],
    weights: [f32; 4],
    position: [f32; 3],
    normal: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let weights = canonical_reference_weights(weights);
    let vertex = Vec3::from_array(position);
    let direction = Vec3::from_array(normal);
    let mut blended_position = Vec3::ZERO;
    let mut blended_normal = Vec3::ZERO;
    for slot in 0..4 {
        let joint = palette
            .get(joints[slot] as usize)
            .copied()
            .unwrap_or(Mat4::IDENTITY);
        blended_position += joint.transform_point3(vertex) * weights[slot];
        blended_normal += joint.transform_vector3(direction) * weights[slot];
    }
    let position = blended_position.to_array();
    let normal = if blended_normal.length_squared() > 1e-12 {
        blended_normal.normalize().to_array()
    } else {
        blended_normal.to_array()
    };
    (position, normal)
}

/// Canonical per-vertex weights for the reference blend (same rule as the
/// CPU path: finite positive sums normalize, otherwise `(1,0,0,0)`).
fn canonical_reference_weights(weights: [f32; 4]) -> [f32; 4] {
    let finite = weights.iter().all(|slot| slot.is_finite());
    let sum: f32 = weights.iter().sum();
    if finite && sum > 1e-6 {
        [
            weights[0] / sum,
            weights[1] / sum,
            weights[2] / sum,
            weights[3] / sum,
        ]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_limit_is_128_as_a_type() {
        assert_eq!(JointLimit::GPU.get(), 128);
        assert_eq!(JointLimit::GPU.index(), 128);
        assert_eq!(u32::from(JointLimit::GPU), 128);
        assert_eq!(usize::from(JointLimit::GPU), 128);
        // The legacy cap constant is the same limit, not a second number.
        assert_eq!(crate::MAX_JOINTS, JointLimit::GPU.index());
    }

    #[test]
    fn joint_count_conversions() {
        let count = JointCount::from_raw(3);
        assert_eq!(count.get(), 3);
        assert_eq!(count.index(), 3);
        assert_eq!(u32::from(count), 3);
        assert_eq!(usize::from(count), 3);
        assert_eq!(JointCount::from(5u32).get(), 5);
    }

    #[test]
    fn mode_defaults_to_cpu() {
        assert_eq!(SkinningMode::default(), SkinningMode::Cpu);
        assert!(!SkinningMode::Cpu.is_gpu());
        assert!(SkinningMode::Gpu.is_gpu());
    }

    #[test]
    fn build_stages_a_fitting_palette() {
        let palette =
            SkinningResources::build(&[Mat4::IDENTITY; 2], SkinningMode::Gpu).expect("fits");
        assert_eq!(palette.mode(), SkinningMode::Gpu);
        assert_eq!(palette.joint_count().get(), 2);
        assert_eq!(palette.palette_matrices(), &[Mat4::IDENTITY; 2]);
    }

    #[test]
    fn build_rejects_empty_and_overflow() {
        assert_eq!(
            SkinningResources::build(&[], SkinningMode::Gpu),
            Err(SkinError::EmptyPalette)
        );
        let limit = JointLimit::GPU.index();
        let fitting = vec![Mat4::IDENTITY; limit];
        assert!(SkinningResources::build(&fitting, SkinningMode::Gpu).is_ok());
        let overflowing = vec![Mat4::IDENTITY; limit + 1];
        let err = SkinningResources::build(&overflowing, SkinningMode::Gpu)
            .expect_err("over-limit palette must fail, never truncate");
        assert_eq!(
            err,
            SkinError::PaletteOverflow {
                count: (limit + 1) as u32,
                limit: limit as u32,
            }
        );
        // thiserror Display names the counts (no silent verdict).
        assert!(err.to_string().contains("129"));
        assert!(err.to_string().contains("128"));
    }

    #[test]
    fn reference_blend_matches_identity_skin() {
        // Identity palette: the vertex passes through untouched.
        let palette = vec![Mat4::IDENTITY; 2];
        let (position, normal) = blend_vertex_reference(
            &palette,
            [0, 1, 0, 0],
            [0.5, 0.5, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 1.0],
        );
        assert_eq!(position, [1.0, 2.0, 3.0]);
        assert_eq!(normal, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn reference_blend_canonicalizes_weights() {
        // Zero-sum weights fall back to the first slot; out-of-range
        // joints read as identity.
        let palette = vec![Mat4::from_translation(Vec3::X); 1];
        let (position, _) = blend_vertex_reference(
            &palette,
            [7, 0, 0, 0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        assert_eq!(position, [1.0, 0.0, 0.0]);
        // First slot in range: the fallback applies joint 0's translation.
        let (position, _) = blend_vertex_reference(
            &palette,
            [0, 0, 0, 0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        assert!((Vec3::from_array(position) - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-6);
    }
}
