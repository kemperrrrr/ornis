//! Clustered lighting geometry (#130): the 16×9×24 cluster grid, the
//! GPU-side span/index-list layout (`ClusterSpan`/`ClusterParams`), and
//! the shared constants every consumer formats from — the shading
//! lookup (`shaders::helpers::cluster_light_span`), the assignment
//! compute pass ([`wgsl`]) and the CPU reference ([`cpu`], test-only).
//! The GPU owner lives in [`gpu`].
use super::*;

mod gpu;
#[cfg(test)]
pub(crate) mod cpu;
mod wgsl;

pub(crate) use gpu::ClusteredLights;

/// Horizontal cluster tiles across the screen (#130 cluster grid 16×9×24).
pub(crate) const CLUSTER_TILE_COUNT_X: u32 = 16;
/// Vertical cluster tiles across the screen.
pub(crate) const CLUSTER_TILE_COUNT_Y: u32 = 9;
/// View-depth slices (exponential between near and far).
pub(crate) const CLUSTER_SLICE_COUNT: u32 = 24;
/// Total clusters: one `ClusterSpan` per entry of the grid buffer.
pub(crate) const CLUSTER_COUNT: u32 =
    CLUSTER_TILE_COUNT_X * CLUSTER_TILE_COUNT_Y * CLUSTER_SLICE_COUNT;
/// View depth where the exponential slice split starts; slice 0 spans
/// `[0, near]`.
pub(crate) const CLUSTER_DEPTH_NEAR: f32 = 0.5;
/// View depth where the exponential slice split ends; the last slice
/// spans up to [`CLUSTER_FAR_LIMIT`].
pub(crate) const CLUSTER_DEPTH_FAR: f32 = 400.0;
/// Upper view depth of the last slice: beyond this nothing is clustered.
pub(crate) const CLUSTER_FAR_LIMIT: f32 = 1.0e9;
/// Light records the cluster storage buffer holds (uniform window head
/// + clustered point/spot records).
pub(crate) const MAX_CLUSTERED_LIGHTS: u32 = 4096;
/// Per-cluster capacity of the index list; a cluster with more
/// intersecting lights keeps the first ones and counts the rest as
/// overflow (the frame never breaks).
pub(crate) const MAX_LIGHTS_PER_CLUSTER: u32 = 128;
/// Total index-list words: one span range per cluster at full capacity.
pub(crate) const CLUSTER_INDEX_LIST_LEN: u32 = CLUSTER_COUNT * MAX_LIGHTS_PER_CLUSTER;
/// Compute-pass workgroup size (one thread per cluster).
pub(crate) const CLUSTER_WORKGROUP_SIZE: u32 = 64;
/// Workgroups dispatched for the whole grid.
pub(crate) const CLUSTER_WORKGROUPS: u32 = CLUSTER_COUNT / CLUSTER_WORKGROUP_SIZE;
/// Outer-cone cosine below which the spot bound falls back to the plain
/// point-light sphere (cone too wide to shrink).
pub(crate) const SPOT_BOUND_COS_FLOOR: f32 = 0.1;

/// `ln(far/near)` of the exponential slice split. A function (not a
/// `const`) because `ln` is not const-evaluable; every consumer — the
/// CPU reference, the fragment consts block and the compute shader —
/// formats this same value, so no second spelling can drift.
pub(crate) fn cluster_log_ratio() -> f32 {
    (CLUSTER_DEPTH_FAR / CLUSTER_DEPTH_NEAR).ln()
}

/// Multiplier turning `ln(depth/near)` into a slice index:
/// `slice = ceil(scale * ln(depth/near))`, clamped to
/// `[0, CLUSTER_SLICE_COUNT - 1]` — the inverse of the exponential
/// slice boundary map (see `clusters::cpu::cluster_slice_z_max`).
pub(crate) fn cluster_slice_scale() -> f32 {
    (CLUSTER_SLICE_COUNT as f32 - 1.0) / cluster_log_ratio()
}

/// One cluster's slice into the shared light index list (#130): the
/// `grid` storage buffer is `array<ClusterSpan>`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "ClusterSpan")]
pub(crate) struct ClusterSpan {
    /// First word of the span in `cluster_light_indices`.
    pub(crate) offset: u32,
    /// Words in the span (intersecting lights, ascending storage order).
    pub(crate) count: u32,
}

/// Cluster-assignment parameters uploaded by
/// [`Renderer3D::set_lights_full`](super::Renderer3D::set_lights_full)
/// and bound in every lit stage plus the compute pass. `clustered` is
/// the runtime flag: the shading lookup multiplies the span by it, so
/// the off path is bit-identical to the legacy window loop.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "ClusterParams")]
pub(crate) struct ClusterParams {
    /// Total light records in the cluster storage (head + clustered).
    pub(crate) light_count: u32,
    /// Head size: the uniform-window records every pixel evaluates
    /// before the cluster slots (equals `lighting.light_count`).
    pub(crate) head_count: u32,
    /// 1 while clustered lighting is enabled, 0 otherwise.
    pub(crate) clustered: u32,
    /// Tail pad keeping the uniform block 16 bytes.
    pub(crate) _pad: u32,
}

/// Cluster-side upload report published through
/// [`Renderer3D::cluster_light_stats`](super::Renderer3D::cluster_light_stats).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClusterLightStats {
    /// Point/spot lights uploaded into the cluster storage beyond the
    /// uniform window.
    pub clustered: u32,
    /// Point/spot lights beyond the storage capacity (never silently).
    pub dropped: u32,
}
