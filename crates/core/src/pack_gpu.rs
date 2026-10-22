//! GPU half of a `#[derive(Pack)]` bundle: SoA columns as storage lanes.
//!
//! `#[pack(gpu)]` on a Pack bundle makes the derive emit [`PackGpu`] for
//! it. One lane per field, in declaration order, each described by a
//! [`PackGpuLane`]: the WGSL element spelling of the column and whether
//! the column is sparse (`Option<T>` fields — absent entities upload as
//! element zero). The descriptors are pure data — no wgpu types cross
//! this seam; layout/render crates turn them into bind-group entries.
//!
//! [`PackGpu::pack_gpu_upload`] serializes the store's lanes into column
//! bytes aligned to one shared index space: the ascending entity-id
//! order of the first non-sparse lane's members (an all-sparse bundle —
//! every field `Option<T>` — takes the sorted union of its lanes
//! instead, so a leading sparse field cannot drop entities from the
//! other columns; CPU-side [`crate::Pack::pack_entities`] keeps its own
//! first-field rule). Each lane is looked up per entity, so
//! dense-order churn from swap-removal never misaligns a column;
//! entities missing from a lane (sparse `None`, or a desynced
//! dense lane) upload as the all-zero element. An empty store uploads
//! one zeroed element per column: wgpu rejects zero-size buffer
//! bindings, so a scene that starts empty must still bind.

use crate::Pack;
use crate::smart_store::SmartStore;

/// One SoA column of a GPU-enabled pack bundle, as a storage resource.
///
/// The lane's WGSL declaration is
/// `@group(0) @binding(i) var<storage, read> {field}: array<{wgsl_elem}>;`
/// with `i` assigned by the consuming pipeline (declaration order of
/// [`PackGpu::pack_gpu_lanes`], continuing whatever binding numbering
/// the pipeline's other resources used).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackGpuLane {
    /// Bundle field name; also the WGSL variable name of the lane.
    pub field: &'static str,
    /// WGSL element spelling of the column (`f32`, a struct-mirror name,
    /// …). Scalars map to themselves; every other field type passes
    /// through as a same-name struct mirror (pair with
    /// `#[derive(WgslStruct)]` or a handwritten WGSL `struct`).
    pub wgsl_elem: &'static str,
    /// `true` for `Option<T>` fields: the lane stores `T`, and entities
    /// without a value upload as the all-zero element.
    pub sparse: bool,
}

/// Aligned GPU upload of every lane of a [`PackGpu`] bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackGpuUpload {
    /// Number of entities in the shared index space (column length).
    pub len: usize,
    /// One byte column per lane, in [`PackGpu::pack_gpu_lanes`] order.
    /// For `len > 0` each column is `len * size_of::<Elem>()` bytes,
    /// laid out exactly as `array<Elem>` expects for the lane's element
    /// type; for `len == 0` a column carries one zeroed element, because
    /// wgpu rejects zero-size buffer bindings.
    pub columns: Vec<Vec<u8>>,
}

/// GPU view of a [`crate::Pack`] bundle: its SoA columns as storage lanes.
///
/// Implemented by `#[derive(Pack)]` when the struct carries
/// `#[pack(gpu)]`; field types must be `bytemuck::Pod` (the upload
/// writes the column with `bytemuck::bytes_of`, and the derive asserts
/// the bound on the field's span).
pub trait PackGpu: Pack {
    /// Lanes in field declaration order. Empty for bundles that are
    /// CPU-only (no `#[pack(gpu)]`).
    fn pack_gpu_lanes() -> &'static [PackGpuLane];

    /// Serializes every lane against the shared index space. `None`
    /// when the store has no lanes registered for this bundle (the
    /// bundle was never `pack_register`ed).
    fn pack_gpu_upload(store: &SmartStore) -> Option<PackGpuUpload>;
}
