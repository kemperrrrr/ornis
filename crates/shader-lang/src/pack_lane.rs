//! Pack-lane storage resources: one SoA column of a GPU-enabled
//! `#[derive(Pack)]` bundle, described for the shader side.
//!
//! The macro that stitches pack lanes into a `#[gpu_pipeline(pack(T))]`
//! shader formats declarations from the runtime descriptors
//! (`ornis_core::PackGpuLane`); [`PackLaneDesc`] is this crate's
//! canonical spelling of the same resource, and integration tests pin
//! the two together so the templates cannot drift.

/// One SoA column of a pack bundle, as a WGSL storage resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackLaneDesc {
    /// Bundle field name; also the WGSL variable name of the lane.
    pub field: &'static str,
    /// WGSL element spelling of the column.
    pub wgsl_elem: &'static str,
    /// `true` for `Option<T>` fields (absent entities upload as zero).
    pub sparse: bool,
}

impl PackLaneDesc {
    /// The module-scope WGSL declaration of this lane:
    /// `@group({group}) @binding({binding}) var<storage, read> {field}: array<{elem}>;`
    ///
    /// Pack lanes are read-only in v1 — writes stay on the CPU side.
    pub fn decl(&self, group: u32, binding: u32) -> String {
        format!(
            "@group({group}) @binding({binding}) var<storage, read> {}: array<{}>;",
            self.field, self.wgsl_elem
        )
    }
}

#[cfg(test)]
mod tests {
    use super::PackLaneDesc;

    fn lane(field: &'static str, elem: &'static str, sparse: bool) -> PackLaneDesc {
        PackLaneDesc {
            field,
            wgsl_elem: elem,
            sparse,
        }
    }

    #[test]
    fn decl_is_the_storage_line() {
        assert_eq!(
            lane("pos", "vec3<f32>", false).decl(0, 2),
            "@group(0) @binding(2) var<storage, read> pos: array<vec3<f32>>;"
        );
        assert_eq!(
            lane("hp", "f32", true).decl(0, 0),
            "@group(0) @binding(0) var<storage, read> hp: array<f32>;"
        );
    }
}
