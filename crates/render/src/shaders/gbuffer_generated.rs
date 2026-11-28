//! G-buffer pass: translated from Rust through LLVM IR.
//!
//! The shader lives in `crates/gpu-gbuffer`. Two vertex entries transform one
//! mesh instance with its own matrices — `vs_main` indexes the per-object
//! record with `instance_index`, `vs_main_indirect` first reads the scene index
//! out of the compacted visibility list — and the fragment stage fills the five
//! lighting targets plus the two temporal ones the resolve reads (motion vector
//! and quantized previous depth).
//!
//! `xtask build-shaders --package ornis-gpu-gbuffer --entry all` writes the
//! checked-in `gbuffer_generated.wgsl` (all three entries, one module);
//! `--entry vs_main` writes `gbuffer_vertex_generated.wgsl`, the vertex-only
//! module the forward-PBR pass shares. `--check` in CI catches drift.

use super::{OPENPBR_WGSL_NAME, Resource, ResourceKind};
use crate::renderer::{CameraUniform, PerObjectGpu};

/// Vertex entry point of the translated module (`instance_index` indexes the
/// object record directly).
pub const VS_ENTRY: &str = "vs_main";

/// Vertex entry point that reads the compacted visibility list first.
pub const VS_INDIRECT_ENTRY: &str = "vs_main_indirect";

/// Fragment entry point of the translated module.
pub const FS_ENTRY: &str = "fs_main";

/// The generated g-buffer module: both vertex entries and the fragment stage.
pub fn wgsl_source() -> String {
    include_str!("gbuffer_generated.wgsl").to_string()
}

/// The vertex-only module, shared with the forward-PBR pass.
///
/// A pipeline layout only has to cover the bindings the entry point it draws
/// with actually uses, so the forward pass builds its vertex stage from this
/// narrower module instead of the whole g-buffer one.
pub fn wgsl_vertex_source() -> String {
    include_str!("gbuffer_vertex_generated.wgsl").to_string()
}

/// Resource layout of the g-buffer pass. Type names come from the Rust
/// side (`WGSL_NAME` / `OPENPBR_WGSL_NAME`) — never retyped.
///
/// The translated kernel reads `per_objects` and `materials` word by word
/// (they mix matrices and counters), which the layout does not describe: a
/// bind group layout only fixes the address space and the access mode.
pub const GBUFFER_RESOURCES: [Resource; 3] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        name: "camera",
        kind: ResourceKind::Uniform(CameraUniform::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::VERTEX,
        name: "per_objects",
        kind: ResourceKind::StorageReadArray(PerObjectGpu::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "materials",
        kind: ResourceKind::StorageReadArray(OPENPBR_WGSL_NAME),
        min_size: None,
    },
];

/// Group-1 compacted source indices for the indirect vertex entry.
pub(crate) const GBUFFER_INDICES_RESOURCES: &[Resource] = &[Resource {
    group: 1,
    binding: 0,
    visibility: wgpu::ShaderStages::VERTEX,
    name: "visible_indices",
    kind: ResourceKind::StorageReadArray("u32"),
    min_size: None,
}];

#[cfg(test)]
mod tests {
    use super::super::{bgl_entry, declared_bindings, entry_stage};
    use super::*;

    fn assert_valid_wgsl(name: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name} must parse: {e}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} must validate: {e}"));
    }

    #[test]
    fn gbuffer_generated_validates_with_naga() {
        assert_valid_wgsl("gbuffer", &wgsl_source());
        assert_valid_wgsl("gbuffer_vertex", &wgsl_vertex_source());
    }

    /// All three entries live in one module, and the fragment stage fills
    /// seven targets.
    #[test]
    fn gbuffer_shader_has_every_entry_and_target() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        assert_eq!(
            entry_stage(&module, VS_ENTRY),
            Some(naga::ShaderStage::Vertex)
        );
        assert_eq!(
            entry_stage(&module, VS_INDIRECT_ENTRY),
            Some(naga::ShaderStage::Vertex)
        );
        assert_eq!(
            entry_stage(&module, FS_ENTRY),
            Some(naga::ShaderStage::Fragment)
        );
        let fragment = module
            .entry_points
            .iter()
            .find(|entry| entry.name == FS_ENTRY)
            .expect("fragment entry");
        let result = fragment.function.result.as_ref().expect("result");
        let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
            panic!("the fragment stage returns a struct");
        };
        assert_eq!(members.len(), 7, "five lighting targets plus motion");
    }

    /// The vertex-only module carries the shared entry and nothing else, so
    /// the forward pass does not inherit the g-buffer's fragment bindings.
    #[test]
    fn gbuffer_vertex_module_is_vertex_only() {
        let module = naga::front::wgsl::parse_str(&wgsl_vertex_source()).expect("parse");
        assert_eq!(module.entry_points.len(), 1, "one entry point");
        assert_eq!(
            entry_stage(&module, VS_ENTRY),
            Some(naga::ShaderStage::Vertex)
        );
        let bindings = declared_bindings(&module);
        assert_eq!(bindings.len(), 2, "{bindings:?}");
    }

    /// Every table row's binding appears in the shader, and every row maps to
    /// a layout entry: shader and pipeline agree.
    #[test]
    fn gbuffer_resources_cover_every_binding() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        let bindings = declared_bindings(&module);
        for resource in GBUFFER_INDICES_RESOURCES {
            assert_eq!(
                bgl_entry(resource, false).binding,
                resource.binding,
                "{}",
                resource.name
            );
            assert!(
                bindings.contains(&(resource.group, resource.binding)),
                "missing {}",
                resource.name
            );
        }
        assert_eq!(GBUFFER_RESOURCES.len(), 3);
        for resource in GBUFFER_RESOURCES {
            assert!(
                bindings.contains(&(resource.group, resource.binding)),
                "missing {}",
                resource.name
            );
            let entry = bgl_entry(&resource, false);
            assert_eq!(
                (entry.binding, entry.visibility),
                (resource.binding, resource.visibility)
            );
        }
    }

    /// The module declares the compacted visibility list the indirect entry
    /// reads, and the table binds it at that same address.
    #[test]
    fn gbuffer_indirect_entry_reads_the_visibility_list() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        let group_one: Vec<(u32, u32)> = declared_bindings(&module)
            .into_iter()
            .filter(|(group, _)| *group == 1)
            .collect();
        let expected: Vec<(u32, u32)> = GBUFFER_INDICES_RESOURCES
            .iter()
            .map(|resource| (resource.group, resource.binding))
            .collect();
        assert_eq!(group_one, expected);
        assert_eq!(
            entry_stage(&module, VS_INDIRECT_ENTRY),
            Some(naga::ShaderStage::Vertex)
        );
    }
}
