//! UI composite pass: translated from Rust through LLVM IR.
//!
//! The shader itself lives in `crates/gpu-composite` and is written against the
//! stable SDK: `#[gpu_entry(vertex)]` draws the quad from two module constants
//! indexed by `@builtin(vertex_index)`, and `#[gpu_entry(fragment)]` samples the
//! PBR and UI layers, decodes the UI texel from sRGB to linear light, and mixes
//! it over the scene by its alpha. `xtask build-shaders --package
//! ornis-gpu-composite --entry all` compiles that crate for
//! `wasm32-unknown-unknown`, translates its LLVM IR, and writes the checked-in
//! `composite_generated.wgsl` next to this file; `--check` in CI catches drift.
//!
//! The resource table below is still the single source of truth for the bind
//! group layout, and it matches the `#[gpu(...)]` attributes in the kernel.

use super::{Resource, ResourceKind};

/// Vertex entry point of the translated module.
pub const VS_ENTRY: &str = "vs_main";

/// Fragment entry point of the translated module.
pub const FS_ENTRY: &str = "fs_main";

/// Resource layout of the composite pass (`CompositePass` builds its layout
/// from this table).
pub const COMPOSITE_RESOURCES: [Resource; 4] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "pbr_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "pbr_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ui_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 3,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ui_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
];

/// Full WGSL source of the pass: the translated module, checked in beside this
/// file and rebuilt by `xtask build-shaders`.
#[must_use]
pub fn wgsl_source() -> String {
    include_str!("composite_generated.wgsl").to_owned()
}

#[cfg(test)]
mod tests {
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
    fn composite_generated_validates_with_naga() {
        assert_valid_wgsl("composite_generated", &wgsl_source());
    }

    #[test]
    fn composite_generated_contains_both_entries() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        // The module keeps the IR's declaration order, so the entries are
        // matched by name and stage rather than by position.
        assert_eq!(
            super::super::entry_stage(&module, VS_ENTRY),
            Some(naga::ShaderStage::Vertex)
        );
        assert_eq!(
            super::super::entry_stage(&module, FS_ENTRY),
            Some(naga::ShaderStage::Fragment)
        );
        assert_eq!(module.entry_points.len(), 2);
    }

    /// The translated bindings are declared where the resource table says.
    #[test]
    fn composite_resources_cover_shader_and_layout() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        let bindings = super::super::declared_bindings(&module);
        assert_eq!(COMPOSITE_RESOURCES.len(), 4);
        for resource in COMPOSITE_RESOURCES {
            assert!(
                bindings.contains(&(resource.group, resource.binding)),
                "missing {}",
                resource.name
            );
        }
    }

    /// The UI texel goes through the sRGB decode before the mix.
    #[test]
    fn composite_generated_decodes_srgb() {
        let src = wgsl_source();
        assert!(src.contains("pow("), "{src}");
        assert!(src.contains("0.04045"), "{src}");
    }
}
