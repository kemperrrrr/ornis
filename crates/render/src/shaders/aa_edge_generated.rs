//! FXAA/SMAA edge filter: translated from Rust through LLVM IR.
//!
//! The shader lives in `crates/gpu-aa-edge`: the luma gradient picks the axis an
//! edge runs along, the two taps across it are blended into the centre, and the
//! mode word in the uniform chooses how they are weighted (FXAA evenly, SMAA by
//! luma similarity to the centre). `xtask build-shaders --package
//! ornis-gpu-aa-edge --entry all` writes the checked-in
//! `aa_edge_generated.wgsl`; `--check` in CI catches drift.

use super::{Resource, ResourceKind};
use crate::renderer::AaUniform;

/// Vertex entry point of the translated module.
pub const VS_ENTRY: &str = "vs_main";

/// Fragment entry point of the translated module.
pub const FS_ENTRY: &str = "fs_main";

/// Resource layout of the edge filter (what the renderer builds its
/// bind-group layout from).
pub const EDGE_RESOURCES: [Resource; 3] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "src_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "src_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "aa_params",
        kind: ResourceKind::Uniform(AaUniform::WGSL_NAME),
        min_size: Some(std::mem::size_of::<AaUniform>() as u64),
    },
];

/// Full WGSL source of the edge filter (the mode is a uniform selector, so one
/// module serves FXAA and SMAA).
#[must_use]
pub fn wgsl_source() -> String {
    include_str!("aa_edge_generated.wgsl").to_owned()
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
    fn edge_shader_validates_with_naga() {
        assert_valid_wgsl("edge", &wgsl_source());
    }

    #[test]
    fn edge_shader_has_both_entries() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        assert_eq!(
            super::super::entry_stage(&module, VS_ENTRY),
            Some(naga::ShaderStage::Vertex)
        );
        assert_eq!(
            super::super::entry_stage(&module, FS_ENTRY),
            Some(naga::ShaderStage::Fragment)
        );
    }

    /// Every table row's binding appears in the shader, and the filter selects
    /// its blend from the uniform.
    #[test]
    fn edge_resources_cover_every_binding() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        let bindings = super::super::declared_bindings(&module);
        assert_eq!(EDGE_RESOURCES.len(), 3);
        for resource in EDGE_RESOURCES {
            assert!(
                bindings.contains(&(resource.group, resource.binding)),
                "missing {}",
                resource.name
            );
        }
        assert!(
            EDGE_RESOURCES[2].min_size == Some(std::mem::size_of::<AaUniform>() as u64),
            "the uniform row carries the explicit minimum size"
        );
    }

    #[test]
    fn edge_shader_branches_on_the_mode_word() {
        let source = wgsl_source();
        assert!(source.contains("f4_"), "the mode field is read: {source}");
        assert!(
            source.contains("smoothstep") || source.contains("abs("),
            "{source}"
        );
    }
}
