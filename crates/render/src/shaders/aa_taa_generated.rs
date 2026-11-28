//! Temporal anti-aliasing history resolve: translated from Rust through LLVM IR.
//!
//! The shader lives in `crates/gpu-aa-taa`: the pixel is reprojected into the
//! previous frame through the motion vector, the history sample is clamped into
//! the YCoCg range of the current neighbourhood, and the two are blended. A
//! pixel with no surface behind it, an invalid motion word, or a reprojection
//! that leaves the screen keeps the current colour. The pass writes two colour
//! attachments, so the entry returns a struct with `@location(0)` and
//! `@location(1)`.
//!
//! `xtask build-shaders --package ornis-gpu-aa-taa --entry all` writes the
//! single-sample artifact; `--multisampled-textures 0:2` the one whose depth
//! binding is a multisampled image (the lighting MSAA contract).

use super::{Resource, ResourceKind};
use crate::renderer::AaUniform;

/// Vertex entry point of the translated module.
pub const VS_ENTRY: &str = "vs_main";

/// Fragment entry point of the translated module.
pub const FS_ENTRY: &str = "fs_main";

/// Resource layout of the temporal resolve (what the renderer builds its
/// bind-group layout from). The uniform carries the explicit minimum size.
pub const TAA_RESOURCES: [Resource; 7] = [
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
        name: "depth_tex",
        kind: ResourceKind::TextureDepth,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 3,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "motion_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "previous_depth_tex",
        kind: ResourceKind::TextureUint,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 5,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "history_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 6,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "aa_params",
        kind: ResourceKind::Uniform(AaUniform::WGSL_NAME),
        min_size: Some(std::mem::size_of::<AaUniform>() as u64),
    },
];

/// Full WGSL source of the temporal resolve at a sample count: the 4x build
/// only changes the declared depth texture type.
#[must_use]
pub fn wgsl_source_for_samples(sample_count: u32) -> String {
    if sample_count > 1 {
        include_str!("aa_taa_msaa_generated.wgsl").to_owned()
    } else {
        wgsl_source()
    }
}

/// Full WGSL source of the single-sample resolve.
#[must_use]
pub fn wgsl_source() -> String {
    include_str!("aa_taa_generated.wgsl").to_owned()
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
    fn taa_shader_validates_with_naga() {
        assert_valid_wgsl("taa", &wgsl_source());
        assert_valid_wgsl("taa_msaa", &wgsl_source_for_samples(4));
    }

    /// Both entries live in one module, and the resolve writes two targets.
    #[test]
    fn taa_shader_has_two_outputs() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        assert_eq!(
            super::super::entry_stage(&module, VS_ENTRY),
            Some(naga::ShaderStage::Vertex)
        );
        assert_eq!(
            super::super::entry_stage(&module, FS_ENTRY),
            Some(naga::ShaderStage::Fragment)
        );
        let fragment = module
            .entry_points
            .iter()
            .find(|entry| entry.name == FS_ENTRY)
            .expect("fragment entry");
        let result = fragment.function.result.as_ref().expect("result");
        let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
            panic!("the resolve returns a struct");
        };
        assert_eq!(members.len(), 2, "colour and history");
    }

    /// Every table row's binding appears in the shader, and the 4x build
    /// changes exactly the depth declaration.
    #[test]
    fn taa_resources_cover_every_binding() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        let bindings = super::super::declared_bindings(&module);
        assert_eq!(TAA_RESOURCES.len(), 7);
        for resource in TAA_RESOURCES {
            assert!(
                bindings.contains(&(resource.group, resource.binding)),
                "missing {}",
                resource.name
            );
        }
        assert!(wgsl_source().contains("texture_2d<u32>"));
        assert!(!wgsl_source().contains("multisampled"));
        assert!(wgsl_source_for_samples(4).contains("texture_depth_multisampled_2d"));
    }
}
