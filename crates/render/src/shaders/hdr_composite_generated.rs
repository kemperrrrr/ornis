//! HDR composite pass: translated from Rust through LLVM IR.
//!
//! The shader lives in `crates/gpu-hdr-composite`: `#[gpu_entry(vertex)]` draws
//! the shared fullscreen quad, and `#[gpu_entry(fragment)]` mixes the deferred
//! and forward layers by the bloom mode word, adds the bloom layer scaled by its
//! intensity, applies the auto-exposure multiplier, and tone maps with the ACES
//! fit. `xtask build-shaders --package ornis-gpu-hdr-composite --entry all`
//! compiles the crate and writes the checked-in `hdr_composite_generated.wgsl`
//! beside this file; `--check` in CI catches drift.
//!
//! The resource table below is still the single source of truth for the bind
//! group layout, and it matches the `#[gpu(...)]` attributes in the kernel.

use super::{Resource, ResourceKind};
use crate::auto_exposure::{EXPOSURE_WGSL_NAME, ExposureParams};
use crate::renderer::BloomUniform;

/// Vertex entry point of the translated module.
pub const VS_ENTRY: &str = "vs_main";

/// Fragment entry point of the translated module.
pub const FS_ENTRY: &str = "fs_main";

/// Full WGSL source of the pass: the translated module.
#[must_use]
pub fn wgsl_source() -> String {
    include_str!("hdr_composite_generated.wgsl").to_owned()
}

/// The same module serves the vertex stage: both entries live in it.
#[must_use]
pub fn wgsl_vertex_source() -> String {
    wgsl_source()
}

/// Resource layout of the HDR composite pass (what `create_composite_pass`
/// builds its layout from). The uniform carries the explicit minimum size
/// the handwritten layout spelled.
pub const HDR_RESOURCES: [Resource; 6] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "deferred_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "forward_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "composite_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 3,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "bloom_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "bloom_params",
        kind: ResourceKind::Uniform(BloomUniform::WGSL_NAME),
        min_size: Some(std::mem::size_of::<BloomUniform>() as u64),
    },
    Resource {
        group: 0,
        binding: 5,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "exposure_params",
        kind: ResourceKind::Uniform(EXPOSURE_WGSL_NAME),
        min_size: Some(std::mem::size_of::<ExposureParams>() as u64),
    },
];

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
    fn hdr_composite_generated_validates_with_naga() {
        assert_valid_wgsl("hdr_composite_generated", &wgsl_source());
    }

    #[test]
    fn hdr_composite_generated_contains_both_entries() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
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

    /// The mode word, the exposure multiplier and the bloom intensity are read
    /// from their uniforms, and the ACES fit is inline.
    #[test]
    fn hdr_composite_generated_reads_its_uniforms() {
        let src = wgsl_source();
        assert!(src.contains("switch "), "{src}");
        assert!(src.contains("2.51f"), "{src}");
        assert!(src.contains("2.43f"), "{src}");
    }

    /// Every table row's declaration appears in the shader, and every row maps
    /// to a layout entry: shader and pipeline agree.
    #[test]
    fn hdr_resources_cover_shader_and_layout() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        let bindings = super::super::declared_bindings(&module);
        assert_eq!(HDR_RESOURCES.len(), 6);
        for resource in HDR_RESOURCES {
            assert!(
                bindings.contains(&(resource.group, resource.binding)),
                "missing {}",
                resource.name
            );
        }
    }
}
