//! Bloom bright-pass: translated from Rust through LLVM IR.
//!
//! The shader lives in `crates/gpu-bloom`: `#[gpu_entry(vertex)]` draws the
//! shared fullscreen quad, and `#[gpu_entry(fragment)]` keeps what is brighter
//! than `threshold + BLOOM_SOFT_KNEE` with the `smoothstep` knee, so the cut
//! does not band. `xtask build-shaders --package ornis-gpu-bloom --entry all`
//! writes the checked-in [`bloom_generated.wgsl`]; `--check` in CI catches
//! drift.

use super::{Resource, ResourceKind};
use crate::renderer::BloomUniform;

/// Vertex entry point of the translated module.
pub const VS_ENTRY: &str = "vs_main";

/// Fragment entry point of the translated module.
pub const FS_ENTRY: &str = "fs_main";

/// Width of the bloom bright-pass soft knee, added above the luminance
/// threshold. The kernel carries the same number; the parity gate pins them.
pub const BLOOM_SOFT_KNEE: f32 = 0.05;

/// Resource layout of the bloom pass. The uniform carries the explicit
/// minimum size the handwritten layout spelled (`size_of::<BloomUniform>`).
pub const BLOOM_RESOURCES: [Resource; 3] = [
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
        name: "bloom_params",
        kind: ResourceKind::Uniform(BloomUniform::WGSL_NAME),
        min_size: Some(std::mem::size_of::<BloomUniform>() as u64),
    },
];

/// Full WGSL source of the pass: the translated module.
#[must_use]
pub fn wgsl_source() -> String {
    include_str!("bloom_generated.wgsl").to_owned()
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
    fn bloom_generated_validates_with_naga() {
        assert_valid_wgsl("bloom_generated", &wgsl_source());
    }

    #[test]
    fn bloom_generated_contains_both_entries() {
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

    /// The bright pass is a `smoothstep` over the threshold, with the knee the
    /// host constant names.
    #[test]
    fn bloom_generated_keeps_the_soft_knee() {
        let src = wgsl_source();
        assert!(src.contains("smoothstep("), "{src}");
        assert!(
            src.contains(&super::super::f32_lit(BLOOM_SOFT_KNEE)),
            "the knee constant is missing"
        );
    }

    /// Every table row's binding appears in the shader, and the uniform row
    /// carries the explicit minimum size the layout spells.
    #[test]
    fn bloom_resources_cover_shader_and_layout() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        let bindings = super::super::declared_bindings(&module);
        assert_eq!(BLOOM_RESOURCES.len(), 3);
        for resource in BLOOM_RESOURCES {
            assert!(
                bindings.contains(&(resource.group, resource.binding)),
                "missing {}",
                resource.name
            );
        }
        assert!(matches!(
            BLOOM_RESOURCES[2].min_size,
            Some(s) if s == std::mem::size_of::<BloomUniform>() as u64
        ));
    }
}
