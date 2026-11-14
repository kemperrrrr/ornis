//! Distance-fog pass: translated from Rust through LLVM IR.
//!
//! The shader lives in `crates/gpu-fog`: `#[gpu_entry(vertex)]` draws the
//! fullscreen quad, and `#[gpu_entry(fragment)]` reads the HDR layer, rebuilds
//! the world position from the hardware depth buffer (`inv_view_proj`), takes
//! the Euclidean distance to the eye, and blends the fog colour in by
//! `1 - exp(-density * distance)` before the composite's ACES curve.
//!
//! Two artifacts come out of that crate: `fog_generated.wgsl` for the
//! single-sample pass and `fog_msaa_generated.wgsl` for the 4x one, whose
//! g-buffer depth is a multisampled image with no resolve target
//! (`--multisampled-textures 0:2`). `xtask build-shaders --check` keeps both in
//! step with the kernel.

use super::{Resource, ResourceKind};
use crate::renderer::{CameraUniform, FogUniform};

/// Vertex entry point of the translated module.
pub const VS_ENTRY: &str = "vs_main";

/// Fragment entry point of the translated module.
pub const FS_ENTRY: &str = "fs_main";

/// Resource layout of the fog pass (what `Renderer3D::create_fog_pass`
/// builds its layout from). The uniform carries the explicit minimum size.
pub const FOG_RESOURCES: [Resource; 5] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "hdr_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "fog_sampler",
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
        name: "camera",
        kind: ResourceKind::Uniform(CameraUniform::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "fog_params",
        kind: ResourceKind::Uniform(FogUniform::WGSL_NAME),
        min_size: Some(std::mem::size_of::<FogUniform>() as u64),
    },
];

/// Full WGSL source for the fog pass at one sample count: the 4x build only
/// changes the declared depth texture type.
#[must_use]
pub fn wgsl_source_for_samples(sample_count: u32) -> String {
    if sample_count > 1 {
        include_str!("fog_msaa_generated.wgsl").to_owned()
    } else {
        wgsl_source()
    }
}

/// Full WGSL source of the single-sample pass.
#[must_use]
pub fn wgsl_source() -> String {
    include_str!("fog_generated.wgsl").to_owned()
}

/// The same module serves the vertex stage: both entries live in it.
#[must_use]
pub fn wgsl_vertex_source() -> String {
    wgsl_source()
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
    fn fog_generated_validates_with_naga() {
        assert_valid_wgsl("fog", &wgsl_source());
        assert_valid_wgsl("fog 4x", &wgsl_source_for_samples(4));
    }

    #[test]
    fn fog_generated_contains_both_entries() {
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

    /// The 4x build changes exactly one declaration: the depth image.
    #[test]
    fn fog_4x_declares_a_multisampled_depth_texture() {
        assert!(wgsl_source().contains("texture_depth_2d"));
        assert!(!wgsl_source().contains("multisampled"));
        assert!(
            wgsl_source_for_samples(4).contains("texture_depth_multisampled_2d"),
            "the 4x pass reads its unresolved g-buffer depth"
        );
    }

    /// Every table row's binding appears in the shader, and every row maps to a
    /// layout entry: shader and pipeline agree.
    #[test]
    fn fog_resources_cover_shader_and_layout() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        let bindings = super::super::declared_bindings(&module);
        assert_eq!(FOG_RESOURCES.len(), 5);
        for resource in FOG_RESOURCES {
            assert!(
                bindings.contains(&(resource.group, resource.binding)),
                "missing {}",
                resource.name
            );
        }
    }
}
