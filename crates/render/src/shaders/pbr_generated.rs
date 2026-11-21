//! Forward-PBR pass: translated from Rust through LLVM IR.
//!
//! The shader lives in `crates/gpu-pbr`: one fragment entry that reads the
//! material record, walks the punctual lights, sums the OpenPBR layers and adds
//! the ambient, emissive and image-based terms. The layer maths it calls is
//! `ornis-gpu-openpbr`, the same ported functions the other passes use.
//!
//! The vertex stage is shared with the g-buffer pass — the two chains transform
//! an instance identically — so [`wgsl_vertex_source`] reuses the g-buffer's
//! vertex-only module instead of duplicating it.
//!
//! `xtask build-shaders --package ornis-gpu-pbr --entry fs_main` writes the
//! checked-in `pbr_generated.wgsl`; `--check` in CI catches drift.

use super::{OPENPBR_WGSL_NAME, Resource, ResourceKind};
use crate::renderer::{
    CameraUniform, ClusterParams, ClusterSpan, GpuLight, LightingUniform, PerObjectGpu,
};

/// Fragment entry point of the translated module.
pub const FS_ENTRY: &str = "fs_main";

/// Forward-PBR vertex shader: instance transforms + world position.
///
/// Delegates to [`super::gbuffer_generated::wgsl_vertex_source`] — both passes use the
/// same instance transform, and the g-buffer's is already translated.
pub fn wgsl_vertex_source() -> String {
    super::gbuffer_generated::wgsl_vertex_source()
}

/// The generated forward-PBR fragment module.
pub fn wgsl_source() -> String {
    include_str!("pbr_generated.wgsl").to_string()
}

/// Resource layout of the forward-PBR pass (vertex 0–1, fragment 0, 2–3).
/// Shared by `create_pbr_bind_group` and `create_forward_pass`, whose
/// handwritten layouts were identical. Type names come from the Rust side.
pub const PBR_RESOURCES: [Resource; 16] = [
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
    Resource {
        group: 0,
        binding: 3,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "lighting",
        kind: ResourceKind::Uniform(LightingUniform::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_tex",
        kind: ResourceKind::TextureDepthArray,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 5,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_sampler",
        kind: ResourceKind::SamplerComparison,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 6,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_cube_tex",
        kind: ResourceKind::TextureDepthCubeArray,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 7,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "prefilter_cube",
        kind: ResourceKind::TextureCube,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 8,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "irradiance_cube",
        kind: ResourceKind::TextureCube,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 9,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "brdf_lut",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 10,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ibl_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 11,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "csm_tex",
        kind: ResourceKind::TextureDepthArray,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 12,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "cluster_params",
        kind: ResourceKind::Uniform(ClusterParams::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 13,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "clustered_lights",
        kind: ResourceKind::StorageReadArray(GpuLight::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 14,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "cluster_light_indices",
        kind: ResourceKind::StorageReadArray("u32"),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 15,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "cluster_grid",
        kind: ResourceKind::StorageReadArray(ClusterSpan::WGSL_NAME),
        min_size: None,
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
    fn pbr_generated_validates_with_naga() {
        assert_valid_wgsl("pbr_vertex", &wgsl_vertex_source());
        assert_valid_wgsl("pbr_fragment", &wgsl_source());
    }

    /// The fragment module declares every binding the layout names, and the
    /// vertex module is the g-buffer's.
    #[test]
    fn pbr_generated_contains_expected_bindings() {
        let vs = wgsl_vertex_source();
        assert_eq!(vs, super::super::gbuffer_generated::wgsl_vertex_source());
        let fs = wgsl_source();
        assert!(fs.contains("fn fs_main("));
        assert!(!fs.contains("fn vs_main("), "fragment-only module");
        for binding in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
            assert!(
                fs.contains(&format!("@group(0) @binding({binding})")),
                "binding {binding} is declared"
            );
        }
    }

    /// The uniform blocks land on the host byte offsets.
    ///
    /// The pixel gate renders both shaders from the same buffers, so a layout
    /// error cancels out there; this test is what pins the ABI. `Lighting`
    /// carries `array<Light, 8>`, which pushes `light_count` from byte 20 to
    /// 1168, and `csm` follows at 1184 — the same numbers
    /// `crate::renderer::LightingUniform` describes.
    #[test]
    fn uniform_offsets_match_the_host_records() {
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("parse");
        let context = naga::proc::GlobalCtx {
            types: &module.types,
            constants: &module.constants,
            overrides: &module.overrides,
            global_expressions: &module.global_expressions,
        };
        naga::proc::Layouter::default()
            .update(context)
            .expect("layout");
        let mut ordering: Vec<(String, naga::Handle<naga::Type>)> = module
            .types
            .iter()
            .filter_map(|(handle, ty)| ty.name.clone().map(|name| (name, handle)))
            .collect();
        ordering.sort();
        let named = |name: &str| {
            ordering
                .iter()
                .find(|(candidate, _)| candidate.trim_end_matches('_') == name)
                .unwrap_or_else(|| panic!("type {name} is not declared"))
                .1
        };
        let members = |handle: naga::Handle<naga::Type>| match &module.types[handle].inner {
            naga::TypeInner::Struct { members, span } => {
                let mut out: Vec<(u32, naga::Handle<naga::Type>)> =
                    members.iter().map(|m| (m.offset, m.ty)).collect();
                out.sort();
                (out, *span)
            }
            other => panic!("expected a struct, found {other:?}"),
        };
        let (lighting, span) = members(named("UniformStruct4"));
        let offsets: Vec<u32> = lighting.iter().map(|(offset, _)| *offset).collect();
        assert_eq!(offsets, [0, 16, 1168, 1172, 1176, 1180, 1184]);
        assert_eq!(span, 1488);
        let (csm, csm_span) = members(lighting.last().expect("the cascade block").1);
        assert_eq!(
            csm.iter().map(|(offset, _)| *offset).collect::<Vec<_>>(),
            [0, 256, 272, 288]
        );
        assert_eq!(csm_span, 304);
        let (cluster, cluster_span) = members(named("UniformStruct5"));
        assert_eq!(cluster_span, 16);
        assert_eq!(cluster.len(), 4);
        // The light array's stride is the host record's size.
        let naga::TypeInner::Array { stride, .. } = &module.types[lighting[1].1].inner else {
            panic!("the second field is the light array");
        };
        assert_eq!(*stride, 144);
    }

    /// Every table row is bound at an address the shaders declare, and every
    /// row maps to a layout entry: shader and pipeline agree.
    #[test]
    fn pbr_resources_cover_stages_and_layout() {
        use super::super::{bgl_entry, declared_bindings};
        let vertex = naga::front::wgsl::parse_str(&wgsl_vertex_source()).expect("vertex");
        let fragment = naga::front::wgsl::parse_str(&wgsl_source()).expect("fragment");
        let mut bindings = declared_bindings(&vertex);
        bindings.extend(declared_bindings(&fragment));
        assert_eq!(PBR_RESOURCES.len(), 16);
        for r in PBR_RESOURCES {
            assert!(
                bindings.contains(&(r.group, r.binding)),
                "missing {} at {}:{}",
                r.name,
                r.group,
                r.binding
            );
            let e = bgl_entry(&r, false);
            assert_eq!((e.binding, e.visibility), (r.binding, r.visibility));
        }
    }
}
