//! Deferred lighting shader: translated from Rust through LLVM IR.
//!
//! Both fragment entries are checked-in llvm2gpu artifacts of
//! `crates/gpu-lighting` (`fs_main_lighting` at 1x,
//! `fs_main_lighting_msaa` at 4x), and the vertex stage is the shared
//! [`quad_vertex_generated`](super::quad_vertex_generated) module — the pass
//! has no hand-written WGSL left. What lives here is the resource table the
//! bind-group layout is built from, the per-sample rule that table follows at
//! 4x, and the mapping from a sample count to the artifact.

use super::{OPENPBR_WGSL_NAME, Resource, ResourceKind};
use crate::renderer::{CameraUniform, ClusterParams, ClusterSpan, GpuLight, LightingUniform};

/// Depth, the material id and the normal stay per-sample at 4x.
///
/// The normal must not go through the hardware box filter: a cleared texel is
/// `(0, 0)`, and the octahedral decode of that is `+Z`, which fringes every
/// silhouette. The other layers resolve, so they bind single-sample even when
/// the pass runs at 4x.
pub(crate) fn keeps_per_sample(name: &str, kind: &ResourceKind) -> bool {
    name == "normal_tex" || matches!(kind, ResourceKind::TextureDepth | ResourceKind::TextureUint)
}

/// Whether an MSAA lighting binding keeps per-sample storage.
///
/// The translated 4x artifact declares exactly these three bindings as
/// multisampled (`--multisampled-textures 0:4,0:5,0:8`), so the layout and the
/// shader read the same sentence.
pub(crate) fn per_sample_flag(sample_count: u32, keeps: bool) -> bool {
    sample_count > 1 && keeps
}

/// Resource layout of the deferred-lighting pass: where each resource
/// binds, when it is visible, under what name. Type names come from the
/// Rust side (`WGSL_NAME` / `OPENPBR_WGSL_NAME`) — never retyped.
pub const LIGHTING_RESOURCES: [Resource; 23] = [
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
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "lighting",
        kind: ResourceKind::Uniform(LightingUniform::WGSL_NAME),
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
        name: "albedo_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "normal_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 5,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "material_id_tex",
        kind: ResourceKind::TextureUint,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 6,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "world_pos_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 7,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "mat_params_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 8,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "depth_tex",
        kind: ResourceKind::TextureDepth,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 9,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "lighting_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 10,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_tex",
        kind: ResourceKind::TextureDepthArray,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 11,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_sampler",
        kind: ResourceKind::SamplerComparison,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 12,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "shadow_cube_tex",
        kind: ResourceKind::TextureDepthCubeArray,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 13,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "prefilter_cube",
        kind: ResourceKind::TextureCube,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 14,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "irradiance_cube",
        kind: ResourceKind::TextureCube,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 15,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "brdf_lut",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 16,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ibl_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 17,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "ao_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 18,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "csm_tex",
        kind: ResourceKind::TextureDepthArray,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 19,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "cluster_params",
        kind: ResourceKind::Uniform(ClusterParams::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 20,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "clustered_lights",
        kind: ResourceKind::StorageReadArray(GpuLight::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 21,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "cluster_light_indices",
        kind: ResourceKind::StorageReadArray("u32"),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 22,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "cluster_grid",
        kind: ResourceKind::StorageReadArray(ClusterSpan::WGSL_NAME),
        min_size: None,
    },
];

/// The translated 1x artifact: `fs_main_lighting` of `ornis-gpu-lighting`.
pub const LIGHTING_1X: &str = include_str!("lighting_generated.wgsl");

/// The translated 4x artifact: `fs_main_lighting_msaa` of `ornis-gpu-lighting`.
///
/// Built with `--multisampled-textures 0:4,0:5,0:8`, which declares depth, the
/// material index and the normal per sample: those three are what the entry
/// loads sample by sample. Every other layer resolves and stays single-sample,
/// so it is still read with `textureSampleLevel`.
pub const LIGHTING_MSAA: &str = include_str!("lighting_msaa_generated.wgsl");

/// Full WGSL source for deferred lighting, assembled from Rust.
pub fn wgsl_source() -> String {
    wgsl_source_for_samples(1)
}

/// Full WGSL source for deferred lighting at a sample count.
///
/// Both paths are translated artifacts now: at 1x `ornis-gpu-lighting`'s
/// `fs_main_lighting`, at 4x its `fs_main_lighting_msaa`. They are separate
/// builds of the same crate because the multisampled binding types belong to
/// the module, not to the entry:
/// `cargo xtask build-shaders --package ornis-gpu-lighting --entry fs_main_lighting`
/// and the same with `--entry fs_main_lighting_msaa --multisampled-textures 0:4,0:5,0:8`.
pub fn wgsl_source_for_samples(sample_count: u32) -> String {
    if sample_count <= 1 {
        LIGHTING_1X.to_string()
    } else {
        LIGHTING_MSAA.to_string()
    }
}

/// Name of the translated 1x fragment entry.
pub const LIGHTING_1X_ENTRY: &str = "fs_main_lighting";

/// Name of the translated 4x fragment entry.
pub const LIGHTING_MSAA_ENTRY: &str = "fs_main_lighting_msaa";

/// Vertex entry-point name (the pipeline-convention ABI).
///
/// The stage is the shared quad module: the lighting pass draws the same
/// triangle-strip quad as every other post pass, so it binds
/// [`quad_vertex_generated`](super::quad_vertex_generated) for the vertex stage
/// and its own artifact for the fragment.
pub use super::quad_vertex_generated::VS_ENTRY;

/// Full WGSL source for the shared quad vertex stage.
#[must_use]
pub fn wgsl_vertex_source() -> String {
    super::quad_vertex_generated::wgsl_vertex_source()
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
    fn lighting_generated_validates_with_naga() {
        assert_valid_wgsl("lighting_generated", &wgsl_source());
    }

    /// The artifact is a single fully-inlined module, so the checks are on the
    /// shapes that survive inlining: the deferred decode, the material table
    /// read and the two shadow families.
    #[test]
    fn lighting_artifact_keeps_the_deferred_shapes() {
        let src = wgsl_source();
        assert!(
            !src.contains("aces_tonemap"),
            "lighting writes scene-linear HDR; ACES belongs to the composite"
        );
        assert!(src.contains("fn fs_main"));
        // Depth is loaded, not sampled: the world position comes back through
        // the inverse view-projection.
        assert!(src.contains("textureLoad(tex_0_8_"), "{src}");
        // The material record is read as storage words and bitcast, never
        // converted: a conversion would turn the packed index into a denormal.
        assert!(
            src.contains("var<storage> storage_0_2_: array<u32>"),
            "{src}"
        );
        assert!(src.contains("bitcast<f32>("), "{src}");
        // Both shadow families: the atlas (10) and the cube pool (12).
        assert!(src.contains("textureSampleCompareLevel(tex_0_10_"), "{src}");
        assert!(src.contains("textureSampleCompareLevel(tex_0_12_"), "{src}");
    }

    /// Deferred lighting keeps both the AO factor from #137 and the CSM
    /// helper from #121. Dropping either binding or the `csm_visibility`
    /// call fails this; naga still has to accept the combined module.
    /// Deferred lighting keeps both the AO layer from #137 and the cascaded
    /// shadow map from #121: drop either binding and the walk loses a term.
    #[test]
    fn lighting_keeps_ambient_occlusion_and_csm() {
        let src = wgsl_source();
        assert_valid_wgsl("lighting ao+csm", &src);
        assert!(src.contains("textureSampleLevel(tex_0_17_"), "{src}");
        assert!(src.contains("textureSampleCompareLevel(tex_0_18_"), "{src}");
    }

    /// The resource table drives both sides: WGSL declarations and `wgpu`
    /// layout entries agree on numbers, visibility and kinds.
    #[test]
    fn lighting_resources_drive_bgl_and_wgsl() {
        use super::super::{bgl_entry, resource_decl};
        use super::LIGHTING_RESOURCES;
        assert_eq!(LIGHTING_RESOURCES.len(), 23);
        for r in LIGHTING_RESOURCES {
            let decl = resource_decl(&r);
            assert!(decl.starts_with(&format!("@group({}) @binding({}) ", r.group, r.binding)));
            let plain = bgl_entry(&r, false);
            let msaa = bgl_entry(&r, true);
            assert_eq!(plain.binding, r.binding);
            assert_eq!(plain.visibility, r.visibility);
            // Only textures observe the MSAA flag.
            let is_texture = matches!(plain.ty, wgpu::BindingType::Texture { .. });
            assert_eq!(
                format!("{:?}", plain.ty) != format!("{:?}", msaa.ty),
                is_texture
            );
        }
        // Binding 0 (camera) is visible to the vertex stage too.
        assert!(
            LIGHTING_RESOURCES[0]
                .visibility
                .contains(wgpu::ShaderStages::VERTEX)
        );
    }
    /// The artifact declares exactly the table's bindings, with the kinds the
    /// pipeline layout builds: a swapped number or a wrong texture type would
    /// pass a textual count and fail here.
    #[test]
    fn lighting_artifact_bindings_match_the_table() {
        use super::LIGHTING_RESOURCES;
        let module = naga::front::wgsl::parse_str(&wgsl_source()).expect("artifact parses");
        let mut declared = Vec::new();
        for (_, global) in module.global_variables.iter() {
            if let Some(binding) = &global.binding {
                declared.push((binding.group, binding.binding, global.ty));
            }
        }
        assert_eq!(declared.len(), LIGHTING_RESOURCES.len());
        for resource in &LIGHTING_RESOURCES {
            let found = declared
                .iter()
                .find(|(group, binding, _)| {
                    *group == resource.group && *binding == resource.binding
                })
                .unwrap_or_else(|| panic!("binding {} is not declared", resource.binding));
            let ty = &module.types[found.2];
            let ok = match resource.kind {
                ResourceKind::Uniform(_) => matches!(ty.inner, naga::TypeInner::Struct { .. }),
                ResourceKind::StorageReadArray(_) => {
                    matches!(ty.inner, naga::TypeInner::Array { .. })
                }
                ResourceKind::TextureFloat => matches!(ty.inner, naga::TypeInner::Image { .. }),
                ResourceKind::TextureUint => matches!(
                    ty.inner,
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Sampled {
                            kind: naga::ScalarKind::Uint,
                            ..
                        },
                        ..
                    }
                ),
                ResourceKind::TextureDepth
                | ResourceKind::TextureDepthArray
                | ResourceKind::TextureDepthCubeArray => matches!(
                    ty.inner,
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Depth { .. },
                        ..
                    }
                ),
                ResourceKind::TextureCube => matches!(
                    ty.inner,
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Sampled { .. },
                        dim: naga::ImageDimension::Cube,
                        ..
                    }
                ),
                ResourceKind::Sampler | ResourceKind::SamplerComparison => {
                    matches!(ty.inner, naga::TypeInner::Sampler { .. })
                }
                ResourceKind::StorageRead(_) | ResourceKind::StorageRw(_) => false,
            };
            assert!(
                ok,
                "binding {} is declared with the wrong type for {:?}",
                resource.binding, resource.kind
            );
        }
    }

    /// The 1x path is the translated artifact: the deferred pass's own module,
    /// declaring the whole binding table and the loop the walk shares with the
    /// forward pass. Asserting the *structure* keeps the file honest without
    /// pinning the translator's variable names.
    #[test]
    fn one_x_source_is_the_translated_artifact() {
        let src = wgsl_source();
        assert!(src.starts_with("struct UniformStruct0_"));
        assert_eq!(src.matches("@binding(").count(), 23);
        for binding in 0..23 {
            assert!(
                src.contains(&format!("@group(0) @binding({binding})")),
                "binding {binding} missing"
            );
        }
        assert!(src.contains("@fragment"));
        assert!(src.contains(
            "fn fs_main_lighting(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>)"
        ));
        // Alpha cut plus the debug views are what the deferred pass adds on top
        // of forward PBR: both must survive the port.
        assert!(src.contains("discard"));
        assert_eq!(wgsl_source_for_samples(1), src);
    }

    /// The 4x path is the translated artifact too: depth, the material index
    /// and the normal bind as multisampled images, and the entry walks the four
    /// samples of one texel before it resolves.
    #[test]
    fn msaa_source_is_the_translated_artifact() {
        let src = wgsl_source_for_samples(4);
        assert_valid_wgsl("lighting_generated_msaa", &src);
        assert_eq!(src.matches("@binding(").count(), 23);
        for binding in 0..23 {
            assert!(
                src.contains(&format!("@group(0) @binding({binding})")),
                "binding {binding} missing"
            );
        }
        assert!(src.contains("fn fs_main_lighting_msaa("), "{src}");
        assert!(src.contains("texture_depth_multisampled_2d"), "{src}");
        assert!(src.contains("texture_multisampled_2d<u32>"), "{src}");
        assert!(src.contains("texture_multisampled_2d<f32>"), "{src}");
        // The three per-sample layers are loaded at the loop's sample index.
        assert!(
            src.matches("textureLoad(").count() >= 4,
            "four per-sample loads: depth, id and the two normal lanes"
        );
        // One sample loop, and the Karis weight and edge test inside it.
        assert_eq!(
            src.matches("loop {").count(),
            2,
            "one sample loop, one light walk"
        );
        // The coverage weight divides by the floored luminance plus the bias.
        assert!(
            src.contains(" + 1f))"),
            "the coverage weight divides by 1 + the floored luminance"
        );
        // The resolved layers still sample, at the pixel centre.
        assert!(src.contains("textureSampleLevel(tex_0_3_"), "{src}");
        let plain = wgsl_source();
        assert!(!plain.contains("multisampled"), "{plain}");
        assert!(!plain.contains("fn fs_main_lighting_msaa"), "{plain}");
        // The 1x entry reads one texel per layer at the fragment centre, so a
        // texel load exists; a sample-indexed one only exists in the resolve.
        assert!(plain.contains("textureLoad("), "{plain}");
    }
}
