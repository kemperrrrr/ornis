//! Sampled glTF material textures feeding the forward-PBR evaluator.
//!
//! The upload side ([`crate::textures`]) owns images, formats and CPU
//! mirrors; this module owns the shader side: a dedicated forward-fragment
//! entry (`fs_main_textured`, translated from `ornis-gpu-pbr`) that samples
//! the bound textures with the material UV (`input.uv`) and folds the texels
//! into the scalar factors before the shared layer evaluators run, plus the
//! CPU twins of those three combines so the arithmetic is pinned without an
//! adapter.
//!
//! The untextured forward entry ([`crate::shaders::pbr_generated::FS_ENTRY`])
//! is untouched: textured materials are a new path (new resource table,
//! new pipeline built by
//! [`Renderer3D::ensure_textured_forward`](crate::renderer::Renderer3D::ensure_textured_forward)),
//! so untextured renders stay pixel-identical by construction. The
//! deferred-lighting pass takes no bindings here on purpose: the g-buffer
//! carries no material UV (only screen UV plus ids), so sampling
//! per-material textures there would need a new MRT and break the
//! pixel-parity gate — the same combines are ready for it once UV storage
//! lands.
//!
//! sRGB/linear contract (mirrors [`crate::textures`]): color roles arrive
//! as `Rgba8UnormSrgb`, so the hardware already decoded the sample to
//! linear light and the multiply stays in linear; the data role arrives
//! as `Rgba8Unorm` (green holds roughness, blue holds metallic) and scales
//! linearly. The scalar `is_metallic` switch never moves — the texture
//! only modulates the metallic factor. Unbound slots sample a 1x1 white
//! fallback (`x * 1.0 == x`, IEEE-exact), which is the legacy no-op.

use super::{OPENPBR_WGSL_NAME, Resource, ResourceKind};
use crate::renderer::{
    CameraUniform, ClusterParams, ClusterSpan, GpuLight, LightingUniform, PerObjectGpu,
};

/// Multiplies a scalar base color by a linear map texel (CPU mirror).
///
/// Pure software twin of the WGSL `apply_base_color_map` the translated
/// module carries, over the
/// [`crate::textures::sample_albedo_linear`] output, so tests pin the
/// combine without an adapter.
pub fn apply_base_color_map_cpu(base_color: [f32; 3], map_rgb: [f32; 3]) -> [f32; 3] {
    [
        base_color[0] * map_rgb[0],
        base_color[1] * map_rgb[1],
        base_color[2] * map_rgb[2],
    ]
}

/// Scales roughness/metallic by a linear `(G, B)` texel (CPU mirror).
///
/// Pure software twin of the WGSL `apply_metallic_roughness_map`, over the
/// [`crate::textures::sample_metallic_roughness`] output.
pub fn apply_metallic_roughness_map_cpu(
    roughness: f32,
    metallic: f32,
    map_gb: [f32; 2],
) -> [f32; 2] {
    [roughness * map_gb[0], metallic * map_gb[1]]
}

/// Multiplies an emission color by a linear map texel (CPU mirror).
///
/// Pure software twin of the WGSL `apply_emissive_map`.
pub fn apply_emissive_map_cpu(emission_color: [f32; 3], map_rgb: [f32; 3]) -> [f32; 3] {
    [
        emission_color[0] * map_rgb[0],
        emission_color[1] * map_rgb[1],
        emission_color[2] * map_rgb[2],
    ]
}

/// Resource layout of the textured-forward pass: the forward-PBR rows
/// (same numbers, same types) plus one texture row per sampled role, the
/// shared sampler and the clustered-lighting set (#130). Type names come
/// from the Rust side, so shader and bind-group layout cannot drift.
pub const TEXTURED_PBR_RESOURCES: [Resource; 20] = [
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
    Resource {
        group: 0,
        binding: 16,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "base_color_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 17,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "metallic_roughness_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 18,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "emissive_tex",
        kind: ResourceKind::TextureFloat,
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 19,
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "material_sampler",
        kind: ResourceKind::Sampler,
        min_size: None,
    },
];

/// Textured forward-PBR fragment shader, translated from `ornis-gpu-pbr`.
///
/// `xtask build-shaders --package ornis-gpu-pbr --entry fs_main_textured`
/// writes `pbr_textured_generated.wgsl`; the entry is the same light walk as
/// the untextured one with three role samples folded into the material
/// factors first. The vertex stage is the g-buffer's (same instance
/// transform).
pub fn wgsl_source_textured() -> String {
    include_str!("pbr_textured_generated.wgsl").to_string()
}

/// Fragment entry point of the translated textured module.
pub const TEXTURED_FS_ENTRY: &str = "fs_main_textured";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textures::{
        CpuImage, TextureRole, sample_albedo_linear, sample_metallic_roughness,
        texture_format_for_role,
    };

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

    /// The three combines are pure multiplies (white is the exact no-op),
    /// and the translated module inlines them per role binding.
    #[test]
    fn sampled_factors_compose_as_plain_multiplies() {
        let module = wgsl_source_textured();
        // One sampler for every role: binding 19, the table's only sampler
        // the material textures may use.
        for role in ["tex_0_16_", "tex_0_17_", "tex_0_18_"] {
            assert!(
                module.contains(&format!(
                    "textureSample({role}, sampler_0_19_, vec2<f32>(in2_.x, in2_.y))"
                )),
                "`{role}` is not sampled with the material sampler at the material UV"
            );
        }
        // DSL spellings must not leak into the committed artifact.
        for leaked in ["Vec3", "glam", "f32_lit"] {
            assert!(!module.contains(leaked), "`{leaked}` leaked into the WGSL");
        }
        assert_eq!(
            super::apply_base_color_map_cpu([1.0, 1.0, 1.0], [0.5, 0.25, 2.0]),
            [0.5, 0.25, 2.0]
        );
        assert_eq!(
            super::apply_metallic_roughness_map_cpu(0.5, 0.5, [0.5, 0.25]),
            [0.25, 0.125]
        );
        assert_eq!(
            super::apply_emissive_map_cpu([1.0, 2.0, 4.0], [0.5, 0.5, 0.5]),
            [0.5, 1.0, 2.0]
        );
    }

    #[test]
    fn textured_assembly_validates_with_naga() {
        assert_valid_wgsl(
            "textured_forward_vertex",
            &super::super::pbr_generated::wgsl_vertex_source(),
        );
        assert_valid_wgsl("textured_forward_fragment", &wgsl_source_textured());
    }

    /// The translated entry samples every role, then runs the one shared
    /// light walk. SSA locals are not readable, so the shape is pinned
    /// structurally; the pixel gate lives in `pbr_parity`.
    #[test]
    fn textured_entry_samples_all_roles_before_lighting() {
        let module = wgsl_source_textured();
        let start = module
            .find(&format!("fn {TEXTURED_FS_ENTRY}("))
            .expect("entry");
        assert!(
            module[..start].trim_end().ends_with("@fragment"),
            "entry lost its @fragment attribute"
        );
        let entry = &module[start..];
        assert!(entry.contains("-> @location(0) vec4<f32>"));
        // One sample per role, all through the shared material sampler, and
        // all before the light loop starts.
        let loop_at = entry.find("loop {").expect("light walk");
        for role in ["tex_0_16_", "tex_0_17_", "tex_0_18_"] {
            let at = entry
                .find(role)
                .unwrap_or_else(|| panic!("role `{role}` is never sampled"));
            assert!(at < loop_at, "`{role}` is sampled after the light walk");
        }
        assert_eq!(
            entry.matches("loop {").count(),
            1,
            "one light walk expected"
        );
    }

    /// Every table row's declaration appears in the assembled fragment, and
    /// every row maps to a layout entry: shader and pipeline agree. The
    /// first seven rows mirror the legacy table one-to-one.
    #[test]
    fn textured_resources_cover_stages_and_layout() {
        use super::super::{bgl_entry, declared_bindings};
        use crate::shaders::pbr_generated::{PBR_RESOURCES, wgsl_vertex_source};
        // Vertex-only rows (per_objects) live in the shared vertex stage,
        // like the legacy table — cover both assemblies together. The vertex
        // stage is the translated g-buffer kernel, so its bindings come from
        // the parsed module rather than from a `resource_decl` line.
        let vertex = naga::front::wgsl::parse_str(&wgsl_vertex_source()).expect("vertex");
        let fragment = naga::front::wgsl::parse_str(&wgsl_source_textured()).expect("fragment");
        let mut bindings = declared_bindings(&vertex);
        bindings.extend(declared_bindings(&fragment));
        assert_eq!(TEXTURED_PBR_RESOURCES.len(), 20);
        for (i, r) in TEXTURED_PBR_RESOURCES.iter().enumerate() {
            assert!(
                bindings.contains(&(r.group, r.binding)),
                "missing {} at {}:{}",
                r.name,
                r.group,
                r.binding
            );
            let e = bgl_entry(r, false);
            assert_eq!((e.binding, e.visibility), (r.binding, r.visibility));
            // The original seven rows stay aligned with the untextured
            // table. IBL is appended after the material textures, so it
            // does not share those binding numbers.
            if i < 7 {
                assert_eq!(
                    (r.binding, r.name),
                    (PBR_RESOURCES[i].binding, PBR_RESOURCES[i].name),
                    "legacy rows must not drift"
                );
            }
        }
    }

    #[test]
    fn sampled_factors_compose_cpu_mirrors_without_adapter() {
        // Adapter-free sample→combine round trip through the same CPU
        // mirrors the GPU path uses: a 2x2 RGBA8 image (the `LoadedImage`
        // contract) samples albedo through the sRGB decode and
        // metallic-roughness through the linear (G, B) pick, then folds
        // into scalar factors exactly like the shader prelude.
        let image = CpuImage::from_rgba8(
            2,
            2,
            vec![
                255, 0, 0, 255, //
                0, 255, 0, 255, //
                0, 0, 255, 255, //
                18, 128, 64, 128,
            ],
        )
        .expect("valid 2x2");
        // Albedo role: red texel decodes to linear red, then scales the
        // scalar base color channel-wise.
        let red = sample_albedo_linear(image.texel(0, 0).expect("red"));
        assert_eq!(red, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            apply_base_color_map_cpu([0.5, 0.25, 1.0], [red[0], red[1], red[2]]),
            [0.5, 0.0, 0.0]
        );
        // Data role: (G, B) = (128/255 roughness, 64/255 metallic), linear.
        let (roughness, metallic) = sample_metallic_roughness(image.texel(1, 1).expect("data"));
        assert!((roughness - 128.0 / 255.0).abs() < 1e-6);
        assert!((metallic - 64.0 / 255.0).abs() < 1e-6);
        let scaled = apply_metallic_roughness_map_cpu(0.8, 0.5, [roughness, metallic]);
        assert!((scaled[0] - 0.8 * 128.0 / 255.0).abs() < 1e-6);
        assert!((scaled[1] - 0.5 * 64.0 / 255.0).abs() < 1e-6);
        // Emissive role: same sRGB decode as albedo, then scales emission.
        let green = sample_albedo_linear(image.texel(1, 0).expect("green"));
        assert_eq!(
            apply_emissive_map_cpu([1.0, 0.5, 0.25], [green[0], green[1], green[2]]),
            [0.0, 0.5, 0.0]
        );
        // White fallbacks are the exact no-op for every role (the legacy
        // path binds them, so untextured materials render unchanged).
        assert_eq!(
            apply_base_color_map_cpu([0.2, 0.4, 0.6], [1.0, 1.0, 1.0]),
            [0.2, 0.4, 0.6]
        );
        assert_eq!(
            apply_metallic_roughness_map_cpu(0.3, 0.9, [1.0, 1.0]),
            [0.3, 0.9]
        );
        assert_eq!(
            apply_emissive_map_cpu([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]),
            [0.1, 0.2, 0.3]
        );
        // The formats behind the samples stay pinned: color roles decode
        // in hardware, the data role never does.
        assert_eq!(
            texture_format_for_role(TextureRole::BaseColor),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            texture_format_for_role(TextureRole::Emissive),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            texture_format_for_role(TextureRole::MetallicRoughness),
            wgpu::TextureFormat::Rgba8Unorm
        );
    }
}
