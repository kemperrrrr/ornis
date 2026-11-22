//! Shader source assembly.
//!
//! Each public function here assembles a complete WGSL module by splicing
//! math-kernel snippets (generated from Rust via `wgsl_source()`, see
//! [`math`](crate::shaders::math)) into a fixed entry-point skeleton, so CPU tests exercise the
//! exact BRDF code the GPU runs.

pub mod aa_edge_generated;
pub mod aa_taa_generated;
pub mod ao_generated;
pub mod bloom_generated;
pub mod composite_generated;
pub mod fog_generated;
pub mod gbuffer_generated;
pub mod hdr_composite_generated;
pub mod helpers;
pub mod lighting_generated;
pub mod material_textures;
pub mod math;
pub mod outline_generated;
pub mod pbr_generated;
pub mod quad_vertex_generated;

/// Splice a derived [`WgslStruct`](ornis_macros::WgslStruct) declaration into
/// assembled WGSL, terminated the way the former handwritten references
/// spelled it (`};` plus newline).
///
/// The derive emits `struct Foo {\n…\n}\n`; every handwritten reference this
/// replaces used `};`, so the terminator keeps assembled modules
/// shaped like the former handwritten sources (pinned by the
/// `*_matches_legacy_shape` tests).
///
/// Test-only: the last production splicer was the quad vertex stage, and that
/// is a translated artifact now. The cluster-pass oracle still assembles its
/// WGSL from the derives.
#[cfg(test)]
pub(crate) fn wgsl_decl(source: &'static str) -> String {
    format!("{};\n", source.trim_end())
}

/// Address-space / resource class of one pass resource.
///
/// The WGSL type name for buffers comes from the Rust side
/// ([`WgslStruct::WGSL_NAME`](ornis_macros::WgslStruct) for derived layouts,
/// `OPENPBR_WGSL_NAME` for the shared material) — never a retyped string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    /// `var<uniform> name: Ty`.
    Uniform(&'static str),
    /// `var<storage, read> name: Ty`.
    StorageRead(&'static str),
    /// `var<storage, read> name: array<Ty>`.
    StorageReadArray(&'static str),
    /// `var<storage, read_write> name: Ty`.
    StorageRw(&'static str),
    /// `var name: texture_2d<f32>`.
    TextureFloat,
    /// `var name: texture_2d<u32>`.
    TextureUint,
    /// `var name: texture_depth_2d`.
    TextureDepth,
    /// `var name: texture_depth_2d_array` (shadow-map layers).
    TextureDepthArray,
    /// `var name: texture_depth_cube_array` (point-light shadow cubes).
    TextureDepthCubeArray,
    /// `var name: texture_cube<f32>` (prefiltered specular / irradiance).
    TextureCube,
    /// `var name: sampler`.
    Sampler,
    /// `var name: sampler_comparison` (shadow PCF).
    SamplerComparison,
}

impl ResourceKind {
    /// Full WGSL type spelling for a `var` declaration. Array wrappers are
    /// structural (a separate variant), so element type names always come
    /// from the Rust side.
    pub fn wgsl_ty_full(&self) -> String {
        match self {
            Self::Uniform(ty) | Self::StorageRead(ty) | Self::StorageRw(ty) => ty.to_string(),
            Self::StorageReadArray(elem) => format!("array<{elem}>"),
            Self::TextureFloat => "texture_2d<f32>".to_string(),
            Self::TextureUint => "texture_2d<u32>".to_string(),
            Self::TextureDepth => "texture_depth_2d".to_string(),
            Self::TextureDepthArray => "texture_depth_2d_array".to_string(),
            Self::TextureDepthCubeArray => "texture_depth_cube_array".to_string(),
            Self::TextureCube => "texture_cube<f32>".to_string(),
            Self::Sampler => "sampler".to_string(),
            Self::SamplerComparison => "sampler_comparison".to_string(),
        }
    }

    /// Full WGSL type spelling for a `var` declaration at a sample count:
    /// the single-sample spelling, except depth and integer layers in MSAA
    /// mode, which spell the multisampled texture types (see
    /// [`resource_stays_multisampled`]) and load sample 0 with the unchanged
    /// `textureLoad` call text.
    pub fn wgsl_ty_for_samples(&self, sample_count: u32) -> String {
        if sample_count > 1 && resource_stays_multisampled(self) {
            self.wgsl_ty_multisampled()
        } else {
            self.wgsl_ty_full()
        }
    }

    /// Multisampled WGSL type spelling of a texture resource (the AO raw
    /// pass keeps its normal per-sample too, so a pass that selects this
    /// spelling does so through its own resource table).
    pub fn wgsl_ty_multisampled(&self) -> String {
        match self {
            Self::TextureFloat => "texture_multisampled_2d<f32>".to_string(),
            Self::TextureUint => "texture_multisampled_2d<u32>".to_string(),
            Self::TextureDepth => "texture_depth_multisampled_2d".to_string(),
            _ => self.wgsl_ty_full(),
        }
    }

    /// The matching `wgpu` binding type; `multisampled` threads the runtime
    /// MSAA flag through to texture resources.
    pub fn bgl_ty(&self, multisampled: bool) -> wgpu::BindingType {
        match self {
            Self::Uniform(_) => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            Self::StorageRead(_) | Self::StorageReadArray(_) => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            Self::StorageRw(_) => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            Self::TextureFloat => wgpu::BindingType::Texture {
                // Multisampled floats are not filterable (wgpu): the MSAA
                // normal is `textureLoad`ed per sample, never sampled.
                sample_type: wgpu::TextureSampleType::Float {
                    filterable: !multisampled,
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled,
            },
            Self::TextureUint => wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled,
            },
            Self::TextureDepth => wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled,
            },
            Self::TextureDepthArray => wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled,
            },
            Self::TextureDepthCubeArray => wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::CubeArray,
                multisampled,
            },
            Self::TextureCube => wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::Cube,
                multisampled,
            },
            Self::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            Self::SamplerComparison => {
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)
            }
        }
    }
}

/// One pass resource: where it binds, when it is visible, under what WGSL
/// name. A pass's `*_RESOURCES` table is the single source of truth for the
/// `wgpu` layout entry ([`bgl_entry`]) and for the artifact tests that parse
/// the checked-in WGSL back — the numbers cannot drift between shader and
/// pipeline layout.
#[derive(Debug, Clone, Copy)]
pub struct Resource {
    /// `@group` index.
    pub group: u32,
    /// `@binding` index.
    pub binding: u32,
    /// Shader stages that can see this resource.
    pub visibility: wgpu::ShaderStages,
    /// WGSL variable name (`camera`, `materials`, …).
    pub name: &'static str,
    /// Address space / resource class.
    pub kind: ResourceKind,
    /// Minimum buffer binding size in bytes (`Some` only where the
    /// handwritten layout carried an explicit size); ignored for
    /// textures/samplers.
    pub min_size: Option<u64>,
}

/// WGSL-valid rendering of an `f32` value: Rust `{}` prints the shortest
/// round-tripping decimal, plus the `.0` fix for whole values (mirrors
/// the lowering's `float_lit` for literal tokens). Non-finite values
/// have no WGSL spelling — loud in debug, clamped caller's problem.
/// Test-only: the last production `const` block went with the DSL.
#[cfg(test)]
pub(crate) fn f32_lit(v: f32) -> String {
    debug_assert!(v.is_finite(), "non-finite f32 has no WGSL spelling");
    let base = format!("{v}");
    if base.contains('.') || base.contains('e') || base.contains('E') {
        base
    } else {
        format!("{base}.0")
    }
}
/// Whether a resource bound to an MSAA target keeps multisampled storage
/// instead of resolving into a single-sample texture upstream.
///
/// Float color layers resolve (`resolve_target`) into single-sample textures,
/// so they stay plain `texture_2d` in both WGSL and layout. Depth
/// ([`ResourceKind::TextureDepth`]) and the integer material-id layer
/// ([`ResourceKind::TextureUint`], loaded but never filtered) have no resolve
/// target in `wgpu` render passes, so they stay multisampled and shaders load
/// sample 0. Shadow-map arrays/cubes stay single-sample (the shadow pre-pass
/// never runs MSAA).
pub fn resource_stays_multisampled(kind: &ResourceKind) -> bool {
    matches!(kind, ResourceKind::TextureDepth | ResourceKind::TextureUint)
}

/// Runtime bind-group entries from a [`Resource`] table: binding numbers
/// come from the table (single source with the WGSL decls and the BGL),
/// so a table reorder propagates here by construction. Only the
/// name → live resource mapping stays per call site.
///
/// Returns `None` when any row fails to resolve — callers skip the pass
/// (or the probe bind group) instead of panicking on a table/call-site drift.
pub fn bind_group_entries<'a>(
    table: &[Resource],
    resolve: impl Fn(&Resource) -> Option<wgpu::BindingResource<'a>>,
) -> Option<Vec<wgpu::BindGroupEntry<'a>>> {
    table
        .iter()
        .map(|r| {
            Some(wgpu::BindGroupEntry {
                binding: r.binding,
                resource: resolve(r)?,
            })
        })
        .collect()
}

/// `wgpu` bind-group-layout entry for one [`Resource`]; `multisampled`
/// threads the runtime MSAA flag through to texture resources.
pub fn bgl_entry(r: &Resource, multisampled: bool) -> wgpu::BindGroupLayoutEntry {
    let min_binding_size = match (r.min_size, &r.kind) {
        (
            Some(bytes),
            ResourceKind::Uniform(_)
            | ResourceKind::StorageRead(_)
            | ResourceKind::StorageReadArray(_)
            | ResourceKind::StorageRw(_),
        ) => std::num::NonZeroU64::new(bytes),
        _ => None,
    };
    wgpu::BindGroupLayoutEntry {
        binding: r.binding,
        visibility: r.visibility,
        ty: match r.kind.bgl_ty(multisampled) {
            wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset,
                ..
            } => wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            },
            other => other,
        },
        count: None,
    }
}

/// `wgpu` bind-group-layout entry for one [`Resource`] at a sample count:
/// identical to [`bgl_entry`] with `false` at 1x; in MSAA mode only the
/// layers that stay multisampled bind multisampled (see
/// [`resource_stays_multisampled`]) — resolved float layers bind the
/// single-sample resolve textures.
pub fn bgl_entry_for_samples(r: &Resource, sample_count: u32) -> wgpu::BindGroupLayoutEntry {
    bgl_entry(r, sample_count > 1 && resource_stays_multisampled(&r.kind))
}

/// Fullscreen-quad corners shared by the bloom/hdr/lighting passes
/// (triangle strip order), as Rust data.
///
/// Test-only: the corners are a literal of the translated
/// [`quad_vertex_generated`] entry now, and this table is what
/// `the_artifact_spells_the_shared_table` checks that literal against.
#[cfg(test)]
pub(crate) const STANDARD_QUAD: [[f32; 4]; 4] = [
    [-1.0, -1.0, 0.0, 1.0],
    [1.0, -1.0, 0.0, 1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0],
];

/// Fullscreen-quad UVs shared by the bloom/hdr/lighting passes, as Rust data.
///
/// Test-only, for the same reason as [`STANDARD_QUAD`].
#[cfg(test)]
pub(crate) const STANDARD_UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]];

/// Stage of `name` in a translated module, for the per-pass artifact tests.
#[cfg(test)]
pub(crate) fn entry_stage(module: &naga::Module, name: &str) -> Option<naga::ShaderStage> {
    module
        .entry_points
        .iter()
        .find(|entry| entry.name == name)
        .map(|entry| entry.stage)
}

/// `(group, binding)` pairs a module declares, for artifact versus table checks.
#[cfg(test)]
pub(crate) fn declared_bindings(module: &naga::Module) -> Vec<(u32, u32)> {
    module
        .global_variables
        .iter()
        .filter_map(|(_, global)| global.binding)
        .map(|binding| (binding.group, binding.binding))
        .collect()
}

/// The WGSL type name of the shared material declaration (no derive:
/// the CPU side is grouped, so the name lives here and the layout is
/// asserted against the artifact).
pub(crate) const OPENPBR_WGSL_NAME: &str = "OpenPBRMaterial";

// ── COMPOSITE_VERTEX ────────────────────────────────────────────────

/// Assemble the full-screen composite vertex shader (triangle-strip quad).
///
/// Single source of truth — [`hdr_composite_generated::wgsl_vertex_source`]
/// (Rust → WGSL, path 2); shape pinned by `hdr_vertex_entry_matches_legacy_shape`.
pub fn composite_vertex() -> String {
    hdr_composite_generated::wgsl_vertex_source()
}

// ── COMPOSITE_FRAGMENT ──────────────────────────────────────────────

/// Assemble the composite fragment shader: HDR mix + bloom, splicing the ACES tonemap and luminance kernels via `wgsl_source()`.
///
/// Single source of truth — [`hdr_composite_generated::wgsl_source`]
/// (Rust → WGSL, path 2); shape pinned by `hdr_fragment_entry_matches_legacy_shape`.
pub fn composite_fragment() -> String {
    hdr_composite_generated::wgsl_source()
}

// ── BLOOM ───────────────────────────────────────────────────────────
/// Single source of truth — `bloom_generated::wgsl_source()`
/// (Rust → WGSL, path 2); `renderer::create_bloom_pass` and this forwarder
/// use only the generated version.
pub fn bloom_fragment() -> String {
    bloom_generated::wgsl_source()
}

// ── FOG ─────────────────────────────────────────────────────────────
/// Single source of truth — `fog_generated::wgsl_source()`
/// (Rust → WGSL, path 2); `renderer::create_fog_pass` and this forwarder
/// use only the generated version.
pub fn fog_fragment() -> String {
    fog_generated::wgsl_source()
}

// ── AMBIENT OCCLUSION ───────────────────────────────────────────────
/// Assemble the AO raw fragment shader for `method` (SSAO/GTAO kernel
/// over the g-buffer depth and normals).
///
/// Single source of truth — [`ao_generated::wgsl_source_raw`]
/// (Rust → WGSL, path 2 with raw-WGSL bodies); `renderer::create_ao_passes`
/// and this forwarder use only the generated version.
pub fn ao_raw_fragment(method: crate::ao::AoMethod) -> String {
    ao_generated::wgsl_source_raw(method)
}

/// Assemble the AO bilateral-blur fragment shader.
///
/// Single source of truth — [`ao_generated::wgsl_source_blur`]; see
/// [`ao_raw_fragment`].
pub fn ao_blur_fragment() -> String {
    ao_generated::wgsl_source_blur()
}

/// Assemble the fullscreen AO vertex shader (triangle-strip quad).
///
/// Single source of truth — [`ao_generated::wgsl_vertex_source`]; see
/// [`ao_raw_fragment`].
pub fn ao_vertex() -> String {
    ao_generated::wgsl_vertex_source()
}

// ── ANTI-ALIASING ─────────────────────────────────────────────────────

/// Assemble the temporal anti-aliasing fragment shader (the history
/// resolve) at a sample count.
///
/// Single source of truth — [`aa_taa_generated::wgsl_source_for_samples`]
/// (Rust → WGSL, path 2 with a raw-WGSL body); the renderer and this
/// forwarder use only the generated version.
pub fn taa_fragment(sample_count: u32) -> String {
    aa_taa_generated::wgsl_source_for_samples(sample_count)
}

/// Assemble the post-process edge filter fragment shader (FXAA or SMAA).
///
/// Single source of truth — [`aa_edge_generated::wgsl_source`]; see
/// [`taa_fragment`].
pub fn aa_edge_fragment() -> String {
    aa_edge_generated::wgsl_source()
}

// ── OUTLINE ───────────────────────────────────────────────────────────

/// Assemble the editor outline vertex shader (inverted hull + flag test).
///
/// Single source of truth — [`outline_generated::wgsl_vertex_source`].
pub fn outline_vertex() -> String {
    outline_generated::wgsl_vertex_source()
}

/// Assemble the editor outline fragment shader (constant silhouette color).
///
/// Single source of truth — [`outline_generated::wgsl_fragment_source`].
pub fn outline_fragment() -> String {
    outline_generated::wgsl_fragment_source()
}

// ── GBUFFER_VERTEX ────────────────────────────────────────────────────

/// The g-buffer module, both pipelines draw from it.
///
/// Single source of truth — [`gbuffer_generated::wgsl_source`] (Rust → WGSL,
/// path 2). It carries every entry of the pass: the direct and the indirect
/// vertex stage and the fragment stage. A pipeline layout only has to cover the
/// bindings the entry point it names actually uses, so one module serves all
/// three. The forward-PBR pass shares the narrower
/// [`gbuffer_generated::wgsl_vertex_source`] instead.
pub fn gbuffer_vertex() -> String {
    gbuffer_generated::wgsl_source()
}

// ── GBUFFER_FRAGMENT ──────────────────────────────────────────────────

/// Assemble the g-buffer fragment shader, splicing the octahedral normal-encoding kernel.
///
/// Single source of truth — [`gbuffer_generated::wgsl_source`]
/// (Rust → WGSL, path 2); shape pinned by `gbuffer_fragment_entry_matches_legacy_shape`.
pub fn gbuffer_fragment() -> String {
    gbuffer_generated::wgsl_source()
}

// ── LIGHTING_VERTEX ─────────────────────────────────────────────────
/// Full-screen lighting vertex shader — single source of truth
/// `lighting_generated::wgsl_vertex_source()` (path 2).
pub fn lighting_vertex() -> String {
    lighting_generated::wgsl_vertex_source()
}

// ── LIGHTING_FRAGMENT ───────────────────────────────────────────────
/// Deferred lighting fragment — single source of truth
/// `lighting_generated::wgsl_source()` (path 2).
pub fn lighting_fragment() -> String {
    lighting_generated::wgsl_source()
}

// ── PBR_VERTEX ────────────────────────────────────────────────────────

/// Assemble the forward PBR vertex shader.
///
/// Single source of truth — [`pbr_generated::wgsl_vertex_source`]
/// (Rust → WGSL, path 2; shares the g-buffer instance-transform vertex).
pub fn pbr_vertex() -> String {
    pbr_generated::wgsl_vertex_source()
}

// ── PBR_FRAGMENT ──────────────────────────────────────────────────────

/// Assemble the forward PBR fragment shader: full OpenPBR evaluation, splicing all BRDF math kernels via `wgsl_source()`.
///
/// Single source of truth — [`pbr_generated::wgsl_source`]
/// (Rust → WGSL, path 2); shape pinned by `pbr_fragment_entry_matches_legacy_shape`.
pub fn pbr_fragment() -> String {
    pbr_generated::wgsl_source()
}

/// Rust mirror of the WGSL `octahedral_decode` used in the fragment shaders;
/// kept next to [`math::octahedral_encode`] so CPU tests can round-trip it.
pub fn octahedral_decode_rust(p: glam::Vec2) -> glam::Vec3 {
    let mut n = glam::Vec3::new(p.x, p.y, 1.0 - p.x.abs() - p.y.abs());
    let t = (-n.z).max(0.0);
    n.x += if n.x >= 0.0 { -t } else { t };
    n.y += if n.y >= 0.0 { -t } else { t };
    n.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every entry name the renderer asks a pipeline for must exist in the
    /// artifact that pipeline is built from, at the stage the pass binds it.
    /// The name alone is not the contract — a typo'd or renamed entry fails at
    /// pipeline creation, on the GPU, at runtime — so each one is parsed out of
    /// the checked-in WGSL and matched per module.
    ///
    /// A pass may bind two modules (the shared quad vertex stage plus its own
    /// fragment), which is why the table is per module rather than per pass.
    #[test]
    fn entry_points_match_pass_abi() {
        use naga::ShaderStage::{Fragment, Vertex};

        /// One entry a pass asks for, and the stage it binds it at.
        type Entry = (&'static str, naga::ShaderStage);
        /// An artifact, and every entry a pass asks that artifact for.
        type Module = (&'static str, String, &'static [Entry]);

        let modules: [Module; 8] = [
            (
                "hdr_composite",
                hdr_composite_generated::wgsl_source(),
                &[
                    (hdr_composite_generated::VS_ENTRY, Vertex),
                    (hdr_composite_generated::FS_ENTRY, Fragment),
                ],
            ),
            (
                "bloom",
                bloom_generated::wgsl_source(),
                &[
                    (bloom_generated::VS_ENTRY, Vertex),
                    (bloom_generated::FS_ENTRY, Fragment),
                ],
            ),
            (
                "composite",
                composite_generated::wgsl_source(),
                &[
                    (composite_generated::VS_ENTRY, Vertex),
                    (composite_generated::FS_ENTRY, Fragment),
                ],
            ),
            (
                "gbuffer",
                gbuffer_generated::wgsl_source(),
                &[
                    (gbuffer_generated::VS_ENTRY, Vertex),
                    (gbuffer_generated::VS_INDIRECT_ENTRY, Vertex),
                    (gbuffer_generated::FS_ENTRY, Fragment),
                ],
            ),
            (
                "pbr",
                pbr_generated::wgsl_source(),
                &[(pbr_generated::FS_ENTRY, Fragment)],
            ),
            (
                "lighting 1x",
                lighting_generated::wgsl_source_for_samples(1),
                &[(lighting_generated::LIGHTING_1X_ENTRY, Fragment)],
            ),
            (
                "lighting 4x",
                lighting_generated::wgsl_source_for_samples(4),
                &[(lighting_generated::LIGHTING_MSAA_ENTRY, Fragment)],
            ),
            (
                "quad vertex",
                quad_vertex_generated::wgsl_vertex_source(),
                &[(quad_vertex_generated::VS_ENTRY, Vertex)],
            ),
        ];
        let mut checked = 0;
        for (label, src, entries) in &modules {
            let module = naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{label} artifact must parse: {e}"));
            for (name, stage) in *entries {
                assert_eq!(
                    entry_stage(&module, name),
                    Some(*stage),
                    "{label} artifact has no {stage:?} entry `{name}`"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 13, "expected 13 entries, checked {checked}");

        // The names are also the external ABI (pipeline caches and frame
        // snapshots key on them), so the plain-stage ones are pinned literally;
        // the artifact check above would follow any rename.
        assert_eq!(hdr_composite_generated::VS_ENTRY, "vs_main");
        assert_eq!(bloom_generated::VS_ENTRY, "vs_main");
        assert_eq!(composite_generated::VS_ENTRY, "vs_main");
        assert_eq!(gbuffer_generated::VS_ENTRY, "vs_main");
        assert_eq!(gbuffer_generated::VS_INDIRECT_ENTRY, "vs_main_indirect");
        assert_eq!(pbr_generated::FS_ENTRY, "fs_main");
        assert_eq!(lighting_generated::LIGHTING_1X_ENTRY, "fs_main_lighting");

        // And the shared quad stage is deliberately not `vs_main`: a pass that
        // keeps its own `vs_main` depends on `ornis-gpu-quad`, so the two
        // entries would collide in that crate's build. The two passes that
        // draw it therefore ask for the same name.
        assert_eq!(quad_vertex_generated::VS_ENTRY, "vs_main_quad");
        assert_eq!(
            lighting_generated::VS_ENTRY,
            quad_vertex_generated::VS_ENTRY
        );
        assert_eq!(ao_generated::VS_ENTRY, quad_vertex_generated::VS_ENTRY);
        assert_eq!(
            lighting_generated::wgsl_vertex_source(),
            quad_vertex_generated::wgsl_vertex_source(),
            "the lighting pass binds the shared quad module"
        );
    }

    /// Parse and fully validate an assembled WGSL module with naga.
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
    fn assembled_shaders_validate_with_naga() {
        let shaders: [(&str, String); 13] = [
            ("composite_vertex", composite_vertex()),
            ("composite_fragment", composite_fragment()),
            ("outline_vertex", outline_vertex()),
            ("outline_fragment", outline_fragment()),
            ("bloom_fragment", bloom_fragment()),
            ("fog_fragment", fog_fragment()),
            ("gbuffer_vertex", gbuffer_vertex()),
            ("gbuffer_fragment", gbuffer_fragment()),
            ("lighting_vertex", lighting_vertex()),
            ("lighting_fragment", lighting_fragment()),
            ("pbr_vertex", pbr_vertex()),
            ("pbr_fragment", pbr_fragment()),
            (
                "textured_forward_fragment",
                material_textures::wgsl_source_textured(),
            ),
        ];
        for (name, source) in &shaders {
            assert_valid_wgsl(name, source);
        }
    }

    /// The grouped CPU-side `OpenPBRMaterial` must lay out its groups in the
    /// exact order the WGSL declaration lists the `vec4` slots:
    /// base(0) → specular(32) → transmission(64) → subsurface(112) →
    /// fuzz(160) → coat(192) → thin_film(240) → emission(256) →
    /// geometry(288), 320 bytes total. Any reordering shifts every word the
    /// shader reads, so the offsets are the contract.
    #[test]
    fn openpbr_rust_layout_matches_wgsl_order() {
        use ornis_core::material::OpenPBRMaterial;
        assert_eq!(std::mem::size_of::<OpenPBRMaterial>(), 320);
        assert_eq!(std::mem::offset_of!(OpenPBRMaterial, base), 0);
        assert_eq!(std::mem::offset_of!(OpenPBRMaterial, specular), 32);
        assert_eq!(std::mem::offset_of!(OpenPBRMaterial, transmission), 64);
        assert_eq!(std::mem::offset_of!(OpenPBRMaterial, subsurface), 112);
        assert_eq!(std::mem::offset_of!(OpenPBRMaterial, fuzz), 160);
        assert_eq!(std::mem::offset_of!(OpenPBRMaterial, coat), 192);
        assert_eq!(std::mem::offset_of!(OpenPBRMaterial, thin_film), 240);
        assert_eq!(std::mem::offset_of!(OpenPBRMaterial, emission), 256);
        assert_eq!(std::mem::offset_of!(OpenPBRMaterial, geometry), 288);
        // 20 `vec4` slots of 16 bytes each fill the 320-byte record.
        // The GPU side addresses the record by word index, so its stride is
        // the CPU record's size: a group added on one side only would make the
        // shader read the neighbouring entity's bytes.
        assert_eq!(
            ornis_gpu_pbr::layout::MATERIAL_STRIDE as usize * 4,
            std::mem::size_of::<OpenPBRMaterial>()
        );
        let words = std::mem::size_of::<OpenPBRMaterial>() / 4;
        let src = super::lighting_generated::wgsl_source_for_samples(1);
        let mut reads = 0;
        let mut highest = 0u32;
        for read in src.match_indices("storage_0_2_[") {
            let call = &src[read.0..];
            let call = &call[..call.find(']').unwrap_or(call.len())];
            // `base | lane` (folded word) or `base + lane` (full word).
            for (at, _) in call.match_indices('|').chain(call.match_indices('+')) {
                let lane: String = call[at + 1..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(lane) = lane.parse::<u32>() {
                    highest = highest.max(lane);
                }
            }
            reads += 1;
        }
        assert!(reads > 20, "expected the material reads, found {reads}");
        assert!(
            highest as usize + 1 <= words,
            "the lighting kernel reads material word {highest} of a {words}-word record"
        );
        for (group, offset) in [
            (std::mem::offset_of!(OpenPBRMaterial, base), 0),
            (std::mem::offset_of!(OpenPBRMaterial, specular), 32),
            (std::mem::offset_of!(OpenPBRMaterial, transmission), 64),
            (std::mem::offset_of!(OpenPBRMaterial, subsurface), 112),
            (std::mem::offset_of!(OpenPBRMaterial, fuzz), 160),
            (std::mem::offset_of!(OpenPBRMaterial, coat), 192),
            (std::mem::offset_of!(OpenPBRMaterial, thin_film), 240),
            (std::mem::offset_of!(OpenPBRMaterial, emission), 256),
            (std::mem::offset_of!(OpenPBRMaterial, geometry), 288),
        ] {
            assert_eq!(group, offset);
            assert_eq!(group % 16, 0, "group must start on a slot boundary");
        }
    }

    /// MSAA declaration policy: at 1x every spelling is the legacy
    /// single-sample one; at 4x only depth and integer layers go
    /// multisampled (resolve has no depth/integer target), while float
    /// colors, shadow arrays/cubes and samplers are untouched.
    #[test]
    fn msaa_decl_policy_touches_only_depth_and_uint() {
        use super::ResourceKind::*;
        // 1x is byte-identical to the legacy spelling.
        for kind in [
            TextureFloat,
            TextureUint,
            TextureDepth,
            TextureDepthArray,
            TextureDepthCubeArray,
            TextureCube,
        ] {
            assert_eq!(kind.wgsl_ty_for_samples(1), kind.wgsl_ty_full());
        }
        // 4x multisampled spellings for the staying layers only.
        assert_eq!(
            TextureUint.wgsl_ty_for_samples(4),
            "texture_multisampled_2d<u32>"
        );
        assert_eq!(
            TextureDepth.wgsl_ty_for_samples(4),
            "texture_depth_multisampled_2d"
        );
        assert_eq!(TextureFloat.wgsl_ty_for_samples(4), "texture_2d<f32>");
        assert_eq!(
            TextureDepthArray.wgsl_ty_for_samples(4),
            "texture_depth_2d_array"
        );
        assert_eq!(
            TextureDepthCubeArray.wgsl_ty_for_samples(4),
            "texture_depth_cube_array"
        );
        assert_eq!(TextureCube.wgsl_ty_for_samples(4), "texture_cube<f32>");
        // The per-resource predicate matches the spelling table.
        assert!(resource_stays_multisampled(&TextureDepth));
        assert!(resource_stays_multisampled(&TextureUint));
        assert!(!resource_stays_multisampled(&TextureFloat));
        assert!(!resource_stays_multisampled(&TextureDepthArray));
        assert!(!resource_stays_multisampled(&TextureDepthCubeArray));
        assert!(!resource_stays_multisampled(&TextureCube));
        assert!(!resource_stays_multisampled(&Sampler));
    }

    /// Layout entries follow the same policy: at 1x every entry equals the
    /// legacy `bgl_entry(r, false)`; at 4x only depth/uint bindings flip to
    /// multisampled.
    #[test]
    fn msaa_bgl_policy_matches_decl_policy() {
        use crate::shaders::lighting_generated::LIGHTING_RESOURCES;
        for r in LIGHTING_RESOURCES {
            let legacy = bgl_entry(&r, false);
            assert_eq!(
                format!("{:?}", bgl_entry_for_samples(&r, 1).ty),
                format!("{:?}", legacy.ty)
            );
            let msaa = bgl_entry_for_samples(&r, 4);
            let expect_multi = resource_stays_multisampled(&r.kind);
            let is_multi = matches!(
                msaa.ty,
                wgpu::BindingType::Texture {
                    multisampled: true,
                    ..
                }
            );
            assert_eq!(is_multi, expect_multi, "binding {}", r.binding);
        }
    }
}
