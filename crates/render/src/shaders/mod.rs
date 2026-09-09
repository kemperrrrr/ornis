//! Shader source assembly.
//!
//! Each public function here assembles a complete WGSL module by splicing
//! math-kernel snippets (generated from Rust via `wgsl_source()`, see
//! [`math`]) into a fixed entry-point skeleton, so CPU tests exercise the
//! exact BRDF code the GPU runs.

pub mod bloom_generated;
pub mod composite_generated;
pub mod gbuffer_generated;
pub mod hdr_composite_generated;
pub mod helpers;
pub mod interface;
pub mod lighting_generated;
pub mod math;
pub mod naga_ir;
pub mod pbr_generated;

/// Splice a derived [`WgslStruct`](ornis_macros::WgslStruct) declaration into
/// assembled WGSL, terminated the way the former handwritten references
/// spelled it (`};` plus newline).
///
/// The derive emits `struct Foo {\n…\n}\n`; every handwritten reference this
/// replaces used `};`, so the terminator keeps assembled modules
/// shaped like the former handwritten sources (pinned by the
/// `*_matches_legacy_shape` tests).
pub(crate) fn wgsl_decl(source: &'static str) -> String {
    format!("{};\n", source.trim_end())
}

/// Address-space / resource class of one pass resource.
///
/// The WGSL type name for buffers comes from the Rust side
/// ([`WgslStruct::WGSL_NAME`](ornis_macros::WgslStruct) for derived layouts,
/// [`OPENPBR_WGSL_NAME`] for the shared material) — never a retyped string.
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
    /// `var name: sampler`.
    Sampler,
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
            Self::Sampler => "sampler".to_string(),
        }
    }

    /// `var<…>` address-space prefix, or `var` for textures/samplers.
    fn wgsl_var(&self) -> &'static str {
        match self {
            Self::Uniform(_) => "var<uniform>",
            Self::StorageRead(_) | Self::StorageReadArray(_) => "var<storage, read>",
            Self::StorageRw(_) => "var<storage, read_write>",
            Self::TextureFloat | Self::TextureUint | Self::TextureDepth | Self::Sampler => "var",
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
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
            Self::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        }
    }
}

/// One pass resource: where it binds, when it is visible, under what WGSL
/// name. A pass's `*_RESOURCES` table is the single source of truth for
/// both the WGSL declaration ([`resource_decl`]) and the `wgpu` layout
/// entry ([`bgl_entry`]) — the numbers cannot drift between shader and
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

/// WGSL declaration lines for a subset of a [`Resource`] table (one
/// shader stage): `bindings` lists the `@binding` indices that stage
/// declares, in order.
pub fn resource_decls(table: &[Resource], bindings: &[u32]) -> String {
    let mut out = String::new();
    for b in bindings {
        let r = table
            .iter()
            .find(|r| r.binding == *b)
            .unwrap_or_else(|| panic!("resource table lacks binding {b}"));
        out.push_str(&resource_decl(r));
    }
    out
}

/// WGSL declaration line for one [`Resource`].
pub fn resource_decl(r: &Resource) -> String {
    format!(
        "@group({}) @binding({}) {} {}: {};\n",
        r.group,
        r.binding,
        r.kind.wgsl_var(),
        r.name,
        r.kind.wgsl_ty_full()
    )
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
        ) => Some(std::num::NonZeroU64::new(bytes).expect("resource min_size must be nonzero")),
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

/// Fullscreen-quad corners shared by the bloom/hdr/lighting passes
/// (triangle strip order), as Rust data.
pub(crate) const STANDARD_QUAD: [[f32; 4]; 4] = [
    [-1.0, -1.0, 0.0, 1.0],
    [1.0, -1.0, 0.0, 1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0],
];

/// Fullscreen-quad UVs shared by the bloom/hdr/lighting passes, as Rust data.
pub(crate) const STANDARD_UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]];

/// DSL-only GPU handle markers: nominal field types for stage-entry context
/// bundles (`ctx: LightingContext`). The Rust functions are never compiled,
/// so these are never instantiated — they exist so context structs are real,
/// name-resolving Rust items whose fields document the WGSL global each maps
/// to (field name === global name, enforced by `stage_globals_declared`).
/// No constructors, no methods, no runtime footprint.
pub struct Texture2d;
/// 2D unsigned-int texture handle marker; see [`Texture2d`].
pub struct Texture2dUint;
/// Depth-texture handle marker; see [`Texture2d`].
pub struct DepthTexture;
/// Sampler handle marker; see [`Texture2d`].
pub struct Sampler;

/// Builtin index newtypes: a bare `u32` cannot say *which* index it is
/// (`vertex_index`, `instance_index`, `@location(0)` and a plain local are
/// all `u32`), so the meaning travels in the type. `#[stage]` maps
/// `VertexIndex` → `@builtin(vertex_index) …: u32` (and likewise for
/// `InstanceIndex`) with no attribute — one source of truth. Newtypes live
/// in signature position only; bodies use the bare ident (no `.0` tax).
pub struct VertexIndex(pub u32);
/// Per-instance index newtype; see [`VertexIndex`].
pub struct InstanceIndex(pub u32);

/// Context-bundle wrapper: `ctx: Context<LightingContext>` marks the
/// parameter as a resource bundle (excluded from the WGSL signature;
/// `ctx.field` lowers to the global `field`), while a bare struct param
/// (`input: VertexInput`) stays a real WGSL function parameter. The two
/// spellings are syntactically identical otherwise, so the wrapper — not
/// an attribute — carries the one disambiguating bit, and signatures stay
/// attribute-free pure Rust. Never instantiated; see the `Texture2d`
/// markers.
pub struct Context<T>(T);

/// Located-value return: `-> Location<0, glam::Vec4>` lowers to
/// `-> @location(0) vec4<f32>`. A bare Rust return type cannot spell a
/// WGSL location attribute, and the old `returns = "@location(0) …"`
/// string embedded WGSL syntax in the signature — the wrapper keeps both
/// the location number and a real, rustc-checked inner type. Never
/// instantiated; combining it with an explicit `returns = "…"` is an
/// error (two sources of truth).
pub struct Location<const N: usize, T>(T, core::marker::PhantomData<[u8; N]>);

/// Quad constants as a stage-entry context bundle: five vertex entries share
/// it (`ctx: QuadContext`, `ctx.quad[idx]`). Field names match the WGSL
/// `const` names exactly — the `#[stage]` `context` lowering strips the
/// `ctx.` prefix, no renaming involved.
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct QuadContext {
    pub quad: [[f32; 4]; 4],
    pub uvs: [[f32; 2]; 4],
}

/// Shared `OpenPBRMaterial` WGSL declaration (20 `vec4` slots), used by the
/// g-buffer, lighting and forward-PBR skeletons.
///
/// Generated from [`OPENPBR_SLOTS`]: the CPU-side
/// [`OpenPBRMaterial`](ornis_core::material::OpenPBRMaterial) is grouped
/// (`BaseGroup`, `SpecularGroup`, …), not flat, so there is no 1:1 field
/// mirror to derive from — the slot-name list is the single source of truth
/// for the WGSL side. The `openpbr_rust_layout_matches_wgsl_order` test
/// below pins the group offsets against this slot order instead.
/// Leading/trailing newlines are part of the legacy assembly contract.
pub(crate) fn openpbr_material_decl() -> String {
    let mut out = String::from("\nstruct OpenPBRMaterial {\n");
    for slot in OPENPBR_SLOTS {
        out.push_str(&format!("    {slot}: vec4<f32>,\n"));
    }
    out.push_str("};\n");
    out
}

/// The WGSL type name of the shared material declaration (no derive:
/// the CPU side is grouped, so the name lives next to the slot list).
pub(crate) const OPENPBR_WGSL_NAME: &str = "OpenPBRMaterial";

/// The 20 `vec4` slots of `OpenPBRMaterial`, in declaration order.
pub(crate) const OPENPBR_SLOTS: [&str; 20] = [
    "base_params",
    "base_color",
    "specular_params",
    "specular_color",
    "transmission_params",
    "transmission_color",
    "transmission_scatter",
    "subsurface_params",
    "subsurface_color",
    "subsurface_radius_scale_gb",
    "fuzz_params",
    "fuzz_color",
    "coat_params",
    "coat_color",
    "coat_ior",
    "thin_film_params",
    "emission_params",
    "emission_color",
    "geometry_params",
    "geometry_params2",
];

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

// ── GBUFFER_VERTEX ────────────────────────────────────────────────────

/// Assemble the gbuffer vertex shader (instance transforms + world position).
///
/// Single source of truth — [`gbuffer_generated::wgsl_vertex_source`]
/// (Rust → WGSL, path 2); shape pinned by `gbuffer_vertex_entry_matches_legacy_shape`.
pub fn gbuffer_vertex() -> String {
    gbuffer_generated::wgsl_vertex_source()
}

// ── GBUFFER_FRAGMENT ──────────────────────────────────────────────────

/// Assemble the 5-MRT gbuffer fragment shader, splicing the octahedral normal-encoding kernel.
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
    let offset = if n.x >= 0.0 {
        glam::Vec2::new(n.y, n.x)
    } else {
        -glam::Vec2::new(n.y, n.x)
    } * t;
    n.x += offset.x;
    n.y += offset.y;
    n.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extract `(Type, [field names])` markers from `Type(args) /* f1, f2 */`
    /// constructor comments emitted by struct-literal translation.
    fn extract_ctors(src: &str) -> Vec<(String, Vec<String>)> {
        let mut out = Vec::new();
        let mut rest = src;
        while let Some(cstart) = rest.find("/* ") {
            let before = rest[..cstart].trim_end();
            let Some(cend) = rest[cstart..].find("*/") else {
                break;
            };
            let names: Vec<String> = rest[cstart + 3..cstart + cend]
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
            // Walk back balanced parens from the ')' ending `before` to the
            // constructor's opening paren; the identifier before it is the type.
            if before.ends_with(')') {
                let mut depth = 0u32;
                let mut open = None;
                for (i, ch) in before.char_indices().rev() {
                    match ch {
                        ')' => depth += 1,
                        '(' => {
                            depth -= 1;
                            if depth == 0 {
                                open = Some(i);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if let Some(i) = open {
                    let ident: String = before[..i]
                        .chars()
                        .rev()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect();
                    if !ident.is_empty() {
                        out.push((ident, names));
                    }
                }
            }
            rest = &rest[cstart + cend + 2..];
        }
        out
    }

    /// Stage-entry struct literals are positional constructors: the emitted
    /// `/* names */` comment must list the mirror's fields in `WGSL_FIELDS`
    /// order. A wrong order would still parse and could pass pixels by
    /// accident — this test pins the contract instead.
    #[test]
    fn interface_field_order() {
        use super::interface::{
            BloomVertexOut, GbufferVertexOutput, HdrFragmentOut, HdrVertexOutput, UiCompositeOut,
        };
        let mirrors: &[(&str, &[&str])] = &[
            (HdrVertexOutput::WGSL_NAME, HdrVertexOutput::WGSL_FIELDS),
            (HdrFragmentOut::WGSL_NAME, HdrFragmentOut::WGSL_FIELDS),
            (BloomVertexOut::WGSL_NAME, BloomVertexOut::WGSL_FIELDS),
            (UiCompositeOut::WGSL_NAME, UiCompositeOut::WGSL_FIELDS),
            (
                GbufferVertexOutput::WGSL_NAME,
                GbufferVertexOutput::WGSL_FIELDS,
            ),
        ];
        let entries: &[String] = &[
            hdr_composite_generated::composite_vertex_entry::wgsl_source().to_string(),
            hdr_composite_generated::hdr_fragment_vs_entry::wgsl_source().to_string(),
            bloom_generated::bloom_vertex_entry::wgsl_source().to_string(),
            lighting_generated::lighting_vertex_entry::wgsl_source().to_string(),
            composite_generated::composite_vs_entry::wgsl_source().to_string(),
            gbuffer_generated::gbuffer_vs_entry::wgsl_source().to_string(),
        ];
        let mut checked = 0;
        for src in entries {
            for (ty, names) in extract_ctors(src) {
                let fields = mirrors
                    .iter()
                    .find(|(n, _)| *n == ty)
                    .unwrap_or_else(|| panic!("constructor {ty} has no registered mirror"));
                assert_eq!(
                    names, fields.1,
                    "struct literal {ty} lists fields out of mirror order"
                );
                checked += 1;
            }
        }
        // Four vertex entries construct varyings today; zero hits would mean
        // the comments vanished and the test passes vacuously.
        assert!(checked >= 4, "expected struct literals, found {checked}");
    }

    /// Every context-bundle / `#[wgsl(global)]` name an entry reads must
    /// exist in the assembled shader it splices into. Bundle structs (via
    /// `#[derive(ShaderContext)]`) are the single source: adding a field
    /// without wiring the global fails here. Catches a typo'd name (the
    /// entry still parses, naga only fails if the name is *used*
    /// undeclared — an unused declaration would pass silently).
    #[test]
    fn stage_globals_declared() {
        let hdr_vertex = hdr_composite_generated::wgsl_vertex_source();
        let hdr_fragment = hdr_composite_generated::wgsl_source();
        let bloom = bloom_generated::wgsl_source();
        let lighting = lighting_generated::wgsl_source();
        let lighting_vertex = lighting_generated::wgsl_vertex_source();
        let composite = composite_generated::wgsl_source();
        let gbuffer_vertex = gbuffer_generated::wgsl_vertex_source();
        let gbuffer_fragment = gbuffer_generated::wgsl_source();
        let pbr = pbr_generated::wgsl_source();
        // (bundle/entry globals, assemblies containing them). QuadContext is
        // shared by five vertex entries, so it pins against all five
        // assemblies carrying the quad constants.
        let pairs: &[(&[&str], Vec<String>)] = &[
            (
                QuadContext::GLOBALS,
                vec![
                    hdr_vertex.clone(),
                    hdr_fragment.clone(),
                    bloom.clone(),
                    lighting_vertex.clone(),
                    composite.clone(),
                ],
            ),
            (
                hdr_composite_generated::HdrContext::GLOBALS,
                vec![hdr_fragment],
            ),
            (bloom_generated::BloomContext::GLOBALS, vec![bloom]),
            (lighting_generated::LightingContext::GLOBALS, vec![lighting]),
            (
                composite_generated::CompositeContext::GLOBALS,
                vec![composite],
            ),
            (
                gbuffer_generated::GbufferVertexContext::GLOBALS,
                vec![gbuffer_vertex],
            ),
            (
                gbuffer_generated::gbuffer_fs_entry::globals(),
                vec![gbuffer_fragment],
            ),
            (pbr_generated::PbrContext::GLOBALS, vec![pbr]),
        ];
        let mut checked = 0;
        for (globals, assemblies) in pairs {
            assert!(!globals.is_empty(), "bundle must declare its globals");
            for global in *globals {
                assert!(
                    assemblies.iter().any(|src| src.contains(global)),
                    "global `{global}` declared by a bundle is missing from its assembled shader"
                );
                checked += 1;
            }
        }
        // Seven bundles plus one lone `global` param declare 30 names today;
        // fewer means a declaration was dropped and the test passes vacuously.
        assert!(
            checked >= 30,
            "expected global declarations, found {checked}"
        );
    }

    /// Entry-point namespace is closed: every generated `entry_point()`
    /// must be one of the four pipeline-convention names. The renderer and
    /// `composite.rs` request entries through these same accessors (no
    /// literals), so a rename propagates by construction — but the names
    /// themselves are external (pipeline caches, debug labels, snapshots),
    /// hence pinned here as a set.
    #[test]
    fn entry_namespace_is_closed() {
        use std::collections::HashSet;
        let entries = [
            hdr_composite_generated::composite_vertex_entry::entry_point(),
            hdr_composite_generated::hdr_fragment_vs_entry::entry_point(),
            hdr_composite_generated::hdr_fragment_entry::entry_point(),
            bloom_generated::bloom_vertex_entry::entry_point(),
            bloom_generated::bloom_fragment_entry::entry_point(),
            lighting_generated::lighting_vertex_entry::entry_point(),
            lighting_generated::lighting_fragment_entry::entry_point(),
            composite_generated::composite_vs_entry::entry_point(),
            composite_generated::composite_fs_entry::entry_point(),
            gbuffer_generated::gbuffer_vs_entry::entry_point(),
            gbuffer_generated::gbuffer_fs_entry::entry_point(),
            pbr_generated::pbr_fragment_entry::entry_point(),
        ];
        // Twelve entries today; fewer means one escaped the pin.
        assert_eq!(entries.len(), 12);
        let set: HashSet<&str> = entries.into_iter().collect();
        assert_eq!(set, HashSet::from(["vs_main", "fs_main", "vs", "fs"]));
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
        let shaders: [(&str, String); 9] = [
            ("composite_vertex", composite_vertex()),
            ("composite_fragment", composite_fragment()),
            ("bloom_fragment", bloom_fragment()),
            ("gbuffer_vertex", gbuffer_vertex()),
            ("gbuffer_fragment", gbuffer_fragment()),
            ("lighting_vertex", lighting_vertex()),
            ("lighting_fragment", lighting_fragment()),
            ("pbr_vertex", pbr_vertex()),
            ("pbr_fragment", pbr_fragment()),
        ];
        for (name, source) in &shaders {
            assert_valid_wgsl(name, source);
        }
    }

    /// The grouped CPU-side `OpenPBRMaterial` must lay out its groups in the
    /// exact order the shared WGSL declaration lists the `vec4` slots:
    /// base(0) → specular(32) → transmission(64) → subsurface(112) →
    /// fuzz(160) → coat(192) → thin_film(240) → emission(256) →
    /// geometry(288), 320 bytes total. Any reordering breaks every pass that
    /// splices [`OPENPBR_MATERIAL_DECL`].
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
        // Within-group slot order mirrors the WGSL field order.
        assert_eq!(openpbr_material_decl().matches("vec4<f32>").count(), 20);
    }
}
