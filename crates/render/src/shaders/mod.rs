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

/// WGSL-valid rendering of an `f32` value: Rust `{}` prints the shortest
/// round-tripping decimal, plus the `.0` fix for whole values (mirrors
/// the lowering's `float_lit` for literal tokens). Non-finite values
/// have no WGSL spelling — loud in debug, clamped caller's problem.
pub(crate) fn f32_lit(v: f32) -> String {
    debug_assert!(v.is_finite(), "non-finite f32 has no WGSL spelling");
    let base = format!("{v}");
    if base.contains('.') || base.contains('e') || base.contains('E') {
        base
    } else {
        format!("{base}.0")
    }
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

/// Runtime bind-group entries from a [`Resource`] table: binding numbers
/// come from the table (single source with the WGSL decls and the BGL),
/// so a table reorder propagates here by construction. Only the
/// name → live resource mapping stays per call site; an unmapped row
/// panics loudly (a table gain without a call-site update).
pub fn bind_group_entries<'a>(
    table: &[Resource],
    resolve: impl Fn(&Resource) -> wgpu::BindingResource<'a>,
) -> Vec<wgpu::BindGroupEntry<'a>> {
    table
        .iter()
        .map(|r| wgpu::BindGroupEntry {
            binding: r.binding,
            resource: resolve(r),
        })
        .collect()
}

/// Ordered shader-module assembly: call sites declare WHAT goes in,
/// [`emit`](ShaderModule::emit) owns section order and separators.
/// WGSL needs decl-before-use; the former per-site `format!` skeletons
/// encoded that order by hand (and could drift) — here the order
/// (types → resources → consts → helpers → entries) is fixed once.
/// Pieces are taken verbatim except edge-newline normalization (leading
/// and trailing `\n` trimmed, sections joined with single `\n`): WGSL is
/// whitespace-insensitive, and naga plus the pixel probes pin the result.
pub struct ShaderModule {
    decls: Vec<String>,
    resources: Vec<String>,
    consts: Vec<String>,
    helpers: Vec<String>,
    entries: Vec<String>,
}

impl ShaderModule {
    /// Empty assembly.
    pub fn new() -> Self {
        Self {
            decls: Vec::new(),
            resources: Vec::new(),
            consts: Vec::new(),
            helpers: Vec::new(),
            entries: Vec::new(),
        }
    }

    /// One type/varying declaration block (derived `WGSL_SOURCE`,
    /// `wgsl_decl`-wrapped or raw).
    pub fn decl(mut self, src: impl Into<String>) -> Self {
        self.decls.push(src.into());
        self
    }

    /// Resource declarations from a [`Resource`] table subset (same rows
    /// the pass layout builds its BGL from).
    pub fn resources(mut self, table: &[Resource], bindings: &[u32]) -> Self {
        self.resources.push(resource_decls(table, bindings));
        self
    }

    /// One `const` block (quad corners/UVs from Rust data, via IR).
    pub fn consts(mut self, src: impl Into<String>) -> Self {
        self.consts.push(src.into());
        self
    }

    /// One helper/kernel function source.
    pub fn helper(mut self, src: impl Into<String>) -> Self {
        self.helpers.push(src.into());
        self
    }

    /// Several helper/kernel sources at once.
    pub fn helpers<I, S>(mut self, srcs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.helpers.extend(srcs.into_iter().map(Into::into));
        self
    }

    /// One entry point (translated body). A module normally carries one;
    /// several are allowed where the legacy assembly did.
    pub fn entry(mut self, src: impl Into<String>) -> Self {
        self.entries.push(src.into());
        self
    }

    /// Render the module: sections in fixed order, normalized whitespace.
    pub fn emit(self) -> String {
        let mut pieces: Vec<String> = Vec::new();
        for section in [
            self.decls,
            self.resources,
            self.consts,
            self.helpers,
            self.entries,
        ] {
            for piece in section {
                let trimmed = piece.trim_matches('\n');
                if !trimmed.is_empty() {
                    pieces.push(trimmed.to_string());
                }
            }
        }
        let mut out = String::from("\n");
        if !pieces.is_empty() {
            out.push_str(&pieces.join("\n"));
            out.push('\n');
        }
        out
    }
}

impl Default for ShaderModule {
    fn default() -> Self {
        Self::new()
    }
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

/// Quad constants as a stage-entry context bundle: the fullscreen-vertex
/// entries share it (`consts: QuadConsts`, `consts.quad[idx]`). A second
/// `Context<T>` parameter needs no new macro machinery — bundle detection
/// is generic over the wrapper — it just names the group. Field names
/// match the WGSL `const` names exactly; the `#[stage]` `context`
/// lowering strips the `consts.` prefix, no renaming involved.
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct QuadConsts {
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
            hdr_composite_generated::vs_main::wgsl_source().to_string(),
            bloom_generated::vs_main::wgsl_source().to_string(),
            lighting_generated::vs_main::wgsl_source().to_string(),
            composite_generated::vs_main::wgsl_source().to_string(),
            gbuffer_generated::vs_main::wgsl_source().to_string(),
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
        // Three vertex entries construct varyings positionally today (the
        // rest use the `var`-out form); zero hits would mean the comments
        // vanished and the test passes vacuously.
        assert!(checked >= 3, "expected struct literals, found {checked}");
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
        // (bundle/entry globals, assemblies containing them). QuadConsts is
        // shared by five vertex entries, so it pins against all five
        // assemblies carrying the quad constants.
        let pairs: &[(&[&str], Vec<String>)] = &[
            (
                QuadConsts::GLOBALS,
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
            (
                lighting_generated::LightingContext::GLOBALS,
                vec![lighting.clone()],
            ),
            (lighting_generated::LightingMaps::GLOBALS, vec![lighting]),
            (
                composite_generated::CompositeContext::GLOBALS,
                vec![composite],
            ),
            (
                gbuffer_generated::GbufferVertexContext::GLOBALS,
                vec![gbuffer_vertex],
            ),
            (
                gbuffer_generated::fs_main::globals(),
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

    /// Entry-point ABI is pinned per pass: each generated `entry_point()`
    /// must be exactly the pipeline-convention name for its pass. A set
    /// cannot catch distribution drift (one entry renamed, another taking
    /// its place), so every entry is asserted individually. The renderer
    /// and `composite.rs` request entries through these same accessors —
    /// the names below are the external ABI (pipeline caches, snapshots).
    #[test]
    fn entry_points_match_pass_abi() {
        assert_eq!(hdr_composite_generated::vs_main::entry_point(), "vs_main");
        assert_eq!(hdr_composite_generated::fs_main::entry_point(), "fs_main");
        assert_eq!(bloom_generated::vs_main::entry_point(), "vs_main");
        assert_eq!(bloom_generated::fs_main::entry_point(), "fs_main");
        assert_eq!(lighting_generated::vs_main::entry_point(), "vs_main");
        assert_eq!(lighting_generated::fs_main::entry_point(), "fs_main");
        assert_eq!(composite_generated::vs_main::entry_point(), "vs_main");
        assert_eq!(composite_generated::fs_main::entry_point(), "fs_main");
        assert_eq!(gbuffer_generated::vs_main::entry_point(), "vs_main");
        assert_eq!(gbuffer_generated::fs_main::entry_point(), "fs_main");
        assert_eq!(pbr_generated::fs_main::entry_point(), "fs_main");
    }

    /// Every non-swizzle `root.field` use in translated entries resolves
    /// against a real definition: `camera`/`lighting`/`bloom_params`
    /// against their buffer layouts (`FIELD_NAMES`), `mat` against
    /// `OPENPBR_SLOTS`, `obj` against `PerObjectGpu`, `input`/`output`
    /// against the pass mirrors (`WGSL_FIELDS`, per entry below).
    /// Anything else dotted is either a vector swizzle (x/y/z/w/r/g/b/a
    /// combos, skipped by pattern) or a hard error: an unresolved root is
    /// a free identifier past the bundles or a typo'd root, and a missing
    /// field is a typo'd member. This is the field-resolution slice of
    /// "valid Rust": no mocks, no second lowering — the oracles are the
    /// real definitions, the text is the real output.
    #[test]
    fn stage_field_uses_resolve() {
        use super::interface::{
            BloomVertexOut, GbufferFragmentInput, GbufferOutput, GbufferVertexInput,
            GbufferVertexOutput, HdrFragmentOut, UiCompositeOut,
        };
        use crate::renderer::{BloomUniform, CameraUniform, LightingUniform, PerObjectGpu};

        fn is_swizzle(field: &str) -> bool {
            field.len() <= 4 && field.chars().all(|c| "xyzwrgba".contains(c))
        }

        // (translated entry source, input mirror fields, output mirror fields)
        struct Entry {
            src: &'static str,
            input: Option<&'static [&'static str]>,
            output: Option<&'static [&'static str]>,
        }
        let entries = [
            Entry {
                src: composite_generated::vs_main::wgsl_source(),
                input: None,
                output: Some(UiCompositeOut::WGSL_FIELDS),
            },
            Entry {
                src: composite_generated::fs_main::wgsl_source(),
                input: Some(UiCompositeOut::WGSL_FIELDS),
                output: None,
            },
            Entry {
                src: gbuffer_generated::vs_main::wgsl_source(),
                input: Some(GbufferVertexInput::WGSL_FIELDS),
                output: Some(GbufferVertexOutput::WGSL_FIELDS),
            },
            Entry {
                src: gbuffer_generated::fs_main::wgsl_source(),
                input: Some(GbufferFragmentInput::WGSL_FIELDS),
                output: Some(GbufferOutput::WGSL_FIELDS),
            },
            Entry {
                src: pbr_generated::fs_main::wgsl_source(),
                input: Some(GbufferFragmentInput::WGSL_FIELDS),
                output: None,
            },
            Entry {
                src: hdr_composite_generated::vs_main::wgsl_source(),
                input: None,
                output: None,
            },
            Entry {
                src: hdr_composite_generated::fs_main::wgsl_source(),
                input: Some(HdrFragmentOut::WGSL_FIELDS),
                output: None,
            },
            Entry {
                src: bloom_generated::vs_main::wgsl_source(),
                input: None,
                output: None,
            },
            Entry {
                src: bloom_generated::fs_main::wgsl_source(),
                input: Some(BloomVertexOut::WGSL_FIELDS),
                output: None,
            },
            Entry {
                src: lighting_generated::vs_main::wgsl_source(),
                input: None,
                output: None,
            },
            Entry {
                src: lighting_generated::fs_main::wgsl_source(),
                input: Some(HdrFragmentOut::WGSL_FIELDS),
                output: None,
            },
        ];
        // Shared roots: uniform buffers by layout, materials by slot list.
        let shared: &[(&str, &[&str])] = &[
            ("camera", CameraUniform::FIELD_NAMES),
            ("lighting", LightingUniform::FIELD_NAMES),
            ("bloom_params", BloomUniform::FIELD_NAMES),
            ("mat", &OPENPBR_SLOTS),
            ("obj", PerObjectGpu::FIELD_NAMES),
        ];
        let mut checked = 0;
        for (n, entry) in entries.iter().enumerate() {
            let mut roots: Vec<(&str, &[&str])> = shared.to_vec();
            if let Some(fields) = entry.input {
                roots.push(("input", fields));
            }
            if let Some(fields) = entry.output {
                // `var`-out locals are spelled `out` (composite) or
                // `output` (g-buffer); both resolve to the output mirror.
                roots.push(("out", fields));
                roots.push(("output", fields));
            }
            let mut rest = entry.src;
            while let Some(dot) = rest.find('.') {
                let before = &rest[..dot];
                let root: String = before
                    .chars()
                    .rev()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                let after = &rest[dot + 1..];
                let field: String = after
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !root.is_empty() && !field.is_empty() && !is_swizzle(&field) {
                    // Float literals (`0.0`) look like `root.field` to the
                    // scanner; numeric roots are never field uses.
                    if root.starts_with(|c: char| c.is_ascii_digit()) {
                        rest = after;
                        continue;
                    }
                    let (_, fields) = roots
                        .iter()
                        .find(|(r, _)| *r == root)
                        .unwrap_or_else(|| panic!("entry {n}: unresolved root `{root}.{field}`"));
                    assert!(
                        fields.contains(&field.as_str()),
                        "entry {n}: `{root}` has no field `{field}`"
                    );
                    checked += 1;
                }
                rest = after;
            }
        }
        // Eleven entries, 151 resolved uses today; fewer means the
        // extraction broke or coverage was dropped — update deliberately.
        assert!(checked >= 151, "expected field uses, found {checked}");
    }

    /// Every call in translated entries resolves: built-ins via the
    /// shared `ornis-shader-lang` registry (single source with the
    /// translator — no second spelling), WGSL natives (texture ops) via a
    /// short stable list, type constructors via `ShaderType`, and
    /// kernels/helpers/mirrors via `fn`/`struct` definitions extracted
    /// from the same assembled shader. A typo'd kernel name can no longer
    /// slide to naga — it fails here with the entry index and callee.
    #[test]
    fn stage_calls_resolve() {
        use ornis_shader_lang::{ShaderBuiltin, ShaderType};

        fn callee_uses(src: &str) -> Vec<(String, usize)> {
            // `name(` occurrences with their byte offset (for messages).
            let mut out = Vec::new();
            let bytes = src.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'(' {
                    let mut j = i;
                    while j > 0 && (bytes[j - 1].is_ascii_alphanumeric() || bytes[j - 1] == b'_') {
                        j -= 1;
                    }
                    if j < i {
                        // `@builtin(` / `@location(` are attributes, not
                        // calls; `if (` etc. are filtered as keywords below.
                        if j == 0 || bytes[j - 1] != b'@' {
                            out.push((src[j..i].to_string(), j));
                        }
                    }
                    i += 1;
                } else {
                    i += 1;
                }
            }
            out
        }

        fn defined_names(assembly: &str) -> Vec<String> {
            // `fn name` and `struct name` declared in the assembled module.
            let mut names = Vec::new();
            for (kw_len, kw) in [(3, "fn "), (7, "struct ")] {
                let mut rest = assembly;
                while let Some(pos) = rest.find(kw) {
                    let after = &rest[pos + kw_len..];
                    let name: String = after
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        names.push(name);
                    }
                    rest = after;
                }
            }
            names
        }

        // WGSL-native spellings: prelude texture ops. Everything else
        // function-like must come from the registry or the assembly.
        const NATIVES: &[&str] = &[
            "textureSample",
            "textureSampleLevel",
            "textureLoad",
            "textureDimensions",
        ];
        const KEYWORDS: &[&str] = &["if", "for", "while", "return", "loop"];
        // (entry source, its assembled shader)
        let hdr_vertex = hdr_composite_generated::wgsl_vertex_source();
        let hdr_fragment = hdr_composite_generated::wgsl_source();
        let bloom = bloom_generated::wgsl_source();
        let lighting = lighting_generated::wgsl_source();
        let lighting_vertex = lighting_generated::wgsl_vertex_source();
        let composite = composite_generated::wgsl_source();
        let gbuffer_vertex = gbuffer_generated::wgsl_vertex_source();
        let gbuffer_fragment = gbuffer_generated::wgsl_source();
        let pbr = pbr_generated::wgsl_source();
        let pairs: &[(&str, &str)] = &[
            (
                composite_generated::vs_main::wgsl_source(),
                composite.as_str(),
            ),
            (
                composite_generated::fs_main::wgsl_source(),
                composite.as_str(),
            ),
            (
                gbuffer_generated::vs_main::wgsl_source(),
                gbuffer_vertex.as_str(),
            ),
            (
                gbuffer_generated::fs_main::wgsl_source(),
                gbuffer_fragment.as_str(),
            ),
            (pbr_generated::fs_main::wgsl_source(), pbr.as_str()),
            (
                hdr_composite_generated::vs_main::wgsl_source(),
                hdr_vertex.as_str(),
            ),
            (
                hdr_composite_generated::fs_main::wgsl_source(),
                hdr_fragment.as_str(),
            ),
            (bloom_generated::vs_main::wgsl_source(), bloom.as_str()),
            (bloom_generated::fs_main::wgsl_source(), bloom.as_str()),
            (
                lighting_generated::vs_main::wgsl_source(),
                lighting_vertex.as_str(),
            ),
            (
                lighting_generated::fs_main::wgsl_source(),
                lighting.as_str(),
            ),
        ];
        let mut checked = 0;
        for (n, (entry, assembly)) in pairs.iter().enumerate() {
            let defs = defined_names(assembly);
            for (callee, _) in callee_uses(entry) {
                if KEYWORDS.contains(&callee.as_str()) {
                    continue;
                }
                let resolved = ShaderBuiltin::from_rust(&callee).is_some()
                    || ShaderType::from_rust(&callee).is_some()
                    || NATIVES.contains(&callee.as_str())
                    || defs.iter().any(|d| d == &callee);
                assert!(
                    resolved,
                    "entry {n}: unresolved call `{callee}` — typo'd kernel/helper or missing definition"
                );
                checked += 1;
            }
        }
        // Eleven entries make 85 calls today; near-zero hits would mean
        // the extraction broke and the test passes vacuously.
        assert!(checked >= 85, "expected calls, found {checked}");
    }

    /// Every table-backed global an entry reads is visible at that
    /// entry's stage. Module consts (`quad`/`uvs`) have no table rows and
    /// are skipped here — their existence is pinned by
    /// `stage_globals_declared`; this test pins the `visibility` column
    /// against actual use, so a vertex entry reading a fragment-only
    /// resource fails before any GPU ever runs.
    #[test]
    fn stage_resources_visible_at_stage() {
        use wgpu::ShaderStages;
        // (globals read by one entry, pass table, entry stage)
        let cases: &[(&[&str], &[Resource], ShaderStages)] = &[
            (
                QuadConsts::GLOBALS,
                &composite_generated::COMPOSITE_RESOURCES,
                ShaderStages::VERTEX,
            ),
            (
                composite_generated::CompositeContext::GLOBALS,
                &composite_generated::COMPOSITE_RESOURCES,
                ShaderStages::FRAGMENT,
            ),
            (
                gbuffer_generated::GbufferVertexContext::GLOBALS,
                &gbuffer_generated::GBUFFER_RESOURCES,
                ShaderStages::VERTEX,
            ),
            (
                gbuffer_generated::fs_main::globals(),
                &gbuffer_generated::GBUFFER_RESOURCES,
                ShaderStages::FRAGMENT,
            ),
            (
                pbr_generated::PbrContext::GLOBALS,
                &pbr_generated::PBR_RESOURCES,
                ShaderStages::FRAGMENT,
            ),
            (
                QuadConsts::GLOBALS,
                &hdr_composite_generated::HDR_RESOURCES,
                ShaderStages::VERTEX,
            ),
            (
                hdr_composite_generated::HdrContext::GLOBALS,
                &hdr_composite_generated::HDR_RESOURCES,
                ShaderStages::FRAGMENT,
            ),
            (
                QuadConsts::GLOBALS,
                &bloom_generated::BLOOM_RESOURCES,
                ShaderStages::VERTEX,
            ),
            (
                bloom_generated::BloomContext::GLOBALS,
                &bloom_generated::BLOOM_RESOURCES,
                ShaderStages::FRAGMENT,
            ),
            (
                QuadConsts::GLOBALS,
                &lighting_generated::LIGHTING_RESOURCES,
                ShaderStages::VERTEX,
            ),
            (
                lighting_generated::LightingContext::GLOBALS,
                &lighting_generated::LIGHTING_RESOURCES,
                ShaderStages::FRAGMENT,
            ),
            (
                lighting_generated::LightingMaps::GLOBALS,
                &lighting_generated::LIGHTING_RESOURCES,
                ShaderStages::FRAGMENT,
            ),
        ];
        let mut checked = 0;
        for (globals, table, stage) in cases {
            for global in *globals {
                if let Some(row) = table.iter().find(|r| r.name == *global) {
                    assert!(
                        row.visibility.contains(*stage),
                        "`{global}` is read at {stage:?} but bound for {:?}",
                        row.visibility
                    );
                    checked += 1;
                }
            }
        }
        // 28 table-backed uses today; fewer means a table lost a row the
        // entries still read — update deliberately.
        assert!(checked >= 28, "expected visible resources, found {checked}");
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
