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

/// Build one `@group/@binding` resource line from its parts, e.g.
/// `binding(0, 2, "var<storage, read> materials: array<OpenPBRMaterial>")`.
/// Group/binding numbers stay explicit at each call site so the layout is
/// reviewable where it is used.
pub(crate) fn binding(group: u32, binding: u32, decl: &str) -> String {
    format!("@group({group}) @binding({binding}) {decl};\n")
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
            Self::Uniform(ty) | Self::StorageRead(ty) | Self::StorageRw(ty) => {
                ty.to_string()
            }
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
            Self::TextureFloat | Self::TextureUint | Self::TextureDepth | Self::Sampler => {
                "var"
            }
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
            Self::StorageRead(_) | Self::StorageReadArray(_) => {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            }
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
    wgpu::BindGroupLayoutEntry {
        binding: r.binding,
        visibility: r.visibility,
        ty: r.kind.bgl_ty(multisampled),
        count: None,
    }
}

/// Spell one `f32` the way the former handwritten sources did (`-1.0`,
/// not `-1`), so generated declaration blocks stay readable.
fn fmt_f32(x: f32) -> String {
    if x.fract() == 0.0 {
        format!("{x:.1}")
    } else {
        format!("{x}")
    }
}

/// Build a `const NAME: array<vec4<f32>, N> = …` block from Rust data.
pub(crate) fn const_vec4_array(name: &str, vals: &[[f32; 4]]) -> String {
    let mut out = format!(
        "const {name}: array<vec4<f32>, {n}> = array<vec4<f32>, {n}>(\n",
        n = vals.len()
    );
    for v in vals {
        out.push_str(&format!(
            "    vec4<f32>({}, {}, {}, {}),\n",
            fmt_f32(v[0]),
            fmt_f32(v[1]),
            fmt_f32(v[2]),
            fmt_f32(v[3])
        ));
    }
    out.push_str(");\n");
    out
}

/// Build a `const NAME: array<vec2<f32>, N> = …` block from Rust data.
pub(crate) fn const_vec2_array(name: &str, vals: &[[f32; 2]]) -> String {
    let mut out = format!(
        "const {name}: array<vec2<f32>, {n}> = array<vec2<f32>, {n}>(\n",
        n = vals.len()
    );
    for v in vals {
        out.push_str(&format!(
            "    vec2<f32>({}, {}),\n",
            fmt_f32(v[0]),
            fmt_f32(v[1])
        ));
    }
    out.push_str(");\n");
    out
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
