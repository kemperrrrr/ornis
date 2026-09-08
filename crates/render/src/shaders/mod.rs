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

/// Shared `OpenPBRMaterial` WGSL declaration (20 `vec4` slots), used by the
/// g-buffer, lighting and forward-PBR skeletons.
///
/// Still handwritten: the CPU-side
/// [`OpenPBRMaterial`](ornis_core::material::OpenPBRMaterial) is grouped
/// (`BaseGroup`, `SpecularGroup`, …), not flat, so there is no 1:1 field
/// mirror to derive from. The `openpbr_rust_layout_matches_wgsl_order` test
/// below pins the group offsets against this declaration order instead.
/// Leading/trailing newlines are part of the legacy assembly contract.
pub(crate) const OPENPBR_MATERIAL_DECL: &str = r#"
struct OpenPBRMaterial {
    base_params: vec4<f32>,
    base_color: vec4<f32>,
    specular_params: vec4<f32>,
    specular_color: vec4<f32>,
    transmission_params: vec4<f32>,
    transmission_color: vec4<f32>,
    transmission_scatter: vec4<f32>,
    subsurface_params: vec4<f32>,
    subsurface_color: vec4<f32>,
    subsurface_radius_scale_gb: vec4<f32>,
    fuzz_params: vec4<f32>,
    fuzz_color: vec4<f32>,
    coat_params: vec4<f32>,
    coat_color: vec4<f32>,
    coat_ior: vec4<f32>,
    thin_film_params: vec4<f32>,
    emission_params: vec4<f32>,
    emission_color: vec4<f32>,
    geometry_params: vec4<f32>,
    geometry_params2: vec4<f32>,
};
"#;

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
        assert_eq!(OPENPBR_MATERIAL_DECL.matches("vec4<f32>").count(), 20);
    }
}
