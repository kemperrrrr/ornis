//! Shader source assembly.
//!
//! Each public function here assembles a complete WGSL module by splicing
//! math-kernel snippets (generated from Rust via `wgsl_source()`, see
//! [`math`]) into a fixed entry-point skeleton, so CPU tests exercise the
//! exact BRDF code the GPU runs.

pub mod math;

/// ── COMPOSITE_VERTEX ────────────────────────────────────────────────
const COMPOSITE_VERTEX_BOILERPLATE: &str = include_str!("wgsl/composite_vertex.wgsl");

/// Assemble the full-screen composite vertex shader (triangle-strip quad).
pub fn composite_vertex() -> String {
    COMPOSITE_VERTEX_BOILERPLATE.to_string()
}

/// ── COMPOSITE_FRAGMENT ──────────────────────────────────────────────
const COMPOSITE_FRAGMENT_BOILERPLATE: &str = include_str!("wgsl/composite_fragment.wgsl");

/// Assemble the composite fragment shader: HDR mix + bloom, splicing the ACES tonemap and luminance kernels via `wgsl_source()`.
pub fn composite_fragment() -> String {
    format!(
        "{}\n{}\n{}",
        COMPOSITE_FRAGMENT_BOILERPLATE,
        math::aces_tonemap::wgsl_source(),
        math::luminance::wgsl_source(),
    )
}

/// ── BLOOM ───────────────────────────────────────────────────────────
///
/// A single quad shader used by the whole bloom chain. Each pass samples
/// the previous (smaller or larger) level and applies a soft threshold on
/// luminance: the first downsample keeps only the bright pixels
/// (threshold ≈ 0.6-0.7), later levels pass everything (threshold = 0),
/// and the upsample passes re-add the level's own content via additive
/// blending (dst = src + previous), recreating the classic Frostbite
/// "downsample chain, upsample with add" cascade.
const BLOOM_FRAGMENT_BOILERPLATE: &str = include_str!("wgsl/bloom_fragment.wgsl");

/// Assemble the bloom fragment shader (bright-pass/blend), splicing the luminance kernel.
pub fn bloom_fragment() -> String {
    format!(
        "{}\n{}",
        BLOOM_FRAGMENT_BOILERPLATE,
        math::luminance::wgsl_source(),
    )
}

/// ── GBUFFER_VERTEX ──────────────────────────────────────────────────
const GBUFFER_VERTEX_BOILERPLATE: &str = include_str!("wgsl/gbuffer_vertex.wgsl");

/// Assemble the gbuffer vertex shader (instance transforms + world position).
pub fn gbuffer_vertex() -> String {
    GBUFFER_VERTEX_BOILERPLATE.to_string()
}

/// ── GBUFFER_FRAGMENT ────────────────────────────────────────────────
const GBUFFER_FRAGMENT_BOILERPLATE: &str = include_str!("wgsl/gbuffer_fragment.wgsl");

/// Assemble the 5-MRT gbuffer fragment shader, splicing the octahedral normal-encoding kernel.
pub fn gbuffer_fragment() -> String {
    format!(
        "{}\n{}",
        GBUFFER_FRAGMENT_BOILERPLATE,
        math::octahedral_encode::wgsl_source(),
    )
}

/// ── LIGHTING_VERTEX ─────────────────────────────────────────────────
const LIGHTING_VERTEX_BOILERPLATE: &str = include_str!("wgsl/lighting_vertex.wgsl");

/// Assemble the full-screen lighting vertex shader.
pub fn lighting_vertex() -> String {
    LIGHTING_VERTEX_BOILERPLATE.to_string()
}

/// ── LIGHTING_FRAGMENT ───────────────────────────────────────────────
const LIGHTING_BOILERPLATE: &str = include_str!("wgsl/lighting.wgsl");

/// Assemble the deferred lighting fragment shader: reconstructs surface data and evaluates OpenPBR by splicing all BRDF math kernels via `wgsl_source()`.
pub fn lighting_fragment() -> String {
    let kernels = [
        math::luminance::wgsl_source(),
        math::aces_tonemap::wgsl_source(),
        math::fresnel0_from_ior::wgsl_source(),
        math::fresnel_schlick::wgsl_source(),
        math::fresnel_schlick_vec::wgsl_source(),
        math::fresnel_f82_tint::wgsl_source(),
        math::ggx_ndf::wgsl_source(),
        math::ggx_ndf_aniso::wgsl_source(),
        math::openpbr_anisotropy::wgsl_source(),
        math::smith_ggx_correlated::wgsl_source(),
        math::smith_ggx_aniso::wgsl_source(),
        math::oren_nayar_brdf::wgsl_source(),
        math::coat_base_darkening::wgsl_source(),
        math::coat_blend_darkened::wgsl_source(),
        math::thin_film_modulation::wgsl_source(),
        math::sheen_brdf::wgsl_source(),
        math::transmission_color_to_extinction::wgsl_source(),
        math::subsurface_brdf::wgsl_source(),
    ];
    let mut src = LIGHTING_BOILERPLATE.to_string();
    for k in &kernels {
        src.push('\n');
        src.push_str(k);
    }
    src
}

/// ── PBR_VERTEX ──────────────────────────────────────────────────────
const PBR_VERTEX_BOILERPLATE: &str = include_str!("wgsl/pbr_vertex.wgsl");

/// Assemble the forward PBR vertex shader.
pub fn pbr_vertex() -> String {
    PBR_VERTEX_BOILERPLATE.to_string()
}

/// ── PBR_FRAGMENT ────────────────────────────────────────────────────
const PBR_FRAGMENT_BOILERPLATE: &str = include_str!("wgsl/pbr_fragment.wgsl");

/// Assemble the forward PBR fragment shader: full OpenPBR evaluation, splicing all BRDF math kernels via `wgsl_source()`.
pub fn pbr_fragment() -> String {
    let kernels = [
        math::luminance::wgsl_source(),
        math::aces_tonemap::wgsl_source(),
        math::fresnel0_from_ior::wgsl_source(),
        math::fresnel_schlick::wgsl_source(),
        math::fresnel_schlick_vec::wgsl_source(),
        math::fresnel_f82_tint::wgsl_source(),
        math::ggx_ndf::wgsl_source(),
        math::ggx_ndf_aniso::wgsl_source(),
        math::openpbr_anisotropy::wgsl_source(),
        math::smith_ggx_correlated::wgsl_source(),
        math::smith_ggx_aniso::wgsl_source(),
        math::oren_nayar_brdf::wgsl_source(),
        math::coat_base_darkening::wgsl_source(),
        math::coat_blend_darkened::wgsl_source(),
        math::thin_film_modulation::wgsl_source(),
        math::sheen_brdf::wgsl_source(),
        math::transmission_color_to_extinction::wgsl_source(),
        math::subsurface_brdf::wgsl_source(),
        math::srgb_to_linear::wgsl_source(),
    ];
    let mut src = PBR_FRAGMENT_BOILERPLATE.to_string();
    for k in &kernels {
        src.push('\n');
        src.push_str(k);
    }
    src
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
}
