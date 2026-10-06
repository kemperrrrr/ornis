//! Shared OpenPBR lighting helpers translated from Rust.
//!
//! The layer evaluators (`evaluate_*`, `transmission_btdf`) are textually
//! identical in the deferred-lighting and forward-PBR passes, so they live
//! here once: each `#[wgsl_fn]` function is replaced by a module exposing
//! `wgsl_source()`, and the passes splice those into the assembled WGSL.
//! [`octahedral_decode`]/[`reconstruct_world_pos`] are lighting-only
//! (g-buffer decode) and splice only there.
//!
//! Bodies are DSL-only and never compiled as Rust (like
//! [`stage`](ornis_macros::stage) entries): parameter types pass through by
//! name (`mat: OpenPBRMaterial`, `camera: Camera`), kernel calls stay bare
//! identifiers, and constructors are spelled `Vec3::new` / `Vec2::new`.
//! The float constants below are `f32`-exact (`format!` prints the short
//! decimal that round-trips to the same bits as the former literals).

/// Single-precision π, matching the former `3.14159265359` literal bit-wise.
pub const PI: f32 = std::f32::consts::PI;
/// Epsilon guard against division by zero (former `1e-6`).
pub const EPS: f32 = 1e-6;
/// Reference-depth bias subtracted in the shadow evaluators (former
/// `0.002` literal, now shared by the 2D and cube branches of both
/// entries): keeps the lit surface's own texels on the lit side of
/// the compare. Larger values erode small blobs uniformly; smaller
/// values risk acne (guarded by
/// `shadowed_directional_without_occluder_has_no_acne`).
pub const SHADOW_REF_BIAS: f32 = 0.001;
/// Single-precision 1/π, matching the former `0.31830988618` bit-wise.
pub const INV_PI: f32 = std::f32::consts::FRAC_1_PI;
/// Cleared g-buffer depth (far plane). A sample at this depth is background:
/// its octahedral clear `(0, 0)` decodes to +Z and must not be shaded.
pub const CLEAR_DEPTH: f32 = 1.0;
/// Dot product above which two sample normals are the same surface.
pub const NORMAL_AGREE: f32 = 0.999;
/// NDC depth gap that splits two samples onto different surfaces.
pub const DEPTH_EDGE: f32 = 0.002;
/// Alpha below which a deferred pixel is absent (former `0.001` discard).
pub const ALPHA_CUTOFF: f32 = 0.001;
/// `MSAA_SAMPLE_COUNT` as `f32`, for coverage comparisons in WGSL.
pub const MSAA_SAMPLES_F: f32 = 4.0;
/// `step` edge between a directional light (kind 0) and a point or spot (kind ≥ 1).
pub const POINT_OR_SPOT_KIND_EDGE: f32 = 0.5;
/// `step` edge between a point light (kind 1) and a spot (kind 2).
pub const SPOT_KIND_EDGE: f32 = 1.5;
/// Inner fraction of a finite light's range where attenuation starts to fall.
///
/// `smoothstep` with equal edges is undefined in WGSL, so the cutoff begins
/// this far inside the light range.
pub const RANGE_CUTOFF_INNER_FRACTION: f32 = 0.99;
/// Half-extent that maps NDC `[-1, 1]` onto a texture UV `[0, 1]`.
pub const NDC_TO_UV_HALF: f32 = 0.5;
/// Near plane of a point-light cube-shadow face. The far plane is the light range.
pub const SHADOW_CUBE_NEAR: f32 = 0.1;
/// Split on a 0/1 light selector: at or below this, the light is not a point light.
pub const LIGHT_SELECTOR_SPLIT: f32 = 0.5;

/// WGSL `const` block for the helpers, generated from the Rust constants
/// above: the name travels via `stringify!` (rename-proof), the value via
/// [`f32_lit`](super::f32_lit) (bit-exact, WGSL-valid). No second spelling.
pub fn wgsl_consts() -> String {
    macro_rules! decl {
        ($name:ident) => {
            format!(
                "const {}: f32 = {};\n",
                stringify!($name),
                super::f32_lit($name)
            )
        };
    }
    let floats = [
        decl!(PI),
        decl!(EPS),
        decl!(INV_PI),
        decl!(SHADOW_REF_BIAS),
        decl!(CLEAR_DEPTH),
        decl!(NORMAL_AGREE),
        decl!(DEPTH_EDGE),
        decl!(ALPHA_CUTOFF),
        decl!(MSAA_SAMPLES_F),
        decl!(POINT_OR_SPOT_KIND_EDGE),
        decl!(SPOT_KIND_EDGE),
        decl!(RANGE_CUTOFF_INNER_FRACTION),
        decl!(NDC_TO_UV_HALF),
        decl!(SHADOW_CUBE_NEAR),
        decl!(LIGHT_SELECTOR_SPLIT),
    ]
    .concat();
    format!("{floats}{}", msaa_samples_wgsl())
}

/// `u32` sample count shared with [`crate::renderer::MSAA_SAMPLE_COUNT`].
fn msaa_samples_wgsl() -> String {
    format!(
        "const MSAA_SAMPLES: u32 = {}u;\n",
        crate::renderer::MSAA_SAMPLE_COUNT
    )
}

/// Base layer: dielectric/metallic mix with anisotropic GGX + Oren-Nayar.
#[ornis_macros::wgsl_fn]
fn evaluate_base_layer(
    n: glam::Vec3,
    v: glam::Vec3,
    l: glam::Vec3,
    h: glam::Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    mat: OpenPBRMaterial,
    base_weight: f32,
    base_color: glam::Vec3,
    metalness: f32,
    diffuse_roughness: f32,
    specular_weight: f32,
    specular_roughness: f32,
    specular_ior: f32,
    specular_anisotropy: f32,
    specular_edge_tint: glam::Vec3,
    t: glam::Vec3,
    b: glam::Vec3,
    thin_film_mod: glam::Vec3,
) -> glam::Vec3 {
    let f0_dielectric = Vec3::new(fresnel0_from_ior(specular_ior));
    let f0_metal = base_color * base_weight;
    let f0 = mix(f0_dielectric, f0_metal, metalness);
    let f = fresnel_f82_tint(voh, f0, specular_edge_tint);
    let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);
    let alpha_u = alpha.x;
    let alpha_v = alpha.y;
    let d = ggx_ndf_aniso(noh, h, t, b, alpha_u, alpha_v);
    // Smith returns V = G / (4 NoV NoL). The specular BRDF is D·F·V;
    // dividing by 4·NoV·NoL again blows up at grazing angles (silhouette
    // fireflies) because the light loop already multiplies by NoL.
    let g = smith_ggx_aniso(nov, nol, v, l, t, b, alpha_u, alpha_v);
    let spec_brdf = d * g * f;
    let diffuse_color = base_color * (1.0 - metalness);
    let diff_roughness = max(diffuse_roughness, specular_roughness);
    let diff_alpha = diff_roughness * diff_roughness;
    let cos_phi = max(dot(normalize(v - n * nov), normalize(l - n * nol)), 0.0);
    let diff_brdf = oren_nayar_brdf(nov, nol, cos_phi, diff_alpha);
    let ks = f * specular_weight;
    let kd = Vec3::new(base_diffuse_energy(luminance(ks)));
    let base_bsdf = kd * diff_brdf * diffuse_color + ks * spec_brdf;
    return base_bsdf * base_weight * thin_film_mod;
}

/// Coat specular lobe. Base darkening is [`evaluate_coat_darkening`],
/// applied to the base lobe by the caller — adding it here used to
/// brighten the coat instead of attenuating the base.
#[ornis_macros::wgsl_fn]
fn evaluate_coat_layer(
    n: glam::Vec3,
    v: glam::Vec3,
    l: glam::Vec3,
    h: glam::Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    mat: OpenPBRMaterial,
    coat_weight: f32,
    coat_roughness: f32,
    coat_anisotropy: f32,
    coat_dark: f32,
    coat_ior: f32,
    coat_color: glam::Vec3,
    base_metalness: f32,
    base_color: glam::Vec3,
    base_weight: f32,
    specular_weight: f32,
    subsurface_weight: f32,
    subsurface_color: glam::Vec3,
    t: glam::Vec3,
    b: glam::Vec3,
) -> glam::Vec3 {
    if coat_weight <= 0.0 {
        return Vec3::new(0.0);
    }
    let coat_f0 = Vec3::new(fresnel0_from_ior(coat_ior));
    let coat_f = fresnel_schlick_vec(voh, coat_f0);
    let coat_alpha = openpbr_anisotropy(coat_roughness, coat_anisotropy);
    let coat_alpha_u = coat_alpha.x;
    let coat_alpha_v = coat_alpha.y;
    let coat_d = ggx_ndf_aniso(noh, h, t, b, coat_alpha_u, coat_alpha_v);
    // Same visibility contract as the base lobe: `coat_g` is already V.
    let coat_g = smith_ggx_aniso(nov, nol, v, l, t, b, coat_alpha_u, coat_alpha_v);
    let coat_brdf = coat_d * coat_g * coat_f;
    return coat_color * coat_brdf * coat_weight;
}

/// Multiplier applied to the base lobe under a coat.
///
/// This is the coat-darkening factor (1 when the coat is off). It used
/// to be added as `darkening * coat_albedo`, which brightened the coat
/// instead of attenuating the base underneath it.
#[ornis_macros::wgsl_fn]
fn evaluate_coat_darkening(
    coat_weight: f32,
    coat_dark: f32,
    coat_ior: f32,
    base_metalness: f32,
    base_color: glam::Vec3,
    base_weight: f32,
    specular_weight: f32,
    subsurface_weight: f32,
    subsurface_color: glam::Vec3,
) -> glam::Vec3 {
    return coat_blend_darkened(
        coat_base_darkening(
            coat_ior,
            base_metalness,
            base_color,
            base_weight,
            specular_weight,
            subsurface_weight,
            subsurface_color,
        ),
        coat_weight,
        coat_dark,
    );
}

/// Fuzz (sheen) layer.
#[ornis_macros::wgsl_fn]
fn evaluate_fuzz_layer(
    n: glam::Vec3,
    v: glam::Vec3,
    l: glam::Vec3,
    h: glam::Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    fuzz_weight: f32,
    fuzz_roughness: f32,
    fuzz_color: glam::Vec3,
) -> glam::Vec3 {
    if fuzz_weight <= 0.0 {
        return Vec3::new(0.0);
    }
    let sheen = sheen_brdf(nov, nol, noh, voh, fuzz_roughness);
    return fuzz_color * sheen * fuzz_weight;
}

/// Transmission layer with Beer-Lambert extinction and BTDF.
#[ornis_macros::wgsl_fn]
fn evaluate_transmission_layer(
    n: glam::Vec3,
    v: glam::Vec3,
    l: glam::Vec3,
    h: glam::Vec3,
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    mat: OpenPBRMaterial,
    transmission_weight: f32,
    transmission_depth: f32,
    transmission_dispersion_scale: f32,
    transmission_dispersion_abbe: f32,
    transmission_color: glam::Vec3,
    transmission_scatter: glam::Vec3,
    transmission_scatter_anisotropy: f32,
    specular_ior: f32,
    specular_roughness: f32,
    specular_anisotropy: f32,
    thin_walled: f32,
) -> glam::Vec3 {
    if transmission_weight <= 0.0 {
        return Vec3::new(0.0);
    }
    let ior_out = 1.0;
    let ior_in = specular_ior;
    let eta = ior_in / ior_out;
    let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);
    let extinction = transmission_color_to_extinction(transmission_color, transmission_depth);
    let distance = transmission_depth;
    let btdf = transmission_btdf(
        nov, nol, noh, voh, ior_in, ior_out, alpha.x, extinction, distance,
    );
    return transmission_color * btdf * transmission_weight;
}

/// Subsurface layer with projected-distance falloff.
#[ornis_macros::wgsl_fn]
fn evaluate_subsurface_layer(
    n: glam::Vec3,
    v: glam::Vec3,
    l: glam::Vec3,
    nov: f32,
    nol: f32,
    subsurface_weight: f32,
    subsurface_radius: f32,
    subsurface_radius_scale_r: f32,
    subsurface_radius_scale_g: f32,
    subsurface_radius_scale_b: f32,
    subsurface_scatter_anisotropy: f32,
    subsurface_color: glam::Vec3,
) -> glam::Vec3 {
    if subsurface_weight <= 0.0 {
        return Vec3::new(0.0);
    }
    let radius = Vec3::new(
        subsurface_radius * subsurface_radius_scale_r,
        subsurface_radius * subsurface_radius_scale_g,
        subsurface_radius * subsurface_radius_scale_b,
    );
    let v_proj = v - n * nov;
    let l_proj = l - n * nol;
    let distance = length(v_proj - l_proj);
    let ss_brdf = subsurface_brdf(nov, nol, distance, radius, subsurface_scatter_anisotropy);
    return subsurface_color * ss_brdf * subsurface_weight;
}

/// Emission with coat-weighted Fresnel modulation.
#[ornis_macros::wgsl_fn]
fn evaluate_emission(
    emission_luminance: f32,
    emission_color: glam::Vec3,
    coat_weight: f32,
    coat_color: glam::Vec3,
    nov: f32,
) -> glam::Vec3 {
    return coated_emission(
        emission_color,
        emission_luminance,
        coat_weight,
        coat_color,
        nov,
    );
}

/// Microfacet transmission BTDF with extinction.
#[ornis_macros::wgsl_fn]
fn transmission_btdf(
    nov: f32,
    nol: f32,
    noh: f32,
    voh: f32,
    ior_in: f32,
    ior_out: f32,
    alpha: f32,
    extinction: glam::Vec3,
    distance: f32,
) -> glam::Vec3 {
    let eta = ior_in / ior_out;
    let cos_theta_t = sqrt(max(1.0 - eta * eta * (1.0 - nov * nov), 0.0));
    let cos_theta_i = nov;
    let f = fresnel_schlick(max(cos_theta_i, EPS), fresnel0_from_ior(ior_in));
    let t = 1.0 - f;
    // The NDF argument is the facet normal `N·H`, not `V·H` (those agree
    // only when the half-vector sits on the normal). `g` is already
    // V = G / (4 NoV NoL); the BTDF is D·T·V, not divided again.
    let d = ggx_ndf(noh, alpha);
    let g = smith_ggx_correlated(nov, nol, alpha);
    let extinction_factor = exp(-extinction * distance);
    return Vec3::new(d * g * t) * extinction_factor;
}

/// Octahedral normal decode (lighting-only g-buffer unpack).
#[ornis_macros::wgsl_fn]
fn octahedral_decode(p: glam::Vec2) -> glam::Vec3 {
    let mut n = Vec3::new(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    // Unfold the lower hemisphere: each component shifts back by its own
    // sign times `t` (the old code multiplied by the other component, so
    // the whole z<0 hemisphere decoded up to ~69° off).
    let t = max(-n.z, 0.0);
    let neg = Vec2::new(0.0 - t, 0.0 - t);
    let pos = Vec2::new(t, t);
    let offset = select(pos, neg, n.xy >= Vec2::new(0.0));
    n.x = n.x + offset.x;
    n.y = n.y + offset.y;
    return normalize(n);
}

/// World-position reconstruction from depth (lighting-only).
///
/// Depth is stored DirectX-style (NDC z in [0, 1], matching
/// `glam::camera::rh::proj::directx`), so it maps to clip space
/// directly — NOT `depth * 2.0 - 1.0` (that OpenGL-style remap halves
/// distances and breaks every absolute-position use: spot cones,
/// point falloff, shadow projection).
///
/// NDC y is `1.0 - uv.y * 2.0`: rasterization puts NDC y+1 at texture
/// row 0 (the fullscreen quad pairs clip (+1,+1) with uv (1,0)), so a
/// plain `uv * 2.0 - 1.0` reconstructs the vertically mirrored pixel —
/// every world-space use in deferred lighting (falloff, cones, shadow
/// lookups) silently overturns.
#[ornis_macros::wgsl_fn]
fn reconstruct_world_pos(uv: glam::Vec2, depth: f32, camera: Camera) -> glam::Vec3 {
    let ndc = Vec3::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let clip = Vec4::new(ndc, 1.0);
    let view = camera.inv_view_proj * clip;
    return view.xyz / view.w;
}

/// All shared evaluator sources concatenated (consts excluded).
pub fn wgsl_shared_helpers() -> String {
    [
        evaluate_base_layer::wgsl_source(),
        evaluate_coat_layer::wgsl_source(),
        evaluate_coat_darkening::wgsl_source(),
        evaluate_fuzz_layer::wgsl_source(),
        evaluate_transmission_layer::wgsl_source(),
        evaluate_subsurface_layer::wgsl_source(),
        evaluate_emission::wgsl_source(),
        transmission_btdf::wgsl_source(),
    ]
    .join("\n")
}

/// Lighting-only decode helpers concatenated.
pub fn wgsl_lighting_decode() -> String {
    [
        octahedral_decode::wgsl_source(),
        reconstruct_world_pos::wgsl_source(),
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_sources_keep_legacy_signatures() {
        assert!(
            evaluate_base_layer::wgsl_source()
                .starts_with("fn evaluate_base_layer(n: vec3<f32>, v: vec3<f32>")
        );
        assert!(evaluate_coat_layer::wgsl_source().contains("mat: OpenPBRMaterial"));
        assert!(
            octahedral_decode::wgsl_source()
                .starts_with("fn octahedral_decode(p: vec2<f32>) -> vec3<f32>")
        );
        assert!(reconstruct_world_pos::wgsl_source().contains("camera: Camera"));
        // DSL spellings must not leak into WGSL.
        for src in [
            evaluate_base_layer::wgsl_source(),
            transmission_btdf::wgsl_source(),
            octahedral_decode::wgsl_source(),
        ] {
            assert!(!src.contains("Vec3"), "glam spelling leaked: {src}");
            assert!(!src.contains("glam"), "glam spelling leaked: {src}");
        }
    }

    /// Smith visibility already includes `G / (4 NoV NoL)`. Dividing again
    /// saturates ACES along a roughness-0.5 grazing rim (the anim mannequin).
    #[test]
    fn grazing_specular_stays_bounded_when_smith_visibility_is_applied_once() {
        let nov = 1.0e-4;
        let nol = 0.5;
        let alpha = 0.25;
        let visibility = crate::shaders::math::smith_ggx_correlated::eval(nov, nol, alpha);
        let distribution = crate::shaders::math::ggx_ndf_aniso::eval(
            1.0,
            glam::Vec3::Z,
            glam::Vec3::X,
            glam::Vec3::Y,
            alpha,
            alpha,
        );
        let fresnel = 0.04;
        let illuminance = 0.6;
        let divided_again = distribution * visibility * fresnel * illuminance / (4.0 * nov);
        let once = distribution * visibility * fresnel * illuminance * nol;
        let white = crate::shaders::math::aces_tonemap::eval(glam::Vec3::splat(divided_again)).x;
        let bounded = crate::shaders::math::aces_tonemap::eval(glam::Vec3::splat(once)).x;
        assert!(
            white > 0.95,
            "double-divided grazing specular should tonemap to white, got {white} from {divided_again}"
        );
        assert!(
            once < 1.0 && bounded < 0.5,
            "single visibility application stays a modest highlight, got {once} -> {bounded}"
        );
    }

    #[test]
    fn specular_layers_do_not_divide_smith_visibility_again() {
        for (name, src) in [
            ("base", evaluate_base_layer::wgsl_source()),
            ("coat", evaluate_coat_layer::wgsl_source()),
            ("transmission", transmission_btdf::wgsl_source()),
        ] {
            assert!(
                !src.contains("4.0") && !src.contains("4f"),
                "{name} still divides by 4·NoV·NoL:\n{src}"
            );
        }
    }

    /// Severity: medium. The transmission NDF must see `N·H`, not `V·H`.
    #[test]
    fn transmission_ndf_uses_facet_normal() {
        let alpha = 0.05;
        let at_noh = crate::shaders::math::ggx_ndf::eval(0.25, alpha);
        let at_voh = crate::shaders::math::ggx_ndf::eval(1.0, alpha);
        assert!(
            at_voh > at_noh * 5.0,
            "the two cosines disagree: noh={at_noh} voh={at_voh}"
        );
        let src = transmission_btdf::wgsl_source();
        assert!(src.contains("ggx_ndf(noh,"), "{src}");
        assert!(!src.contains("ggx_ndf(voh,"), "{src}");
    }

    /// Severity: medium. Coat darkening is a base multiplier, not extra light.
    #[test]
    fn coat_layer_does_not_add_the_darkening_term() {
        let lobe = evaluate_coat_layer::wgsl_source();
        assert!(
            !lobe.contains("coat_albedo_approx") && !lobe.contains("darkening *"),
            "{lobe}"
        );
        let factor = evaluate_coat_darkening::wgsl_source();
        assert!(factor.contains("coat_blend_darkened("), "{factor}");
        assert!(!factor.contains("if ("), "{factor}");
    }

    #[test]
    fn rust_consts_are_f32_exact() {
        // The former decimal literals and the Rust constants round to the
        // same f32 bits (differences sit far below f32 epsilon).
        assert_eq!(PI.to_bits(), (std::f64::consts::PI as f32).to_bits());
        assert_eq!(EPS.to_bits(), 1e-6f32.to_bits());
        assert_eq!(
            INV_PI.to_bits(),
            (std::f64::consts::FRAC_1_PI as f32).to_bits()
        );
        let consts = wgsl_consts();
        assert!(consts.contains("const PI: f32"));
        assert!(consts.contains("const EPS: f32"));
        assert!(consts.contains("const INV_PI: f32"));
    }
}
