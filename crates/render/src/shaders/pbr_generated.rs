//! Forward-PBR shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module: the full OpenPBR
//! fragment skeleton (layer evaluators + `fs_main`) lives here as a Rust
//! string, and the 19 BRDF math kernels are spliced in from
//! [`crate::shaders::math`] (single source of truth via `#[kernel]`).
//! The former handwritten `shaders/wgsl/pbr_*.wgsl` sources were deleted
//! after the `#[stage]` translation of `fs_main`; the
//! `pbr_fragment_entry_matches_legacy_shape` test pins the entry shape.
//!
//! Note: the vertex stage is shared with the g-buffer pass (same instance
//! transform), so [`wgsl_vertex_source`] reuses
//! [`crate::shaders::gbuffer_generated::wgsl_vertex_source`] instead of
//! duplicating it.

use super::interface::GbufferFragmentInput as FragmentInput;
use super::{binding, helpers, openpbr_material_decl, wgsl_decl};
use crate::renderer::{CameraUniform, GpuLight, LightingUniform};
use crate::shaders::{gbuffer_generated, math};
use ornis_macros::stage;

/// Forward-PBR vertex shader: instance transforms + world position.
///
/// Delegates to [`gbuffer_generated::wgsl_vertex_source`] — the two legacy
/// vertex files are byte-identical. Entry point name `vs_main` is kept for
/// compatibility with `create_pbr_pass`.
pub fn wgsl_vertex_source() -> String {
    gbuffer_generated::wgsl_vertex_source()
}

/// Forward-PBR fragment shader: full OpenPBR evaluation.
///
/// Assembled as `{skeleton}\\n{kernel × 19}`, exactly like the legacy
/// `shaders::pbr_fragment()`; entry point `fs_main` is kept.
pub fn wgsl_source() -> String {
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
    let mut src = format!(
        "\n{cam}\n{light}\n{lighting}{mat}\n{restA}\n{fin}\n{consts}\n{helpers}\n{entry}",
        cam = CameraUniform::WGSL_SOURCE,
        light = wgsl_decl(GpuLight::WGSL_SOURCE),
        lighting = wgsl_decl(LightingUniform::WGSL_SOURCE),
        mat = openpbr_material_decl(),
        restA = fragment_bindings(),
        fin = wgsl_decl(FragmentInput::WGSL_SOURCE),
        consts = helpers::wgsl_consts(),
        helpers = helpers::wgsl_shared_helpers(),
        entry = pbr_fragment_entry::wgsl_source(),
    );
    for k in &kernels {
        src.push('\n');
        src.push_str(k);
    }
    src
}

/// Forward-PBR fragment entry, translated by [`stage`](ornis_macros::stage):
/// full OpenPBR evaluation over the light array. DSL-only — free
/// `camera`/`lighting`/`materials` binding identifiers. Layer evaluators
/// stay handwritten below and are called by name.
#[stage(fragment, entry = "fs_main", returns = "@location(0) vec4<f32>")]
fn pbr_fragment_entry(input: FragmentInput) -> glam::Vec4 {
    let mat = materials[input.material_index];
    let n = normalize(input.world_normal);
    let v = normalize(camera.camera_pos.xyz - input.world_position);
    let nov = max(dot(n, v), EPS);
    let t = normalize(input.world_tangent);
    let b = cross(n, t);
    let base_weight = mat.base_params.x;
    let base_color = mat.base_color.rgb;
    let metalness = mat.base_params.z;
    let diffuse_roughness = mat.base_params.y;
    let specular_weight = mat.specular_params.x;
    let specular_roughness = mat.specular_params.y;
    let specular_ior = mat.specular_params.z;
    let specular_anisotropy = mat.specular_params.w;
    let specular_edge_tint = mat.specular_color.rgb;
    let transmission_weight = mat.transmission_params.x;
    let transmission_depth = mat.transmission_params.y;
    let transmission_dispersion_scale = mat.transmission_params.z;
    let transmission_dispersion_abbe = mat.transmission_params.w;
    let transmission_color = mat.transmission_color.rgb;
    let transmission_scatter = mat.transmission_scatter.rgb;
    let transmission_scatter_anisotropy = mat.transmission_scatter.a;
    let subsurface_weight = mat.subsurface_params.x;
    let subsurface_radius = mat.subsurface_params.y;
    let subsurface_radius_scale_r = mat.subsurface_params.z;
    let subsurface_scatter_anisotropy = mat.subsurface_params.w;
    let subsurface_color = mat.subsurface_color.rgb;
    let subsurface_radius_scale_g = mat.subsurface_radius_scale_gb.x;
    let subsurface_radius_scale_b = mat.subsurface_radius_scale_gb.y;
    let fuzz_weight = mat.fuzz_params.x;
    let fuzz_roughness = mat.fuzz_params.y;
    let fuzz_color = mat.fuzz_color.rgb;
    let coat_weight = mat.coat_params.x;
    let coat_roughness = mat.coat_params.y;
    let coat_anisotropy = mat.coat_params.z;
    let coat_darkening = mat.coat_params.w;
    let coat_color = mat.coat_color.rgb;
    let coat_ior = mat.coat_ior.x;
    let thin_film_weight = mat.thin_film_params.x;
    let thin_film_thickness_um = mat.thin_film_params.y;
    let thin_film_ior = mat.thin_film_params.z;
    let emission_luminance = mat.emission_params.x;
    let emission_color = mat.emission_color.rgb;
    let opacity = mat.geometry_params.x;
    let thin_walled = mat.geometry_params.y;
    let mut lo = Vec3::new(0.0, 0.0, 0.0);
    let thin_film_mod = thin_film_modulation(nov, thin_film_ior, thin_film_thickness_um, 1.0);
    for i in 0u..lighting.light_count {
        let l = normalize(lighting.lights[i].direction.xyz);
        let h = normalize(v + l);
        let light_color = lighting.lights[i].color.rgb;
        let intensity = lighting.lights[i].color.w;
        let radiance = light_color * intensity;
        let nol = max(dot(n, l), EPS);
        let noh = max(dot(n, h), EPS);
        let voh = max(dot(v, h), EPS);
        if nol <= EPS {
            continue;
        }
        let base_bsdf = evaluate_base_layer(
            n,
            v,
            l,
            h,
            nov,
            nol,
            noh,
            voh,
            mat,
            base_weight,
            base_color,
            metalness,
            diffuse_roughness,
            specular_weight,
            specular_roughness,
            specular_ior,
            specular_anisotropy,
            specular_edge_tint,
            t,
            b,
            thin_film_mod,
        );
        let coat_bsdf = evaluate_coat_layer(
            n,
            v,
            l,
            h,
            nov,
            nol,
            noh,
            voh,
            mat,
            coat_weight,
            coat_roughness,
            coat_anisotropy,
            coat_darkening,
            coat_ior,
            coat_color,
            metalness,
            base_color,
            base_weight,
            specular_weight,
            subsurface_weight,
            subsurface_color,
            t,
            b,
        );
        let fuzz_bsdf = evaluate_fuzz_layer(
            n,
            v,
            l,
            h,
            nov,
            nol,
            noh,
            voh,
            fuzz_weight,
            fuzz_roughness,
            fuzz_color,
        );
        let trans_bsdf = evaluate_transmission_layer(
            n,
            v,
            l,
            h,
            nov,
            nol,
            noh,
            voh,
            mat,
            transmission_weight,
            transmission_depth,
            transmission_dispersion_scale,
            transmission_dispersion_abbe,
            transmission_color,
            transmission_scatter,
            transmission_scatter_anisotropy,
            specular_ior,
            specular_roughness,
            specular_anisotropy,
            thin_walled,
        );
        let ss_bsdf = evaluate_subsurface_layer(
            n,
            v,
            l,
            nov,
            nol,
            subsurface_weight,
            subsurface_radius,
            subsurface_radius_scale_r,
            subsurface_radius_scale_g,
            subsurface_radius_scale_b,
            subsurface_scatter_anisotropy,
            subsurface_color,
        );
        let layer_bsdf = base_bsdf + coat_bsdf + fuzz_bsdf + trans_bsdf + ss_bsdf;
        lo = lo + layer_bsdf * radiance * nol;
    }
    let ambient =
        lighting.ambient_color.rgb * mix(base_color, base_color * specular_weight, metalness);
    let emission = evaluate_emission(
        emission_luminance,
        emission_color,
        coat_weight,
        coat_color,
        nov,
    );
    let color = ambient + lo + emission;
    let tone_mapped = aces_tonemap(color);
    return glam::Vec4::new(tone_mapped, opacity);
}

/// Static view for naga validation in tests.
pub fn wgsl_source_static() -> String {
    wgsl_source()
}

/// Fragment resource bindings, assembled from Rust.
fn fragment_bindings() -> String {
    let mut out = String::new();
    out.push_str(&binding(0, 0, "var<uniform> camera: Camera"));
    out.push_str(&binding(
        0,
        2,
        "var<storage, read> materials: array<OpenPBRMaterial>",
    ));
    out.push_str(&binding(0, 3, "var<uniform> lighting: Lighting"));
    out
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
    fn pbr_generated_validates_with_naga() {
        assert_valid_wgsl("pbr_vertex", &wgsl_vertex_source());
        assert_valid_wgsl("pbr_fragment", &wgsl_source());
    }

    #[test]
    fn pbr_generated_contains_expected_bindings() {
        let vs = wgsl_vertex_source();
        assert!(vs.contains("@group(0) @binding(1) var<storage, read> per_objects"));
        assert!(vs.contains("fn vs_main("));
        let fs = wgsl_source();
        assert!(fs.contains("@group(0) @binding(3) var<uniform> lighting"));
        assert!(fs.contains("fn fs_main("));
        assert!(fs.contains("fn evaluate_base_layer"));
        assert!(fs.contains("fn aces_tonemap"));
    }

    /// The translated fragment entry must keep the legacy shape: same
    /// signature, same layer-evaluator calls, same light loop with the
    /// early-out. (Byte-parity no longer applies — the generated entry is
    /// single-line with normalized int suffixes.)
    #[test]
    fn pbr_fragment_entry_matches_legacy_shape() {
        let entry = pbr_fragment_entry::wgsl_source();
        assert!(entry.starts_with("@fragment\nfn fs_main(input: FragmentInput)"));
        assert!(entry.contains("-> @location(0) vec4<f32>"));
        assert!(entry.contains("let mat = materials[input.material_index];"));
        assert!(entry.contains("for (var i: u32 = 0; i < lighting.light_count; i = i + 1)"));
        assert!(entry.contains("continue;"));
        assert!(entry.contains("let base_bsdf = evaluate_base_layer("));
        assert!(entry.contains(
            "let layer_bsdf = base_bsdf + coat_bsdf + fuzz_bsdf + trans_bsdf + ss_bsdf;"
        ));
        assert!(entry.contains("return vec4<f32>(tone_mapped, opacity);"));
    }

    #[test]
    fn pbr_generated_parity_with_legacy_assembly() {
        // Vertex is shared with gbuffer (translated — see
        // `gbuffer_vertex_entry_matches_legacy_shape`); pin the sharing.
        assert_eq!(
            wgsl_vertex_source(),
            super::super::gbuffer_generated::wgsl_vertex_source()
        );
        // Fragment layouts still splice the derived declarations (Camera
        // raw — that declaration terminates with bare `}`).
        let src = wgsl_source();
        assert!(src.contains(CameraUniform::WGSL_SOURCE));
        assert!(src.contains(&wgsl_decl(GpuLight::WGSL_SOURCE)));
        assert!(src.contains(&wgsl_decl(LightingUniform::WGSL_SOURCE)));
        assert!(src.contains(openpbr_material_decl().as_str()));
    }
}
