//! Behavioural checks on the ported layers.
//!
//! These are not the numerical gate — that is the pass's pixel parity against
//! the frozen reference, since the reference bodies are DSL-only. They pin the
//! contracts the layer stack promises its callers: a layer whose weight is zero
//! contributes nothing, the sanitizer and the debug views only touch what they
//! say they touch, and the split alpha pair agrees with the pair the host math
//! exposes.

use super::*;
use crate::math::anisotropy_alpha_u;
use crate::math::anisotropy_alpha_v;
use crate::math::openpbr_anisotropy;
use crate::math::splat3;
use crate::math::v3;
use ornis_gpu_shader::Vec3;

/// The terms the per-sample layers all take, in a configuration nothing
/// degenerates: front-facing, normal-incidence-ish.
fn terms() -> (Vec3, Vec3, Vec3, Vec3) {
    let n = v3(0.0, 0.0, 1.0);
    let v = v3(0.0, 0.0, 1.0);
    let l = v3(0.6, 0.0, 0.8);
    let h = v3(0.3, 0.0, 0.95);
    (n, v, l, h)
}

/// Lanes of `value`, for the per-lane assertions below.
fn lanes(value: Vec3) -> [f32; 3] {
    [value.x, value.y, value.z]
}

#[test]
fn a_zero_weight_layer_contributes_nothing() {
    let (n, v, l, h) = terms();
    let color = v3(0.4, 0.5, 0.6);
    let zero = [0.0_f32; 3];

    let t = v3(1.0, 0.0, 0.0);
    let b = v3(0.0, 1.0, 0.0);
    let coat = evaluate_coat_layer(
        n, v, l, h, 1.0, 0.8, 0.5, 0.9, // per-sample terms
        0.0, 0.5, 0.0, 1.0, 1.5, color, // coat fields, weight off
        0.0, color, 1.0, 1.0, 0.0, color, // base fields the darkening reads
        t, b,
    );
    assert_eq!(lanes(coat), zero);

    let fuzz = evaluate_fuzz_layer(n, v, l, h, 1.0, 0.8, 0.5, 0.9, 0.0, 0.5, color);
    assert_eq!(lanes(fuzz), zero);

    let transmission = evaluate_transmission_layer(
        n, v, l, h, 1.0, 0.8, 0.5, 0.9, // per-sample terms
        0.0, 1.0, 1.0, 1.0, color, color, 0.0, // transmission fields, weight off
        1.5, 0.5, 0.0, 0.0,
    );
    assert_eq!(lanes(transmission), zero);

    let subsurface =
        evaluate_subsurface_layer(n, v, l, 1.0, 0.8, 0.0, 0.1, 1.0, 1.0, 1.0, 0.0, color);
    assert_eq!(lanes(subsurface), zero);
}

#[test]
fn no_coat_leaves_the_base_undarkened() {
    let color = v3(0.4, 0.5, 0.6);
    let darkening = evaluate_coat_darkening(0.0, 1.0, 1.5, 0.0, color, 1.0, 1.0, 0.0, color);
    assert_eq!(lanes(darkening), [1.0, 1.0, 1.0]);
}

#[test]
fn hdr_sanitize_keeps_finite_colors() {
    assert_eq!(
        lanes(sanitize_hdr(v3(0.5, 12.0, 300.0))),
        [0.5, 12.0, 300.0]
    );
}

#[test]
fn hdr_sanitize_zeroes_one_bad_channel_and_keeps_its_siblings() {
    // Per-channel selection: a magnitude past the limit zeroes that channel
    // and nothing else. `1e21` rather than an infinity keeps the assertion
    // about the select and not about IEEE `∞ · 0`.
    assert_eq!(
        lanes(sanitize_hdr(v3(1e21, 12.0, 300.0))),
        [0.0, 12.0, 300.0]
    );
}

#[test]
fn hdr_sanitize_scrubs_a_nan_channel() {
    // Masking could not: `NaN * 0` is `NaN`. Selecting the replacement returns
    // `0.0` untouched, which is the point of the form.
    let sanitized = lanes(sanitize_hdr(v3(f32::NAN, 12.0, 300.0)));
    assert_eq!(sanitized, [0.0, 12.0, 300.0]);
}

#[test]
fn specular_aa_without_a_neighbour_returns_the_input_roughness() {
    // The host arms of `dpdx`/`dpdy` are zero: a single pixel has no
    // neighbour, so the variance term vanishes and the double square root
    // undoes the squaring the BRDF would do.
    let n = v3(0.0, 0.0, 1.0);
    for roughness in [0.05_f32, 0.3, 0.9] {
        let widened = specular_aa_roughness(roughness, n);
        assert!(
            (widened - roughness).abs() < 1e-6,
            "roughness {roughness} became {widened}"
        );
    }
}

#[test]
fn each_debug_view_returns_its_own_term() {
    let beauty = v3(0.1, 0.2, 0.3);
    let other = v3(9.0, 9.0, 9.0);
    let select = |view: u32| {
        select_shading_debug(view, beauty, other, other, other, 0.25, 0.75, 0.5, beauty)
    };
    assert_eq!(lanes(select(debug_view::BEAUTY)), lanes(beauty));
    assert_eq!(lanes(select(debug_view::ROUGHNESS)), [0.25; 3]);
    assert_eq!(lanes(select(debug_view::METALLIC)), [0.75; 3]);
    assert_eq!(lanes(select(debug_view::SHADOW)), [0.5; 3]);
    assert_eq!(lanes(select(debug_view::DIFFUSE)), lanes(other));
}

#[test]
fn emission_without_a_coat_is_luminance_times_color() {
    let color = v3(0.2, 0.4, 0.8);
    let emission = lanes(evaluate_emission(3.0, color, 0.0, color, 1.0));
    assert_eq!(emission, [0.6, 1.2, 2.4]);
}

#[test]
fn the_alpha_pair_agrees_with_the_two_scalars() {
    for roughness in [0.1_f32, 0.5, 0.95] {
        for anisotropy in [-0.8_f32, 0.0, 0.6] {
            let pair = openpbr_anisotropy(roughness, anisotropy);
            assert_eq!(pair.0, anisotropy_alpha_u(roughness, anisotropy));
            assert_eq!(pair.1, anisotropy_alpha_v(roughness, anisotropy));
            assert_eq!(pair.0, splat3(pair.0).x);
        }
    }
}
