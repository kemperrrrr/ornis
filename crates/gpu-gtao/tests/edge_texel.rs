//! `uv == 1` must load the last depth texel, matching master
//! `min(u32(uv * dims), dims - 1)`.
//!
//! The unclamped cast is `dim`, which is outside the texture. A backend that
//! returns zero for that load is not the edge texel the hand-written horizon
//! read.

use ornis_gpu_gtao::clamped_texel;
use ornis_gpu_gtao::homogeneous_divide;
use ornis_gpu_shader::dot2;

/// Master's interleaved-gradient weights.
const HASH_X: f32 = 0.06711056;
const HASH_Y: f32 = 0.00583715;
const HASH_Z: f32 = 52.982_918;

/// Master's `min(u32(uv * dims), u32(dims) - 1)` for one axis.
fn master_texel(uv: f32, dim: f32) -> u32 {
    let raw = (uv * dim) as u32;
    let last = (dim as u32).saturating_sub(1);
    raw.min(last)
}

#[test]
fn uv_one_clamps_to_the_last_texel() {
    for dim in [1.0_f32, 2.0, 64.0, 128.0, 320.0, 960.0] {
        let last = (dim as u32) - 1;
        assert_eq!(
            (1.0_f32 * dim) as u32,
            dim as u32,
            "unclamped uv=1 is out of bounds"
        );
        assert_eq!(clamped_texel(1.0, dim), last);
        assert_eq!(clamped_texel(1.0, dim), master_texel(1.0, dim));
        assert_eq!(clamped_texel(0.0, dim), 0);
        let mid = 0.5;
        assert_eq!(clamped_texel(mid, dim), master_texel(mid, dim));
        let almost = 1.0 - f32::EPSILON;
        assert_eq!(clamped_texel(almost, dim), master_texel(almost, dim));
        assert!(
            clamped_texel(almost, dim) < dim as u32,
            "a uv just below 1 stays inside the texture"
        );
    }
}

#[test]
fn ssao_master_does_not_clamp_uv_one() {
    // The hand-written SSAO tap is `textureLoad(..., vec2<u32>(suv * dims))`
    // after rejecting only `suv > 1`. At exactly 1 the index is out of bounds
    // on master; the port keeps that cast.
    let dim = 64.0_f32;
    let suv = 1.0_f32;
    let index = (suv * dim) as u32;
    assert_eq!(index, dim as u32);
    assert_ne!(index, clamped_texel(suv, dim));
}

/// `dot(vec2(pixel), magic.xy)` contracts as `fma(y, cy, x * cx)`.
fn master_ign_dot(x: f32, y: f32) -> f32 {
    y.mul_add(HASH_Y, x * HASH_X)
}

fn scalar_ign_sum(x: f32, y: f32) -> f32 {
    x * HASH_X + y * HASH_Y
}

fn fract(value: f32) -> f32 {
    value - value.floor()
}

fn master_ign(pixel_x: u32, pixel_y: u32) -> f32 {
    let dot = master_ign_dot(pixel_x as f32, pixel_y as f32);
    fract(HASH_Z * fract(dot))
}

#[test]
fn ign_dot_matches_master_and_not_the_scalar_sum() {
    let mut dot_differs = false;
    let mut hash_differs = false;
    for y in 0..128u32 {
        for x in 0..128u32 {
            let xf = x as f32;
            let yf = y as f32;
            let dot = dot2(xf, yf, HASH_X, HASH_Y);
            assert_eq!(dot.to_bits(), master_ign_dot(xf, yf).to_bits());
            let scalar = scalar_ign_sum(xf, yf);
            if scalar.to_bits() != dot.to_bits() {
                dot_differs = true;
                let scalar_hash = fract(HASH_Z * fract(scalar));
                if scalar_hash.to_bits() != master_ign(x, y).to_bits() {
                    hash_differs = true;
                }
            }
        }
    }
    assert!(
        dot_differs,
        "scalar sum and vec2 dot must disagree on some texel"
    );
    assert!(
        hash_differs,
        "that disagreement must move the interleaved-gradient hash"
    );
}

#[test]
fn reconstruction_divides_by_raw_w_below_eps() {
    let eps = 1.0e-6_f32;
    let value = 3.5_f32;
    for w in [eps * 0.5, -eps * 0.25, -0.5, eps, 0.25] {
        let got = homogeneous_divide(value, w);
        assert_eq!(got.to_bits(), (value / w).to_bits(), "w={w}");
        if w < eps {
            let clamped = value / w.max(eps);
            assert_ne!(got.to_bits(), clamped.to_bits(), "w={w}");
        }
    }
}
