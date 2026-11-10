//! `uv == 1` must load the last depth texel, matching master
//! `min(u32(uv * dims), dims - 1)`.
//!
//! The unclamped cast is `dim`, which is outside the texture. A backend that
//! returns zero for that load is not the edge texel the hand-written horizon
//! read.

use ornis_gpu_gtao::clamped_texel;

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
