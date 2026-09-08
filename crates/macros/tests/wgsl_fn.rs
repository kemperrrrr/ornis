//! Integration tests for `#[wgsl_fn]`: plain helper emission without the
//! `@vertex`/`@fragment` wrapper. Input functions are DSL-only — `Mat`,
//! `kernel` and friends intentionally do not exist as Rust items.

use ornis_macros::wgsl_fn;

/// Helper shape: plain params, glam + passthrough types, constructor call.
#[wgsl_fn]
fn mix_helper(n: Vec3, mat: Mat, k: f32) -> Vec3 {
    let base = Vec3::new(1.0);
    return mix(base, n, k);
}

#[test]
fn wgsl_fn_plain_helper_shape() {
    let src = mix_helper::wgsl_source();
    assert!(
        src.starts_with("fn mix_helper(n: vec3<f32>, mat: Mat, k: f32) -> vec3<f32>"),
        "{src}"
    );
    assert!(src.contains("let base = vec3<f32>(1.0);"), "{src}");
    assert!(src.contains("return mix(base, n, k);"), "{src}");
    assert!(
        !src.contains("@vertex") && !src.contains("@fragment"),
        "{src}"
    );
}
