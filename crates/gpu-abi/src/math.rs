//! Scalar math helpers that compile to WebAssembly core float operations.

/// Return the smaller of two finite scalar values.
#[inline(always)]
pub fn gpu_min_f32(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// Return the larger of two finite scalar values.
#[inline(always)]
pub fn gpu_max_f32(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Clamp a finite value to finite, ordered bounds.
///
/// The lower bound must not exceed the upper bound.
#[inline(always)]
pub fn gpu_clamp_f32(value: f32, low: f32, high: f32) -> f32 {
    value.clamp(low, high)
}
