//! UI test: `kernel` rejects namespaced scalar paths.

use ornis_macros::kernel;

// A scalar type hidden below `core`/`std` has no WGSL spelling either:
// `is_scalar_namespaced` checks every path segment but the last, so
// these used to slip through the validator and fail only at naga.
// Use a plain `f32`, `u32` or `i32` literal instead.
#[kernel]
fn core_u8_max(x: f32) -> f32 {
    x + core::u8::MAX as u32 as f32
}

#[kernel]
fn std_primitive_u8_max(x: f32) -> f32 {
    x + std::primitive::u8::MAX as u32 as f32
}

#[kernel]
fn std_f32_const_pi(x: f32) -> f32 {
    x + std::f32::consts::PI
}

fn main() {}
