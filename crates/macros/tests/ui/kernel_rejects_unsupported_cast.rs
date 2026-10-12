//! UI test: `kernel` rejects casts without a WGSL spelling.

use ornis_macros::kernel;

// `bool` has no conversion constructor (`f32(x)`/`u32(x)`/`i32(x)` are
// the only supported targets), so the cast must fail at macro time
// instead of reaching naga as WGSL text.
#[kernel]
fn cast_bool(x: f32) -> f32 {
    let flag = x as bool;
    return flag as f32;
}

// Narrow targets would need masking to match Rust's wrapping (`300u32
// as u8 == 44` on CPU vs `300` on GPU), so they are rejected too.
#[kernel]
fn cast_u8(x: u32) -> u32 {
    let narrow = x as u8;
    return narrow as u32;
}

// Wide targets would silently narrow to 32 bits, so they are rejected.
#[kernel]
fn cast_f64(x: f32) -> f32 {
    let wide = x as f64;
    return wide as f32;
}

fn main() {}
