//! UI test: `kernel` rejects casts without a WGSL spelling.

use ornis_macros::kernel;

// `bool` has no conversion constructor (`u32(x)`/`i32(x)`/`f32(x)` are
// the only supported targets), so the cast must fail at macro time
// instead of reaching naga as WGSL text.
#[kernel]
fn cast_bool(x: f32) -> f32 {
    let flag = x as bool;
    return flag as f32;
}

fn main() {}
