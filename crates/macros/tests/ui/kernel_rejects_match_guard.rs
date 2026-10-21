//! UI test: `kernel` rejects match guards (issue #110, feature 3).

use ornis_macros::kernel;

// A match guard (`pat if cond`) has no WGSL spelling: WGSL has no
// per-arm predicate. Restructure with an explicit `if` in the body, or
// split into an extra arm.
#[kernel]
fn guarded(x: Option<f32>) -> f32 {
    match x {
        Some(v) if v > 0.0 => v,
        _ => 0.0,
    }
}

fn main() {}
