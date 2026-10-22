//! UI test: `kernel` rejects struct patterns (issue #110, feature 3).

use ornis_macros::kernel;

// A struct pattern has no WGSL spelling in `#[kernel]`; match on
// scalars or `Option` payloads instead.
#[kernel]
fn structured(x: f32) -> f32 {
    match x {
        f32 { .. } => x,
    }
}

fn main() {}
