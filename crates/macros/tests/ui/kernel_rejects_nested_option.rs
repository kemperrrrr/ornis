//! UI test: `kernel` rejects `Option<Option<T>>` (issue #110,
//! feature 3).

use ornis_macros::kernel;

// Nested `Option`s would need a recursive struct layout, which WGSL
// does not have. Flatten to one level.
#[kernel]
fn nested(x: Option<Option<f32>>) -> f32 {
    match x {
        Some(inner) => inner.unwrap_or(0.0),
        None => 0.0,
    }
}

fn main() {}
