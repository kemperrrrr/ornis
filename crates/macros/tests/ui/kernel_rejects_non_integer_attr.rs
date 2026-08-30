//! UI test: `kernel` rejects non-integer attribute.

use ornis_macros::kernel;

// `#[kernel]` accepts an optional integer dispatch id; a string literal is
// rejected (expected an integer literal).
#[kernel("not_an_int")]
fn double(x: f32) -> f32 {
    x * 2.0
}

fn main() {}
