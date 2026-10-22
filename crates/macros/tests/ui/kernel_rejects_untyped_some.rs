//! UI test: `kernel` requires a type context for `Some`/`None`
//! (issue #110, feature 3).

use ornis_macros::kernel;

// `Some`/`None` lower to a generated two-field struct, whose name comes
// from a static `Option<T>` type context: a `let` annotation or the
// function's return type. Without one there is nothing to name.
#[kernel]
fn untyped() -> f32 {
    let a = Some(1.0);
    let b = None;
    a.unwrap_or(b.unwrap_or(0.0))
}

fn main() {}
