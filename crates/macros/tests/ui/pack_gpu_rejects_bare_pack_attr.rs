//! UI test: on `#[derive(Pack)]`, `pack(...)` has exactly one spelling
//! — `#[pack(gpu)]`; bare `#[pack]` fails instead of silently staying
//! CPU-only.

use ornis_macros::Pack;

#[derive(Pack)]
#[pack]
struct Bare {
    x: f32,
}

fn main() {}
