//! UI test: `#[pack(gpu)]` rejects nested `Option` — the outer
//! `Option<T>` field is already the sparse-column form.

use ornis_macros::Pack;

#[derive(Pack)]
#[pack(gpu)]
struct Nested {
    x: Option<Option<f32>>,
}

fn main() {}
