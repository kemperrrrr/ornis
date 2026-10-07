//! UI test: `#[smart_system]` rejects two `&Resources` parameters.

use ornis_macros::smart_system;

#[smart_system]
fn two(a: &ornis_core::Resources, b: &ornis_core::Resources) {
    let _ = (a.len(), b.len());
}

fn main() {}
