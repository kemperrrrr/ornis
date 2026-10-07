//! UI test: `#[smart_system]` rejects `unsafe fn`.

use ornis_macros::smart_system;

#[smart_system]
unsafe fn unsafe_system(resources: &ornis_core::Resources) {
    let _ = resources.len();
}

fn main() {}
