//! UI test: `#[smart_system]` rejects a bare `name` without value.

use ornis_macros::smart_system;

#[smart_system(name)]
fn bare_name(resources: &ornis_core::Resources) {
    let _ = resources.len();
}

fn main() {}
