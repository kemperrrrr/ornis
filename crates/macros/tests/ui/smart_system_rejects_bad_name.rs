//! UI test: `#[smart_system]` rejects a non-string `name`.

use ornis_macros::smart_system;

#[smart_system(name = 1)]
fn bad_name(resources: &ornis_core::Resources) {
    let _ = resources.len();
}

fn main() {}
