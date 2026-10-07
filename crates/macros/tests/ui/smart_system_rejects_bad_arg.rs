//! UI test: `#[smart_system]` rejects unknown arguments.

use ornis_macros::smart_system;

#[smart_system(mode = "fast")]
fn bad_arg(resources: &ornis_core::Resources) {
    let _ = resources.len();
}

fn main() {}
