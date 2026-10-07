//! UI test: `#[smart_system]` rejects `&str` parameters — an unsized
//! target cannot be stored in the system struct.

use ornis_macros::smart_system;

#[smart_system]
fn by_ref(resources: &ornis_core::Resources, name: &str) {
    let _ = (resources.len(), name.len());
}

fn main() {}
