//! UI test: `#[smart_system]` rejects generic functions — the generated
//! struct cannot carry type parameters.

use ornis_macros::smart_system;

#[smart_system]
fn generic_system<T>(resources: &ornis_core::Resources, value: T) {
    let _ = (resources.len(), value);
}

fn main() {}
