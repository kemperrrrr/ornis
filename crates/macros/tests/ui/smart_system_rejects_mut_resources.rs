//! UI test: `#[smart_system]` rejects `&mut Resources` — `run` receives
//! `&Resources`.

use ornis_macros::smart_system;

#[smart_system]
fn mut_resources(resources: &mut ornis_core::Resources) {
    let _ = resources.len();
}

fn main() {}
