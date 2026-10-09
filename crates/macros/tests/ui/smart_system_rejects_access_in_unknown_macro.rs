//! UI test: `#[smart_system]` rejects `resources` used inside an unknown
//! macro invocation — the access cannot be derived.

use ornis_macros::smart_system;

#[allow(unused_macros)]
macro_rules! check_present {
    ($resources:expr) => {
        $resources.len() > 0
    };
}

#[smart_system]
fn in_unknown_macro(resources: &ornis_core::Resources) {
    let _ = check_present!(resources);
}

fn main() {}
