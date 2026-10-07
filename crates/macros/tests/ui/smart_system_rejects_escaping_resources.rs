//! UI test: `#[smart_system]` rejects `resources` escaping into a helper —
//! the helper's accesses cannot be derived.

use ornis_macros::smart_system;

struct Cfg {
    k: f32,
}

fn helper(resources: &ornis_core::Resources) -> usize {
    resources.get::<Cfg>().map(|c| c.k as usize).unwrap_or(0)
}

#[smart_system]
fn via_helper(resources: &ornis_core::Resources) {
    let _ = helper(resources);
}

fn main() {}
