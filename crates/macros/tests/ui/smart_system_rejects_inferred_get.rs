//! UI test: `#[smart_system]` rejects `resources.get()` without turbofish —
//! the type cannot be derived for the access set.

use ornis_macros::smart_system;

struct Cfg {
    k: f32,
}

#[smart_system]
fn inferred(resources: &ornis_core::Resources) {
    let cfg: &Cfg = resources.get().expect("cfg");
    let _ = cfg.k;
}

fn main() {}
