//! UI test: an explicit `writes(...)` for one type does not blanket-approve
//! a `resources` escape into a helper reading another type — that needs the
//! `opaque` opt-in with every touched resource listed.

use ornis_macros::smart_system;

struct Counter(u32);

struct Cfg {
    k: f32,
}

fn cfg_helper(resources: &ornis_core::Resources) -> f32 {
    resources.get::<Cfg>().map(|cfg| cfg.k).unwrap_or(0.0)
}

#[smart_system(writes(std::sync::Mutex<Counter>))]
fn partial_explicit(resources: &ornis_core::Resources) {
    resources
        .get::<std::sync::Mutex<Counter>>()
        .expect("counter")
        .lock()
        .expect("lock")
        .0 += 1;
    let _ = cfg_helper(resources);
}

fn main() {}
