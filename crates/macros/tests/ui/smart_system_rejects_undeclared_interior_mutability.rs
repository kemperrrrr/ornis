//! UI test: `#[smart_system]` rejects an undeclared interior-mutability
//! resource — a silent `reads` would mislead the level scheduler.

use ornis_macros::smart_system;

struct Counter(u32);

#[smart_system]
fn bump_undeclared(resources: &ornis_core::Resources) {
    resources
        .get::<std::sync::Mutex<Counter>>()
        .expect("counter")
        .lock()
        .expect("lock")
        .0 += 1;
}

fn main() {}
