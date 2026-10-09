//! UI test: `#[smart_system]` rejects a wrapped interior-mutability
//! resource (`Arc<Mutex<_>>`) without an explicit declaration — a silent
//! `reads` would mislead the level scheduler.

use ornis_macros::smart_system;

struct Counter(u32);

#[smart_system]
fn arc_bump_undeclared(resources: &ornis_core::Resources) {
    resources
        .get::<std::sync::Arc<std::sync::Mutex<Counter>>>()
        .expect("counter")
        .lock()
        .expect("lock")
        .0 += 1;
}

fn main() {}
