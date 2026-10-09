//! UI test: `#[smart_system]` rejects a store ident escaping into a helper —
//! the helper's lane accesses cannot be derived.

use ornis_macros::smart_system;

struct V {
    x: f32,
}

fn lane_helper(store: &ornis_core::SmartStore) -> usize {
    store.read_lane::<V>().map(|lane| lane.len()).unwrap_or(0)
}

#[smart_system]
fn store_escape(resources: &ornis_core::Resources) {
    let store = resources.get::<ornis_core::SmartStore>().expect("store");
    let _ = lane_helper(store);
}

fn main() {}
