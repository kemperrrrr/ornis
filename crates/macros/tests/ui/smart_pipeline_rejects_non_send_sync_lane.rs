//! UI test: `#[smart_pipeline]` (R6) rejects lanes whose type is not
//! `Send + Sync` — parallel iteration would move lane data across threads.

use ornis_core::SmartStore;
use ornis_macros::smart_pipeline;

struct NotSendSync {
    _raw: *mut u8,
}

#[smart_pipeline]
fn integrate_not_send(store: &SmartStore) {
    let mut lane = store.write_lane::<NotSendSync>().expect("lane");
    for x in lane.iter_mut() {
        let _ = x;
    }
}

fn main() {}
