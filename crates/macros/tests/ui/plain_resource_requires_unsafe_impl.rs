//! UI test: hand-written PlainResource implementations must be explicitly unsafe.

use ornis_core::{PlainResource, ResMutResource};

struct Shared;

unsafe impl ResMutResource for Shared {}
impl PlainResource for Shared {}

fn main() {
    let _ = Shared;
}
