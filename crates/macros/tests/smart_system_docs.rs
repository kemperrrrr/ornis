//! Docs probe (REVIEW-smart_system.md item 7): generated `pub` items carry
//! documentation, so `#[smart_system]` is usable under `#![deny(missing_docs)]`.

#![deny(missing_docs)]

use ornis_core::{Resources, System};
use ornis_macros::smart_system;

/// A documented public system.
#[smart_system]
pub fn documented(resources: &Resources) {
    let _ = resources.len();
}

#[test]
fn generated_items_satisfy_missing_docs() {
    let system = DocumentedSystem::new();
    assert_eq!(system.name(), "documented");
    assert!(system.access().reads.is_empty());
}
