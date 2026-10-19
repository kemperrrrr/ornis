//! Process-wide clock that versions entity-sequence layouts.
//!
//! [`ComponentStore::layout`](crate::component_store::ComponentStore::layout)
//! takes a fresh value from here on every membership or order change, so a
//! cached alignment check can be revalidated by comparing two integers.

use std::sync::atomic::{AtomicU64, Ordering};

static LAYOUT_CLOCK: AtomicU64 = AtomicU64::new(0);

/// Returns a value never handed out before in this process.
pub(crate) fn next_layout() -> u64 {
    LAYOUT_CLOCK.fetch_add(1, Ordering::Relaxed) + 1
}
