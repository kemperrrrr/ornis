//! UI test: `kernel` rejects `==`/`!=` on whole `Entity` values
//! (RAT-49).
//!
//! WGSL has no struct equality; compare `.index` (and `.generation`)
//! explicitly.

use ornis_macros::kernel;

#[kernel]
fn entity_eq(t: Entity, other: Entity) -> u32 {
    if t == other { 1u32 } else { 0u32 }
}

fn main() {}
