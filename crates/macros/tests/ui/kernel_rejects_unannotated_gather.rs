//! UI test: `kernel` rejects un-annotated entity field reads (RAT-49).
//!
//! Macros cannot see component struct definitions, so a gather's type
//! must come from the use-site annotation — a bare `t.position` has no
//! known lane type and no WGSL spelling.

use ornis_macros::kernel;

#[kernel]
fn unannotated(t: Entity) -> f32 {
    let p = t.position;
    p.x
}

fn main() {}
