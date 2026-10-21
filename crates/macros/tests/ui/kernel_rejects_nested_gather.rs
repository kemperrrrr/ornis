//! UI test: `kernel` rejects nested entity field reads (RAT-49).
//!
//! `t.position.x` cannot carry one annotation for two projections —
//! bind the field first, then use the component.

use ornis_macros::kernel;

#[kernel]
fn nested(t: Entity) -> f32 {
    let px: f32 = t.position.x;
    px
}

fn main() {}
