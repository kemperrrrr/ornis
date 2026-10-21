//! UI test: `kernel` rejects `bool` gather annotations (RAT-49).
//!
//! WGSL storage arrays cannot hold `bool`; the lane element must be a
//! registry scalar/vector (use `u32` for flags) or `Entity`.

use ornis_macros::kernel;

#[kernel]
fn bool_lane(t: Entity) -> bool {
    let alive: bool = t.alive;
    alive
}

fn main() {}
