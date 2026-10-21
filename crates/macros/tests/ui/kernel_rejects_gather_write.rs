//! UI test: `kernel` rejects writes to entity fields (RAT-49).
//!
//! Foreign entity fields are read-only gathers; WGSL has no write
//! spelling for a field of another component's storage lane.

use ornis_macros::kernel;

#[kernel]
fn write_field(t: Entity) {
    t.position = glam::Vec3::ZERO;
}

fn main() {}
