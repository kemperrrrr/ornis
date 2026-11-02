//! UI test: a compute entry takes exactly one `u32` invocation id.

use ornis_macros::compute;

#[compute(workgroup_size = 4)]
fn cull(id: f32) {
    let _ = id;
}

fn main() {}
