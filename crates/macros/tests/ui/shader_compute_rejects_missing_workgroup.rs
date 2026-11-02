//! UI test: `#[compute]` requires a workgroup size.

use ornis_macros::compute;

#[compute]
fn cull(id: u32) {
    let _ = id;
}

fn main() {}
