//! UI test: `#[compute]` requires a workgroup size.

use ornis_gpu_abi_macros::compute;

#[compute]
fn cull(id: u32) {
    let _ = id;
}

fn main() {}
