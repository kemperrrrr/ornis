//! UI test: `workgroup_size` must be positive.

use ornis_gpu_abi_macros::compute;

#[compute(workgroup_size = 0)]
fn cull(id: u32) {
    let _ = id;
}

fn main() {}
