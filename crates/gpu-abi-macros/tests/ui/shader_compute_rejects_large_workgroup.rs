//! UI test: `workgroup_size` is capped at 256 invocations.

use ornis_gpu_abi_macros::compute;

#[compute(workgroup_size = 257)]
fn cull(id: u32) {
    let _ = id;
}

fn main() {}
