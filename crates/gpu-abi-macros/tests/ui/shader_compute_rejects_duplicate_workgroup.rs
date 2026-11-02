//! UI test: a repeated `workgroup_size` is an error on the repeat.

use ornis_gpu_abi_macros::compute;

#[compute(workgroup_size = 4, workgroup_size = 8)]
fn cull(id: u32) {
    let _ = id;
}

fn main() {}
