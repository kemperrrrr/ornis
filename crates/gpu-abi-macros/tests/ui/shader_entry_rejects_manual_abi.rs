//! UI test: the attribute adds `extern "C"` itself.

use ornis_gpu_abi_macros::compute;

#[compute(workgroup_size = 4)]
extern "C" fn cull(id: u32) {
    let _ = id;
}

fn main() {}
