//! UI test: entry parameters are limited to wasm scalars.

use ornis_gpu_abi_macros::vertex;

#[vertex]
fn quad(v: u64) {
    let _ = v;
}

fn main() {}
