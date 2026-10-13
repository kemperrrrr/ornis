//! UI test: `kernel` rejects closures (WGSL has none).

use ornis_macros::kernel;

// A closure used to slip through the validator while the lowerer
// smuggled its `compile_error!` text into the WGSL string, failing
// only at naga. Reject it at macro time instead — inline the
// computation.
#[kernel]
fn closure_u8(x: f32) -> f32 {
    let f = |y: u8| y;
    x + (f(0) as f32)
}

fn main() {}
