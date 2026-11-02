//! UI test: entries cannot be generic or `async`.

use ornis_gpu_abi_macros::{fragment, vertex};

#[vertex]
fn quad<T>(x: f32) -> f32 {
    x
}

#[fragment]
async fn shade(u: f32) -> f32 {
    u
}

fn main() {}
