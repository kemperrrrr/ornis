//! UI test: vertex and fragment entries take no arguments.

use ornis_gpu_abi_macros::{fragment, vertex};

#[vertex(workgroup_size = 4)]
fn quad(x: f32) -> f32 {
    x
}

#[fragment(workgroup_size = 4)]
fn shade(u: f32) -> f32 {
    u
}

fn main() {}
