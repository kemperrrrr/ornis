//! UI test: `gpu_pipeline` rejects `pack(...)` before an explicit
//! binding — pack lanes bind after every explicit binding, so the
//! option must come last.

use ornis_macros::gpu_pipeline;

#[gpu_pipeline(pack(Anything), storage(buf: [f32; 8], read_write))]
fn would_collide() {}

fn main() {}
