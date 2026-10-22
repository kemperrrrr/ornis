//! UI test: GPU lanes are opt-in — a plain `#[derive(Pack)]` without
//! `#[pack(gpu)]` must not implement `PackGpu`.

use ornis_core::PackGpu;
use ornis_macros::Pack;

#[derive(Clone, Pack)]
struct CpuOnly {
    v: f32,
}

fn main() {
    let _ = CpuOnly::pack_gpu_lanes();
}
