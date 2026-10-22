//! UI test: GPU lanes are opt-in — a plain `#[derive(Pack)]` without
//! `#[pack(gpu)]` must not implement `PackGpu` (trait-bound check, so
//! the diagnostic stays a plain E0277 instead of rustc's unstable
//! method-resolution help notes).

use ornis_macros::Pack;

#[derive(Clone, Pack)]
struct CpuOnly {
    v: f32,
}

fn needs_pack_gpu<T: ornis_core::PackGpu>() {}

fn main() {
    needs_pack_gpu::<CpuOnly>();
}
