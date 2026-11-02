//! GPU parity and A/B harness for the wasm2gpu skinning guest (issue #228).
//!
//! The guest compute kernel (`skin` in `ornis-wasm2gpu-shader`) writes
//! skinned positions and normals into a storage buffer; this harness owns
//! the lowering, the persistent buffers, the packing from render-side
//! skin entries and the readback the parity test asserts against
//! `ornis_animation::skin_vertices`. The production DSL vertex-stage
//! skinning in the parent module is untouched.

mod bench;
mod data;
mod gpu;
mod lower;

pub use bench::run_skin_ab;
pub(crate) use data::SkinInputs;
pub(crate) use data::SyntheticSkin;
pub(crate) use data::synthetic_skin;
pub(crate) use gpu::SkinGpu;
pub(crate) use lower::lower_guest;

#[cfg(test)]
mod tests;
