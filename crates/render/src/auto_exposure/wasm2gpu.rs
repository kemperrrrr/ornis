//! GPU parity and A/B harness for the scalarized Rust→Wasm→WGSL guest.

mod bench;
mod gpu;
mod lower;

pub use bench::run_auto_exposure_ab;
pub(crate) use gpu::GuestGpu;
pub(crate) use gpu::GuestInputs;
pub(crate) use gpu::GuestOutput;
pub(crate) use lower::Lowered;
pub(crate) use lower::lower_guest;
