//! UI test: a uniform token has no `write` — the method lives only on
//! `StorageSpace`, with no cfg around it, so the failure is E0599 on
//! every target.

use ornis_gpu_abi::Uniform;

fn main() {
    let uniform = Uniform::<0, 0, f32>::bind();
    uniform.as_ref().write(0.0);
}
