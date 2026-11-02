//! `#[shader::*]` keeps entry points callable and adds the export wrapper.
//!
//! The custom section is wasm-only; these tests pin the host-visible side:
//! the signature the author wrote still runs as a plain Rust function and the
//! macro does not require a manual `extern "C"`.

mod shader {
    pub use ornis_gpu_abi_macros::compute;
    pub use ornis_gpu_abi_macros::fragment;
    pub use ornis_gpu_abi_macros::vertex;
}

#[shader::compute(workgroup_size = 4)]
pub fn compute_entry(id: u32) {
    let _ = id + 1;
}

#[shader::vertex]
pub fn vertex_entry(x: f32, y: f32) -> f32 {
    x + y
}

#[shader::fragment]
pub fn fragment_entry(value: f32) -> f32 {
    value * 2.0
}

#[test]
fn entries_keep_their_rust_signatures() {
    compute_entry(2);
    assert_eq!(vertex_entry(1.5, 2.0), 3.5);
    assert_eq!(fragment_entry(2.5), 5.0);
}
