//! no_std GPU kernels for the stable Rust → LLVM IR → WGSL path.
//!
//! The crate is a normal `no_std` library: cargo checks it on the host, while
//! `cargo xtask build-shaders` compiles the same source for
//! `wasm32-unknown-unknown` with `--emit=llvm-ir` and feeds the IR to
//! `ornis-llvm2gpu`. Entry points use the scalar `cs_main` ABI for now;
//! typed `Storage`/`Uniform` wrappers and `#[gpu_entry]` metadata come with
//! the cull/compaction slice.

#![no_std]
#![warn(missing_docs)]

/// Smoke compute entry: writes `id * 2 + 1` through the scalar return ABI.
///
/// `ornis-llvm2gpu` stores the return value at `storage_0_0[global_id.x]`.
#[unsafe(no_mangle)]
pub extern "C" fn cs_main(id: u32) -> f32 {
    (id as f32) * 2.0 + 1.0
}
