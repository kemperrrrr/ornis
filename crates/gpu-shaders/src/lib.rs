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

unsafe extern "C" {
    fn storage_len_u32x4(group: i32, binding: i32) -> i32;
    fn storage_read_u32_lane(group: i32, binding: i32, index: i32, lane: i32) -> i32;
    fn storage_copy_u32x4(
        src_group: i32,
        src_binding: i32,
        src_index: i32,
        dst_group: i32,
        dst_binding: i32,
        dst_index: i32,
    );
}

/// Copy one 16-byte record chunk and read one lane through the vec4 ABI.
///
/// `cs_main` stores the returned bits at `storage_0_0[global_id.x]`, so the
/// builder smoke test can inspect the lowered WGSL for a vec4 load/store,
/// `arrayLength`, and a read-only source binding.
#[unsafe(no_mangle)]
pub extern "C" fn cs_main(id: u32) -> f32 {
    let index = id as i32;
    // SAFETY: the import reads the array length of binding 1; the host
    // binds it as a read-only storage buffer.
    let len = unsafe { storage_len_u32x4(0, 1) };
    if index < len {
        // SAFETY: `index` is guarded by `len`; both bindings are vec4<u32>
        // arrays with the same record layout.
        unsafe {
            storage_copy_u32x4(0, 1, index, 0, 2, index);
        }
    }
    // SAFETY: lane 3 of chunk 0 exists in every vec4<u32> record.
    let word = unsafe { storage_read_u32_lane(0, 1, 0, 3) };
    f32::from_bits(word as u32)
}
