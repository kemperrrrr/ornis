//! A read-only storage binding cannot be written.
//!
//! `Storage<u32, ReadOnly>` has no `store` method: the access mode is part of
//! the type, not a runtime flag, so the mistake is caught here instead of by
//! bind-group validation or by a WGSL store into a `var<storage>`.

use ornis_gpu_shader::{gpu_entry, ReadOnly, Storage};

#[gpu_entry(compute(threads(1)))]
#[unsafe(no_mangle)]
pub extern "C" fn cs_main(
    id: u32,
    #[gpu(storage_buffer, group = 0, binding = 1, read_only)] src: Storage<u32, ReadOnly>,
) {
    src.store(id, 0);
}

fn main() {}
