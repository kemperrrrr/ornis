//! Atomic and synchronization import wrappers for wasm32.

// SAFETY: these imports use the documented ornis_gpu integer ABI. Wrappers
// pass only scalar indices and values; they expose no guest-memory pointers.
#[link(wasm_import_module = "ornis_gpu")]
unsafe extern "C" {
    fn storage_atomic_add_u32(group: u32, binding: u32, index: u32, value: u32) -> u32;
    fn workgroup_load_u32(index: u32) -> u32;
    fn workgroup_store_u32(index: u32, value: u32);
    fn workgroup_atomic_add_u32(index: u32, value: u32) -> u32;
    fn workgroup_barrier();
    fn storage_barrier();
}

#[inline(always)]
pub(in crate::gpu_abi) fn call_storage_atomic_add_u32(
    group: u32,
    binding: u32,
    index: u32,
    value: u32,
) -> u32 {
    call_gpu_import!(storage_atomic_add_u32(group, binding, index, value))
}

#[inline(always)]
pub(in crate::gpu_abi) fn call_workgroup_load_u32(index: u32) -> u32 {
    call_gpu_import!(workgroup_load_u32(index))
}

#[inline(always)]
pub(in crate::gpu_abi) fn call_workgroup_store_u32(index: u32, value: u32) {
    call_gpu_import!(workgroup_store_u32(index, value))
}

#[inline(always)]
pub(in crate::gpu_abi) fn call_workgroup_atomic_add_u32(index: u32, value: u32) -> u32 {
    call_gpu_import!(workgroup_atomic_add_u32(index, value))
}

#[inline(always)]
pub(in crate::gpu_abi) fn call_workgroup_barrier() {
    call_gpu_import!(workgroup_barrier())
}

#[inline(always)]
pub(in crate::gpu_abi) fn call_storage_barrier() {
    call_gpu_import!(storage_barrier())
}
