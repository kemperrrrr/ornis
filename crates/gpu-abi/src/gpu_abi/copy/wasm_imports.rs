//! Thin wrappers around the wasm GPU import ABI.

use super::*;

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_uniform_field_f32(group: u32, binding: u32, sid: u32, field: u32) -> f32 {
    import_uniform_field_f32!(group, binding, sid, field)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_uniform_field_u32(group: u32, binding: u32, sid: u32, field: u32) -> u32 {
    import_uniform_field_u32!(group, binding, sid, field)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_uniform_read_f32(group: u32, binding: u32, len: u32, index: u32) -> f32 {
    import_uniform_read_f32!(group, binding, len, index)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_storage_read_f32(group: u32, binding: u32, index: u32) -> f32 {
    import_storage_read_f32!(group, binding, index)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_storage_write_f32(group: u32, binding: u32, index: u32, value: f32) {
    import_storage_write_f32!(group, binding, index, value)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_storage_read_u32(group: u32, binding: u32, index: u32) -> u32 {
    import_storage_read_u32!(group, binding, index)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_storage_write_u32(group: u32, binding: u32, index: u32, value: u32) {
    import_storage_write_u32!(group, binding, index, value)
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_texture_load_2d(group: u32, binding: u32, x: i32, y: i32) -> GpuVec4 {
    // SAFETY: the import returns one texel as four initialized f32 lanes.
    let word = call_gpu_import!(texture_load_2d(group, binding, x, y));
    GpuVec4 {
        x: word.x,
        y: word.y,
        z: word.z,
        w: word.w,
    }
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_texture_width_2d(group: u32, binding: u32) -> u32 {
    // SAFETY: this import returns the bound texture's width, not a pointer.
    call_gpu_import!(texture_width_2d(group, binding))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_texture_height_2d(group: u32, binding: u32) -> u32 {
    // SAFETY: this import returns the bound texture's height, not a pointer.
    call_gpu_import!(texture_height_2d(group, binding))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_math_dot3(a: GpuVec3, b: GpuVec3) -> f32 {
    // SAFETY: the import accepts six scalar lanes and returns one scalar.
    call_gpu_import!(math_dot3(a.x, a.y, a.z, b.x, b.y, b.z))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_math_cross3(a: GpuVec3, b: GpuVec3) -> GpuVec3 {
    // SAFETY: the import returns three initialized f32 lanes.
    let word = call_gpu_import!(math_cross3(a.x, a.y, a.z, b.x, b.y, b.z));
    GpuVec3 {
        x: word.x,
        y: word.y,
        z: word.z,
    }
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_math_length3(value: GpuVec3) -> f32 {
    // SAFETY: the import accepts three scalar lanes and returns one scalar.
    call_gpu_import!(math_length3(value.x, value.y, value.z))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_math_normalize3(value: GpuVec3) -> GpuVec3 {
    // SAFETY: the import returns three initialized f32 lanes.
    let word = call_gpu_import!(math_normalize3(value.x, value.y, value.z));
    GpuVec3 {
        x: word.x,
        y: word.y,
        z: word.z,
    }
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_storage_atomic_add_u32(group: u32, binding: u32, index: u32, value: u32) -> u32 {
    // SAFETY: the import performs one atomic operation on its named storage binding.
    call_gpu_import!(storage_atomic_add_u32(group, binding, index, value))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_workgroup_load_u32(index: u32) -> u32 {
    // SAFETY: the import reads a bounded word from workgroup scratch.
    call_gpu_import!(workgroup_load_u32(index))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_workgroup_store_u32(index: u32, value: u32) {
    // SAFETY: the import writes a bounded word to workgroup scratch.
    call_gpu_import!(workgroup_store_u32(index, value))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_workgroup_atomic_add_u32(index: u32, value: u32) -> u32 {
    // SAFETY: the import performs one atomic operation in workgroup scratch.
    call_gpu_import!(workgroup_atomic_add_u32(index, value))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_workgroup_barrier() {
    // SAFETY: the import lowers to a workgroup control barrier.
    call_gpu_import!(workgroup_barrier())
}

#[cfg(target_arch = "wasm32")]
pub(super) fn call_storage_barrier() {
    // SAFETY: the import lowers to a storage memory barrier.
    call_gpu_import!(storage_barrier())
}
