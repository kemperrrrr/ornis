//! Vec3 math import wrappers for wasm32.

use crate::layout::GpuVec3;

#[repr(C)]
struct Word3 {
    x: f32,
    y: f32,
    z: f32,
}

// SAFETY: these math imports use the documented ornis_gpu scalar-lane ABI.
// The wrappers below pass and return only initialized scalar lanes.
#[link(wasm_import_module = "ornis_gpu")]
unsafe extern "C" {
    fn math_dot3(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32;
    fn math_cross3(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> Word3;
    fn math_length3(x: f32, y: f32, z: f32) -> f32;
    fn math_normalize3(x: f32, y: f32, z: f32) -> Word3;
}

#[inline(always)]
pub(in crate::gpu_abi) fn call_math_dot3(a: GpuVec3, b: GpuVec3) -> f32 {
    call_gpu_import!(math_dot3(a.x, a.y, a.z, b.x, b.y, b.z))
}

#[inline(always)]
pub(in crate::gpu_abi) fn call_math_cross3(a: GpuVec3, b: GpuVec3) -> GpuVec3 {
    let word = call_gpu_import!(math_cross3(a.x, a.y, a.z, b.x, b.y, b.z));
    GpuVec3 {
        x: word.x,
        y: word.y,
        z: word.z,
    }
}

#[inline(always)]
pub(in crate::gpu_abi) fn call_math_length3_xyz(x: f32, y: f32, z: f32) -> f32 {
    call_gpu_import!(math_length3(x, y, z))
}

#[inline(always)]
pub(in crate::gpu_abi) fn call_math_normalize3(value: GpuVec3) -> GpuVec3 {
    let word = call_gpu_import!(math_normalize3(value.x, value.y, value.z));
    GpuVec3 {
        x: word.x,
        y: word.y,
        z: word.z,
    }
}
