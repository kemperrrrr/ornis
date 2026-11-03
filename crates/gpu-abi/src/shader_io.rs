//! Scalar uniform and storage calls a wasm32 shader can make.
//!
//! The offline translator rejects linear memory, so a shader body cannot
//! return a struct or load through `i32.load`. These functions stay safe on
//! the host: they read the shadow installed by
//! [`with_host`](crate::gpu_abi::with_host) through [`GpuRef`](crate::GpuRef).
//! On `wasm32` the same functions call `ornis_gpu` imports. `OFFSET` is the
//! host byte offset from [`gpu_struct!`](crate::gpu_struct). The wasm operand
//! is the constant field index, because WGSL struct fields are not
//! dynamically indexed. An array length is a const generic; the element
//! index may be a runtime value.

#[cfg(not(target_arch = "wasm32"))]
use crate::GpuSafeLayout;
#[cfg(not(target_arch = "wasm32"))]
use crate::Storage;
#[cfg(not(target_arch = "wasm32"))]
use crate::Uniform;
use crate::GpuVec3;

/// Read one `f32` field of uniform struct `SID` at host byte `OFFSET`.
///
/// `FIELD` is the declaration index. On `wasm32` it is the import operand
/// and `OFFSET` is not passed: the translator places the field itself.
#[inline(always)]
pub fn uniform_field_f32<
    const GROUP: u32,
    const BINDING: u32,
    const SID: u32,
    const FIELD: u32,
    const OFFSET: u32,
>() -> f32 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Uniform::<GROUP, BINDING, f32>::bind()
            .as_ref()
            .field::<f32>(OFFSET)
            .read()
    }
    #[cfg(target_arch = "wasm32")]
    {
        crate::gpu_abi::uniform_field_f32(GROUP, BINDING, SID, FIELD)
    }
}

/// Read one `u32` field of uniform struct `SID` at host byte `OFFSET`.
///
/// Same split as [`uniform_field_f32`]: the host uses `OFFSET`, wasm uses
/// `FIELD`.
#[inline(always)]
pub fn uniform_field_u32<
    const GROUP: u32,
    const BINDING: u32,
    const SID: u32,
    const FIELD: u32,
    const OFFSET: u32,
>() -> u32 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Uniform::<GROUP, BINDING, u32>::bind()
            .as_ref()
            .field::<u32>(OFFSET)
            .read()
    }
    #[cfg(target_arch = "wasm32")]
    {
        crate::gpu_abi::uniform_field_u32(GROUP, BINDING, SID, FIELD)
    }
}

/// Read `array<f32, LEN>[index]` from a uniform binding.
///
/// The element stride is the uniform stride of `f32` (16 bytes), not the
/// packed storage stride. `index` may be a runtime value.
#[inline(always)]
pub fn uniform_array_f32<const GROUP: u32, const BINDING: u32, const LEN: u32>(index: u32) -> f32 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Uniform::<GROUP, BINDING, f32>::bind()
            .as_ref()
            .index(index, f32::UNIFORM_STRIDE)
            .read()
    }
    #[cfg(target_arch = "wasm32")]
    {
        crate::gpu_abi::uniform_read_f32(GROUP, BINDING, LEN, index)
    }
}

/// Read `array<f32>[index]` from a storage binding.
///
/// The stride is the packed storage stride of `f32`.
#[inline(always)]
pub fn storage_at_f32<const GROUP: u32, const BINDING: u32>(index: u32) -> f32 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Storage::<GROUP, BINDING, f32>::bind()
            .as_ref()
            .index_storage(index)
            .read()
    }
    #[cfg(target_arch = "wasm32")]
    {
        crate::gpu_abi::storage_read_f32(GROUP, BINDING, index)
    }
}

/// Read `array<u32>[index]` from a storage binding.
///
/// The stride is the packed storage stride of `u32`.
#[inline(always)]
pub fn storage_at_u32<const GROUP: u32, const BINDING: u32>(index: u32) -> u32 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Storage::<GROUP, BINDING, u32>::bind()
            .as_ref()
            .index_storage(index)
            .read()
    }
    #[cfg(target_arch = "wasm32")]
    {
        crate::gpu_abi::storage_read_u32(GROUP, BINDING, index)
    }
}

/// Write one `u32` into `array<u32>[index]` of a storage binding.
#[inline(always)]
pub fn storage_store_u32<const GROUP: u32, const BINDING: u32>(index: u32, value: u32) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Storage::<GROUP, BINDING, u32>::bind()
            .as_ref()
            .index_storage(index)
            .write(value);
    }
    #[cfg(target_arch = "wasm32")]
    {
        crate::gpu_abi::storage_write_u32(GROUP, BINDING, index, value);
    }
}

/// Write one `f32` into `array<f32>[index]` of a storage binding.
#[inline(always)]
pub fn storage_store_f32<const GROUP: u32, const BINDING: u32>(index: u32, value: f32) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Storage::<GROUP, BINDING, f32>::bind()
            .as_ref()
            .index_storage(index)
            .write(value);
    }
    #[cfg(target_arch = "wasm32")]
    {
        crate::gpu_abi::storage_write_f32(GROUP, BINDING, index, value);
    }
}

/// Dot product of two `vec3<f32>` values.
#[inline(always)]
pub fn gpu_dot3(a: GpuVec3, b: GpuVec3) -> f32 {
    gpu_abi::math_dot3(a, b)
}

/// Cross product of two `vec3<f32>` values.
#[inline(always)]
pub fn gpu_cross3(a: GpuVec3, b: GpuVec3) -> GpuVec3 {
    gpu_abi::math_cross3(a, b)
}

/// Euclidean length of a `vec3<f32>` value.
#[inline(always)]
pub fn gpu_length3(value: GpuVec3) -> f32 {
    gpu_abi::math_length3(value)
}

/// Normalize a `vec3<f32>` value to unit length.
///
/// As with WGSL `normalize`, a zero vector has no defined direction.
#[inline(always)]
pub fn gpu_normalize3(value: GpuVec3) -> GpuVec3 {
    gpu_abi::math_normalize3(value)
}

/// Atomically add to one `u32` in storage and return its previous value.
///
/// On the host this is serialized by the shadow; shader execution uses WGSL
/// `atomicAdd` and requires an atomic-compatible storage binding.
#[inline(always)]
pub fn storage_atomic_add_u32<const GROUP: u32, const BINDING: u32>(
    index: u32,
    value: u32,
) -> u32 {
    crate::gpu_abi::storage_atomic_add_u32(GROUP, BINDING, index, value)
}

/// Read one word from the fixed-size workgroup scratch array.
///
/// `index` must be less than [`crate::symbols::WORKGROUP_WORDS`]. The host
/// shadow checks this; shader code must keep the index in bounds.
#[inline(always)]
pub fn workgroup_load_u32(index: u32) -> u32 {
    crate::gpu_abi::workgroup_load_u32(index)
}

/// Write one word to the fixed-size workgroup scratch array.
///
/// `index` must be less than [`crate::symbols::WORKGROUP_WORDS`]. The host
/// shadow checks this; shader code must keep the index in bounds.
#[inline(always)]
pub fn workgroup_store_u32(index: u32, value: u32) {
    crate::gpu_abi::workgroup_store_u32(index, value)
}

/// Atomically add to a workgroup scratch word and return the previous value.
///
/// `index` must be less than [`crate::symbols::WORKGROUP_WORDS`]. Atomic
/// scratch is a separate array from the ordinary workgroup scratch words.
#[inline(always)]
pub fn workgroup_atomic_add_u32(index: u32, value: u32) -> u32 {
    crate::gpu_abi::workgroup_atomic_add_u32(index, value)
}

/// Synchronize workgroup execution and workgroup-memory accesses.
#[inline(always)]
pub fn workgroup_barrier() {
    crate::gpu_abi::workgroup_barrier()
}

/// Order storage-buffer memory accesses within a workgroup.
#[inline(always)]
pub fn storage_barrier() {
    crate::gpu_abi::storage_barrier()
}
