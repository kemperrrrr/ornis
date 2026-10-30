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
