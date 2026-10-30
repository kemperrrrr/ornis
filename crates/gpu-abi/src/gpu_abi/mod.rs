//! The only module in the SDK that contains `unsafe`.
//!
//! `copy` holds the byte copies (and, on `wasm32`, the import calls).
//! Everything else in this module checks bounds and alignment first, then
//! asks `copy` to move plain bytes. Shader code outside this module is safe
//! Rust: it never sees a raw pointer.
//!
//! # Miri
//!
//! `cargo +nightly-2026-10-09 miri test -p ornis-gpu-abi --lib` with
//! `MIRIFLAGS=-Zmiri-strict-provenance` runs the host shadow. That path
//! allocates real aligned bytes and copies them with `ptr::read` /
//! `ptr::write`, so a missed bounds or alignment check is UB that Miri
//! reports. Provenance stays strict: offsets are integers until the host
//! shadow adds them to a real allocation. The `wasm32` import shim is not
//! executed under Miri:
//! Miri does not provide those imports, and the shim does not dereference a
//! GPU address. Host tests cover the same byte contract the import is
//! specified to implement.

#[cfg(not(target_arch = "wasm32"))]
mod bytes;
mod copy;

#[cfg(not(target_arch = "wasm32"))]
mod host;

#[cfg(not(target_arch = "wasm32"))]
pub use host::HostMemory;
#[cfg(not(target_arch = "wasm32"))]
pub use host::with_host;

use crate::layout::GpuSafeLayout;
use crate::layout::GpuVec2;
use crate::layout::GpuVec4;

/// Stop a broken GPU-ABI contract.
///
/// The assert panics (so tests can use `#[should_panic]`). `abort` /
/// `unreachable` only exist so the signature is `!` after the assert, which
/// the type system does not treat as diverging.
pub(crate) fn contract_breach(msg: &str) -> ! {
    assert!(core::hint::black_box(false), "{msg}");
    #[cfg(target_arch = "wasm32")]
    core::arch::wasm32::unreachable();
    #[cfg(not(target_arch = "wasm32"))]
    std::process::abort();
}

pub(crate) fn read_at<T: GpuSafeLayout>(group: u32, binding: u32, offset: u32) -> T {
    #[cfg(not(target_arch = "wasm32"))]
    {
        host::host_read::<T>(group, binding, offset)
    }
    #[cfg(target_arch = "wasm32")]
    {
        copy::wasm_read::<T>(group, binding, offset)
    }
}

pub(crate) fn write_at<T: GpuSafeLayout>(group: u32, binding: u32, offset: u32, value: &T) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        host::host_write::<T>(group, binding, offset, value);
    }
    #[cfg(target_arch = "wasm32")]
    {
        copy::wasm_write::<T>(group, binding, offset, value);
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn storage_base<const G: u32, const B: u32>() -> u32 {
    copy::wasm_storage_base::<G, B>()
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn ptr_load<T: GpuSafeLayout>(addr: u32) -> T {
    copy::ptr_load::<T>(addr)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn ptr_store<T: GpuSafeLayout>(addr: u32, value: T) {
    copy::ptr_store::<T>(addr, value)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn uniform_field_f32(group: u32, binding: u32, sid: u32, field: u32) -> f32 {
    copy::call_uniform_field_f32(group, binding, sid, field)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn uniform_field_u32(group: u32, binding: u32, sid: u32, field: u32) -> u32 {
    copy::call_uniform_field_u32(group, binding, sid, field)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn uniform_read_f32(group: u32, binding: u32, len: u32, index: u32) -> f32 {
    copy::call_uniform_read_f32(group, binding, len, index)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn storage_read_f32(group: u32, binding: u32, index: u32) -> f32 {
    copy::call_storage_read_f32(group, binding, index)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn storage_write_f32(group: u32, binding: u32, index: u32, value: f32) {
    copy::call_storage_write_f32(group, binding, index, value);
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn storage_read_u32(group: u32, binding: u32, index: u32) -> u32 {
    copy::call_storage_read_u32(group, binding, index)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn storage_write_u32(group: u32, binding: u32, index: u32, value: u32) {
    copy::call_storage_write_u32(group, binding, index, value);
}

pub(crate) fn texture_sample_2d<const TG: u32, const TB: u32, const SG: u32, const SB: u32>(
    uv: GpuVec2,
) -> GpuVec4 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        host::host_sample::<TG, TB, SG, SB>(uv)
    }
    #[cfg(target_arch = "wasm32")]
    {
        copy::wasm_sample::<TG, TB, SG, SB>(uv)
    }
}
