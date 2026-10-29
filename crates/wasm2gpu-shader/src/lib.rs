//! Shader body compiled to wasm32 and lowered by `ornis-wasm2gpu`.
//!
//! The host build of [`shade`] is the CPU reference. On `wasm32` the same
//! function is an `extern "C"` export named `shade`. `#[unsafe(no_mangle)]`
//! keeps that export name: the gpu-test lifter looks it up, and Rust 2024
//! makes `no_mangle` an unsafe attribute. It is an export attribute, not a
//! memory-unsafe block; the only `unsafe` code stays in `gpu-abi`. Returning
//! a struct would spill through linear memory, which the translator rejects.
//! This crate is not the production shader DSL.

#![cfg_attr(target_arch = "wasm32", no_std)]
#![warn(missing_docs)]

use ornis_gpu_abi::gpu_struct;
use ornis_gpu_abi::storage_at_f32;
use ornis_gpu_abi::storage_store_f32;
use ornis_gpu_abi::uniform_array_f32;
use ornis_gpu_abi::uniform_field_f32;
use ornis_gpu_abi::uniform_field_u32;

gpu_struct! {
    /// Uniform parameters. Field 0 is the dynamic index. Field 1 is the scale.
    pub struct ShadeParams {
        /// Element index shared by the storage arrays and the uniform array.
        pub index: u32,
        /// Scale applied to the storage sample.
        pub intensity: f32,
        /// Padding so the struct span stays 16 bytes.
        pub pad0: u32,
        /// Padding so the struct span stays 16 bytes.
        pub pad1: u32,
    }
}

gpu_struct! {
    /// One `f32` at the uniform stride of 16 bytes.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub struct UniformF32Lane {
        /// Value the shader reads.
        pub value: f32,
        /// Padding to the uniform stride.
        pub pad0: u32,
        /// Padding to the uniform stride.
        pub pad1: u32,
        /// Padding to the uniform stride.
        pub pad2: u32,
    }
}

/// Length of the uniform `f32` array [`shade`] indexes.
pub const UNIFORM_LEN: u32 = 4;

/// Element count of the storage buffers [`shade`] reads and writes.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
// qual:api — the gpu e2e harness sizes storage buffers from this length.
pub const STORAGE_LEN: u32 = 4;

const OUT_BINDING: u32 = 3;

/// Fragment body.
///
/// `u` and `v` match the vertex outputs and are not read. The dynamic index
/// comes from the uniform struct, so the vertex/fragment locations stay two
/// `f32`s.
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
// qual:api — host reference for the wasm export; the gpu e2e test calls it.
pub fn shade(u: f32, v: f32) -> f32 {
    shade_body(u, v)
}

/// Wasm export named `shade`.
///
/// `#[unsafe(no_mangle)]` is the Rust 2024 spelling of the export attribute.
/// The lifter copies this body into the toy `fs_main` only when the export
/// name is `shade`.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn shade(u: f32, v: f32) -> f32 {
    shade_body(u, v)
}

#[inline(always)]
fn shade_body(_u: f32, _v: f32) -> f32 {
    let index = uniform_field_u32::<0, 0, 0, 0, { ShadeParams::FIELD_OFFSETS[0] }>();
    let intensity = uniform_field_f32::<0, 0, 0, 1, { ShadeParams::FIELD_OFFSETS[1] }>();
    let src = storage_at_f32::<0, 1>(index);
    let extra = uniform_array_f32::<0, 2, UNIFORM_LEN>(index);
    let out = src * intensity + extra;
    storage_store_f32::<0, OUT_BINDING>(index, out);
    out
}

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    core::arch::wasm32::unreachable()
}
