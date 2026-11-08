//! Stable `no_std` shader SDK for the llvm2gpu path.
//!
//! The crate follows rust-gpu's user-facing shape where stable Rust allows it:
//! `#[gpu_entry(compute(threads(64)))]` declares the entry and its workgroup
//! size, and typed resource parameters replace the raw `ornis_gpu` imports
//! inside kernels:
//!
//! ```ignore
//! use ornis_gpu_shader::{gpu_entry, GpuStruct, ReadOnly, ReadWrite, Storage, Uniform};
//!
//! /// Group-0 binding 0 of the packed `CullParams` uniform.
//! pub struct CullParams;
//! impl GpuStruct for CullParams {
//!     const ID: u32 = 0;
//!     const FIELD_TAGS: &'static [u8] = &[6, 6, 6, 6, 6, 6, 2];
//! }
//!
//! #[gpu_entry(compute(threads(64)))]
//! pub fn cs_main(
//!     id: u32,
//!     #[gpu(uniform, group = 0, binding = 0)] params: Uniform<CullParams>,
//!     #[gpu(storage_buffer, group = 0, binding = 1, read_only)] src: Storage<f32>,
//!     #[gpu(storage_buffer, group = 0, binding = 2)] dst: Storage<u32, ReadWrite>,
//! ) {
//!     let count = params.field_u32::<6>();
//!     if id < count && id < src.len() {
//!         dst.store(id, id);
//!     }
//! }
//! ```
//!
//! The annotated parameters stay in the signature and carry the group and
//! binding in their types. They are zero-sized, so rustc drops them from the
//! compiled entry: the translator still lowers `cs_main(i32)`. The attribute
//! also attaches the export name, so the kernel is a plain safe `fn` with no
//! `extern "C"` and no `#[unsafe(no_mangle)]`. Field reads are checked against [`GpuStruct::FIELD_TAGS`] at
//! compile time, and a read-only [`Storage`] has no `store` method, so a
//! mismatched use fails the build instead of producing invalid WGSL.
//!
//! A new GPU operation is added here, as a method or a function. Kernel crates
//! do not declare `extern "C"` and do not name an `ornis_gpu` import: that
//! block lives only in the private `abi` module, and `#[inline(always)]`
//! methods lower to it. Resource methods only exist on `wasm32`; on the host
//! they compile and panic, because the host build of a kernel crate has no
//! bindings to read.

#![no_std]
#![warn(missing_docs)]

#[cfg(target_arch = "wasm32")]
mod abi;
mod access;
mod builtin;
/// Element types of a storage binding and the import each lowers to.
pub mod elements;
#[cfg(not(target_arch = "wasm32"))]
mod host;
mod math;
/// Entry metadata format shared with `ornis-llvm2gpu`.
pub mod meta;
mod resource;
mod texture;
mod uniform;
mod vec3;
mod workgroup;

pub use access::{ReadOnly, ReadWrite, StorageAccess};
pub use builtin::{fragment_store, fragment_uv, global_invocation_lane, local_invocation_index};
pub use elements::{AtomicU32, Vec4U32};
pub use math::{
    abs, acos, clamp, cos, cross_lane, dot3, exp2, fract, length3, log2, mix, normalize_lane, sign,
    sin,
};
pub use ornis_gpu_shader_macros::gpu_entry;
pub use resource::Storage;
pub use texture::{DepthTexture2d, Sampler, Texture2d};
pub use uniform::{GpuStruct, Uniform};
pub use vec3::Vec3;
pub use workgroup::{WorkgroupAtomicU32, WorkgroupU32, workgroup_barrier};

/// Base-e exponential, lowered to WGSL `exp`.
///
/// `no_std` has no `f32::exp`, and a software series would not match the GPU
/// instruction. On the host this diverges: a kernel crate is type-checked
/// there, not executed.
#[inline(always)]
#[must_use]
pub fn exp(value: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::exp(value)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = value;
        host_unavailable()
    }
}

/// Field tag of an `f32` in `ornis_gpu.types` (mirrors `ornis_gpu_abi`).
pub const TYPE_TAG_F32: u8 = 1;
/// Field tag of a `u32` in `ornis_gpu.types`.
pub const TYPE_TAG_U32: u8 = 2;
/// Field tag of an `i32` in `ornis_gpu.types`.
pub const TYPE_TAG_I32: u8 = 3;
/// Field tag of a `vec2<f32>` in `ornis_gpu.types`.
pub const TYPE_TAG_VEC2: u8 = 4;
/// Field tag of a `vec3<f32>` in `ornis_gpu.types`.
pub const TYPE_TAG_VEC3: u8 = 5;
/// Field tag of a `vec4<f32>` in `ornis_gpu.types`.
pub const TYPE_TAG_VEC4: u8 = 6;
/// Field tag of a nested uniform struct. The next `FIELD_TAGS` byte is its id.
pub const TYPE_TAG_STRUCT: u8 = 7;

/// Diverging stub behind every resource method outside `wasm32`.
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub(crate) fn host_unavailable() -> ! {
    host::unavailable()
}

/// Square root for the host reference path.
///
/// `no_std` has no `f32::sqrt`; the host build only exists so kernel crates can
/// be type-checked and unit-tested, and the test harness links `std`, which
/// provides the intrinsic. On `wasm32` the same operation is the
/// `gpu_length3_xyz` import instead, so this function is never in a shader.
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub(crate) fn host_sqrt(value: f32) -> f32 {
    extern crate std;
    std::primitive::f32::sqrt(value)
}
