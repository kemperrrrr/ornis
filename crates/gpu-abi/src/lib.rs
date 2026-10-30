//! Toy GPU ABI for the Wasm → GPU research path.
//!
//! Binding provenance lives in const generics (`GROUP`, `BINDING`). The only
//! runtime bits in a [`GpuRef`] are a byte offset. Textures are calls, not
//! loads from linear memory. This crate is not the production shader DSL
//! (`ornis-shader-lang` / `#[stage]`); see `docs/rendering/wasm-segmented-memory.md`.
//!
//! On the host, [`gpu_abi::with_host`] installs a shadow of those bindings so
//! [`GpuRef::read`] / [`GpuRef::write`] can be executed and checked under Miri.
//! On `wasm32` the same methods lower to `ornis_gpu` imports
//! ([`symbols`]). Every `unsafe` block in this crate lives in `gpu_abi::copy`.

#![cfg_attr(target_arch = "wasm32", no_std)]
#![warn(missing_docs)]

mod gpu_ref;
mod layout;
mod macros;
mod shader_io;
mod texture;

/// Host shadow memory and the only `unsafe` in the SDK.
pub mod gpu_abi;
/// Import names the offline translator recognizes.
pub mod symbols;

#[cfg(not(target_arch = "wasm32"))]
pub use gpu_abi::{HostMemory, with_host};
pub use gpu_ref::{GpuRef, Storage, StorageSpace, Uniform, UniformSpace};
#[cfg(not(target_arch = "wasm32"))]
pub use layout::encode_types_section;
pub use layout::{
    F32_ALIGN, F32_SIZE, FieldLayout, GpuSafeLayout, GpuVec2, GpuVec3, GpuVec4, LayoutMismatch,
    TYPE_TAG_F32, TYPE_TAG_I32, TYPE_TAG_U32, TYPE_TAG_VEC2, TYPE_TAG_VEC3, TYPE_TAG_VEC4,
    UNIFORM_STRUCT_ALIGN, VEC2_ALIGN, VEC2_SIZE, VEC3_ALIGN, VEC3_SIZE, VEC4_ALIGN, VEC4_SIZE,
    place_fields, round_up, storage_layout_mismatch, storage_stride, uniform_layout_mismatch,
    uniform_stride,
};
pub use shader_io::{
    storage_at_f32, storage_at_u32, storage_store_f32, storage_store_u32, uniform_array_f32,
    uniform_field_f32, uniform_field_u32,
};
pub use texture::{Sampler, Texture2d};

#[cfg(test)]
mod abi_ptr_tests;
#[cfg(test)]
mod abi_tests;
