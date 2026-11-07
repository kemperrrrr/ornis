//! Typed uniform bindings.
//!
//! A [`Uniform<T>`](Uniform) names one `var<uniform>` binding and carries the
//! `ornis_gpu.types` metadata of `T` through [`GpuStruct`]. Field reads are
//! const-generic (`field_u32::<6>()`), so the field index reaches the
//! translator as a literal and is checked against the declared tag list at
//! compile time: reading field 6 of a struct whose sixth tag is not `u32`
//! fails the build instead of producing a mismatched WGSL load.
//!
//! The indices follow rust-gpu's `spirv-std` naming (`StorageBuffer<T>` has
//! `.index()`), except that the field index is the `ornis_gpu.types`
//! declaration index rather than a Rust struct field, because the host struct
//! lives in another crate (`ornis-render`) that the shader crate must not
//! depend on.

use crate::TYPE_TAG_U32;
use crate::TYPE_TAG_VEC4;

/// A struct whose fields occupy a uniform binding.
///
/// Implementors describe themselves the way the host side already does: an id
/// into the `ornis_gpu.types` table plus one tag per field, in declaration
/// order. `#[gpu_entry]` serializes those tags into the shader metadata, so a
/// kernel cannot disagree with the struct it reads.
pub trait GpuStruct {
    /// Struct id: the index into the `ornis_gpu.types` table.
    const ID: u32;
    /// Field tags, in declaration order (`1` `f32`, `2` `u32`, `6` `vec4`).
    const FIELD_TAGS: &'static [u8];
}

/// Typed view of a uniform binding `(group, binding)` of `T`.
pub struct Uniform<T: GpuStruct, const G: u32, const B: u32> {
    layout: core::marker::PhantomData<fn() -> T>,
}

impl<T: GpuStruct, const G: u32, const B: u32> Uniform<T, G, B> {
    /// Bind the parameter that `#[gpu_entry]` stripped from the signature.
    #[doc(hidden)]
    #[inline(always)]
    pub const fn __bind() -> Self {
        Self {
            layout: core::marker::PhantomData,
        }
    }

    /// Fail the build when field `FIELD` is not declared with `tag`.
    const fn require_tag<const FIELD: usize>(tag: u8) {
        assert!(
            FIELD < T::FIELD_TAGS.len(),
            "uniform field index is past the declared field count"
        );
        assert!(
            T::FIELD_TAGS[FIELD] == tag,
            "uniform field has a different declared tag"
        );
    }

    /// Read scalar `u32` field `FIELD`.
    #[inline(always)]
    pub fn field_u32<const FIELD: usize>(&self) -> u32 {
        const { Self::require_tag::<FIELD>(TYPE_TAG_U32) };
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::field_u32(G, B, T::ID, FIELD as u32)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }

    /// Signed distance from `(x, y, z)` to `vec4` plane field `FIELD`:
    /// `dot((x, y, z, 1), field)`.
    ///
    /// One uniform vector load and one `dot`, which is what a frustum test
    /// wants; reading the four lanes separately would cost four uniform loads
    /// and a scalar FMA chain.
    #[inline(always)]
    pub fn plane_distance<const FIELD: usize>(&self, x: f32, y: f32, z: f32) -> f32 {
        const { Self::require_tag::<FIELD>(TYPE_TAG_VEC4) };
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::plane_distance(G, B, T::ID, FIELD as u32, x, y, z)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (x, y, z);
            crate::host_unavailable()
        }
    }
}

impl<T: GpuStruct, const G: u32, const B: u32> Clone for Uniform<T, G, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: GpuStruct, const G: u32, const B: u32> Copy for Uniform<T, G, B> {}
