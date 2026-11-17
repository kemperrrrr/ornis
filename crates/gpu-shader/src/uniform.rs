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

use crate::TYPE_TAG_ARRAY;
use crate::TYPE_TAG_F32;
use crate::TYPE_TAG_STRUCT;
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
// qual:allow(srp, god_struct) reason: "one method per import kind the binding can offer (scalar, vector, plane, nested field, array element); splitting it would fork the const-generic address every caller names"
pub struct Uniform<T: GpuStruct, const G: u32, const B: u32> {
    layout: core::marker::PhantomData<fn() -> T>,
}

impl<T: GpuStruct, const G: u32, const B: u32> Uniform<T, G, B> {
    /// Fail the build when logical field `FIELD` is not declared with `tag`.
    const fn require_tag<const FIELD: usize>(tag: u8) {
        assert!(
            logical_tag(T::FIELD_TAGS, FIELD) == tag,
            "uniform field has a different declared tag"
        );
    }

    /// Read scalar `f32` field `FIELD`.
    #[inline(always)]
    pub fn field_f32<const FIELD: usize>(&self) -> f32 {
        const { Self::require_tag::<FIELD>(TYPE_TAG_F32) };
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::field_f32(G, B, T::ID, FIELD as u32)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
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

    /// Lane `COMPONENT` (`0..4`) of `vec4` field `FIELD`.
    #[inline(always)]
    pub fn vec4_lane<const FIELD: usize, const COMPONENT: usize>(&self) -> f32 {
        const { Self::require_tag::<FIELD>(TYPE_TAG_VEC4) };
        const { assert!(COMPONENT < 4, "a vec4 has four lanes") };
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::vec4_component(G, B, T::ID, FIELD as u32, COMPONENT as u32)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }

    /// Scalar `f32` field `INNER` of the struct stored in field `FIELD`.
    // qual:allow(test_quality, untested) reason: "shader method; the nested-struct test covers the lowered WGSL, and calling it on the host diverges"
    #[inline(always)]
    pub fn nested_field_f32<const FIELD: usize, const INNER: usize>(&self) -> f32 {
        const { Self::require_tag::<FIELD>(TYPE_TAG_STRUCT) };
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::nested_field_f32(G, B, T::ID, FIELD as u32, INNER as u32)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = INNER;
            crate::host_unavailable()
        }
    }

    /// Lane `COMPONENT` of `vec4` field `ELEMENT_FIELD` of element `index` of
    /// the fixed-size array field `FIELD`.
    ///
    /// The element index is the only runtime value: WGSL allows a dynamic index
    /// into a fixed-size uniform array, but not into a struct or a vector.
    // qual:allow(test_quality, untested) reason: "shader method; the translator test covers the lowered WGSL, and calling it on the host diverges"
    #[inline(always)]
    pub fn array_vec4_lane<
        const FIELD: usize,
        const ELEMENT_FIELD: usize,
        const COMPONENT: usize,
    >(
        &self,
        index: u32,
    ) -> f32 {
        const { Self::require_tag::<FIELD>(TYPE_TAG_ARRAY) };
        const { assert!(COMPONENT < 4, "a vec4 has four lanes") };
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::array_vec4_lane(
                G,
                B,
                T::ID,
                FIELD as u32,
                index,
                ELEMENT_FIELD as u32,
                COMPONENT as u32,
            )
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = index;
            crate::host_unavailable()
        }
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

/// Tag of logical field `field`.
///
/// The slice is not one tag per field: [`TYPE_TAG_STRUCT`] consumes the next
/// byte as the nested struct id, and [`TYPE_TAG_ARRAY`] consumes the element's
/// tag (itself possibly two bytes) and the length. Hence the cursor.
const fn logical_tag(tags: &[u8], field: usize) -> u8 {
    let mut cursor = 0;
    let mut seen = 0;
    while seen < field {
        assert!(
            cursor < tags.len(),
            "uniform field index is past the declared field count"
        );
        cursor += field_width(tags, cursor);
        seen += 1;
    }
    assert!(
        cursor < tags.len(),
        "uniform field index is past the declared field count"
    );
    tags[cursor]
}

/// Bytes the field whose leading tag sits at `cursor` occupies.
const fn field_width(tags: &[u8], cursor: usize) -> usize {
    let tag = tags[cursor];
    if tag == TYPE_TAG_ARRAY {
        // The element tag starts right after the array tag and is itself one
        // byte, or two when it names a nested struct.
        let element = cursor + 1;
        assert!(element < tags.len(), "array field has no element tag");
        let element_width = if tags[element] == TYPE_TAG_STRUCT {
            2
        } else {
            1
        };
        let length = element + element_width;
        assert!(length < tags.len(), "array field has no length");
        1 + element_width + 1
    } else if tag == TYPE_TAG_STRUCT {
        2
    } else {
        1
    }
}

impl<T: GpuStruct, const G: u32, const B: u32> Clone for Uniform<T, G, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: GpuStruct, const G: u32, const B: u32> Copy for Uniform<T, G, B> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TYPE_TAG_U32, TYPE_TAG_VEC4};

    #[test]
    fn a_scalar_after_an_array_of_structs_is_found() {
        // `struct Lighting { ambient: vec4, lights: array<Light, 8>, count: u32 }`
        // with `Light` at struct id 3.
        const TAGS: [u8; 6] = [
            TYPE_TAG_VEC4,
            TYPE_TAG_ARRAY,
            TYPE_TAG_STRUCT,
            3,
            8,
            TYPE_TAG_U32,
        ];
        assert_eq!(logical_tag(&TAGS, 0), TYPE_TAG_VEC4);
        assert_eq!(logical_tag(&TAGS, 1), TYPE_TAG_ARRAY);
        assert_eq!(logical_tag(&TAGS, 2), TYPE_TAG_U32);
    }

    #[test]
    fn a_field_after_an_array_of_vectors_is_found() {
        const TAGS: [u8; 4] = [TYPE_TAG_ARRAY, TYPE_TAG_VEC4, 4, TYPE_TAG_U32];
        assert_eq!(logical_tag(&TAGS, 0), TYPE_TAG_ARRAY);
        assert_eq!(logical_tag(&TAGS, 1), TYPE_TAG_U32);
    }

    #[test]
    fn a_nested_struct_still_costs_two_bytes() {
        const TAGS: [u8; 4] = [TYPE_TAG_STRUCT, 3, TYPE_TAG_F32, TYPE_TAG_U32];
        assert_eq!(logical_tag(&TAGS, 1), TYPE_TAG_F32);
        assert_eq!(logical_tag(&TAGS, 2), TYPE_TAG_U32);
    }
}
