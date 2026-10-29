//! GPU layout marker and WGSL uniform/storage placement.
//!
//! [`GpuSafeLayout`] is unsafe to implement: it promises a `repr(C)` plain-data
//! type whose every bit pattern is valid (no `bool`, references, or niches) and
//! whose size and alignment match the associated constants. [`place_fields`]
//! follows WGSL rules: a member's offset is rounded up to its own alignment,
//! and a uniform struct's alignment is at least 16. A `vec3` followed by an
//! `f32` therefore places the scalar at byte 12; the 16-byte hole appears
//! before the next `vec3`, not before that scalar.
//!
//! The vector types are [`GpuVec2`], [`GpuVec3`], and [`GpuVec4`]. Those names
//! stay distinct from `glam::Vec2` / `Vec3` / `Vec4`, which the rest of the
//! workspace already calls.

/// Byte size of an `f32` / `u32` / `i32` lane.
pub const F32_SIZE: u32 = 4;
/// Alignment of a 32-bit scalar.
pub const F32_ALIGN: u32 = 4;
/// Byte size of a two-lane vector.
pub const VEC2_SIZE: u32 = 8;
/// Alignment of a two-lane vector.
pub const VEC2_ALIGN: u32 = 8;
/// Byte size of a three-lane vector (not padded to 16).
pub const VEC3_SIZE: u32 = 12;
/// Alignment of a three-lane vector.
pub const VEC3_ALIGN: u32 = 16;
/// Byte size and alignment of a four-lane vector.
pub const VEC4_SIZE: u32 = 16;
/// Alignment of a four-lane vector.
pub const VEC4_ALIGN: u32 = 16;
/// Minimum alignment of a struct in the uniform address space.
pub const UNIFORM_STRUCT_ALIGN: u32 = 16;
/// `ornis_gpu.types` tag for an `f32` field.
pub const TYPE_TAG_F32: u8 = 1;
/// `ornis_gpu.types` tag for a `u32` field.
pub const TYPE_TAG_U32: u8 = 2;
/// `ornis_gpu.types` tag for an `i32` field.
pub const TYPE_TAG_I32: u8 = 3;
/// `ornis_gpu.types` tag for a `vec2<f32>` field.
pub const TYPE_TAG_VEC2: u8 = 4;
/// `ornis_gpu.types` tag for a `vec3<f32>` field.
pub const TYPE_TAG_VEC3: u8 = 5;
/// `ornis_gpu.types` tag for a `vec4<f32>` field.
pub const TYPE_TAG_VEC4: u8 = 6;

/// One struct field's size and alignment, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldLayout {
    /// Unpadded byte size of the field.
    pub size: u32,
    /// Power-of-two alignment of the field.
    pub align: u32,
}

/// Round `value` up to a multiple of `align`.
///
/// `align` must be a non-zero power of two. Overflow panics.
pub const fn round_up(value: u32, align: u32) -> u32 {
    assert!(align.is_power_of_two(), "align must be a power of two");
    let mask = align - 1;
    assert!(
        value.checked_add(mask).is_some(),
        "layout round_up overflow"
    );
    value.wrapping_add(mask) & !mask
}

/// Array stride in the storage address space (`std430`-style).
pub const fn storage_stride(size: u32, align: u32) -> u32 {
    round_up(size, align)
}

/// Array stride in the uniform address space (at least 16 bytes).
pub const fn uniform_stride(size: u32, align: u32) -> u32 {
    round_up(storage_stride(size, align), UNIFORM_STRUCT_ALIGN)
}

/// Place `fields` in order and return `(offsets, struct span)`.
///
/// `offsets` must be the same length as `fields`. When `uniform` is set, the
/// struct alignment is raised to [`UNIFORM_STRUCT_ALIGN`].
pub const fn place_fields(fields: &[FieldLayout], uniform: bool, offsets: &mut [u32]) -> u32 {
    assert!(offsets.len() == fields.len(), "offset buffer length");
    let mut cursor = 0u32;
    let mut struct_align = F32_ALIGN;
    let mut index = 0usize;
    while index < fields.len() {
        let field = fields[index];
        assert!(field.align.is_power_of_two(), "field align");
        cursor = round_up(cursor, field.align);
        offsets[index] = cursor;
        assert!(
            cursor.checked_add(field.size).is_some(),
            "struct layout overflow"
        );
        cursor = cursor.wrapping_add(field.size);
        if field.align > struct_align {
            struct_align = field.align;
        }
        index += 1;
    }
    if uniform {
        struct_align = round_up(struct_align, UNIFORM_STRUCT_ALIGN);
    }
    round_up(cursor, struct_align)
}

/// Plain data whose bytes match a WGSL uniform or storage value.
///
/// # Safety
///
/// Implementors must be `repr(C)` (or a primitive), must contain no
/// references, `bool`s, enums, or other niches, and [`SIZE`](Self::SIZE) must
/// equal `size_of`. [`ALIGN`](Self::ALIGN) is the GPU alignment: a power of
/// two and a multiple of `align_of` (it may be stricter — `vec3` is 12 bytes
/// in Rust and 16-byte aligned on the GPU). Every bit pattern of [`SIZE`](Self::SIZE)
/// bytes must be a valid value so a host byte copy is not UB.
pub unsafe trait GpuSafeLayout: Copy + Send + Sync + 'static {
    /// Byte size of the value.
    const SIZE: u32;
    /// Required alignment, in bytes.
    const ALIGN: u32;
    /// Stride when this type is an element of a uniform array.
    const UNIFORM_STRIDE: u32;
    /// Stride when this type is an element of a storage array.
    const STORAGE_STRIDE: u32;
    /// Tag written into `ornis_gpu.types` when this type is a struct field.
    ///
    /// `0` is not a field. `u8` and arrays are host padding, and a struct's
    /// own tag is `0` because the section lists its fields, not the struct.
    const TYPE_TAG: u8;
}

macro_rules! impl_word {
    ($ty:ty, $tag:expr) => {
        // SAFETY: `$ty` is a 32-bit primitive with no niches. `SIZE` is
        // `size_of`; `ALIGN` is the GPU alignment of a scalar.
        unsafe impl GpuSafeLayout for $ty {
            const SIZE: u32 = F32_SIZE;
            const ALIGN: u32 = F32_ALIGN;
            const STORAGE_STRIDE: u32 = F32_SIZE;
            const UNIFORM_STRIDE: u32 = UNIFORM_STRUCT_ALIGN;
            const TYPE_TAG: u8 = $tag;
        }
    };
}

impl_word!(f32, TYPE_TAG_F32);
impl_word!(u32, TYPE_TAG_U32);
impl_word!(i32, TYPE_TAG_I32);

// SAFETY: `u8` has no niches. `SIZE` and `ALIGN` are 1, matching `size_of`.
unsafe impl GpuSafeLayout for u8 {
    const SIZE: u32 = 1;
    const ALIGN: u32 = 1;
    const STORAGE_STRIDE: u32 = 1;
    const UNIFORM_STRIDE: u32 = UNIFORM_STRUCT_ALIGN;
    const TYPE_TAG: u8 = 0;
}

/// Two-lane `f32` vector. Size 8. GPU alignment is 8; Rust alignment is 4.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuVec2 {
    /// X lane.
    pub x: f32,
    /// Y lane.
    pub y: f32,
}

/// Three-lane `f32` vector. Size 12.
///
/// The Rust alignment is 4 so a following `f32` in a `repr(C)` struct sits at
/// byte 12 (WGSL). [`GpuSafeLayout::ALIGN`] is 16: a binding that *is* a
/// `vec3` is still placed on a 16-byte boundary.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuVec3 {
    /// X lane.
    pub x: f32,
    /// Y lane.
    pub y: f32,
    /// Z lane.
    pub z: f32,
}

/// Four-lane `f32` vector. Size 16. GPU alignment is 16; Rust alignment is 4.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuVec4 {
    /// X lane.
    pub x: f32,
    /// Y lane.
    pub y: f32,
    /// Z lane.
    pub z: f32,
    /// W lane.
    pub w: f32,
}

// SAFETY: `repr(C)` of two `f32`s, so every bit pattern is valid. `SIZE` is
// `size_of`. `ALIGN` is the GPU alignment and a multiple of the Rust alignment.
unsafe impl GpuSafeLayout for GpuVec2 {
    const SIZE: u32 = VEC2_SIZE;
    const ALIGN: u32 = VEC2_ALIGN;
    const STORAGE_STRIDE: u32 = storage_stride(VEC2_SIZE, VEC2_ALIGN);
    const UNIFORM_STRIDE: u32 = uniform_stride(VEC2_SIZE, VEC2_ALIGN);
    const TYPE_TAG: u8 = TYPE_TAG_VEC2;
}

// SAFETY: `repr(C)` of three `f32`s. `SIZE` is 12 (`size_of`). `ALIGN` is 16,
// which is a multiple of the Rust alignment of 4.
unsafe impl GpuSafeLayout for GpuVec3 {
    const SIZE: u32 = VEC3_SIZE;
    const ALIGN: u32 = VEC3_ALIGN;
    const STORAGE_STRIDE: u32 = storage_stride(VEC3_SIZE, VEC3_ALIGN);
    const UNIFORM_STRIDE: u32 = uniform_stride(VEC3_SIZE, VEC3_ALIGN);
    const TYPE_TAG: u8 = TYPE_TAG_VEC3;
}

// SAFETY: `repr(C)` of four `f32`s. `SIZE` is `size_of`. `ALIGN` is the GPU
// alignment and a multiple of the Rust alignment.
unsafe impl GpuSafeLayout for GpuVec4 {
    const SIZE: u32 = VEC4_SIZE;
    const ALIGN: u32 = VEC4_ALIGN;
    const STORAGE_STRIDE: u32 = storage_stride(VEC4_SIZE, VEC4_ALIGN);
    const UNIFORM_STRIDE: u32 = uniform_stride(VEC4_SIZE, VEC4_ALIGN);
    const TYPE_TAG: u8 = TYPE_TAG_VEC4;
}

// SAFETY: an array of plain data is plain data. `SIZE` is `size_of`. `ALIGN`
// is at least the element's GPU alignment, so it is a multiple of `align_of`.
unsafe impl<T: GpuSafeLayout, const N: usize> GpuSafeLayout for [T; N] {
    const SIZE: u32 = {
        let size = core::mem::size_of::<Self>();
        assert!(size <= u32::MAX as usize, "array exceeds u32");
        size as u32
    };
    const ALIGN: u32 = {
        let align = core::mem::align_of::<Self>() as u32;
        if T::ALIGN > align { T::ALIGN } else { align }
    };
    const STORAGE_STRIDE: u32 = storage_stride(Self::SIZE, Self::ALIGN);
    const UNIFORM_STRIDE: u32 = uniform_stride(Self::SIZE, Self::ALIGN);
    const TYPE_TAG: u8 = 0;
}

/// Bytes of one `ornis_gpu.types` payload (magic `OGT1`).
///
/// Each slice is one struct's [`GpuSafeLayout::TYPE_TAG`] list, in id order.
/// A tag of `0` is host padding and is not a wasm field; callers drop those
/// before encoding. `None` means a count does not fit in `u16`.
#[cfg(not(target_arch = "wasm32"))]
pub fn encode_types_section(structs: &[&[u8]]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(b"OGT1");
    write_u16(&mut out, structs.len())?;
    for fields in structs {
        write_u16(&mut out, fields.len())?;
        out.extend_from_slice(fields);
    }
    Some(out)
}

#[cfg(not(target_arch = "wasm32"))]
fn write_u16(out: &mut Vec<u8>, value: usize) -> Option<()> {
    let value = u16::try_from(value).ok()?;
    out.extend_from_slice(&value.to_le_bytes());
    Some(())
}

const _: () = {
    assert!(core::mem::size_of::<GpuVec2>() == VEC2_SIZE as usize);
    assert!(core::mem::align_of::<GpuVec2>() == F32_ALIGN as usize);
    assert!(VEC2_ALIGN.is_multiple_of(F32_ALIGN));
    assert!(core::mem::size_of::<GpuVec3>() == VEC3_SIZE as usize);
    assert!(core::mem::align_of::<GpuVec3>() == F32_ALIGN as usize);
    assert!(VEC3_ALIGN.is_multiple_of(F32_ALIGN));
    assert!(core::mem::size_of::<GpuVec4>() == VEC4_SIZE as usize);
    assert!(core::mem::align_of::<GpuVec4>() == F32_ALIGN as usize);
    assert!(VEC4_ALIGN.is_multiple_of(F32_ALIGN));
};
