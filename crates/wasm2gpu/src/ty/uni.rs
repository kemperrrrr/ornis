//! Uniform import names and the stride they share with host `gpu_struct!`.
//!
//! An `f32` element is 16 bytes in a uniform array and 4 bytes in storage.
//! The WGSL writer omits `@stride`, so a tighter natural stride is padded
//! with a lane struct rather than a raw array.

use ornis_gpu_abi::symbols;

use super::Elem;
use super::FieldKind;
use super::StructDef;
use super::elem_layout;
use super::named;

use crate::error::TranslateError;

/// Uniform access. Group and binding stay immediates, as on storage.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Uni {
    /// One field of a uniform struct. The field index is a constant.
    Field(FieldKind),
    /// `array<elem, len>` with a constant length. The index may be dynamic.
    Array(Elem),
    /// The binding is the struct value itself.
    Struct,
    /// `array<Struct, len>[index]`. Length is constant; the index may be dynamic.
    IndexStruct,
}

pub(super) fn imp(name: &str) -> Option<Uni> {
    if name == symbols::UNIFORM_READ_STRUCT {
        return Some(Uni::Struct);
    }
    if name == symbols::UNIFORM_INDEX_STRUCT {
        return Some(Uni::IndexStruct);
    }
    if let Some(kind) = named(name, FIELDS) {
        return Some(Uni::Field(kind));
    }
    named(name, ARRAYS).map(Uni::Array)
}

/// Byte stride of one element in a uniform array.
///
/// This is [`uniform_stride`](ornis_gpu_abi::uniform_stride), not the storage
/// stride. An `f32` element is 16 bytes here and 4 bytes in a storage array.
pub(crate) fn uniform_elem_stride(
    elem: Elem,
    structs: &[StructDef],
) -> Result<u32, TranslateError> {
    let layout = elem_layout(elem, structs)?;
    Ok(ornis_gpu_abi::uniform_stride(layout.size, layout.align))
}

/// `true` when WGSL's natural array stride is tighter than the uniform stride.
///
/// The writer does not emit `@align` / `@size`, so a scalar `f32` would come
/// back from the WGSL round-trip with stride 4. A lane struct of copies of
/// the element pads that stride out to 16 without changing byte 0.
pub(crate) fn uniform_needs_lane(
    elem: Elem,
    structs: &[StructDef],
) -> Result<bool, TranslateError> {
    let layout = elem_layout(elem, structs)?;
    let natural = ornis_gpu_abi::storage_stride(layout.size, layout.align);
    Ok(natural != ornis_gpu_abi::uniform_stride(layout.size, layout.align))
}

const FIELDS: &[(&str, FieldKind)] = &[
    (symbols::UNIFORM_FIELD_F32, FieldKind::F32),
    (symbols::UNIFORM_FIELD_U32, FieldKind::U32),
    (symbols::UNIFORM_FIELD_I32, FieldKind::I32),
    (symbols::UNIFORM_FIELD_VEC2, FieldKind::Vec2),
    (symbols::UNIFORM_FIELD_VEC3, FieldKind::Vec3),
    (symbols::UNIFORM_FIELD_VEC4, FieldKind::Vec4),
];

const ARRAYS: &[(&str, Elem)] = &[
    (symbols::UNIFORM_READ_F32, Elem::F32),
    (symbols::UNIFORM_READ_U32, Elem::U32),
    (symbols::UNIFORM_READ_I32, Elem::I32),
    (symbols::UNIFORM_READ_VEC2, Elem::Vec2),
    (symbols::UNIFORM_READ_VEC3, Elem::Vec3),
    (symbols::UNIFORM_READ_VEC4, Elem::Vec4),
];
