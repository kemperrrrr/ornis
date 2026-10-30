//! Storage element types and the layout they share with [`GpuSafeLayout`](ornis_gpu_abi::GpuSafeLayout).
//!
//! Structs are placed with [`place_fields`](ornis_gpu_abi::place_fields) in the
//! uniform (16-byte) alignment mode. That is the layout `gpu_struct!` gives a
//! host [`GpuRef`](ornis_gpu_abi::GpuRef), including when the binding is storage.
//! A scalar or vector uses that type's `STORAGE_STRIDE`.

use ornis_gpu_abi::FieldLayout;
use ornis_gpu_abi::GpuSafeLayout;
use ornis_gpu_abi::GpuVec2;
use ornis_gpu_abi::GpuVec3;
use ornis_gpu_abi::GpuVec4;
use ornis_gpu_abi::place_fields;
use ornis_gpu_abi::symbols;

use crate::error::TranslateError;
use crate::ir::Kind;

const TAG_F32: u8 = ornis_gpu_abi::TYPE_TAG_F32;
const TAG_U32: u8 = ornis_gpu_abi::TYPE_TAG_U32;
const TAG_I32: u8 = ornis_gpu_abi::TYPE_TAG_I32;
const TAG_VEC2: u8 = ornis_gpu_abi::TYPE_TAG_VEC2;
const TAG_VEC3: u8 = ornis_gpu_abi::TYPE_TAG_VEC3;
const TAG_VEC4: u8 = ornis_gpu_abi::TYPE_TAG_VEC4;
const MAX_FIELDS: usize = 8;
const STRUCT_HEADER: u32 = 4;
const PROV_PARAMS: u32 = 3;

mod sigs;
mod uni;

pub(super) use sigs::sig_ok;
pub(crate) use uni::Uni;
pub(crate) use uni::uniform_elem_stride;
pub(crate) use uni::uniform_needs_lane;

/// One field of a storage struct. Nested structs are rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum FieldKind {
    /// `f32`.
    F32,
    /// `u32`.
    U32,
    /// `i32`.
    I32,
    /// `vec2<f32>`.
    Vec2,
    /// `vec3<f32>`.
    Vec3,
    /// `vec4<f32>`.
    Vec4,
}

/// A struct declared in the types custom section.
#[derive(Clone, Debug)]
pub(super) struct StructDef {
    /// Fields in declaration order.
    pub(super) fields: Vec<FieldKind>,
}

/// Element of a storage array.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Elem {
    /// `f32`.
    F32,
    /// `u32`.
    U32,
    /// `i32`.
    I32,
    /// `vec2<f32>`.
    Vec2,
    /// `vec3<f32>`.
    Vec3,
    /// `vec4<f32>`.
    Vec4,
    /// Struct `id` from the types section.
    Struct(u16),
}

/// How an import moves values.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Imp {
    /// `storage_base`: the provenance pair folds into an address.
    Base,
    /// Uniform `raw_read` of a `vec4<f32>` at byte offset 0.
    ReadUniform,
    /// `texture_sample_2d`.
    Sample,
    /// Storage load of `elem`. `Struct` ids are an immediate operand.
    Read(Spec),
    /// Storage store of `elem`.
    Write(Spec),
    /// Uniform struct field or fixed-size array. See [`Uni`].
    Uniform(Uni),
}

/// Fixed element, or a struct id popped from the stack.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Spec {
    /// The element is named by the import.
    Fixed(Elem),
    /// The element is `Elem::Struct(id)` and `id` is the next immediate.
    Struct,
}

/// Byte layout of one storage element. `stride` is the array stride.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    /// Unpadded size. For a struct this is the placed span.
    pub(super) size: u32,
    /// Alignment used for the stride.
    pub(super) align: u32,
    /// Storage array stride. Matches `GpuSafeLayout::STORAGE_STRIDE`.
    pub(super) stride: u32,
    /// Field offsets. Empty unless this is a struct.
    pub(super) offsets: Vec<u32>,
}

/// Classify a toy import. `None` means the name is not a gpu opcode we lower.
pub(super) fn classify(name: &str) -> Option<Imp> {
    if name == symbols::STORAGE_BASE {
        return Some(Imp::Base);
    }
    if name == symbols::RAW_READ {
        return Some(Imp::ReadUniform);
    }
    if name == symbols::TEXTURE_SAMPLE_2D {
        return Some(Imp::Sample);
    }
    if let Some(uni) = uni::imp(name) {
        return Some(Imp::Uniform(uni));
    }
    storage_imp(name)
}

/// Layout of `elem`, using `structs` for [`Elem::Struct`].
pub(super) fn elem_layout(elem: Elem, structs: &[StructDef]) -> Result<Layout, TranslateError> {
    match elem {
        Elem::F32 => Ok(scalar_layout::<f32>()),
        Elem::U32 => Ok(scalar_layout::<u32>()),
        Elem::I32 => Ok(scalar_layout::<i32>()),
        Elem::Vec2 => Ok(scalar_layout::<GpuVec2>()),
        Elem::Vec3 => Ok(scalar_layout::<GpuVec3>()),
        Elem::Vec4 => Ok(scalar_layout::<GpuVec4>()),
        Elem::Struct(id) => struct_layout_id(id, structs),
    }
}

/// Place `fields` the way `gpu_struct!` places a host struct.
pub(super) fn struct_layout(fields: &[FieldKind]) -> Result<Layout, TranslateError> {
    if fields.is_empty() {
        return Err(type_err("unsupported empty struct"));
    }
    if fields.len() > MAX_FIELDS {
        return Err(type_err("unsupported struct width"));
    }
    let mut specs = Vec::new();
    let mut index = 0usize;
    while index < fields.len() {
        let (size, align) = field_size_align(fields[index]);
        specs.push(FieldLayout { size, align });
        index += 1;
    }
    let mut offsets = vec![0u32; fields.len()];
    let span = place_fields(&specs, true, &mut offsets);
    let align = ornis_gpu_abi::UNIFORM_STRUCT_ALIGN;
    Ok(Layout {
        size: span,
        align,
        stride: ornis_gpu_abi::storage_stride(span, align),
        offsets,
    })
}

/// Flatten a struct into wasm result lanes, in field order.
pub(super) fn struct_lanes(fields: &[FieldKind]) -> Result<Vec<Kind>, TranslateError> {
    let mut lanes = Vec::new();
    let mut index = 0usize;
    while index < fields.len() {
        sigs::push_lanes(&mut lanes, fields[index])?;
        index += 1;
    }
    Ok(lanes)
}

/// Parse the `ornis_gpu.types` payload.
pub(super) fn parse_types(data: &[u8]) -> Result<Vec<StructDef>, TranslateError> {
    let mut cur = Cursor { data, at: 0 };
    if cur.take(MAGIC.len()) != Some(MAGIC) {
        return Err(type_err("ornis_gpu.types magic is not OGT1"));
    }
    let count = cur.u16()?;
    let mut structs = Vec::new();
    let mut index = 0u16;
    while index < count {
        structs.push(one_struct(&mut cur)?);
        index += 1;
    }
    if cur.at != data.len() {
        return Err(type_err("ornis_gpu.types has trailing bytes"));
    }
    Ok(structs)
}

/// Resolve a struct immediate. Out-of-range ids are a type error.
pub(super) fn struct_def(
    structs: &[StructDef],
    id: u32,
) -> Result<(u16, &StructDef), TranslateError> {
    let id = u16::try_from(id).map_err(|_| type_err("struct type id is out of range"))?;
    let def = structs
        .get(id as usize)
        .ok_or_else(|| type_err("struct type id is out of range"))?;
    Ok((id, def))
}

/// Field tag from the custom section.
pub(super) fn field_tag(tag: u8) -> Result<FieldKind, TranslateError> {
    match tag {
        TAG_F32 => Ok(FieldKind::F32),
        TAG_U32 => Ok(FieldKind::U32),
        TAG_I32 => Ok(FieldKind::I32),
        TAG_VEC2 => Ok(FieldKind::Vec2),
        TAG_VEC3 => Ok(FieldKind::Vec3),
        TAG_VEC4 => Ok(FieldKind::Vec4),
        _ => Err(type_err("unsupported storage field")),
    }
}

/// Wasm lanes of one field.
pub(super) fn field_lanes(kind: FieldKind) -> u32 {
    match kind {
        FieldKind::F32 | FieldKind::U32 | FieldKind::I32 => 1,
        FieldKind::Vec2 => 2,
        FieldKind::Vec3 => VEC3_LANES,
        FieldKind::Vec4 => VEC4_LANES,
    }
}

/// `true` when the field's lanes are `f32` rather than `i32`.
pub(super) fn field_is_float(kind: FieldKind) -> bool {
    matches!(
        kind,
        FieldKind::F32 | FieldKind::Vec2 | FieldKind::Vec3 | FieldKind::Vec4
    )
}

const VEC3_LANES: u32 = 3;
const VEC4_LANES: u32 = 4;
const MAGIC: &[u8] = b"OGT1";

fn storage_imp(name: &str) -> Option<Imp> {
    if let Some(spec) = named(name, READS) {
        return Some(Imp::Read(spec));
    }
    named(name, WRITES).map(Imp::Write)
}

fn named<T: Copy>(name: &str, table: &[(&str, T)]) -> Option<T> {
    let mut index = 0usize;
    while index < table.len() {
        if table[index].0 == name {
            return Some(table[index].1);
        }
        index += 1;
    }
    None
}

const READS: &[(&str, Spec)] = &[
    (symbols::STORAGE_READ, Spec::Fixed(Elem::Vec4)),
    (symbols::STORAGE_READ_F32, Spec::Fixed(Elem::F32)),
    (symbols::STORAGE_READ_U32, Spec::Fixed(Elem::U32)),
    (symbols::STORAGE_READ_I32, Spec::Fixed(Elem::I32)),
    (symbols::STORAGE_READ_VEC2, Spec::Fixed(Elem::Vec2)),
    (symbols::STORAGE_READ_VEC3, Spec::Fixed(Elem::Vec3)),
    (symbols::STORAGE_READ_STRUCT, Spec::Struct),
];

const WRITES: &[(&str, Spec)] = &[
    (symbols::STORAGE_WRITE, Spec::Fixed(Elem::Vec4)),
    (symbols::STORAGE_WRITE_F32, Spec::Fixed(Elem::F32)),
    (symbols::STORAGE_WRITE_U32, Spec::Fixed(Elem::U32)),
    (symbols::STORAGE_WRITE_I32, Spec::Fixed(Elem::I32)),
    (symbols::STORAGE_WRITE_VEC2, Spec::Fixed(Elem::Vec2)),
    (symbols::STORAGE_WRITE_VEC3, Spec::Fixed(Elem::Vec3)),
    (symbols::STORAGE_WRITE_STRUCT, Spec::Struct),
];

pub(super) fn scalar_layout<T: GpuSafeLayout>() -> Layout {
    Layout {
        size: T::SIZE,
        align: T::ALIGN,
        stride: T::STORAGE_STRIDE,
        offsets: Vec::new(),
    }
}

pub(super) fn field_size_align(kind: FieldKind) -> (u32, u32) {
    match kind {
        FieldKind::F32 => (f32::SIZE, f32::ALIGN),
        FieldKind::U32 => (u32::SIZE, u32::ALIGN),
        FieldKind::I32 => (i32::SIZE, i32::ALIGN),
        FieldKind::Vec2 => (GpuVec2::SIZE, GpuVec2::ALIGN),
        FieldKind::Vec3 => (GpuVec3::SIZE, GpuVec3::ALIGN),
        FieldKind::Vec4 => (GpuVec4::SIZE, GpuVec4::ALIGN),
    }
}

pub(super) fn struct_layout_id(id: u16, structs: &[StructDef]) -> Result<Layout, TranslateError> {
    let def = structs
        .get(id as usize)
        .ok_or_else(|| type_err("struct type id is out of range"))?;
    struct_layout(&def.fields)
}

fn one_struct(cur: &mut Cursor<'_>) -> Result<StructDef, TranslateError> {
    let count = cur.u16()? as usize;
    if count == 0 || count > MAX_FIELDS {
        return Err(type_err("unsupported struct width"));
    }
    let mut fields = Vec::new();
    let mut index = 0usize;
    while index < count {
        fields.push(field_tag(cur.u8()?)?);
        index += 1;
    }
    Ok(StructDef { fields })
}

struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let end = self.at.checked_add(n)?;
        let slice = self.data.get(self.at..end)?;
        self.at = end;
        Some(slice)
    }

    fn u8(&mut self) -> Result<u8, TranslateError> {
        self.take(1)
            .and_then(|b| b.first().copied())
            .ok_or_else(|| type_err("ornis_gpu.types is truncated"))
    }

    fn u16(&mut self) -> Result<u16, TranslateError> {
        let bytes = self
            .take(2)
            .ok_or_else(|| type_err("ornis_gpu.types is truncated"))?;
        let pair: [u8; 2] = [bytes[0], bytes[1]];
        Ok(u16::from_le_bytes(pair))
    }
}

fn type_err(msg: &str) -> TranslateError {
    TranslateError::UnsupportedType(msg.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_gpu_abi::gpu_struct;

    gpu_struct! {
        struct TintHost {
            color: GpuVec3,
            intensity: f32,
        }
    }

    #[test]
    fn vec3_then_f32_matches_host_gpu_ref() {
        let layout = struct_layout(&[FieldKind::Vec3, FieldKind::F32]).expect("layout");
        assert!(layout.offsets[0] == TintHost::FIELD_OFFSETS[0]);
        assert!(layout.offsets[1] == TintHost::FIELD_OFFSETS[1]);
        assert!(layout.offsets[1] == GpuVec3::SIZE);
        assert!(layout.size == TintHost::SIZE);
        assert!(layout.stride == TintHost::STORAGE_STRIDE);
    }

    #[test]
    fn two_vec3_fields_are_sixteen_apart() {
        let layout = struct_layout(&[FieldKind::Vec3, FieldKind::Vec3]).expect("layout");
        assert!(layout.offsets[1] == GpuVec3::ALIGN);
    }

    #[test]
    fn scalar_and_vector_strides_match_the_host() {
        assert!(elem_layout(Elem::F32, &[]).expect("f32").stride == f32::STORAGE_STRIDE);
        assert!(elem_layout(Elem::U32, &[]).expect("u32").stride == u32::STORAGE_STRIDE);
        assert!(elem_layout(Elem::I32, &[]).expect("i32").stride == i32::STORAGE_STRIDE);
        assert!(elem_layout(Elem::Vec2, &[]).expect("v2").stride == GpuVec2::STORAGE_STRIDE);
        assert!(elem_layout(Elem::Vec3, &[]).expect("v3").stride == GpuVec3::STORAGE_STRIDE);
        assert!(elem_layout(Elem::Vec4, &[]).expect("v4").stride == GpuVec4::STORAGE_STRIDE);
    }

    #[test]
    fn unknown_field_tag_is_rejected() {
        let err = field_tag(0).expect_err("tag");
        assert!(matches!(err, TranslateError::UnsupportedType(_)));
    }
}
