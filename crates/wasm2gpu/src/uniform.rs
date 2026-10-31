//! Uniform loads: struct fields and fixed-size arrays.
//!
//! Group and binding are immediates, the same provenance contract as storage.
//! A struct field index is a constant (`AccessIndex`). An array length is a
//! constant. The element index may be a runtime `u32` and becomes `Access`,
//! which WGSL allows on a fixed-size array in the uniform address space.
//! Field layout is [`place_fields`](ornis_gpu_abi::place_fields) with uniform
//! alignment, matching host `gpu_struct!`. A dynamic field index, a dynamic
//! length, a zero length, and a binding larger than 64 KiB are rejected.

use naga::Expression;

use crate::cx::UniformShape;
use crate::cx::Val;
use crate::env::Emit;
use crate::error::TranslateError;
use crate::globals;
use crate::ix::Ix;
use crate::ix::pop_index;
use crate::storage;
use crate::ty::Elem;
use crate::ty::FieldKind;
use crate::ty::Uni;

pub(super) fn load(em: &mut Emit<'_, '_>, uni: Uni) -> Result<(), TranslateError> {
    match uni {
        Uni::Field(kind) => field(em, kind),
        Uni::Array(elem) => array(em, elem),
        Uni::Struct => whole_struct(em),
        Uni::IndexStruct => index_struct(em),
    }
}

fn field(em: &mut Emit<'_, '_>, expect: FieldKind) -> Result<(), TranslateError> {
    let index = pop_field(em)?;
    let id = pop_sid(em)?;
    let (group, binding) = pop_prov(em)?;
    let kind = field_at(em, id, index)?;
    if kind != expect {
        return Err(msg("uniform field type does not match the import"));
    }
    let shape = UniformShape::Bare(Elem::Struct(id));
    let var = globals::uniform_global(em.cx, group, binding, shape, em.structs)?;
    let base = em.ex(Expression::GlobalVariable(var));
    let pointer = em.ex(Expression::AccessIndex { base, index });
    let loaded = em.ex(Expression::Load { pointer });
    storage::push_field(em, loaded, kind)
}

fn array(em: &mut Emit<'_, '_>, elem: Elem) -> Result<(), TranslateError> {
    let index = pop_index(em)?;
    let len = pop_len(em)?;
    let (group, binding) = pop_prov(em)?;
    fits(em, elem, len)?;
    let pointer = array_pointer(em, group, binding, UniformShape::Array { elem, len }, index)?;
    storage::push_elem(em, elem, pointer)
}

fn whole_struct(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let id = pop_sid(em)?;
    let (group, binding) = pop_prov(em)?;
    let shape = UniformShape::Bare(Elem::Struct(id));
    let var = globals::uniform_global(em.cx, group, binding, shape, em.structs)?;
    let pointer = em.ex(Expression::GlobalVariable(var));
    let loaded = em.ex(Expression::Load { pointer });
    storage::push_struct(em, loaded, id)
}

fn index_struct(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let id = pop_sid(em)?;
    let index = pop_index(em)?;
    let len = pop_len(em)?;
    let (group, binding) = pop_prov(em)?;
    let elem = Elem::Struct(id);
    fits(em, elem, len)?;
    let pointer = array_pointer(em, group, binding, UniformShape::Array { elem, len }, index)?;
    let loaded = em.ex(Expression::Load { pointer });
    storage::push_struct(em, loaded, id)
}

fn fits(em: &Emit<'_, '_>, elem: Elem, len: u32) -> Result<(), TranslateError> {
    let stride = crate::ty::uniform_elem_stride(elem, em.structs)?;
    let bytes = stride
        .checked_mul(len)
        .ok_or_else(|| msg("uniform binding exceeds 64KiB"))?;
    if bytes > crate::cx::UNIFORM_BYTES {
        return Err(msg("uniform binding exceeds 64KiB"));
    }
    Ok(())
}

fn field_at(em: &Emit<'_, '_>, id: u16, index: u32) -> Result<FieldKind, TranslateError> {
    let def = em
        .structs
        .get(id as usize)
        .ok_or_else(|| msg("struct type id is out of range"))?;
    let slot = usize::try_from(index).map_err(|_| msg("uniform field index is out of range"))?;
    def.fields
        .get(slot)
        .copied()
        .ok_or_else(|| msg("uniform field index is out of range"))
}

fn pop_field(em: &mut Emit<'_, '_>) -> Result<u32, TranslateError> {
    match em.pop()? {
        Val::Imm(value) => Ok(value),
        Val::Index(_) | Val::Bool(_) => Err(msg("uniform field index must be a constant")),
        _ => Err(TranslateError::Unsupported),
    }
}

fn pop_len(em: &mut Emit<'_, '_>) -> Result<u32, TranslateError> {
    match em.pop()? {
        Val::Imm(0) => Err(msg("uniform array length must be positive")),
        Val::Imm(value) => Ok(value),
        Val::Index(_) | Val::Bool(_) => Err(msg("uniform array length must be a constant")),
        _ => Err(TranslateError::Unsupported),
    }
}

fn pop_sid(em: &mut Emit<'_, '_>) -> Result<u16, TranslateError> {
    let id = em.pop_imm()?;
    let (id, _) = crate::ty::struct_def(em.structs, id)?;
    Ok(id)
}

fn pop_prov(em: &mut Emit<'_, '_>) -> Result<(u32, u32), TranslateError> {
    let binding = em.pop_imm()?;
    let group = em.pop_imm()?;
    Ok((group, binding))
}

fn array_pointer(
    em: &mut Emit<'_, '_>,
    group: u32,
    binding: u32,
    shape: UniformShape,
    index: Ix,
) -> Result<naga::Handle<Expression>, TranslateError> {
    let UniformShape::Array { elem, .. } = shape else {
        return Err(TranslateError::Unsupported);
    };
    let var = globals::uniform_global(em.cx, group, binding, shape, em.structs)?;
    element_ptr(em, var, elem, index)
}

fn element_ptr(
    em: &mut Emit<'_, '_>,
    var: naga::Handle<naga::GlobalVariable>,
    elem: Elem,
    index: Ix,
) -> Result<naga::Handle<Expression>, TranslateError> {
    let global = em.ex(Expression::GlobalVariable(var));
    let array = em.ex(Expression::AccessIndex {
        base: global,
        index: 0,
    });
    let index = match index {
        Ix::Const(value) => em.u32(value),
        Ix::Dyn(handle) => handle,
    };
    let mut pointer = em.ex(Expression::Access { base: array, index });
    if crate::ty::uniform_needs_lane(elem, em.structs)? {
        pointer = em.ex(Expression::AccessIndex {
            base: pointer,
            index: 0,
        });
    }
    Ok(pointer)
}

fn msg(text: &str) -> TranslateError {
    TranslateError::UnsupportedType(text.to_string())
}
