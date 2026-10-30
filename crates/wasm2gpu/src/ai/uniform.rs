//! Abstract interpretation of uniform field and array operands.
//!
//! Group, binding, struct id, field index, and array length must be constants.
//! A joined value used as one of those is [`TranslateError::AmbiguousProvenance`].
//! A runtime value used as a field index or a length is a typed rejection.
//! The element index may be a phi.

use super::Abs;
use super::Num;
use super::St;
use super::mem::pop_prov;
use super::mem::pop_sid;
use super::type_msg;
use super::value::push_elem;
use super::value::push_field;
use super::value::push_struct;

use crate::error::TranslateError;
use crate::ir::Import;
use crate::ty;
use crate::ty::Elem;
use crate::ty::FieldKind;
use crate::ty::StructDef;
use crate::ty::Uni;

pub(super) fn on_uniform(
    st: &mut St,
    import: &Import,
    structs: &[StructDef],
    uni: Uni,
) -> Result<(), TranslateError> {
    match uni {
        Uni::Field(kind) => field(st, import, structs, kind),
        Uni::Array(elem) => array(st, import, structs, elem),
        Uni::Struct => whole(st, import, structs),
        Uni::IndexStruct => indexed(st, import, structs),
    }
}

fn field(
    st: &mut St,
    import: &Import,
    structs: &[StructDef],
    expect: FieldKind,
) -> Result<(), TranslateError> {
    ty::sig_ok(
        &import.sig,
        ty::Imp::Uniform(Uni::Field(expect)),
        structs,
        None,
    )?;
    let index = pop_const(st, ConstKind::Field)?;
    let id = pop_sid(st)?;
    let kind = field_at(structs, id, index)?;
    if kind != expect {
        return Err(type_msg("uniform field type does not match the import"));
    }
    pop_prov(st)?;
    pop_prov(st)?;
    push_field(st, kind)
}

fn array(
    st: &mut St,
    import: &Import,
    structs: &[StructDef],
    elem: Elem,
) -> Result<(), TranslateError> {
    ty::sig_ok(
        &import.sig,
        ty::Imp::Uniform(Uni::Array(elem)),
        structs,
        None,
    )?;
    pop_index(st)?;
    let len = pop_const(st, ConstKind::Len)?;
    check_len(elem, len, structs)?;
    pop_prov(st)?;
    pop_prov(st)?;
    push_elem(st, elem, structs)
}

fn whole(st: &mut St, import: &Import, structs: &[StructDef]) -> Result<(), TranslateError> {
    let id = pop_sid(st)?;
    ty::sig_ok(
        &import.sig,
        ty::Imp::Uniform(Uni::Struct),
        structs,
        Some(id),
    )?;
    pop_prov(st)?;
    pop_prov(st)?;
    push_struct(st, id, structs)
}

fn indexed(st: &mut St, import: &Import, structs: &[StructDef]) -> Result<(), TranslateError> {
    let id = pop_sid(st)?;
    ty::sig_ok(
        &import.sig,
        ty::Imp::Uniform(Uni::IndexStruct),
        structs,
        Some(id),
    )?;
    pop_index(st)?;
    let len = pop_const(st, ConstKind::Len)?;
    check_len(Elem::Struct(id), len, structs)?;
    pop_prov(st)?;
    pop_prov(st)?;
    push_struct(st, id, structs)
}

fn check_len(elem: Elem, len: u32, structs: &[StructDef]) -> Result<(), TranslateError> {
    if len == 0 {
        return Err(type_msg("uniform array length must be positive"));
    }
    let stride = ty::uniform_elem_stride(elem, structs)?;
    let bytes = stride
        .checked_mul(len)
        .ok_or_else(|| type_msg("uniform binding exceeds 64KiB"))?;
    if bytes > crate::cx::UNIFORM_BYTES {
        return Err(type_msg("uniform binding exceeds 64KiB"));
    }
    Ok(())
}

fn field_at(structs: &[StructDef], id: u16, index: u32) -> Result<FieldKind, TranslateError> {
    let def = structs
        .get(id as usize)
        .ok_or_else(|| type_msg("struct type id is out of range"))?;
    let slot =
        usize::try_from(index).map_err(|_| type_msg("uniform field index is out of range"))?;
    def.fields
        .get(slot)
        .copied()
        .ok_or_else(|| type_msg("uniform field index is out of range"))
}

enum ConstKind {
    Field,
    Len,
}

fn pop_const(st: &mut St, kind: ConstKind) -> Result<u32, TranslateError> {
    let dynamic = match kind {
        ConstKind::Field => "uniform field index must be a constant",
        ConstKind::Len => "uniform array length must be a constant",
    };
    match st.stack.pop() {
        Some(Abs::I32(Num::Const(0))) if matches!(kind, ConstKind::Len) => {
            Err(type_msg("uniform array length must be positive"))
        }
        Some(Abs::I32(Num::Const(value))) => Ok(value),
        Some(Abs::I32(Num::Dyn { joined: true })) => Err(TranslateError::AmbiguousProvenance),
        Some(Abs::I32(Num::Dyn { joined: false })) => Err(type_msg(dynamic)),
        Some(_) => Err(type_msg(dynamic)),
        None => Err(TranslateError::Unsupported),
    }
}

fn pop_index(st: &mut St) -> Result<(), TranslateError> {
    match st.stack.pop() {
        Some(Abs::I32(_)) => Ok(()),
        Some(_) => Err(TranslateError::Unsupported),
        None => Err(TranslateError::Unsupported),
    }
}
