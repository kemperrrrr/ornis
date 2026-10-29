//! Storage and arithmetic steps of the address lattice.

use super::Abs;
use super::Ai;
use super::Fold;
use super::Num;
use super::STRUCT_HEADER;
use super::St;
use super::fold;
use super::join::join_abs;
use super::local_get;
use super::local_set;
use super::pop;
use super::pop_num;
use super::push;
use super::type_msg;

use crate::error::TranslateError;
use crate::ir::Import;
use crate::ir::Op;
use crate::ty;
use crate::ty::Elem;
use crate::ty::Imp;
use crate::ty::Spec;
use crate::ty::StructDef;

pub(super) fn on_op(ai: &mut Ai<'_>, st: &mut St, op: Op) -> Result<(), TranslateError> {
    match op {
        Op::I32(value) => push(st, Abs::I32(Num::Const(value))),
        Op::F32(_) => push(st, Abs::F32),
        Op::Local(index) | Op::LocalI32(index) => local_get(st, index),
        Op::LocalSet(index) => local_set(st, index, false),
        Op::LocalTee(index) => local_set(st, index, true),
        Op::Drop => pop(st).map(|_| ()),
        Op::Call(index) => call(ai, st, index),
        Op::Mul | Op::Add => f32_bin(st),
        Op::I32Add => i32_bin(st, u32::wrapping_add),
        Op::I32Mul => i32_bin(st, u32::wrapping_mul),
        Op::I32Sub => i32_bin(st, u32::wrapping_sub),
        Op::I32Eqz => i32_eqz(st),
        Op::I32Eq => i32_cmp(st, |a, b| a == b),
        Op::I32Ne => i32_cmp(st, |a, b| a != b),
        Op::I32LtU => i32_cmp(st, |a, b| a < b),
        Op::I32GtU => i32_cmp(st, |a, b| a > b),
        Op::Select => select(st),
    }
}

pub(super) fn call(ai: &Ai<'_>, st: &mut St, index: u32) -> Result<(), TranslateError> {
    let import = ai
        .imports
        .get(index as usize)
        .ok_or(TranslateError::Unsupported)?;
    let Some(imp) = ty::classify(&import.name) else {
        return Err(TranslateError::Unsupported);
    };
    match imp {
        Imp::ReadUniform => read_uniform(st),
        Imp::Sample => sample(st, import),
        Imp::Read(spec) => read_storage(ai, st, import, spec),
        Imp::Write(spec) => write_storage(ai, st, import, spec),
    }
}

pub(super) fn read_uniform(st: &mut St) -> Result<(), TranslateError> {
    let offset = pop_num(st)?;
    pop_prov(st)?;
    pop_prov(st)?;
    match offset {
        Num::Const(0) => {}
        Num::Const(_) => return Err(TranslateError::Unsupported),
        Num::Dyn { joined: true } => return Err(TranslateError::AmbiguousProvenance),
        Num::Dyn { joined: false } => return Err(TranslateError::DynamicBinding),
    }
    push(st, Abs::Vec)
}

pub(super) fn sample(st: &mut St, import: &Import) -> Result<(), TranslateError> {
    ty::sig_ok(&import.sig, Imp::Sample, &[], None)?;
    pop_f32(st)?;
    pop_f32(st)?;
    pop_prov(st)?;
    pop_prov(st)?;
    pop_prov(st)?;
    pop_prov(st)?;
    push(st, Abs::Vec)
}

pub(super) fn read_storage(
    ai: &Ai<'_>,
    st: &mut St,
    import: &Import,
    spec: Spec,
) -> Result<(), TranslateError> {
    let elem = resolve_read(ai, st, import, spec)?;
    pop_index(st)?;
    pop_prov(st)?;
    pop_prov(st)?;
    push_elem(st, elem, ai.structs)
}

pub(super) fn write_storage(
    ai: &Ai<'_>,
    st: &mut St,
    import: &Import,
    spec: Spec,
) -> Result<(), TranslateError> {
    let elem = resolve_write(ai, st, import, spec)?;
    pop_index(st)?;
    pop_prov(st)?;
    pop_prov(st)?;
    let _ = elem;
    Ok(())
}

pub(super) fn resolve_read(
    ai: &Ai<'_>,
    st: &mut St,
    import: &Import,
    spec: Spec,
) -> Result<Elem, TranslateError> {
    match spec {
        Spec::Fixed(elem) => {
            ty::sig_ok(&import.sig, Imp::Read(spec), ai.structs, None)?;
            Ok(elem)
        }
        Spec::Struct => {
            let id = pop_sid(st)?;
            ty::sig_ok(&import.sig, Imp::Read(spec), ai.structs, Some(id))?;
            Ok(Elem::Struct(id))
        }
    }
}

pub(super) fn resolve_write(
    ai: &Ai<'_>,
    st: &mut St,
    import: &Import,
    spec: Spec,
) -> Result<Elem, TranslateError> {
    match spec {
        Spec::Fixed(elem) => {
            ty::sig_ok(&import.sig, Imp::Write(spec), ai.structs, None)?;
            pop_n(st, lanes_of(elem)?)?;
            Ok(elem)
        }
        Spec::Struct => {
            let n = lane_params(import)?;
            pop_n(st, n)?;
            let id = pop_sid(st)?;
            ty::sig_ok(&import.sig, Imp::Write(spec), ai.structs, Some(id))?;
            Ok(Elem::Struct(id))
        }
    }
}

pub(super) fn push_elem(
    st: &mut St,
    elem: Elem,
    structs: &[StructDef],
) -> Result<(), TranslateError> {
    match elem {
        Elem::F32 => push(st, Abs::F32),
        Elem::U32 | Elem::I32 => push(st, Abs::I32(Num::Dyn { joined: false })),
        Elem::Vec2 => push_f32s(st, 2),
        Elem::Vec3 => push_f32s(st, crate::cx::VEC3_LANES),
        Elem::Vec4 => push(st, Abs::Vec),
        Elem::Struct(id) => push_struct(st, id, structs),
    }
}

pub(super) fn push_struct(
    st: &mut St,
    id: u16,
    structs: &[StructDef],
) -> Result<(), TranslateError> {
    let def = structs
        .get(id as usize)
        .ok_or_else(|| type_msg("struct type id is out of range"))?;
    for field in &def.fields {
        push_field(st, *field)?;
    }
    Ok(())
}

pub(super) fn push_field(st: &mut St, field: crate::ty::FieldKind) -> Result<(), TranslateError> {
    if ty::field_is_float(field) {
        push_f32s(st, ty::field_lanes(field))
    } else {
        let mut index = 0u32;
        let n = ty::field_lanes(field);
        while index < n {
            push(st, Abs::I32(Num::Dyn { joined: false }))?;
            index += 1;
        }
        Ok(())
    }
}

pub(super) fn lanes_of(elem: Elem) -> Result<u32, TranslateError> {
    match elem {
        Elem::F32 | Elem::U32 | Elem::I32 => Ok(1),
        Elem::Vec2 => Ok(2),
        Elem::Vec3 => Ok(crate::cx::VEC3_LANES),
        Elem::Vec4 => Ok(crate::cx::LANES),
        Elem::Struct(_) => Err(type_msg("struct lanes need a type id")),
    }
}

pub(super) fn lane_params(import: &Import) -> Result<u32, TranslateError> {
    import
        .sig
        .params
        .checked_sub(STRUCT_HEADER)
        .ok_or(TranslateError::Unsupported)
}

pub(super) fn push_f32s(st: &mut St, n: u32) -> Result<(), TranslateError> {
    let mut index = 0u32;
    while index < n {
        push(st, Abs::F32)?;
        index += 1;
    }
    Ok(())
}

pub(super) fn pop_n(st: &mut St, n: u32) -> Result<(), TranslateError> {
    let mut index = 0u32;
    while index < n {
        pop(st)?;
        index += 1;
    }
    Ok(())
}

pub(super) fn pop_sid(st: &mut St) -> Result<u16, TranslateError> {
    match st.stack.pop() {
        Some(Abs::I32(Num::Const(value))) => {
            u16::try_from(value).map_err(|_| type_msg("struct type id is out of range"))
        }
        Some(Abs::I32(Num::Dyn { joined: true })) => Err(TranslateError::AmbiguousProvenance),
        Some(_) => Err(type_msg("struct type id is not a constant")),
        None => Err(TranslateError::Unsupported),
    }
}

pub(super) fn pop_prov(st: &mut St) -> Result<(), TranslateError> {
    match st.stack.pop() {
        Some(Abs::I32(Num::Const(_))) => Ok(()),
        Some(Abs::I32(Num::Dyn { joined: true })) => Err(TranslateError::AmbiguousProvenance),
        Some(Abs::I32(Num::Dyn { joined: false }) | Abs::Bot) => {
            Err(TranslateError::DynamicBinding)
        }
        Some(_) => Err(TranslateError::DynamicBinding),
        None => Err(TranslateError::Unsupported),
    }
}

pub(super) fn pop_index(st: &mut St) -> Result<(), TranslateError> {
    match st.stack.pop() {
        Some(Abs::I32(_) | Abs::Bot) => Ok(()),
        Some(_) => Err(TranslateError::Unsupported),
        None => Err(TranslateError::Unsupported),
    }
}

pub(super) fn pop_f32(st: &mut St) -> Result<(), TranslateError> {
    match pop(st)? {
        Abs::F32 => Ok(()),
        _ => Err(TranslateError::Unsupported),
    }
}

pub(super) fn f32_bin(st: &mut St) -> Result<(), TranslateError> {
    let right = pop(st)?;
    let left = pop(st)?;
    match (left, right) {
        (Abs::F32, Abs::F32) => push(st, Abs::F32),
        (Abs::Vec, Abs::Vec) => push(st, Abs::Vec),
        _ => Err(TranslateError::Unsupported),
    }
}

pub(super) fn i32_bin(st: &mut St, op: impl Fn(u32, u32) -> u32) -> Result<(), TranslateError> {
    let right = pop_num(st)?;
    let left = pop_num(st)?;
    let num = match (left, right) {
        (Num::Const(a), Num::Const(b)) => Num::Const(op(a, b)),
        _ => Num::Dyn { joined: false },
    };
    push(st, Abs::I32(num))
}

pub(super) fn i32_cmp(st: &mut St, pred: fn(u32, u32) -> bool) -> Result<(), TranslateError> {
    i32_bin(st, |a, b| u32::from(pred(a, b)))
}

pub(super) fn i32_eqz(st: &mut St) -> Result<(), TranslateError> {
    let num = match pop_num(st)? {
        Num::Const(value) => Num::Const(u32::from(value == 0)),
        Num::Dyn { .. } => Num::Dyn { joined: false },
    };
    push(st, Abs::I32(num))
}

pub(super) fn select(st: &mut St) -> Result<(), TranslateError> {
    let cond = pop_num(st)?;
    let reject = pop(st)?;
    let accept = pop(st)?;
    let value = match fold(cond) {
        Fold::True => accept,
        Fold::False => reject,
        Fold::Dyn => join_abs(accept, reject)?,
    };
    push(st, value)
}
