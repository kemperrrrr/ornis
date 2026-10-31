//! Compose stored lanes into naga values and address one element.

use crate::ix::Ix;

use naga::Expression;

use crate::cx::VEC3_LANES;
use crate::cx::Val;
use crate::env::Emit;
use crate::error::TranslateError;
use crate::ty::FieldKind;

pub(super) fn compose_struct(
    em: &mut Emit<'_, '_>,
    ty: naga::Handle<naga::Type>,
    fields: &[FieldKind],
    values: &[Val],
) -> Result<naga::Handle<Expression>, TranslateError> {
    let mut parts = Vec::new();
    let mut cursor = 0usize;
    for field in fields {
        let (next, expr) = take_field(em, *field, values, cursor)?;
        cursor = next;
        parts.push(expr);
    }
    let pads = pad_count(em, ty, fields.len())?;
    let mut index = 0u32;
    while index < pads {
        parts.push(em.u32(0));
        index += 1;
    }
    compose(em, ty, &parts)
}

fn pad_count(
    em: &Emit<'_, '_>,
    ty: naga::Handle<naga::Type>,
    fields: usize,
) -> Result<u32, TranslateError> {
    let naga::TypeInner::Struct { members, .. } = &em.cx.module.types[ty].inner else {
        return Err(TranslateError::Unsupported);
    };
    let extra = members
        .len()
        .checked_sub(fields)
        .ok_or(TranslateError::Unsupported)?;
    u32::try_from(extra).map_err(|_| TranslateError::Unsupported)
}

pub(super) fn take_field(
    em: &mut Emit<'_, '_>,
    field: FieldKind,
    values: &[Val],
    cursor: usize,
) -> Result<(usize, naga::Handle<Expression>), TranslateError> {
    match field {
        FieldKind::F32 => Ok((cursor + 1, float_at(em, values, cursor)?)),
        FieldKind::U32 => Ok((cursor + 1, uint_at(em, values, cursor)?)),
        FieldKind::I32 => {
            let bits = uint_at(em, values, cursor)?;
            Ok((cursor + 1, sint(em, bits)))
        }
        FieldKind::Vec2 => {
            let ty = em.cx.vec2;
            take_vec(em, values, cursor, 2, ty)
        }
        FieldKind::Vec3 => {
            let ty = em.cx.vec3;
            take_vec(em, values, cursor, VEC3_LANES, ty)
        }
        FieldKind::Vec4 => {
            let ty = em.cx.vec4;
            take_vec(em, values, cursor, crate::cx::LANES, ty)
        }
    }
}

pub(super) fn take_vec(
    em: &mut Emit<'_, '_>,
    values: &[Val],
    cursor: usize,
    n: u32,
    ty: naga::Handle<naga::Type>,
) -> Result<(usize, naga::Handle<Expression>), TranslateError> {
    let mut parts = Vec::new();
    let mut index = 0u32;
    while index < n {
        parts.push(float_at(em, values, cursor + index as usize)?);
        index += 1;
    }
    Ok((cursor + n as usize, compose(em, ty, &parts)?))
}

pub(super) fn float_at(
    _em: &mut Emit<'_, '_>,
    values: &[Val],
    index: usize,
) -> Result<naga::Handle<Expression>, TranslateError> {
    let val = *values.get(index).ok_or(TranslateError::Unsupported)?;
    crate::env::float_of(val)
}

pub(super) fn uint_at(
    em: &mut Emit<'_, '_>,
    values: &[Val],
    index: usize,
) -> Result<naga::Handle<Expression>, TranslateError> {
    let val = *values.get(index).ok_or(TranslateError::Unsupported)?;
    em.as_uint(val)
}

pub(super) fn compose(
    em: &mut Emit<'_, '_>,
    ty: naga::Handle<naga::Type>,
    parts: &[naga::Handle<Expression>],
) -> Result<naga::Handle<Expression>, TranslateError> {
    Ok(em.ex(Expression::Compose {
        ty,
        components: parts.to_vec(),
    }))
}

pub(super) fn pop_floats(
    em: &mut Emit<'_, '_>,
    n: u32,
) -> Result<Vec<naga::Handle<Expression>>, TranslateError> {
    let vals = pop_vals(em, n)?;
    let mut parts = Vec::new();
    for val in vals {
        parts.push(crate::env::float_of(val)?);
    }
    Ok(parts)
}

pub(super) fn pop_vals(em: &mut Emit<'_, '_>, n: u32) -> Result<Vec<Val>, TranslateError> {
    let count = usize::try_from(n).map_err(|_| TranslateError::Unsupported)?;
    em.take(count)
}

pub(super) fn pop_prov(em: &mut Emit<'_, '_>) -> Result<(u32, u32), TranslateError> {
    let binding = em.pop_imm()?;
    let group = em.pop_imm()?;
    Ok((group, binding))
}

pub(super) fn element_ptr(
    em: &mut Emit<'_, '_>,
    var: naga::Handle<naga::GlobalVariable>,
    index: Ix,
) -> naga::Handle<Expression> {
    let base = em.ex(Expression::GlobalVariable(var));
    let index = match index {
        Ix::Const(value) => em.u32(value),
        Ix::Dyn(handle) => handle,
    };
    em.ex(Expression::Access { base, index })
}

pub(super) fn sint(
    em: &mut Emit<'_, '_>,
    bits: naga::Handle<Expression>,
) -> naga::Handle<Expression> {
    em.ex(Expression::As {
        expr: bits,
        kind: naga::ScalarKind::Sint,
        convert: None,
    })
}

pub(super) fn uint_bits(
    em: &mut Emit<'_, '_>,
    bits: naga::Handle<Expression>,
) -> naga::Handle<Expression> {
    em.ex(Expression::As {
        expr: bits,
        kind: naga::ScalarKind::Uint,
        convert: None,
    })
}
