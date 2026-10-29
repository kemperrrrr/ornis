//! Storage load and store: one global per (group, binding, element).
//!
//! Group and binding are immediates. The element index may be a runtime
//! `u32` and becomes an access chain. A `vec4<f32>` load stays one value so
//! existing straight-line shaders keep their stack. Other elements push one
//! wasm lane per scalar component. A struct uses the host field offsets.

use naga::Expression;

use crate::cx::VEC3_LANES;
use crate::cx::Val;
use crate::cx::sp;
use crate::env::Emit;
use crate::error::TranslateError;
use crate::globals;
use crate::ty::Elem;
use crate::ty::FieldKind;
use crate::ty::Spec;

mod parts;

use parts::compose;
use parts::compose_struct;
use parts::element_ptr;
use parts::pop_floats;
use parts::pop_index;
use parts::pop_prov;
use parts::pop_vals;
use parts::sint;
use parts::uint_bits;

const STRUCT_HEADER: u32 = 4;

enum Ix {
    Const(u32),
    Dyn(naga::Handle<Expression>),
}

pub(super) fn load(em: &mut Emit<'_, '_>, spec: Spec) -> Result<(), TranslateError> {
    let elem = read_elem(em, spec)?;
    let index = pop_index(em)?;
    let (group, binding) = pop_prov(em)?;
    let var = globals::storage(em.cx, group, binding, elem, em.structs)?;
    let pointer = element_ptr(em, var, index);
    push_elem(em, elem, pointer)
}

pub(super) fn store(em: &mut Emit<'_, '_>, spec: Spec, import: u32) -> Result<(), TranslateError> {
    let (elem, value) = pop_stored(em, spec, import)?;
    let index = pop_index(em)?;
    let (group, binding) = pop_prov(em)?;
    let var = globals::storage(em.cx, group, binding, elem, em.structs)?;
    let pointer = element_ptr(em, var, index);
    em.block
        .push(naga::Statement::Store { pointer, value }, sp());
    Ok(())
}

fn read_elem(em: &mut Emit<'_, '_>, spec: Spec) -> Result<Elem, TranslateError> {
    match spec {
        Spec::Fixed(elem) => Ok(elem),
        Spec::Struct => {
            let id = em.pop_imm()?;
            let (id, _) = crate::ty::struct_def(em.structs, id)?;
            Ok(Elem::Struct(id))
        }
    }
}

fn pop_stored(
    em: &mut Emit<'_, '_>,
    spec: Spec,
    import: u32,
) -> Result<(Elem, naga::Handle<Expression>), TranslateError> {
    match spec {
        Spec::Fixed(elem) => Ok((elem, pop_fixed(em, elem)?)),
        Spec::Struct => pop_struct(em, import),
    }
}

fn pop_struct(
    em: &mut Emit<'_, '_>,
    import: u32,
) -> Result<(Elem, naga::Handle<Expression>), TranslateError> {
    let params = em
        .imports
        .get(import as usize)
        .ok_or(TranslateError::Unsupported)?
        .sig
        .params;
    let lanes = params
        .checked_sub(STRUCT_HEADER)
        .ok_or(TranslateError::Unsupported)?;
    let values = pop_vals(em, lanes)?;
    let id = em.pop_imm()?;
    let (id, def) = crate::ty::struct_def(em.structs, id)?;
    let fields = def.fields.clone();
    let ty = globals::struct_ty(em.cx, id, &fields)?;
    let value = compose_struct(em, ty, &fields, &values)?;
    Ok((Elem::Struct(id), value))
}

fn pop_fixed(
    em: &mut Emit<'_, '_>,
    elem: Elem,
) -> Result<naga::Handle<Expression>, TranslateError> {
    match elem {
        Elem::F32 => {
            let val = em.pop()?;
            crate::env::float_of(val)
        }
        Elem::U32 => em.pop_uint(),
        Elem::I32 => {
            let val = em.pop()?;
            let bits = em.as_uint(val)?;
            Ok(sint(em, bits))
        }
        Elem::Vec2 => fixed_vec(em, 2, true),
        Elem::Vec3 => fixed_vec(em, VEC3_LANES, false),
        Elem::Vec4 => fixed_vec(em, crate::cx::LANES, false),
        Elem::Struct(_) => Err(TranslateError::Unsupported),
    }
}

fn push_elem(
    em: &mut Emit<'_, '_>,
    elem: Elem,
    pointer: naga::Handle<Expression>,
) -> Result<(), TranslateError> {
    let loaded = em.ex(Expression::Load { pointer });
    match elem {
        Elem::F32 => em.push(Val::Scalar(loaded)),
        Elem::U32 => em.push(Val::Index(loaded)),
        Elem::I32 => {
            let bits = uint_bits(em, loaded);
            em.push(Val::Index(bits))
        }
        Elem::Vec2 => push_lanes(em, loaded, 2),
        Elem::Vec3 => push_lanes(em, loaded, VEC3_LANES),
        Elem::Vec4 => em.push(Val::Vec(loaded)),
        Elem::Struct(id) => push_struct(em, loaded, id),
    }
}

fn push_struct(
    em: &mut Emit<'_, '_>,
    loaded: naga::Handle<Expression>,
    id: u16,
) -> Result<(), TranslateError> {
    let def = em
        .structs
        .get(id as usize)
        .ok_or(TranslateError::Unsupported)?;
    let fields = def.fields.clone();
    let mut index = 0u32;
    while (index as usize) < fields.len() {
        let field = em.ex(Expression::AccessIndex {
            base: loaded,
            index,
        });
        push_field(em, field, fields[index as usize])?;
        index += 1;
    }
    Ok(())
}

fn push_field(
    em: &mut Emit<'_, '_>,
    field: naga::Handle<Expression>,
    kind: FieldKind,
) -> Result<(), TranslateError> {
    match kind {
        FieldKind::F32 => em.push(Val::Scalar(field)),
        FieldKind::U32 => em.push(Val::Index(field)),
        FieldKind::I32 => {
            let bits = uint_bits(em, field);
            em.push(Val::Index(bits))
        }
        FieldKind::Vec2 => push_lanes(em, field, 2),
        FieldKind::Vec3 => push_lanes(em, field, VEC3_LANES),
        FieldKind::Vec4 => push_lanes(em, field, crate::cx::LANES),
    }
}

fn push_lanes(
    em: &mut Emit<'_, '_>,
    base: naga::Handle<Expression>,
    n: u32,
) -> Result<(), TranslateError> {
    let mut index = 0u32;
    while index < n {
        let lane = em.ex(Expression::AccessIndex { base, index });
        em.stack.push(Val::Scalar(lane));
        index += 1;
    }
    Ok(())
}

fn fixed_vec(
    em: &mut Emit<'_, '_>,
    n: u32,
    bi: bool,
) -> Result<naga::Handle<Expression>, TranslateError> {
    let ty = if bi {
        em.cx.vec2
    } else if n == VEC3_LANES {
        em.cx.vec3
    } else {
        em.cx.vec4
    };
    let parts = pop_floats(em, n)?;
    compose(em, ty, &parts)
}
