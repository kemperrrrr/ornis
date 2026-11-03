//! Vector math import lowering.

use crate::cx::Val;
use crate::env::Emit;
use crate::error::TranslateError;
use naga::Expression;

pub(super) fn math_dot3(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let right = pop_vec3(em)?;
    let left = pop_vec3(em)?;
    let value = em.ex(Expression::Math {
        fun: naga::MathFunction::Dot,
        arg: left,
        arg1: Some(right),
        arg2: None,
        arg3: None,
    });
    em.stack.push(Val::Scalar(value));
    Ok(())
}

pub(super) fn math_length3(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let vector = pop_vec3(em)?;
    let value = em.ex(Expression::Math {
        fun: naga::MathFunction::Length,
        arg: vector,
        arg1: None,
        arg2: None,
        arg3: None,
    });
    em.stack.push(Val::Scalar(value));
    Ok(())
}

pub(super) fn math_vec3(
    em: &mut Emit<'_, '_>,
    fun: naga::MathFunction,
    binary: bool,
) -> Result<(), TranslateError> {
    let right = if binary { Some(pop_vec3(em)?) } else { None };
    let left = pop_vec3(em)?;
    let vector = em.ex(Expression::Math {
        fun,
        arg: left,
        arg1: right,
        arg2: None,
        arg3: None,
    });
    let mut index = 0;
    while index < crate::cx::VEC3_LANES {
        let lane = em.ex(Expression::AccessIndex {
            base: vector,
            index,
        });
        em.stack.push(Val::Scalar(lane));
        index += 1;
    }
    Ok(())
}

pub(super) fn pop_vec3(em: &mut Emit<'_, '_>) -> Result<naga::Handle<Expression>, TranslateError> {
    let z = crate::env::float_of(em.pop()?)?;
    let y = crate::env::float_of(em.pop()?)?;
    let x = crate::env::float_of(em.pop()?)?;
    let ty = em.cx.vec3;
    Ok(em.ex(Expression::Compose {
        ty,
        components: vec![x, y, z],
    }))
}

pub(super) fn math_scalar(
    em: &mut Emit<'_, '_>,
    fun: naga::MathFunction,
    arity: usize,
) -> Result<(), TranslateError> {
    let mut args = Vec::with_capacity(arity);
    while args.len() < arity {
        args.push(crate::env::float_of(em.pop()?)?);
    }
    args.reverse();
    let mut args = args.into_iter();
    let arg = args.next().ok_or(TranslateError::Unsupported)?;
    let arg1 = args.next();
    let arg2 = args.next();
    let value = em.ex(Expression::Math {
        fun,
        arg,
        arg1,
        arg2,
        arg3: None,
    });
    em.stack.push(Val::Scalar(value));
    Ok(())
}
