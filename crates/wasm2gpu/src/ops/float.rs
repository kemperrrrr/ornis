//! Scalar and vector floating-point operators for the translator stack.

use naga::BinaryOperator;
use naga::Expression;

use crate::cx::Val;
use crate::env::Emit;
use crate::error::TranslateError;

pub(super) fn bin(em: &mut Emit<'_, '_>, op: BinaryOperator) -> Result<(), TranslateError> {
    let right = em.pop()?;
    let left = em.pop()?;
    match (left, right) {
        (Val::Vec(left), Val::Vec(right)) => {
            let value = arith(em, op, left, right);
            em.stack.push(Val::Vec(value));
            Ok(())
        }
        (Val::Scalar(left), Val::Scalar(right)) => {
            let value = arith(em, op, left, right);
            em.stack.push(Val::Scalar(value));
            Ok(())
        }
        _ => Err(TranslateError::Unsupported),
    }
}

pub(super) fn arith(
    em: &mut Emit<'_, '_>,
    op: BinaryOperator,
    left: naga::Handle<Expression>,
    right: naga::Handle<Expression>,
) -> naga::Handle<Expression> {
    em.ex(Expression::Binary { op, left, right })
}

pub(super) fn f32_cmp(em: &mut Emit<'_, '_>, op: BinaryOperator) -> Result<(), TranslateError> {
    let right = crate::env::float_of(em.pop()?)?;
    let left = crate::env::float_of(em.pop()?)?;
    let condition = arith(em, op, left, right);
    em.push(Val::Bool(condition))
}

pub(super) fn f32_from_u32(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let raw = em.pop()?;
    let value = em.as_uint(raw)?;
    let converted = em.ex(Expression::As {
        expr: value,
        kind: naga::ScalarKind::Float,
        convert: Some(crate::cx::F32_WIDTH),
    });
    em.push(Val::Scalar(converted))
}
