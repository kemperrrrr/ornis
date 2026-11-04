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

const F32_SIGN_BIT: u32 = 1 << 31;
const F32_CANONICAL_NAN_BITS: u32 = 0x7fc0_0000;
const F32_POSITIVE_ZERO: f32 = 0.0;
const F32_NEGATIVE_ZERO: f32 = f32::from_bits(F32_SIGN_BIT);
const U32_ZERO: u32 = 0;

pub(super) fn scalar_binary(
    em: &mut Emit<'_, '_>,
    fun: naga::MathFunction,
) -> Result<(), TranslateError> {
    let right = crate::env::float_of(em.pop()?)?;
    let left = crate::env::float_of(em.pop()?)?;
    let value = match fun {
        naga::MathFunction::Min | naga::MathFunction::Max => wasm_min_max(em, fun, left, right),
        _ => em.ex(Expression::Math {
            fun,
            arg: left,
            arg1: Some(right),
            arg2: None,
            arg3: None,
        }),
    };
    em.stack.push(Val::Scalar(value));
    Ok(())
}

fn wasm_min_max(
    em: &mut Emit<'_, '_>,
    fun: naga::MathFunction,
    left: naga::Handle<Expression>,
    right: naga::Handle<Expression>,
) -> naga::Handle<Expression> {
    let ordinary = em.ex(Expression::Math {
        fun,
        arg: left,
        arg1: Some(right),
        arg2: None,
        arg3: None,
    });
    let zero_corrected = signed_zero_min_max(em, fun, left, right, ordinary);
    canonical_nan_min_max(em, left, right, zero_corrected)
}

fn signed_zero_min_max(
    em: &mut Emit<'_, '_>,
    fun: naga::MathFunction,
    left: naga::Handle<Expression>,
    right: naga::Handle<Expression>,
    ordinary: naga::Handle<Expression>,
) -> naga::Handle<Expression> {
    let zero = em.ex(Expression::Literal(naga::Literal::F32(F32_POSITIVE_ZERO)));
    let left_zero = binary(em, naga::BinaryOperator::Equal, left, zero);
    let right_zero = binary(em, naga::BinaryOperator::Equal, right, zero);
    let both_zero = binary(em, naga::BinaryOperator::LogicalAnd, left_zero, right_zero);
    let left_negative = sign_bit_set(em, left);
    let right_negative = sign_bit_set(em, right);
    let zero_is_negative = binary(
        em,
        if fun == naga::MathFunction::Min {
            naga::BinaryOperator::LogicalOr
        } else {
            naga::BinaryOperator::LogicalAnd
        },
        left_negative,
        right_negative,
    );
    let negative_zero = em.ex(Expression::Literal(naga::Literal::F32(F32_NEGATIVE_ZERO)));
    let signed_zero = select(em, zero_is_negative, negative_zero, zero);
    select(em, both_zero, signed_zero, ordinary)
}

fn canonical_nan_min_max(
    em: &mut Emit<'_, '_>,
    left: naga::Handle<Expression>,
    right: naga::Handle<Expression>,
    numeric: naga::Handle<Expression>,
) -> naga::Handle<Expression> {
    let left_nan = is_nan(em, left);
    let right_nan = is_nan(em, right);
    let either_nan = binary(em, naga::BinaryOperator::LogicalOr, left_nan, right_nan);
    let canonical_nan_bits = em.ex(Expression::Literal(naga::Literal::U32(
        F32_CANONICAL_NAN_BITS,
    )));
    let canonical_nan = em.ex(Expression::As {
        expr: canonical_nan_bits,
        kind: naga::ScalarKind::Float,
        convert: None,
    });
    select(em, either_nan, canonical_nan, numeric)
}

fn select(
    em: &mut Emit<'_, '_>,
    condition: naga::Handle<Expression>,
    accept: naga::Handle<Expression>,
    reject: naga::Handle<Expression>,
) -> naga::Handle<Expression> {
    em.ex(Expression::Select {
        condition,
        accept,
        reject,
    })
}

fn binary(
    em: &mut Emit<'_, '_>,
    op: naga::BinaryOperator,
    left: naga::Handle<Expression>,
    right: naga::Handle<Expression>,
) -> naga::Handle<Expression> {
    em.ex(Expression::Binary { op, left, right })
}

fn is_nan(em: &mut Emit<'_, '_>, value: naga::Handle<Expression>) -> naga::Handle<Expression> {
    em.ex(Expression::Relational {
        fun: naga::RelationalFunction::IsNan,
        argument: value,
    })
}

fn sign_bit_set(
    em: &mut Emit<'_, '_>,
    value: naga::Handle<Expression>,
) -> naga::Handle<Expression> {
    let bits = em.ex(Expression::As {
        expr: value,
        kind: naga::ScalarKind::Uint,
        convert: None,
    });
    let mask = em.ex(Expression::Literal(naga::Literal::U32(F32_SIGN_BIT)));
    let sign = binary(em, naga::BinaryOperator::And, bits, mask);
    let zero = em.ex(Expression::Literal(naga::Literal::U32(U32_ZERO)));
    binary(em, naga::BinaryOperator::NotEqual, sign, zero)
}
