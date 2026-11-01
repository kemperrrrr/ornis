//! Integer arithmetic, comparisons, and WebAssembly selection.

use naga::BinaryOperator;
use naga::Expression;

use crate::cx::Val;
use crate::env::Emit;
use crate::error::TranslateError;

use super::Ready;
use super::ptr::i32_pair;
use super::ptr::i32_ready;

pub(super) fn i32_bin(
    em: &mut Emit<'_, '_>,
    fold: fn(u32, u32) -> u32,
    op: BinaryOperator,
) -> Result<(), TranslateError> {
    let right = em.pop()?;
    let left = em.pop()?;
    i32_pair(em, left, right, fold, op)
}

pub(super) fn i32_cmp(
    em: &mut Emit<'_, '_>,
    fold: fn(u32, u32) -> bool,
    op: BinaryOperator,
) -> Result<(), TranslateError> {
    match i32_ready(em, |a, b| u32::from(fold(a, b)))? {
        Ready::Imm(value) => em.push(Val::Imm(value)),
        Ready::Expr(left, right) => {
            let cond = em.ex(Expression::Binary { op, left, right });
            em.push(Val::Bool(cond))
        }
    }
}

pub(super) fn i32_eqz(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let val = em.pop()?;
    if let Val::Imm(value) = val {
        em.stack.push(Val::Imm(u32::from(value == 0)));
        return Ok(());
    }
    if let Val::Bool(handle) = val {
        let negated = crate::cx::bool_not(em.func, &mut em.block, handle);
        em.stack.push(Val::Bool(negated));
        return Ok(());
    }
    let bits = em.as_uint(val)?;
    let zero = em.u32(0);
    let cond = em.ex(Expression::Binary {
        op: BinaryOperator::Equal,
        left: bits,
        right: zero,
    });
    em.stack.push(Val::Bool(cond));
    Ok(())
}

/// Arithmetic right shift on the `i32` value of a `u32` bit pattern.
pub(super) fn shr_s(a: u32, b: u32) -> u32 {
    (a as i32).wrapping_shr(b) as u32
}

pub(super) fn wasm_select(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let cond_val = em.pop()?;
    let cond = em.as_bool(cond_val)?;
    let reject = em.pop()?;
    let accept = em.pop()?;
    if let Some(merged) = em.merge_ptr(cond, accept, reject) {
        return em.push(merged);
    }
    let value = match (accept, reject) {
        (Val::Scalar(left), Val::Scalar(right)) => Val::Scalar(em.pick(cond, left, right)),
        (Val::Vec(left), Val::Vec(right)) => Val::Vec(em.pick(cond, left, right)),
        // Both arms are `0`/`1`, so the result stays a boolean.
        (Val::Bool(left), Val::Bool(right)) => Val::Bool(em.pick(cond, left, right)),
        (left, right) => {
            let accept = em.as_uint(left)?;
            let reject = em.as_uint(right)?;
            Val::Index(em.pick(cond, accept, reject))
        }
    };
    em.stack.push(value);
    Ok(())
}
