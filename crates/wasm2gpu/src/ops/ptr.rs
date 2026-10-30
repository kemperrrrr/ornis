//! Pointer provenance on the value stack, with the integer fallback it
//! shares with plain `i32` arithmetic.

use naga::BinaryOperator;

use super::Ready;
use super::arith;
use super::zero_one;
use crate::cx::PtrOff;
use crate::cx::Val;
use crate::env::Emit;
use crate::error::TranslateError;

/// Fold `storage_base`'s provenance pair into an address.
///
/// The binding's base tag is the binding itself: only byte offsets are
/// ever added, and the access names the global through group and binding.
pub(super) fn storage_base(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let binding = em.pop_imm()?;
    let group = em.pop_imm()?;
    em.push(Val::Ptr {
        group,
        binding,
        off: PtrOff::Imm(0),
    })
}

/// Shadowed pointer value of a local slot, or `None` for a plain local.
pub(super) fn local_ptr(em: &mut Emit<'_, '_>, slot: usize) -> Result<Option<Val>, TranslateError> {
    if let Some((group, binding)) = em.ptr_slots.get(slot).copied().flatten() {
        let var = em
            .locals
            .get(slot)
            .copied()
            .flatten()
            .ok_or(TranslateError::Unsupported)?;
        let loaded = em.load_var(var);
        return Ok(Some(Val::Ptr {
            group,
            binding,
            off: PtrOff::Dyn(loaded),
        }));
    }
    Ok(None)
}

/// The provenance pair of a value: `Some` only for a storage pointer.
pub(super) fn provenance(val: &Val) -> Option<(u32, u32)> {
    match val {
        Val::Ptr { group, binding, .. } => Some((*group, *binding)),
        _ => None,
    }
}

/// `i32.add` / `i32.sub` that keep a storage pointer's provenance.
///
/// A pointer plus or minus an offset stays a pointer; anything else falls
/// back to plain integer arithmetic, where a pointer operand is rejected.
pub(super) fn ptr_bin(
    em: &mut Emit<'_, '_>,
    add: bool,
    fold: fn(u32, u32) -> u32,
) -> Result<(), TranslateError> {
    let op = if add {
        BinaryOperator::Add
    } else {
        BinaryOperator::Subtract
    };
    let right = em.pop()?;
    let left = em.pop()?;
    let (group, binding, off) = match (left, right) {
        (Val::Ptr { .. }, Val::Ptr { .. }) => return Err(TranslateError::Unsupported),
        (
            Val::Ptr {
                group,
                binding,
                off,
            },
            operand,
        ) => {
            let next = off_math(em, off, operand, add, fold)?;
            (group, binding, next)
        }
        (
            operand,
            Val::Ptr {
                group,
                binding,
                off,
            },
        ) if add => {
            let off = off_math(em, off, operand, add, fold)?;
            (group, binding, off)
        }
        (left, right) => return i32_pair(em, left, right, fold, op),
    };
    em.push(Val::Ptr {
        group,
        binding,
        off,
    })
}

/// Fold or emit `off ± operand` as a pointer's new byte offset.
fn off_math(
    em: &mut Emit<'_, '_>,
    off: PtrOff,
    operand: Val,
    add: bool,
    fold: fn(u32, u32) -> u32,
) -> Result<PtrOff, TranslateError> {
    let op = if add {
        BinaryOperator::Add
    } else {
        BinaryOperator::Subtract
    };
    match (off, operand) {
        (PtrOff::Imm(value), Val::Imm(delta)) => Ok(PtrOff::Imm(fold(value, delta))),
        (off, operand) => {
            let left = em.off_expr(off);
            let right = em.as_uint(operand)?;
            Ok(PtrOff::Dyn(arith(em, op, left, right)))
        }
    }
}

pub(super) fn f32_lt(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let right = crate::env::float_of(em.pop()?)?;
    let left = crate::env::float_of(em.pop()?)?;
    let cond = arith(em, BinaryOperator::Less, left, right);
    let value = zero_one(em, cond);
    em.push(Val::Index(value))
}

fn ready_of(
    em: &mut Emit<'_, '_>,
    left: Val,
    right: Val,
    fold: impl Fn(u32, u32) -> u32,
) -> Result<Ready, TranslateError> {
    if let (Val::Imm(a), Val::Imm(b)) = (left, right) {
        return Ok(Ready::Imm(fold(a, b)));
    }
    Ok(Ready::Expr(em.as_uint(left)?, em.as_uint(right)?))
}

pub(super) fn i32_ready(
    em: &mut Emit<'_, '_>,
    fold: impl Fn(u32, u32) -> u32,
) -> Result<Ready, TranslateError> {
    let right = em.pop()?;
    let left = em.pop()?;
    ready_of(em, left, right, fold)
}

pub(super) fn i32_pair(
    em: &mut Emit<'_, '_>,
    left: Val,
    right: Val,
    fold: fn(u32, u32) -> u32,
    op: BinaryOperator,
) -> Result<(), TranslateError> {
    match ready_of(em, left, right, fold)? {
        Ready::Imm(value) => em.push(Val::Imm(value)),
        Ready::Expr(left, right) => {
            let value = arith(em, op, left, right);
            em.push(Val::Index(value))
        }
    }
}
