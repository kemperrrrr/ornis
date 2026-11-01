//! Integer arithmetic and selection in the abstract-value interpreter.

use super::Abs;
use super::Fold;
use super::Num;
use super::St;
use super::join::join_abs;
use super::mem::num_fold;
use super::pop;
use super::pop_num;
use super::push;

use crate::error::TranslateError;

/// Arithmetic right shift on the `i32` value of a `u32` bit pattern.
pub(super) fn shr_s(a: u32, b: u32) -> u32 {
    (a as i32).wrapping_shr(b) as u32
}

pub(super) fn i32_bin(st: &mut St, op: impl Fn(u32, u32) -> u32) -> Result<(), TranslateError> {
    let right = pop_num(st)?;
    let left = pop_num(st)?;
    push(st, Abs::I32(num_fold(left, right, op)))
}

pub(super) fn i32_cmp(st: &mut St, pred: fn(u32, u32) -> bool) -> Result<(), TranslateError> {
    i32_bin(st, |a, b| u32::from(pred(a, b)))
}

pub(super) fn i32_eqz(st: &mut St) -> Result<(), TranslateError> {
    let num = match pop_num(st)? {
        Num::Const(value) => Num::Const(u32::from(value == 0)),
        Num::Dyn { joined } => Num::Dyn { joined },
    };
    push(st, Abs::I32(num))
}

pub(super) fn select(st: &mut St) -> Result<(), TranslateError> {
    let cond = pop_num(st)?;
    let reject = pop(st)?;
    let accept = pop(st)?;
    let value = match super::fold(cond) {
        Fold::True => accept,
        Fold::False => reject,
        Fold::Dyn => join_abs(accept, reject)?,
    };
    push(st, value)
}
