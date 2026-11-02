//! Floating-point operations of the abstract-value interpreter.

use super::Abs;
use super::St;
use crate::error::TranslateError;

pub(super) fn f32_abs(st: &mut St) -> Result<(), TranslateError> {
    unary_f32(st)
}

pub(super) fn f32_neg(st: &mut St) -> Result<(), TranslateError> {
    unary_f32(st)
}

pub(super) fn f32_sqrt(st: &mut St) -> Result<(), TranslateError> {
    unary_f32(st)
}

pub(super) fn f32_bin(st: &mut St) -> Result<(), TranslateError> {
    let right = super::pop(st)?;
    let left = super::pop(st)?;
    match (left, right) {
        (Abs::F32, Abs::F32) => super::push(st, Abs::F32),
        (Abs::Vec, Abs::Vec) => super::push(st, Abs::Vec),
        _ => Err(TranslateError::Unsupported),
    }
}

fn unary_f32(st: &mut St) -> Result<(), TranslateError> {
    match super::pop(st)? {
        Abs::F32 => super::push(st, Abs::F32),
        _ => Err(TranslateError::Unsupported),
    }
}
