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

fn unary_f32(st: &mut St) -> Result<(), TranslateError> {
    match super::pop(st)? {
        Abs::F32 => super::push(st, Abs::F32),
        _ => Err(TranslateError::Unsupported),
    }
}
