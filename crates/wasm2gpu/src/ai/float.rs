//! Floating-point operations of the abstract-value interpreter.

use super::Abs;
use super::St;
use crate::error::TranslateError;

pub(super) fn f32_abs(st: &mut St) -> Result<(), TranslateError> {
    match super::pop(st)? {
        Abs::F32 => super::push(st, Abs::F32),
        _ => Err(TranslateError::Unsupported),
    }
}
