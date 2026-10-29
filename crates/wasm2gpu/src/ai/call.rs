//! Direct calls of defined functions during abstract interpretation.
//!
//! Constant `i32` arguments stay constant in the callee, so a group or
//! binding passed as an immediate is still an immediate at the import.
//! A dynamic index stays dynamic and may come back as a result.

use super::Abs;
use super::Ai;
use super::St;
use super::interpret;
use super::join::entry;
use super::pop;
use super::push;

use crate::error::TranslateError;
use crate::ir::Kind;

pub(super) fn defined(ai: &mut Ai<'_>, st: &mut St, index: u32) -> Result<(), TranslateError> {
    if ai.active.contains(&index) {
        return Err(TranslateError::Recursion);
    }
    let func = ai
        .funcs
        .get(index as usize)
        .ok_or(TranslateError::Unsupported)?;
    let params = func.sig.params;
    let results = func.sig.results;
    let args = take_args(st, params)?;
    let body = func.body.clone();
    let mut callee = entry(func);
    seed(&mut callee, func, &args)?;
    ai.active.push(index);
    let saved = std::mem::take(&mut ai.labels);
    let result = interpret(ai, &mut callee, &body);
    ai.labels = saved;
    ai.active.pop();
    result?;
    if callee.stop || callee.stack.len() != results as usize {
        return Err(TranslateError::Unsupported);
    }
    for value in callee.stack {
        push(st, value)?;
    }
    Ok(())
}

fn seed(st: &mut St, func: &crate::ir::Func, args: &[Abs]) -> Result<(), TranslateError> {
    let mut index = 0usize;
    while index < args.len() {
        if !kind_ok(func.locals[index], &args[index]) {
            return Err(TranslateError::Unsupported);
        }
        st.locals[index] = args[index].clone();
        index += 1;
    }
    Ok(())
}

fn kind_ok(kind: Kind, value: &Abs) -> bool {
    matches!(
        (kind, value),
        (Kind::I32, Abs::I32(_)) | (Kind::F32, Abs::F32)
    )
}

fn take_args(st: &mut St, n: u32) -> Result<Vec<Abs>, TranslateError> {
    let mut args = Vec::new();
    let mut index = 0u32;
    while index < n {
        args.push(pop(st)?);
        index += 1;
    }
    args.reverse();
    Ok(args)
}
