//! Direct calls of defined functions during abstract interpretation.
//!
//! Constant `i32` arguments stay constant in the callee, so a group or
//! binding passed as an immediate is still an immediate at the import.
//! A dynamic index stays dynamic and may come back as a result. The same
//! function and the same abstract arguments are interpreted once.
//!
//! The memo key is that pair, and the stored value is only the result
//! stack. A callee does not change anything the caller reads after return.
//! When globals appear, the key has to grow with them.

use super::Abs;
use super::Ai;
use super::St;
use super::interpret;
use super::join::entry;
use super::pop;
use super::push;

use crate::error::TranslateError;
use crate::ir::Kind;

/// One entry's memo of defined calls, and how many new pairs it has entered.
pub(super) struct Seen {
    entries: u32,
    hits: Vec<Hit>,
}

struct Hit {
    func: u32,
    args: Vec<Abs>,
    out: Vec<Abs>,
}

impl Seen {
    pub(super) fn new() -> Self {
        Self {
            entries: 0,
            hits: Vec::new(),
        }
    }
}

/// Count one new `(function, arguments)` pair.
///
/// Instructions and loop iterations do not pass through here. A pair already
/// in the memo is not counted again.
pub(super) fn charge(ai: &mut Ai<'_>) -> Result<(), TranslateError> {
    ai.seen.entries += 1;
    if ai.seen.entries > crate::cx::MAX_AI_ENTRIES {
        return Err(TranslateError::Budget);
    }
    Ok(())
}

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
    if let Some(out) = recall(ai, index, &args) {
        return give(st, &out);
    }
    // The key is `(function, args)` only. Extend it when globals appear.
    charge(ai)?;
    let body = func.body.clone();
    let mut callee = entry(func);
    seed(&mut callee, func, &args)?;
    ai.active.push(index);
    let saved = std::mem::take(&mut ai.labels);
    let result = interpret(ai, &mut callee, &body);
    ai.labels = saved;
    ai.active.pop();
    result?;
    if callee.stop || callee.stack.len() != results as usize || returns_ptr(&callee.stack) {
        return Err(TranslateError::Unsupported);
    }
    remember(ai, index, &args, &callee.stack);
    give(st, &callee.stack)
}

/// A pointer result would leave the callee with provenance this subset
/// does not carry across calls; the arguments reject it first too.
fn returns_ptr(values: &[Abs]) -> bool {
    values.iter().any(|value| matches!(value, Abs::Ptr { .. }))
}

pub(super) fn recall(ai: &Ai<'_>, func: u32, args: &[Abs]) -> Option<Vec<Abs>> {
    let index = find_hit(ai, func, args)?;
    Some(ai.seen.hits[index].out.clone())
}

pub(super) fn find_hit(ai: &Ai<'_>, func: u32, args: &[Abs]) -> Option<usize> {
    let mut index = 0usize;
    while index < ai.seen.hits.len() {
        let hit = &ai.seen.hits[index];
        if hit.func == func && hit.args == args {
            return Some(index);
        }
        index += 1;
    }
    None
}

pub(super) fn remember(ai: &mut Ai<'_>, func: u32, args: &[Abs], out: &[Abs]) {
    ai.seen.hits.push(Hit {
        func,
        args: args.to_vec(),
        out: out.to_vec(),
    });
}

pub(super) fn give(st: &mut St, values: &[Abs]) -> Result<(), TranslateError> {
    for value in values {
        push(st, value.clone())?;
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
