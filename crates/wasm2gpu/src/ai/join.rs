//! Join of abstract values where control-flow paths meet.

use super::Abs;
use super::Num;
use super::St;

use crate::error::TranslateError;
use crate::ir::Func;
use crate::ir::Kind;

pub(super) fn join_header(entry: &St, jumps: &[St]) -> Result<St, TranslateError> {
    let mut acc = entry.clone();
    acc.stop = false;
    for jump in jumps {
        acc = join_two(acc, jump.clone())?;
    }
    Ok(acc)
}

pub(super) fn join_list(states: Vec<St>) -> Result<St, TranslateError> {
    let mut live = Vec::new();
    for state in states {
        if !state.stop {
            live.push(state);
        }
    }
    let Some(mut acc) = live.pop() else {
        return Ok(stopped());
    };
    while let Some(state) = live.pop() {
        acc = join_two(acc, state)?;
    }
    acc.stop = false;
    Ok(acc)
}

pub(super) fn join_two(left: St, right: St) -> Result<St, TranslateError> {
    Ok(St {
        stack: join_vec(left.stack, right.stack)?,
        locals: join_vec(left.locals, right.locals)?,
        stop: false,
    })
}

pub(super) fn join_vec(left: Vec<Abs>, right: Vec<Abs>) -> Result<Vec<Abs>, TranslateError> {
    if left.len() != right.len() {
        return Err(TranslateError::Unsupported);
    }
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < left.len() {
        out.push(join_abs(left[index].clone(), right[index].clone())?);
        index += 1;
    }
    Ok(out)
}

pub(super) fn join_abs(left: Abs, right: Abs) -> Result<Abs, TranslateError> {
    match (left, right) {
        (Abs::Bot, other) | (other, Abs::Bot) => Ok(other),
        (Abs::F32, Abs::F32) => Ok(Abs::F32),
        (Abs::Vec, Abs::Vec) => Ok(Abs::Vec),
        (Abs::I32(left), Abs::I32(right)) => Ok(Abs::I32(join_num(left, right))),
        _ => Err(TranslateError::Unsupported),
    }
}

pub(super) fn join_num(left: Num, right: Num) -> Num {
    match (left, right) {
        (Num::Const(a), Num::Const(b)) if a == b => Num::Const(a),
        (Num::Const(_), Num::Const(_)) => Num::Dyn { joined: true },
        (Num::Const(_), Num::Dyn { joined }) | (Num::Dyn { joined }, Num::Const(_)) => {
            Num::Dyn { joined }
        }
        (Num::Dyn { joined: a }, Num::Dyn { joined: b }) => Num::Dyn { joined: a || b },
    }
}

pub(super) fn same(left: &St, right: &St) -> bool {
    left.stack == right.stack && left.locals == right.locals
}

pub(super) fn entry(func: &Func) -> St {
    let mut locals = Vec::new();
    let mut index = 0usize;
    while index < func.locals.len() {
        locals.push(local_abs(func, index));
        index += 1;
    }
    St {
        stack: Vec::new(),
        locals,
        stop: false,
    }
}

pub(super) fn local_abs(func: &Func, index: usize) -> Abs {
    if (index as u32) >= func.sig.params {
        return Abs::Bot;
    }
    match func.locals[index] {
        Kind::I32 => Abs::I32(Num::Dyn { joined: false }),
        Kind::F32 => Abs::F32,
    }
}
pub(super) fn stopped() -> St {
    St {
        stack: Vec::new(),
        locals: Vec::new(),
        stop: true,
    }
}
