//! Abstract values for group, binding, and element index across joins.
//!
//! A constant group or binding stays constant when every predecessor agrees.
//! Two different constants become [`Num::Dyn`] `{ joined: true }` and are
//! rejected only when that value is *used* as a group or binding. An index may
//! be that phi. `i32` arithmetic keeps `joined`, so a binding phi stays
//! ambiguous after `i32.add`. A loop counter that is not a phi stays
//! `joined: false`. Non-parameter locals start at zero, matching Wasm.

use crate::error::TranslateError;
use crate::ir::Func;
use crate::ir::Import;
use crate::ir::Stmt;
use crate::ty::StructDef;

mod join;
mod uniform;
mod value;

use join::entry;
use join::join_header;
use join::join_list;
use join::same;
use value::on_op;

const MAX_ITER: u32 = {
    const STEPS: u32 = 8;
    STEPS
};
const STRUCT_HEADER: u32 = 4;

#[derive(Clone, PartialEq, Eq)]
enum Abs {
    F32,
    Vec,
    I32(Num),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Num {
    Const(u32),
    Dyn { joined: bool },
}

#[derive(Clone, PartialEq, Eq)]
struct St {
    stack: Vec<Abs>,
    locals: Vec<Abs>,
    stop: bool,
}

enum Fold {
    True,
    False,
    Dyn,
}

struct Lab {
    jumps: Vec<St>,
}

struct Ai<'a> {
    imports: &'a [Import],
    structs: &'a [StructDef],
    labels: Vec<Lab>,
}

pub(super) fn check(
    func: &Func,
    imports: &[Import],
    structs: &[StructDef],
) -> Result<(), TranslateError> {
    let mut ai = Ai {
        imports,
        structs,
        labels: Vec::new(),
    };
    let mut st = entry(func);
    interpret(&mut ai, &mut st, &func.body)
}

// qual:recursive
fn interpret(ai: &mut Ai<'_>, st: &mut St, stmts: &[Stmt]) -> Result<(), TranslateError> {
    for stmt in stmts {
        if st.stop {
            return Ok(());
        }
        on_stmt(ai, st, stmt)?;
    }
    Ok(())
}

fn on_stmt(ai: &mut Ai<'_>, st: &mut St, stmt: &Stmt) -> Result<(), TranslateError> {
    match stmt {
        Stmt::Op(op) => on_op(ai, st, *op),
        Stmt::Block { body, .. } => region(ai, st, body),
        Stmt::Loop { body, .. } => loop_region(ai, st, body),
        Stmt::If {
            then_body,
            else_body,
            ..
        } => if_region(ai, st, then_body, else_body),
        Stmt::Br(depth) => branch(ai, st, *depth, true),
        Stmt::BrIf(depth) => br_if(ai, st, *depth),
    }
}

fn region(ai: &mut Ai<'_>, st: &mut St, body: &[Stmt]) -> Result<(), TranslateError> {
    ai.labels.push(Lab { jumps: Vec::new() });
    interpret(ai, st, body)?;
    let lab = pop_lab(ai)?;
    let mut outs = Vec::new();
    if !st.stop {
        outs.push(st.clone());
    }
    outs.extend(lab.jumps);
    *st = join_list(outs)?;
    Ok(())
}

fn if_region(
    ai: &mut Ai<'_>,
    st: &mut St,
    then_body: &[Stmt],
    else_body: &[Stmt],
) -> Result<(), TranslateError> {
    let cond = pop_num(st)?;
    ai.labels.push(Lab { jumps: Vec::new() });
    let outs = match fold(cond) {
        Fold::True => one_side(ai, st, then_body)?,
        Fold::False => one_side(ai, st, else_body)?,
        Fold::Dyn => both_sides(ai, st, then_body, else_body)?,
    };
    let lab = pop_lab(ai)?;
    let mut outs = outs;
    outs.extend(lab.jumps);
    *st = join_list(outs)?;
    Ok(())
}

fn one_side(ai: &mut Ai<'_>, st: &mut St, body: &[Stmt]) -> Result<Vec<St>, TranslateError> {
    interpret(ai, st, body)?;
    let mut outs = Vec::new();
    if !st.stop {
        outs.push(st.clone());
    }
    Ok(outs)
}

fn both_sides(
    ai: &mut Ai<'_>,
    st: &St,
    then_body: &[Stmt],
    else_body: &[Stmt],
) -> Result<Vec<St>, TranslateError> {
    let mut then_st = st.clone();
    interpret(ai, &mut then_st, then_body)?;
    let mut else_st = st.clone();
    interpret(ai, &mut else_st, else_body)?;
    let mut outs = Vec::new();
    if !then_st.stop {
        outs.push(then_st);
    }
    if !else_st.stop {
        outs.push(else_st);
    }
    Ok(outs)
}

fn loop_region(ai: &mut Ai<'_>, st: &mut St, body: &[Stmt]) -> Result<(), TranslateError> {
    let entry = st.clone();
    let mut header = entry.clone();
    let mut guard = 0u32;
    let exit = loop {
        if guard >= MAX_ITER {
            return Err(TranslateError::Unsupported);
        }
        guard += 1;
        ai.labels.push(Lab { jumps: Vec::new() });
        let mut cur = header.clone();
        cur.stop = false;
        interpret(ai, &mut cur, body)?;
        let lab = pop_lab(ai)?;
        let new_header = join_header(&entry, &lab.jumps)?;
        let stable = same(&header, &new_header);
        header = new_header;
        if stable {
            break cur;
        }
    };
    *st = exit;
    Ok(())
}

fn branch(ai: &mut Ai<'_>, st: &mut St, depth: u32, stop: bool) -> Result<(), TranslateError> {
    let n = ai.labels.len();
    if depth as usize == n {
        return Err(TranslateError::FunctionBranch);
    }
    let index = n
        .checked_sub(depth as usize + 1)
        .ok_or(TranslateError::Unsupported)?;
    ai.labels[index].jumps.push(st.clone());
    st.stop = stop;
    Ok(())
}

fn br_if(ai: &mut Ai<'_>, st: &mut St, depth: u32) -> Result<(), TranslateError> {
    let cond = pop_num(st)?;
    match fold(cond) {
        Fold::True => branch(ai, st, depth, true),
        Fold::False => Ok(()),
        Fold::Dyn => {
            let mut jumped = st.clone();
            branch(ai, &mut jumped, depth, false)?;
            Ok(())
        }
    }
}

fn local_get(st: &mut St, index: u32) -> Result<(), TranslateError> {
    let value = st
        .locals
        .get(index as usize)
        .cloned()
        .ok_or(TranslateError::Unsupported)?;
    push(st, value)
}

fn local_set(st: &mut St, index: u32, tee: bool) -> Result<(), TranslateError> {
    let value = pop(st)?;
    let slot = st
        .locals
        .get_mut(index as usize)
        .ok_or(TranslateError::Unsupported)?;
    *slot = value.clone();
    if tee {
        push(st, value)?;
    }
    Ok(())
}

fn pop_num(st: &mut St) -> Result<Num, TranslateError> {
    match pop(st)? {
        Abs::I32(num) => Ok(num),
        _ => Err(TranslateError::Unsupported),
    }
}

fn pop(st: &mut St) -> Result<Abs, TranslateError> {
    st.stack.pop().ok_or(TranslateError::Unsupported)
}

fn push(st: &mut St, value: Abs) -> Result<(), TranslateError> {
    st.stack.push(value);
    Ok(())
}

fn fold(num: Num) -> Fold {
    match num {
        Num::Const(0) => Fold::False,
        Num::Const(_) => Fold::True,
        Num::Dyn { .. } => Fold::Dyn,
    }
}

fn pop_lab(ai: &mut Ai<'_>) -> Result<Lab, TranslateError> {
    ai.labels.pop().ok_or(TranslateError::Unsupported)
}

fn type_msg(msg: &str) -> TranslateError {
    TranslateError::UnsupportedType(msg.to_string())
}
