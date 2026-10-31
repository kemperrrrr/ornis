//! What a function does with each of its wasm locals.
//!
//! A local written once and read only after that write keeps its value in
//! the expression itself: no shadow variable, no store, no load. The write
//! must dominate every read, so a local written inside a branch or read
//! before its write keeps a shadow.

use crate::ir::Op;
use crate::ir::Stmt;

/// What a function does with each of its wasm locals.
pub(super) struct Slots {
    /// The local is written.
    pub(super) written: Vec<bool>,
    /// The local keeps its value in an expression.
    pub(super) expr: Vec<bool>,
}

impl Slots {
    /// Read the writes and the reads of every local of a body.
    ///
    /// `slots` is the number of wasm locals of the function, parameters
    /// included.
    pub(super) fn of(stmts: &[Stmt], slots: usize) -> Slots {
        let mut written = vec![false; slots];
        for stmt in stmts {
            mark(stmt, &mut written);
        }
        let mut writes = vec![0u32; slots];
        let mut reads = vec![0u32; slots];
        for stmt in stmts {
            count(stmt, &mut writes, &mut reads);
        }
        let mut ok = vec![true; slots];
        let definite = vec![false; slots];
        scan(stmts, definite, &mut ok);
        let mut expr = vec![false; slots];
        let mut index = 0usize;
        while index < slots {
            expr[index] = writes[index] == 1 && reads[index] > 0 && ok[index];
            index += 1;
        }
        Slots { written, expr }
    }
}

/// Count the writes and the reads of every local.
///
/// A `local.tee` stores the value and leaves it on the stack; it does not
/// read the old value.
// qual:recursive
fn count(stmt: &Stmt, writes: &mut [u32], reads: &mut [u32]) {
    match stmt {
        Stmt::Op(Op::LocalSet(index) | Op::LocalTee(index)) => {
            if let Some(slot) = writes.get_mut(*index as usize) {
                *slot += 1;
            }
        }
        Stmt::Op(Op::Local(index) | Op::LocalI32(index)) => {
            if let Some(slot) = reads.get_mut(*index as usize) {
                *slot += 1;
            }
        }
        Stmt::Op(_) => {}
        Stmt::Block { body, .. } | Stmt::Loop { body, .. } => {
            for stmt in body {
                count(stmt, writes, reads);
            }
        }
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            for stmt in then_body {
                count(stmt, writes, reads);
            }
            for stmt in else_body {
                count(stmt, writes, reads);
            }
        }
        Stmt::Br(_) | Stmt::BrIf(_) | Stmt::BrTable { .. } | Stmt::Return => {}
    }
}

/// Mark the slots a read reaches before their write.
///
/// `definite` holds the slots written on every path to this point. A read of
/// a slot that is not definite yet is not dominated by the write, so the
/// slot keeps a shadow.
// qual:recursive
fn scan(stmts: &[Stmt], mut definite: Vec<bool>, ok: &mut [bool]) -> Vec<bool> {
    for stmt in stmts {
        match stmt {
            Stmt::Op(Op::LocalSet(index)) => {
                if let Some(slot) = definite.get_mut(*index as usize) {
                    *slot = true;
                }
            }
            Stmt::Op(Op::LocalTee(index)) => {
                // A `local.tee` does not read the old value.
                if let Some(slot) = definite.get_mut(*index as usize) {
                    *slot = true;
                }
            }
            Stmt::Op(Op::Local(index) | Op::LocalI32(index)) => read(*index, &definite, ok),
            Stmt::Op(_) => {}
            Stmt::Block { body, .. } => {
                let end = scan(body, definite.clone(), ok);
                for index in 0..definite.len() {
                    definite[index] = definite[index] || end[index];
                }
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                let then_end = scan(then_body, definite.clone(), ok);
                let else_end = scan(else_body, definite.clone(), ok);
                // A slot written in one arm only is not written on every path.
                for index in 0..definite.len() {
                    definite[index] = definite[index] || (then_end[index] && else_end[index]);
                }
            }
            Stmt::Loop { body, .. } => {
                // The loop may run zero times, so its writes are not definite
                // after it. Inside the body they are.
                scan(body, definite.clone(), ok);
            }
            Stmt::Br(_) | Stmt::Return => break,
            Stmt::BrIf(_) | Stmt::BrTable { .. } => {}
        }
    }
    definite
}

/// Mark a slot as read before its write.
fn read(index: u32, definite: &[bool], ok: &mut [bool]) {
    let slot = index as usize;
    if definite.get(slot).copied().unwrap_or(false) {
        return;
    }
    if let Some(entry) = ok.get_mut(slot) {
        *entry = false;
    }
}

/// Mark the locals a body writes.
// qual:recursive
fn mark(stmt: &Stmt, out: &mut [bool]) {
    match stmt {
        Stmt::Op(Op::LocalSet(index) | Op::LocalTee(index)) => {
            if let Some(slot) = out.get_mut(*index as usize) {
                *slot = true;
            }
        }
        Stmt::Op(_) => {}
        Stmt::Block { body, .. } | Stmt::Loop { body, .. } => {
            for stmt in body {
                mark(stmt, out);
            }
        }
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            for stmt in then_body {
                mark(stmt, out);
            }
            for stmt in else_body {
                mark(stmt, out);
            }
        }
        Stmt::Br(_) | Stmt::BrIf(_) | Stmt::BrTable { .. } | Stmt::Return => {}
    }
}
