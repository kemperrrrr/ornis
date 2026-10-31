//! Canonical forms the emitter lowers without branch flags.
//!
//! Two rewrites run on the structured wasm IR before emission:
//!
//! - A branch to the function's last block leaves the function, so it
//!   becomes a `return`. Without it the branch would cross two regions and
//!   force both of them onto flag variables.
//! - A block whose first branch is a `br_if 0` guards the rest of the
//!   block: the rest runs when the condition is false. Such a block is an
//!   `if`, not a region with a label, so the emitter needs no loop for it.
//!
//! Both rewrites keep the label depth of every branch they move: a wasm
//! `block` and a wasm `if` push one label each, and a `return` pushes none.

use crate::ir::BlockTy;
use crate::ir::Func;
use crate::ir::Op;
use crate::ir::Stmt;

use super::shape::Escape;
use super::shape::targeted;

/// Rewrite `func`'s body into the shapes the emitter lowers directly.
pub(super) fn rewrite(func: &Func) -> Vec<Stmt> {
    let mut body = func.body.clone();
    if func.sig.results == 0 {
        tail_returns(&mut body);
    }
    guards(&mut body);
    body
}

/// Turn a branch to the function's last block into a `return`.
///
/// The last block ends the function, so leaving it leaves the function.
fn tail_returns(body: &mut [Stmt]) {
    let Some(Stmt::Block { body: last, ty }) = body.last_mut() else {
        return;
    };
    // A block that carries values keeps them: a branch to it hands them over.
    if !ty.params.is_empty() || !ty.results.is_empty() {
        return;
    }
    returns(last, 0);
}

// qual:recursive
fn returns(stmts: &mut [Stmt], level: u32) {
    for stmt in stmts.iter_mut() {
        match stmt {
            // A branch in the tail block itself already ends the function
            // when it leaves the block, so only a branch from a nested
            // region needs to become a `return`.
            Stmt::Br(depth) if level > 0 && *depth == level => *stmt = Stmt::Return,
            Stmt::BrIf(depth) if level > 0 && *depth == level => {
                *stmt = Stmt::If {
                    ty: void_ty(),
                    then_body: vec![Stmt::Return],
                    else_body: Vec::new(),
                };
            }
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } => returns(body, level + 1),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                returns(then_body, level + 1);
                returns(else_body, level + 1);
            }
            _ => {}
        }
    }
}

/// Rewrite a guarded block into an `if`.
// qual:recursive
fn guards(stmts: &mut Vec<Stmt>) {
    let old = std::mem::take(stmts);
    let mut out: Vec<Stmt> = Vec::with_capacity(old.len());
    for stmt in old {
        match stmt {
            Stmt::Block { ty, mut body } => {
                guards(&mut body);
                match guard(&ty, &body) {
                    Some((prefix, rest)) => {
                        out.extend(prefix);
                        out.push(Stmt::Op(Op::I32Eqz));
                        out.push(guarded(rest));
                    }
                    None => out.push(Stmt::Block { ty, body }),
                }
            }
            Stmt::Loop { ty, mut body } => {
                guards(&mut body);
                out.push(Stmt::Loop { ty, body });
            }
            Stmt::If {
                ty,
                mut then_body,
                mut else_body,
            } => {
                guards(&mut then_body);
                guards(&mut else_body);
                out.push(Stmt::If {
                    ty,
                    then_body,
                    else_body,
                });
            }
            other => out.push(other),
        }
    }
    *stmts = out;
}

/// The `if` a guarded block becomes.
fn guarded(body: Vec<Stmt>) -> Stmt {
    Stmt::If {
        ty: void_ty(),
        then_body: body,
        else_body: Vec::new(),
    }
}

/// Split a block into the guard prefix and the guarded rest.
///
/// The guard is a leading `br_if 0`: it leaves the block when the condition
/// holds, so the rest runs when the condition is false. The rest may not
/// branch out of the block, or the block would still need a label.
fn guard(ty: &BlockTy, body: &[Stmt]) -> Option<(Vec<Stmt>, Vec<Stmt>)> {
    if !ty.params.is_empty() || !ty.results.is_empty() {
        return None;
    }
    let split = body.iter().position(is_branch)?;
    if !matches!(body[split], Stmt::BrIf(0)) {
        return None;
    }
    let prefix = &body[..split];
    if Escape::of(prefix).control {
        // The prefix leaves the block. Lifting it out would move its
        // branches one label up, and they would target another block.
        return None;
    }
    let rest = &body[split + 1..];
    if targeted(rest, 0) || targeted(rest, 1) {
        return None;
    }
    Some((prefix.to_vec(), rest.to_vec()))
}

fn is_branch(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::Br(_) | Stmt::BrIf(_) | Stmt::BrTable { .. } | Stmt::Return
    )
}

fn void_ty() -> BlockTy {
    BlockTy {
        params: Vec::new(),
        results: Vec::new(),
    }
}
