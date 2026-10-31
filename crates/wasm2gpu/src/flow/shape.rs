//! Shape of structured control flow: which branches leave a region, which
//! locals need a shadow variable, and which parameters a function rewrites.

use crate::env::Emit;
use crate::error::TranslateError;
use crate::ir::Func;
use crate::ir::Op;
use crate::ir::Stmt;
use crate::locals::shadow;

use super::slots::Slots;

pub(super) fn is_nested(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::Block { .. } | Stmt::Loop { .. } | Stmt::If { .. }
    )
}

/// Shadows of the parameters a function writes.
///
/// naga wants a constant initializer, so a parameter that keeps its value
/// in a variable gets that variable before the body runs, with the argument
/// value stored into it. A parameter nobody writes, or one that keeps its
/// value as an expression, needs no variable at all.
pub(super) fn bind_params(
    func: &mut naga::Function,
    ir: &Func,
    cx: &crate::cx::Cx<'_>,
    slots: &Slots,
) -> Result<Vec<Option<naga::Handle<naga::LocalVariable>>>, TranslateError> {
    let mut out = vec![None; ir.locals.len()];
    let mut index = 0usize;
    while index < ir.locals.len() {
        let is_param = index < ir.sig.params as usize;
        let keeps = slots.written.get(index).copied().unwrap_or(false)
            && !slots.expr.get(index).copied().unwrap_or(false);
        if is_param && keeps {
            out[index] = Some(shadow(func, cx, ir.locals[index])?);
        }
        index += 1;
    }
    Ok(out)
}

/// Copy the argument value of every parameter into its shadow.
pub(super) fn copy_params(em: &mut Emit<'_, '_>, params: u32) -> Result<(), TranslateError> {
    let mut index = 0u32;
    while index < params {
        let Some(var) = em.locals.get(index as usize).copied().flatten() else {
            index += 1;
            continue;
        };
        if em.baked.get(index as usize).copied().flatten().is_some() {
            index += 1;
            continue;
        }
        let value = em.param_expr(index)?;
        em.store_var(var, value);
        index += 1;
    }
    Ok(())
}

/// How a wasm `if` lowers.
pub(super) enum IfKind {
    Bad,
    Pure,
    Target,
    Stmt,
}

pub(super) fn if_kind(ty: &crate::ir::BlockTy, then_body: &[Stmt], else_body: &[Stmt]) -> IfKind {
    match ty.params.is_empty() {
        false => bad_if(),
        true => kind_from(
            targeted(then_body, 0),
            targeted(else_body, 0),
            pure_list(then_body),
            pure_list(else_body),
        ),
    }
}

fn bad_if() -> IfKind {
    IfKind::Bad
}

fn kind_from(target_then: bool, target_else: bool, pure_then: bool, pure_else: bool) -> IfKind {
    if target_then || target_else {
        IfKind::Target
    } else if pure_then && pure_else {
        IfKind::Pure
    } else {
        IfKind::Stmt
    }
}

pub(super) fn pure_list(stmts: &[Stmt]) -> bool {
    stmts.iter().all(pure_stmt)
}

pub(super) fn pure_stmt(stmt: &Stmt) -> bool {
    let Stmt::Op(op) = stmt else {
        return false;
    };
    matches!(
        op,
        Op::I32(_)
            | Op::F32(_)
            | Op::Local(_)
            | Op::LocalI32(_)
            | Op::I32Add
            | Op::I32Mul
            | Op::I32Sub
            | Op::I32Eqz
            | Op::I32Eq
            | Op::I32Ne
            | Op::I32LtU
            | Op::I32GtU
            | Op::I32Shl
            | Op::I32ShrU
            | Op::F32Lt
            | Op::Add
            | Op::Mul
            | Op::Select
    )
}

/// Branches inside a region, seen from that region.
///
/// A branch that leaves the region needs the region's `leave` flag, and a
/// branch that restarts it needs `again`. A region whose branches all stay
/// inside it needs neither.
pub(super) struct Escape {
    /// A branch leaves the region.
    pub beyond: bool,
    /// A branch inside a nested region targets the region itself.
    pub self_nested: bool,
    /// The region holds a branch or a return.
    pub control: bool,
}

impl Escape {
    /// Collect the branches of a region body.
    ///
    /// `level` counts the labels between the statements and the region, so a
    /// branch to `level` targets the region and a deeper one leaves it.
    pub(super) fn of(body: &[Stmt]) -> Escape {
        let mut out = Escape {
            beyond: false,
            self_nested: false,
            control: false,
        };
        out.walk(body, 0);
        out
    }

    fn walk(&mut self, stmts: &[Stmt], level: u32) {
        for stmt in stmts {
            match stmt {
                Stmt::Br(depth) | Stmt::BrIf(depth) => {
                    self.control = true;
                    self.branch(*depth, level);
                }
                Stmt::BrTable { targets, default } => {
                    self.control = true;
                    for target in targets {
                        self.branch(*target, level);
                    }
                    self.branch(*default, level);
                }
                Stmt::Return => self.control = true,
                Stmt::Block { body, .. } | Stmt::Loop { body, .. } => self.walk(body, level + 1),
                Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    self.walk(then_body, level + 1);
                    self.walk(else_body, level + 1);
                }
                Stmt::Op(_) => {}
            }
        }
    }

    fn branch(&mut self, depth: u32, level: u32) {
        if depth > level {
            self.beyond = true;
        } else if depth == level && level > 0 {
            self.self_nested = true;
        }
    }
}

// qual:recursive
pub(super) fn targeted(stmts: &[Stmt], depth: u32) -> bool {
    let mut index = 0usize;
    while index < stmts.len() {
        let hit = match &stmts[index] {
            Stmt::Br(found) | Stmt::BrIf(found) => *found == depth,
            Stmt::BrTable { targets, default } => targets.contains(&depth) || *default == depth,
            Stmt::Return => false,
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } => targeted(body, depth + 1),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => targeted(then_body, depth + 1) || targeted(else_body, depth + 1),
            Stmt::Op(_) => false,
        };
        if hit {
            return true;
        }
        index += 1;
    }
    false
}
