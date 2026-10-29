//! Predicates and shadow locals for structured control flow.

use naga::Expression;
use naga::Literal;

use crate::cx::sp;
use crate::env::Emit;
use crate::env::const_expr;
use crate::error::TranslateError;
use crate::ir::Func;
use crate::ir::Kind;
use crate::ir::Op;
use crate::ir::Stmt;

pub(super) fn is_nested(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::Block { .. } | Stmt::Loop { .. } | Stmt::If { .. }
    )
}

pub(super) fn needs_shadow(stmts: &[Stmt]) -> bool {
    stmts.iter().any(shadow_stmt)
}

pub(super) fn shadow_stmt(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Op(Op::LocalSet(_) | Op::LocalTee(_)) => true,
        Stmt::Op(_) => false,
        Stmt::Br(_) | Stmt::BrIf(_) | Stmt::BrTable { .. } | Stmt::Return => true,
        Stmt::Block { .. } | Stmt::Loop { .. } | Stmt::If { .. } => true,
    }
}

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
            | Op::Add
            | Op::Mul
            | Op::Select
    )
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

pub(super) fn bind_locals(
    func: &mut naga::Function,
    ir: &Func,
    cx: &crate::cx::Cx<'_>,
    shadowed: bool,
) -> Result<Vec<Option<naga::Handle<naga::LocalVariable>>>, TranslateError> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < ir.locals.len() {
        if shadowed {
            out.push(Some(shadow(func, cx, ir.locals[index])?));
        } else {
            out.push(None);
        }
        index += 1;
    }
    Ok(out)
}

pub(super) fn shadow(
    func: &mut naga::Function,
    cx: &crate::cx::Cx<'_>,
    kind: Kind,
) -> Result<naga::Handle<naga::LocalVariable>, TranslateError> {
    let (ty, init) = if kind == Kind::I32 {
        (
            cx.uint,
            const_expr(func, Expression::Literal(Literal::U32(0))),
        )
    } else {
        (
            cx.scalar,
            const_expr(func, Expression::Literal(Literal::F32(0.0))),
        )
    };
    Ok(func.local_variables.append(
        naga::LocalVariable {
            name: None,
            ty,
            init: Some(init),
        },
        sp(),
    ))
}

pub(super) fn copy_params(em: &mut Emit<'_, '_>, params: u32) -> Result<(), TranslateError> {
    let mut index = 0u32;
    while index < params {
        let var = em
            .locals
            .get(index as usize)
            .copied()
            .flatten()
            .ok_or(TranslateError::Unsupported)?;
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
