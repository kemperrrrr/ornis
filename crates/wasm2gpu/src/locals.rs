//! The wasm locals of the function being lowered.
//!
//! A local keeps three things: its kind, whether its value stays an
//! expression, and the expression itself while the value is one. They live
//! together because they describe the same slot.

use naga::Expression;
use naga::Literal;

use crate::cx::Cx;
use crate::cx::Val;
use crate::cx::sp;
use crate::error::TranslateError;
use crate::ir::Func;

/// The wasm locals of the function being lowered.
pub(super) struct Locals {
    /// Kind of every wasm local.
    pub(super) kinds: Vec<crate::ir::Kind>,
    /// Slots whose value stays an expression.
    pub(super) expr: Vec<bool>,
    /// The value of an expression slot, while it is still one expression.
    ///
    /// A slot written once keeps its value here. A read uses it directly, so
    /// the value never reaches a variable.
    pub(super) pending: Vec<Option<Val>>,
}

impl Locals {
    /// Describe the locals of a function body.
    pub(super) fn of(func: &Func, expr: Vec<bool>) -> Locals {
        let pending = vec![None; expr.len()];
        Locals {
            kinds: func.locals.clone(),
            expr,
            pending,
        }
    }

    /// The shadow variable of a slot, created when it is first needed.
    ///
    /// The variable starts from the wasm zero: naga wants a constant
    /// initializer. A parameter keeps its argument value in a store, so its
    /// shadow is made before the body runs.
    pub(super) fn shadow_for(
        &mut self,
        func: &mut naga::Function,
        cx: &Cx<'_>,
        shadows: &mut [Option<naga::Handle<naga::LocalVariable>>],
        slot: usize,
    ) -> Result<naga::Handle<naga::LocalVariable>, TranslateError> {
        if let Some(var) = shadows.get(slot).copied().flatten() {
            return Ok(var);
        }
        let kind = self
            .kinds
            .get(slot)
            .copied()
            .ok_or(TranslateError::Unsupported)?;
        let var = shadow(func, cx, kind)?;
        if let Some(entry) = shadows.get_mut(slot) {
            *entry = Some(var);
        }
        // The value now lives in the variable, not in an expression.
        self.clear_pending(slot);
        Ok(var)
    }

    /// Keep the value of an expression slot as an expression.
    pub(super) fn set_pending(&mut self, slot: usize, val: Val) {
        if let Some(entry) = self.pending.get_mut(slot) {
            *entry = Some(val);
        }
    }

    /// Forget the expression of a slot: its value lives in a variable now.
    pub(super) fn clear_pending(&mut self, slot: usize) {
        if let Some(entry) = self.pending.get_mut(slot) {
            *entry = None;
        }
    }
}

/// A shadow variable of a wasm local.
///
/// The variable starts from the wasm zero: naga wants a constant
/// initializer, so the value a local holds comes from a store.
pub(super) fn shadow(
    func: &mut naga::Function,
    cx: &Cx<'_>,
    kind: crate::ir::Kind,
) -> Result<naga::Handle<naga::LocalVariable>, TranslateError> {
    let (ty, init) = if kind == crate::ir::Kind::I32 {
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

/// Where the argument value of every parameter comes from.
///
/// A baked parameter has no argument: its value is a constant. The compute
/// entry takes the invocation id instead of the argument vector.
pub(super) fn arg_map(
    func: &Func,
    baked: &[Option<u32>],
    id_param: Option<u32>,
) -> Vec<Option<u32>> {
    let mut out = vec![None; func.locals.len()];
    if id_param.is_some() {
        if func.sig.params > 0 {
            out[0] = Some(0);
        }
        return out;
    }
    let mut slot = 0u32;
    let mut index = 0u32;
    while index < func.sig.params {
        let immediate = baked.get(index as usize).copied().flatten();
        if immediate.is_none() {
            out[index as usize] = Some(slot);
            slot += 1;
        }
        index += 1;
    }
    out
}

fn const_expr(func: &mut naga::Function, value: Expression) -> naga::Handle<Expression> {
    func.expressions.append(value, naga::Span::default())
}
