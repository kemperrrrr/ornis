//! Emit a naga `Call` and push the callee's results onto the caller stack.
//!
//! A constant result is an immediate. One dynamic result is the call value.
//! Several dynamic results are fields of the anonymous struct the callee
//! returns.

use naga::Expression;
use naga::Span;
use naga::Statement;

use crate::cx::Spec;
use crate::cx::Val;
use crate::cx::sp;
use crate::env::Emit;
use crate::error::TranslateError;
use crate::ir::Kind;

pub(super) fn invoke(
    em: &mut Emit<'_, '_>,
    spec: &Spec,
    args: &[Val],
) -> Result<(), TranslateError> {
    let mut arguments = Vec::new();
    let mut index = 0usize;
    while index < spec.nparams as usize {
        if spec.baked[index].is_none() {
            arguments.push(arg_expr(em, args[index])?);
        }
        index += 1;
    }
    let dyn_count = count_dynamic(spec);
    let result = if dyn_count == 0 {
        None
    } else {
        Some(
            em.func
                .expressions
                .append(Expression::CallResult(spec.handle), Span::default()),
        )
    };
    em.block.push(
        Statement::Call {
            function: spec.handle,
            arguments,
            result,
        },
        sp(),
    );
    push_results(em, spec, result, dyn_count)
}

fn arg_expr(em: &mut Emit<'_, '_>, val: Val) -> Result<naga::Handle<Expression>, TranslateError> {
    match val {
        Val::Index(handle) | Val::Scalar(handle) => Ok(handle),
        Val::Imm(value) => Ok(em.u32(value)),
        Val::Vec(_) => Err(TranslateError::Unsupported),
    }
}

fn count_dynamic(spec: &Spec) -> u32 {
    let mut count = 0u32;
    let mut index = 0usize;
    while index < spec.nresults as usize {
        if spec.result_imm[index].is_none() {
            count += 1;
        }
        index += 1;
    }
    count
}

fn push_results(
    em: &mut Emit<'_, '_>,
    spec: &Spec,
    result: Option<naga::Handle<Expression>>,
    dyn_count: u32,
) -> Result<(), TranslateError> {
    let mut dyn_at = 0u32;
    let mut index = 0usize;
    while index < spec.nresults as usize {
        if let Some(value) = spec.result_imm[index] {
            em.push(Val::Imm(value))?;
        } else {
            let base = result.ok_or(TranslateError::Unsupported)?;
            let handle = if dyn_count == 1 {
                base
            } else {
                em.ex(Expression::AccessIndex {
                    base,
                    index: dyn_at,
                })
            };
            let val = if spec.result_kind[index] == Kind::F32 {
                Val::Scalar(handle)
            } else {
                Val::Index(handle)
            };
            em.push(val)?;
            dyn_at += 1;
        }
        index += 1;
    }
    Ok(())
}
