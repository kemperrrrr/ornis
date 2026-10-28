//! Storage `read` / `write`: one global, element index as an access chain.
//!
//! Group and binding are popped as immediates (the binding's provenance).
//! A runtime index becomes `OpAccessChain` on that global. A runtime group
//! or binding is [`crate::TranslateError::DynamicBinding`] — bindless stays
//! out of scope.

use naga::Expression;
use naga::Function;
use naga::Span;
use naga::Statement;

use crate::cx::Cx;
use crate::cx::Val;
use crate::cx::expr;
use crate::error::TranslateError;
use crate::globals;
use crate::ops;

enum Ix {
    Const(u32),
    Dyn(naga::Handle<Expression>),
}

pub(super) fn storage_load(
    cx: &mut Cx<'_>,
    body: &mut Function,
    stack: &mut Vec<Val>,
) -> Result<(), TranslateError> {
    let index = pop_index(stack)?;
    let (group, binding) = pop_provenance(stack)?;
    let pointer = element_ptr(cx, body, group, binding, index);
    stack.push(Val::Vec(expr(body, Expression::Load { pointer })));
    Ok(())
}

pub(super) fn storage_store(
    cx: &mut Cx<'_>,
    body: &mut Function,
    stack: &mut Vec<Val>,
) -> Result<(), TranslateError> {
    let value = pop_vec4(cx, body, stack)?;
    let index = pop_index(stack)?;
    let (group, binding) = pop_provenance(stack)?;
    let pointer = element_ptr(cx, body, group, binding, index);
    body.body
        .push(Statement::Store { pointer, value }, Span::default());
    Ok(())
}

fn pop_index(stack: &mut Vec<Val>) -> Result<Ix, TranslateError> {
    match stack.pop() {
        Some(Val::Imm(value)) => Ok(Ix::Const(value)),
        Some(Val::Index(handle)) => Ok(Ix::Dyn(handle)),
        Some(_) => Err(TranslateError::Unsupported),
        None => Err(TranslateError::Unsupported),
    }
}

fn pop_provenance(stack: &mut Vec<Val>) -> Result<(u32, u32), TranslateError> {
    let binding = ops::pop_imm(stack)?;
    let group = ops::pop_imm(stack)?;
    Ok((group, binding))
}

fn element_ptr(
    cx: &mut Cx<'_>,
    body: &mut Function,
    group: u32,
    binding: u32,
    index: Ix,
) -> naga::Handle<Expression> {
    let base = expr(
        body,
        Expression::GlobalVariable(globals::storage(cx, group, binding)),
    );
    let index = match index {
        Ix::Const(value) => expr(body, Expression::Literal(naga::Literal::U32(value))),
        Ix::Dyn(handle) => handle,
    };
    expr(body, Expression::Access { base, index })
}

fn pop_vec4(
    cx: &Cx<'_>,
    body: &mut Function,
    stack: &mut Vec<Val>,
) -> Result<naga::Handle<Expression>, TranslateError> {
    let w = ops::pop_scalar(stack)?;
    let z = ops::pop_scalar(stack)?;
    let y = ops::pop_scalar(stack)?;
    let x = ops::pop_scalar(stack)?;
    Ok(expr(
        body,
        Expression::Compose {
            ty: cx.vec4,
            components: vec![x, y, z, w],
        },
    ))
}
