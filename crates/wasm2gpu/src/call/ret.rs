//! Pack a specialized function's results into a naga return.
//!
//! Constant `i32` results stay immediates on the caller. Dynamic results
//! become the function result: one scalar, or an anonymous struct when
//! there are several.

use naga::Expression;
use naga::Function;
use naga::FunctionResult;
use naga::Span;
use naga::Statement;
use naga::StructMember;
use naga::Type;
use naga::TypeInner;

use crate::cx::Cx;
use crate::cx::Val;
use crate::cx::expr;
use crate::cx::sp;
use crate::error::TranslateError;
use crate::ir::MAX_PARAMS;
use crate::ir::Sig;

pub(super) fn finish(
    body: &mut Function,
    cx: &mut Cx<'_>,
    stack: &[Val],
    sig: &Sig,
) -> Result<([Option<u32>; MAX_PARAMS], u8), TranslateError> {
    if stack.len() != sig.results as usize {
        return Err(TranslateError::Unsupported);
    }
    let mut result_imm = [None; MAX_PARAMS];
    let mut dynamic = Vec::new();
    let mut index = 0usize;
    while index < stack.len() {
        match stack[index] {
            Val::Imm(value) => result_imm[index] = Some(value),
            Val::Scalar(handle) => dynamic.push((cx.scalar, handle)),
            Val::Index(handle) => dynamic.push((cx.uint, handle)),
            Val::Vec(_) => return Err(TranslateError::Unsupported),
        }
        index += 1;
    }
    if let Some((ty, value)) = pack(body, cx, &dynamic)? {
        body.result = Some(FunctionResult { ty, binding: None });
        body.body
            .push(Statement::Return { value: Some(value) }, sp());
    }
    let nresults = u8::try_from(stack.len()).map_err(|_| TranslateError::Unsupported)?;
    Ok((result_imm, nresults))
}

type Packed = Option<(naga::Handle<Type>, naga::Handle<Expression>)>;

fn pack(
    body: &mut Function,
    cx: &mut Cx<'_>,
    dynamic: &[(naga::Handle<Type>, naga::Handle<Expression>)],
) -> Result<Packed, TranslateError> {
    match dynamic {
        [] => Ok(None),
        [(ty, handle)] => Ok(Some((*ty, *handle))),
        _ => {
            let ty = packed_ty(cx, dynamic)?;
            let components = dynamic.iter().map(|(_, handle)| *handle).collect();
            let value = expr(body, Expression::Compose { ty, components });
            Ok(Some((ty, value)))
        }
    }
}

fn packed_ty(
    cx: &mut Cx<'_>,
    dynamic: &[(naga::Handle<Type>, naga::Handle<Expression>)],
) -> Result<naga::Handle<Type>, TranslateError> {
    let mut members = Vec::new();
    let mut offset = 0u32;
    for (ty, _) in dynamic {
        members.push(StructMember {
            name: None,
            ty: *ty,
            binding: None,
            offset,
        });
        offset = offset
            .checked_add(ornis_gpu_abi::F32_SIZE)
            .ok_or(TranslateError::Unsupported)?;
    }
    Ok(cx.module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Struct {
                members,
                span: offset,
            },
        },
        Span::default(),
    ))
}
