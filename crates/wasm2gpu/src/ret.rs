//! Entry-point results: vertex struct and fragment `vec4`.

use naga::Binding;
use naga::BuiltIn;
use naga::Expression;
use naga::Function;
use naga::FunctionResult;
use naga::Span;
use naga::Statement;
use naga::StructMember;
use naga::Type;
use naga::TypeInner;

use crate::cx::Cx;
use crate::cx::LANES;
use crate::cx::VS_RESULTS;
use crate::cx::Val;
use crate::cx::expr;
use crate::cx::location;
use crate::error::TranslateError;

const CLIP_OFFSET: u32 = 0;
const UV_U_OFFSET: u32 = 16;
const UV_V_OFFSET: u32 = 20;
const VS_SPAN: u32 = 32;

pub(super) fn vertex(
    cx: &mut Cx<'_>,
    body: &mut Function,
    stack: &[Val],
) -> Result<naga::Handle<Expression>, TranslateError> {
    if stack.len() != VS_RESULTS as usize {
        return Err(TranslateError::Unsupported);
    }
    let clip = expr(
        body,
        Expression::Compose {
            ty: cx.vec4,
            components: lanes(stack)?,
        },
    );
    let u = scalar_at(stack, LANES as usize)?;
    let v = scalar_at(stack, (LANES + 1) as usize)?;
    let ty = vs_out(cx);
    body.result = Some(FunctionResult { ty, binding: None });
    Ok(expr(
        body,
        Expression::Compose {
            ty,
            components: vec![clip, u, v],
        },
    ))
}

pub(super) fn fragment(
    cx: &mut Cx<'_>,
    body: &mut Function,
    stack: &[Val],
) -> Result<naga::Handle<Expression>, TranslateError> {
    body.result = Some(FunctionResult {
        ty: cx.vec4,
        binding: Some(location(0)),
    });
    match stack {
        [Val::Vec(handle)] => Ok(*handle),
        _ => Err(TranslateError::Unsupported),
    }
}

pub(super) fn finish(body: &mut Function, value: naga::Handle<Expression>) {
    body.body
        .push(Statement::Return { value: Some(value) }, Span::default());
}

fn lanes(stack: &[Val]) -> Result<Vec<naga::Handle<Expression>>, TranslateError> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < LANES as usize {
        out.push(scalar_at(stack, index)?);
        index += 1;
    }
    Ok(out)
}

fn scalar_at(stack: &[Val], index: usize) -> Result<naga::Handle<Expression>, TranslateError> {
    match stack.get(index) {
        Some(Val::Scalar(handle)) => Ok(*handle),
        _ => Err(TranslateError::Unsupported),
    }
}

fn vs_out(cx: &mut Cx<'_>) -> naga::Handle<Type> {
    let members = vec![
        member(
            "clip",
            cx.vec4,
            CLIP_OFFSET,
            Some(Binding::BuiltIn(BuiltIn::Position { invariant: false })),
        ),
        member("u", cx.scalar, UV_U_OFFSET, Some(location(0))),
        member("v", cx.scalar, UV_V_OFFSET, Some(location(1))),
    ];
    cx.module.types.insert(
        Type {
            name: Some("VsOut".to_string()),
            inner: TypeInner::Struct {
                members,
                span: VS_SPAN,
            },
        },
        Span::default(),
    )
}

fn member(
    name: &str,
    ty: naga::Handle<Type>,
    offset: u32,
    binding: Option<Binding>,
) -> StructMember {
    StructMember {
        name: Some(name.to_string()),
        ty,
        binding,
        offset,
    }
}
