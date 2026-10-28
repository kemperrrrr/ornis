//! Stack steps for the straight-line subset.

use naga::BinaryOperator;
use naga::Expression;
use naga::Function;
use naga::SampleLevel;

use crate::cx::Cx;
use crate::cx::LANES;
use crate::cx::READ_PARAMS;
use crate::cx::SAMPLE_PARAMS;
use crate::cx::Val;
use crate::cx::expr;
use crate::cx::import;
use crate::error::TranslateError;
use crate::globals;
use crate::ir::Import;
use crate::ir::Op;
use ornis_gpu_abi::symbols;

pub(super) fn step(
    cx: &mut Cx<'_>,
    imports: &[Import],
    body: &mut Function,
    stack: &mut Vec<Val>,
    op: Op,
) -> Result<(), TranslateError> {
    match op {
        Op::I32(value) => stack.push(Val::Imm(value)),
        Op::F32(bits) => stack.push(Val::Scalar(lit(body, f32::from_bits(bits)))),
        Op::Local(index) => stack.push(Val::Scalar(arg(body, index))),
        Op::Call(index) => call(cx, imports, body, stack, index)?,
        Op::Mul => bin(body, stack, BinaryOperator::Multiply)?,
        Op::Add => bin(body, stack, BinaryOperator::Add)?,
    }
    Ok(())
}

fn lit(body: &mut Function, value: f32) -> naga::Handle<Expression> {
    expr(body, Expression::Literal(naga::Literal::F32(value)))
}

fn arg(body: &mut Function, index: u32) -> naga::Handle<Expression> {
    expr(body, Expression::FunctionArgument(index))
}

fn call(
    cx: &mut Cx<'_>,
    imports: &[Import],
    body: &mut Function,
    stack: &mut Vec<Val>,
    index: u32,
) -> Result<(), TranslateError> {
    let import = import(imports, index)?;
    if import.name == symbols::RAW_READ
        && import.sig.params == READ_PARAMS
        && import.sig.results == LANES
    {
        read_uniform(cx, body, stack)
    } else if import.name == symbols::TEXTURE_SAMPLE_2D
        && import.sig.params == SAMPLE_PARAMS
        && import.sig.results == LANES
    {
        sample(cx, body, stack)
    } else {
        Err(TranslateError::Unsupported)
    }
}

fn read_uniform(
    cx: &mut Cx<'_>,
    body: &mut Function,
    stack: &mut Vec<Val>,
) -> Result<(), TranslateError> {
    let offset = pop_imm(stack)?;
    let binding = pop_imm(stack)?;
    let group = pop_imm(stack)?;
    if offset != 0 {
        return Err(TranslateError::Unsupported);
    }
    let var = globals::uniform(cx, group, binding);
    let pointer = expr(body, Expression::GlobalVariable(var));
    stack.push(Val::Vec(expr(body, Expression::Load { pointer })));
    Ok(())
}

fn sample(
    cx: &mut Cx<'_>,
    body: &mut Function,
    stack: &mut Vec<Val>,
) -> Result<(), TranslateError> {
    let v = pop_scalar(stack)?;
    let u = pop_scalar(stack)?;
    let samp_binding = pop_imm(stack)?;
    let samp_group = pop_imm(stack)?;
    let tex_binding = pop_imm(stack)?;
    let tex_group = pop_imm(stack)?;
    let image = expr(
        body,
        Expression::GlobalVariable(globals::texture(cx, tex_group, tex_binding)),
    );
    let sampler = expr(
        body,
        Expression::GlobalVariable(globals::sampler(cx, samp_group, samp_binding)),
    );
    let coordinate = expr(
        body,
        Expression::Compose {
            ty: cx.vec2,
            components: vec![u, v],
        },
    );
    stack.push(Val::Vec(expr(
        body,
        Expression::ImageSample {
            image,
            sampler,
            gather: None,
            coordinate,
            array_index: None,
            offset: None,
            level: SampleLevel::Auto,
            depth_ref: None,
            clamp_to_edge: false,
        },
    )));
    Ok(())
}

fn pop_imm(stack: &mut Vec<Val>) -> Result<u32, TranslateError> {
    match stack.pop() {
        Some(Val::Imm(value)) => Ok(value),
        Some(_) => Err(TranslateError::DynamicBinding),
        None => Err(TranslateError::Unsupported),
    }
}

fn pop_scalar(stack: &mut Vec<Val>) -> Result<naga::Handle<Expression>, TranslateError> {
    match stack.pop() {
        Some(Val::Scalar(handle)) => Ok(handle),
        _ => Err(TranslateError::Unsupported),
    }
}

fn binary(
    body: &mut Function,
    op: BinaryOperator,
    left: naga::Handle<Expression>,
    right: naga::Handle<Expression>,
) -> naga::Handle<Expression> {
    expr(body, Expression::Binary { op, left, right })
}

fn bin(
    body: &mut Function,
    stack: &mut Vec<Val>,
    op: BinaryOperator,
) -> Result<(), TranslateError> {
    let right = stack.pop().ok_or(TranslateError::Unsupported)?;
    let left = stack.pop().ok_or(TranslateError::Unsupported)?;
    match (left, right) {
        (Val::Vec(left), Val::Vec(right)) => {
            stack.push(Val::Vec(binary(body, op, left, right)));
            Ok(())
        }
        (Val::Scalar(left), Val::Scalar(right)) => {
            stack.push(Val::Scalar(binary(body, op, left, right)));
            Ok(())
        }
        _ => Err(TranslateError::Unsupported),
    }
}
