//! Stack steps for arithmetic, locals, and gpu imports.

use naga::BinaryOperator;
use naga::Expression;

use crate::cx::Val;
use crate::env::Emit;
use crate::error::TranslateError;
use crate::globals;
use crate::ir::Op;
use crate::storage;
use crate::ty;
use crate::ty::Imp;
use crate::uniform;

pub(super) fn step(em: &mut Emit<'_, '_>, op: Op) -> Result<(), TranslateError> {
    match op {
        Op::I32(value) => em.push(Val::Imm(value)),
        Op::F32(bits) => {
            let lit = em.f32(f32::from_bits(bits));
            em.push(Val::Scalar(lit))
        }
        Op::Local(index) => local_get(em, index, false),
        Op::LocalI32(index) => local_get(em, index, true),
        Op::LocalSet(index) => local_set(em, index, false),
        Op::LocalTee(index) => local_set(em, index, true),
        Op::Drop => em.pop().map(|_| ()),
        Op::Call(index) => call(em, index),
        Op::Mul => bin(em, BinaryOperator::Multiply),
        Op::Add => bin(em, BinaryOperator::Add),
        Op::I32Add => i32_bin(em, u32::wrapping_add, BinaryOperator::Add),
        Op::I32Mul => i32_bin(em, u32::wrapping_mul, BinaryOperator::Multiply),
        Op::I32Sub => i32_bin(em, u32::wrapping_sub, BinaryOperator::Subtract),
        Op::I32Eqz => i32_eqz(em),
        Op::I32Eq => i32_cmp(em, |a, b| a == b, BinaryOperator::Equal),
        Op::I32Ne => i32_cmp(em, |a, b| a != b, BinaryOperator::NotEqual),
        Op::I32LtU => i32_cmp(em, |a, b| a < b, BinaryOperator::Less),
        Op::I32GtU => i32_cmp(em, |a, b| a > b, BinaryOperator::Greater),
        Op::Select => wasm_select(em),
    }
}

fn call(em: &mut Emit<'_, '_>, index: u32) -> Result<(), TranslateError> {
    let imp = em
        .imports
        .get(index as usize)
        .and_then(|import| ty::classify(&import.name));
    let Some(imp) = imp else {
        return crate::call::call_defined(em, index);
    };
    match imp {
        Imp::ReadUniform => read_uniform(em),
        Imp::Sample => sample(em),
        Imp::Read(spec) => storage::load(em, spec),
        Imp::Write(spec) => storage::store(em, spec, index),
        Imp::Uniform(uni) => uniform::load(em, uni),
    }
}

fn read_uniform(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let offset = em.pop_imm()?;
    let binding = em.pop_imm()?;
    let group = em.pop_imm()?;
    if offset != 0 {
        return Err(TranslateError::Unsupported);
    }
    let var = globals::uniform(em.cx, group, binding)?;
    let pointer = em.ex(Expression::GlobalVariable(var));
    let loaded = em.ex(Expression::Load { pointer });
    em.stack.push(Val::Vec(loaded));
    Ok(())
}

fn sample(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let v_val = em.pop()?;
    let u_val = em.pop()?;
    let v = crate::env::float_of(v_val)?;
    let u = crate::env::float_of(u_val)?;
    let samp_binding = em.pop_imm()?;
    let samp_group = em.pop_imm()?;
    let tex_binding = em.pop_imm()?;
    let tex_group = em.pop_imm()?;
    let tex = globals::texture(em.cx, tex_group, tex_binding);
    let samp = globals::sampler(em.cx, samp_group, samp_binding);
    let image = em.ex(Expression::GlobalVariable(tex));
    let sampler = em.ex(Expression::GlobalVariable(samp));
    let coordinate = em.ex(Expression::Compose {
        ty: em.cx.vec2,
        components: vec![u, v],
    });
    let sampled = em.ex(Expression::ImageSample {
        image,
        sampler,
        gather: None,
        coordinate,
        array_index: None,
        offset: None,
        level: naga::SampleLevel::Auto,
        depth_ref: None,
        clamp_to_edge: false,
    });
    em.stack.push(Val::Vec(sampled));
    Ok(())
}

fn local_get(em: &mut Emit<'_, '_>, index: u32, i32: bool) -> Result<(), TranslateError> {
    let slot = index as usize;
    if let Some(bits) = em.baked.get(slot).copied().flatten() {
        return em.push(Val::Imm(bits));
    }
    let var = em.locals.get(slot).copied().flatten();
    if let Some(var) = var {
        let loaded = em.load_var(var);
        return push_loaded(em, loaded, i32);
    }
    if em.args_at.get(slot).copied().flatten().is_some() {
        let loaded = em.param_expr(index)?;
        return push_loaded(em, loaded, i32);
    }
    // Wasm zeros locals that are not parameters. A slot with no shadow was
    // never written, so the value is the constant zero the analysis recorded.
    if slot < em.locals.len() {
        return zero_local(em, i32);
    }
    Err(TranslateError::Unsupported)
}

fn push_loaded(
    em: &mut Emit<'_, '_>,
    loaded: naga::Handle<Expression>,
    i32: bool,
) -> Result<(), TranslateError> {
    let val = if i32 {
        Val::Index(loaded)
    } else {
        Val::Scalar(loaded)
    };
    em.push(val)
}

fn zero_local(em: &mut Emit<'_, '_>, i32: bool) -> Result<(), TranslateError> {
    if i32 {
        return em.push(Val::Imm(0));
    }
    let lit = em.f32(0.0);
    em.push(Val::Scalar(lit))
}

fn local_set(em: &mut Emit<'_, '_>, index: u32, tee: bool) -> Result<(), TranslateError> {
    let val = em.pop()?;
    if let Some(slot) = em.baked.get_mut(index as usize) {
        *slot = None;
    }
    if tee {
        em.stack.push(val);
    }
    let var = em
        .locals
        .get(index as usize)
        .copied()
        .flatten()
        .ok_or(TranslateError::Unsupported)?;
    let bits = match val {
        Val::Imm(value) => em.u32(value),
        Val::Index(handle) | Val::Scalar(handle) => handle,
        Val::Vec(_) => return Err(TranslateError::Unsupported),
    };
    em.store_var(var, bits);
    Ok(())
}

fn bin(em: &mut Emit<'_, '_>, op: BinaryOperator) -> Result<(), TranslateError> {
    let right = em.pop()?;
    let left = em.pop()?;
    match (left, right) {
        (Val::Vec(left), Val::Vec(right)) => {
            let value = arith(em, op, left, right);
            em.stack.push(Val::Vec(value));
            Ok(())
        }
        (Val::Scalar(left), Val::Scalar(right)) => {
            let value = arith(em, op, left, right);
            em.stack.push(Val::Scalar(value));
            Ok(())
        }
        _ => Err(TranslateError::Unsupported),
    }
}

fn arith(
    em: &mut Emit<'_, '_>,
    op: BinaryOperator,
    left: naga::Handle<Expression>,
    right: naga::Handle<Expression>,
) -> naga::Handle<Expression> {
    em.ex(Expression::Binary { op, left, right })
}

/// Immediate bits, or the two operands already lowered to `u32`.
enum Ready {
    Imm(u32),
    Expr(naga::Handle<Expression>, naga::Handle<Expression>),
}

fn i32_ready(
    em: &mut Emit<'_, '_>,
    fold: impl Fn(u32, u32) -> u32,
) -> Result<Ready, TranslateError> {
    let right = em.pop()?;
    let left = em.pop()?;
    if let (Val::Imm(a), Val::Imm(b)) = (left, right) {
        return Ok(Ready::Imm(fold(a, b)));
    }
    Ok(Ready::Expr(em.as_uint(left)?, em.as_uint(right)?))
}

fn i32_bin(
    em: &mut Emit<'_, '_>,
    fold: fn(u32, u32) -> u32,
    op: BinaryOperator,
) -> Result<(), TranslateError> {
    match i32_ready(em, fold)? {
        Ready::Imm(value) => em.push(Val::Imm(value)),
        Ready::Expr(left, right) => {
            let value = arith(em, op, left, right);
            em.push(Val::Index(value))
        }
    }
}

fn i32_cmp(
    em: &mut Emit<'_, '_>,
    fold: fn(u32, u32) -> bool,
    op: BinaryOperator,
) -> Result<(), TranslateError> {
    match i32_ready(em, |a, b| u32::from(fold(a, b)))? {
        Ready::Imm(value) => em.push(Val::Imm(value)),
        Ready::Expr(left, right) => {
            let cond = arith(em, op, left, right);
            let value = zero_one(em, cond);
            em.push(Val::Index(value))
        }
    }
}

fn i32_eqz(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let val = em.pop()?;
    if let Val::Imm(value) = val {
        em.stack.push(Val::Imm(u32::from(value == 0)));
        return Ok(());
    }
    let bits = em.as_uint(val)?;
    let zero = em.u32(0);
    let cond = arith(em, BinaryOperator::Equal, bits, zero);
    let value = zero_one(em, cond);
    em.stack.push(Val::Index(value));
    Ok(())
}

fn zero_one(em: &mut Emit<'_, '_>, cond: naga::Handle<Expression>) -> naga::Handle<Expression> {
    let accept = em.u32(1);
    let reject = em.u32(0);
    em.pick(cond, accept, reject)
}

fn wasm_select(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let cond_val = em.pop()?;
    let cond = em.as_bool(cond_val)?;
    let reject = em.pop()?;
    let accept = em.pop()?;
    let value = match (accept, reject) {
        (Val::Scalar(left), Val::Scalar(right)) => Val::Scalar(em.pick(cond, left, right)),
        (Val::Vec(left), Val::Vec(right)) => Val::Vec(em.pick(cond, left, right)),
        (left, right) => {
            let accept = em.as_uint(left)?;
            let reject = em.as_uint(right)?;
            Val::Index(em.pick(cond, accept, reject))
        }
    };
    em.stack.push(value);
    Ok(())
}
