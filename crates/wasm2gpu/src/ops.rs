//! Stack steps for arithmetic, locals, and gpu imports.

use naga::BinaryOperator;
use naga::Expression;

use crate::cx::PtrOff;
use crate::cx::Val;
use crate::env::Emit;
use crate::error::TranslateError;
use crate::globals;
use crate::ir::Op;
use crate::storage;
use crate::ty;
use crate::ty::Elem;
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
        Op::I32Add => ptr_bin(em, true, u32::wrapping_add),
        Op::I32Mul => i32_bin(em, u32::wrapping_mul, BinaryOperator::Multiply),
        Op::I32Sub => ptr_bin(em, false, u32::wrapping_sub),
        Op::I32Shl => i32_bin(em, u32::wrapping_shl, BinaryOperator::ShiftLeft),
        Op::I32ShrU => i32_bin(em, u32::wrapping_shr, BinaryOperator::ShiftRight),
        Op::I32Eqz => i32_eqz(em),
        Op::I32Eq => i32_cmp(em, |a, b| a == b, BinaryOperator::Equal),
        Op::I32Ne => i32_cmp(em, |a, b| a != b, BinaryOperator::NotEqual),
        Op::I32LtU => i32_cmp(em, |a, b| a < b, BinaryOperator::Less),
        Op::I32GtU => i32_cmp(em, |a, b| a > b, BinaryOperator::Greater),
        Op::F32Lt => f32_lt(em),
        Op::LoadI32(offset) => storage::load_ptr(em, offset, Elem::U32),
        Op::LoadF32(offset) => storage::load_ptr(em, offset, Elem::F32),
        Op::StoreI32(offset) => storage::store_ptr(em, offset, Elem::U32),
        Op::StoreF32(offset) => storage::store_ptr(em, offset, Elem::F32),
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
        Imp::Base => storage_base(em),
        Imp::ReadUniform => read_uniform(em),
        Imp::Sample => sample(em),
        Imp::Read(spec) => storage::load(em, spec),
        Imp::Write(spec) => storage::store(em, spec, index),
        Imp::Uniform(uni) => uniform::load(em, uni),
    }
}

/// Fold `storage_base`'s provenance pair into an address.
///
/// The binding's base tag is the binding itself: only byte offsets are
/// ever added, and the access names the global through group and binding.
fn storage_base(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let binding = em.pop_imm()?;
    let group = em.pop_imm()?;
    em.push(Val::Ptr {
        group,
        binding,
        off: PtrOff::Imm(0),
    })
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
    if let Some((group, binding)) = em.ptr_slots.get(slot).copied().flatten() {
        let var = em
            .locals
            .get(slot)
            .copied()
            .flatten()
            .ok_or(TranslateError::Unsupported)?;
        let loaded = em.load_var(var);
        return em.push(Val::Ptr {
            group,
            binding,
            off: PtrOff::Dyn(loaded),
        });
    }
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
    let provenance = match val {
        Val::Ptr { group, binding, .. } => Some((group, binding)),
        _ => None,
    };
    if let Some(slot) = em.ptr_slots.get_mut(index as usize) {
        *slot = provenance;
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
        Val::Ptr { off, .. } => em.off_expr(off),
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

fn ready_of(
    em: &mut Emit<'_, '_>,
    left: Val,
    right: Val,
    fold: impl Fn(u32, u32) -> u32,
) -> Result<Ready, TranslateError> {
    if let (Val::Imm(a), Val::Imm(b)) = (left, right) {
        return Ok(Ready::Imm(fold(a, b)));
    }
    Ok(Ready::Expr(em.as_uint(left)?, em.as_uint(right)?))
}

fn i32_ready(
    em: &mut Emit<'_, '_>,
    fold: impl Fn(u32, u32) -> u32,
) -> Result<Ready, TranslateError> {
    let right = em.pop()?;
    let left = em.pop()?;
    ready_of(em, left, right, fold)
}

fn i32_bin(
    em: &mut Emit<'_, '_>,
    fold: fn(u32, u32) -> u32,
    op: BinaryOperator,
) -> Result<(), TranslateError> {
    let right = em.pop()?;
    let left = em.pop()?;
    i32_pair(em, left, right, fold, op)
}

fn i32_pair(
    em: &mut Emit<'_, '_>,
    left: Val,
    right: Val,
    fold: fn(u32, u32) -> u32,
    op: BinaryOperator,
) -> Result<(), TranslateError> {
    match ready_of(em, left, right, fold)? {
        Ready::Imm(value) => em.push(Val::Imm(value)),
        Ready::Expr(left, right) => {
            let value = arith(em, op, left, right);
            em.push(Val::Index(value))
        }
    }
}

/// `i32.add` / `i32.sub` that keep a storage pointer's provenance.
///
/// A pointer plus or minus an offset stays a pointer; anything else falls
/// back to plain integer arithmetic, where a pointer operand is rejected.
fn ptr_bin(
    em: &mut Emit<'_, '_>,
    add: bool,
    fold: fn(u32, u32) -> u32,
) -> Result<(), TranslateError> {
    let op = if add {
        BinaryOperator::Add
    } else {
        BinaryOperator::Subtract
    };
    let right = em.pop()?;
    let left = em.pop()?;
    match (left, right) {
        (Val::Ptr { .. }, Val::Ptr { .. }) => Err(TranslateError::Unsupported),
        (Val::Ptr { group, binding, off }, operand) => {
            let off = off_math(em, off, operand, add, fold)?;
            em.push(Val::Ptr {
                group,
                binding,
                off,
            })
        }
        (operand, Val::Ptr { group, binding, off }) if add => {
            let off = off_math(em, off, operand, add, fold)?;
            em.push(Val::Ptr {
                group,
                binding,
                off,
            })
        }
        (left, right) => i32_pair(em, left, right, fold, op),
    }
}

/// Fold or emit `off ± operand` as a pointer's new byte offset.
fn off_math(
    em: &mut Emit<'_, '_>,
    off: PtrOff,
    operand: Val,
    add: bool,
    fold: fn(u32, u32) -> u32,
) -> Result<PtrOff, TranslateError> {
    let op = if add {
        BinaryOperator::Add
    } else {
        BinaryOperator::Subtract
    };
    match (off, operand) {
        (PtrOff::Imm(value), Val::Imm(delta)) => Ok(PtrOff::Imm(fold(value, delta))),
        (off, operand) => {
            let left = em.off_expr(off);
            let right = em.as_uint(operand)?;
            Ok(PtrOff::Dyn(arith(em, op, left, right)))
        }
    }
}

fn f32_lt(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let right = crate::env::float_of(em.pop()?)?;
    let left = crate::env::float_of(em.pop()?)?;
    let cond = arith(em, BinaryOperator::Less, left, right);
    let value = zero_one(em, cond);
    em.push(Val::Index(value))
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
    if let Some(merged) = em.merge_ptr(cond, accept, reject) {
        return em.push(merged);
    }
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
