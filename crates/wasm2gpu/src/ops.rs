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
use crate::ty::Elem;
use crate::ty::Imp;
use crate::uniform;

mod compute;
mod float;
mod int;
mod math;
mod ptr;
mod texture;

use float::bin;
use float::f32_abs;
use float::f32_cmp;
use float::f32_from_u32;
use float::f32_neg;
use float::f32_sqrt;
use int::i32_bin;
use int::i32_cmp;
use int::i32_eqz;
use int::shr_s;
use int::wasm_select;
use ptr::local_ptr;
use ptr::provenance;
use ptr::ptr_bin;
use ptr::storage_base;

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
        Op::Sub => bin(em, BinaryOperator::Subtract),
        Op::Div => bin(em, BinaryOperator::Divide),
        Op::F32Abs => f32_abs(em),
        Op::F32Neg => f32_neg(em),
        Op::F32Sqrt => f32_sqrt(em),
        Op::F32Min => math::scalar_binary(em, naga::MathFunction::Min),
        Op::F32Max => math::scalar_binary(em, naga::MathFunction::Max),
        Op::F32ConvertI32U => f32_from_u32(em),
        Op::I32Add => ptr_bin(em, true, u32::wrapping_add),
        Op::I32Mul => i32_bin(em, u32::wrapping_mul, BinaryOperator::Multiply),
        Op::I32Sub => ptr_bin(em, false, u32::wrapping_sub),
        Op::I32Shl => i32_bin(em, u32::wrapping_shl, BinaryOperator::ShiftLeft),
        Op::I32ShrU => i32_bin(em, u32::wrapping_shr, BinaryOperator::ShiftRight),
        Op::I32ShrS => i32_bin(em, shr_s, BinaryOperator::ShiftRight),
        Op::I32And => i32_bin(em, |a, b| a & b, BinaryOperator::And),
        Op::I32Or => i32_bin(em, |a, b| a | b, BinaryOperator::InclusiveOr),
        Op::I32Xor => i32_bin(em, |a, b| a ^ b, BinaryOperator::ExclusiveOr),
        Op::I32Eqz => i32_eqz(em),
        Op::I32Eq => i32_cmp(em, |a, b| a == b, BinaryOperator::Equal),
        Op::I32Ne => i32_cmp(em, |a, b| a != b, BinaryOperator::NotEqual),
        Op::I32LtU => i32_cmp(em, |a, b| a < b, BinaryOperator::Less),
        Op::I32GtU => i32_cmp(em, |a, b| a > b, BinaryOperator::Greater),
        Op::I32GeU => i32_cmp(em, |a, b| a >= b, BinaryOperator::GreaterEqual),
        Op::I32LeU => i32_cmp(em, |a, b| a <= b, BinaryOperator::LessEqual),
        Op::F32Lt => f32_cmp(em, BinaryOperator::Less),
        Op::F32Le => f32_cmp(em, BinaryOperator::LessEqual),
        Op::F32Gt => f32_cmp(em, BinaryOperator::Greater),
        Op::F32Ge => f32_cmp(em, BinaryOperator::GreaterEqual),
        Op::F32Eq => f32_cmp(em, BinaryOperator::Equal),
        Op::F32Ne => f32_cmp(em, BinaryOperator::NotEqual),
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
        Imp::TextureLoad => texture::texture_load(em),
        Imp::TextureWidth => texture_dimension(em, 0),
        Imp::TextureHeight => texture_dimension(em, 1),
        Imp::MathDot3 => math::math_dot3(em),
        Imp::MathCross3 => math::math_vec3(em, naga::MathFunction::Cross, true),
        Imp::MathLength3 => math::math_length3(em),
        Imp::MathNormalize3 => math::math_vec3(em, naga::MathFunction::Normalize, false),
        Imp::StorageAtomicAddU32 => compute::storage_atomic_add(em),
        Imp::WorkgroupLoadU32 => compute::workgroup_load(em),
        Imp::WorkgroupStoreU32 => compute::workgroup_store(em),
        Imp::WorkgroupAtomicAddU32 => compute::workgroup_atomic_add(em),
        Imp::WorkgroupBarrier => compute::workgroup_barrier(em),
        Imp::StorageBarrier => compute::storage_barrier(em),
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

fn texture_dimension(em: &mut Emit<'_, '_>, axis: u32) -> Result<(), TranslateError> {
    let binding = em.pop_imm()?;
    let group = em.pop_imm()?;
    let texture = globals::texture(em.cx, group, binding);
    let image = em.ex(Expression::GlobalVariable(texture));
    let size = em.ex(Expression::ImageQuery {
        image,
        query: naga::ImageQuery::Size { level: None },
    });
    let lane = em.ex(Expression::AccessIndex {
        base: size,
        index: axis,
    });
    em.stack.push(Val::Index(lane));
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
    if let Some(value) = em.wasm_locals.pending.get(slot).copied().flatten() {
        // The value is still an expression, so reading it costs nothing.
        return em.push(value);
    }
    if let Some(value) = local_ptr(em, slot)? {
        return em.push(value);
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
    if let Some(slot) = em.ptr_slots.get_mut(index as usize) {
        *slot = provenance(&val);
    }
    if tee {
        em.stack.push(val);
    }
    let slot = index as usize;
    if em.wasm_locals.expr.get(slot).copied().unwrap_or(false) && !matches!(val, Val::Ptr { .. }) {
        // The slot is written once and read after that write, so its value
        // stays an expression: no variable, no store, no load.
        em.wasm_locals.set_pending(slot, val);
        return Ok(());
    }
    let var = em
        .wasm_locals
        .shadow_for(&mut *em.func, &*em.cx, &mut em.locals, index as usize)?;
    // The value now lives in the variable, not in an expression.
    em.wasm_locals.clear_pending(slot);
    let bits = match val {
        Val::Imm(value) => em.u32(value),
        Val::Index(handle) | Val::Scalar(handle) => handle,
        Val::Bool(handle) => crate::cx::bool_bits(em.func, &mut em.block, handle),
        Val::Ptr { off, .. } => em.off_expr(off),
        Val::Vec(_) => return Err(TranslateError::Unsupported),
    };
    em.store_var(var, bits);
    Ok(())
}

/// Immediate bits, or the two operands already lowered to `u32`.
enum Ready {
    Imm(u32),
    Expr(naga::Handle<Expression>, naga::Handle<Expression>),
}
