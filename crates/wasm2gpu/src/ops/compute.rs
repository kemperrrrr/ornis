//! GPU synchronization and atomic operation lowering.

use crate::cx::Val;
use crate::env::Emit;
use crate::error::TranslateError;
use crate::globals;
use naga::Expression;

pub(super) fn storage_atomic_add(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let value = em.pop_uint()?;
    let index = em.pop_uint()?;
    let binding = em.pop_imm()?;
    let group = em.pop_imm()?;
    let var = globals::atomic_storage(em.cx, group, binding)?;
    let base = em.ex(Expression::GlobalVariable(var));
    let pointer = em.ex(Expression::Access { base, index });
    let ty = em.cx.uint;
    let result = em.func.expressions.append(
        Expression::AtomicResult {
            ty,
            comparison: false,
        },
        crate::cx::sp(),
    );
    em.block.push(
        naga::Statement::Atomic {
            pointer,
            fun: naga::AtomicFunction::Add,
            value,
            result: Some(result),
        },
        crate::cx::sp(),
    );
    em.stack.push(Val::Index(result));
    Ok(())
}

pub(super) fn workgroup_load(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let index = em.pop_uint()?;
    let var = globals::workgroup(em.cx);
    let base = em.ex(Expression::GlobalVariable(var));
    let pointer = em.ex(Expression::Access { base, index });
    let value = em.ex(Expression::Load { pointer });
    em.stack.push(Val::Index(value));
    Ok(())
}

pub(super) fn workgroup_store(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let value = em.pop_uint()?;
    let index = em.pop_uint()?;
    let var = globals::workgroup(em.cx);
    let base = em.ex(Expression::GlobalVariable(var));
    let pointer = em.ex(Expression::Access { base, index });
    em.block
        .push(naga::Statement::Store { pointer, value }, crate::cx::sp());
    Ok(())
}

// qual:allow(dry, fragment) reason: "The two atomic operations intentionally share Naga pointer setup while selecting distinct address spaces and globals."
pub(super) fn workgroup_atomic_add(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let value = em.pop_uint()?;
    let index = em.pop_uint()?;
    let var = globals::workgroup_atomics(em.cx);
    let base = em.ex(Expression::GlobalVariable(var));
    let pointer = em.ex(Expression::Access { base, index });
    let ty = em.cx.uint;
    let result = em.func.expressions.append(
        Expression::AtomicResult {
            ty,
            comparison: false,
        },
        crate::cx::sp(),
    );
    em.block.push(
        naga::Statement::Atomic {
            pointer,
            fun: naga::AtomicFunction::Add,
            value,
            result: Some(result),
        },
        crate::cx::sp(),
    );
    em.stack.push(Val::Index(result));
    Ok(())
}

pub(super) fn workgroup_barrier(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    em.block.push(
        naga::Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
        crate::cx::sp(),
    );
    Ok(())
}

pub(super) fn storage_barrier(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    em.block.push(
        naga::Statement::MemoryBarrier(naga::Barrier::STORAGE),
        crate::cx::sp(),
    );
    Ok(())
}
