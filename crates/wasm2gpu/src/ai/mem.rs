//! Provenance arithmetic: storage bases, pointer offsets, and the stack
//! effects of the scalar load and store operators.

use super::Abs;
use super::Num;
use super::St;
use super::pop;
use super::push;
use super::type_msg;
use super::value::pop_f32;

use crate::error::TranslateError;
use crate::ir::Import;
use crate::ty;
use crate::ty::Imp;

/// Fold `storage_base`'s provenance pair into an address.
pub(super) fn storage_base(st: &mut St, import: &Import) -> Result<(), TranslateError> {
    ty::sig_ok(&import.sig, Imp::Base, &[], None)?;
    let binding = pop_imm(st)?;
    let group = pop_imm(st)?;
    let address = Abs::Ptr {
        group,
        binding,
        offset: Num::Const(0),
    };
    push(st, address)
}

pub(super) fn pop_sid(st: &mut St) -> Result<u16, TranslateError> {
    match st.stack.pop() {
        Some(Abs::I32(Num::Const(value))) => {
            u16::try_from(value).map_err(|_| type_msg("struct type id is out of range"))
        }
        Some(Abs::I32(Num::Dyn { joined: true })) => Err(TranslateError::AmbiguousProvenance),
        Some(_) => Err(type_msg("struct type id is not a constant")),
        None => Err(TranslateError::Unsupported),
    }
}

pub(super) fn pop_prov(st: &mut St) -> Result<(), TranslateError> {
    pop_imm(st)?;
    Ok(())
}

/// Pop a group or binding operand: the value must be an `i32.const`.
pub(super) fn pop_imm(st: &mut St) -> Result<u32, TranslateError> {
    match st.stack.pop() {
        Some(Abs::I32(Num::Const(value))) => Ok(value),
        Some(Abs::I32(Num::Dyn { joined: true })) => Err(TranslateError::AmbiguousProvenance),
        Some(Abs::I32(Num::Dyn { joined: false })) => Err(TranslateError::DynamicBinding),
        Some(_) => Err(TranslateError::DynamicBinding),
        None => Err(TranslateError::Unsupported),
    }
}

/// Fold two abstract integers, keeping `joined` when either side is a phi.
pub(super) fn num_fold(left: Num, right: Num, op: impl Fn(u32, u32) -> u32) -> Num {
    match (left, right) {
        (Num::Const(a), Num::Const(b)) => Num::Const(op(a, b)),
        _ => Num::Dyn {
            joined: num_joined(left) || num_joined(right),
        },
    }
}

fn num_joined(num: Num) -> bool {
    match num {
        Num::Const(_) => false,
        Num::Dyn { joined } => joined,
    }
}

/// `i32.add` / `i32.sub`, keeping a storage pointer's provenance.
///
/// A pointer plus or minus an integer stays a pointer with the folded
/// offset; a pointer on both sides is meaningless and rejected, and any
/// other pairing is plain integer arithmetic, where a pointer operand
/// fails the pop.
pub(super) fn i32_offset(
    st: &mut St,
    add: bool,
    op: impl Fn(u32, u32) -> u32,
) -> Result<(), TranslateError> {
    let right = pop(st)?;
    let left = pop(st)?;
    let (group, binding, offset) = match (left, right) {
        (
            Abs::Ptr {
                group,
                binding,
                offset,
            },
            Abs::I32(num),
        ) => {
            let next = num_fold(offset, num, op);
            (group, binding, next)
        }
        (
            Abs::I32(num),
            Abs::Ptr {
                group,
                binding,
                offset,
            },
        ) if add => {
            let next = num_fold(num, offset, op);
            (group, binding, next)
        }
        (Abs::I32(left_num), Abs::I32(right_num)) => {
            return push(st, Abs::I32(num_fold(left_num, right_num, op)));
        }
        _ => return Err(TranslateError::Unsupported),
    };
    push(st, Abs::Ptr {
        group,
        binding,
        offset,
    })
}

/// Pop a load or store address: only a storage pointer names a buffer.
fn ptr_operand(st: &mut St) -> Result<(), TranslateError> {
    match pop(st)? {
        Abs::Ptr { .. } => Ok(()),
        Abs::I32(_) => Err(TranslateError::LinearMemory),
        _ => Err(TranslateError::Unsupported),
    }
}

fn pop_i32(st: &mut St) -> Result<(), TranslateError> {
    match pop(st)? {
        Abs::I32(_) => Ok(()),
        _ => Err(TranslateError::Unsupported),
    }
}

pub(super) fn f32_lt(st: &mut St) -> Result<(), TranslateError> {
    pop_f32(st)?;
    pop_f32(st)?;
    push(st, Abs::I32(Num::Dyn { joined: false }))
}

/// `i32.load` analysis: the address is already checked provenance.
pub(super) fn load_i32(st: &mut St) -> Result<(), TranslateError> {
    ptr_operand(st)?;
    push(st, Abs::I32(Num::Dyn { joined: false }))
}

/// `f32.load` analysis: the address is already checked provenance.
pub(super) fn load_f32(st: &mut St) -> Result<(), TranslateError> {
    ptr_operand(st)?;
    push(st, Abs::F32)
}

/// `i32.store` analysis: an integer value leaves a checked address.
pub(super) fn store_i32(st: &mut St) -> Result<(), TranslateError> {
    pop_i32(st)?;
    ptr_operand(st)
}

/// `f32.store` analysis: a float value leaves a checked address.
pub(super) fn store_f32(st: &mut St) -> Result<(), TranslateError> {
    pop_f32(st)?;
    ptr_operand(st)
}
