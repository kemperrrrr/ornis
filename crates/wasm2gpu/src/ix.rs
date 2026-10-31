//! One element index of a storage or uniform access.
//!
//! The index is a constant, or a runtime `u32` that becomes an access chain.
//! A `0`/`1` boolean is a wasm `i32`, so it addresses an element like any
//! other index.

use naga::Expression;

use crate::cx::Val;
use crate::env::Emit;
use crate::error::TranslateError;

pub(super) enum Ix {
    Const(u32),
    Dyn(naga::Handle<Expression>),
}

pub(super) fn pop_index(em: &mut Emit<'_, '_>) -> Result<Ix, TranslateError> {
    match em.pop()? {
        Val::Imm(value) => Ok(Ix::Const(value)),
        Val::Index(handle) => Ok(Ix::Dyn(handle)),
        Val::Bool(handle) => {
            let bits = crate::cx::bool_bits(em.func, &mut em.block, handle);
            Ok(Ix::Dyn(bits))
        }
        _ => Err(TranslateError::Unsupported),
    }
}
