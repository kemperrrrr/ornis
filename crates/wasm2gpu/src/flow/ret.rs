//! Lowering for explicit Wasm function returns.

use crate::cx::Val;
use crate::cx::sp;
use crate::env::Emit;
use crate::error::TranslateError;

/// The fallthrough stack and explicit-return status of one lowered function.
pub(crate) struct Output {
    /// Values left by the path that falls through.
    pub(crate) stack: Vec<Val>,
    /// Whether the body emitted an explicit return on any path.
    pub(crate) returned: bool,
}

impl Emit<'_, '_> {
    pub(super) fn fn_return(&mut self) -> Result<(), TranslateError> {
        let value = match self.results {
            0 => {
                self.stack.clear();
                None
            }
            1 => {
                let result = self
                    .stack
                    .last()
                    .copied()
                    .ok_or(TranslateError::Unsupported)?;
                self.stack.clear();
                match result {
                    Val::Scalar(handle) => {
                        self.stack.push(Val::Scalar(handle));
                        Some(handle)
                    }
                    Val::Index(handle) => {
                        self.stack.push(Val::Index(handle));
                        Some(handle)
                    }
                    Val::Imm(_) => return Err(TranslateError::Unsupported),
                    Val::Bool(condition) => {
                        let handle = crate::cx::bool_bits(self.func, &mut self.block, condition);
                        self.stack.push(Val::Index(handle));
                        Some(handle)
                    }
                    Val::Ptr { .. } | Val::Vec(_) => return Err(TranslateError::Unsupported),
                }
            }
            _ => return Err(TranslateError::Unsupported),
        };
        self.block.push(naga::Statement::Return { value }, sp());
        self.halted = true;
        self.returned = true;
        Ok(())
    }
}
