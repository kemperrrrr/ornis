//! Calls into helper functions lowered alongside the entry point.

use super::lowerer::Lowerer;
use super::memory::Helper;
use crate::error::TranslateError;
use crate::parse::{Operand, Ty};

impl<'a, 'b> Lowerer<'a, 'b> {
    /// Call a helper function and return its result expression.
    pub(crate) fn call_helper(
        &mut self,
        out: &mut naga::Block,
        helper: Helper,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        // The result expression is defined by the `Call` statement and must not
        // appear in an `Emit` range, so it is appended outside `Self::expr`.
        let result = self.func.expressions.append(
            naga::Expression::CallResult(helper.function),
            naga::Span::default(),
        );
        let arguments = self.helper_arguments(out, args)?;
        out.push(
            naga::Statement::Call {
                function: helper.function,
                arguments,
                result: Some(result),
            },
            naga::Span::default(),
        );
        if helper.ret == Ty::Void {
            return Err(TranslateError::unsupported());
        }
        Ok(result)
    }

    /// Call a helper that returns nothing.
    pub(crate) fn call_helper_void(
        &mut self,
        out: &mut naga::Block,
        helper: Helper,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        let arguments = self.helper_arguments(out, args)?;
        out.push(
            naga::Statement::Call {
                function: helper.function,
                arguments,
                result: None,
            },
            naga::Span::default(),
        );
        Ok(())
    }

    /// Operands of a helper call, in signature order.
    pub(crate) fn helper_arguments(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<Vec<naga::Handle<naga::Expression>>, TranslateError> {
        let mut arguments = Vec::with_capacity(args.len());
        for arg in args {
            arguments.push(self.operand(out, arg)?);
        }
        Ok(arguments)
    }
}
