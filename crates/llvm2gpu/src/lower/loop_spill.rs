//! Loop exit dispatch and the spilling of values live across a loop.

use super::EXIT;
use super::cfg::{Ctx, Flow, def_ty, push_local, push_term_local};
use super::lowerer::Lowerer;
use super::memory::local_label;
use crate::error::TranslateError;
use std::collections::HashSet;

impl<'a, 'b> Lowerer<'a, 'b> {
    /// Continue at `exit` after its `Loop`, in the enclosing region.
    pub(crate) fn resume_after_loop(
        &mut self,
        exit: &'a str,
        stop: Option<&'a str>,
        out: &mut naga::Block,
        outer: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
        if stop == Some(exit) {
            return Ok(Flow::Next);
        }
        if let Ctx::Loop {
            header: outer_header,
        } = outer
        {
            if exit == outer_header {
                out.push(naga::Statement::Continue, naga::Span::default());
                return Ok(Flow::Continue);
            }
            if self.leaves_outer(exit, outer) {
                // `break 'outer`: this `break` leaves the enclosing loop too.
                out.push(naga::Statement::Break, naga::Span::default());
                return Ok(Flow::Break);
            }
        }
        self.emit_region(exit, stop, out, outer)
    }

    /// Resume at whichever exit the selector names, after the loop unwound.
    ///
    /// Every exit runs until the loop's merge, so the code after the loop is
    /// emitted once, after the dispatch. An exit that leaves the enclosing loop
    /// breaks it instead, and the enclosing loop's own dispatch resumes there.
    pub(crate) fn dispatch_exits(
        &mut self,
        header: &'a str,
        exits: &[&'a str],
        stop: Option<&'a str>,
        out: &mut naga::Block,
        outer: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
        let merge = self.cfg.merge(header).unwrap_or(EXIT);
        let selector = self.exit_id_var.ok_or_else(TranslateError::unsupported)?;
        let mut returned = 0usize;
        for exit in exits {
            let id = self.exit_id(exit);
            let pointer = self.expr(out, naga::Expression::LocalVariable(selector));
            let value = self.expr(out, naga::Expression::Load { pointer });
            let expected = self.u32_literal(out, id);
            let condition = self.binary(out, naga::BinaryOperator::Equal, value, expected);
            let mut accept = naga::Block::new();
            if self.leaves_outer(exit, outer) {
                accept.push(naga::Statement::Break, naga::Span::default());
            } else {
                let flow = self.emit_region(exit, Some(merge), &mut accept, outer)?;
                // `Continue`/`Break` leave the enclosing loop from inside the
                // case, which is a terminal flow for this dispatch too.
                if flow == Flow::Returned {
                    returned += 1;
                }
            }
            out.push(
                naga::Statement::If {
                    condition,
                    accept,
                    reject: naga::Block::new(),
                },
                naga::Span::default(),
            );
        }
        if merge == EXIT {
            if returned == exits.len() {
                return Ok(Flow::Returned);
            }
            return Err(TranslateError::unsupported());
        }
        if self.leaves_outer(merge, outer) {
            // The merge is past the enclosing loop: break it and let its own
            // dispatch resume there.
            out.push(naga::Statement::Break, naga::Span::default());
            return Ok(Flow::Break);
        }
        if stop == Some(merge) {
            return Ok(Flow::Next);
        }
        self.emit_region(merge, stop, out, outer)
    }

    /// Values the exits read that are defined inside the loop.
    pub(crate) fn loop_live_outs(
        &mut self,
        body: &HashSet<&'a str>,
    ) -> Result<Vec<String>, TranslateError> {
        let mut used = Vec::new();
        // Anything defined in the loop and read after it has to be spilled:
        // the exit block is not the only reader. Later blocks and their phis
        // run once the loop's expressions are out of scope.
        for label in &self.cfg.order {
            if body.contains(label) {
                continue;
            }
            let block = self.cfg.blocks[label];
            for instr in &block.instrs {
                instr.each_operand(|op| push_local(op, &mut used));
            }
            push_term_local(&block.term, &mut used);
        }
        let mut live = Vec::new();
        for name in used {
            self.spill_live(body, &name, &mut live)?;
        }
        Ok(live)
    }

    pub(crate) fn spill_live(
        &mut self,
        body: &HashSet<&str>,
        name: &str,
        live: &mut Vec<String>,
    ) -> Result<(), TranslateError> {
        if live.iter().any(|found| found == name) {
            return Ok(());
        }
        if self.env.contains_key(name) {
            return Ok(());
        }
        if self.spill.contains_key(name) {
            live.push(name.to_string());
            return Ok(());
        }
        let Some(ty) = def_ty(self.cfg, body, name) else {
            return Ok(());
        };
        let ty = self.ty(ty)?;
        let var = self.func.local_variables.append(
            naga::LocalVariable {
                name: Some(format!("out.{}", local_label(name))),
                ty,
                init: None,
            },
            naga::Span::default(),
        );
        self.spill.insert(name.to_string(), var);
        live.push(name.to_string());
        Ok(())
    }

    pub(crate) fn reload_live(
        &mut self,
        names: &[String],
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        for name in names {
            let var = *self
                .spill
                .get(name)
                .ok_or_else(TranslateError::unsupported)?;
            let pointer = self.expr(out, naga::Expression::LocalVariable(var));
            let value = self.expr(out, naga::Expression::Load { pointer });
            self.env.insert(name.clone(), value);
        }
        Ok(())
    }
}
