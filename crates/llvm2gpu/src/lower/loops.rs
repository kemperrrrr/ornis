//! Loop emission: exit branches, double exits and live-out spilling.

use super::cfg::{Ctx, Flow, loop_exits};
use super::flow::loop_statement;
use super::lowerer::Lowerer;
use crate::error::TranslateError;
use crate::parse::Operand;

impl<'a, 'b> Lowerer<'a, 'b> {
    /// Loop-header or latch branch where exactly one successor stays inside
    /// the loop and the other leaves it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_loop_exit_branch(
        &mut self,
        header: &'a str,
        start: &'a str,
        cond: &Operand,
        true_label: &'a str,
        false_label: &'a str,
        true_in: bool,
        out: &mut naga::Block,
    ) -> Result<Flow, TranslateError> {
        let (inside, outside) = if true_in {
            (true_label, false_label)
        } else {
            (false_label, true_label)
        };
        let condition = self.operand(out, cond)?;
        let outer_env = self.env.clone();
        let mut inside_block = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(start, inside, &outer_env, &mut inside_block)?;
        let inside_flow = if inside == header {
            // A branch back to the loop header means the loop repeats.
            inside_block.push(naga::Statement::Continue, naga::Span::default());
            Flow::Continue
        } else {
            let ctx = Ctx::Loop { header };
            self.emit_region(inside, None, &mut inside_block, ctx)?
        };
        // `Next` means the arm ended at a nested exit branch inside the loop,
        // which already emitted its own `break`/`continue`.
        if !matches!(
            inside_flow,
            Flow::Next | Flow::Continue | Flow::Break | Flow::Returned
        ) {
            return Err(TranslateError::unsupported());
        }
        let mut outside_block = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(start, outside, &outer_env, &mut outside_block)?;
        self.store_exit_id(&mut outside_block, outside)?;
        outside_block.push(naga::Statement::Break, naga::Span::default());
        self.env = outer_env;
        let (accept, reject) = if true_in {
            (inside_block, outside_block)
        } else {
            (outside_block, inside_block)
        };
        out.push(
            naga::Statement::If {
                condition,
                accept,
                reject,
            },
            naga::Span::default(),
        );
        Ok(Flow::Next)
    }

    /// Both branch targets leave the loop; each arm breaks.
    pub(crate) fn emit_loop_double_exit(
        &mut self,
        start: &'a str,
        cond: &Operand,
        true_label: &'a str,
        false_label: &'a str,
        out: &mut naga::Block,
    ) -> Result<Flow, TranslateError> {
        let condition = self.operand(out, cond)?;
        let outer_env = self.env.clone();
        let mut accept = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(start, true_label, &outer_env, &mut accept)?;
        self.store_exit_id(&mut accept, true_label)?;
        accept.push(naga::Statement::Break, naga::Span::default());
        let mut reject = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(start, false_label, &outer_env, &mut reject)?;
        self.store_exit_id(&mut reject, false_label)?;
        reject.push(naga::Statement::Break, naga::Span::default());
        self.env = outer_env;
        out.push(
            naga::Statement::If {
                condition,
                accept,
                reject,
            },
            naga::Span::default(),
        );
        Ok(Flow::Next)
    }

    /// `naga::Loop` for a natural loop with one header.
    ///
    /// `outer` is the region that contains this loop. The exit block is
    /// emitted there, so a nested loop's `break` does not leave the outer
    /// `loop` and a branch back to the outer header stays a `continue`.
    ///
    /// A loop with one exit keeps the plain `break` shape. Several exits, or an
    /// exit that leaves more than one loop (`break 'outer`), record which block
    /// control has to resume at in a selector local: each `break` unwinds one
    /// `Loop`, and the dispatch after it either resumes the exit block or
    /// breaks the next loop out.
    pub(crate) fn emit_loop(
        &mut self,
        header: &'a str,
        stop: Option<&'a str>,
        out: &mut naga::Block,
        outer: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
        let body = self
            .cfg
            .loops
            .get(header)
            .ok_or_else(TranslateError::unsupported)?
            .clone();
        let mut exits: Vec<&'a str> = loop_exits(self.cfg, &body).into_iter().collect();
        exits.sort_unstable();
        let live = self.loop_live_outs(&body)?;
        // A single exit that stays in the enclosing region needs no selector.
        let single = exits.len() == 1 && !self.leaves_outer(exits[0], outer);
        if !single {
            for exit in &exits {
                self.exit_id(exit);
            }
        }
        let ctx = Ctx::Loop { header };
        let outer_env = self.env.clone();
        let outer_depth = self.divergent_depth;
        if !self.loop_is_uniform(header, &body) {
            self.divergent_depth += 1;
        }
        let mut loop_body = naga::Block::new();
        let flow = self.emit_region(header, None, &mut loop_body, ctx)?;
        self.divergent_depth = outer_depth;
        // Expressions created inside the loop body are out of scope after
        // `Loop`; the exit block reloads whatever its phis need.
        self.env = outer_env;
        if flow == Flow::Continue {
            return Err(TranslateError::unsupported());
        }
        if exits.is_empty() {
            if flow == Flow::Returned {
                out.push(loop_statement(loop_body), naga::Span::default());
                return Ok(Flow::Returned);
            }
            return Err(TranslateError::unsupported());
        }
        out.push(loop_statement(loop_body), naga::Span::default());
        if flow == Flow::Returned {
            return Ok(Flow::Returned);
        }
        self.reload_live(&live, out)?;
        if single {
            return self.resume_after_loop(exits[0], stop, out, outer);
        }
        self.dispatch_exits(header, &exits, stop, out, outer)
    }

    /// Whether `target` is outside the loop that contains this region.
    pub(crate) fn leaves_outer(&self, target: &'a str, outer: Ctx<'a>) -> bool {
        let Ctx::Loop { header } = outer else {
            return false;
        };
        self.cfg
            .loops
            .get(header)
            .is_none_or(|body| !body.contains(target))
    }
}
