//! Branch emission: exits, conditionals and switches.

use super::EXIT;
use super::cfg::{Ctx, Env, Flow};
use super::emit::scalar;
use super::flow::{case_flow_returns, switch_case};
use super::lowerer::Lowerer;
use crate::error::TranslateError;
use crate::parse::{Operand, Terminator};
use std::collections::HashSet;

impl<'a, 'b> Lowerer<'a, 'b> {
    /// Selector value for `target`, assigned the first time it is needed.
    pub(crate) fn exit_id(&mut self, target: &'a str) -> u32 {
        if let Some(id) = self.exit_ids.get(target) {
            return *id;
        }
        let id = self.exit_ids.len() as u32 + 1;
        self.exit_ids.insert(target, id);
        id
    }

    /// Record that the current `break` leaves the loop towards `target`.
    ///
    /// Loops with a single exit never assign ids, so their WGSL keeps the plain
    /// `break` it always had.
    pub(crate) fn store_exit_id(
        &mut self,
        out: &mut naga::Block,
        target: &'a str,
    ) -> Result<(), TranslateError> {
        let Some(id) = self.exit_ids.get(target).copied() else {
            return Ok(());
        };
        let var = match self.exit_id_var {
            Some(var) => var,
            None => {
                let var = self.func.local_variables.append(
                    naga::LocalVariable {
                        name: Some("exit_id".to_string()),
                        ty: scalar(&mut self.builders.module, naga::ScalarKind::Uint),
                        init: None,
                    },
                    naga::Span::default(),
                );
                self.exit_id_var = Some(var);
                var
            }
        };
        let pointer = self.expr(out, naga::Expression::LocalVariable(var));
        let value = self.u32_literal(out, id);
        out.push(
            naga::Statement::Store { pointer, value },
            naga::Span::default(),
        );
        Ok(())
    }

    /// Mark the control region a condition opens, returning the depth to restore.
    ///
    /// A proven-uniform condition keeps the region uniform, so a barrier inside
    /// it stays valid; anything else makes the enclosing region divergent.
    pub(crate) fn enter_control(&mut self, cond: &Operand) -> u32 {
        let outer = self.divergent_depth;
        if !self.is_uniform(cond) {
            self.divergent_depth += 1;
        }
        outer
    }

    /// Whether `op` has the same value in every invocation of the workgroup.
    pub(crate) fn is_uniform(&self, op: &Operand) -> bool {
        match op {
            Operand::Local(name) => self.uniform.contains(name),
            Operand::Bool(_) | Operand::F32(_) | Operand::U32(_) | Operand::Symbol(_) => true,
        }
    }

    /// Whether a loop runs the same number of iterations for every invocation.
    ///
    /// Conservative: every edge that leaves the loop must come from a
    /// conditional whose condition is uniform, and the loop must not leave
    /// unconditionally. A `break` behind a non-uniform branch, or a plain
    /// `break`, therefore makes the loop divergent — a barrier inside it is
    /// rejected rather than assumed safe.
    pub(crate) fn loop_is_uniform(&self, header: &str, body: &HashSet<&'a str>) -> bool {
        body.iter().all(|label| {
            let block = self.cfg.blocks[label];
            let leaves: Vec<&str> = self
                .cfg
                .succs
                .get(label)
                .into_iter()
                .flatten()
                .filter(|target| !body.contains(*target))
                .copied()
                .collect();
            let exits_all = !leaves.is_empty();
            if !exits_all {
                // Still check a conditional that can reach the header again.
                return match &block.term {
                    Terminator::BrCond { cond, .. } if *label == header => self.is_uniform(cond),
                    _ => true,
                };
            }
            match &block.term {
                Terminator::BrCond { cond, .. } => self.is_uniform(cond),
                _ => false,
            }
        })
    }

    /// Reject a merge that lies outside the enclosing loop.
    ///
    /// A natural loop cannot have a merge outside it that both arms reach: any
    /// block on the way to the back edge belongs to the loop. Rejecting keeps
    /// the structurer from guessing.
    pub(crate) fn check_merge_in_loop(
        &self,
        merge: &str,
        ctx: Ctx<'a>,
    ) -> Result<(), TranslateError> {
        let Ctx::Loop { header } = ctx else {
            return Ok(());
        };
        let merges_in_loop = self
            .cfg
            .loops
            .get(header)
            .is_some_and(|body| body.contains(merge));
        if merge != EXIT && !merges_in_loop {
            return Err(TranslateError::unsupported());
        }
        Ok(())
    }

    /// Emit one arm of a branch: enter it from `from`, run it until `merge`.
    ///
    /// The caller sets `self.env` before calling; the arm's phi stores are
    /// written from `env`, which is the environment at the branch.
    pub(crate) fn emit_arm(
        &mut self,
        from: &'a str,
        enter: &'a str,
        merge: &'a str,
        env: &Env,
        ctx: Ctx<'a>,
    ) -> Result<(naga::Block, Flow), TranslateError> {
        let mut body = naga::Block::new();
        self.env = env.clone();
        self.store_edge_into(from, enter, env, &mut body)?;
        let flow = self.emit_region(enter, Some(merge), &mut body, ctx)?;
        Ok((body, flow))
    }

    /// Diamond `if`: both arms run until the closest common post-dominator.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_if(
        &mut self,
        label: &'a str,
        cond: &Operand,
        true_label: &'a str,
        false_label: &'a str,
        stop: Option<&'a str>,
        out: &mut naga::Block,
        ctx: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
        let merge = self
            .cfg
            .branch_merge(label, true_label, false_label)
            .ok_or_else(TranslateError::unsupported)?;
        self.check_merge_in_loop(merge, ctx)?;
        let condition = self.operand(out, cond)?;
        let outer_env = self.env.clone();
        let outer_depth = self.enter_control(cond);
        let (accept, accept_flow) = self.emit_arm(label, true_label, merge, &outer_env, ctx)?;
        let (reject, reject_flow) = self.emit_arm(label, false_label, merge, &outer_env, ctx)?;
        self.divergent_depth = outer_depth;
        self.env = outer_env;
        if accept_flow == Flow::Returned || reject_flow == Flow::Returned {
            // The merge block only sees the invocations whose arm fell
            // through; a barrier there would not be reached by all lanes.
            self.diverged = true;
        }
        if accept_flow != Flow::Next && accept_flow != Flow::Returned {
            return Err(TranslateError::unsupported());
        }
        if reject_flow != Flow::Next && reject_flow != Flow::Returned {
            return Err(TranslateError::unsupported());
        }
        out.push(
            naga::Statement::If {
                condition,
                accept,
                reject,
            },
            naga::Span::default(),
        );
        if merge == EXIT {
            if accept_flow == Flow::Returned && reject_flow == Flow::Returned {
                return Ok(Flow::Returned);
            }
            return Err(TranslateError::unsupported());
        }
        if stop == Some(merge) {
            return Ok(Flow::Next);
        }
        self.emit_region(merge, stop, out, ctx)
    }

    /// `switch`: every case runs until the closest common post-dominator.
    ///
    /// Cases that share a target label share one body (the first values fall
    /// through into it), which is what Rust emits for `x | y => …` arms.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_switch(
        &mut self,
        label: &'a str,
        selector: &Operand,
        default: &'a str,
        cases: &'a [(u32, String)],
        stop: Option<&'a str>,
        out: &mut naga::Block,
        ctx: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
        let merge = self
            .cfg
            .merge(label)
            .ok_or_else(TranslateError::unsupported)?;
        self.check_merge_in_loop(merge, ctx)?;
        let selector_expr = self.operand(out, selector)?;
        let outer_env = self.env.clone();
        let outer_depth = self.enter_control(selector);
        let mut targets: Vec<(&'a str, Vec<u32>)> = Vec::new();
        for (value, target) in cases {
            let target = target.as_str();
            match targets.iter_mut().find(|(label, _)| *label == target) {
                Some((_, values)) => values.push(*value),
                None => targets.push((target, vec![*value])),
            }
        }
        let mut naga_cases = Vec::new();
        let mut returned = false;
        for (target, values) in targets {
            let (mut body, flow) = self.emit_arm(label, target, merge, &outer_env, ctx)?;
            returned |= case_flow_returns(flow)?;
            for (index, value) in values.iter().enumerate() {
                let last = index + 1 == values.len();
                let filled = if last {
                    std::mem::take(&mut body)
                } else {
                    naga::Block::new()
                };
                naga_cases.push(switch_case(naga::SwitchValue::U32(*value), filled, !last));
            }
        }
        let (default_body, default_flow) = self.emit_arm(label, default, merge, &outer_env, ctx)?;
        returned |= case_flow_returns(default_flow)?;
        naga_cases.push(switch_case(naga::SwitchValue::Default, default_body, false));
        self.divergent_depth = outer_depth;
        self.env = outer_env;
        if returned {
            // The merge only runs on the lanes whose case fell through.
            self.diverged = true;
        }
        out.push(
            naga::Statement::Switch {
                selector: selector_expr,
                cases: naga_cases,
            },
            naga::Span::default(),
        );
        if merge == EXIT {
            return Err(TranslateError::unsupported());
        }
        if stop == Some(merge) {
            return Ok(Flow::Next);
        }
        self.emit_region(merge, stop, out, ctx)
    }
}
