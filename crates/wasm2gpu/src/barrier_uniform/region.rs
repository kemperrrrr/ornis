//! Walk structured control flow and record where invocations diverge.
//!
//! A branch or a loop is marked only when the invocation id decides which
//! invocations stay inside it. A barrier later in that region reads the mark.

use super::Lab;
use super::LabKind;
use super::Leave;
use super::Split;
use super::St;
use super::Uni;
use super::merge;
use super::tag::Tag;
use super::tag::apply;
use crate::error::BarrierSite;
use crate::error::TranslateError;
use crate::ir::Op;
use crate::ir::Stmt;

impl Uni<'_> {
    pub(super) fn walk(&mut self, stmts: &[Stmt]) -> Result<(), TranslateError> {
        for stmt in stmts {
            if self.st.leave != Leave::Fall {
                break;
            }
            self.one(stmt)?;
        }
        Ok(())
    }

    fn one(&mut self, stmt: &Stmt) -> Result<(), TranslateError> {
        self.at = self.pos;
        self.pos = self.pos.saturating_add(1);
        match stmt {
            Stmt::Op(Op::Call(index)) => self.on_call(*index),
            Stmt::Op(op) => {
                apply(&mut self.st, *op);
                Ok(())
            }
            Stmt::Block { body, .. } => self.block_region(body),
            Stmt::Loop { body, .. } => self.loop_region(body),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => self.if_region(then_body, else_body),
            Stmt::Br(depth) => self.br(*depth),
            Stmt::BrIf(depth) => self.br_if(*depth),
            Stmt::BrTable { targets, default } => self.br_table(targets, *default),
            Stmt::Return => {
                self.st.leave = Leave::Return;
                Ok(())
            }
        }
    }

    fn block_region(&mut self, body: &[Stmt]) -> Result<(), TranslateError> {
        let parent = self.st.split;
        let index = self.labels.len();
        self.labels.push(Lab::new(LabKind::Block, parent.is_none()));
        self.walk(body)?;
        self.labels.pop();
        self.finish_block(parent, index);
        Ok(())
    }

    fn if_region(&mut self, then_body: &[Stmt], else_body: &[Stmt]) -> Result<(), TranslateError> {
        let cond = self.pop();
        let parent = self.st.split;
        let index = self.labels.len();
        self.labels.push(Lab::new(LabKind::If, parent.is_none()));
        let arm = arm_split(parent, cond, index);
        let then_end = self.arm(then_body, arm)?;
        let else_end = self.arm(else_body, arm)?;
        if cond == Tag::Inv {
            self.mark_disagreement(then_end.leave, else_end.leave);
        }
        self.labels.pop();
        let (cont, parked) = merge::place_if(merge::IfEnds {
            cond,
            parent,
            index,
            then_end,
            else_end,
        });
        if let Some(parked) = parked {
            self.pending.push(parked);
        }
        self.st = cont;
        Ok(())
    }

    fn arm(&mut self, body: &[Stmt], split: Option<Split>) -> Result<St, TranslateError> {
        let saved = self.st.clone();
        self.st.split = split;
        self.st.leave = Leave::Fall;
        self.walk(body)?;
        let end = self.st.clone();
        self.st = saved;
        Ok(end)
    }

    fn br(&mut self, depth: u32) -> Result<(), TranslateError> {
        let target = self.target(depth)?;
        self.note_back(target);
        self.st.leave = Leave::Br { target };
        Ok(())
    }

    fn br_if(&mut self, depth: u32) -> Result<(), TranslateError> {
        let cond = self.pop();
        let target = self.target(depth)?;
        self.note_back(target);
        if cond != Tag::Inv {
            return Ok(());
        }
        self.note_continue(target);
        self.note_exit(target);
        self.st.split = Some(Split {
            site: self
                .st
                .split
                .map(|item| item.site)
                .unwrap_or(BarrierSite::Branch),
            at: self.keep_open_split(target),
        });
        let mut taken = self.st.clone();
        taken.leave = Leave::Br { target };
        self.pending.push(taken);
        Ok(())
    }

    pub(super) fn note_back(&mut self, target: usize) {
        let Some(lab) = self.labels.get_mut(target) else {
            return;
        };
        if lab.kind != LabKind::Loop {
            return;
        }
        if lab.carry.is_empty() {
            lab.carry = self.st.locals.clone();
            return;
        }
        lab.carry = super::tag::widen_vec(&lab.carry, &self.st.locals);
    }

    pub(super) fn note_continue(&mut self, target: usize) {
        let Some(lab) = self.labels.get_mut(target) else {
            return;
        };
        if lab.kind == LabKind::Loop && lab.clear {
            lab.varies = true;
        }
    }

    /// A jump to `target` leaves every loop nested strictly inside it.
    pub(super) fn note_exit(&mut self, target: usize) {
        let mut index = target + 1;
        while index < self.labels.len() {
            if self.labels[index].kind == LabKind::Loop && self.labels[index].clear {
                self.labels[index].varies = true;
                let exit_at = self.labels[index].exit_at;
                self.labels[index].exit_at =
                    Some(exit_at.map(|at| at.min(target)).unwrap_or(target));
            }
            index += 1;
        }
    }

    fn mark_disagreement(&mut self, then_leave: Leave, else_leave: Leave) {
        let mut index = 0usize;
        while index < self.labels.len() {
            if self.labels[index].kind == LabKind::Loop && self.labels[index].clear {
                let then_exits = merge::exits_loop(then_leave, index);
                let else_exits = merge::exits_loop(else_leave, index);
                if then_exits != else_exits {
                    self.labels[index].varies = true;
                }
            }
            index += 1;
        }
    }

    /// Keep a split that already lasts until the function returns.
    pub(super) fn keep_open_split(&self, target: usize) -> Option<usize> {
        match self.st.split {
            Some(split) if split.at.is_none() => None,
            _ => Some(target),
        }
    }

    pub(super) fn target(&self, depth: u32) -> Result<usize, TranslateError> {
        let n = self.labels.len();
        let depth = depth as usize;
        if depth >= n {
            return Err(TranslateError::FunctionBranch);
        }
        Ok(n - 1 - depth)
    }

    pub(super) fn pop(&mut self) -> Tag {
        self.st.stack.pop().unwrap_or(Tag::Unknown)
    }
}

fn arm_split(parent: Option<Split>, cond: Tag, if_index: usize) -> Option<Split> {
    if cond != Tag::Inv {
        return parent;
    }
    Some(Split {
        site: parent.map(|item| item.site).unwrap_or(BarrierSite::Branch),
        at: match parent {
            Some(item) if item.at.is_none() => None,
            _ => Some(if_index),
        },
    })
}
