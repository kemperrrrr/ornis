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
        self.reconverge(parent, index);
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
        self.st = merge::merge_if(cond, parent, index, then_end, else_end);
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

    fn loop_region(&mut self, body: &[Stmt]) -> Result<(), TranslateError> {
        let parent = self.st.split;
        let saved = self.st.clone();
        let index = self.labels.len();
        self.labels.push(Lab::new(LabKind::Loop, parent.is_none()));
        self.walk(body)?;
        let varies = self.labels.last().is_some_and(|lab| lab.varies);
        let after = self.st.clone();
        self.labels.pop();
        if varies && parent.is_none() {
            self.rewalk_divergent_loop(&saved, body)?;
        }
        self.st = after;
        if varies {
            self.st.leave = Leave::Fall;
            self.st.split = Some(Split {
                site: parent.map(|item| item.site).unwrap_or(BarrierSite::Loop),
                at: None,
            });
        } else {
            self.reconverge(parent, index);
        }
        Ok(())
    }

    /// Second walk of a loop whose trip count depends on the invocation id.
    ///
    /// The first walk has to see the continue or the conflicting exit before
    /// it knows the header was divergent, so a barrier that textually
    /// precedes that branch is rejected here.
    fn rewalk_divergent_loop(&mut self, saved: &St, body: &[Stmt]) -> Result<(), TranslateError> {
        self.st = saved.clone();
        self.st.split = Some(Split {
            site: BarrierSite::Loop,
            at: None,
        });
        self.st.leave = Leave::Fall;
        self.labels.push(Lab::new(LabKind::Loop, false));
        let walked = self.walk(body);
        self.labels.pop();
        walked
    }

    fn br(&mut self, depth: u32) -> Result<(), TranslateError> {
        let target = self.target(depth)?;
        self.st.leave = Leave::Br { target };
        Ok(())
    }

    fn br_if(&mut self, depth: u32) -> Result<(), TranslateError> {
        let cond = self.pop();
        if cond != Tag::Inv {
            return Ok(());
        }
        let target = self.target(depth)?;
        self.note_continue(target);
        self.st.split = Some(Split {
            site: self
                .st
                .split
                .map(|item| item.site)
                .unwrap_or(BarrierSite::Branch),
            at: self.keep_open_split(target),
        });
        Ok(())
    }

    fn br_table(&mut self, targets: &[u32], default: u32) -> Result<(), TranslateError> {
        let index = self.pop();
        if index == Tag::Inv {
            self.note_table(targets, default)?;
        }
        self.st.leave = Leave::Gone;
        Ok(())
    }

    fn note_table(&mut self, targets: &[u32], default: u32) -> Result<(), TranslateError> {
        let mut index = 0usize;
        while index < targets.len() {
            let target = self.target(targets[index])?;
            self.note_continue(target);
            self.note_exit(target);
            index += 1;
        }
        let target = self.target(default)?;
        self.note_continue(target);
        self.note_exit(target);
        Ok(())
    }

    fn note_continue(&mut self, target: usize) {
        let Some(lab) = self.labels.get_mut(target) else {
            return;
        };
        if lab.kind == LabKind::Loop && lab.clear {
            lab.varies = true;
        }
    }

    /// A jump to `target` leaves every loop nested strictly inside it.
    fn note_exit(&mut self, target: usize) {
        let mut index = target + 1;
        while index < self.labels.len() {
            if self.labels[index].kind == LabKind::Loop && self.labels[index].clear {
                self.labels[index].varies = true;
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

    fn reconverge(&mut self, parent: Option<Split>, region: usize) {
        if matches!(self.st.leave, Leave::Br { target } if target == region) {
            self.st.leave = Leave::Fall;
        }
        if self.st.leave != Leave::Fall {
            return;
        }
        if matches!(self.st.split, Some(Split { at: Some(at), .. }) if at == region) {
            self.st.split = parent;
        }
    }

    /// Keep a split that already lasts until the function returns.
    fn keep_open_split(&self, target: usize) -> Option<usize> {
        match self.st.split {
            Some(split) if split.at.is_none() => None,
            _ => Some(target),
        }
    }

    fn target(&self, depth: u32) -> Result<usize, TranslateError> {
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
