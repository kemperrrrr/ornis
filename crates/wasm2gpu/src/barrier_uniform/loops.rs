//! Fixed point of loop-carried locals, then one rewalk if the loop diverges.
//!
//! The lattice has three tags, so each local moves at most twice. Once a loop
//! exit depends on the invocation id, one more round treats every definition
//! in the body as invocation-dependent, including a counter updated before
//! the exit test. A barrier after the loop is uniform once every exit has
//! reached the loop's own merge; an exit to an outer label keeps the split
//! until that label.

use super::Lab;
use super::LabKind;
use super::Split;
use super::St;
use super::Uni;
use super::exit;
use super::tag;
use crate::error::BarrierSite;
use crate::error::TranslateError;
use crate::ir::Stmt;

impl Uni<'_> {
    pub(super) fn loop_region(&mut self, body: &[Stmt]) -> Result<(), TranslateError> {
        let parent = self.st.split;
        let entry = self.st.clone();
        let index = self.labels.len();
        let pos = self.pos;
        let pending_base = self.pending.len();
        self.labels.push(Lab::new(LabKind::Loop, parent.is_none()));
        let (header, varies, exit_at) = self.fix_loop(&LoopRound {
            body,
            entry: &entry,
            index,
            pos,
            pending_base,
            parent,
        })?;
        self.labels.pop();
        if varies && parent.is_none() {
            self.rewalk_divergent_loop(&entry, &header, pos, pending_base, body)?;
        }
        self.finish_loop(parent, index, varies, exit_at);
        Ok(())
    }

    fn fix_loop(
        &mut self,
        round: &LoopRound<'_>,
    ) -> Result<(Vec<tag::Tag>, bool, Option<usize>), TranslateError> {
        let mut header = round.entry.locals.clone();
        let mut rounds = 0u32;
        let mut poison = false;
        let limit = (header.len() as u32).saturating_mul(2).saturating_add(1);
        loop {
            let seen = poison;
            self.prepare_round(round, &header);
            self.st.taint = poison || round.entry.taint;
            self.walk(round.body)?;
            let carried = self
                .labels
                .last()
                .map(|lab| lab.carry.clone())
                .unwrap_or_default();
            self.note_fallthrough(round.index);
            let next = self
                .absorb_loop_pending(round.index, carried_header(round.entry, &header, &carried));
            let stable = next == header;
            header = next;
            if self.labels.last().is_some_and(|lab| lab.varies) {
                poison = true;
            }
            if (stable && poison == seen) || rounds >= limit {
                break;
            }
            rounds += 1;
        }
        self.st.taint = round.entry.taint;
        let varies = self.labels.last().is_some_and(|lab| lab.varies);
        let exit_at = self.labels.last().and_then(|lab| lab.exit_at);
        Ok((header, varies, exit_at))
    }

    fn note_fallthrough(&mut self, index: usize) {
        if self.st.leave != super::Leave::Fall {
            return;
        }
        let Some(lab) = self.labels.last_mut() else {
            return;
        };
        lab.exit_at = Some(lab.exit_at.map(|at| at.min(index)).unwrap_or(index));
    }

    fn prepare_round(&mut self, round: &LoopRound<'_>, header: &[tag::Tag]) {
        self.pos = round.pos;
        self.pending.truncate(round.pending_base);
        self.st = round.entry.clone();
        self.st.locals = header.to_vec();
        self.st.leave = super::Leave::Fall;
        self.st.split = round.parent;
        if let Some(lab) = self.labels.last_mut() {
            lab.varies = false;
            lab.carry.clear();
            lab.exit_at = None;
            lab.clear = round.parent.is_none();
        }
    }

    fn absorb_loop_pending(&mut self, index: usize, mut next: Vec<tag::Tag>) -> Vec<tag::Tag> {
        if matches!(self.st.leave, super::Leave::Br { target } if target == index) {
            next = tag::widen_vec(&next, &self.st.locals);
        }
        let pending = std::mem::take(&mut self.pending);
        let mut rest = Vec::new();
        for alt in pending {
            if matches!(alt.leave, super::Leave::Br { target } if target == index) {
                next = tag::widen_vec(&next, &alt.locals);
            } else {
                rest.push(alt);
            }
        }
        self.pending = rest;
        next
    }

    /// Second walk once the header is known to be divergent.
    ///
    /// The fixed point has to see the continue before it knows that, so a
    /// barrier that textually precedes the branch is rejected here.
    fn rewalk_divergent_loop(
        &mut self,
        entry: &St,
        header: &[tag::Tag],
        pos: u32,
        pending_base: usize,
        body: &[Stmt],
    ) -> Result<(), TranslateError> {
        let saved = self.st.clone();
        let saved_pending = self.pending.clone();
        self.pos = pos;
        self.pending.truncate(pending_base);
        self.st = entry.clone();
        self.st.locals = header.to_vec();
        self.st.leave = super::Leave::Fall;
        self.st.split = Some(Split {
            site: BarrierSite::Loop,
            at: None,
        });
        self.labels.push(Lab::new(LabKind::Loop, false));
        let walked = self.walk(body);
        self.labels.pop();
        self.st = saved;
        self.pending = saved_pending;
        walked
    }

    fn finish_loop(
        &mut self,
        parent: Option<Split>,
        index: usize,
        varies: bool,
        exit_at: Option<usize>,
    ) {
        if !varies {
            exit::reconverge_at(&mut self.st, parent, index);
            return;
        }
        let at = exit_at.unwrap_or(index);
        self.st.leave = super::Leave::Fall;
        if at == index {
            self.st.split = parent;
            return;
        }
        self.st.split = Some(Split {
            site: parent.map(|item| item.site).unwrap_or(BarrierSite::Loop),
            at: Some(at),
        });
    }
}

struct LoopRound<'a> {
    body: &'a [Stmt],
    entry: &'a St,
    index: usize,
    pos: u32,
    pending_base: usize,
    parent: Option<Split>,
}

fn carried_header(entry: &St, header: &[tag::Tag], carried: &[tag::Tag]) -> Vec<tag::Tag> {
    if carried.is_empty() {
        header.to_vec()
    } else {
        tag::widen_vec(&entry.locals, carried)
    }
}
