//! Multi-target exits for `br_table`, and the merge at a block.
//!
//! A uniform or unknown index is a branch to every target: the check keeps
//! walking each continuation, so a later proved divergence is not skipped.
//! An invocation-dependent index opens one split until the outermost target.

use super::Leave;
use super::Split;
use super::St;
use super::Uni;
use super::tag;
use crate::error::BarrierSite;
use crate::error::TranslateError;

impl Uni<'_> {
    pub(super) fn br_table(&mut self, targets: &[u32], default: u32) -> Result<(), TranslateError> {
        let index_tag = self.pop();
        let labels = self.table_labels(targets, default)?;
        let mut index = 0usize;
        while index < labels.len() {
            self.note_back(labels[index]);
            index += 1;
        }
        if index_tag == tag::Tag::Inv {
            self.open_table_split(&labels);
        }
        self.fork_table(&labels);
        Ok(())
    }

    fn table_labels(&self, targets: &[u32], default: u32) -> Result<Vec<usize>, TranslateError> {
        let mut labels = Vec::new();
        let mut index = 0usize;
        while index < targets.len() {
            labels.push(self.target(targets[index])?);
            index += 1;
        }
        labels.push(self.target(default)?);
        labels.sort_unstable();
        labels.dedup();
        Ok(labels)
    }

    fn open_table_split(&mut self, labels: &[usize]) {
        let mut index = 0usize;
        while index < labels.len() {
            self.note_continue(labels[index]);
            self.note_exit(labels[index]);
            index += 1;
        }
        let Some(outermost) = labels.first().copied() else {
            return;
        };
        self.st.split = Some(Split {
            site: self
                .st
                .split
                .map(|item| item.site)
                .unwrap_or(BarrierSite::Branch),
            at: self.keep_open_split(outermost),
        });
    }

    fn fork_table(&mut self, labels: &[usize]) {
        let Some(first) = labels.first().copied() else {
            return;
        };
        let mut index = 1usize;
        while index < labels.len() {
            let mut alt = self.st.clone();
            alt.leave = Leave::Br {
                target: labels[index],
            };
            self.pending.push(alt);
            index += 1;
        }
        self.st.leave = Leave::Br { target: first };
    }

    pub(super) fn finish_block(&mut self, parent: Option<Split>, index: usize) {
        let mut outward = Vec::new();
        let mut cont = None;
        take_cont(&mut cont, &mut outward, parent, index, self.st.clone());
        let pending = std::mem::take(&mut self.pending);
        for alt in pending {
            take_cont(&mut cont, &mut outward, parent, index, alt);
        }
        self.pending = outward;
        if let Some(live) = cont {
            self.st = live;
        } else if let Some(out) = self.pending.pop() {
            self.st = out;
        }
    }
}

fn take_cont(
    cont: &mut Option<St>,
    outward: &mut Vec<St>,
    parent: Option<Split>,
    index: usize,
    mut st: St,
) {
    if !continues_here(st.leave, index) {
        outward.push(st);
        return;
    }
    reconverge_at(&mut st, parent, index);
    *cont = Some(match cont.take() {
        Some(prev) => join_cont(prev, st),
        None => st,
    });
}

pub(super) fn reconverge_at(st: &mut St, parent: Option<Split>, region: usize) {
    if matches!(st.leave, Leave::Br { target } if target == region) {
        st.leave = Leave::Fall;
    }
    if st.leave != Leave::Fall {
        return;
    }
    if matches!(st.split, Some(Split { at: Some(at), .. }) if at == region) {
        st.split = parent;
    }
}

fn continues_here(leave: Leave, region: usize) -> bool {
    match leave {
        Leave::Fall => true,
        Leave::Br { target } => target == region,
        Leave::Return | Leave::Gone => false,
    }
}

fn join_cont(mut left: St, right: St) -> St {
    left.stack = tag::join_vec(&left.stack, &right.stack);
    left.locals = tag::join_vec(&left.locals, &right.locals);
    left.split = wider_split(left.split, right.split);
    left.leave = Leave::Fall;
    left
}

fn wider_split(left: Option<Split>, right: Option<Split>) -> Option<Split> {
    match (left, right) {
        (None, other) | (other, None) => other,
        (Some(left), Some(right)) => Some(if outer_than(left, right) { left } else { right }),
    }
}

/// `at: None` lasts to the function end, so it is outside every label.
fn outer_than(left: Split, right: Split) -> bool {
    match (left.at, right.at) {
        (None, _) => true,
        (_, None) => false,
        (Some(left), Some(right)) => left < right,
    }
}
