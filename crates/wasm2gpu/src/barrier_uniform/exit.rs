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
        let mut states = vec![self.st.clone()];
        states.append(&mut self.pending);
        let (arrivals, outward) = split_arrivals(states, index);
        self.pending = outward;
        let Some(merged) = fold_arrivals(arrivals) else {
            if let Some(out) = self.pending.pop() {
                self.st = out;
            }
            return;
        };
        self.st = settle(merged, &self.pending, parent, index);
    }
}

fn split_arrivals(states: Vec<St>, index: usize) -> (Vec<St>, Vec<St>) {
    let mut arrivals = Vec::new();
    let mut outward = Vec::new();
    for st in states {
        if continues_here(st.leave, index) {
            arrivals.push(st);
        } else {
            outward.push(st);
        }
    }
    (arrivals, outward)
}

fn fold_arrivals(mut arrivals: Vec<St>) -> Option<St> {
    let mut merged = arrivals.pop()?;
    for alt in arrivals {
        let control = merged.split.is_some() || alt.split.is_some();
        merged = join_cont(merged, alt, control);
    }
    Some(merged)
}

/// Keep a split open when some divergent exit did not arrive here.
fn settle(mut merged: St, elsewhere: &[St], parent: Option<Split>, index: usize) -> St {
    if elsewhere.iter().any(|st| st.split.is_some()) {
        merged.leave = Leave::Fall;
        merged.split = hold_split(merged.split, elsewhere, parent);
        return merged;
    }
    reconverge_at(&mut merged, parent, index);
    merged
}

fn hold_split(split: Option<Split>, elsewhere: &[St], parent: Option<Split>) -> Option<Split> {
    let mut early = matches!(
        split,
        Some(Split {
            site: BarrierSite::EarlyReturn,
            ..
        })
    );
    let mut at = split.and_then(|item| item.at);
    for st in elsewhere {
        if st.split.is_none() {
            continue;
        }
        match st.leave {
            Leave::Return | Leave::Gone => {
                early = true;
                at = None;
            }
            Leave::Br { target } if !early => {
                at = Some(at.map(|cur| cur.min(target)).unwrap_or(target));
            }
            Leave::Br { .. } | Leave::Fall => {}
        }
    }
    let site = if early {
        BarrierSite::EarlyReturn
    } else {
        split.map(|item| item.site).unwrap_or(BarrierSite::Branch)
    };
    wider_split(parent, Some(Split { site, at }))
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

fn join_cont(mut left: St, right: St, control: bool) -> St {
    if control {
        left.stack = tag::control_join_vec(&left.stack, &right.stack);
        left.locals = tag::control_join_vec(&left.locals, &right.locals);
    } else {
        left.stack = tag::join_vec(&left.stack, &right.stack);
        left.locals = tag::join_vec(&left.locals, &right.locals);
    }
    left.split = wider_split(left.split, right.split);
    left.leave = Leave::Fall;
    left
}

pub(super) fn wider_split(left: Option<Split>, right: Option<Split>) -> Option<Split> {
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
