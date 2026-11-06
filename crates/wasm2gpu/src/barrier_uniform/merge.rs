//! Join the two arms of an `if` back into one uniformity state.
//!
//! An invocation-dependent condition that takes only one arm past the `if`
//! leaves the continuation split. A condition this pass did not prove
//! invocation-dependent does not open a split, but its `br` arm is still
//! carried to the label and every exit is still walked.

use super::Leave;
use super::Split;
use super::St;
use super::exit;
use super::tag;
use crate::error::BarrierSite;

/// Both arms of one `if`, plus the condition that chose them.
pub(super) struct IfEnds {
    pub(super) cond: tag::Tag,
    pub(super) parent: Option<Split>,
    pub(super) index: usize,
    pub(super) then_end: St,
    pub(super) else_end: St,
}

/// The state after the `if`, and an arm that left by `br` or `return`.
///
/// The parked arm is carried to its label with its locals. Both arms are
/// parked when neither falls through, even if the condition is not the
/// invocation id: collapsing them used to stop the walk, so a later barrier
/// was never checked. A parked `Return` stays in the pending list until the
/// function ends and still meets every later label.
pub(super) fn place_if(ends: IfEnds) -> (St, Option<St>) {
    let then_live = live_at(ends.then_end.leave, ends.index);
    let else_live = live_at(ends.else_end.leave, ends.index);
    match (then_live, else_live) {
        (true, true) => (
            joined(
                ends.then_end,
                ends.else_end,
                ends.parent,
                ends.cond == tag::Tag::Inv,
            ),
            None,
        ),
        (true, false) => park(ends.cond, ends.parent, ends.then_end, ends.else_end),
        (false, true) => park(ends.cond, ends.parent, ends.else_end, ends.then_end),
        (false, false) => fork_both(ends),
    }
}

/// An invocation-dependent arm leaves `loop_index` while the other stays.
pub(super) fn exits_loop(leave: Leave, loop_index: usize) -> bool {
    match leave {
        Leave::Return => true,
        Leave::Br { target } => target < loop_index,
        Leave::Fall => false,
    }
}

fn joined(then_end: St, else_end: St, parent: Option<Split>, control: bool) -> St {
    let (stack, locals) = if control {
        (
            tag::control_join_vec(&then_end.stack, &else_end.stack),
            tag::control_join_vec(&then_end.locals, &else_end.locals),
        )
    } else {
        (
            tag::join_vec(&then_end.stack, &else_end.stack),
            tag::join_vec(&then_end.locals, &else_end.locals),
        )
    };
    St {
        stack,
        locals,
        split: parent,
        leave: Leave::Fall,
        taint: then_end.taint,
    }
}

fn fall_through(mut live: St, parent: Option<Split>) -> St {
    live.leave = Leave::Fall;
    live.split = parent;
    live
}

fn diverge_after(mut live: St, dead: Leave, parent: Option<Split>) -> St {
    live.leave = Leave::Fall;
    live.split = Some(opened_by(dead, parent));
    live
}

fn opened_by(dead: Leave, parent: Option<Split>) -> Split {
    if let Some(parent) = parent.filter(|item| item.at.is_none()) {
        return parent;
    }
    match dead {
        Leave::Return => Split {
            site: BarrierSite::EarlyReturn,
            at: None,
        },
        Leave::Br { target } => Split {
            site: parent.map(|item| item.site).unwrap_or(BarrierSite::Branch),
            at: Some(target),
        },
        Leave::Fall => Split {
            site: parent.map(|item| item.site).unwrap_or(BarrierSite::Branch),
            at: None,
        },
    }
}

/// Split carried by exits that do not meet again at the `if`.
///
/// A `return` beside a `br` never rejoins, so the split stays open. Two
/// branches to different labels stay open until the outer one. The same
/// label, or two returns, is not a partial exit.
pub(super) fn cover_exit(
    parent: Option<Split>,
    then_leave: Leave,
    else_leave: Leave,
) -> Option<Split> {
    let extra = match (then_leave, else_leave) {
        (Leave::Return, Leave::Return) => None,
        (Leave::Br { target: left }, Leave::Br { target: right }) if left == right => None,
        (Leave::Br { .. }, Leave::Return) | (Leave::Return, Leave::Br { .. }) => Some(Split {
            site: BarrierSite::EarlyReturn,
            at: None,
        }),
        (Leave::Br { target: left }, Leave::Br { target: right }) => Some(Split {
            site: BarrierSite::Branch,
            at: Some(if left < right { left } else { right }),
        }),
        _ => Some(Split {
            site: BarrierSite::Branch,
            at: None,
        }),
    };
    exit::wider_split(parent, extra)
}

fn park(cond: tag::Tag, parent: Option<Split>, live: St, mut dead: St) -> (St, Option<St>) {
    let leave = dead.leave;
    let cont = if cond == tag::Tag::Inv {
        // The arm's own split ends at this `if`, which has already been
        // popped. Keep the split opened by the exit, not that stale one.
        dead.split = Some(opened_by(leave, parent));
        diverge_after(live, leave, parent)
    } else {
        dead.split = exit::wider_split(dead.split, parent);
        fall_through(live, parent)
    };
    (cont, Some(dead))
}

fn fork_both(ends: IfEnds) -> (St, Option<St>) {
    let mut then_end = ends.then_end;
    let mut else_end = ends.else_end;
    if ends.cond == tag::Tag::Inv {
        // Both arms left. The same label is one destination: `cover_exit`
        // returns the parent split, and a barrier after that label stays
        // uniform. Different labels stay split until the outer one. The
        // split opened for the arm body ends at this `if` and must not
        // outlive it.
        let split = cover_exit(ends.parent, then_end.leave, else_end.leave);
        then_end.split = split;
        else_end.split = split;
    } else {
        then_end.split = exit::wider_split(then_end.split, ends.parent);
        else_end.split = exit::wider_split(else_end.split, ends.parent);
    }
    (then_end, Some(else_end))
}

fn live_at(leave: Leave, if_index: usize) -> bool {
    match leave {
        Leave::Fall => true,
        Leave::Br { target } => target == if_index,
        Leave::Return => false,
    }
}
