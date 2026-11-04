//! Join the two arms of an `if` back into one uniformity state.
//!
//! An invocation-dependent condition that takes only one arm past the `if`
//! leaves the continuation split. A condition this pass did not prove
//! invocation-dependent joins as if both arms were uniform.

use super::Leave;
use super::Split;
use super::St;
use super::tag;
use crate::error::BarrierSite;

pub(super) fn merge_if(
    cond: tag::Tag,
    parent: Option<Split>,
    if_index: usize,
    then_end: St,
    else_end: St,
) -> St {
    let then_live = live_at(then_end.leave, if_index);
    let else_live = live_at(else_end.leave, if_index);
    if cond != tag::Tag::Inv {
        return merge_unproven(parent, then_end, else_end, then_live, else_live);
    }
    match (then_live, else_live) {
        (true, true) => joined(then_end, else_end, parent),
        (true, false) => diverge_after(then_end, else_end.leave, parent),
        (false, true) => diverge_after(else_end, then_end.leave, parent),
        (false, false) => stopped(then_end, else_end, parent),
    }
}

/// An invocation-dependent arm leaves `loop_index` while the other stays.
pub(super) fn exits_loop(leave: Leave, loop_index: usize) -> bool {
    match leave {
        Leave::Return => true,
        Leave::Br { target } => target < loop_index,
        Leave::Fall | Leave::Gone => false,
    }
}

fn merge_unproven(
    parent: Option<Split>,
    then_end: St,
    else_end: St,
    then_live: bool,
    else_live: bool,
) -> St {
    match (then_live, else_live) {
        (true, true) => joined(then_end, else_end, parent),
        (true, false) => fall_through(then_end, parent),
        (false, true) => fall_through(else_end, parent),
        (false, false) => stopped(then_end, else_end, parent),
    }
}

fn joined(then_end: St, else_end: St, parent: Option<Split>) -> St {
    St {
        stack: tag::join_vec(&then_end.stack, &else_end.stack),
        locals: tag::join_vec(&then_end.locals, &else_end.locals),
        split: parent,
        leave: Leave::Fall,
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
        Leave::Fall | Leave::Gone => Split {
            site: parent.map(|item| item.site).unwrap_or(BarrierSite::Branch),
            at: None,
        },
    }
}

fn stopped(mut then_end: St, else_end: St, parent: Option<Split>) -> St {
    then_end.leave = outer_leave(then_end.leave, else_end.leave);
    then_end.split = parent;
    then_end
}

fn outer_leave(then_leave: Leave, else_leave: Leave) -> Leave {
    match (then_leave, else_leave) {
        (Leave::Return, Leave::Return) => Leave::Return,
        (Leave::Br { target: left }, Leave::Br { target: right }) if left == right => {
            Leave::Br { target: left }
        }
        (Leave::Br { target }, Leave::Return) | (Leave::Return, Leave::Br { target }) => {
            Leave::Br { target }
        }
        _ => Leave::Gone,
    }
}

fn live_at(leave: Leave, if_index: usize) -> bool {
    match leave {
        Leave::Fall => true,
        Leave::Br { target } => target == if_index,
        Leave::Return | Leave::Gone => false,
    }
}
