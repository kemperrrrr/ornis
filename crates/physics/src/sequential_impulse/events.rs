//! Trigger and contact overlap reconciliation for the sequential-impulse step.

use rustc_hash::FxHashSet;

use crate::body::RigidBody;
use crate::distance;
use crate::trigger::{TriggerEvent, TriggerEventKind};

/// Detect actual (not speculative) overlaps for pairs containing a trigger.
///
/// Trigger geometry uses the exact distance oracle rather than the contact
/// margin: a nearby but non-overlapping body must not emit `Entered`. The
/// broadphase has already applied the mutual layer masks, so this pass only
/// performs the shape-level check.
pub(crate) fn detect_trigger_overlaps(
    bodies: &[RigidBody],
    active: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let mut overlaps = Vec::new();
    for &(i, j) in active {
        let a = &bodies[i];
        let b = &bodies[j];
        if !(a.is_trigger || b.is_trigger) || !a.can_collide_with(b) {
            continue;
        }
        let distance = distance::shape_distance(
            distance::ShapeRef {
                shape: &a.shape,
                pos: a.position,
                rot: a.orientation,
            },
            distance::ShapeRef {
                shape: &b.shape,
                pos: b.position,
                rot: b.orientation,
            },
        );
        if distance.dist <= 0.0 {
            overlaps.push((i, j));
        }
    }
    overlaps
}

/// Reconcile the current overlap set with the previous step and queue sorted
/// enter/exit events. Sorting keeps event order independent of broadphase
/// sweep-axis rotation and hash-set iteration order.
pub(crate) fn update_trigger_events(
    previous: &FxHashSet<(usize, usize)>,
    current: Vec<(usize, usize)>,
    events: &mut Vec<TriggerEvent>,
) -> FxHashSet<(usize, usize)> {
    let current_set: FxHashSet<(usize, usize)> = current.into_iter().collect();
    let mut entered: Vec<_> = current_set.difference(previous).copied().collect();
    let mut exited: Vec<_> = previous.difference(&current_set).copied().collect();
    entered.sort_unstable();
    exited.sort_unstable();
    events.extend(entered.into_iter().map(|(body_a, body_b)| TriggerEvent {
        body_a,
        body_b,
        kind: TriggerEventKind::Entered,
    }));
    events.extend(exited.into_iter().map(|(body_a, body_b)| TriggerEvent {
        body_a,
        body_b,
        kind: TriggerEventKind::Exited,
    }));
    current_set
}
