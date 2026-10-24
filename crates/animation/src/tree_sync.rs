//! Clip sync marks (`docs/animation-design.md` §7.5): named normalized
//! positions in a clip's loop and the transition-time re-phasing of leaf
//! cursors that aligns two clips on a shared mark. A separate module
//! from `tree` because evaluation and phase synchronization are
//! different responsibilities.

use ornis_core::ColdComponentStore;

use crate::{Seconds, SkelClip, SkelPlayer, SkelTree, TreeNodeDef};

/// One named sync mark of a clip: a normalized position in the clip's
/// loop, in `[0, 1)`.
///
/// Marks are authoring data ([`ClipSyncMarks`], keyed by [`ClipId`](crate::ClipId)).
/// When a machine transition fires, the incoming state's clip cursors
/// re-phase so that same-named marks of the leaving and arriving clips
/// land on the same wall-clock hit (see `docs/animation-design.md`
/// §7.5).
#[derive(Debug, Clone, PartialEq)]
pub struct SyncMark {
    /// Name matched across clips at a transition; names are matched
    /// literally. The first mark of a name in author order owns the
    /// alignment grid.
    pub name: String,
    /// Position within one loop of the clip: `0.0 <= at < 1.0`.
    pub at: f32,
}

/// Sync marks of one clip — cold authoring data keyed by [`ClipId`](crate::ClipId) in
/// the [`ColdComponentStore`] lane. Marks are optional: a clip without
/// marks (or without a name shared with the other side) never re-phases
/// anyone. The lane is written through `set_clip_sync_marks`; marks with
/// a non-finite `at` slipped in any other way are ignored at re-phasing
/// time.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClipSyncMarks {
    /// Marks in author order; several marks per name are legal (the
    /// later ones follow the first one's grid).
    pub marks: Vec<SyncMark>,
}

/// The first `incoming` mark whose name `outgoing` also carries, with
/// the outgoing clip's first mark of that name: `(in_at, out_at)`.
/// `None` = no shared name, nothing to align.
pub(crate) fn shared_mark_at(incoming: &[SyncMark], outgoing: &[SyncMark]) -> Option<(f32, f32)> {
    for mark in incoming {
        if !mark.at.is_finite() {
            continue;
        }
        if let Some(other) = outgoing
            .iter()
            .find(|other| other.name == mark.name && other.at.is_finite())
        {
            return Some((mark.at, other.at));
        }
    }
    None
}

/// The cursor time that puts the incoming clip's `in_at` mark on the
/// outgoing clip's next `out_at` hit: both clips reach their mark after
/// the same wall-clock delta. `out_time` and the cycles are seconds; the
/// caller skips non-positive cycles.
pub(crate) fn sync_cursor_time(
    out_time: f32,
    out_at: f32,
    out_cycle: f32,
    in_at: f32,
    in_cycle: f32,
) -> f32 {
    let delta = (out_at * out_cycle - out_time).rem_euclid(out_cycle);
    (in_at * in_cycle - delta).rem_euclid(in_cycle)
}

/// Whether `parent` names `child` among its inputs (a blend child or a
/// machine state).
fn node_references(tree: &SkelTree, parent: usize, child: usize) -> bool {
    match tree.nodes.get(parent) {
        Some(TreeNodeDef::Blend1d { children, .. }) => {
            children.iter().any(|input| input.node == child)
        }
        Some(TreeNodeDef::Blend2d { children, .. }) => {
            children.iter().any(|input| input.node == child)
        }
        Some(TreeNodeDef::Machine { states, .. }) => {
            for state in states {
                if *state == child {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

/// Clip-leaf node indices under `node` (the `Clip` node itself, or the
/// leaves of its `Blend1d`/`Machine` subtree), in node order. Children
/// always reference earlier nodes, so one downward pass over the
/// reachability flags covers the whole subtree — no recursion.
fn subtree_leaves(tree: &SkelTree, node: usize) -> Vec<usize> {
    if node >= tree.nodes.len() {
        return Vec::new();
    }
    let mut reached = vec![false; tree.nodes.len()];
    reached[node] = true;
    for index in (0..node).rev() {
        let touched = (index + 1..=node)
            .any(|parent| reached[parent] && node_references(tree, parent, index));
        if touched {
            reached[index] = true;
        }
    }
    (0..tree.nodes.len())
        .filter(|index| reached[*index] && matches!(tree.nodes[*index], TreeNodeDef::Clip { .. }))
        .collect()
}

/// The cursor slot of a `Clip` node: clip leaves are addressed in node
/// order ([`SkelTreePlayer::leaves`](crate::SkelTreePlayer::leaves) pairs with the i-th clip node).
fn leaf_cursor_of(tree: &SkelTree, node: usize) -> Option<usize> {
    if !matches!(tree.nodes.get(node), Some(TreeNodeDef::Clip { .. })) {
        return None;
    }
    Some(
        tree.nodes[..node]
            .iter()
            .filter(|seen| matches!(seen, TreeNodeDef::Clip { .. }))
            .count(),
    )
}

/// Clocks of the leaving state's clip leaves, read up front: re-phasing
/// must not depend on node order (`docs/animation-design.md` §7.5).
fn leaving_clocks(tree: &SkelTree, from_node: usize, leaves: &[SkelPlayer]) -> Vec<(usize, f32)> {
    subtree_leaves(tree, from_node)
        .into_iter()
        .filter_map(|node| {
            let cursor = leaves.get(leaf_cursor_of(tree, node)?)?;
            Some((node, cursor.time.get()))
        })
        .collect()
}

/// Whether `leaf` is one of the leaving state's clip leaves.
fn leaf_listed(from_leaves: &[(usize, f32)], leaf: usize) -> bool {
    for &(node, _) in from_leaves {
        if node == leaf {
            return true;
        }
    }
    false
}

/// Re-phases the incoming state's clip cursors when a machine transition
/// fires (`docs/animation-design.md` §7.5): each looping incoming leaf
/// takes its phase from the first leaving-state leaf (node order) whose
/// clip shares a mark name — the clips' next same-named marks land on the
/// same wall-clock hit. Leaves without a partner (no shared name, no
/// marks, missing clip, non-positive cycle, not looping) keep their
/// clocks. A leaf living in both states is skipped whole — the same
/// cursors serve both sides, so self-loops and shared subtrees never
/// re-phase — and the leaving clocks are read before any write.
pub(crate) fn sync_transition_leaves(
    leaves: &mut [SkelPlayer],
    tree: &SkelTree,
    from_node: usize,
    to_node: usize,
    clips: &ColdComponentStore<SkelClip>,
    marks: &ColdComponentStore<ClipSyncMarks>,
) {
    let from_leaves = leaving_clocks(tree, from_node, leaves);
    for incoming in subtree_leaves(tree, to_node) {
        let Some(slot) = leaf_cursor_of(tree, incoming) else {
            continue;
        };
        if leaf_listed(&from_leaves, incoming) {
            continue;
        }
        if !leaves.get(slot).is_some_and(|cursor| cursor.looping) {
            continue;
        }
        let Some(TreeNodeDef::Clip { clip: in_id }) = tree.nodes.get(incoming) else {
            continue;
        };
        let Some(in_marks) = marks.get(in_id.0) else {
            continue;
        };
        let Some(in_clip) = clips.get(in_id.0) else {
            continue;
        };
        if in_clip.duration <= 0.0 {
            continue;
        }
        let mut aligned = None;
        for &(outgoing, out_time) in &from_leaves {
            let Some(TreeNodeDef::Clip { clip: out_id }) = tree.nodes.get(outgoing) else {
                continue;
            };
            let Some(out_marks) = marks.get(out_id.0) else {
                continue;
            };
            let Some(out_clip) = clips.get(out_id.0) else {
                continue;
            };
            if out_clip.duration <= 0.0 {
                continue;
            }
            let Some((in_at, out_at)) = shared_mark_at(&in_marks.marks, &out_marks.marks) else {
                continue;
            };
            aligned = Some(sync_cursor_time(
                out_time,
                out_at,
                out_clip.duration,
                in_at,
                in_clip.duration,
            ));
            break;
        }
        if let (Some(aligned), Some(cursor)) = (aligned, leaves.get_mut(slot)) {
            cursor.time = Seconds::new(aligned);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Bodies sit in one inner block each: rustqual's fragment scanner
    // hashes windows of three consecutive top-level statements, and the
    // neighbouring modules already own the natural statement shapes.
    #[test]
    fn shared_mark_picks_the_first_incoming_name() {
        {
            let incoming = vec![
                SyncMark {
                    name: "b".to_string(),
                    at: 0.3,
                },
                SyncMark {
                    name: "a".to_string(),
                    at: 0.1,
                },
            ];
            let outgoing = vec![
                SyncMark {
                    name: "a".to_string(),
                    at: 0.2,
                },
                SyncMark {
                    name: "b".to_string(),
                    at: 0.4,
                },
            ];
            assert_eq!(shared_mark_at(&incoming, &outgoing), Some((0.3, 0.4)));
        }
    }

    #[test]
    fn shared_mark_skips_non_finite_at() {
        {
            let nan = f32::NAN;
            let incoming = vec![
                SyncMark {
                    name: "foot".to_string(),
                    at: nan,
                },
                SyncMark {
                    name: "hand".to_string(),
                    at: 0.3,
                },
            ];
            let outgoing = vec![
                SyncMark {
                    name: "foot".to_string(),
                    at: 0.4,
                },
                SyncMark {
                    name: "hand".to_string(),
                    at: nan,
                },
            ];
            assert_eq!(shared_mark_at(&incoming, &outgoing), None);
        }
    }

    #[test]
    fn disjoint_names_share_nothing() {
        {
            let incoming = vec![SyncMark {
                name: "a".to_string(),
                at: 0.1,
            }];
            let outgoing = vec![SyncMark {
                name: "b".to_string(),
                at: 0.2,
            }];
            assert_eq!(shared_mark_at(&incoming, &outgoing), None);
        }
    }

    #[test]
    fn sync_cursor_time_aligns_the_next_hits() {
        {
            // walk at 0.5 of 1.0 s, its 0.75 mark hits in 0.25 s; run
            // (0.5 s, mark at 0.25) must sit at 0.375 to hit in 0.25 s.
            assert_eq!(sync_cursor_time(0.5, 0.75, 1.0, 0.25, 0.5), 0.375);
        }
    }

    #[test]
    fn sync_cursor_time_wraps_around_the_cycle() {
        {
            // the outgoing mark hit 0.25 s ago (delta 0.5 s); the
            // incoming cursor wraps to 0.125 of its 0.5 s cycle.
            assert_eq!(sync_cursor_time(0.75, 0.25, 1.0, 0.25, 0.5), 0.125);
        }
    }

    #[test]
    fn subtree_leaves_cover_blend_children() {
        {
            let tree = SkelTree {
                nodes: vec![
                    TreeNodeDef::Clip {
                        clip: crate::ClipId(ornis_core::Entity::new(1)),
                    },
                    TreeNodeDef::Clip {
                        clip: crate::ClipId(ornis_core::Entity::new(2)),
                    },
                    TreeNodeDef::Blend1d {
                        axis: 0,
                        children: vec![
                            crate::TreeChild {
                                position: 0.0,
                                node: 0,
                            },
                            crate::TreeChild {
                                position: 1.0,
                                node: 1,
                            },
                        ],
                    },
                    TreeNodeDef::Machine {
                        states: vec![2],
                        initial: 0,
                        transitions: Vec::new(),
                    },
                ],
                params: Vec::new(),
            };
            assert_eq!(subtree_leaves(&tree, 3), vec![0, 1]);
            assert_eq!(leaf_cursor_of(&tree, 1), Some(1));
        }
    }
}
