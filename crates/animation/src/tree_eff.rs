//! Effective leaf weights of an animation tree (`docs/animation-design.md`
//! §8.1): the share one clip leaf contributes to the sampled pose — the
//! product of the blend-node weights along the path from the tree root and
//! the machine-state activity of every machine on the path. The event emit
//! scales each leaf's cursor by this share, so the host filters on one
//! number ([`crate::events::AnimEventOut::weight`]).
//!
//! Separate from [`crate::tree`] (the pose evaluator): this walk runs at
//! cursor-advance time and never touches poses. It reuses the blend tables
//! of [`crate::tree_weights`] and the machine contract of [`crate::tree`],
//! so a broken tree answers the same way in both. The walk allocates
//! nothing per frame: its two buffers live in [`WeightsScratch`] and
//! refill in place (design §8.1).

use crate::blend::fade_progress;
use crate::tree::{
    MachineCursor, SkelTree, SkelTreePlayer, SwitchCursor, TreeChild, TreeChild2d, TreeNodeDef,
};
use crate::tree_eval::{machine_states_valid, switch_children_valid};
use crate::tree_weights::{BLEND2D_CORNERS, blend1d_weights, blend2d_weights};

/// Reusable buffers of the [`leaf_effective_weights`] walk — node weights
/// and the leaf result. The caller keeps one in its frame scratch (like
/// `scratch.finished`), so the walk allocates nothing per frame.
#[derive(Default)]
pub(crate) struct WeightsScratch {
    eff: Vec<f32>,
    leaves: Vec<f32>,
}

/// Effective blend weight of every [`TreeNodeDef::Clip`] leaf, in leaf
/// order — the product of the blend-node weights along the path from the
/// tree root, times the machine-state activity of every machine on the
/// path (`0` while a state is not the machine's current one, the
/// crossfade share while a transition flies). A leaf's own cursor weight
/// ([`SkelPlayer::weight`](crate::SkelPlayer::weight)) is not included —
/// the cursor carries it; the product lands in
/// [`crate::events::AnimEventOut::weight`] so the host filters on one
/// number (design §8.1). The result reuses `scratch` and always covers
/// `player.leaves.len()` slots.
///
/// A missing tree, or one whose contract [`crate::tree_eval::eval_tree`] would
/// read as bad skin, answers `1.0` everywhere: the cursors' own weights
/// stand alone. The flag is `false` in that case — event emit still uses
/// the `1.0` weights (§8.1), while root motion treats the tree as no
/// contribution (§8.4, a broken tree publishes no pose).
pub(crate) fn leaf_effective_weights<'a>(
    scratch: &'a mut WeightsScratch,
    tree: Option<&SkelTree>,
    player: &SkelTreePlayer,
) -> (&'a [f32], bool) {
    if let Some(tree) = tree
        && eff_weights(scratch, tree, player).is_some()
    {
        return (&scratch.leaves, true);
    }
    scratch.leaves.clear();
    scratch.leaves.resize(player.leaves.len(), 1.0);
    (&scratch.leaves, false)
}

/// The [`leaf_effective_weights`] walk over one tree; `None` = the tree
/// contract is broken here (the caller answers `1.0`).
fn eff_weights(
    scratch: &mut WeightsScratch,
    tree: &SkelTree,
    player: &SkelTreePlayer,
) -> Option<()> {
    scratch.eff.clear();
    scratch.eff.resize(tree.nodes.len(), 0.0);
    *scratch.eff.last_mut()? = 1.0;
    // Children sit strictly earlier than their parents, so one reverse
    // walk hands a node's weight to its children before it folds itself.
    let mut machine_cursor = tree
        .nodes
        .iter()
        .filter(|node| matches!(node, TreeNodeDef::Machine { .. }))
        .count();
    let mut switch_cursor = tree
        .nodes
        .iter()
        .filter(|node| matches!(node, TreeNodeDef::Switch { .. }))
        .count();
    for index in (0..tree.nodes.len()).rev() {
        let weight = scratch.eff[index];
        match &tree.nodes[index] {
            TreeNodeDef::Clip { .. } => {}
            TreeNodeDef::Blend1d { axis, children } => {
                let value = player.params.get(*axis).copied()?;
                fold_blend1d_eff(&mut scratch.eff, index, weight, children, value)?;
            }
            TreeNodeDef::Blend2d { axes, children } => {
                let value = [
                    player.params.get(axes[0]).copied()?,
                    player.params.get(axes[1]).copied()?,
                ];
                fold_blend2d_eff(&mut scratch.eff, index, weight, children, value)?;
            }
            TreeNodeDef::Machine { states, .. } => {
                machine_cursor = machine_cursor.checked_sub(1)?;
                let cursor = player.machines.get(machine_cursor)?;
                fold_machine_eff(&mut scratch.eff, index, weight, states, cursor)?;
            }
            TreeNodeDef::Switch {
                param, children, ..
            } => {
                switch_cursor = switch_cursor.checked_sub(1)?;
                let cursor = player.switches.get(switch_cursor)?;
                player
                    .params
                    .get(*param)
                    .copied()
                    .filter(|value| value.is_finite())?;
                fold_switch_eff(&mut scratch.eff, index, weight, children, cursor)?;
            }
        }
    }
    scratch.leaves.clear();
    for (node, weight) in tree.nodes.iter().zip(scratch.eff.iter()) {
        if matches!(node, TreeNodeDef::Clip { .. }) {
            scratch.leaves.push(*weight);
        }
    }
    scratch.leaves.resize(player.leaves.len(), 1.0);
    Some(())
}

/// Folds one [`TreeNodeDef::Blend1d`] node's weight into its weighted
/// children; `None` on the [`crate::tree_eval::eval_tree`] contract violations
/// (non-finite value/position, a child not earlier than the node).
fn fold_blend1d_eff(
    eff: &mut [f32],
    index: usize,
    weight: f32,
    children: &[TreeChild],
    value: f32,
) -> Option<()> {
    if !value.is_finite() {
        return None;
    }
    for child in children {
        if child.node >= index || !child.position.is_finite() {
            return None;
        }
    }
    spread_blend1d(eff, weight, children, value);
    Some(())
}

/// The [`blend1d_weights`] table of a 1-D blend node, spread over its
/// live child slots.
fn spread_blend1d(eff: &mut [f32], weight: f32, children: &[TreeChild], value: f32) {
    let mut weights = [(0, 0.0f32); 2];
    let filled = blend1d_weights(children, value, &mut weights);
    for (child, child_weight) in weights.iter().take(filled) {
        eff[children[*child].node] += weight * child_weight;
    }
}

/// Folds one [`TreeNodeDef::Blend2d`] node's weight into its weighted
/// cell corners; `None` on the [`crate::tree_eval::eval_tree`] contract
/// violations.
fn fold_blend2d_eff(
    eff: &mut [f32],
    index: usize,
    weight: f32,
    children: &[TreeChild2d],
    value: [f32; 2],
) -> Option<()> {
    if !value[0].is_finite() || !value[1].is_finite() {
        return None;
    }
    for child in children {
        if child.node >= index || !child.position[0].is_finite() || !child.position[1].is_finite() {
            return None;
        }
    }
    spread_blend2d(eff, weight, children, value);
    Some(())
}

/// The [`blend2d_weights`] table of a 2-D blend node, spread over its
/// live cell corners.
fn spread_blend2d(eff: &mut [f32], weight: f32, children: &[TreeChild2d], value: [f32; 2]) {
    let mut weights = [(0, 0.0f32); BLEND2D_CORNERS];
    let filled = blend2d_weights(children, value, &mut weights);
    for (child, child_weight) in weights.iter().take(filled) {
        eff[children[*child].node] += weight * child_weight;
    }
}

/// Folds one [`TreeNodeDef::Machine`] node's weight into its states: the
/// current one owns the pure share, an in-flight crossfade splits it with
/// [`crate::blend::fade_progress`], every other state stays at `0`;
/// `None` on the machine contract violations of `machine_states_valid`
/// or a non-finite crossfade clock.
fn fold_machine_eff(
    eff: &mut [f32],
    index: usize,
    weight: f32,
    states: &[usize],
    cursor: &MachineCursor,
) -> Option<()> {
    machine_states_valid(index, states, cursor)?;
    match cursor.from {
        Some(from) => {
            let progress = fade_progress(cursor.duration, cursor.elapsed);
            if !progress.is_finite() {
                return None;
            }
            fade_split(eff, weight, states[from], states[cursor.state], progress);
        }
        None => eff[states[cursor.state]] += weight,
    }
    Some(())
}

/// Folds one [`TreeNodeDef::Switch`] node's weight into its children: the
/// standing selection owns the pure share, an in-flight crossfade splits
/// it with [`crate::blend::fade_progress`], every other child stays at
/// `0`; `None` on the switch contract violations of
/// [`switch_children_valid`] or a non-finite crossfade clock. The walk
/// itself rejects a missing or non-finite parameter before the fold runs.
pub(crate) fn fold_switch_eff(
    eff: &mut [f32],
    index: usize,
    weight: f32,
    children: &[usize],
    cursor: &SwitchCursor,
) -> Option<()> {
    switch_children_valid(index, children, cursor)?;
    match cursor.from {
        Some(from) => {
            let progress = fade_progress(cursor.duration, cursor.elapsed);
            if !progress.is_finite() {
                return None;
            }
            fade_split(
                eff,
                weight,
                children[from],
                children[cursor.child],
                progress,
            );
        }
        None => eff[children[cursor.child]] += weight,
    }
    Some(())
}

/// The shared crossfade split of a machine or switch node: the leaving
/// node takes `weight · (1 − progress)`, the heading one `weight ·
/// progress`.
fn fade_split(eff: &mut [f32], weight: f32, from_node: usize, to_node: usize, progress: f32) {
    eff[from_node] += weight * (1.0 - progress);
    eff[to_node] += weight * progress;
}
