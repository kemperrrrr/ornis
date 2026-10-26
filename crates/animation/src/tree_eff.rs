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
//! so a broken tree answers the same way in both.

use crate::blend::fade_progress;
use crate::tree::{
    MachineCursor, SkelTree, SkelTreePlayer, TreeChild, TreeChild2d, TreeNodeDef,
    machine_states_valid,
};
use crate::tree_weights::{BLEND2D_CORNERS, blend1d_weights, blend2d_weights};

/// Effective blend weight of every [`TreeNodeDef::Clip`] leaf, in leaf
/// order — the product of the blend-node weights along the path from the
/// tree root, times the machine-state activity of every machine on the
/// path (`0` while a state is not the machine's current one, the
/// crossfade share while a transition flies). A leaf's own cursor weight
/// ([`SkelPlayer::weight`](crate::SkelPlayer::weight)) is not included —
/// the cursor carries it; the product lands in
/// [`crate::events::AnimEventOut::weight`] so the host filters on one
/// number (design §8.1).
///
/// A missing tree, or one whose contract [`crate::tree::eval_tree`] would
/// read as bad skin, answers `1.0` everywhere: the cursors' own weights
/// stand alone.
pub(crate) fn leaf_effective_weights(tree: Option<&SkelTree>, player: &SkelTreePlayer) -> Vec<f32> {
    let Some(tree) = tree else {
        return vec![1.0; player.leaves.len()];
    };
    eff_weights(tree, player).unwrap_or_else(|| vec![1.0; player.leaves.len()])
}

/// The [`leaf_effective_weights`] walk over one tree; `None` = the tree
/// contract is broken here.
fn eff_weights(tree: &SkelTree, player: &SkelTreePlayer) -> Option<Vec<f32>> {
    let mut eff = vec![0.0; tree.nodes.len()];
    *eff.last_mut()? = 1.0;
    // Children sit strictly earlier than their parents, so one reverse
    // walk hands a node's weight to its children before it folds itself.
    let mut machine_cursor = tree
        .nodes
        .iter()
        .filter(|node| matches!(node, TreeNodeDef::Machine { .. }))
        .count();
    for index in (0..tree.nodes.len()).rev() {
        let weight = eff[index];
        match &tree.nodes[index] {
            TreeNodeDef::Clip { .. } => {}
            TreeNodeDef::Blend1d { axis, children } => {
                let value = player.params.get(*axis).copied()?;
                fold_blend1d_eff(&mut eff, index, weight, children, value)?;
            }
            TreeNodeDef::Blend2d { axes, children } => {
                let value = [
                    player.params.get(axes[0]).copied()?,
                    player.params.get(axes[1]).copied()?,
                ];
                fold_blend2d_eff(&mut eff, index, weight, children, value)?;
            }
            TreeNodeDef::Machine { states, .. } => {
                machine_cursor = machine_cursor.checked_sub(1)?;
                let cursor = player.machines.get(machine_cursor)?;
                fold_machine_eff(&mut eff, index, weight, states, cursor)?;
            }
        }
    }
    Some(
        tree.nodes
            .iter()
            .zip(&eff)
            .filter_map(|(node, weight)| {
                matches!(node, TreeNodeDef::Clip { .. }).then_some(*weight)
            })
            .collect(),
    )
}

/// Folds one [`TreeNodeDef::Blend1d`] node's weight into its weighted
/// children; `None` on the [`crate::tree::eval_tree`] contract violations
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
/// cell corners; `None` on the [`crate::tree::eval_tree`] contract
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
            eff[states[from]] += weight * (1.0 - progress);
            eff[states[cursor.state]] += weight * progress;
        }
        None => eff[states[cursor.state]] += weight,
    }
    Some(())
}
