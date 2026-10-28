//! Animation-tree evaluation (`docs/animation-design.md` §7): folds an
//! authored [`SkelTree`](crate::tree::SkelTree) with its cursor state
//! ([`SkelTreePlayer`](crate::tree::SkelTreePlayer)) into the phase-E pose
//! pipeline. The model lives in [`crate::tree`]; this module is the
//! deterministic per-frame fold — node order is the evaluation order, and
//! a weighted fold (`BlendedPose::fold_pose`) builds the result the same
//! way `mix_towards` blends a crossfade.

use ornis_core::ColdComponentStore;

use crate::blend::{BlendedPose, SampleBuf, fade_progress, sample_joints};
use crate::tree::{
    MachineCursor, SkelTree, SkelTreePlayer, SwitchCursor, TreeChild2d, TreeNodeDef, TreeTransition,
};
use crate::tree_weights::{BLEND2D_CORNERS, blend1d_weights, blend2d_weights};
use crate::{ClipId, JointTrs, RetargetMap, SkelClip};

/// Scratch buffers of [`eval_tree`], grouped to stay inside clippy's
/// argument budget (same pattern as the type aliases).
pub(crate) struct EvalBuf<'a> {
    /// One scratch pose per node, in node order.
    pub poses: &'a mut [BlendedPose],
    /// Reusable sample scratch (channels plus paint marks).
    pub joints: &'a mut SampleBuf,
    /// Clip ids of the evaluated leaves, appended in node order.
    pub contributors: &'a mut Vec<ClipId>,
    /// Retarget of the root (§8.3); every leaf samples through it.
    pub map: Option<&'a RetargetMap>,
}

/// Evaluates `tree` into `buf.poses` — one scratch pose per node, in node
/// order (the root is the last one) — sampling the [`SkelClip`] leaves at
/// their cursors in `player`.
///
/// The clip ids of every leaf are appended to `buf.contributors`. Leaf
/// cursors apply their own `weight` as in phase E (pull toward rest);
/// `Blend1d` nodes fold their weighted children the same way a crossfade
/// mixes poses. `buf.map` is the root's retarget (§8.3): every leaf
/// samples through it, the same boundary as the flat player, the
/// crossfade and the overlay layers. A violation of the tree contract
/// (missing cursor
/// or clip, bad `axis`, a non-finite parameter or child position, a child
/// not earlier than its parent, missing joint counts) returns `None` and
/// the caller publishes nothing — the "bad skin" rule. The child-order
/// check covers **all** children of a blend node, not just the weighted
/// ones, so one broken tree reads the same way at every parameter value.
pub(crate) fn eval_tree(
    tree: &SkelTree,
    player: &SkelTreePlayer,
    clips: &ColdComponentStore<SkelClip>,
    count: usize,
    rest: &[JointTrs],
    buf: EvalBuf<'_>,
) -> Option<()> {
    let EvalBuf {
        poses,
        joints,
        contributors,
        map,
    } = buf;
    if tree.nodes.is_empty() {
        return None;
    }
    let mut leaves = 0;
    let mut machines = 0;
    let mut switches = 0;
    for (index, node) in tree.nodes.iter().enumerate() {
        match node {
            TreeNodeDef::Clip { clip } => {
                let cursor = player.leaves.get(leaves)?;
                leaves += 1;
                let def = clips.get(clip.0)?;
                sample_joints(def, count, cursor.time.get(), joints, map);
                poses[index].refill(&joints.samples, rest, cursor.weight.get());
                contributors.push(*clip);
            }
            TreeNodeDef::Blend1d { axis, children } => {
                let value = player.params.get(*axis).copied()?;
                if !value.is_finite() {
                    return None;
                }
                // Every child is validated, not just the weighted ones —
                // the same broken tree must not answer differently at
                // different parameter values.
                for child in children {
                    if child.node >= index || !child.position.is_finite() {
                        return None;
                    }
                }
                poses[index].refill_rest(rest);
                let mut weights = [(0, 0.0f32); 2];
                let filled = blend1d_weights(children, value, &mut weights);
                let mut total = 0.0;
                for (child, weight) in weights.iter().take(filled) {
                    let node_index = children[*child].node;
                    let (earlier, tail) = poses.split_at_mut(index);
                    tail[0].fold_pose(&earlier[node_index], *weight, &mut total);
                }
            }
            TreeNodeDef::Blend2d { axes, children } => {
                eval_blend2d(*axes, children, &player.params, rest, index, poses)?;
            }
            TreeNodeDef::Machine {
                states,
                initial: _,
                transitions,
            } => {
                let cursor = player.machines.get(machines)?;
                machines += 1;
                eval_machine(states, transitions, cursor, &player.params, index, poses)?;
            }
            TreeNodeDef::Switch {
                param,
                children,
                fade: _,
            } => {
                let cursor = player.switches.get(switches)?;
                switches += 1;
                let value = player.params.get(*param).copied()?;
                eval_switch(children, cursor, value, index, poses)?;
            }
        }
    }
    Some(())
}

/// Evaluates one [`TreeNodeDef::Blend2d`] node into `poses[index]`. The
/// hot path re-checks the fold-poisoning part of the contract every frame
/// — both axes in range, finite parameter values and positions, children
/// among the earlier nodes — so the same broken tree never answers
/// differently at different parameter values; the lattice *shape* is
/// validated once at install (`set_skel_tree` rejects a broken grid),
/// and a torn tail just falls off deterministically
/// ([`blend2d_weights`]). The output is the bilinear fold of the weighted
/// cell corners; `None` = bad skin.
pub(crate) fn eval_blend2d(
    axes: [usize; 2],
    children: &[TreeChild2d],
    params: &[f32],
    rest: &[JointTrs],
    index: usize,
    poses: &mut [BlendedPose],
) -> Option<()> {
    let value = [params.get(axes[0]).copied()?, params.get(axes[1]).copied()?];
    if !value[0].is_finite() || !value[1].is_finite() {
        return None;
    }
    // Every child is validated, not just the weighted ones — the same
    // broken tree must not answer differently at different parameter
    // values.
    for child in children {
        if child.node >= index || !child.position[0].is_finite() || !child.position[1].is_finite() {
            return None;
        }
    }
    poses[index].refill_rest(rest);
    let mut weights = [(0, 0.0f32); BLEND2D_CORNERS];
    let filled = blend2d_weights(children, value, &mut weights);
    let mut total = 0.0;
    for (child, weight) in weights.iter().take(filled) {
        let node_index = children[*child].node;
        let (earlier, tail) = poses.split_at_mut(index);
        tail[0].fold_pose(&earlier[node_index], *weight, &mut total);
    }
    Some(())
}

/// The [`eval_tree`] machine contract, checked every frame by both the
/// pose fold ([`eval_machine`]) and the event-weight walk
/// ([`crate::tree_eff::fold_machine_eff`]): non-empty states among the
/// earlier nodes, in-range cursors — the same broken machine must not
/// answer differently after a transition.
pub(crate) fn machine_states_valid(
    index: usize,
    states: &[usize],
    cursor: &MachineCursor,
) -> Option<()> {
    if states.is_empty() || cursor.state >= states.len() {
        return None;
    }
    if cursor.from.is_some_and(|from| from >= states.len()) {
        return None;
    }
    for state in states {
        if *state >= index {
            return None;
        }
    }
    Some(())
}

/// Evaluates one [`TreeNodeDef::Machine`] node into `poses[index]`: the
/// whole machine contract is validated every frame — non-empty `states`
/// among the earlier nodes, in-range cursors and transitions, finite
/// guards — so a broken machine must not answer differently after a
/// transition. The output is a pure state pose or a
/// [`SkelFade`](crate::SkelFade)-progress mix of the leaving and heading
/// states; `None` = bad skin.
fn eval_machine(
    states: &[usize],
    transitions: &[TreeTransition],
    cursor: &MachineCursor,
    params: &[f32],
    index: usize,
    poses: &mut [BlendedPose],
) -> Option<()> {
    machine_states_valid(index, states, cursor)?;
    for transition in transitions {
        if transition.from >= states.len()
            || transition.to >= states.len()
            || !transition.when.value.is_finite()
            || !transition.fade.get().is_finite()
        {
            return None;
        }
        let value = params.get(transition.when.param).copied()?;
        if !value.is_finite() {
            return None;
        }
    }
    let (earlier, tail) = poses.split_at_mut(index);
    match cursor.from {
        Some(from) => {
            let progress = fade_progress(cursor.duration, cursor.elapsed);
            if !progress.is_finite() {
                return None;
            }
            fade_fold(
                earlier,
                &mut tail[0],
                states[from],
                states[cursor.state],
                progress,
            );
        }
        None => tail[0].clone_from(&earlier[states[cursor.state]]),
    }
    Some(())
}

/// The [`eval_tree`] switch contract, checked every frame by both the
/// pose fold ([`eval_switch`]) and the event-weight walk
/// ([`crate::tree_eff::fold_switch_eff`]): non-empty children among the
/// earlier nodes, in-range cursors — the same broken switch must not
/// answer differently after a selection change.
pub(crate) fn switch_children_valid(
    index: usize,
    children: &[usize],
    cursor: &SwitchCursor,
) -> Option<()> {
    if children.is_empty() || cursor.child >= children.len() {
        return None;
    }
    if cursor.from.is_some_and(|from| from >= children.len()) {
        return None;
    }
    for child in children {
        if *child >= index {
            return None;
        }
    }
    Some(())
}

/// Evaluates one [`TreeNodeDef::Switch`] node into `poses[index]`: the
/// whole switch contract is validated every frame — non-empty `children`
/// among the earlier nodes, in-range cursors, finite parameter — so a
/// broken switch must not answer differently after a selection change.
/// The output is a pure child pose or a [`SkelFade`](crate::SkelFade)-
/// progress mix of the leaving and heading children; `None` = bad skin.
fn eval_switch(
    children: &[usize],
    cursor: &SwitchCursor,
    value: f32,
    index: usize,
    poses: &mut [BlendedPose],
) -> Option<()> {
    switch_children_valid(index, children, cursor)?;
    if !value.is_finite() {
        return None;
    }
    let (earlier, tail) = poses.split_at_mut(index);
    match cursor.from {
        Some(from) => {
            let progress = fade_progress(cursor.duration, cursor.elapsed);
            if !progress.is_finite() {
                return None;
            }
            fade_fold(
                earlier,
                &mut tail[0],
                children[from],
                children[cursor.child],
                progress,
            );
        }
        None => tail[0].clone_from(&earlier[children[cursor.child]]),
    }
    Some(())
}

/// The shared crossfade fold of a machine or switch node: `out` takes the
/// leaving node's pose, mixed toward the heading node's by `progress`
/// ([`SkelFade`](crate::SkelFade) mechanics).
fn fade_fold(
    earlier: &[BlendedPose],
    out: &mut BlendedPose,
    from_node: usize,
    to_node: usize,
    progress: f32,
) {
    out.clone_from(&earlier[from_node]);
    out.mix_towards(&earlier[to_node], progress);
}
