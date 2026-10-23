//! Animation trees (`docs/animation-design.md` §7): cold authoring
//! graphs ([`SkelTree`]), hot cursor state ([`SkelTreePlayer`]) and the
//! deterministic blend1d weights that evaluate them into the phase-E pose
//! pipeline.
//!
//! The model keeps the phase-E rules: deterministic and stateless per
//! frame (only cursor clocks mutate), evaluation order is the node order
//! (`Blend1d` children reference earlier nodes, the last node is the
//! root), and a weighted fold (`BlendedPose::fold_pose`) builds the
//! result the same way `mix_towards` blends a crossfade.

use ornis_core::ColdComponentStore;

use crate::blend::{BlendedPose, JointSample, fade_progress, sample_joints};
use crate::{ClipId, JointTrs, Seconds, SkelClip, SkelPlayer};

/// One named parameter of an [`SkelTree`], addressed by index.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeParam {
    /// Host-facing name (the key of the typed parameter input).
    pub name: String,
    /// Value the parameter keeps until a host sets it.
    pub default: f32,
}

/// One input of a [`TreeNodeDef::Blend1d`] node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeChild {
    /// Position of the input on the node's parameter axis; `children`
    /// carry strictly ascending positions.
    pub position: f32,
    /// Index of the child node in [`SkelTree::nodes`], strictly smaller
    /// than the index of the blend node itself.
    pub node: usize,
}

/// Comparison of a tree parameter against [`TreeCond::value`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeCmp {
    /// The parameter's live value is greater than the threshold.
    Greater,
    /// The parameter's live value is less than the threshold.
    Less,
}

/// Guard of a [`TreeTransition`]: one comparison of a live parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeCond {
    /// Index into [`SkelTree::params`] the guard reads.
    pub param: usize,
    /// Comparison applied to the parameter's live value.
    pub op: TreeCmp,
    /// Finite threshold the value is compared against.
    pub value: f32,
}

impl TreeCond {
    /// Whether `params` satisfy the condition. A missing or non-finite
    /// parameter (or threshold) never fires.
    pub(crate) fn holds(&self, params: &[f32]) -> bool {
        let Some(value) = params.get(self.param).copied() else {
            return false;
        };
        if !value.is_finite() || !self.value.is_finite() {
            return false;
        }
        match self.op {
            TreeCmp::Greater => value > self.value,
            TreeCmp::Less => value < self.value,
        }
    }
}

/// One transition of a [`TreeNodeDef::Machine`]: crossfades the machine
/// between two of its states while its `when` guard holds. `from == to`
/// is a legal self-loop — it restarts the state's crossfade into itself
/// after each completion, a harmless way to keep a guard live.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeTransition {
    /// State index (into the machine's `states`) the machine stands on.
    pub from: usize,
    /// State index to crossfade into.
    pub to: usize,
    /// Guard on the tree parameters.
    pub when: TreeCond,
    /// Crossfade length in seconds, [`SkelFade`](crate::SkelFade)
    /// mechanics: linear progress, the outgoing state's cursors keep
    /// stepping; non-positive cuts at once. Must be finite.
    pub fade: Seconds,
}

/// One node of an [`SkelTree`] (authoring data, cold).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TreeNodeDef {
    /// Leaf: one clip, sampled at its cursor in [`SkelTreePlayer::leaves`]
    /// (clip leaves are addressed in node order).
    Clip {
        /// Playlist entity of the [`SkelClip`](crate::SkelClip).
        clip: ClipId,
    },
    /// 1-D blend of `children` along parameter `axis`: at most two
    /// neighbouring inputs carry weight, summing to one; outside the
    /// covered range the nearest end input owns the pose (`blend1d_weights`).
    Blend1d {
        /// Index into [`SkelTree::params`] that drives this node.
        axis: usize,
        /// Inputs at strictly ascending `position`s; empty reads as rest.
        children: Vec<TreeChild>,
    },
    /// State machine over earlier nodes: stands on `initial` and
    /// crossfades between `states` entries as [`TreeTransition`]s fire on
    /// the live parameter values (first match in `transitions` order
    /// wins; an in-flight crossfade is never interrupted — guards are
    /// re-checked once it completes). A crossfade mixes the two state
    /// poses with [`SkelFade`](crate::SkelFade) mechanics.
    Machine {
        /// Indices into [`SkelTree::nodes`], all strictly smaller than
        /// this node's index.
        states: Vec<usize>,
        /// Initial state index into `states`.
        initial: usize,
        /// Transition candidates, in evaluation order.
        transitions: Vec<TreeTransition>,
    },
}

/// Authoring catalog of one animation tree (cold lane, on the skeleton
/// root).
///
/// `nodes` is the evaluation order: every [`TreeNodeDef::Blend1d`]
/// references strictly earlier nodes, and the last node is the tree root.
#[derive(Debug, Clone, PartialEq)]
pub struct SkelTree {
    /// Nodes in evaluation order; the last one is the root.
    pub nodes: Vec<TreeNodeDef>,
    /// Parameters referenced by index from [`TreeNodeDef::Blend1d`] and
    /// [`TreeTransition`]-guards.
    pub params: Vec<TreeParam>,
}

/// Runtime state of one [`TreeNodeDef::Machine`] — one entry of
/// [`SkelTreePlayer::machines`], in machine-node order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MachineCursor {
    /// Index into the machine's `states`: where the machine stands, or
    /// where an in-flight crossfade is heading.
    pub state: usize,
    /// State being left while a crossfade flies; `None` at rest.
    pub from: Option<usize>,
    /// Time spent in the in-flight crossfade.
    pub elapsed: Seconds,
    /// Length of the in-flight crossfade; non-positive cuts at once.
    pub duration: Seconds,
}

/// Runtime state of one [`SkelTree`] (hot lane, on the skeleton root).
///
/// The only mutable state of the tree, and it is all cursor clocks plus
/// host-set parameter values — no pose ever feeds back into the next
/// frame, matching the phase-E contract.
///
/// `set_skel_tree` pairs `leaves[i]` with the i-th [`TreeNodeDef::Clip`]
/// and mirrors that node's id in `leaves[i].clip`; the sampler reads the
/// node's clip def on the cursor's clock. A hand-inserted player that lets
/// the two diverge still samples correctly, but its cursor metadata lies —
/// install through the facade or keep them aligned. The same pairing holds
/// for `machines[i]` and the i-th [`TreeNodeDef::Machine`]: a lane whose
/// counts diverge steps the matching prefix and reads as bad skin
/// (`eval_tree` demands one cursor per machine node).
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct SkelTreePlayer {
    /// Live parameter values, aligned with [`SkelTree::params`].
    pub params: Vec<f32>,
    /// Cursors of the [`TreeNodeDef::Clip`] leaves, in node order.
    pub leaves: Vec<SkelPlayer>,
    /// Cursors of the [`TreeNodeDef::Machine`] nodes, in node order.
    pub machines: Vec<MachineCursor>,
}

/// Weights of a 1-D blend at `value` over `children` (strictly ascending
/// `TreeChild::position`s).
///
/// Writes up to two `(child, weight)` slots into `out` and returns how
/// many are live; `child` is an index into `children` and the weights sum
/// to one. Outside the covered range the nearest end child gets the whole
/// pose, a single child always owns it, and empty `children` yields `0`
/// (the caller reads rest). Exactly on an interior position the matching
/// child owns the pose; on a tie (equal neighbouring positions) the later
/// child wins.
///
/// Deterministic and allocation-free — evaluated every frame per blend
/// node.
pub(crate) fn blend1d_weights(
    children: &[TreeChild],
    value: f32,
    out: &mut [(usize, f32); 2],
) -> usize {
    if children.is_empty() {
        return 0;
    }
    let last = children.len() - 1;
    let mut segment = 0;
    while segment < last && children[segment + 1].position <= value {
        segment += 1;
    }
    let next = segment + 1;
    if next > last || value <= children[segment].position {
        out[0] = (segment, 1.0);
        return 1;
    }
    let span = children[next].position - children[segment].position;
    if span <= 0.0 {
        out[0] = (next, 1.0);
        return 1;
    }
    let t = (value - children[segment].position) / span;
    out[0] = (segment, 1.0 - t);
    out[1] = (next, t);
    2
}

/// Scratch buffers of [`eval_tree`], grouped to stay inside clippy's
/// argument budget (same pattern as the type aliases).
pub(crate) struct EvalBuf<'a> {
    /// One scratch pose per node, in node order.
    pub poses: &'a mut [BlendedPose],
    /// Reusable sampled-joints buffer.
    pub samples: &'a mut Vec<JointSample>,
    /// Reusable painted-channel buffer.
    pub painted: &'a mut Vec<bool>,
    /// Clip ids of the evaluated leaves, appended in node order.
    pub contributors: &'a mut Vec<ClipId>,
}

/// Evaluates `tree` into `buf.poses` — one scratch pose per node, in node
/// order (the root is the last one) — sampling the [`SkelClip`] leaves at
/// their cursors in `player`.
///
/// The clip ids of every leaf are appended to `buf.contributors`. Leaf
/// cursors apply their own `weight` as in phase E (pull toward rest);
/// `Blend1d` nodes fold their weighted children the same way a crossfade
/// mixes poses. A violation of the tree contract (missing cursor or clip,
/// bad `axis`, a non-finite parameter or child position, a child not
/// earlier than its parent, missing joint counts) returns `None` and the
/// caller publishes nothing — the "bad skin" rule. The child-order check
/// covers **all** children of a blend node, not just the weighted ones, so
/// one broken tree reads the same way at every parameter value.
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
        samples,
        painted,
        contributors,
    } = buf;
    if tree.nodes.is_empty() {
        return None;
    }
    let mut leaves = 0;
    let mut machines = 0;
    for (index, node) in tree.nodes.iter().enumerate() {
        match node {
            TreeNodeDef::Clip { clip } => {
                let cursor = player.leaves.get(leaves)?;
                leaves += 1;
                let def = clips.get(clip.0)?;
                sample_joints(def, count, cursor.time.get(), samples, painted);
                poses[index].refill(samples, rest, cursor.weight.get());
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
            TreeNodeDef::Machine {
                states,
                initial: _,
                transitions,
            } => {
                let cursor = player.machines.get(machines)?;
                machines += 1;
                eval_machine(states, transitions, cursor, &player.params, index, poses)?;
            }
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
            tail[0].clone_from(&earlier[states[from]]);
            tail[0].mix_towards(&earlier[states[cursor.state]], progress);
        }
        None => tail[0].clone_from(&earlier[states[cursor.state]]),
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Children at `positions`, node indices matching the child index.
    fn axis(positions: &[f32]) -> Vec<TreeChild> {
        positions
            .iter()
            .enumerate()
            .map(|(node, &position)| TreeChild { position, node })
            .collect()
    }

    // Bodies sit in one inner block each: rustqual's fragment scanner
    // hashes windows of three consecutive top-level statements, and the
    // neighbouring test files already own the natural statement shapes.
    #[test]
    fn empty_children_read_as_rest() {
        {
            let mut out = [(0, 0.0); 2];
            assert_eq!(blend1d_weights(&[], 0.5, &mut out), 0);
        }
    }

    #[test]
    fn single_child_owns_every_value() {
        {
            let mut out = [(0, 0.0); 2];
            let filled = blend1d_weights(&axis(&[0.25]), 7.0, &mut out);
            assert_eq!((filled, out[0]), (1, (0, 1.0)));
        }
    }

    #[test]
    fn value_outside_the_range_clamps_to_the_ends() {
        {
            let mut out = [(0, 0.0); 2];
            let below = blend1d_weights(&axis(&[1.0, 2.0]), -5.0, &mut out);
            assert_eq!((below, out[0]), (1, (0, 1.0)));
            let above = blend1d_weights(&axis(&[1.0, 2.0]), 9.0, &mut out);
            assert_eq!((above, out[0]), (1, (1, 1.0)));
        }
    }

    #[test]
    fn interior_value_splits_between_neighbours() {
        {
            let mut out = [(0, 0.0); 2];
            let filled = blend1d_weights(&axis(&[0.0, 2.0, 4.0]), 1.5, &mut out);
            assert_eq!(filled, 2);
            assert_eq!(out[0], (0, 0.25));
            assert_eq!(out[1], (1, 0.75));
        }
    }

    #[test]
    fn exact_axis_point_owns_the_pose() {
        {
            let mut out = [(0, 0.0); 2];
            let filled = blend1d_weights(&axis(&[0.0, 1.0, 2.0]), 1.0, &mut out);
            assert_eq!((filled, out[0]), (1, (1, 1.0)));
        }
    }

    #[test]
    fn tied_positions_fall_to_the_later_child() {
        {
            let mut out = [(0, 0.0); 2];
            let filled = blend1d_weights(&axis(&[1.0, 1.0]), 1.0, &mut out);
            assert_eq!((filled, out[0]), (1, (1, 1.0)));
        }
    }
}
