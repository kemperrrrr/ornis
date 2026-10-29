//! Animation-tree model (`docs/animation-design.md` §7): cold authoring
//! graphs ([`SkelTree`]) and hot cursor state ([`SkelTreePlayer`]).
//!
//! The model keeps the phase-E rules: deterministic and stateless per
//! frame (only cursor clocks mutate), and evaluation order is the node
//! order (`Blend1d` children reference earlier nodes, the last node is
//! the root). The per-frame fold itself lives in [`crate::tree_eval`].

use crate::{ClipId, Seconds, SkelPlayer};

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

/// One input of a [`TreeNodeDef::Blend2d`] or [`TreeNodeDef::Freeform2d`]
/// node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeChild2d {
    /// Position of the input on the node's two parameter axes. For
    /// [`TreeNodeDef::Blend2d`] the `children` form a complete row-major
    /// lattice: rows share `position[1]`, columns share `position[0]`,
    /// and both ascend strictly along the lattice. For
    /// [`TreeNodeDef::Freeform2d`] the positions are arbitrary distinct
    /// points — no lattice, at most [`FREEFORM_MAX_POINTS`] of them.
    pub position: [f32; 2],
    /// Index of the child node in [`SkelTree::nodes`], strictly smaller
    /// than the index of the blend node itself.
    pub node: usize,
}

/// Cap of one [`TreeNodeDef::Freeform2d`] input set: the triangulation
/// folds in fixed buffers on the hot path, and authored blend spaces
/// (speed × direction and friends) stay far below this. More points —
/// the tree is refused at install and reads as bad skin at eval.
pub const FREEFORM_MAX_POINTS: usize = 16;

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
    /// 2-D blend of `children` over parameters `axes` (x, y): the
    /// corners of the containing lattice cell carry bilinear weights
    /// summing to one; outside the covered lattice the nearest edge or
    /// corner owns the pose (`blend2d_weights`).
    Blend2d {
        /// Two distinct indices into [`SkelTree::params`] (x, y) that
        /// drive this node.
        axes: [usize; 2],
        /// Inputs on a complete row-major lattice ([`TreeChild2d`]);
        /// empty reads as rest.
        children: Vec<TreeChild2d>,
    },
    /// 2-D **freeform** blend of `children` over parameters `axes`
    /// (x, y): the inputs sit at arbitrary distinct points of the plane
    /// (e.g. speed × direction) — no lattice. The points are
    /// triangulated once at install into a [`FreeformCache`]
    /// (deterministic empty-circumcircle set, no allocations;
    /// cocircular sets tie-break to one fixed diagonal by a symbolic
    /// lexicographic perturbation — a geometric SoS rule, independent
    /// of the arrival order — so the triangles partition the hull) and
    /// the containing triangle's corners carry
    /// barycentric weights summing to one (the hot fold reads the cache
    /// triangle by triangle); the
    /// weights are continuous across shared triangle edges. Exactly on
    /// a hull edge or corner the edge or corner owns the pose; outside
    /// the convex hull the value clamps to its nearest hull segment
    /// (the nearest edge carries the pose — the same «outside = nearest
    /// owns» rule as [`TreeNodeDef::Blend2d`]); a collinear set
    /// degenerates to its hull segment and a one-point set is that
    /// point. Empty reads as rest. At most [`FREEFORM_MAX_POINTS`]
    /// distinct points.
    Freeform2d {
        /// Two distinct indices into [`SkelTree::params`] (x, y) that
        /// drive this node.
        axes: [usize; 2],
        /// Inputs at arbitrary distinct positions ([`TreeChild2d`]);
        /// empty reads as rest.
        children: Vec<TreeChild2d>,
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
    /// Selector over earlier nodes: stands on the child chosen by
    /// rounding parameter `param` to the nearest integer (halves away
    /// from zero, clamped to the covered range — [`switch_selection`]),
    /// and crossfades between selections with
    /// [`SkelFade`](crate::SkelFade) mechanics (an in-flight crossfade is
    /// never interrupted — a new selection re-checks once it completes).
    /// Empty `children` is a broken tree (bad skin): the node owns a
    /// cursor, like [`TreeNodeDef::Machine`].
    Switch {
        /// Index into [`SkelTree::params`] that drives this node.
        param: usize,
        /// Candidate indices into [`SkelTree::nodes`], all strictly
        /// smaller than this node's index; non-empty.
        children: Vec<usize>,
        /// Crossfade length in seconds, [`SkelFade`](crate::SkelFade)
        /// mechanics: linear progress, the outgoing child's cursors keep
        /// stepping; non-positive cuts at once. Must be finite.
        fade: Seconds,
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

/// Runtime state of one [`TreeNodeDef::Switch`] — one entry of
/// [`SkelTreePlayer::switches`], in switch-node order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwitchCursor {
    /// Index into the node's `children`: where the switch stands, or
    /// where an in-flight crossfade is heading.
    pub child: usize,
    /// Child being left while a crossfade flies; `None` at rest.
    pub from: Option<usize>,
    /// Time spent in the in-flight crossfade.
    pub elapsed: Seconds,
    /// Length of the in-flight crossfade; non-positive cuts at once.
    pub duration: Seconds,
}

/// Child selection of a [`TreeNodeDef::Switch`]: `value` rounded to the
/// nearest integer (halves away from zero), clamped to `0..len`.
/// `len == 0` reads as `0` — the eval contract rejects empty `children`
/// before the index is ever used.
pub fn switch_selection(value: f32, len: usize) -> usize {
    let rounded = value.round();
    if rounded <= 0.0 || len == 0 {
        0
    } else if rounded >= len as f32 {
        len - 1
    } else {
        rounded as usize
    }
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
/// for `machines[i]` and the i-th [`TreeNodeDef::Machine`], and for
/// `switches[i]` and the i-th [`TreeNodeDef::Switch`]: a lane whose counts
/// diverge steps the matching prefix and reads as bad skin (`eval_tree`
/// demands one cursor per machine or switch node).
/// Triangle capacity of one [`FreeformCache`] — the Delaunay
/// triangulation of `n` points has at most `2n − 5` triangles, so
/// [`FREEFORM_MAX_POINTS`] points never need more.
pub const FREEFORM_MAX_TRIANGLES: usize = 2 * FREEFORM_MAX_POINTS - 5;

/// The surviving Delaunay triangles of one [`TreeNodeDef::Freeform2d`]
/// node — the install-time triangulation of its points
/// ([`freeform_cache`](crate::freeform::freeform_cache)) that the hot
/// fold reads triangle by triangle. The hot path never re-triangulates:
/// the node's points are fixed at install, and every `set_skel_tree`
/// install rebuilds the cache — a raw [`SkelTree`] lane write leaves it
/// untouched.
///
/// Contract: replace a tree only through `set_skel_tree` (the typed
/// facade). A raw [`SkelTree`] lane write does not rebuild the cache —
/// with another point count the frame reads as bad skin (`None` in both
/// folds), with the same count it answers from the stale triangulation,
/// which is the caller's bug. Build the cache only from the node's own
/// children: the frame trusts the stored triples completely.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FreeformCache {
    pub(crate) kept: [[usize; 3]; FREEFORM_MAX_TRIANGLES],
    pub(crate) count: usize,
    /// The `children.len()` this cache was built for — a frame whose
    /// children disagree reads as bad skin, never past the end.
    pub(crate) n_points: usize,
}

impl FreeformCache {
    /// The surviving `(child, child, child)` triples, in the
    /// triangulation's deterministic scan order.
    pub fn triples(&self) -> &[[usize; 3]] {
        &self.kept[..self.count]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct SkelTreePlayer {
    /// Live parameter values, aligned with [`SkelTree::params`].
    pub params: Vec<f32>,
    /// Cursors of the [`TreeNodeDef::Clip`] leaves, in node order.
    pub leaves: Vec<SkelPlayer>,
    /// Cursors of the [`TreeNodeDef::Machine`] nodes, in node order.
    pub machines: Vec<MachineCursor>,
    /// Cursors of the [`TreeNodeDef::Switch`] nodes, in node order.
    pub switches: Vec<SwitchCursor>,
    /// Install-time triangulations of the [`TreeNodeDef::Freeform2d`]
    /// nodes, in node order — one [`FreeformCache`] per freeform node;
    /// a lane whose count diverges reads as bad skin like the cursors
    /// above. Rebuilt by every `set_skel_tree` install — the points of
    /// an installed node never change, and a raw [`SkelTree`] lane
    /// write leaves the caches stale (the [`FreeformCache`] contract:
    /// the same point count answers from the stale triangulation — the
    /// caller's bug; another count reads as bad skin). Each cache is
    /// built from the same `children` the node carries.
    pub freeforms: Vec<FreeformCache>,
}
