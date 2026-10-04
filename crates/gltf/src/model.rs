//! glTF scene kept as a node tree: local TRS, parents, and primitives.
//!
//! [`Model`] is the import result of [`load_slice`](crate::load_slice).
//! Nodes stay in topological order (a parent's index is always lower than
//! its child's), including nodes that carry no mesh. Primitives hang off
//! the node that owned them. World TRS is derived by [`Model::world_transform`]
//! from that chain — the same matrix product the old flattener baked into
//! each mesh entity.

use ornis_core::{Transform, UnitQuat};

use crate::geom;
use crate::{ImportStats, LoadedAnimClip, LoadedMaterial, LoadedMesh, LoadedSkelClip, LoadedSkin};

/// Index into [`Model::nodes`].
///
/// Tracks and skins address nodes with this, not with the glTF document
/// index: document order is not topological, and nodes outside the chosen
/// scene are not imported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NodeIdx(pub u32);

impl NodeIdx {
    /// Index into [`Model::nodes`] (`as usize`).
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// One glTF node: local pose, optional parent, and the primitives it owns.
///
/// `parent` is `None` for a root of the imported scene. `primitives` indexes
/// [`Model::primitives`] (empty when the node has no imported mesh). `skin`
/// indexes [`Model::skins`] when the node is a skinned mesh instance.
#[derive(Debug, Clone)]
pub struct ModelNode {
    /// glTF node name (`None` when the document omits it).
    pub name: Option<String>,
    /// Parent in [`Model::nodes`], or `None` for a scene root.
    pub parent: Option<NodeIdx>,
    /// TRS relative to [`Self::parent`].
    pub local: Transform,
    /// Indices into [`Model::primitives`], in primitive order.
    pub primitives: Vec<usize>,
    /// Index into [`Model::skins`] when this node instances a skin.
    pub skin: Option<usize>,
}

/// One imported mesh primitive, attached to the node that owned it.
///
/// The primitive does not carry a transform: pose lives on [`ModelNode`].
/// A flattened editor scene composes [`Model::world_transform`] of `node`.
#[derive(Debug, Clone)]
pub struct ModelPrimitive {
    /// Node this primitive is attached to.
    pub node: NodeIdx,
    /// Triangle soup plus optional source attributes.
    pub mesh: LoadedMesh,
    /// Scalar PBR factors plus decoded texture slots.
    pub material: LoadedMaterial,
}

/// Imported glTF scene: node tree, primitives, skins, and clips.
///
/// `nodes` is a topological order of the default scene (else the first
/// scene): for every node, `parent` is either `None` or a strictly smaller
/// index. Mesh-less nodes are included. Object and joint tracks address
/// [`NodeIdx`] into this list. [`LoadedSkin::parents`] stays in skin-joint
/// order.
///
/// `name` is the glTF scene label (`"scene"` when absent). It is not a
/// node; the editor flattener (`scene_from_model` in `ornis-assets`) copies
/// it onto a flat scene. The field sits beside the spawn-hierarchy contract
/// so that label survives.
#[derive(Debug, Clone)]
pub struct Model {
    /// glTF scene name, or `"scene"`.
    pub name: String,
    /// Scene nodes, parent before children.
    pub nodes: Vec<ModelNode>,
    /// Scene roots, in glTF scene order. Each has `parent == None`.
    pub roots: Vec<NodeIdx>,
    /// Imported mesh primitives. [`ModelNode::primitives`] indexes this.
    pub primitives: Vec<ModelPrimitive>,
    /// Resolved `skins[]`, in first-use order. [`ModelNode::skin`] links here.
    pub skins: Vec<LoadedSkin>,
    /// Skeletal clips. [`LoadedJointTrack::node`](crate::LoadedJointTrack::node)
    /// is a [`NodeIdx`]; [`LoadedJointTrack::joint`](crate::LoadedJointTrack::joint)
    /// stays the skin-order index the animation runtime samples.
    pub skel_clips: Vec<LoadedSkelClip>,
    /// Object clips. [`LoadedAnimTrack::node`](crate::LoadedAnimTrack::node)
    /// is a [`NodeIdx`], including mesh-less parents.
    pub anim_clips: Vec<LoadedAnimClip>,
    /// Primitive and skip counters.
    pub stats: ImportStats,
}

impl Model {
    /// First node whose name equals `name`, in topological order.
    ///
    /// Comparison is exact (`hand_r`, not `Hand_R`). Duplicate names keep
    /// the earlier node.
    pub fn node_by_name(&self, name: &str) -> Option<NodeIdx> {
        self.nodes.iter().enumerate().find_map(|(index, node)| {
            (node.name.as_deref() == Some(name)).then_some(NodeIdx(index as u32))
        })
    }

    /// World TRS of `node`: the parent chain's local matrices, decomposed.
    ///
    /// Composition is the matrix product the old flattener used (`parent *
    /// local`, then TRS decompose), so a nested node's result matches the
    /// world translation/rotation/scale that used to be stored on its mesh
    /// entity. An out-of-range index yields [`Transform::IDENTITY`]. A
    /// parent cycle stops at the walk limit and returns what it composed.
    pub fn world_transform(&self, node: NodeIdx) -> Transform {
        let mut chain = Vec::new();
        let mut current = Some(node);
        let mut guard = 0usize;
        while let Some(index) = current {
            if guard > self.nodes.len() {
                break;
            }
            let Some(entry) = self.nodes.get(index.index()) else {
                break;
            };
            chain.push(index);
            current = entry.parent;
            guard += 1;
        }
        if chain.is_empty() {
            return Transform::IDENTITY;
        }
        chain.reverse();
        let mut world = geom::IDENTITY;
        for index in chain {
            let local = self.nodes[index.index()].local.to_mat4().to_cols_array_2d();
            world = geom::mat_mul(&world, &local);
        }
        transform_from_decomposed(geom::decompose(&world))
    }

    /// Direct children of `node`, in model order (glTF child order).
    ///
    /// An unknown index yields an empty iterator.
    pub fn children(&self, node: NodeIdx) -> impl Iterator<Item = NodeIdx> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(move |(index, child)| {
                (child.parent == Some(node)).then_some(NodeIdx(index as u32))
            })
    }
}

/// TRS newtype from a decomposed column-major matrix.
///
/// A degenerate rotation falls back to identity, matching
/// [`UnitQuat::normalize`].
pub(crate) fn transform_from_decomposed(decomposed: ([f32; 3], [f32; 4], [f32; 3])) -> Transform {
    let (translation, rotation, scale) = decomposed;
    Transform {
        translation: glam::Vec3::from_array(translation),
        rotation: UnitQuat::normalize(glam::Quat::from_array(rotation))
            .unwrap_or(UnitQuat::IDENTITY),
        scale: glam::Vec3::from_array(scale),
    }
}
