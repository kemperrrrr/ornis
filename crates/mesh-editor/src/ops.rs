//! Edit operations: data describing an edit, never the work itself.
//!
//! [`EditOp`] values are cheap to record every frame (e.g. during a drag)
//! and are classified later by the preview/exact split: cheap regional ops
//! stay in the frame budget, topological ops go to the background worker.
//! Undo history stores these ops, not mesh snapshots.

/// Boolean combination kind for [`EditOp::Boolean`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanKind {
    /// Union of base and tool volumes.
    Union,
    /// Base volume minus the tool volume.
    Subtract,
    /// Overlap of base and tool volumes.
    Intersect,
}

/// One recorded mesh edit (intent, not execution).
#[derive(Debug, Clone)]
pub enum EditOp {
    /// Rigid transform of the whole mesh (column-major `glam::Mat4`).
    Transform {
        /// World-space transform to apply to every vertex.
        matrix: glam::Mat4,
    },
    /// Push selected faces along their normals.
    Extrude {
        /// Face indices (into `indices` triples) to push.
        faces: Vec<u32>,
        /// Push distance in engine units.
        depth: f32,
    },
    /// Chamfer selected edges (closed on the Ornis side: upstream has no
    /// 3D bevel, only 2D offset `JoinType::Bevel`).
    Bevel {
        /// Edge endpoint pairs (vertex indices).
        edges: Vec<(u32, u32)>,
        /// Chamfer width in engine units.
        width: f32,
        /// Segments per chamfer (1 = flat).
        segments: u32,
    },
    /// Midpoint subdivision levels (upstream has no crease-controlled
    /// Catmull-Clark; each level quadruples triangles).
    Subdivide {
        /// Subdivision levels to add.
        levels: u32,
    },
    /// CSG combination with a tool mesh.
    Boolean {
        /// Combination kind.
        kind: BooleanKind,
        /// Tool volume in the base mesh's local space.
        tool: crate::MeshData,
        /// Transform applied to the tool before combining.
        tool_matrix: glam::Mat4,
    },
    /// Accept the pending preview as the new base version.
    CommitExact,
    /// Drop the pending preview, keep the base version.
    CancelPreview,
}
