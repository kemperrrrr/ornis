//! Joint (constraint) definitions for the builtin physics engine (G5).
//!
//! Modeled on Box3D `spherical_joint`/`revolute_joint` and Jolt `Constraint`:
//! joints are persistent equality constraints with warm-started accumulated
//! impulses, solved as dedicated sub-solvers inside the substep loop.

use glam::Vec3;

use crate::body::BodyHandle;

/// Stable index of a joint inside its owning engine.
///
/// Like [`BodyHandle`] this is a vector index:
/// removal shifts subsequent handles.
pub type JointHandle = usize;

/// What the user supplies when creating a joint. Local anchors/axes are
/// specified in each body's frame; the joint is satisfied when the world
/// anchors coincide (and, for revolute, the world axes are parallel).
#[derive(Debug, Clone)]
pub enum JointKind {
    /// Ball-and-socket (spherical): anchor points coincide.
    /// 3 linear equality constraints along the world axes.
    Ball {
        /// Anchor point in body A's local frame.
        local_anchor_a: Vec3,
        /// Anchor point in body B's local frame.
        local_anchor_b: Vec3,
    },
    /// Revolute (hinge): ball joint + the hinge axes stay parallel,
    /// leaving exactly one rotational degree of freedom around the axis.
    /// 3 linear + 2 angular equality constraints.
    Revolute {
        /// Anchor point in body A's local frame (hinge center).
        local_anchor_a: Vec3,
        /// Anchor point in body B's local frame (hinge center).
        local_anchor_b: Vec3,
        /// Hinge axis in each body's local frame. The axes must coincide in
        /// world space when the joint is assembled (normalized on creation).
        local_axis_a: Vec3,
        /// Hinge axis in body B's local frame; see `local_axis_a`.
        local_axis_b: Vec3,
    },
}

/// A joint plus its persistent solver state (warm-start accumulators).
pub(crate) struct Joint {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub kind: JointKind,
    /// Accumulated linear impulses per world axis (X/Y/Z), reused as the
    /// warm start of the next substep — same pattern as the contact cache.
    pub acc_lin: [f32; 3],
    /// Accumulated angular impulses per constraint axis (revolute only).
    pub acc_ang: [f32; 2],
}

impl Joint {
    pub fn new(body_a: BodyHandle, body_b: BodyHandle, kind: JointKind) -> Self {
        Self {
            body_a,
            body_b,
            kind,
            acc_lin: [0.0; 3],
            acc_ang: [0.0; 2],
        }
    }

    /// Local anchor on each body, regardless of the joint kind.
    pub fn local_anchors(&self) -> (Vec3, Vec3) {
        match &self.kind {
            JointKind::Ball {
                local_anchor_a,
                local_anchor_b,
            }
            | JointKind::Revolute {
                local_anchor_a,
                local_anchor_b,
                ..
            } => (*local_anchor_a, *local_anchor_b),
        }
    }
}
