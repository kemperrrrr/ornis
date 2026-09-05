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
    /// 3 linear + 2 angular equality constraints, plus optional limit/motor
    /// drive on the remaining axis (`None` = free unpowered hinge).
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
        /// Travel window (rad, relative to the pose at creation).
        limit: Option<RevoluteLimit>,
        /// Velocity motor on the hinge axis.
        motor: Option<RevoluteMotor>,
    },
}

/// Angular travel window for a revolute joint, in radians relative to the
/// reference twist captured at creation. The solver blocks rotation past
/// either bound with a one-sided velocity constraint (plus a small slop);
/// the hinge moves freely inside the window.
///
/// Requires `min <= max`. A degenerate window (`min == max`) locks the axis.
#[derive(Debug, Clone, Copy)]
pub struct RevoluteLimit {
    /// Lower bound (rad, relative to the reference twist).
    pub min: f32,
    /// Upper bound (rad, relative to the reference twist).
    pub max: f32,
}

/// Velocity motor for a revolute joint: drives the hinge toward
/// `target_speed` (rad/s, signed about the hinge axis). The per-substep
/// impulse is clamped to `max_torque * sub_dt`, so a weak motor spins up
/// gradually instead of teleporting the hinge. The motor pauses while a
/// limit is violated and resumes inside the window.
#[derive(Debug, Clone, Copy)]
pub struct RevoluteMotor {
    /// Desired hinge speed (rad/s, signed about the hinge axis).
    pub target_speed: f32,
    /// Torque budget: bounds the per-substep motor impulse.
    pub max_torque: f32,
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
    /// Hinge twist at creation (rad): limits measure travel relative to
    /// this, Box2D `m_referenceAngle` style. Ignored by ball joints.
    pub reference_angle: f32,
    /// Accumulated one-sided limit impulse (warm start). Positive = lower
    /// bound active, negative = upper; zero when inside the window. The
    /// clamp logic self-corrects on side flips, so no side state is stored.
    pub acc_limit: f32,
}

impl Joint {
    pub fn new(body_a: BodyHandle, body_b: BodyHandle, kind: JointKind) -> Self {
        Self {
            body_a,
            body_b,
            kind,
            acc_lin: [0.0; 3],
            acc_ang: [0.0; 2],
            reference_angle: 0.0,
            acc_limit: 0.0,
        }
    }

    /// (limit, motor) drive of a revolute joint; (None, None) for ball.
    pub(crate) fn drive(&self) -> (Option<RevoluteLimit>, Option<RevoluteMotor>) {
        match &self.kind {
            JointKind::Revolute { limit, motor, .. } => (*limit, *motor),
            JointKind::Ball { .. } => (None, None),
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
