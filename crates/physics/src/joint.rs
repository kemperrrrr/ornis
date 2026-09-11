//! Joint (constraint) definitions for the builtin physics engine (G5).
//!
//! Modeled on Box3D `spherical_joint`/`revolute_joint` and Jolt `Constraint`:
//! joints are persistent equality constraints with warm-started accumulated
//! impulses, solved as dedicated sub-solvers inside the substep loop.

use glam::{Quat, Vec3};

use crate::body::BodyHandle;

/// Stable index of a joint inside its owning engine.
///
/// Like [`BodyHandle`] this is a vector index:
/// removal shifts subsequent handles.
pub type JointHandle = usize;

/// What the user supplies when creating a joint. Local anchors/axes are
/// specified in each body's frame; the joint is satisfied when the world
/// anchors coincide (and, for revolute, the world axes are parallel).
#[derive(Debug, Clone, Copy)]
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
    /// Prismatic (slider, Box2D formulation): the anchors may separate only
    /// along the slide axis; perpendicular motion and axis misalignment are
    /// constrained, leaving translation along the axis (plus spin about it)
    /// free. 2 linear (perp basis) + 2 angular equality constraints, plus
    /// optional limit/motor drive along the axis.
    Prismatic {
        /// Anchor point in body A's local frame (slide origin).
        local_anchor_a: Vec3,
        /// Anchor point in body B's local frame (slide origin).
        local_anchor_b: Vec3,
        /// Slide axis in each body's local frame. The axes must coincide in
        /// world space when the joint is assembled (normalized on creation).
        local_axis_a: Vec3,
        /// Slide axis in body B's local frame; see `local_axis_a`.
        local_axis_b: Vec3,
        /// Travel window (m, relative to the assembly separation).
        limit: Option<PrismaticLimit>,
        /// Velocity motor along the slide axis.
        motor: Option<PrismaticMotor>,
    },
    /// Fixed (weld): the two bodies keep their assembly transform rigidly.
    /// 3 linear + 3 angular equality constraints — a ball joint with the
    /// rotation locked. Cheaper and more stable than freezing a stack with
    /// contacts; breaks never (no breakable extension — document if added).
    Fixed {
        /// Anchor point in body A's local frame (weld origin).
        local_anchor_a: Vec3,
        /// Anchor point in body B's local frame (weld origin).
        local_anchor_b: Vec3,
    },
    /// Distance (rigid rod, Box2D `b2DistanceJoint` with zero
    /// frequency/damping): the anchor separation keeps its assembly length.
    /// 1 linear equality constraint along the anchor delta axis; rotation
    /// stays fully free on both ends (a rod with ball ends, not a weld).
    Distance {
        /// Anchor point in body A's local frame (rod end).
        local_anchor_a: Vec3,
        /// Anchor point in body B's local frame (rod end).
        local_anchor_b: Vec3,
    },
    /// Wheel (suspension, Box2D `b2WheelJoint` formulation): a prismatic
    /// slide along the suspension axis with a spring (frequency/damping)
    /// instead of a rigid drive, plus free spin about a designated axle
    /// axis with an optional motor. 2 linear (perp basis) + 2 angular
    /// equality constraints (spin about the suspension axis and the third
    /// axis are locked, spin about the axle is free).
    Wheel {
        /// Anchor point in body A's local frame (suspension origin).
        local_anchor_a: Vec3,
        /// Anchor point in body B's local frame (suspension origin).
        local_anchor_b: Vec3,
        /// Suspension axis in each body's local frame (normalized and
        /// orthogonalized against the axle on creation).
        local_suspension_a: Vec3,
        /// Suspension axis in body B's local frame.
        local_suspension_b: Vec3,
        /// Spin axle in each body's local frame (must be non-parallel to
        /// the suspension; orthogonalized on creation).
        local_axle_a: Vec3,
        /// Spin axle in body B's local frame.
        local_axle_b: Vec3,
        /// Spring parameters (Box2D frequency/damping semantics).
        suspension: WheelSuspension,
        /// Velocity motor about the axle (`None` = free-spinning wheel).
        motor: Option<RevoluteMotor>,
    },
    /// Gear (Box2D `b2GearJoint` formulation): couples the coordinates of
    /// two existing revolute/prismatic joints,
    /// `coord_a + ratio * coord_b = const` (angles in rad, slides in m —
    /// mixed units are the driver's responsibility, as in Box2D). The
    /// constant is captured at creation. Holds no bodies of its own:
    /// `Joint::body_a/body_b` mirror the first bodies of the two referenced
    /// joints for the sleep-skip heuristic; island union resolves all four
    /// bodies through the references.
    Gear {
        /// Index of the first coordinated joint in the engine's joint list.
        joint_a: JointHandle,
        /// Index of the second coordinated joint.
        joint_b: JointHandle,
        /// Transmission ratio (`coord_a + ratio * coord_b = const`).
        ratio: f32,
    },
    /// Six-DOF generic (Jolt `SixDOFConstraint` idea, axis-aligned): each of
    /// the 3 linear and 3 angular axes in body A's assembly frame is
    /// independently free, locked, or limited. Locked axes reuse the
    /// equality iterations, limited linear axes the one-sided drive;
    /// limited ANGULAR axes measure per-axis twist about the assembly frame
    /// (an axis-twist approximation, not a true swing cone — documented).
    /// All-locked degenerates to [`JointKind::Fixed`].
    SixDof {
        /// Anchor point in body A's local frame (joint origin).
        local_anchor_a: Vec3,
        /// Anchor point in body B's local frame (joint origin).
        local_anchor_b: Vec3,
        /// Per-axis linear configuration in body A's assembly frame.
        linear: [AxisConfig; 3],
        /// Per-axis angular configuration in body A's assembly frame.
        angular: [AxisConfig; 3],
    },
}

/// Spring parameters of a [`JointKind::Wheel`] suspension, Box2D
/// `b2WheelJoint` semantics: `frequency_hz` is the suspension resonance,
/// `damping_ratio` the dimensionless damping (1 = critically damped).
#[derive(Debug, Clone, Copy)]
pub struct WheelSuspension {
    /// Suspension resonance frequency (Hz, must be > 0 for a live spring).
    pub frequency_hz: f32,
    /// Dimensionless damping ratio (0 = undamped, 1 = critical).
    pub damping_ratio: f32,
}

/// Per-axis configuration of a [`JointKind::SixDof`] joint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AxisConfig {
    /// Axis moves freely (no constraint).
    Free,
    /// Axis is locked (equality constraint).
    Locked,
    /// Axis travels inside `[min, max]` measured from the assembly pose
    /// (linear: meters along the axis; angular: radians of twist about the
    /// axis). Requires `min <= max`; degenerate (`min == max`) locks.
    Limited {
        /// Lower bound (m or rad, relative to the assembly pose).
        min: f32,
        /// Upper bound (m or rad, relative to the assembly pose).
        max: f32,
    },
}

/// Linear travel window for a prismatic joint, in meters relative to the
/// anchor separation captured at creation. Same one-sided velocity-block
/// semantics as [`RevoluteLimit`]; requires `min <= max`.
#[derive(Debug, Clone, Copy)]
pub struct PrismaticLimit {
    /// Lower bound (m, relative to the assembly separation).
    pub min: f32,
    /// Upper bound (m, relative to the assembly separation).
    pub max: f32,
}

/// Velocity motor along a prismatic joint's slide axis. Same
/// force-clamped target-speed semantics as [`RevoluteMotor`].
#[derive(Debug, Clone, Copy)]
pub struct PrismaticMotor {
    /// Desired slide speed (m/s, signed about the slide axis).
    pub target_speed: f32,
    /// Force budget: bounds the per-substep motor impulse.
    pub max_force: f32,
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
    /// Accumulated angular impulses per constraint axis. Revolute and wheel
    /// joints use slots 0..2 (hinge tangents / lock axes); a fixed joint
    /// uses all three (world-axis locks).
    pub acc_ang: [f32; 3],
    /// Hinge twist at creation (rad): limits measure travel relative to
    /// this, Box2D `m_referenceAngle` style. Revolute measures the hinge
    /// twist, wheel the axle twist. Ignored by other joints.
    pub reference_angle: f32,
    /// Anchor separation along the slide axis at creation (m): prismatic
    /// limits and the wheel spring measure travel relative to this.
    /// Ignored by other joints.
    pub reference_length: f32,
    /// Anchor distance at creation (m): the distance-rod rest length, or
    /// the gear constraint constant (`coord_a + ratio * coord_b`).
    /// Ignored by other joints.
    pub reference_distance: f32,
    /// Relative orientation at creation (`qa^-1 * qb`): fixed, wheel and
    /// six-DOF angular locks measure drift relative to this.
    /// Ignored by other joints.
    pub reference_quat: Quat,
    /// Anchor separation at creation, in body A's local frame
    /// (`qa^-1 * ((pb + rb) - (pa + ra))`): fixed, wheel and six-DOF
    /// locked-axis position steps measure drift relative to this instead
    /// of pulling offset assemblies into coincidence. Legacy
    /// ball/revolute/prismatic joints assemble coincidently and ignore it.
    pub reference_anchor_delta: Vec3,
    /// Accumulated one-sided limit impulse (warm start). Positive = lower
    /// bound active, negative = upper; zero when inside the window. The
    /// clamp logic self-corrects on side flips, so no side state is stored.
    /// Shared by revolute/prismatic drives and the wheel spring (one
    /// one-sided axis per joint at most).
    pub acc_limit: f32,
    /// Accumulated rod-constraint impulse (distance joints only).
    pub acc_dist: f32,
    /// Accumulated gear-constraint impulse (gear joints only).
    pub acc_gear: f32,
    /// One-sided accumulators of six-DOF limited axes: slots 0..3 linear
    /// (X/Y/Z), 3..6 angular. Free/locked axes never touch these.
    pub acc_6dof: [f32; 6],
}

impl Joint {
    pub fn new(body_a: BodyHandle, body_b: BodyHandle, kind: JointKind) -> Self {
        Self {
            body_a,
            body_b,
            kind,
            acc_lin: [0.0; 3],
            acc_ang: [0.0; 3],
            reference_angle: 0.0,
            reference_length: 0.0,
            reference_distance: 0.0,
            reference_quat: Quat::IDENTITY,
            reference_anchor_delta: Vec3::ZERO,
            acc_limit: 0.0,
            acc_dist: 0.0,
            acc_gear: 0.0,
            acc_6dof: [0.0; 6],
        }
    }

    /// (limit, motor) drive of a revolute joint; (None, None) for the rest.
    pub(crate) fn drive(&self) -> (Option<RevoluteLimit>, Option<RevoluteMotor>) {
        match &self.kind {
            JointKind::Revolute { limit, motor, .. } => (*limit, *motor),
            JointKind::Ball { .. }
            | JointKind::Prismatic { .. }
            | JointKind::Fixed { .. }
            | JointKind::Distance { .. }
            | JointKind::Wheel { .. }
            | JointKind::Gear { .. }
            | JointKind::SixDof { .. } => (None, None),
        }
    }

    /// (limit, motor) drive of a prismatic joint; (None, None) otherwise.
    pub(crate) fn prismatic_drive(&self) -> (Option<PrismaticLimit>, Option<PrismaticMotor>) {
        match &self.kind {
            JointKind::Prismatic { limit, motor, .. } => (*limit, *motor),
            JointKind::Ball { .. }
            | JointKind::Revolute { .. }
            | JointKind::Fixed { .. }
            | JointKind::Distance { .. }
            | JointKind::Wheel { .. }
            | JointKind::Gear { .. }
            | JointKind::SixDof { .. } => (None, None),
        }
    }

    /// Slide axis of a prismatic joint in each local frame; None otherwise.
    pub(crate) fn prismatic_axes(&self) -> Option<(Vec3, Vec3)> {
        match &self.kind {
            JointKind::Prismatic {
                local_axis_a,
                local_axis_b,
                ..
            } => Some((*local_axis_a, *local_axis_b)),
            JointKind::Ball { .. }
            | JointKind::Revolute { .. }
            | JointKind::Fixed { .. }
            | JointKind::Distance { .. }
            | JointKind::Wheel { .. }
            | JointKind::Gear { .. }
            | JointKind::SixDof { .. } => None,
        }
    }

    /// Suspension + axle axes of a wheel joint in body A's local frame plus
    /// its spring and motor; None otherwise.
    pub(crate) fn wheel_drive(
        &self,
    ) -> Option<(Vec3, Vec3, WheelSuspension, Option<RevoluteMotor>)> {
        match &self.kind {
            JointKind::Wheel {
                local_suspension_a,
                local_axle_a,
                suspension,
                motor,
                ..
            } => Some((*local_suspension_a, *local_axle_a, *suspension, *motor)),
            _ => None,
        }
    }

    /// (linear, angular) axis configs of a six-DOF joint; None otherwise.
    pub(crate) fn sixdof_config(&self) -> Option<([AxisConfig; 3], [AxisConfig; 3])> {
        match &self.kind {
            JointKind::SixDof {
                linear, angular, ..
            } => Some((*linear, *angular)),
            _ => None,
        }
    }

    /// Local anchor on each body. Gear joints hold no anchors — returns
    /// zeros (they coordinate other joints instead of constraining bodies).
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
            }
            | JointKind::Prismatic {
                local_anchor_a,
                local_anchor_b,
                ..
            }
            | JointKind::Fixed {
                local_anchor_a,
                local_anchor_b,
            }
            | JointKind::Distance {
                local_anchor_a,
                local_anchor_b,
            }
            | JointKind::Wheel {
                local_anchor_a,
                local_anchor_b,
                ..
            }
            | JointKind::SixDof {
                local_anchor_a,
                local_anchor_b,
                ..
            } => (*local_anchor_a, *local_anchor_b),
            JointKind::Gear { .. } => (Vec3::ZERO, Vec3::ZERO),
        }
    }
}
