//! Joint (constraint) definitions for the sequential-impulse physics engine (G5).
//!
//! Modeled on Box3D `spherical_joint`/`revolute_joint` and Jolt `Constraint`:
//! joints are persistent equality constraints with warm-started accumulated
//! impulses, solved as dedicated sub-solvers inside the substep loop.
//!
//! P6 adds Rapier-style rope/spring joints and a generalized motor model
//! (`MotorModel`/`JointMotor`, after Rapier `motor_model.rs` + `JointMotor`):
//! velocity, position and servo drives with force- or acceleration-based
//! spring interpretation, shared by every engine through one drive equation
//! (see [`JointMotor::servo_impulse`]) instead of per-joint duplicates.
//!
//! # Multibody: maximal coordinates by default, Featherstone opt-in (R6)
//!
//! The P6 verdict ("no reduced-coordinate solver; chains from the existing
//! joints suffice") was overturned by the owner as wrong (Box3D-audit
//! program, R6). Chains composed from [`JointKind`]s in the SI/AVBD/XPBD
//! engines remain the default and the only path that couples articulated
//! bodies with contacts, islands and sleep. For drift-free, exact tree
//! dynamics of free/loaded mechanisms there is now the opt-in
//! reduced-coordinate engine [`crate::articulated::FeatherstoneEngine`]
//! (ABA, revolute/fixed/floating-base joints, no contacts in v1 — see its
//! module docs for the full scope).

use glam::{Quat, Vec3};

use crate::body::BodyHandle;
use crate::engine::joints::hinge_twist;
use crate::math::orthogonalize_axle;

/// Stable index of a joint inside its owning engine.
///
/// Like [`BodyHandle`](crate::body::BodyHandle) this is a dense vector index:
/// removal rebuilds the dense map (see `rebuild_joints`).
///
/// Newtype over `u32` so joint handles never mix with body handles at the
/// type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JointHandle(u32);

impl JointHandle {
    /// Wraps a raw `u32` joint index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` joint index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Joint index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for JointHandle {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for JointHandle {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<JointHandle> for u32 {
    fn from(h: JointHandle) -> Self {
        h.0
    }
}

impl From<JointHandle> for usize {
    fn from(h: JointHandle) -> Self {
        h.0 as usize
    }
}

/// Local joint index inside an [`AvbdEngine`](crate::avbd::AvbdEngine).
///
/// Same `u32` representation as [`JointHandle`], but a different handle
/// space: the AVBD engine's dense joint table, not the
/// [`Engine`](crate::Engine) global registry. Convert explicitly at the
/// split-registry boundary (`split.rs`); lossless `u32` reinterpretation,
/// bit-identical behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalAvbdJoint(u32);

impl LocalAvbdJoint {
    /// Wraps a raw `u32` local joint index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` local joint index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Local joint index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for LocalAvbdJoint {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for LocalAvbdJoint {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<LocalAvbdJoint> for u32 {
    fn from(h: LocalAvbdJoint) -> Self {
        h.0
    }
}

impl From<LocalAvbdJoint> for usize {
    fn from(h: LocalAvbdJoint) -> Self {
        h.0 as usize
    }
}

impl From<LocalAvbdJoint> for JointHandle {
    fn from(h: LocalAvbdJoint) -> Self {
        Self::from_raw(h.0)
    }
}

impl From<JointHandle> for LocalAvbdJoint {
    fn from(h: JointHandle) -> Self {
        Self::from_raw(h.as_u32())
    }
}

/// Local joint index inside a [`SequentialImpulseEngine`](crate::engine::SequentialImpulseEngine).
///
/// Same contract as [`LocalAvbdJoint`]: the SI engine's dense joint table,
/// not the global registry. Explicit conversions only, at the split boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalSiJoint(u32);

impl LocalSiJoint {
    /// Wraps a raw `u32` local joint index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` local joint index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Local joint index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for LocalSiJoint {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for LocalSiJoint {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<LocalSiJoint> for u32 {
    fn from(h: LocalSiJoint) -> Self {
        h.0
    }
}

impl From<LocalSiJoint> for usize {
    fn from(h: LocalSiJoint) -> Self {
        h.0 as usize
    }
}

impl From<LocalSiJoint> for JointHandle {
    fn from(h: LocalSiJoint) -> Self {
        Self::from_raw(h.0)
    }
}

impl From<JointHandle> for LocalSiJoint {
    fn from(h: JointHandle) -> Self {
        Self::from_raw(h.as_u32())
    }
}

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
    /// Rope (Rapier `RopeJoint` idea): a distance INEQUALITY — the anchor
    /// separation may shrink freely but never exceeds `max_distance`.
    /// 1 one-sided linear constraint along the anchor delta axis; slack is
    /// free (a slack rope pushes nothing), tension pulls like a rod.
    /// Unlike [`JointKind::Distance`] the limit is explicit, not captured
    /// from the assembly pose, so a rope is always assembled slack-or-taut
    /// and never shorter than its own maximum.
    Rope {
        /// Anchor point in body A's local frame (rope end).
        local_anchor_a: Vec3,
        /// Anchor point in body B's local frame (rope end).
        local_anchor_b: Vec3,
        /// Maximum anchor separation in meters (must be finite and `> 0`).
        max_distance: f32,
    },
    /// Spring (Rapier `SpringJoint` idea): a compliant distance row pulling
    /// the anchors toward the motor's target position (the rest length)
    /// with stiffness/damping from [`JointMotor`]. A velocity-kind motor is
    /// a pure damper (shock absorber without centering); position/servo
    /// kinds center on the rest length. Solved implicitly by default
    /// (unconditionally stable, wheel-spring discipline); [`SpringIntegration::Explicit`]
    /// selects semi-explicit Euler (conditionally stable — see its docs).
    Spring {
        /// Anchor point in body A's local frame (spring end).
        local_anchor_a: Vec3,
        /// Anchor point in body B's local frame (spring end).
        local_anchor_b: Vec3,
        /// Spring-damper drive: rest length, stiffness, damping, force
        /// budget and force/acceleration interpretation.
        motor: JointMotor,
        /// Implicit (stable) or explicit (cheap, bounded) integration.
        integration: SpringIntegration,
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
    /// Free motor (Box3D `b3MotorJoint` spirit): drives body B toward the
    /// target linear + angular velocity RELATIVE to body A, with no locked
    /// axes and no anchors. A static body A is the world frame (the
    /// Box3D "motor vs ground" setup); two dynamics drive their relative
    /// motion (chaser/drone disciplines).
    ///
    /// Targets are world-frame vectors (m/s and rad/s of B-minus-A). The
    /// budgets clamp the ACCUMULATED impulse (`±max_force·dt`,
    /// `±max_torque·dt`, limit-row discipline): a weak motor ramps up
    /// gradually and under-delivers against overload instead of snapping
    /// to the target — never a hard velocity clamp.
    ///
    /// `correction` is the Baumgarte share of the assembly-pose error
    /// removed per position pass (`0..=1`, default
    /// [`DEFAULT_MOTOR_CORRECTION`], Box2D `b2MotorJointDef`
    /// `correctionFactor` parity): `0` is a pure velocity drive with no
    /// pose memory, `> 0` weakly pulls the assembly transform back (drift
    /// control, not a pose hold — under a sustained velocity drive the
    /// pose trails by ~`v·h/correction` per substep, millimetric at
    /// default). Pose holding is the separate [`JointMotor::servo`] drive
    /// (fixed setpoint + stiffness); the motor tracks a VELOCITY, the
    /// servo a POSE — do not mix them.
    ///
    /// Solver coverage: SI runs 6 accumulated velocity rows plus the weak
    /// position pull; AVBD/XPBD run the velocity drive only (no positional
    /// rows — correction is SI-only there). Cross-solver has no structural
    /// row, so a cross motor is explicitly `Unsupported` (pin both ends to
    /// one solver). Boundary with [`KinematicMover`](crate::KinematicMover):
    /// the motor drives DYNAMIC bodies through impulses, the mover carries
    /// KINEMATIC bodies through displacement — different bodies, no overlap.
    Motor {
        /// Desired relative linear velocity (m/s, B-minus-A, world frame).
        linear_target: Vec3,
        /// Desired relative angular velocity (rad/s, B-minus-A, world frame).
        angular_target: Vec3,
        /// Linear force budget (N): clamps the accumulated linear impulse.
        max_force: f32,
        /// Angular torque budget (N·m): clamps the accumulated angular impulse.
        max_torque: f32,
        /// Assembly-pose pull per position pass (`0..=1`, SI only).
        correction: f32,
    },
}

/// Default motor position-correction share (Box2D `b2MotorJointDef`
/// `correctionFactor` parity: `0.3`).
pub const DEFAULT_MOTOR_CORRECTION: f32 = 0.3;

/// Drive parameters of a [`JointKind::Motor`] joint, by value for the
/// solver rows and sleep votes (the spec variant stays the canonical
/// store; see [`JointKind::motor_drive`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotorDrive {
    /// Desired relative linear velocity (m/s, B-minus-A, world frame).
    pub linear_target: Vec3,
    /// Desired relative angular velocity (rad/s, B-minus-A, world frame).
    pub angular_target: Vec3,
    /// Linear force budget (N): clamps the accumulated linear impulse.
    pub max_force: f32,
    /// Angular torque budget (N·m): clamps the accumulated angular impulse.
    pub max_torque: f32,
    /// Assembly-pose pull per position pass (`0..=1`, SI only).
    pub correction: f32,
}

impl MotorDrive {
    /// Checked constructor: `None` unless both targets are finite, both
    /// budgets are finite and `>= 0`, and `correction` is finite in
    /// `0..=1` (a pull share outside the unit range would over/anti-correct).
    pub fn try_new(
        linear_target: Vec3,
        angular_target: Vec3,
        max_force: f32,
        max_torque: f32,
        correction: f32,
    ) -> Option<Self> {
        let m = Self {
            linear_target,
            angular_target,
            max_force,
            max_torque,
            correction,
        };
        m.check().then_some(m)
    }

    /// Validity predicate behind [`MotorDrive::try_new`].
    pub fn check(&self) -> bool {
        self.linear_target.is_finite()
            && self.angular_target.is_finite()
            && self.max_force.is_finite()
            && self.max_force >= 0.0
            && self.max_torque.is_finite()
            && self.max_torque >= 0.0
            && self.correction.is_finite()
            && (0.0..=1.0).contains(&self.correction)
    }

    /// Whether this drive disturbs sleep (a live velocity vote for the
    /// engine sleep heuristics): any budgeted nonzero target. A pure
    /// correction pull (`correction > 0`, zero targets) is NOT a vote —
    /// the assembly-residual check covers the unsettled case and a settled
    /// motor must sleep (same split as the spring rest residual).
    pub fn keeps_awake(&self) -> bool {
        (self.max_force > 0.0 && self.linear_target != Vec3::ZERO)
            || (self.max_torque > 0.0 && self.angular_target != Vec3::ZERO)
    }

    /// Budget-clamped deadbeat linear impulse for one drive axis: the
    /// exact impulse driving the axis rate to the target
    /// (`rate_error / inv_eff_mass`, clamped to `±max_force·dt`). The
    /// single-shot twin of the SI accumulated rows, shared by the
    /// AVBD/XPBD velocity passes; non-positive `dt`, budget or mass
    /// yields zero (never NaN).
    pub fn linear_impulse(&self, rate_error: f32, inv_eff_mass: f32, dt: f32) -> f32 {
        if dt <= 0.0 || self.max_force <= 0.0 || inv_eff_mass <= 0.0 {
            return 0.0;
        }
        (rate_error / inv_eff_mass).clamp(-self.max_force * dt, self.max_force * dt)
    }

    /// Budget-clamped deadbeat angular impulse for one drive axis (same
    /// equation against `max_torque`).
    pub fn angular_impulse(&self, rate_error: f32, inv_eff_mass: f32, dt: f32) -> f32 {
        if dt <= 0.0 || self.max_torque <= 0.0 || inv_eff_mass <= 0.0 {
            return 0.0;
        }
        (rate_error / inv_eff_mass).clamp(-self.max_torque * dt, self.max_torque * dt)
    }
}

impl JointKind {
    /// Checked gear coupling: `None` unless `ratio` is finite and non-zero
    /// (a zero ratio would freeze `coord_a` at its assembly constant).
    pub fn gear_checked(joint_a: JointHandle, joint_b: JointHandle, ratio: f32) -> Option<Self> {
        ornis_core::units::GearRatio::try_new(ratio).map(|r| Self::Gear {
            joint_a,
            joint_b,
            ratio: r.get(),
        })
    }

    /// Typed gear coupling over [`ornis_core::units::GearRatio`] (infallible:
    /// the ratio is valid by construction).
    pub fn gear_checked_units(
        joint_a: JointHandle,
        joint_b: JointHandle,
        ratio: ornis_core::units::GearRatio,
    ) -> Self {
        Self::Gear {
            joint_a,
            joint_b,
            ratio: ratio.get(),
        }
    }

    /// Transmission ratio of a [`JointKind::Gear`] as
    /// [`ornis_core::units::GearRatio`], or `None` for other joints and for
    /// degenerate stored ratios (legacy raw construction paths).
    pub fn gear_ratio_units(&self) -> Option<ornis_core::units::GearRatio> {
        match self {
            Self::Gear { ratio, .. } => ornis_core::units::GearRatio::try_new(*ratio),
            _ => None,
        }
    }

    /// Checked rope: `None` unless `max_distance` is finite and `> 0`
    /// (a rope must allow some separation).
    pub fn rope_checked(
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        max_distance: ornis_core::units::Meters,
    ) -> Option<Self> {
        if max_distance.get().is_finite() && max_distance.get() > 0.0 {
            Some(Self::Rope {
                local_anchor_a,
                local_anchor_b,
                max_distance: max_distance.get(),
            })
        } else {
            None
        }
    }

    /// Checked spring: `None` unless the motor is valid
    /// (see [`JointMotor::check`]).
    pub fn spring_checked(
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        motor: JointMotor,
        integration: SpringIntegration,
    ) -> Option<Self> {
        if motor.check() {
            Some(Self::Spring {
                local_anchor_a,
                local_anchor_b,
                motor,
                integration,
            })
        } else {
            None
        }
    }

    /// Maximum separation of a [`JointKind::Rope`] in meters,
    /// or `None` for other joints.
    pub fn rope_max_units(&self) -> Option<ornis_core::units::Meters> {
        match self {
            Self::Rope { max_distance, .. } => Some(ornis_core::units::Meters::new(*max_distance)),
            _ => None,
        }
    }

    /// Spring motor of a [`JointKind::Spring`], or `None` for other joints.
    pub fn spring_motor(&self) -> Option<JointMotor> {
        match self {
            Self::Spring { motor, .. } => Some(*motor),
            _ => None,
        }
    }

    /// Checked motor: `None` unless the drive validates (see
    /// [`MotorDrive::check`]).
    pub fn motor_checked(
        linear_target: Vec3,
        angular_target: Vec3,
        max_force: f32,
        max_torque: f32,
        correction: f32,
    ) -> Option<Self> {
        MotorDrive::try_new(
            linear_target,
            angular_target,
            max_force,
            max_torque,
            correction,
        )
        .map(
            |MotorDrive {
                 linear_target,
                 angular_target,
                 max_force,
                 max_torque,
                 correction,
             }| Self::Motor {
                linear_target,
                angular_target,
                max_force,
                max_torque,
                correction,
            },
        )
    }

    /// Drive parameters of a [`JointKind::Motor`], or `None` for other joints.
    pub fn motor_drive(&self) -> Option<MotorDrive> {
        match *self {
            Self::Motor {
                linear_target,
                angular_target,
                max_force,
                max_torque,
                correction,
            } => Some(MotorDrive {
                linear_target,
                angular_target,
                max_force,
                max_torque,
                correction,
            }),
            _ => None,
        }
    }
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

impl WheelSuspension {
    /// Checked constructor: `None` unless the frequency is a live spring
    /// (`finite && > 0`) and the damping ratio is finite and `>= 0`.
    pub fn try_new(frequency: ornis_core::units::Hertz, damping_ratio: f32) -> Option<Self> {
        if !frequency.is_live() || !damping_ratio.is_finite() || damping_ratio < 0.0 {
            return None;
        }
        Some(Self {
            frequency_hz: frequency.get(),
            damping_ratio,
        })
    }

    /// Resonance frequency in hertz.
    pub fn frequency_units(&self) -> ornis_core::units::Hertz {
        ornis_core::units::Hertz::new(self.frequency_hz)
    }
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

impl PrismaticLimit {
    /// Checked constructor: `None` unless both bounds are finite and
    /// `min <= max`.
    pub fn try_new(min: ornis_core::units::Meters, max: ornis_core::units::Meters) -> Option<Self> {
        let (lo, hi) = (min.get(), max.get());
        if lo.is_finite() && hi.is_finite() && lo <= hi {
            Some(Self { min: lo, max: hi })
        } else {
            None
        }
    }

    /// Lower bound in meters.
    pub fn min_units(&self) -> ornis_core::units::Meters {
        ornis_core::units::Meters::new(self.min)
    }

    /// Upper bound in meters.
    pub fn max_units(&self) -> ornis_core::units::Meters {
        ornis_core::units::Meters::new(self.max)
    }
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

impl PrismaticMotor {
    /// Checked constructor: `None` unless the target speed is finite and
    /// the force budget is finite and `>= 0`.
    pub fn try_new(
        target_speed: ornis_core::units::MetersPerSecond,
        max_force: f32,
    ) -> Option<Self> {
        if !target_speed.is_finite() || !max_force.is_finite() || max_force < 0.0 {
            return None;
        }
        Some(Self {
            target_speed: target_speed.get(),
            max_force,
        })
    }

    /// Desired slide speed in m/s.
    pub fn target_speed_units(&self) -> ornis_core::units::MetersPerSecond {
        ornis_core::units::MetersPerSecond::new(self.target_speed)
    }

    /// Raw slide-speed getter (kept for solver-adjacent code; prefer
    /// [`PrismaticMotor::target_speed_units`]).
    pub fn target_speed_raw(&self) -> f32 {
        self.target_speed
    }

    /// Generalized velocity drive with the same target and budget
    /// (stiffness/damping zero, model inert for a pure velocity solve).
    pub fn as_general(&self) -> JointMotor {
        JointMotor {
            kind: MotorKind::Velocity,
            target_position: 0.0,
            target_velocity: self.target_speed,
            stiffness: 0.0,
            damping: 0.0,
            max_force: self.max_force,
            model: MotorModel::default(),
        }
    }
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

impl RevoluteLimit {
    /// Checked constructor: `None` unless both bounds are finite and
    /// `min <= max` (degenerate `min == max` locks the axis).
    pub fn try_new(
        min: ornis_core::units::Radians,
        max: ornis_core::units::Radians,
    ) -> Option<Self> {
        let (lo, hi) = (min.get(), max.get());
        if lo.is_finite() && hi.is_finite() && lo <= hi {
            Some(Self { min: lo, max: hi })
        } else {
            None
        }
    }

    /// Lower bound in radians.
    pub fn min_units(&self) -> ornis_core::units::Radians {
        ornis_core::units::Radians::new(self.min)
    }

    /// Upper bound in radians.
    pub fn max_units(&self) -> ornis_core::units::Radians {
        ornis_core::units::Radians::new(self.max)
    }
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

impl RevoluteMotor {
    /// Checked constructor: `None` unless the target speed is finite and
    /// the torque budget is finite and `>= 0`.
    pub fn try_new(target_speed: f32, max_torque: f32) -> Option<Self> {
        if !target_speed.is_finite() || !max_torque.is_finite() || max_torque < 0.0 {
            return None;
        }
        Some(Self {
            target_speed,
            max_torque,
        })
    }

    /// Typed constructor over [`ornis_core::units::RadiansPerSecond`]: the
    /// same finiteness checks as [`RevoluteMotor::try_new`] with the hinge
    /// speed pinned to rad/s at the type level.
    pub fn try_new_units(
        target_speed: ornis_core::units::RadiansPerSecond,
        max_torque: f32,
    ) -> Option<Self> {
        Self::try_new(target_speed.get(), max_torque)
    }

    /// Desired hinge speed in rad/s.
    pub fn target_speed_units(&self) -> ornis_core::units::RadiansPerSecond {
        ornis_core::units::RadiansPerSecond::new(self.target_speed)
    }

    /// Raw hinge-speed getter (kept for solver-adjacent code; prefer
    /// [`RevoluteMotor::target_speed_units`]).
    pub fn target_speed_raw(&self) -> f32 {
        self.target_speed
    }

    /// Generalized velocity drive with the same target and budget
    /// (stiffness/damping zero, model inert for a pure velocity solve).
    pub fn as_general(&self) -> JointMotor {
        JointMotor {
            kind: MotorKind::Velocity,
            target_position: 0.0,
            target_velocity: self.target_speed,
            stiffness: 0.0,
            damping: 0.0,
            max_force: self.max_torque,
            model: MotorModel::default(),
        }
    }
}

/// How spring constants are interpreted: mass-dependent
/// (acceleration targets) vs mass-independent (force targets).
/// Rapier `MotorModel` parity, including the default.
///
/// - [`MotorModel::AccelerationBased`] (default, recommended): spring
///   constants auto-scale with the driven effective mass, so heavy and
///   light assemblies respond alike —
///   `acceleration = stiffness * error + damping * velocity_error`.
/// - [`MotorModel::ForceBased`]: constants produce absolute forces —
///   `force = stiffness * error + damping * velocity_error` — so the same
///   values behave differently across masses (more physical, retune on
///   mass changes, Rapier `SpringJoint` default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MotorModel {
    /// Spring constants auto-scale with mass (easier to tune, recommended).
    #[default]
    AccelerationBased,
    /// Spring constants produce absolute forces (mass-dependent response).
    ForceBased,
}

impl MotorModel {
    /// Combines the stiffness/damping coefficients for a step of size `dt`
    /// (Rapier `combine_coefficients` parity): returns
    /// `(erp_inv_dt, cfm_coeff, cfm_gain)`. Acceleration-based legs put the
    /// compliance on the mass-scaled term, force-based legs on the gain.
    /// A zero denominator yields zero (no division by dust), never NaN.
    pub fn combine_coefficients(self, dt: f32, stiffness: f32, damping: f32) -> (f32, f32, f32) {
        fn inv(x: f32) -> f32 {
            if x == 0.0 || !x.is_finite() {
                0.0
            } else {
                1.0 / x
            }
        }
        match self {
            MotorModel::AccelerationBased => {
                let erp_inv_dt = stiffness * inv(dt * stiffness + damping);
                let cfm_coeff = inv(dt * dt * stiffness + dt * damping);
                (erp_inv_dt, cfm_coeff, 0.0)
            }
            MotorModel::ForceBased => {
                let erp_inv_dt = stiffness * inv(dt * stiffness + damping);
                let cfm_gain = inv(dt * dt * stiffness + dt * damping);
                (erp_inv_dt, 0.0, cfm_gain)
            }
        }
    }
}

/// What a generalized [`JointMotor`] drives toward (Rapier `JointMotor`
/// control-mode parity: velocity control, position control, or both).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotorKind {
    /// Constant-speed drive toward `target_velocity` (spring terms inert).
    Velocity,
    /// Spring-damper toward `target_position` (`target_velocity` is the
    /// settle-point velocity, usually 0).
    Position,
    /// Servo: position spring plus velocity tracking combined.
    Servo,
}

/// Generalized joint motor (Rapier `JointMotor` idea): one drive type for
/// every powered axis — revolute hinges (radians), prismatic slides and
/// springs (meters), wheel spin about the axle (radians, a revolute
/// special case) — instead of per-joint duplicates.
///
/// Position targets are RELATIVE to the assembly pose (the same Box2D
/// `m_referenceAngle` convention as the travel limits): radians of twist
/// from the assembly twist for hinges, meters of separation from the
/// assembly separation for slides, meters from the rest length for
/// springs. The solvers convert the legacy [`RevoluteMotor`] /
/// [`PrismaticMotor`] into [`JointMotor::velocity`] internally, so all
/// three engines share one drive equation ([`JointMotor::servo_impulse`]).
///
/// Attach to an assembled joint with the engine's `set_joint_motor`
/// (an explicit override — `None` clears it and the spec motor resumes);
/// a [`JointKind::Spring`] carries its motor inline in the spec instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointMotor {
    /// Control mode (which targets participate).
    pub kind: MotorKind,
    /// Position target (rad or m, assembly-relative; see above).
    pub target_position: f32,
    /// Velocity target (rad/s or m/s).
    pub target_velocity: f32,
    /// Spring constant (force/length for [`MotorModel::ForceBased`],
    /// 1/s² for [`MotorModel::AccelerationBased`]).
    pub stiffness: f32,
    /// Damping coefficient (force·time/length vs 1/s, same split).
    pub damping: f32,
    /// Force/torque budget: bounds the per-step motor impulse
    /// (`max_force * dt`).
    pub max_force: f32,
    /// Force vs acceleration interpretation of the spring terms.
    pub model: MotorModel,
}

impl JointMotor {
    /// Checked constructor: `None` unless every scalar is finite,
    /// `stiffness`/`damping`/`max_force` are `>= 0`.
    pub fn try_new(
        kind: MotorKind,
        target_position: f32,
        target_velocity: f32,
        stiffness: f32,
        damping: f32,
        max_force: f32,
        model: MotorModel,
    ) -> Option<Self> {
        let m = Self {
            kind,
            target_position,
            target_velocity,
            stiffness,
            damping,
            max_force,
            model,
        };
        m.check().then_some(m)
    }

    /// Pure velocity drive (the [`RevoluteMotor`]/[`PrismaticMotor`]
    /// semantics, generalized): `None` unless the target is finite and the
    /// budget is finite and `>= 0`.
    pub fn velocity(target_velocity: f32, max_force: f32) -> Option<Self> {
        Self::try_new(
            MotorKind::Velocity,
            0.0,
            target_velocity,
            0.0,
            0.0,
            max_force,
            MotorModel::default(),
        )
    }

    /// Position spring-damper settling at `target_position`:
    /// `None` unless the targets are finite and
    /// stiffness/damping/max are finite with stiffness `> 0`
    /// (a spring with no stiffness is not a spring — use
    /// [`JointMotor::velocity`] for a pure damper... which still needs
    /// `damping > 0` to do anything) and damping/max `>= 0`.
    pub fn position(
        target_position: f32,
        stiffness: f32,
        damping: f32,
        max_force: f32,
    ) -> Option<Self> {
        if !(stiffness.is_finite() && stiffness > 0.0) {
            return None;
        }
        Self::try_new(
            MotorKind::Position,
            target_position,
            0.0,
            stiffness,
            damping,
            max_force,
            MotorModel::default(),
        )
    }

    /// Servo: position spring plus velocity tracking. Same admission as
    /// [`JointMotor::position`] (stiffness `> 0`), plus a finite
    /// `target_velocity`.
    pub fn servo(
        target_position: f32,
        target_velocity: f32,
        stiffness: f32,
        damping: f32,
        max_force: f32,
    ) -> Option<Self> {
        if !(stiffness.is_finite() && stiffness > 0.0) {
            return None;
        }
        Self::try_new(
            MotorKind::Servo,
            target_position,
            target_velocity,
            stiffness,
            damping,
            max_force,
            MotorModel::default(),
        )
    }

    /// Builder-style model override (force vs acceleration targets).
    pub fn with_model(mut self, model: MotorModel) -> Self {
        self.model = model;
        self
    }

    /// Builder-style force-budget override.
    pub fn with_max_force(mut self, max_force: f32) -> Self {
        self.max_force = max_force;
        self
    }

    /// Validity predicate behind [`JointMotor::try_new`]: finite targets,
    /// finite non-negative spring terms and budget.
    pub fn check(&self) -> bool {
        self.target_position.is_finite()
            && self.target_velocity.is_finite()
            && self.stiffness.is_finite()
            && self.stiffness >= 0.0
            && self.damping.is_finite()
            && self.damping >= 0.0
            && self.max_force.is_finite()
            && self.max_force >= 0.0
    }

    /// Whether this motor disturbs sleep (a live drive vote for the
    /// engine sleep heuristics): any budgeted drive with a nonzero target
    /// (velocity), or a live spring (position/servo with stiffness).
    /// Mirrors the legacy "nonzero speed" vote, extended to springs.
    pub fn keeps_awake(&self) -> bool {
        if self.max_force <= 0.0 {
            return false;
        }
        match self.kind {
            MotorKind::Velocity => self.target_velocity != 0.0,
            MotorKind::Position | MotorKind::Servo => {
                self.stiffness > 0.0 || self.target_velocity != 0.0
            }
        }
    }

    /// Spring terms as absolute (force, damping-force) coefficients for an
    /// axis with inverse effective mass `inv_eff_mass` (linear + angular
    /// terms, what the solvers call `k_eff`): force-based values pass
    /// through, acceleration-based values scale by the driven mass
    /// (`1 / inv_eff_mass`). Non-positive mass reads as zero (no drive).
    pub fn pd_coefficients(&self, inv_eff_mass: f32) -> (f32, f32) {
        match self.model {
            MotorModel::ForceBased => (self.stiffness, self.damping),
            MotorModel::AccelerationBased => {
                if inv_eff_mass > 0.0 {
                    let m = 1.0 / inv_eff_mass;
                    (self.stiffness * m, self.damping * m)
                } else {
                    (0.0, 0.0)
                }
            }
        }
    }

    /// Desired servo force for position error `pos_err` (target − current,
    /// rad or m) and velocity error `vel_err` (target rate − current rate):
    /// `stiffness * pos_err + damping * vel_err`, model-scaled by
    /// [`JointMotor::pd_coefficients`]. Unclamped — clamp with
    /// [`JointMotor::servo_impulse`].
    pub fn servo_force(&self, pos_err: f32, vel_err: f32, inv_eff_mass: f32) -> f32 {
        let (k, c) = self.pd_coefficients(inv_eff_mass);
        k * pos_err + c * vel_err
    }

    /// Budget-clamped servo impulse for a step of size `dt`
    /// (`servo_force * dt`, clamped to `±max_force * dt`). Non-positive
    /// `dt` or budget yields zero (never NaN).
    pub fn servo_impulse(&self, pos_err: f32, vel_err: f32, inv_eff_mass: f32, dt: f32) -> f32 {
        if dt <= 0.0 || self.max_force <= 0.0 {
            return 0.0;
        }
        (self.servo_force(pos_err, vel_err, inv_eff_mass) * dt)
            .clamp(-self.max_force * dt, self.max_force * dt)
    }

    /// Budget-clamped deadbeat velocity impulse: the exact impulse driving
    /// the axis rate to the target (`rate_error / inv_eff_mass`, clamped
    /// to `±max_force * dt`). The velocity-kind drive equation shared by
    /// every engine; the model is inert here (no spring terms).
    pub fn velocity_impulse(&self, rate_error: f32, inv_eff_mass: f32, dt: f32) -> f32 {
        if dt <= 0.0 || self.max_force <= 0.0 || inv_eff_mass <= 0.0 {
            return 0.0;
        }
        (rate_error / inv_eff_mass).clamp(-self.max_force * dt, self.max_force * dt)
    }
}

/// Spring integration tactic for [`JointKind::Spring`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpringIntegration {
    /// Implicit (closed-form, wheel-spring discipline): unconditionally
    /// stable for any stiffness at any substep count. Default.
    #[default]
    Implicit,
    /// Semi-explicit Euler force (`F = −(k·s + c·v)` applied as an
    /// impulse): cheaper per step, conditionally stable — needs
    /// `stiffness * dt² * inv_eff_mass < ~4` (stiff springs on light
    /// bodies at large steps explode; halve the step or switch to
    /// [`SpringIntegration::Implicit`).
    Explicit,
}

/// Solver-agnostic joint setup, resolved once at creation from a
/// [`JointKind`] and the two assembly poses. Both engines (SI and
/// AVBD) consume this instead of parsing `JointKind` twice: axis
/// normalization, axle orthogonalization and reference capture live here,
/// per-solver row math stays in the engines.
///
/// All reference values measure the ASSEMBLY pose (Box2D `m_referenceAngle`
/// style): limits, motors and springs read travel relative to these.
/// Expressions mirror the historical per-engine ones op-for-op so resolved
/// values are bit-identical to what each engine computed before.
///
/// Degenerate axes (squared length below [`DEGENERATE_LEN2`]) fall back infallibly
/// (`Z` for hinge/slide, `Y` for suspension — the SI policy) and set
/// [`ResolvedJoint::degenerate`]; each engine keeps its own admission
/// policy (SI accepts, AVBD rejects revolute/prismatic).
#[derive(Debug, Clone, Copy)]
pub struct ResolvedJoint {
    /// Anchor in body A's local frame (passthrough).
    pub la: Vec3,
    /// Anchor in body B's local frame (passthrough).
    pub lb: Vec3,
    /// Hinge/slide/suspension axis in A's frame (normalized or fallback).
    pub ax_a: Vec3,
    /// Hinge/slide/suspension axis in B's frame (normalized or fallback).
    pub ax_b: Vec3,
    /// Wheel spin axle in A's frame (orthogonalized; `X` otherwise).
    pub bx_a: Vec3,
    /// Wheel spin axle in B's frame (orthogonalized; `X` otherwise).
    pub bx_b: Vec3,
    /// Hinge twist (revolute) or axle twist (wheel) at assembly, rad.
    pub ref_angle: f32,
    /// Anchor separation along the axis (prismatic/wheel) at assembly, m.
    pub ref_length: f32,
    /// Anchor distance (distance rod) at assembly, m.
    pub ref_distance: f32,
    /// Relative orientation (fixed/wheel/sixdof) at assembly.
    pub ref_quat: Quat,
    /// Anchor separation (fixed/wheel/sixdof) at assembly, in A's frame.
    pub ref_anchor_delta: Vec3,
    /// True when an axis needed its fallback (see above).
    /// Prefer [`Self::status`] with [`crate::flags::AxisStatus`].
    pub degenerate: bool,
}

impl ResolvedJoint {
    /// Axis health as a [`crate::flags::AxisStatus`].
    pub fn status(&self) -> crate::flags::AxisStatus {
        crate::flags::AxisStatus::from(self.degenerate)
    }
}

use crate::constants::DEGENERATE_LEN2;

/// Resolve joint frames and assembly references. Returns `None` for
/// [`JointKind::Gear`], which coordinates other joints (needs the engine's
/// joint list, not just poses) and is handled engine-side.
pub fn resolve_joint(
    kind: &JointKind,
    pa: Vec3,
    qa: Quat,
    pb: Vec3,
    qb: Quat,
) -> Option<ResolvedJoint> {
    let mut r = ResolvedJoint {
        la: Vec3::ZERO,
        lb: Vec3::ZERO,
        ax_a: Vec3::X,
        ax_b: Vec3::X,
        bx_a: Vec3::X,
        bx_b: Vec3::X,
        ref_angle: 0.0,
        ref_length: 0.0,
        ref_distance: 0.0,
        ref_quat: Quat::IDENTITY,
        ref_anchor_delta: Vec3::ZERO,
        degenerate: false,
    };
    match kind {
        JointKind::Ball {
            local_anchor_a,
            local_anchor_b,
        } => {
            r.la = *local_anchor_a;
            r.lb = *local_anchor_b;
        }
        JointKind::Revolute {
            local_anchor_a,
            local_anchor_b,
            local_axis_a,
            local_axis_b,
            ..
        } => {
            r.la = *local_anchor_a;
            r.lb = *local_anchor_b;
            r.degenerate = local_axis_a.length_squared() < DEGENERATE_LEN2
                || local_axis_b.length_squared() < DEGENERATE_LEN2;
            r.ax_a = local_axis_a.normalize_or(Vec3::Z);
            r.ax_b = local_axis_b.normalize_or(Vec3::Z);
            r.ref_angle = hinge_twist(qa, qb, r.ax_a);
        }
        JointKind::Prismatic {
            local_anchor_a,
            local_anchor_b,
            local_axis_a,
            local_axis_b,
            ..
        } => {
            r.la = *local_anchor_a;
            r.lb = *local_anchor_b;
            r.degenerate = local_axis_a.length_squared() < DEGENERATE_LEN2
                || local_axis_b.length_squared() < DEGENERATE_LEN2;
            r.ax_a = local_axis_a.normalize_or(Vec3::Z);
            r.ax_b = local_axis_b.normalize_or(Vec3::Z);
            let wa = (qa * r.ax_a).normalize_or(Vec3::Z);
            let pa0 = pa + qa * r.la;
            let pb0 = pb + qb * r.lb;
            r.ref_length = (pb0 - pa0).dot(wa);
        }
        JointKind::Fixed {
            local_anchor_a,
            local_anchor_b,
        } => {
            r.la = *local_anchor_a;
            r.lb = *local_anchor_b;
            r.ref_quat = qa.conjugate() * qb;
            let ra = qa * r.la;
            let rb = qb * r.lb;
            r.ref_anchor_delta = qa.conjugate() * ((pb + rb) - (pa + ra));
        }
        JointKind::Distance {
            local_anchor_a,
            local_anchor_b,
        } => {
            r.la = *local_anchor_a;
            r.lb = *local_anchor_b;
            let ra = qa * r.la;
            let rb = qb * r.lb;
            r.ref_distance = ((pb + rb) - (pa + ra)).length();
        }
        JointKind::Rope {
            local_anchor_a,
            local_anchor_b,
            ..
        }
        | JointKind::Spring {
            local_anchor_a,
            local_anchor_b,
            ..
        } => {
            r.la = *local_anchor_a;
            r.lb = *local_anchor_b;
            // Assembly separation as a diagnostic baseline (the live limit
            // — rope maximum, spring rest length — rides in the spec).
            let ra = qa * r.la;
            let rb = qb * r.lb;
            r.ref_distance = ((pb + rb) - (pa + ra)).length();
        }
        JointKind::Wheel {
            local_anchor_a,
            local_anchor_b,
            local_suspension_a,
            local_suspension_b,
            local_axle_a,
            local_axle_b,
            ..
        } => {
            r.la = *local_anchor_a;
            r.lb = *local_anchor_b;
            r.degenerate = local_suspension_a.length_squared() < DEGENERATE_LEN2
                || local_suspension_b.length_squared() < DEGENERATE_LEN2;
            let sa = local_suspension_a.normalize_or(Vec3::Y);
            let sb = local_suspension_b.normalize_or(Vec3::Y);
            r.ax_a = sa;
            r.ax_b = sb;
            r.bx_a = orthogonalize_axle(sa, *local_axle_a);
            r.bx_b = orthogonalize_axle(sb, *local_axle_b);
            r.ref_angle = hinge_twist(qa, qb, r.bx_a);
            let wa = (qa * sa).normalize_or(Vec3::Z);
            let pa0 = pa + qa * r.la;
            let pb0 = pb + qb * r.lb;
            r.ref_length = (pb0 - pa0).dot(wa);
            r.ref_quat = qa.conjugate() * qb;
            let ra = qa * r.la;
            let rb = qb * r.lb;
            r.ref_anchor_delta = qa.conjugate() * ((pb + rb) - (pa + ra));
        }
        JointKind::SixDof {
            local_anchor_a,
            local_anchor_b,
            ..
        } => {
            r.la = *local_anchor_a;
            r.lb = *local_anchor_b;
            r.ref_quat = qa.conjugate() * qb;
            let ra = qa * r.la;
            let rb = qb * r.lb;
            r.ref_anchor_delta = qa.conjugate() * ((pb + rb) - (pa + ra));
        }
        JointKind::Motor { .. } => {
            // Free drive: no anchors (COM-level rows), but the assembly
            // relative pose is still captured for the weak SI position
            // pull (`correction > 0`) and the AVBD rest residual.
            r.ref_quat = qa.conjugate() * qb;
            r.ref_anchor_delta = qa.conjugate() * (pb - pa);
        }
        JointKind::Gear { .. } => return None,
    }
    Some(r)
}

/// Cross-solver coupling row of a joint whose ends live in different
/// solvers under [`RoutingKind::Islands`](crate::RoutingKind) routing.
/// Only structural point rows qualify: a ball row pins two world anchors
/// together (3 linear equalities), a distance row pins their separation to
/// the assembly rest length (1 linear equality along the anchor delta).
/// Every other [`JointKind`] has no cross row — the split coupling pass
/// leaves it unmirrored and [`Engine::cross_joint_status`](crate::Engine::cross_joint_status)
/// reports it as unsupported instead of silently mis-solving it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossRowKind {
    /// Ball-and-socket anchor coincidence (3 positional/velocity equalities).
    Ball,
    /// Rigid-rod rest-length row (1 equality along the anchor delta axis).
    Distance,
    /// Rope maximum-length row (1 one-sided inequality along the anchor
    /// delta: pulls a stretched rope, ignores a slack one).
    Rope,
}

/// Cross-solver row of a joint spec, or `None` when the kind has no
/// structural point row. Ball, distance and rope couple across solvers
/// (rope one-sided: the coupling pass only pulls a stretched rope);
/// revolute, prismatic, fixed, spring, wheel, gear, six-DOF and motor
/// return `None` explicitly — a cross joint of those kinds is never
/// half-solved (springs have no compliant cross row in v1; the motor is a
/// velocity drive with no positional row to couple — see `split.rs`).
pub fn cross_row_kind(kind: &JointKind) -> Option<CrossRowKind> {
    match kind {
        JointKind::Ball { .. } => Some(CrossRowKind::Ball),
        JointKind::Distance { .. } => Some(CrossRowKind::Distance),
        JointKind::Rope { .. } => Some(CrossRowKind::Rope),
        JointKind::Revolute { .. }
        | JointKind::Prismatic { .. }
        | JointKind::Fixed { .. }
        | JointKind::Spring { .. }
        | JointKind::Wheel { .. }
        | JointKind::Gear { .. }
        | JointKind::SixDof { .. }
        | JointKind::Motor { .. } => None,
    }
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
    /// One-sided rope accumulator (rope joints only): non-positive pulling
    /// impulse along the anchor delta (upper-bound convention, like the
    /// hinge/slide limit clamps). Clamp memory, not a warm start — slack
    /// zeroes it instead of re-applying a stale pull.
    pub acc_rope: f32,
    /// Generalized motor override (`set_joint_motor`: revolute, prismatic
    /// and wheel-axle joints). `Some` replaces the spec motor for the
    /// solve; `None` resumes the spec motor. Rides the migration snapshot
    /// so solver switches never silently drop it.
    pub servo: Option<JointMotor>,
    /// Accumulated gear-constraint impulse (gear joints only).
    pub acc_gear: f32,
    /// Raw and continuous coordinates of the gear's two referenced joints.
    pub gear_mem: Option<([f32; 2], [f32; 2])>,
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
            acc_rope: 0.0,
            servo: None,
            acc_gear: 0.0,
            gear_mem: None,
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
            | JointKind::Rope { .. }
            | JointKind::Spring { .. }
            | JointKind::Wheel { .. }
            | JointKind::Gear { .. }
            | JointKind::SixDof { .. }
            | JointKind::Motor { .. } => (None, None),
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
            | JointKind::Rope { .. }
            | JointKind::Spring { .. }
            | JointKind::Wheel { .. }
            | JointKind::Gear { .. }
            | JointKind::SixDof { .. }
            | JointKind::Motor { .. } => (None, None),
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
            | JointKind::Rope { .. }
            | JointKind::Spring { .. }
            | JointKind::Wheel { .. }
            | JointKind::Gear { .. }
            | JointKind::SixDof { .. }
            | JointKind::Motor { .. } => None,
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

    /// Local anchor on each body. Gear and motor joints hold no anchors —
    /// returns zeros (gears coordinate other joints instead of constraining
    /// bodies; motors drive center-of-mass velocities directly).
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
            | JointKind::Rope {
                local_anchor_a,
                local_anchor_b,
                ..
            }
            | JointKind::Spring {
                local_anchor_a,
                local_anchor_b,
                ..
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
            JointKind::Gear { .. } | JointKind::Motor { .. } => (Vec3::ZERO, Vec3::ZERO),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: Quat = Quat::IDENTITY;

    #[test]
    fn resolve_revolute_captures_twist_and_normalizes() {
        let r = resolve_joint(
            &JointKind::Revolute {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_axis_a: Vec3::new(0.0, 2.0, 0.0),
                local_axis_b: Vec3::Y,
                limit: None,
                motor: None,
            },
            Vec3::ZERO,
            ID,
            Vec3::X,
            ID,
        )
        .expect("revolute resolves");
        assert!(!r.degenerate);
        assert_eq!(r.ax_a, Vec3::Y);
        assert_eq!(r.ref_angle, 0.0);
    }

    #[test]
    fn resolve_flags_degenerate_axes_for_engine_policy() {
        let r = resolve_joint(
            &JointKind::Prismatic {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_axis_a: Vec3::ZERO,
                local_axis_b: Vec3::Y,
                limit: None,
                motor: None,
            },
            Vec3::ZERO,
            ID,
            Vec3::ZERO,
            ID,
        )
        .expect("prismatic resolves (flagged)");
        assert!(r.degenerate);
        // Fallback is the builtin substitute, not a NaN.
        assert_eq!(r.ax_a, Vec3::Z);
    }

    #[test]
    fn resolve_wheel_captures_slide_and_orthogonalizes_axle() {
        let r = resolve_joint(
            &JointKind::Wheel {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_suspension_a: Vec3::Y,
                local_suspension_b: Vec3::Y,
                local_axle_a: Vec3::Y, // parallel: deterministic fallback
                local_axle_b: Vec3::Z,
                suspension: WheelSuspension {
                    frequency_hz: 2.0,
                    damping_ratio: 0.5,
                },
                motor: None,
            },
            Vec3::new(0.0, 1.0, 0.0),
            ID,
            Vec3::ZERO,
            ID,
        )
        .expect("wheel resolves");
        assert!(!r.degenerate);
        // 1m separation along +Y reads as -1m slide (B-minus-A convention).
        assert!((r.ref_length + 1.0).abs() < 1e-6);
        // Parallel axle fell back perpendicular to the suspension.
        assert!(r.bx_a.dot(Vec3::Y).abs() < 1e-6);
        assert!((r.bx_a.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn resolve_gear_needs_engine_list() {
        assert!(
            resolve_joint(
                &JointKind::Gear {
                    joint_a: JointHandle::from_raw(0),
                    joint_b: JointHandle::from_raw(1),
                    ratio: 2.0
                },
                Vec3::ZERO,
                ID,
                Vec3::ZERO,
                ID,
            )
            .is_none()
        );
    }

    #[test]
    fn motor_model_combine_splits_force_and_acceleration_legs() {
        let dt = 1.0 / 60.0;
        let (erp_a, cfm_a, gain_a) =
            MotorModel::AccelerationBased.combine_coefficients(dt, 100.0, 10.0);
        let (erp_f, cfm_f, gain_f) = MotorModel::ForceBased.combine_coefficients(dt, 100.0, 10.0);
        // Same error reduction, compliance on opposite legs (Rapier parity).
        assert!((erp_a - erp_f).abs() < 1e-6);
        assert!(cfm_a > 0.0 && gain_a == 0.0);
        assert!(cfm_f == 0.0 && gain_f > 0.0);
        // Zero step never divides by dust (safe inverse yields zero).
        assert_eq!(
            MotorModel::ForceBased.combine_coefficients(0.0, 100.0, 10.0),
            (10.0, 0.0, 0.0)
        );
    }

    #[test]
    fn joint_motor_constructors_validate_ranges() {
        assert!(JointMotor::velocity(3.0, 50.0).is_some());
        assert!(JointMotor::velocity(f32::NAN, 50.0).is_none());
        assert!(JointMotor::velocity(3.0, -1.0).is_none());
        assert!(JointMotor::position(1.0, 40.0, 5.0, 50.0).is_some());
        // A spring with no stiffness is not a spring.
        assert!(JointMotor::position(1.0, 0.0, 5.0, 50.0).is_none());
        assert!(JointMotor::servo(1.0, 2.0, 40.0, 5.0, 50.0).is_some());
        assert!(JointMotor::servo(1.0, f32::INFINITY, 40.0, 5.0, 50.0).is_none());
        // Legacy velocity motors convert losslessly into the unified drive.
        let legacy = RevoluteMotor {
            target_speed: 3.0,
            max_torque: 50.0,
        };
        let general = legacy.as_general();
        assert_eq!(general.kind, MotorKind::Velocity);
        assert_eq!(general.target_velocity, 3.0);
        assert_eq!(general.max_force, 50.0);
        let slide = PrismaticMotor {
            target_speed: -1.5,
            max_force: 20.0,
        };
        assert_eq!(slide.as_general().target_velocity, -1.5);
    }

    #[test]
    fn servo_impulse_is_budget_clamped_and_deadbeat_is_exact() {
        let servo = JointMotor::position(1.0, 100.0, 10.0, 50.0).expect("valid servo");
        // At rest on target with no rate error: zero impulse.
        assert_eq!(servo.servo_impulse(0.0, 0.0, 1.0, 1.0 / 60.0), 0.0);
        // A huge error clamps to the budget, never past it.
        let big = servo.servo_impulse(10.0, 0.0, 1.0, 1.0 / 60.0);
        assert!((big - 50.0 / 60.0).abs() < 1e-6 && big <= 50.0 / 60.0 + 1e-6);
        // Force-based values pass through; acceleration-based scale by mass.
        let force = JointMotor::position(1.0, 100.0, 10.0, 50.0)
            .expect("valid")
            .with_model(MotorModel::ForceBased);
        assert_eq!(force.pd_coefficients(0.5), (100.0, 10.0));
        let accel = force.with_model(MotorModel::AccelerationBased);
        assert_eq!(accel.pd_coefficients(0.5), (200.0, 20.0));
        // Velocity deadbeat: exact rate correction clamped by the budget.
        let vel = JointMotor::velocity(3.0, 50.0).expect("valid");
        assert!((vel.velocity_impulse(2.0, 0.5, 1.0 / 60.0) - 50.0 / 60.0).abs() < 1e-6);
        assert_eq!(vel.velocity_impulse(0.5, 0.5, 1.0), 1.0);
    }

    #[test]
    fn rope_and_spring_checked_constructors_gate_ranges() {
        use ornis_core::units::Meters;
        assert!(JointKind::rope_checked(Vec3::ZERO, Vec3::ZERO, Meters::new(2.0)).is_some());
        assert!(JointKind::rope_checked(Vec3::ZERO, Vec3::ZERO, Meters::new(0.0)).is_none());
        assert!(JointKind::rope_checked(Vec3::ZERO, Vec3::ZERO, Meters::new(f32::NAN)).is_none());
        let motor = JointMotor::position(1.0, 40.0, 5.0, 100.0).expect("valid");
        assert!(
            JointKind::spring_checked(Vec3::ZERO, Vec3::ZERO, motor, SpringIntegration::Implicit)
                .is_some()
        );
        let bad = JointMotor {
            stiffness: f32::NAN,
            ..motor
        };
        assert!(
            JointKind::spring_checked(Vec3::ZERO, Vec3::ZERO, bad, SpringIntegration::Explicit)
                .is_none()
        );
    }

    #[test]
    fn resolve_rope_and_spring_capture_assembly_distance() {
        let rope =
            JointKind::rope_checked(Vec3::ZERO, Vec3::ZERO, ornis_core::units::Meters::new(2.0))
                .expect("valid rope");
        let r = resolve_joint(&rope, Vec3::ZERO, ID, Vec3::new(0.0, -1.5, 0.0), ID)
            .expect("rope resolves");
        assert!((r.ref_distance - 1.5).abs() < 1e-6);
        assert_eq!(rope.rope_max_units().map(|m| m.get()), Some(2.0));
        let motor = JointMotor::position(1.0, 40.0, 5.0, 100.0).expect("valid");
        let spring =
            JointKind::spring_checked(Vec3::ZERO, Vec3::ZERO, motor, SpringIntegration::Implicit)
                .expect("valid spring");
        assert_eq!(spring.spring_motor().map(|m| m.target_position), Some(1.0));
        // Rope couples across solvers (one-sided); spring has no cross row.
        assert_eq!(cross_row_kind(&rope), Some(CrossRowKind::Rope));
        assert_eq!(cross_row_kind(&spring), None);
        let (la, lb) = Joint {
            body_a: BodyHandle::from_raw(0),
            body_b: BodyHandle::from_raw(1),
            kind: rope,
            acc_lin: [0.0; 3],
            acc_ang: [0.0; 3],
            reference_angle: 0.0,
            reference_length: 0.0,
            reference_distance: 0.0,
            reference_quat: Quat::IDENTITY,
            reference_anchor_delta: Vec3::ZERO,
            acc_limit: 0.0,
            acc_dist: 0.0,
            acc_rope: 0.0,
            servo: None,
            acc_gear: 0.0,
            gear_mem: None,
            acc_6dof: [0.0; 6],
        }
        .local_anchors();
        assert_eq!((la, lb), (Vec3::ZERO, Vec3::ZERO));
    }

    #[test]
    fn motor_checked_gates_targets_budgets_and_correction() {
        let good = JointKind::motor_checked(Vec3::X, Vec3::Z, 50.0, 10.0, 0.3);
        assert!(good.is_some());
        let drive = good
            .expect("valid motor")
            .motor_drive()
            .expect("drive reads back");
        assert_eq!(drive.linear_target, Vec3::X);
        assert_eq!(drive.angular_target, Vec3::Z);
        assert_eq!(drive.max_force, 50.0);
        assert_eq!(drive.max_torque, 10.0);
        assert_eq!(drive.correction, 0.3);
        // Non-finite targets, negative budgets and out-of-range correction refuse.
        assert!(JointKind::motor_checked(Vec3::NAN, Vec3::Z, 50.0, 10.0, 0.3).is_none());
        assert!(JointKind::motor_checked(Vec3::X, Vec3::Z, -1.0, 10.0, 0.3).is_none());
        assert!(JointKind::motor_checked(Vec3::X, Vec3::Z, 50.0, f32::INFINITY, 0.3).is_none());
        assert!(JointKind::motor_checked(Vec3::X, Vec3::Z, 50.0, 10.0, 1.5).is_none());
        assert!(JointKind::motor_checked(Vec3::X, Vec3::Z, 50.0, 10.0, f32::NAN).is_none());
        // Non-motor kinds expose no drive.
        assert!(
            JointKind::Ball {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
            }
            .motor_drive()
            .is_none()
        );
    }

    #[test]
    fn motor_drive_sleep_vote_needs_a_budgeted_target() {
        let coasting =
            MotorDrive::try_new(Vec3::ZERO, Vec3::ZERO, 50.0, 10.0, 0.3).expect("valid drive");
        assert!(!coasting.keeps_awake());
        let pushing =
            MotorDrive::try_new(Vec3::X, Vec3::ZERO, 50.0, 10.0, 0.0).expect("valid drive");
        assert!(pushing.keeps_awake());
        let spinning =
            MotorDrive::try_new(Vec3::ZERO, Vec3::Z, 50.0, 10.0, 0.0).expect("valid drive");
        assert!(spinning.keeps_awake());
        // A budgeted target with a zeroed budget on its own axis is no vote.
        let unbudgeted = MotorDrive {
            max_force: 0.0,
            ..pushing
        };
        assert!(!unbudgeted.keeps_awake());
    }

    #[test]
    fn resolve_motor_captures_assembly_pose_without_anchors() {
        let motor =
            JointKind::motor_checked(Vec3::X, Vec3::ZERO, 50.0, 10.0, DEFAULT_MOTOR_CORRECTION)
                .expect("valid motor");
        let r = resolve_joint(
            &motor,
            Vec3::new(1.0, 0.0, 0.0),
            ID,
            Vec3::new(4.0, 0.0, 0.0),
            ID,
        )
        .expect("motor resolves");
        assert!(!r.degenerate);
        assert_eq!((r.la, r.lb), (Vec3::ZERO, Vec3::ZERO));
        // COM separation in A's frame survives for the weak position pull.
        assert!((r.ref_anchor_delta - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-6);
        assert_eq!(r.ref_quat, Quat::IDENTITY);
        // A velocity drive has no positional row to couple across solvers.
        assert_eq!(cross_row_kind(&motor), None);
        let (la, lb) = Joint {
            body_a: BodyHandle::from_raw(0),
            body_b: BodyHandle::from_raw(1),
            kind: motor,
            acc_lin: [0.0; 3],
            acc_ang: [0.0; 3],
            reference_angle: 0.0,
            reference_length: 0.0,
            reference_distance: 0.0,
            reference_quat: Quat::IDENTITY,
            reference_anchor_delta: Vec3::ZERO,
            acc_limit: 0.0,
            acc_dist: 0.0,
            acc_rope: 0.0,
            servo: None,
            acc_gear: 0.0,
            gear_mem: None,
            acc_6dof: [0.0; 6],
        }
        .local_anchors();
        assert_eq!((la, lb), (Vec3::ZERO, Vec3::ZERO));
    }
}
