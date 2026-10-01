//! Joint (constraint) definitions for the sequential-impulse physics engine (G5).
//!
//! Modeled on Box3D `spherical_joint`/`revolute_joint` and Jolt `Constraint`:
//! joints are persistent equality constraints with warm-started accumulated
//! impulses, solved as dedicated sub-solvers inside the substep loop.

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
/// Degenerate axes (squared length below 1e-12) fall back infallibly
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

/// Degenerate-axis boundary (squared length): mirrors AVBD's historical
/// reject threshold so its admission policy is preserved bit-for-bit.
const DEGENERATE_LEN2: f32 = 1e-12;

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
}

/// Cross-solver row of a joint spec, or `None` when the kind has no
/// structural point row. Ball and distance couple across solvers;
/// revolute, prismatic, fixed, wheel, gear and six-DOF return `None`
/// explicitly — a cross joint of those kinds is never half-solved.
pub fn cross_row_kind(kind: &JointKind) -> Option<CrossRowKind> {
    match kind {
        JointKind::Ball { .. } => Some(CrossRowKind::Ball),
        JointKind::Distance { .. } => Some(CrossRowKind::Distance),
        JointKind::Revolute { .. }
        | JointKind::Prismatic { .. }
        | JointKind::Fixed { .. }
        | JointKind::Wheel { .. }
        | JointKind::Gear { .. }
        | JointKind::SixDof { .. } => None,
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
}
