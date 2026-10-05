//! Versioned serde world snapshots (P8): deterministic `WorldSnapshot`
//! capture plus RON round-trips, in the spirit of Rapier's
//! `serde-serialize` snapshot support.
//!
//! The migration [`SceneSnapshot`](crate::migration) this module is often
//! confused with is a transient solver-handoff struct (never serialized,
//! drops warm starts and sleep by design). [`WorldSnapshot`] is the
//! opposite: a versioned, serializable, bit-faithful capture of a
//! sequential-impulse world — every body field verbatim (including
//! sleep-zeroed inverse masses), driver baselines, event baselines and
//! pending queues, joint specs with assembly references and warm
//! accumulators, the contact warm-start cache, sleep timers, solver tuning
//! and gravity — so `capture → to_ron → from_ron → instantiate` continues
//! bit-identically.
//!
//! Scope (v1):
//!
//! - Engines: [`SequentialImpulseEngine`](crate::engine::SequentialImpulseEngine)
//!   directly, plus the [`Engine`](crate::Engine) orchestrator in
//!   [`RoutingKind::Single`](crate::RoutingKind) with the sequential-impulse
//!   solver. AVBD/XPBD engines and Islands routing are refused with a typed
//!   [`SnapshotError::Unsupported`] (follow-up, not silent loss).
//! - Soft bodies travel as parked payload only (persistent particle /
//!   constraint / surface data). Substep-live multipliers (`lambda`,
//!   `volume_lambda`, self-collision) are reset by
//!   [`SoftBody::begin_substep`](crate::soft::SoftBody::begin_substep) every
//!   substep, so they are not carried. Restoring soft bodies into a bare
//!   sequential-impulse engine is refused (it cannot own them); restore
//!   through [`Engine`](crate::Engine), which re-parks them.
//! - Not carried (documented, never silent): contact hooks and GPU solver
//!   attachments (capture refuses while they are attached), the
//!   narrowphase/SAT derived caches (rebuilt naturally), scratch buffers,
//!   diagnostics, and custom uniform-grid cell sizes (the backend kind is
//!   carried, the cell size restores to its default).
//! - Debug-render data and mesh converters are render/asset domains and
//!   stay out of scope (follow-up).
//!
//! Float fidelity: RON floats print with Rust's shortest round-trip
//! formatting, so finite `f32` values survive `to_ron`/`from_ron`
//! bit-identically (pinned by test, including `-0.0` and `inf`); `NaN`
//! payloads canonicalize (physics state must be finite anyway).

use serde::{Deserialize, Serialize};

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::errors::{JointError, SnapshotError};
use crate::joint::{
    AxisConfig, JointHandle, JointKind, JointMotor, MotorKind, MotorModel, PrismaticLimit,
    PrismaticMotor, RevoluteLimit, RevoluteMotor, SpringIntegration, WheelSuspension,
};
use crate::shape::{ConvexHull, Heightfield, Pose, Shape, TriMesh, Triangle};
use crate::soft::{DeformConstraint, DeformKind, Particle, ParticleIdx, SoftBody};
use crate::trigger::{
    ContactEvent, ContactEventKind, ContactForceEvent, TriggerEvent, TriggerEventKind,
};
use crate::xpbd::SoftContactEvent;

/// Serialized [`WorldSnapshot`] format version. Bumped only for
/// incompatible changes; [`WorldSnapshot::from_ron`] refuses anything
/// else with [`SnapshotError::VersionMismatch`].
pub const WORLD_SNAPSHOT_VERSION: u32 = 1;

/// Plain-data world snapshot: every field below is `serde`-round-tripped
/// through RON (see [`WorldSnapshot::to_ron`]/[`WorldSnapshot::from_ron`]).
/// Handles are raw `u32` indices into the parallel vectors; field order is
/// handle order throughout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldSnapshot {
    /// Format version (see [`WORLD_SNAPSHOT_VERSION`]).
    pub version: u32,
    /// World-space gravity (m/s²).
    pub gravity: [f32; 3],
    /// Solver tuning (substeps, iterations, budget, backend kind, ...).
    pub tuning: TuningSnapshot,
    /// Bodies in handle order (every field verbatim).
    pub bodies: Vec<BodySnapshot>,
    /// Sleep flags, parallel to `bodies`.
    pub asleep: Vec<bool>,
    /// Ever-moved flags, parallel to `bodies` (frozen-fast-track input).
    pub moved: Vec<bool>,
    /// Island ids, parallel to `bodies` (wake coherence across restore).
    pub islands: Vec<u32>,
    /// Sleep timers `(island root, seconds)`, sorted.
    pub island_timers: Vec<(u32, f32)>,
    /// Post-wake grace counters `(island root, steps left)`, sorted.
    pub island_grace: Vec<(u32, u32)>,
    /// Step-start driver baselines, parallel to `bodies`.
    pub prev: Vec<PoseSnapshot>,
    /// Solid-contact touch baseline (canonical pairs), sorted.
    pub event_contacts: Vec<(u32, u32)>,
    /// Trigger overlap baseline (canonical pairs), sorted.
    pub event_triggers: Vec<(u32, u32)>,
    /// Undrained contact transitions at capture time, in drain order.
    pub pending_contacts: Vec<ContactEventSnapshot>,
    /// Undrained trigger transitions at capture time, in drain order.
    pub pending_triggers: Vec<TriggerEventSnapshot>,
    /// Undrained contact-force reports at capture time, in drain order.
    pub pending_forces: Vec<ContactForceEventSnapshot>,
    /// Contact warm-start cache (last substep), sorted by pair.
    pub warm: Vec<WarmPairSnapshot>,
    /// Joints in handle order (spec + assembly reference + warm state).
    pub joints: Vec<JointSnapshot>,
    /// Parked soft bodies in handle order (persistent data only; the
    /// substep-live multipliers are reset by `begin_substep`, never stored).
    pub soft_bodies: Vec<SoftBodySnapshot>,
    /// Parked soft↔rigid `(soft, particle, body)` touch triples, sorted.
    pub soft_touch: Vec<(u32, u32, u32)>,
    /// Soft-side Coulomb coefficient (`None` = keep the target default;
    /// the sequential-impulse capture writes `None`).
    pub soft_friction: Option<f32>,
    /// Undrained soft↔rigid transitions at capture time, in drain order.
    pub soft_contact_events: Vec<SoftContactEventSnapshot>,
    /// Undrained fracture reports at capture time, in drain order.
    pub fracture_events: Vec<FractureEventSnapshot>,
}

/// Solver tuning carried by [`WorldSnapshot::tuning`]. Restored through
/// the public setters, so a snapshot continues under the exact
/// configuration it was captured with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuningSnapshot {
    /// Substeps per `step` call.
    pub substeps: u32,
    /// Velocity iterations per substep.
    pub velocity_iterations: u32,
    /// Position iterations per substep.
    pub position_iterations: u32,
    /// CFM softness of the positional pass.
    pub contact_softness: f32,
    /// Worst-case step budget (`None` = shedding disabled).
    pub step_budget: Option<(usize, u32)>,
    /// Angular-sweep iteration budget.
    pub max_ccd_substeps: usize,
    /// Candidate-pair backend kind (custom grid cell sizes are NOT
    /// carried — the backend restores with default tuning).
    pub broadphase: BroadPhaseSnapshot,
    /// `true` = SIMD-wide single-point path, `false` = scalar path.
    pub wide_solver: bool,
}

/// Broadphase backend kind mirror (the engine enum carries no serde impl).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BroadPhaseSnapshot {
    /// Axis sweep baseline.
    SweepAndPrune,
    /// Uniform spatial grid (default cell size on restore).
    UniformGrid,
    /// Experimental dynamic AABB tree.
    DynamicAabbTree,
    /// Analytic routing (hysteresis state restarts fresh).
    Auto,
}

/// One body with every field verbatim (floats as raw `f32`, vectors as
/// arrays): sleep-zeroed inverse masses included, so sleepers restore as
/// sleepers instead of waking up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BodySnapshot {
    /// World-space center of mass.
    pub position: [f32; 3],
    /// Orientation (unit quaternion `xyzw`).
    pub orientation: [f32; 4],
    /// Linear velocity (m/s).
    pub velocity: [f32; 3],
    /// Angular velocity (rad/s, world space).
    pub angular_velocity: [f32; 3],
    /// Total mass (kg).
    pub mass: f32,
    /// Cached inverse mass (zeroed while asleep — carried verbatim).
    pub inv_mass: f32,
    /// Diagonal body-frame inertia (zeroed while asleep — verbatim).
    pub inertia: [f32; 3],
    /// Accumulated external torque (N·m).
    pub torque: [f32; 3],
    /// Restitution in `[0, 1]`.
    pub restitution: f32,
    /// Coulomb friction.
    pub friction: f32,
    /// Transverse friction (`mu2`).
    pub friction_transverse: f32,
    /// Anisotropy axis (`None` = isotropic).
    pub friction_dir: Option<[f32; 3]>,
    /// Rolling resistance (m).
    pub rolling_friction: f32,
    /// Torsional friction (m).
    pub torsion_friction: f32,
    /// Collision primitive.
    pub shape: ShapeSnapshot,
    /// Collision layer bits.
    pub collision_layer: u32,
    /// Collision mask bits.
    pub collision_mask: u32,
    /// Trigger (sensor) role flag.
    pub is_trigger: bool,
    /// Fracture impact speed (m/s, `inf` = never).
    pub fracture_impact_speed: f32,
    /// Simulation role.
    pub body_type: BodyTypeSnapshot,
    /// Bullet (full-CCD) upgrade flag.
    pub ccd_enabled: bool,
    /// Contact-force report threshold (N, `inf` = disabled).
    pub contact_force_threshold: f32,
}

/// Body role mirror (the engine enum carries no serde impl).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyTypeSnapshot {
    /// Never moves.
    Static,
    /// Fully simulated.
    Dynamic,
    /// Driver-moved.
    Kinematic,
}

/// Collision shape mirror: plain data, recursive for compounds and
/// rounded shapes. Meshes carry their exact triangle soup (per-triangle
/// hulls plus centroids) so restore rebuilds the derived BVH
/// bit-identically from the same ordered input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShapeSnapshot {
    /// Uniform ball.
    Sphere {
        /// Surface radius.
        radius: f32,
    },
    /// Oriented box.
    Box {
        /// Half-size per local axis.
        half_extents: [f32; 3],
    },
    /// Capped cylinder along local +Y.
    Capsule {
        /// Cylinder/cap radius.
        radius: f32,
        /// Cylinder half-length excluding caps.
        half_height: f32,
    },
    /// Flat-capped cylinder along local +Y.
    Cylinder {
        /// End-disk radius.
        radius: f32,
        /// Half-height along +Y.
        half_height: f32,
    },
    /// Solid cone along local +Y.
    Cone {
        /// Base-disk radius.
        radius: f32,
        /// Half-height (apex at `+half_height`).
        half_height: f32,
    },
    /// Explicit convex polyhedron (body-local vertices + faces).
    ConvexHull {
        /// Deduplicated vertices.
        vertices: Vec<[f32; 3]>,
        /// Outward triangles as vertex indices.
        faces: Vec<[u32; 3]>,
    },
    /// Heightfield terrain grid.
    Heightfield {
        /// Row-major heights (`rows * cols` entries).
        heights: Vec<f32>,
        /// Sample count along +Z.
        rows: usize,
        /// Sample count along +X.
        cols: usize,
        /// Uniform grid spacing (m).
        cell: f32,
    },
    /// Triangle-soup mesh (per-triangle hulls + centroids + bounds; the
    /// BVH rebuilds deterministically from this ordered input).
    TriMesh {
        /// One convex hull per surviving triangle (centroid-relative).
        tris: Vec<ShapeSnapshot>,
        /// Triangle centroids (mesh-local placement origins).
        centroids: Vec<[f32; 3]>,
        /// Mesh-local bounding-box min.
        local_min: [f32; 3],
        /// Mesh-local bounding-box max.
        local_max: [f32; 3],
        /// Support radius about the body origin.
        bound_radius: f32,
        /// Smallest triangle edge over the soup.
        min_feature: f32,
    },
    /// Rigid union of placed children.
    Compound {
        /// Children with compound-local placements.
        shapes: Vec<(ShapeSnapshot, PoseSnapshot)>,
    },
    /// Minkowski-dilated inner shape.
    Round {
        /// Dilated inner shape.
        inner: Box<ShapeSnapshot>,
        /// Dilation radius (m).
        border_radius: f32,
    },
    /// Static infinite plane (unit outward normal, body-local).
    HalfSpace {
        /// Outward unit normal.
        normal: [f32; 3],
    },
}

/// Rigid placement mirror (compound children, driver baselines).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoseSnapshot {
    /// Child/step-start origin.
    pub position: [f32; 3],
    /// Child/step-start orientation (`xyzw`).
    pub rotation: [f32; 4],
}

/// One joint: spec plus assembly reference, motor override and the full
/// warm-start accumulator set, so restored joints continue with the exact
/// solver state вместо cold restart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointSnapshot {
    /// First body index.
    pub a: u32,
    /// Second body index.
    pub b: u32,
    /// Joint spec.
    pub spec: JointKindSnapshot,
    /// Assembly reference (angle, length, distance, quat, anchor delta).
    pub reference: JointReferenceSnapshot,
    /// Generalized motor override (`None` = spec motor applies).
    pub servo: Option<JointMotorSnapshot>,
    /// Accumulated linear impulses per world axis.
    pub acc_lin: [f32; 3],
    /// Accumulated angular impulses per constraint axis.
    pub acc_ang: [f32; 3],
    /// One-sided limit impulse.
    pub acc_limit: f32,
    /// Distance-rod impulse.
    pub acc_dist: f32,
    /// One-sided rope impulse.
    pub acc_rope: f32,
    /// Gear-constraint impulse.
    pub acc_gear: f32,
    /// Gear coordinate memory (raw + continuous, both sides).
    pub gear_mem: Option<([f32; 2], [f32; 2])>,
    /// Six-DOF limited-axis accumulators.
    pub acc_6dof: [f32; 6],
}

/// Assembly-reference mirror (hinge twist, slide length, rod/gear
/// constant, relative orientation, anchor separation).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JointReferenceSnapshot {
    /// Hinge twist at assembly (rad).
    pub angle: f32,
    /// Slide/spring rest length at assembly (m).
    pub length: f32,
    /// Distance-rod / gear constant at assembly (m).
    pub distance: f32,
    /// Relative orientation at assembly (`xyzw`).
    pub rotation: [f32; 4],
    /// Anchor separation at assembly (m, body-A frame).
    pub anchor_delta: [f32; 3],
}

/// Joint spec mirror (all ten joint kinds with their drives).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JointKindSnapshot {
    /// Ball-and-socket.
    Ball {
        /// Anchor in body A's frame.
        local_anchor_a: [f32; 3],
        /// Anchor in body B's frame.
        local_anchor_b: [f32; 3],
    },
    /// Hinge with optional limit/motor.
    Revolute {
        /// Hinge center in body A's frame.
        local_anchor_a: [f32; 3],
        /// Hinge center in body B's frame.
        local_anchor_b: [f32; 3],
        /// Hinge axis in body A's frame.
        local_axis_a: [f32; 3],
        /// Hinge axis in body B's frame.
        local_axis_b: [f32; 3],
        /// Travel window (rad, assembly-relative).
        limit: Option<RevoluteLimitSnapshot>,
        /// Hinge velocity motor.
        motor: Option<RevoluteMotorSnapshot>,
    },
    /// Slider with optional limit/motor.
    Prismatic {
        /// Slide origin in body A's frame.
        local_anchor_a: [f32; 3],
        /// Slide origin in body B's frame.
        local_anchor_b: [f32; 3],
        /// Slide axis in body A's frame.
        local_axis_a: [f32; 3],
        /// Slide axis in body B's frame.
        local_axis_b: [f32; 3],
        /// Travel window (m, assembly-relative).
        limit: Option<PrismaticLimitSnapshot>,
        /// Slide velocity motor.
        motor: Option<PrismaticMotorSnapshot>,
    },
    /// Weld.
    Fixed {
        /// Weld origin in body A's frame.
        local_anchor_a: [f32; 3],
        /// Weld origin in body B's frame.
        local_anchor_b: [f32; 3],
    },
    /// Rigid rod.
    Distance {
        /// Rod end in body A's frame.
        local_anchor_a: [f32; 3],
        /// Rod end in body B's frame.
        local_anchor_b: [f32; 3],
    },
    /// One-sided maximum-distance rope.
    Rope {
        /// Rope end in body A's frame.
        local_anchor_a: [f32; 3],
        /// Rope end in body B's frame.
        local_anchor_b: [f32; 3],
        /// Maximum separation (m).
        max_distance: f32,
    },
    /// Compliant distance row with inline spring motor.
    Spring {
        /// Spring end in body A's frame.
        local_anchor_a: [f32; 3],
        /// Spring end in body B's frame.
        local_anchor_b: [f32; 3],
        /// Spring-damper drive.
        motor: JointMotorSnapshot,
        /// Implicit vs explicit integration.
        integration: SpringIntegrationSnapshot,
    },
    /// Suspension slide plus free spin about the axle.
    Wheel {
        /// Suspension origin in body A's frame.
        local_anchor_a: [f32; 3],
        /// Suspension origin in body B's frame.
        local_anchor_b: [f32; 3],
        /// Suspension axis in body A's frame.
        local_suspension_a: [f32; 3],
        /// Suspension axis in body B's frame.
        local_suspension_b: [f32; 3],
        /// Spin axle in body A's frame.
        local_axle_a: [f32; 3],
        /// Spin axle in body B's frame.
        local_axle_b: [f32; 3],
        /// Spring parameters.
        suspension: WheelSuspensionSnapshot,
        /// Axle velocity motor.
        motor: Option<RevoluteMotorSnapshot>,
    },
    /// Coordinate coupling of two revolute/prismatic joints.
    Gear {
        /// First coordinated joint index.
        joint_a: u32,
        /// Second coordinated joint index.
        joint_b: u32,
        /// Transmission ratio.
        ratio: f32,
    },
    /// Per-axis free/locked/limited generic.
    SixDof {
        /// Joint origin in body A's frame.
        local_anchor_a: [f32; 3],
        /// Joint origin in body B's frame.
        local_anchor_b: [f32; 3],
        /// Linear-axis configurations.
        linear: [AxisConfigSnapshot; 3],
        /// Angular-axis configurations.
        angular: [AxisConfigSnapshot; 3],
    },
    /// Free 6-DOF velocity drive (no anchors — COM-level rows).
    Motor {
        /// Desired relative linear velocity (m/s, B-minus-A, world frame).
        linear_target: [f32; 3],
        /// Desired relative angular velocity (rad/s, B-minus-A, world frame).
        angular_target: [f32; 3],
        /// Linear force budget (N).
        max_force: f32,
        /// Angular torque budget (N·m).
        max_torque: f32,
        /// Assembly-pose pull per position pass (`0..=1`, SI only).
        correction: f32,
    },
}

/// Per-axis six-DOF configuration mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AxisConfigSnapshot {
    /// Free axis.
    Free,
    /// Locked axis.
    Locked,
    /// Travel window from the assembly pose.
    Limited {
        /// Lower bound (m or rad).
        min: f32,
        /// Upper bound (m or rad).
        max: f32,
    },
}

/// Prismatic travel-window mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrismaticLimitSnapshot {
    /// Lower bound (m, assembly-relative).
    pub min: f32,
    /// Upper bound (m, assembly-relative).
    pub max: f32,
}

/// Prismatic velocity-motor mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrismaticMotorSnapshot {
    /// Desired slide speed (m/s).
    pub target_speed: f32,
    /// Force budget.
    pub max_force: f32,
}

/// Revolute travel-window mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RevoluteLimitSnapshot {
    /// Lower bound (rad, assembly-relative).
    pub min: f32,
    /// Upper bound (rad, assembly-relative).
    pub max: f32,
}

/// Revolute velocity-motor mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RevoluteMotorSnapshot {
    /// Desired hinge speed (rad/s).
    pub target_speed: f32,
    /// Torque budget.
    pub max_torque: f32,
}

/// Wheel suspension-parameter mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WheelSuspensionSnapshot {
    /// Resonance frequency (Hz).
    pub frequency_hz: f32,
    /// Dimensionless damping ratio.
    pub damping_ratio: f32,
}

/// Generalized joint-motor mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JointMotorSnapshot {
    /// Control mode.
    pub kind: MotorKindSnapshot,
    /// Position target (assembly-relative).
    pub target_position: f32,
    /// Velocity target.
    pub target_velocity: f32,
    /// Spring constant.
    pub stiffness: f32,
    /// Damping coefficient.
    pub damping: f32,
    /// Force/torque budget.
    pub max_force: f32,
    /// Force vs acceleration interpretation.
    pub model: MotorModelSnapshot,
}

/// Motor control-mode mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotorKindSnapshot {
    /// Constant-speed drive.
    Velocity,
    /// Spring-damper to a position.
    Position,
    /// Position spring plus velocity tracking.
    Servo,
}

/// Motor spring-interpretation mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotorModelSnapshot {
    /// Mass-scaled (acceleration) targets.
    AccelerationBased,
    /// Absolute-force targets.
    ForceBased,
}

/// Spring-integration mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpringIntegrationSnapshot {
    /// Unconditionally stable implicit solve.
    Implicit,
    /// Cheap conditionally stable semi-explicit Euler.
    Explicit,
}

/// Solid-contact transition mirror (begin/end/hit with impact data).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactEventSnapshot {
    /// First body index.
    pub a: u32,
    /// Second body index.
    pub b: u32,
    /// Begin, end or hit.
    pub kind: ContactEventKindSnapshot,
}

/// Contact-transition kind mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ContactEventKindSnapshot {
    /// Pair started touching.
    Begin,
    /// Pair stopped touching.
    End,
    /// Hard impact with approach data.
    Hit {
        /// World contact point of the hardest approach.
        point: [f32; 3],
        /// Solved contact normal (A toward B).
        normal: [f32; 3],
        /// Closing speed along the normal (m/s, positive).
        approach_speed: f32,
    },
}

/// Trigger overlap-transition mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerEventSnapshot {
    /// First body index.
    pub a: u32,
    /// Second body index.
    pub b: u32,
    /// `true` = entered, `false` = exited.
    pub entered: bool,
}

/// Contact-force report mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactForceEventSnapshot {
    /// Lower body index of the canonical pair.
    pub a: u32,
    /// Higher body index of the canonical pair.
    pub b: u32,
    /// Pair force (N): last-substep normal impulse over substep length.
    pub force: f32,
    /// Deepest contact point (world space).
    pub point: [f32; 3],
}

/// Fracture-report mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FractureEventSnapshot {
    /// Pre-split (stale) parent index.
    pub parent: u32,
    /// Live piece indices in split order.
    pub pieces: [u32; 2],
}

/// Soft↔rigid transition mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SoftContactEventSnapshot {
    /// Soft body index.
    pub soft: u32,
    /// Rigid body index.
    pub body: u32,
    /// Begin or end.
    pub kind: ContactEventKindSnapshot,
}

/// One warm-cache entry: canonical pair plus matched per-point impulses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WarmPairSnapshot {
    /// Canonical pair `(min, max)` body indices.
    pub pair: (usize, usize),
    /// Matched contact points with accumulated normal impulses.
    pub points: Vec<WarmPointSnapshot>,
}

/// One warm-started contact point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WarmPointSnapshot {
    /// Body-frame anchor on body A.
    pub la: [f32; 3],
    /// Body-frame anchor on body B.
    pub lb: [f32; 3],
    /// Contact normal at cache time.
    pub normal: [f32; 3],
    /// Accumulated normal impulse.
    pub impulse: f32,
}

/// Deformable-body mirror: persistent particles, topology and surface
/// data. Substep-live multipliers (`lambda`, `volume_lambda`,
/// self-collision) are intentionally absent — `begin_substep` resets
/// them every substep, so carrying them would be dead weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftBodySnapshot {
    /// Particles in index order.
    pub particles: Vec<ParticleSnapshot>,
    /// Distance rows.
    pub constraints: Vec<DeformConstraintSnapshot>,
    /// Closed volume-surface triangles (particle indices).
    pub triangles: Vec<[u32; 3]>,
    /// Render-only open-sheet topology (particle indices).
    pub surface: Vec<[u32; 3]>,
    /// Rest volume (m³).
    pub volume_rest: f32,
    /// Volume compliance.
    pub volume_compliance: f32,
    /// Velocity damping rate (1/s).
    pub damping: f32,
    /// Coupling particle radius (m, 0 = no rigid coupling).
    pub contact_radius: f32,
    /// Breakage stretch ratio (0 = unbreakable).
    pub tear_strain: f32,
    /// Collision layer bits.
    pub collision_layer: u32,
    /// Collision mask bits.
    pub collision_mask: u32,
}

/// Point-mass mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParticleSnapshot {
    /// World-space position.
    pub position: [f32; 3],
    /// Substep-start position (velocity baseline).
    pub prev_position: [f32; 3],
    /// Linear velocity (m/s).
    pub velocity: [f32; 3],
    /// Cached inverse mass (0 = pinned).
    pub inv_mass: f32,
}

/// Distance-row mirror (topology + rest state, no live multiplier).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeformConstraintSnapshot {
    /// First particle index.
    pub a: u32,
    /// Second particle index.
    pub b: u32,
    /// Rest length (m).
    pub rest: f32,
    /// Compliance (m/N, 0 = inextensible).
    pub compliance: f32,
    /// Topology group.
    pub kind: DeformKindSnapshot,
}

/// Constraint-group mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeformKindSnapshot {
    /// Primary topology.
    Structural,
    /// Shear diagonals.
    Shear,
    /// Skip-one bending rows.
    Bend,
}

fn v3(v: glam::Vec3) -> [f32; 3] {
    v.to_array()
}

fn q4(q: glam::Quat) -> [f32; 4] {
    [q.x, q.y, q.z, q.w]
}

fn vec3(v: [f32; 3]) -> glam::Vec3 {
    glam::Vec3::from_array(v)
}

fn quat(q: [f32; 4]) -> glam::Quat {
    glam::Quat::from_xyzw(q[0], q[1], q[2], q[3])
}

fn finite(v: f32, what: &str) -> Result<f32, SnapshotError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(SnapshotError::InvalidData {
            detail: format!("non-finite `{what}`"),
        })
    }
}

fn finite_array<const N: usize>(v: [f32; N], what: &str) -> Result<[f32; N], SnapshotError> {
    for x in v {
        finite(x, what)?;
    }
    Ok(v)
}

impl WorldSnapshot {
    /// Serializes this snapshot to compact RON (deterministic field
    /// order; scenes in this workspace already speak RON).
    ///
    /// # Errors
    ///
    /// [`SnapshotError::Decode`] when the serializer rejects the data
    /// (only on genuinely unserializable input — plain data never hits
    /// this in practice).
    pub fn to_ron(&self) -> Result<String, SnapshotError> {
        ron::ser::to_string(self).map_err(|e| SnapshotError::Decode {
            detail: e.to_string(),
        })
    }

    /// Parses a snapshot from RON and gates it on
    /// [`WORLD_SNAPSHOT_VERSION`]: a version mismatch is a loud
    /// [`SnapshotError::VersionMismatch`], never a silent reinterpretation.
    /// Unknown extra fields are ignored (forward-tolerant); missing or
    /// mistyped fields fail as [`SnapshotError::Decode`].
    ///
    /// # Errors
    ///
    /// [`SnapshotError::VersionMismatch`] on a version skew,
    /// [`SnapshotError::Decode`] on malformed input.
    pub fn from_ron(text: &str) -> Result<Self, SnapshotError> {
        let snap: Self = ron::de::from_str(text).map_err(|e| SnapshotError::Decode {
            detail: e.to_string(),
        })?;
        if snap.version != WORLD_SNAPSHOT_VERSION {
            return Err(SnapshotError::VersionMismatch {
                expected: WORLD_SNAPSHOT_VERSION,
                found: snap.version,
            });
        }
        Ok(snap)
    }

    /// Builds a fresh sequential-impulse engine holding exactly this
    /// snapshot's state: same tuning, same bodies/joints/warm/sleep/event
    /// state, so stepping it continues the captured trajectory
    /// bit-identically.
    ///
    /// # Errors
    ///
    /// [`SnapshotError::InvalidData`] on dangling handles, non-finite
    /// scalars or rejected shapes/joints; [`SnapshotError::Unsupported`]
    /// when the snapshot carries soft bodies (a bare
    /// sequential-impulse engine cannot own them — restore through
    /// [`Engine::restore_world_snapshot`](crate::Engine::restore_world_snapshot),
    /// which re-parks them).
    pub fn instantiate_si(&self) -> Result<crate::SequentialImpulseEngine, SnapshotError> {
        if !self.soft_bodies.is_empty() {
            return Err(SnapshotError::Unsupported {
                detail:
                    "snapshot holds soft bodies: restore through Engine::restore_world_snapshot"
                        .to_string(),
            });
        }
        let mut engine = crate::SequentialImpulseEngine::new(vec3(self.gravity));
        engine.restore_world(self)?;
        Ok(engine)
    }
}

impl BodyTypeSnapshot {
    pub(crate) fn from_type(t: BodyType) -> Self {
        match t {
            BodyType::Static => Self::Static,
            BodyType::Dynamic => Self::Dynamic,
            BodyType::Kinematic => Self::Kinematic,
        }
    }

    pub(crate) fn to_type(self) -> BodyType {
        match self {
            Self::Static => BodyType::Static,
            Self::Dynamic => BodyType::Dynamic,
            Self::Kinematic => BodyType::Kinematic,
        }
    }
}

impl PoseSnapshot {
    /// Builds a snapshot pose from raw parts (driver baselines carry the
    /// same data under different field names).
    pub(crate) fn from_parts(position: glam::Vec3, rotation: glam::Quat) -> Self {
        Self {
            position: v3(position),
            rotation: q4(rotation),
        }
    }

    pub(crate) fn from_pose(p: Pose) -> Self {
        Self::from_parts(p.position, p.rotation)
    }

    pub(crate) fn to_pose(self) -> Pose {
        Pose {
            position: vec3(self.position),
            rotation: quat(self.rotation),
        }
    }
}

impl ShapeSnapshot {
    /// Mirrors a live shape into plain data (every variant, recursively).
    pub(crate) fn from_shape(shape: &Shape) -> Self {
        match shape {
            Shape::Sphere { radius } => Self::Sphere { radius: *radius },
            Shape::Box { half_extents } => Self::Box {
                half_extents: v3(*half_extents),
            },
            Shape::Capsule {
                radius,
                half_height,
            } => Self::Capsule {
                radius: *radius,
                half_height: *half_height,
            },
            Shape::Cylinder {
                radius,
                half_height,
            } => Self::Cylinder {
                radius: *radius,
                half_height: *half_height,
            },
            Shape::Cone {
                radius,
                half_height,
            } => Self::Cone {
                radius: *radius,
                half_height: *half_height,
            },
            Shape::ConvexHull(hull) => Self::ConvexHull {
                vertices: hull.vertices.iter().map(|v| v3(*v)).collect(),
                faces: hull.faces.iter().map(|t| t.as_u32()).collect(),
            },
            Shape::Heightfield(hf) => Self::Heightfield {
                heights: hf.heights.clone(),
                rows: hf.rows,
                cols: hf.cols,
                cell: hf.cell,
            },
            Shape::TriMesh(mesh) => Self::TriMesh {
                tris: mesh.tris.iter().map(Self::from_shape).collect(),
                centroids: mesh.centroids.iter().map(|v| v3(*v)).collect(),
                local_min: v3(mesh.local_min),
                local_max: v3(mesh.local_max),
                bound_radius: mesh.bound_radius,
                min_feature: mesh.min_feature,
            },
            Shape::Compound { shapes } => Self::Compound {
                shapes: shapes
                    .iter()
                    .map(|(s, p)| (Self::from_shape(s), PoseSnapshot::from_pose(*p)))
                    .collect(),
            },
            Shape::Round {
                inner,
                border_radius,
            } => Self::Round {
                inner: Box::new(Self::from_shape(inner)),
                border_radius: *border_radius,
            },
            Shape::HalfSpace { normal } => Self::HalfSpace {
                normal: v3(normal.get()),
            },
        }
    }

    /// Rebuilds the live shape. Constructions that went through checked
    /// builders are rebuilt through the same builders (same deterministic
    /// output); `pub`-field structs (`ConvexHull`, `Heightfield`) are
    /// rebuilt literally with explicit validation. The mesh BVH rebuilds
    /// from the snapshotted ordered triangles, so it comes back
    /// bit-identical.
    ///
    /// # Errors
    ///
    /// [`SnapshotError::InvalidData`] on non-finite scalars, empty
    /// compounds, dangling mesh triangles or rejected half-space normals.
    pub(crate) fn to_shape(&self) -> Result<Shape, SnapshotError> {
        let bad = |detail: &str| SnapshotError::InvalidData {
            detail: detail.to_string(),
        };
        match self {
            Self::Sphere { radius } => Ok(Shape::Sphere {
                radius: finite(*radius, "sphere radius")?,
            }),
            Self::Box { half_extents } => Ok(Shape::Box {
                half_extents: vec3(finite_array(*half_extents, "box half_extents")?),
            }),
            Self::Capsule {
                radius,
                half_height,
            } => Ok(Shape::Capsule {
                radius: finite(*radius, "capsule radius")?,
                half_height: finite(*half_height, "capsule half_height")?,
            }),
            Self::Cylinder {
                radius,
                half_height,
            } => Ok(Shape::Cylinder {
                radius: finite(*radius, "cylinder radius")?,
                half_height: finite(*half_height, "cylinder half_height")?,
            }),
            Self::Cone {
                radius,
                half_height,
            } => Ok(Shape::Cone {
                radius: finite(*radius, "cone radius")?,
                half_height: finite(*half_height, "cone half_height")?,
            }),
            Self::ConvexHull { vertices, faces } => {
                let mut verts = Vec::with_capacity(vertices.len());
                for v in vertices {
                    verts.push(vec3(finite_array(*v, "hull vertex")?));
                }
                let mut tris = Vec::with_capacity(faces.len());
                for f in faces {
                    tris.push(Triangle::from_raw(*f));
                }
                Ok(Shape::ConvexHull(ConvexHull {
                    vertices: verts,
                    faces: tris,
                }))
            }
            Self::Heightfield {
                heights,
                rows,
                cols,
                cell,
            } => {
                if *rows == 0 || *cols == 0 {
                    return Err(bad("heightfield grid is empty"));
                }
                if rows.checked_mul(*cols) != Some(heights.len()) {
                    return Err(bad("heightfield heights.len() != rows * cols"));
                }
                if !cell.is_finite() || *cell <= 0.0 {
                    return Err(bad("heightfield cell must be positive finite"));
                }
                if heights.iter().any(|h| !h.is_finite()) {
                    return Err(bad("heightfield has non-finite samples"));
                }
                Ok(Shape::Heightfield(Heightfield {
                    heights: heights.clone(),
                    rows: *rows,
                    cols: *cols,
                    cell: *cell,
                }))
            }
            Self::TriMesh {
                tris,
                centroids,
                local_min,
                local_max,
                bound_radius,
                min_feature,
            } => {
                if tris.len() != centroids.len() {
                    return Err(bad("trimesh tris/centroids length mismatch"));
                }
                let mut live_tris = Vec::with_capacity(tris.len());
                for t in tris {
                    let shape = t.to_shape()?;
                    if !matches!(shape, Shape::ConvexHull(_)) {
                        return Err(bad("trimesh child is not a convex hull"));
                    }
                    live_tris.push(shape);
                }
                let live_centroids: Vec<glam::Vec3> = centroids
                    .iter()
                    .map(|c| finite_array(*c, "trimesh centroid").map(vec3))
                    .collect::<Result<_, _>>()?;
                let (nodes, order) = TriMesh::build_bvh(&live_tris, &live_centroids);
                Ok(Shape::TriMesh(TriMesh {
                    tris: live_tris,
                    centroids: live_centroids,
                    nodes,
                    order,
                    local_min: vec3(finite_array(*local_min, "trimesh local_min")?),
                    local_max: vec3(finite_array(*local_max, "trimesh local_max")?),
                    bound_radius: finite(*bound_radius, "trimesh bound_radius")?,
                    min_feature: finite(*min_feature, "trimesh min_feature")?,
                }))
            }
            Self::Compound { shapes } => {
                if shapes.is_empty() {
                    return Err(bad("compound shape has no children"));
                }
                let mut live = Vec::with_capacity(shapes.len());
                for (s, p) in shapes {
                    live.push((s.to_shape()?, p.to_pose()));
                }
                Ok(Shape::Compound { shapes: live })
            }
            Self::Round {
                inner,
                border_radius,
            } => Ok(Shape::Round {
                inner: Box::new(inner.to_shape()?),
                border_radius: finite(*border_radius, "round border_radius")?,
            }),
            Self::HalfSpace { normal } => {
                let n = vec3(finite_array(*normal, "half-space normal")?);
                // The stored normal came from a validated unit vector;
                // re-wrapping must not re-normalize (that would cost a bit
                // of precision for nothing).
                crate::invariants::UnitVec3::try_from(n)
                    .map(|_| Shape::HalfSpace {
                        // SAFETY: `try_from` just proved unit length on the
                        // same value; the unchecked wrap preserves its bits
                        // exactly instead of dividing by ~1.0.
                        normal: unsafe { crate::invariants::UnitVec3::from_unit_unchecked(n) },
                    })
                    .ok_or_else(|| bad("half-space normal is not unit length"))
            }
        }
    }
}

impl BodySnapshot {
    /// Mirrors a live body into plain data (every field verbatim,
    /// including sleep-zeroed inverse masses).
    pub(crate) fn from_body(body: &RigidBody) -> Self {
        Self {
            position: v3(body.position),
            orientation: q4(body.orientation),
            velocity: v3(body.velocity),
            angular_velocity: v3(body.angular_velocity),
            mass: body.mass,
            inv_mass: body.inv_mass,
            inertia: v3(body.inertia),
            torque: v3(body.torque),
            restitution: body.restitution,
            friction: body.friction,
            friction_transverse: body.friction_transverse,
            friction_dir: body.friction_dir.map(v3),
            rolling_friction: body.rolling_friction,
            torsion_friction: body.torsion_friction,
            shape: ShapeSnapshot::from_shape(&body.shape),
            collision_layer: body.collision_layer,
            collision_mask: body.collision_mask,
            is_trigger: body.is_trigger,
            fracture_impact_speed: body.fracture_impact_speed,
            body_type: BodyTypeSnapshot::from_type(body.body_type),
            ccd_enabled: body.ccd_enabled,
            contact_force_threshold: body.contact_force_threshold,
        }
    }

    /// Rebuilds the live body with every field verbatim (no mass-model
    /// re-derivation: a sleeper's zeroed triple restores zeroed).
    ///
    /// # Errors
    ///
    /// [`SnapshotError::InvalidData`] on a rejected shape (scalars stay
    /// unchecked here — a live body may legitimately hold infinities such
    /// as the default fracture threshold; only the shape constructors
    /// validate).
    pub(crate) fn to_body(&self) -> Result<RigidBody, SnapshotError> {
        Ok(RigidBody {
            position: vec3(self.position),
            orientation: quat(self.orientation),
            velocity: vec3(self.velocity),
            angular_velocity: vec3(self.angular_velocity),
            mass: self.mass,
            inv_mass: self.inv_mass,
            inertia: vec3(self.inertia),
            torque: vec3(self.torque),
            restitution: self.restitution,
            friction: self.friction,
            friction_transverse: self.friction_transverse,
            friction_dir: self.friction_dir.map(vec3),
            rolling_friction: self.rolling_friction,
            torsion_friction: self.torsion_friction,
            shape: self.shape.to_shape()?,
            collision_layer: self.collision_layer,
            collision_mask: self.collision_mask,
            is_trigger: self.is_trigger,
            fracture_impact_speed: self.fracture_impact_speed,
            body_type: self.body_type.to_type(),
            ccd_enabled: self.ccd_enabled,
            contact_force_threshold: self.contact_force_threshold,
        })
    }
}

impl AxisConfigSnapshot {
    pub(crate) fn from_config(c: AxisConfig) -> Self {
        match c {
            AxisConfig::Free => Self::Free,
            AxisConfig::Locked => Self::Locked,
            AxisConfig::Limited { min, max } => Self::Limited { min, max },
        }
    }

    pub(crate) fn to_config(self) -> AxisConfig {
        match self {
            Self::Free => AxisConfig::Free,
            Self::Locked => AxisConfig::Locked,
            Self::Limited { min, max } => AxisConfig::Limited { min, max },
        }
    }
}

impl JointMotorSnapshot {
    pub(crate) fn from_motor(m: JointMotor) -> Self {
        Self {
            kind: match m.kind {
                MotorKind::Velocity => MotorKindSnapshot::Velocity,
                MotorKind::Position => MotorKindSnapshot::Position,
                MotorKind::Servo => MotorKindSnapshot::Servo,
            },
            target_position: m.target_position,
            target_velocity: m.target_velocity,
            stiffness: m.stiffness,
            damping: m.damping,
            max_force: m.max_force,
            model: match m.model {
                MotorModel::AccelerationBased => MotorModelSnapshot::AccelerationBased,
                MotorModel::ForceBased => MotorModelSnapshot::ForceBased,
            },
        }
    }

    pub(crate) fn to_motor(self) -> JointMotor {
        JointMotor {
            kind: match self.kind {
                MotorKindSnapshot::Velocity => MotorKind::Velocity,
                MotorKindSnapshot::Position => MotorKind::Position,
                MotorKindSnapshot::Servo => MotorKind::Servo,
            },
            target_position: self.target_position,
            target_velocity: self.target_velocity,
            stiffness: self.stiffness,
            damping: self.damping,
            max_force: self.max_force,
            model: match self.model {
                MotorModelSnapshot::AccelerationBased => MotorModel::AccelerationBased,
                MotorModelSnapshot::ForceBased => MotorModel::ForceBased,
            },
        }
    }
}

impl JointKindSnapshot {
    /// Mirrors a live joint spec into plain data.
    pub(crate) fn from_spec(kind: JointKind) -> Self {
        match kind {
            JointKind::Ball {
                local_anchor_a,
                local_anchor_b,
            } => Self::Ball {
                local_anchor_a: v3(local_anchor_a),
                local_anchor_b: v3(local_anchor_b),
            },
            JointKind::Revolute {
                local_anchor_a,
                local_anchor_b,
                local_axis_a,
                local_axis_b,
                limit,
                motor,
            } => Self::Revolute {
                local_anchor_a: v3(local_anchor_a),
                local_anchor_b: v3(local_anchor_b),
                local_axis_a: v3(local_axis_a),
                local_axis_b: v3(local_axis_b),
                limit: limit.map(|l| RevoluteLimitSnapshot {
                    min: l.min,
                    max: l.max,
                }),
                motor: motor.map(|m| RevoluteMotorSnapshot {
                    target_speed: m.target_speed,
                    max_torque: m.max_torque,
                }),
            },
            JointKind::Prismatic {
                local_anchor_a,
                local_anchor_b,
                local_axis_a,
                local_axis_b,
                limit,
                motor,
            } => Self::Prismatic {
                local_anchor_a: v3(local_anchor_a),
                local_anchor_b: v3(local_anchor_b),
                local_axis_a: v3(local_axis_a),
                local_axis_b: v3(local_axis_b),
                limit: limit.map(|l| PrismaticLimitSnapshot {
                    min: l.min,
                    max: l.max,
                }),
                motor: motor.map(|m| PrismaticMotorSnapshot {
                    target_speed: m.target_speed,
                    max_force: m.max_force,
                }),
            },
            JointKind::Fixed {
                local_anchor_a,
                local_anchor_b,
            } => Self::Fixed {
                local_anchor_a: v3(local_anchor_a),
                local_anchor_b: v3(local_anchor_b),
            },
            JointKind::Distance {
                local_anchor_a,
                local_anchor_b,
            } => Self::Distance {
                local_anchor_a: v3(local_anchor_a),
                local_anchor_b: v3(local_anchor_b),
            },
            JointKind::Rope {
                local_anchor_a,
                local_anchor_b,
                max_distance,
            } => Self::Rope {
                local_anchor_a: v3(local_anchor_a),
                local_anchor_b: v3(local_anchor_b),
                max_distance,
            },
            JointKind::Spring {
                local_anchor_a,
                local_anchor_b,
                motor,
                integration,
            } => Self::Spring {
                local_anchor_a: v3(local_anchor_a),
                local_anchor_b: v3(local_anchor_b),
                motor: JointMotorSnapshot::from_motor(motor),
                integration: match integration {
                    SpringIntegration::Implicit => SpringIntegrationSnapshot::Implicit,
                    SpringIntegration::Explicit => SpringIntegrationSnapshot::Explicit,
                },
            },
            JointKind::Wheel {
                local_anchor_a,
                local_anchor_b,
                local_suspension_a,
                local_suspension_b,
                local_axle_a,
                local_axle_b,
                suspension,
                motor,
            } => Self::Wheel {
                local_anchor_a: v3(local_anchor_a),
                local_anchor_b: v3(local_anchor_b),
                local_suspension_a: v3(local_suspension_a),
                local_suspension_b: v3(local_suspension_b),
                local_axle_a: v3(local_axle_a),
                local_axle_b: v3(local_axle_b),
                suspension: WheelSuspensionSnapshot {
                    frequency_hz: suspension.frequency_hz,
                    damping_ratio: suspension.damping_ratio,
                },
                motor: motor.map(|m| RevoluteMotorSnapshot {
                    target_speed: m.target_speed,
                    max_torque: m.max_torque,
                }),
            },
            JointKind::Gear {
                joint_a,
                joint_b,
                ratio,
            } => Self::Gear {
                joint_a: joint_a.as_u32(),
                joint_b: joint_b.as_u32(),
                ratio,
            },
            JointKind::SixDof {
                local_anchor_a,
                local_anchor_b,
                linear,
                angular,
            } => Self::SixDof {
                local_anchor_a: v3(local_anchor_a),
                local_anchor_b: v3(local_anchor_b),
                linear: linear.map(AxisConfigSnapshot::from_config),
                angular: angular.map(AxisConfigSnapshot::from_config),
            },
            JointKind::Motor {
                linear_target,
                angular_target,
                max_force,
                max_torque,
                correction,
            } => Self::Motor {
                linear_target: v3(linear_target),
                angular_target: v3(angular_target),
                max_force,
                max_torque,
                correction,
            },
        }
    }

    /// Rebuilds the live spec. Scalar validation (finite axes, ordered
    /// bounds, known gear refs) happens in the restore path, which owns
    /// the handle-space context — this only translates shapes.
    pub(crate) fn to_spec(&self) -> JointKind {
        match *self {
            Self::Ball {
                local_anchor_a,
                local_anchor_b,
            } => JointKind::Ball {
                local_anchor_a: vec3(local_anchor_a),
                local_anchor_b: vec3(local_anchor_b),
            },
            Self::Revolute {
                local_anchor_a,
                local_anchor_b,
                local_axis_a,
                local_axis_b,
                limit,
                motor,
            } => JointKind::Revolute {
                local_anchor_a: vec3(local_anchor_a),
                local_anchor_b: vec3(local_anchor_b),
                local_axis_a: vec3(local_axis_a),
                local_axis_b: vec3(local_axis_b),
                limit: limit.map(|l| RevoluteLimit {
                    min: l.min,
                    max: l.max,
                }),
                motor: motor.map(|m| RevoluteMotor {
                    target_speed: m.target_speed,
                    max_torque: m.max_torque,
                }),
            },
            Self::Prismatic {
                local_anchor_a,
                local_anchor_b,
                local_axis_a,
                local_axis_b,
                limit,
                motor,
            } => JointKind::Prismatic {
                local_anchor_a: vec3(local_anchor_a),
                local_anchor_b: vec3(local_anchor_b),
                local_axis_a: vec3(local_axis_a),
                local_axis_b: vec3(local_axis_b),
                limit: limit.map(|l| PrismaticLimit {
                    min: l.min,
                    max: l.max,
                }),
                motor: motor.map(|m| PrismaticMotor {
                    target_speed: m.target_speed,
                    max_force: m.max_force,
                }),
            },
            Self::Fixed {
                local_anchor_a,
                local_anchor_b,
            } => JointKind::Fixed {
                local_anchor_a: vec3(local_anchor_a),
                local_anchor_b: vec3(local_anchor_b),
            },
            Self::Distance {
                local_anchor_a,
                local_anchor_b,
            } => JointKind::Distance {
                local_anchor_a: vec3(local_anchor_a),
                local_anchor_b: vec3(local_anchor_b),
            },
            Self::Rope {
                local_anchor_a,
                local_anchor_b,
                max_distance,
            } => JointKind::Rope {
                local_anchor_a: vec3(local_anchor_a),
                local_anchor_b: vec3(local_anchor_b),
                max_distance,
            },
            Self::Spring {
                local_anchor_a,
                local_anchor_b,
                motor,
                integration,
            } => JointKind::Spring {
                local_anchor_a: vec3(local_anchor_a),
                local_anchor_b: vec3(local_anchor_b),
                motor: motor.to_motor(),
                integration: match integration {
                    SpringIntegrationSnapshot::Implicit => SpringIntegration::Implicit,
                    SpringIntegrationSnapshot::Explicit => SpringIntegration::Explicit,
                },
            },
            Self::Wheel {
                local_anchor_a,
                local_anchor_b,
                local_suspension_a,
                local_suspension_b,
                local_axle_a,
                local_axle_b,
                suspension,
                motor,
            } => JointKind::Wheel {
                local_anchor_a: vec3(local_anchor_a),
                local_anchor_b: vec3(local_anchor_b),
                local_suspension_a: vec3(local_suspension_a),
                local_suspension_b: vec3(local_suspension_b),
                local_axle_a: vec3(local_axle_a),
                local_axle_b: vec3(local_axle_b),
                suspension: WheelSuspension {
                    frequency_hz: suspension.frequency_hz,
                    damping_ratio: suspension.damping_ratio,
                },
                motor: motor.map(|m| RevoluteMotor {
                    target_speed: m.target_speed,
                    max_torque: m.max_torque,
                }),
            },
            Self::Gear {
                joint_a,
                joint_b,
                ratio,
            } => JointKind::Gear {
                joint_a: JointHandle::from_raw(joint_a),
                joint_b: JointHandle::from_raw(joint_b),
                ratio,
            },
            Self::SixDof {
                local_anchor_a,
                local_anchor_b,
                linear,
                angular,
            } => JointKind::SixDof {
                local_anchor_a: vec3(local_anchor_a),
                local_anchor_b: vec3(local_anchor_b),
                linear: linear.map(AxisConfigSnapshot::to_config),
                angular: angular.map(AxisConfigSnapshot::to_config),
            },
            Self::Motor {
                linear_target,
                angular_target,
                max_force,
                max_torque,
                correction,
            } => JointKind::Motor {
                linear_target: vec3(linear_target),
                angular_target: vec3(angular_target),
                max_force,
                max_torque,
                correction,
            },
        }
    }
}

impl ContactEventKindSnapshot {
    pub(crate) fn from_kind(kind: ContactEventKind) -> Self {
        match kind {
            ContactEventKind::Begin => Self::Begin,
            ContactEventKind::End => Self::End,
            ContactEventKind::Hit {
                point,
                normal,
                approach_speed,
            } => Self::Hit {
                point: v3(point),
                normal: v3(normal),
                approach_speed,
            },
        }
    }

    pub(crate) fn to_kind(self) -> ContactEventKind {
        match self {
            Self::Begin => ContactEventKind::Begin,
            Self::End => ContactEventKind::End,
            Self::Hit {
                point,
                normal,
                approach_speed,
            } => ContactEventKind::Hit {
                point: vec3(point),
                normal: vec3(normal),
                approach_speed,
            },
        }
    }
}

impl ContactEventSnapshot {
    pub(crate) fn from_event(e: &ContactEvent) -> Self {
        Self {
            a: e.body_a.as_u32(),
            b: e.body_b.as_u32(),
            kind: ContactEventKindSnapshot::from_kind(e.kind),
        }
    }

    pub(crate) fn to_event(self) -> ContactEvent {
        ContactEvent {
            body_a: BodyHandle::from_raw(self.a),
            body_b: BodyHandle::from_raw(self.b),
            kind: self.kind.to_kind(),
        }
    }
}

impl TriggerEventSnapshot {
    pub(crate) fn from_event(e: &TriggerEvent) -> Self {
        Self {
            a: e.body_a.as_u32(),
            b: e.body_b.as_u32(),
            entered: e.kind == TriggerEventKind::Entered,
        }
    }

    pub(crate) fn to_event(self) -> TriggerEvent {
        TriggerEvent {
            body_a: BodyHandle::from_raw(self.a),
            body_b: BodyHandle::from_raw(self.b),
            kind: if self.entered {
                TriggerEventKind::Entered
            } else {
                TriggerEventKind::Exited
            },
        }
    }
}

impl ContactForceEventSnapshot {
    pub(crate) fn from_event(e: &ContactForceEvent) -> Self {
        Self {
            a: e.a.as_u32(),
            b: e.b.as_u32(),
            force: e.force,
            point: v3(e.point),
        }
    }

    pub(crate) fn to_event(self) -> ContactForceEvent {
        ContactForceEvent {
            a: BodyHandle::from_raw(self.a),
            b: BodyHandle::from_raw(self.b),
            force: self.force,
            point: vec3(self.point),
        }
    }
}

impl FractureEventSnapshot {
    pub(crate) fn from_event(e: &crate::trigger::FractureEvent) -> Self {
        Self {
            parent: e.parent.as_u32(),
            pieces: [e.pieces[0].as_u32(), e.pieces[1].as_u32()],
        }
    }

    pub(crate) fn to_event(self) -> crate::trigger::FractureEvent {
        crate::trigger::FractureEvent {
            parent: BodyHandle::from_raw(self.parent),
            pieces: [
                BodyHandle::from_raw(self.pieces[0]),
                BodyHandle::from_raw(self.pieces[1]),
            ],
        }
    }
}

impl SoftContactEventSnapshot {
    pub(crate) fn from_event(e: &SoftContactEvent) -> Self {
        Self {
            soft: e.soft.as_u32(),
            body: e.body.as_u32(),
            kind: ContactEventKindSnapshot::from_kind(e.kind),
        }
    }

    pub(crate) fn to_event(self) -> SoftContactEvent {
        SoftContactEvent {
            soft: crate::soft::SoftHandle::from_raw(self.soft),
            body: BodyHandle::from_raw(self.body),
            kind: self.kind.to_kind(),
        }
    }
}

impl WarmPointSnapshot {
    pub(crate) fn from_point(p: &crate::sequential_impulse::WarmPoint) -> Self {
        Self {
            la: v3(p.la),
            lb: v3(p.lb),
            normal: v3(p.normal),
            impulse: p.impulse,
        }
    }

    pub(crate) fn to_point(self) -> crate::sequential_impulse::WarmPoint {
        crate::sequential_impulse::WarmPoint {
            la: vec3(self.la),
            lb: vec3(self.lb),
            normal: vec3(self.normal),
            impulse: self.impulse,
        }
    }
}

impl DeformKindSnapshot {
    fn from_kind(k: DeformKind) -> Self {
        match k {
            DeformKind::Structural => Self::Structural,
            DeformKind::Shear => Self::Shear,
            DeformKind::Bend => Self::Bend,
        }
    }

    fn to_kind(self) -> DeformKind {
        match self {
            Self::Structural => DeformKind::Structural,
            Self::Shear => DeformKind::Shear,
            Self::Bend => DeformKind::Bend,
        }
    }
}

impl SoftBodySnapshot {
    /// Mirrors a parked soft body into plain data (persistent state
    /// only — the substep-live multipliers are `begin_substep` scratch).
    pub(crate) fn from_soft(body: &SoftBody) -> Self {
        Self {
            particles: body
                .particles
                .iter()
                .map(|p| ParticleSnapshot {
                    position: v3(p.position),
                    prev_position: v3(p.prev_position),
                    velocity: v3(p.velocity),
                    inv_mass: p.inv_mass,
                })
                .collect(),
            constraints: body
                .constraints
                .iter()
                .map(|c| DeformConstraintSnapshot {
                    a: c.a.as_u32(),
                    b: c.b.as_u32(),
                    rest: c.rest,
                    compliance: c.compliance,
                    kind: DeformKindSnapshot::from_kind(c.kind),
                })
                .collect(),
            triangles: body
                .triangles
                .iter()
                .map(|t| [t[0].as_u32(), t[1].as_u32(), t[2].as_u32()])
                .collect(),
            surface: body
                .surface
                .iter()
                .map(|t| [t[0].as_u32(), t[1].as_u32(), t[2].as_u32()])
                .collect(),
            volume_rest: body.volume_rest,
            volume_compliance: body.volume_compliance,
            damping: body.damping,
            contact_radius: body.contact_radius,
            tear_strain: body.tear_strain,
            collision_layer: body.collision_layer,
            collision_mask: body.collision_mask,
        }
    }

    /// Rebuilds the parked body. Particle indices are validated against
    /// the restored particle count; triangle windings are preserved
    /// verbatim.
    ///
    /// # Errors
    ///
    /// [`SnapshotError::InvalidData`] on dangling particle indices.
    pub(crate) fn to_soft(&self) -> Result<SoftBody, SnapshotError> {
        let bad = |detail: &str| SnapshotError::InvalidData {
            detail: detail.to_string(),
        };
        let n = self.particles.len();
        let particles: Vec<Particle> = self
            .particles
            .iter()
            .map(|p| Particle {
                position: vec3(p.position),
                prev_position: vec3(p.prev_position),
                velocity: vec3(p.velocity),
                inv_mass: p.inv_mass,
            })
            .collect();
        let idx = |v: u32, what: &str| {
            if (v as usize) < n {
                Ok(ParticleIdx::from_raw(v))
            } else {
                Err(bad(what))
            }
        };
        let mut constraints = Vec::with_capacity(self.constraints.len());
        for c in &self.constraints {
            constraints.push(DeformConstraint {
                a: idx(c.a, "soft constraint index out of range")?,
                b: idx(c.b, "soft constraint index out of range")?,
                rest: c.rest,
                compliance: c.compliance,
                kind: c.kind.to_kind(),
                lambda: 0.0,
            });
        }
        let mut tris = Vec::with_capacity(self.triangles.len());
        for t in &self.triangles {
            tris.push([
                idx(t[0], "soft triangle index out of range")?,
                idx(t[1], "soft triangle index out of range")?,
                idx(t[2], "soft triangle index out of range")?,
            ]);
        }
        let mut surface = Vec::with_capacity(self.surface.len());
        for t in &self.surface {
            surface.push([
                idx(t[0], "soft surface index out of range")?,
                idx(t[1], "soft surface index out of range")?,
                idx(t[2], "soft surface index out of range")?,
            ]);
        }
        Ok(SoftBody {
            particles,
            constraints,
            triangles: tris,
            surface,
            volume_rest: self.volume_rest,
            volume_compliance: self.volume_compliance,
            volume_lambda: 0.0,
            damping: self.damping,
            contact_radius: self.contact_radius,
            tear_strain: self.tear_strain,
            collision_layer: self.collision_layer,
            collision_mask: self.collision_mask,
            self_collision_lambda: Default::default(),
        })
    }
}

/// Validates a restored joint spec against the handle-space context and
/// maps engine [`JointError`] into [`SnapshotError::InvalidData`].
pub(crate) fn check_joint_spec(
    spec: &JointKind,
    a: usize,
    b: usize,
    bodies: usize,
    joints: usize,
) -> Result<(), SnapshotError> {
    let invalid = |e: JointError| SnapshotError::InvalidData {
        detail: format!("joint spec rejected: {e}"),
    };
    if a >= bodies || b >= bodies {
        return Err(SnapshotError::InvalidData {
            detail: format!("joint references out-of-range bodies {a} and {b}"),
        });
    }
    if !matches!(spec, JointKind::Gear { .. }) && a == b {
        return Err(SnapshotError::InvalidData {
            detail: "self joint".to_string(),
        });
    }
    if let JointKind::Gear {
        joint_a, joint_b, ..
    } = spec
    {
        for r in [*joint_a, *joint_b] {
            if r.index() >= joints {
                return Err(SnapshotError::InvalidData {
                    detail: format!("gear references unknown joint {}", r.index()),
                });
            }
        }
    }
    crate::migration::validate_joint(spec).map_err(invalid)
}

/// Validates a restored motor override.
pub(crate) fn check_joint_motor(motor: &JointMotor) -> Result<(), SnapshotError> {
    crate::migration::validate_motor(motor).map_err(|e| SnapshotError::InvalidData {
        detail: format!("joint motor rejected: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RON must round-trip every finite `f32` bit-identically (the
    /// continuation promise rests on this), including signed zero and
    /// infinities.
    #[test]
    fn ron_preserves_float_bits() {
        let values: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            1.0 / 3.0,
            1e-7,
            123.456,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE,
            std::f32::consts::PI,
            9.81,
        ];
        for v in values {
            let snap = WorldSnapshot {
                version: WORLD_SNAPSHOT_VERSION,
                gravity: [0.0, -9.81, 0.0],
                tuning: TuningSnapshot {
                    substeps: 12,
                    velocity_iterations: 8,
                    position_iterations: 4,
                    contact_softness: 0.0,
                    step_budget: Some((200_000, 4)),
                    max_ccd_substeps: 32,
                    broadphase: BroadPhaseSnapshot::UniformGrid,
                    wide_solver: true,
                },
                bodies: vec![BodySnapshot {
                    position: [v, v, v],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    velocity: [v, v, v],
                    angular_velocity: [v, v, v],
                    mass: v,
                    inv_mass: v,
                    inertia: [v, v, v],
                    torque: [v, v, v],
                    restitution: v,
                    friction: v,
                    friction_transverse: v,
                    friction_dir: Some([v, v, v]),
                    rolling_friction: v,
                    torsion_friction: v,
                    shape: ShapeSnapshot::Sphere { radius: 0.5 },
                    collision_layer: 1,
                    collision_mask: u32::MAX,
                    is_trigger: false,
                    fracture_impact_speed: f32::INFINITY,
                    body_type: BodyTypeSnapshot::Dynamic,
                    ccd_enabled: false,
                    contact_force_threshold: f32::INFINITY,
                }],
                asleep: vec![false],
                moved: vec![true],
                islands: vec![0],
                island_timers: vec![(0, v)],
                island_grace: vec![],
                prev: vec![PoseSnapshot {
                    position: [v, v, v],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                }],
                event_contacts: vec![],
                event_triggers: vec![],
                pending_contacts: vec![],
                pending_triggers: vec![],
                pending_forces: vec![ContactForceEventSnapshot {
                    a: 0,
                    b: 1,
                    force: v,
                    point: [v, v, v],
                }],
                warm: vec![],
                joints: vec![],
                soft_bodies: vec![],
                soft_touch: vec![],
                soft_friction: None,
                soft_contact_events: vec![],
                fracture_events: vec![],
            };
            let text = snap.to_ron().expect("serializes");
            let back = WorldSnapshot::from_ron(&text).expect("parses");
            assert_eq!(snap, back, "RON must round-trip {v:e} exactly");
            assert_eq!(
                back.bodies[0].position[0].to_bits(),
                v.to_bits(),
                "float bits must survive RON for {v:e}"
            );
        }
    }

    /// The version gate refuses anything but the current version with a
    /// typed error naming both sides.
    #[test]
    fn version_gate_refuses_other_versions() {
        let mut snap = WorldSnapshot {
            version: WORLD_SNAPSHOT_VERSION + 1,
            gravity: [0.0, -9.81, 0.0],
            tuning: TuningSnapshot {
                substeps: 12,
                velocity_iterations: 8,
                position_iterations: 4,
                contact_softness: 0.0,
                step_budget: None,
                max_ccd_substeps: 32,
                broadphase: BroadPhaseSnapshot::UniformGrid,
                wide_solver: true,
            },
            bodies: vec![],
            asleep: vec![],
            moved: vec![],
            islands: vec![],
            island_timers: vec![],
            island_grace: vec![],
            prev: vec![],
            event_contacts: vec![],
            event_triggers: vec![],
            pending_contacts: vec![],
            pending_triggers: vec![],
            pending_forces: vec![],
            warm: vec![],
            joints: vec![],
            soft_bodies: vec![],
            soft_touch: vec![],
            soft_friction: None,
            soft_contact_events: vec![],
            fracture_events: vec![],
        };
        // Serialize with a forged version (the struct serializer writes
        // whatever `version` holds), then parse: the gate must fire.
        let text = snap.to_ron().expect("serializes");
        assert!(matches!(
            WorldSnapshot::from_ron(&text),
            Err(SnapshotError::VersionMismatch { .. })
        ));
        snap.version = WORLD_SNAPSHOT_VERSION;
        assert!(WorldSnapshot::from_ron(&snap.to_ron().expect("serializes")).is_ok());
    }
}
