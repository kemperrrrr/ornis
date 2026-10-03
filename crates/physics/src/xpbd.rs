//! Extended Position-Based Dynamics (XPBD) rigid-body engine.
//!
//! Textbook implementation of Macklin–Müller–Chentanez 2016 ("XPBD:
//! Position-Based Simulation of Compliant Constrained Dynamics", MIG'16)
//! with the rigid-body extensions from Müller et al. ("Detailed Rigid Body
//! Simulation with Extended Position Based Dynamics") and the substepping
//! regime from Macklin et al. 2019 ("Small Steps in Physics Simulation"):
//! the frame step is split into many small substeps with a single
//! constraint iteration each, which converges faster per unit of work than
//! spending the same budget on solver iterations inside one large step.
//!
//! # Method (one substep of size `h`)
//!
//! 1. Integrate: `v += h·g`, `x += h·v` (plus torque/gyroscopic angular
//!    integration); orientations advance through the exact exponential map
//!    and are renormalized.
//! 2. Solve: for every scalar constraint the XPBD Gauss–Seidel update
//!    `Δλ = (−C − α̃·λ) / (w + α̃)` with `α̃ = α/h²` is applied as
//!    `Δx = M⁻¹·∇Cᵀ·Δλ`, where `w = ∇C·M⁻¹·∇Cᵀ` is the generalized inverse
//!    mass (linear + angular terms). Contacts are inequalities (`λ ≥ 0`);
//!    joints are equalities. With `α = 0` this reduces exactly to PBD.
//! 3. Velocities are re-derived from positions (`v = (x − xₙ)/h`, angular
//!    from the quaternion delta); restitution and Coulomb friction run as a
//!    velocity pass clamped by the position-level normal force `λ/h²`.
//!
//! # Scope (deliberate, documented)
//!
//! - Rigid↔rigid contacts are discrete (no CCD): substeps shrink the
//!   tunneling window but fast/thin rigid bodies can still tunnel.
//!   Particles get a once-per-substep conservative-advancement clamp
//!   through [`crate::distance::cast_shape`] (each advance bounded by the
//!   exact current gap): a fast particle stops at the first rigid wall
//!   instead of tunneling through it. Resting contact stays the discrete
//!   solver's job (a sweep reports no hit from a touching start).
//! - Contacts solve on the single `shape_distance` witness pair per body
//!   pair; face-stable multi-point manifolds are a non-goal here. Soft
//!   rows additionally warm-start their normal impulse `λ` from the
//!   previous substep (exact-key persistence on `(soft, particle, body)` —
//!   stronger than witness-proximity matching because particle identity is
//!   stable): a resting stack inherits last step's support instead of
//!   rebuilding it from zero, which is what keeps it still.
//! - Joints supported structurally: ball, distance, rope (one-sided),
//!   spring (compliant row plus the velocity damping pass), fixed,
//!   revolute and prismatic. Limits and motors are NOT driven (accepted
//!   joints constrain the free axes and ignore the drive — a stored servo
//!   override migrates losslessly but never fires here). Wheel, gear and
//!   six-DOF joints are rejected (`add_joint` returns `None`).
//! - No islands, single-threaded: bodies sleep individually (never as one
//!   coherent island), so a jointed assembly has no group freeze — an awake
//!   neighbour re-wakes a sleeper on impact instead. Soft↔rigid begin/end
//!   transitions drain through [`XpbdEngine::drain_soft_contact_events`];
//!   rigid↔rigid and trigger events are not produced here: use
//!   [`crate::engine::SequentialImpulseEngine`] or
//!   [`crate::avbd::AvbdEngine`] when those are needed. Full islands remain
//!   the documented next step.
//! - Soft bodies ([`crate::soft::SoftBody`], PLAN B2/D1): particles +
//!   distance rows + global volume row + self-collision + breakage step in
//!   the same substep loop, with a render-upload surface
//!   ([`SoftBody::surface`] / [`SoftBody::positions_snapshot`]). Particles
//!   couple against rigid bodies as spheres with Coulomb friction (a
//!   velocity pass over the BDF1 velocities clamped by `µ·λ/h`, position
//!   corrections alone cannot hold at Small-Steps sizes) and mutual
//!   layer/mask filtering ([`SoftBody::can_couple_with`]).
//! - Wired into the [`crate::Engine`] orchestrator as
//!   [`crate::SolverKind::Xpbd`]: a world with soft bodies migrates wholly
//!   onto this path (rigid + soft step in the one substep loop, so
//!   soft↔rigid coupling actually runs). Worlds without soft bodies stay on
//!   sequential-impulse/AVBD; soft bodies parked outside the XPBD path keep
//!   their order (and handles) but do not step.

use std::collections::BTreeSet;

use glam::{Quat, Vec3};
use rustc_hash::FxHashMap;

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::broadphase::PrevPose;
use crate::constants::{AXIS_REST_LEN2, COINCIDENT_LEN2, DEGENERATE_LEN2, NEAR_ZERO};
use crate::distance::{ShapeRef, cast_shape, shape_distance};
use crate::engine::{PhysicsEngine, raycast_shape_hit};
use crate::errors::{JointError, QueryError};
use crate::joint::{JointHandle, JointKind, JointMotor, resolve_joint};
use crate::math::{Ray, RaycastHit, tangent_basis};
use crate::migration::{JointReference, JointSnapshot, validate_joint};
use crate::shape::Shape;
use crate::soft::{SoftBody, SoftHandle};
use crate::trigger::{CONTACT_BEGIN_SLOP, ContactEventKind};

/// Structural joint model supported by [`XpbdEngine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XpbdJointKind {
    /// Anchor coincidence (3 positional rows).
    Ball,
    /// Ball rows plus hinge-axis alignment (spin about the axis stays free).
    Revolute,
    /// Perpendicular anchor rows plus axis alignment (slide and spin about
    /// the axis stay free).
    Prismatic,
    /// Ball rows plus full orientation lock.
    Fixed,
    /// Single distance row between the anchors (rod with ball ends).
    Distance,
    /// One-sided distance row (pulls past the maximum, ignores slack).
    Rope,
    /// Compliant distance row about the spec rest length (stiffness via
    /// per-joint compliance, damping via the velocity pass below).
    Spring,
}

/// Persistent joint state: structural kind plus assembly-time frames.
///
/// The original [`JointKind`] spec and its assembly [`JointReference`] ride
/// along untouched by the solve (migration payload for the [`crate::Engine`]
/// orchestrator — the structural rows alone cannot rebuild the spec).
#[derive(Debug, Clone, Copy)]
struct XpbdJoint {
    /// First body handle.
    a: usize,
    /// Second body handle.
    b: usize,
    /// Structural model.
    kind: XpbdJointKind,
    /// Original joint spec (migration payload: rebuilds re-resolve it).
    spec: JointKind,
    /// Assembly references (migration payload: preserved verbatim for
    /// solvers that understand them, e.g. a roundtrip back to
    /// sequential-impulse).
    reference: JointReference,
    /// Anchor in body A's local frame.
    la: Vec3,
    /// Anchor in body B's local frame.
    lb: Vec3,
    /// Hinge/slide axis in A's local frame.
    ax_a: Vec3,
    /// Hinge/slide axis in B's local frame.
    ax_b: Vec3,
    /// Rest length of a distance rod, captured at creation. For rope
    /// joints the spec maximum, for spring joints the spec rest length
    /// (both spec-immutable — restores never overwrite them, see
    /// [`XpbdEngine::restore_joint_reference`]).
    rest_length: f32,
    /// Relative rotation `qa⁻¹·qb` at creation (fixed joints).
    q_ref: Quat,
    /// Generalized motor override (`set_joint_motor`, revolute/prismatic
    /// only): migration payload, ignored by the solve — XPBD drives no
    /// motors (see the module scope docs). Spring joints carry their motor
    /// inline in the spec and never use this slot.
    servo: Option<JointMotor>,
}

/// One discrete contact for a single substep: fixed normal and body-local
/// anchors (captured at discovery), live Lagrange multiplier.
#[derive(Debug, Clone)]
struct Contact {
    /// First body handle.
    a: usize,
    /// Second body handle.
    b: usize,
    /// Contact normal from A to B.
    n: Vec3,
    /// Contact anchor in A's local frame.
    la: Vec3,
    /// Contact anchor in B's local frame.
    lb: Vec3,
    /// Accumulated normal multiplier (force estimate: `λ/h²`).
    lambda: f32,
}

/// Particle↔rigid contact for a single substep: the particle is side A (a
/// bare position, no orientation), the rigid body side B with a body-local
/// anchor. The normal row is an inequality (`λ ≥ 0`, warm-started from the
/// previous substep); Coulomb friction runs as a velocity pass clamped by
/// the position-level normal impulse (`µ·λ/h`), mirroring the rigid
/// [`XpbdEngine::solve_velocities`]. Both sides derive post-solve
/// velocities via BDF1 first, then the friction pass trims them.
#[derive(Debug, Clone)]
struct SoftContact {
    /// Soft body handle.
    soft: usize,
    /// Particle index inside the soft body.
    particle: usize,
    /// Rigid body handle.
    body: usize,
    /// Contact normal from the particle toward the rigid body.
    n: Vec3,
    /// Surface witness offset from the particle center at discovery
    /// (particles don't rotate, so the world offset stays fixed): the gap
    /// is measured from the sphere surface, not its center.
    off: Vec3,
    /// Contact anchor in the rigid body's local frame.
    lb: Vec3,
    /// Accumulated normal multiplier (`λ ≥ 0`).
    lambda: f32,
}

/// Linear speed below which a body counts as quiet for sleep (m/s,
/// sequential-impulse parity).
const SLEEP_LIN: f32 = 0.15;
/// Midpoint / half-extent scale.
const HALF: f32 = 0.5;
/// Angular speed below which a rigid body counts as quiet (rad/s,
/// sequential-impulse parity).
const SLEEP_ANG: f32 = 0.15;
/// Continuous quiet time before a body freezes (s).
const SLEEP_TIME: f32 = HALF;
/// Relative normal speed at a contact that wakes a sleeper (m/s,
/// sequential-impulse `WAKE_IMPACT_SPEED` parity): a fast impact disturbs
/// the sleeper, slow settling does not.
const WAKE_IMPACT_SPEED: f32 = HALF;
/// CCD backoff (m): a clamped particle stops this far short of the swept
/// hit so the discrete pass sees a clean near-touch, not a zero-gap
/// flicker (same order as the sequential-impulse TOI backoff).
const CCD_BACKOFF: f32 = 1e-3;

/// Per-soft-body sleep state, parallel to [`XpbdEngine`]'s soft registry.
#[derive(Debug, Clone, Copy)]
struct SoftSleep {
    /// Whether the body is frozen (skips integration and solves).
    asleep: bool,
    /// Accumulated continuous quiet time (s).
    quiet: f32,
}

/// A solid-contact transition between a soft-body particle cloud and a
/// rigid body, drained after the step like [`crate::trigger::TriggerEvent`].
/// Pairs are reported per `(soft body, rigid body)` — individual particle
/// touches aggregate to their body pair — in deterministic sorted order
/// (begins before ends).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftContactEvent {
    /// Soft body handle (this engine's soft registry).
    pub soft: SoftHandle,
    /// Rigid body handle (this engine's rigid registry).
    pub body: BodyHandle,
    /// What happened: begin or end. [`ContactEventKind::Hit`] is never
    /// emitted here (no impact-speed tracking in D1) — match only on
    /// begin/end.
    pub kind: ContactEventKind,
}

/// XPBD rigid-body engine; see the module docs for formulation and scope.
///
/// Bodies live in dense handle order (`swap_remove` on removal, like the
/// other engines). Constraints are rebuilt from scratch every substep, so
/// there is no cross-step warm-start state to migrate or invalidate — with
/// one exception: soft normal impulses persist in a keyed cache across
/// substeps (see the module scope docs).
/// Soft bodies ([`crate::soft::SoftBody`]) live in a second dense registry
/// and step in the same substep loop: particles + distance rows + global
/// volume row + self-collision + breakage, coupled against rigid bodies
/// with Coulomb friction and layer/mask filtering.
#[derive(Debug)]
pub struct XpbdEngine {
    /// Constant world-space acceleration applied to dynamic bodies.
    gravity: Vec3,
    /// Bodies in handle order.
    bodies: Vec<RigidBody>,
    /// Live joints in handle order.
    joints: Vec<XpbdJoint>,
    /// Canonical body pairs joined by a non-fixed joint: their contacts are
    /// skipped (`collide_connected = false` parity; fixed/weld assemblies
    /// keep structural contacts).
    joint_pairs: BTreeSet<(usize, usize)>,
    /// Substeps per `step` call (Small-Steps regime).
    substeps: u32,
    /// Constraint iterations per substep (1 is the Small-Steps optimum).
    iterations: u32,
    /// Contact compliance `α` (inverse stiffness, 0 = rigid).
    contact_compliance: f32,
    /// Joint compliance `α` (inverse stiffness, 0 = rigid).
    joint_compliance: f32,
    /// Approach speed below which contacts are inelastic (m/s).
    restitution_threshold: f32,
    /// Soft bodies in handle order (PLAN B2/D1).
    soft_bodies: Vec<SoftBody>,
    /// Coulomb coefficient on the soft side: each soft↔rigid pair combines
    /// `sqrt(soft_friction · body.friction)` (geometric-mean parity with
    /// the rigid↔rigid velocity pass).
    soft_friction: f32,
    /// Per-body rigid sleep flags, parallel to `bodies` (dynamics only;
    /// statics/kinematics are never asleep).
    rigid_asleep: Vec<bool>,
    /// Accumulated continuous quiet time per rigid body (s), parallel to
    /// `bodies`.
    rigid_quiet: Vec<f32>,
    /// Per-soft-body sleep state, parallel to `soft_bodies`.
    soft_sleep: Vec<SoftSleep>,
    /// Previous substep's soft normal impulses, keyed by
    /// `(soft, particle, body)`: the cross-substep warm-start cache.
    /// Seeded into fresh contacts at discovery, written back after the
    /// iterations, pruned to the live touch set at step end, and cleared
    /// by body removals (topology edits drop warm state).
    soft_lambda_cache: FxHashMap<(usize, usize, usize), f32>,
    /// Soft↔rigid `(soft, particle, body)` triples touching at the previous
    /// completed step (overlap, plus a [`CONTACT_BEGIN_SLOP`] grace band so
    /// resting jitter emits no flicker). Cleared by body removals, like the
    /// sequential-impulse touch sets.
    soft_touch: BTreeSet<(usize, usize, usize)>,
    /// Queued soft↔rigid begin/end transitions, drained by
    /// [`XpbdEngine::drain_soft_contact_events`].
    soft_contact_events: Vec<SoftContactEvent>,
}

impl XpbdEngine {
    /// Empty engine with Small-Steps defaults: 20 substeps × 1 iteration,
    /// rigid contacts and joints, 1 m/s restitution threshold, 0.5 soft
    /// friction, everything awake and no cached contacts.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            gravity,
            bodies: Vec::new(),
            joints: Vec::new(),
            joint_pairs: BTreeSet::new(),
            substeps: 20,
            iterations: 1,
            contact_compliance: 0.0,
            joint_compliance: 0.0,
            restitution_threshold: 1.0,
            soft_bodies: Vec::new(),
            soft_friction: HALF,
            rigid_asleep: Vec::new(),
            rigid_quiet: Vec::new(),
            soft_sleep: Vec::new(),
            soft_lambda_cache: FxHashMap::default(),
            soft_touch: BTreeSet::new(),
            soft_contact_events: Vec::new(),
        }
    }

    /// Substeps per [`PhysicsEngine::step`]: the frame step is divided into
    /// this many XPBD substeps. More substeps = stiffer stacks and smaller
    /// tunneling windows at linear cost. Ignored when zero or absurd
    /// (clamped to 1..=1000).
    pub fn set_substeps(&mut self, n: u32) {
        self.substeps = n.clamp(1, 1000);
    }

    /// Constraint iterations per substep. The Small-Steps result says a
    /// fixed budget is best spent on substeps, so the default is 1;
    /// raising this stiffens individual substeps instead. Clamped to 1..=100.
    pub fn set_iterations(&mut self, n: u32) {
        self.iterations = n.clamp(1, 100);
    }

    /// Contact compliance `α` (inverse stiffness in m/N). Zero (default) is
    /// a rigid contact; positive values let contacts sink proportionally to
    /// the contact force. Must be finite; negative values are clamped to 0.
    pub fn set_contact_compliance(&mut self, alpha: f32) {
        self.contact_compliance = if alpha.is_finite() {
            alpha.max(0.0)
        } else {
            0.0
        };
    }

    /// Joint compliance `α` (inverse stiffness). Zero (default) is a rigid
    /// joint; positive values let joints stretch under load. Same guards as
    /// [`XpbdEngine::set_contact_compliance`].
    pub fn set_joint_compliance(&mut self, alpha: f32) {
        self.joint_compliance = if alpha.is_finite() {
            alpha.max(0.0)
        } else {
            0.0
        };
    }

    /// Approach speed (m/s) below which contacts are perfectly inelastic.
    /// Faster impacts bounce with the bodies' mean restitution. Must be
    /// non-negative and finite; invalid values are ignored.
    pub fn set_restitution_threshold(&mut self, threshold: f32) {
        if threshold.is_finite() && threshold >= 0.0 {
            self.restitution_threshold = threshold;
        }
    }

    /// Coulomb coefficient on the soft side (≥ 0, default 0.5): each
    /// soft↔rigid pair uses `sqrt(soft_friction · body.friction)`, so 0
    /// disables soft friction and 1 keeps the rigid body's value. Must be
    /// finite; negative or non-finite values are clamped to 0.
    pub fn set_soft_friction(&mut self, mu: f32) {
        self.soft_friction = if mu.is_finite() { mu.max(0.0) } else { 0.0 };
    }

    /// Current soft-side Coulomb coefficient (see
    /// [`XpbdEngine::set_soft_friction`]).
    pub fn soft_friction(&self) -> f32 {
        self.soft_friction
    }

    /// Whether the rigid body is currently sleeping (frozen: zeroed inverse
    /// mass/inertia, skipped integration). Static and kinematic bodies are
    /// never asleep; invalid handles report `false`.
    pub fn is_asleep(&self, handle: BodyHandle) -> bool {
        self.rigid_asleep
            .get(handle.index())
            .copied()
            .unwrap_or(false)
    }

    /// Whether the soft body is currently sleeping (frozen: skips
    /// integration and solves, velocities zeroed). Invalid handles report
    /// `false`.
    pub fn is_soft_asleep(&self, handle: SoftHandle) -> bool {
        self.soft_sleep
            .get(handle.index())
            .is_some_and(|s| s.asleep)
    }

    /// Wake a sleeping soft body without moving it: velocities stay zeroed,
    /// but the BDF1 baseline (`prev_position`) is rebased onto the live
    /// positions, so waking after a teleport through
    /// [`XpbdEngine::get_soft_body_mut`] emits no velocity spike. No-op for
    /// invalid handles. Contact impacts wake bodies on their own (no
    /// baseline change there — motion continues); call this after direct
    /// pose edits, like the orchestrator does for rigid bodies.
    pub fn wake_soft_body(&mut self, handle: SoftHandle) {
        let Some(soft) = self.soft_bodies.get_mut(handle.index()) else {
            return;
        };
        for p in &mut soft.particles {
            p.prev_position = p.position;
        }
        if let Some(st) = self.soft_sleep.get_mut(handle.index()) {
            st.asleep = false;
            st.quiet = 0.0;
        }
    }

    /// Drain soft↔rigid begin/end transitions produced by completed steps
    /// (per `(soft body, rigid body)` pair, begins before ends, sorted).
    /// Queued until drained; body removals clear the queue.
    pub fn drain_soft_contact_events(&mut self) -> Vec<SoftContactEvent> {
        std::mem::take(&mut self.soft_contact_events)
    }

    /// Number of registered bodies (dense handles).
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Number of live joints (dense handles).
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// Register a soft body and return its handle (PLAN B2/D1).
    pub fn add_soft_body(&mut self, body: SoftBody) -> SoftHandle {
        self.soft_bodies.push(body);
        self.soft_sleep.push(SoftSleep {
            asleep: false,
            quiet: 0.0,
        });
        SoftHandle::from(self.soft_bodies.len() - 1)
    }

    /// Remove a soft body, swapping the last into its slot. Invalid
    /// handles are a no-op. No joints reference particles, so no remap is
    /// needed beyond the swap; the warm-start cache, touch set and queued
    /// events are cleared (indices shift, like the sequential-impulse touch
    /// sets on removal).
    pub fn remove_soft_body(&mut self, handle: SoftHandle) {
        if handle.index() < self.soft_bodies.len() {
            self.soft_bodies.swap_remove(handle.index());
            self.soft_sleep.swap_remove(handle.index());
            self.soft_lambda_cache.clear();
            self.soft_touch.clear();
            self.soft_contact_events.clear();
        }
    }

    /// Number of registered soft bodies (dense handles).
    pub fn soft_body_count(&self) -> usize {
        self.soft_bodies.len()
    }

    /// Read-only access to a soft body, or `None` for an invalid handle.
    pub fn get_soft_body(&self, handle: SoftHandle) -> Option<&SoftBody> {
        self.soft_bodies.get(handle.index())
    }

    /// Mutable access to a soft body, or `None` for an invalid handle.
    /// Direct particle edits take effect at the next [`PhysicsEngine::step`].
    pub fn get_soft_body_mut(&mut self, handle: SoftHandle) -> Option<&mut SoftBody> {
        self.soft_bodies.get_mut(handle.index())
    }

    /// Bodies in handle order, cloned for solver migration ([`crate::Engine`]
    /// re-registers them 1:1, so handles stay valid across the switch).
    /// Sleepers migrate awake with their mass model restored (warm-start
    /// state never migrates, and a zeroed inverse mass must not leak into
    /// the new engine — AVBD parity).
    pub(crate) fn bodies_snapshot(&self) -> Vec<RigidBody> {
        self.bodies
            .iter()
            .map(|b| {
                let mut c = b.clone();
                c.restore_sleep_triple();
                c
            })
            .collect()
    }

    /// Driver baselines for migration: XPBD derives velocities from substep
    /// positions and keeps no cross-step driver history, so the live poses
    /// seed the target without manufacturing motion.
    pub(crate) fn body_baselines(&self) -> Vec<PrevPose> {
        self.bodies
            .iter()
            .map(|b| PrevPose {
                pos: b.position,
                rot: b.orientation,
            })
            .collect()
    }

    /// Baseline restore after rebuilding: no-op (see
    /// [`XpbdEngine::body_baselines`] — there is no driver history to seed).
    pub(crate) fn restore_body_baseline(&mut self, _h: BodyHandle, _pose: PrevPose) {}

    /// Completed-step rigid event baseline: XPBD tracks no rigid↔rigid
    /// contact/trigger pairs (those stay a sequential-impulse/AVBD job —
    /// see the module scope docs), so migration seeds nothing.
    pub(crate) fn event_state(&self) -> crate::migration::EventState {
        crate::migration::EventState::default()
    }

    /// Event-baseline restore after rebuilding: no-op (see
    /// [`XpbdEngine::event_state`]).
    pub(crate) fn restore_event_state(&mut self, _state: crate::migration::EventState) {}

    /// Physical joint state in handle order: the stored spec plus the stored
    /// assembly reference (numerical state never migrates).
    pub(crate) fn joint_snapshots(&self) -> Vec<JointSnapshot> {
        self.joints
            .iter()
            .map(|j| JointSnapshot {
                a: BodyHandle::from(j.a),
                b: BodyHandle::from(j.b),
                spec: j.spec,
                reference: j.reference,
                servo: j.servo,
            })
            .collect()
    }

    /// Restore the assembly reference after a solver migration: the stored
    /// payload is replaced verbatim (so a later migration back out stays
    /// lossless) and the structural rest state (distance-rod length,
    /// fixed-joint relative rotation) follows it.
    pub(crate) fn restore_joint_reference(&mut self, h: JointHandle, r: JointReference) {
        let Some(j) = self.joints.get_mut(h.index()) else {
            return;
        };
        j.reference = r;
        // Distance rest length follows the assembly reference; rope
        // maximum and spring rest ride in the spec (immutable after
        // creation) and must survive the restore untouched.
        if j.kind == XpbdJointKind::Distance {
            j.rest_length = r.distance.0;
        }
        j.q_ref = r.rotation;
    }

    /// Generalized motor override on an assembled joint (see
    /// [`crate::joint::JointMotor`]): stored as a migration payload and
    /// reported by [`XpbdEngine::joint_motor`], but never driven — XPBD
    /// solves no motors (see the module scope docs). Only revolute and
    /// prismatic joints take one (there are no wheel joints here); spring
    /// joints carry theirs inline in the spec.
    ///
    /// # Errors
    ///
    /// [`JointError::UnknownRef`] for a stale handle, `Unsupported` for a
    /// joint kind without a driven axis, `NonFinite` for an invalid motor.
    pub fn set_joint_motor(
        &mut self,
        handle: JointHandle,
        motor: Option<JointMotor>,
    ) -> Result<(), JointError> {
        if let Some(m) = motor {
            crate::migration::validate_motor(&m)?;
        }
        let Some(j) = self.joints.get(handle.index()) else {
            return Err(JointError::UnknownRef {
                handle: handle.index(),
            });
        };
        if !matches!(j.kind, XpbdJointKind::Revolute | XpbdJointKind::Prismatic) {
            return Err(JointError::Unsupported {
                detail: "set_joint_motor needs a revolute or prismatic joint".to_string(),
            });
        }
        self.joints[handle.index()].servo = motor;
        let (a, b) = (self.joints[handle.index()].a, self.joints[handle.index()].b);
        self.wake_rigid(a);
        self.wake_rigid(b);
        Ok(())
    }

    /// Current motor override of a joint (`None` = spec motor applies, if
    /// any — still undriven here). `None` for an invalid handle.
    pub fn joint_motor(&self, handle: JointHandle) -> Option<JointMotor> {
        self.joints.get(handle.index())?.servo
    }

    /// Restore the motor override after a solver migration (verbatim).
    pub(crate) fn restore_joint_motor(&mut self, h: JointHandle, motor: Option<JointMotor>) {
        if let Some(j) = self.joints.get_mut(h.index()) {
            j.servo = motor;
        }
    }

    /// Soft bodies in handle order, cloned for solver migration (the
    /// orchestrator parks them outside the XPBD path and re-registers them
    /// 1:1 on return, so handles stay valid across the switch).
    pub(crate) fn soft_bodies_snapshot(&self) -> Vec<SoftBody> {
        self.soft_bodies.clone()
    }
    /// Live soft↔rigid touch triples for migration: indices survive the
    /// ordered rebuild (rigid and soft registries both restore 1:1), so the
    /// set transfers verbatim and the target emits no manufactured begins.
    pub(crate) fn soft_touch_state(&self) -> BTreeSet<(usize, usize, usize)> {
        self.soft_touch.clone()
    }

    /// Restore the migrated touch set (see [`XpbdEngine::soft_touch_state`]).
    /// Triples outside the rebuilt registries are dropped, never indexed.
    pub(crate) fn restore_soft_touch_state(&mut self, state: BTreeSet<(usize, usize, usize)>) {
        self.soft_touch = state
            .into_iter()
            .filter(|&(s, p, b)| {
                s < self.soft_bodies.len()
                    && b < self.bodies.len()
                    && p < self.soft_bodies[s].particles.len()
            })
            .collect();
    }

    /// Hand the whole soft registry to the orchestrator's park (Islands
    /// routing): bodies in handle order plus the live touch set. Sleep,
    /// warm-start and queued events are dropped — parking is a migration,
    /// and sleep/warm state never migrates.
    pub(crate) fn drain_soft_registry(
        &mut self,
    ) -> (Vec<SoftBody>, BTreeSet<(usize, usize, usize)>) {
        self.soft_sleep.clear();
        self.soft_lambda_cache.clear();
        self.soft_contact_events.clear();
        (
            std::mem::take(&mut self.soft_bodies),
            std::mem::take(&mut self.soft_touch),
        )
    }

    /// Whether the body is simulated by this engine (dynamic with mass).
    fn solvable(&self, h: usize) -> bool {
        self.bodies[h].body_type == BodyType::Dynamic && self.bodies[h].inv_mass > 0.0
    }

    /// One substep of size `h`: integrate → CCD clamp → discover → solve →
    /// velocities. Sleeping bodies skip integration (rigid sleepers via
    /// zeroed inverse mass, soft sleepers via an explicit flag) but still
    /// anchor contacts: corrections apply to the awake side only.
    fn substep(&mut self, h: f32) {
        let n = self.bodies.len();
        let mut prev_pos = vec![Vec3::ZERO; n];
        let mut prev_rot = vec![Quat::IDENTITY; n];
        for (i, b) in self.bodies.iter_mut().enumerate() {
            prev_pos[i] = b.position;
            prev_rot[i] = b.orientation;
            if b.body_type != BodyType::Dynamic || b.inv_mass <= 0.0 {
                continue;
            }
            b.velocity += h * self.gravity;
            // Newton–Euler angular integration with the gyroscopic term.
            let iw = world_inertia(b.inertia, b.orientation, b.angular_velocity);
            let ang_acc = apply_inv_inertia(
                b.inertia,
                b.orientation,
                b.torque - b.angular_velocity.cross(iw),
            );
            b.angular_velocity += h * ang_acc;
            b.torque = Vec3::ZERO;
            b.position += h * b.velocity;
            b.orientation = integrate_orientation(b.orientation, b.angular_velocity, h);
        }

        let mut contacts = self.discover_contacts();
        // Time-scaled compliance: α̃ = α/h² (Macklin et al. 2016, §4).
        let alpha_c = self.contact_compliance / (h * h);
        let alpha_j = self.joint_compliance / (h * h);
        for (s, body) in self.soft_bodies.iter_mut().enumerate() {
            if self.soft_sleep[s].asleep {
                continue;
            }
            body.integrate(h, self.gravity);
            body.begin_substep();
        }
        // Particle CCD, once per substep: clamp the integrated prediction
        // before discovery so no witness can start across a wall.
        for (s, body) in self.soft_bodies.iter_mut().enumerate() {
            if self.soft_sleep[s].asleep {
                continue;
            }
            Self::clamp_soft_ccd(&self.bodies, body);
        }
        let mut soft_contacts = self.discover_soft_contacts();
        self.wake_on_impact(&contacts, &soft_contacts);
        for _ in 0..self.iterations {
            for i in 0..contacts.len() {
                self.solve_contact(i, &mut contacts, alpha_c);
            }
            for j in 0..self.joints.len() {
                self.solve_joint(j, alpha_j, h);
            }
            for (s, body) in self.soft_bodies.iter_mut().enumerate() {
                if self.soft_sleep[s].asleep {
                    continue;
                }
                body.solve_constraints(h);
                body.solve_volume(h);
                crate::soft_self::solve_self_collision(body, h);
            }
            for i in 0..soft_contacts.len() {
                self.solve_soft_contact(i, &mut soft_contacts, alpha_c);
            }
            // Write the normal impulses back to the warm-start cache while
            // the witnesses are still live.
            for c in &soft_contacts {
                self.soft_lambda_cache
                    .insert((c.soft, c.particle, c.body), c.lambda);
            }
        }
        // Topology surgery outside the constraint iterations: broken rows
        // must not shift indices mid-sweep.
        for (s, body) in self.soft_bodies.iter_mut().enumerate() {
            if self.soft_sleep[s].asleep {
                continue;
            }
            body.apply_breakage();
        }
        for b in &mut self.bodies {
            b.orientation = b.orientation.normalize();
        }

        // BDF1-style velocity update from the solved positions.
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if b.body_type != BodyType::Dynamic || b.inv_mass <= 0.0 {
                continue;
            }
            b.velocity = (b.position - prev_pos[i]) / h;
            b.angular_velocity = angular_velocity_from_delta(b.orientation, prev_rot[i], h);
        }
        for body in &mut self.soft_bodies {
            body.update_velocities(h);
        }
        self.solve_velocities(&contacts, h);
        self.solve_soft_velocities(&soft_contacts, h);
        self.solve_spring_damping(h);
    }

    /// Discrete contact discovery at the current poses: AABB prefilter plus
    /// collision filters, trigger skip and the no-collide joint set, then an
    /// exact `shape_distance` query per surviving pair.
    fn discover_contacts(&self) -> Vec<Contact> {
        let n = self.bodies.len();
        let aabbs: Vec<crate::math::AABB> = self
            .bodies
            .iter()
            .map(|b| b.shape.aabb(b.position, b.orientation))
            .collect();
        let mut out = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                let (ba, bb) = (&self.bodies[a], &self.bodies[b]);
                if !ba.can_collide_with(bb) {
                    continue;
                }
                if ba.is_trigger || bb.is_trigger {
                    continue;
                }
                // Pairs with no dynamics on either side cannot move.
                if !self.solvable(a) && !self.solvable(b) {
                    continue;
                }
                // Two sleepers are frozen relative to each other: no work,
                // and no event flicker downstream.
                if self.rigid_asleep[a] && self.rigid_asleep[b] {
                    continue;
                }
                if self.joint_pairs.contains(&(a, b)) {
                    continue;
                }
                if !aabbs[a].overlaps(&aabbs[b]) {
                    continue;
                }
                let da = ShapeRef {
                    shape: &ba.shape,
                    pos: ba.position,
                    rot: ba.orientation,
                };
                let db = ShapeRef {
                    shape: &bb.shape,
                    pos: bb.position,
                    rot: bb.orientation,
                };
                let d = shape_distance(da, db);
                if !d.dist.is_finite() || d.dist > 0.0 {
                    continue;
                }
                let mut normal = d.point_b - d.point_a;
                if normal.length_squared() < DEGENERATE_LEN2 {
                    normal = bb.position - ba.position;
                }
                let normal = normal.normalize_or(Vec3::Y);
                out.push(Contact {
                    a,
                    b,
                    n: normal,
                    la: ba.orientation.inverse() * (d.point_a - ba.position),
                    lb: bb.orientation.inverse() * (d.point_b - bb.position),
                    lambda: 0.0,
                });
            }
        }
        out
    }

    /// Particle↔rigid discovery: every particle of a body with a positive
    /// `contact_radius` is queried as a sphere against every non-trigger
    /// rigid body passing the mutual layer/mask filter
    /// ([`SoftBody::can_couple_with`]). Pairs where the soft body sleeps
    /// and the rigid side cannot move (static or asleep) are skipped — the
    /// frozen-pair rule, mirrored from the rigid path. Fresh contacts seed
    /// their normal impulse from the cross-substep warm-start cache.
    fn discover_soft_contacts(&self) -> Vec<SoftContact> {
        let mut out = Vec::new();
        for (s, soft) in self.soft_bodies.iter().enumerate() {
            if soft.contact_radius <= 0.0 {
                continue;
            }
            let sphere = Shape::Sphere {
                radius: soft.contact_radius,
            };
            for (p, particle) in soft.particles.iter().enumerate() {
                for (b, body) in self.bodies.iter().enumerate() {
                    if body.is_trigger {
                        continue;
                    }
                    if !soft.can_couple_with(body) {
                        continue;
                    }
                    if self.soft_sleep[s].asleep
                        && (body.body_type != BodyType::Dynamic || self.rigid_asleep[b])
                    {
                        continue;
                    }
                    let d = shape_distance(
                        ShapeRef {
                            shape: &sphere,
                            pos: particle.position,
                            rot: Quat::IDENTITY,
                        },
                        ShapeRef {
                            shape: &body.shape,
                            pos: body.position,
                            rot: body.orientation,
                        },
                    );
                    if !d.dist.is_finite() || d.dist > 0.0 {
                        continue;
                    }
                    let mut normal = d.point_b - d.point_a;
                    if normal.length_squared() < DEGENERATE_LEN2 {
                        normal = body.position - particle.position;
                    }
                    out.push(SoftContact {
                        soft: s,
                        particle: p,
                        body: b,
                        n: normal.normalize_or(Vec3::Y),
                        off: d.point_a - particle.position,
                        lb: body.orientation.inverse() * (d.point_b - body.position),
                        lambda: self
                            .soft_lambda_cache
                            .get(&(s, p, b))
                            .copied()
                            .unwrap_or(0.0),
                    });
                }
            }
        }
        out
    }

    /// Position-level solve for one particle↔rigid contact (inequality,
    /// `λ ≥ 0`): the particle carries side A (`+n`), the rigid body side B
    /// (`−n` with its rotation lever). Contact compliance is shared with
    /// the rigid path. Friction is NOT positional here — at Small-Steps
    /// sizes a positional correction is O(h²) against O(h) of sliding and
    /// cannot hold; it runs instead as a velocity pass over the BDF1
    /// velocities (see [`XpbdEngine::solve_soft_velocities`]).
    fn solve_soft_contact(&mut self, i: usize, contacts: &mut [SoftContact], alpha_tilde: f32) {
        let soft_awake = self
            .soft_sleep
            .get(contacts[i].soft)
            .is_none_or(|s| !s.asleep);
        let c = &mut contacts[i];
        let Some(soft) = self.soft_bodies.get_mut(c.soft) else {
            return;
        };
        let Some(particle) = soft.particles.get_mut(c.particle) else {
            return;
        };
        let Some(body) = self.bodies.get_mut(c.body) else {
            return;
        };
        let pb = body.position + body.orientation * c.lb;
        // Signed gap from the particle's surface witness (not its center —
        // at a clean touch the witness gap is 0 while the center is a full
        // radius away along −n).
        let gap = (particle.position + c.off - pb).dot(c.n);
        if gap >= 0.0 && c.lambda == 0.0 {
            return;
        }
        let rb = pb - body.position;
        // Rigid gradient enters with −n, but the mass term is quadratic.
        let t = rb.cross(c.n);
        let w = particle.inv_mass
            + body.inv_mass
            + t.dot(apply_inv_inertia(body.inertia, body.orientation, t));
        if w <= 0.0 {
            return;
        }
        let dlambda = delta_lambda(gap, c.lambda, w, alpha_tilde);
        let next = (c.lambda + dlambda).max(0.0);
        let applied = next - c.lambda;
        c.lambda = next;
        if applied != 0.0 {
            if soft_awake {
                particle.position += c.n * (applied * particle.inv_mass);
            }
            apply_position_correction(body, -1.0, c.n, rb, applied);
        }
    }

    /// Velocity-level Coulomb friction over the substep's soft contacts,
    /// mirroring the rigid [`XpbdEngine::solve_velocities`]: runs after the
    /// BDF1 update, kills the tangential slip velocity up to `µ·λn/h` with
    /// `µ = sqrt(soft_friction · body.friction)`. Sleeping sides stay
    /// frozen (rigid sleepers no-op through zero inverse mass, soft
    /// sleepers are skipped explicitly).
    fn solve_soft_velocities(&mut self, contacts: &[SoftContact], h: f32) {
        for c in contacts {
            let normal_impulse = c.lambda / h;
            if normal_impulse <= 0.0 {
                continue;
            }
            if self.soft_sleep.get(c.soft).is_some_and(|s| s.asleep) {
                continue;
            }
            let (Some(soft), Some(body)) = (self.soft_bodies.get(c.soft), self.bodies.get(c.body))
            else {
                continue;
            };
            let mu = (self.soft_friction * body.friction).sqrt();
            if mu <= 0.0 {
                continue;
            }
            let Some(particle) = soft.particles.get(c.particle) else {
                continue;
            };
            let rb = body.orientation * c.lb;
            let vrel = particle.velocity - point_velocity(body, rb);
            // Tangential slip in the contact frame (tangent_basis parity
            // with the joint/prismatic rows: deterministic frame, slip
            // projected onto it).
            let (t1, t2) = tangent_basis(c.n);
            let vt = t1 * vrel.dot(t1) + t2 * vrel.dot(t2);
            let speed = vt.length();
            if speed < NEAR_ZERO {
                continue;
            }
            let t = vt / speed;
            let rt = rb.cross(t);
            let w = particle.inv_mass
                + body.inv_mass
                + rt.dot(apply_inv_inertia(body.inertia, body.orientation, rt));
            if w <= 0.0 {
                continue;
            }
            let max_friction = mu * normal_impulse;
            let jt = (-speed / w).clamp(-max_friction, max_friction);
            if jt != 0.0 {
                if let Some(particle) = self.soft_bodies[c.soft].particles.get_mut(c.particle) {
                    particle.velocity += t * (jt * particle.inv_mass);
                }
                if let Some(body) = self.bodies.get_mut(c.body) {
                    apply_velocity_impulse(body, -1.0, t, rb, jt);
                }
            }
        }
    }

    /// Position-level normal solve for one contact (inequality, `λ ≥ 0`).
    fn solve_contact(&mut self, i: usize, contacts: &mut [Contact], alpha_tilde: f32) {
        let c = &mut contacts[i];
        let (ba, bb) = pair_mut(&mut self.bodies, c.a, c.b);
        let pa = ba.position + ba.orientation * c.la;
        let pb = bb.position + bb.orientation * c.lb;
        // Signed gap: negative while penetrating, the XPBD `C(x) ≥ 0` form.
        let gap = (pa - pb).dot(c.n);
        if gap >= 0.0 && c.lambda == 0.0 {
            return;
        }
        let ra = pa - ba.position;
        let rb = pb - bb.position;
        let w = generalized_inverse_mass(ba, bb, c.n, ra, rb);
        if w <= 0.0 {
            return;
        }
        let dlambda = delta_lambda(gap, c.lambda, w, alpha_tilde);
        let next = (c.lambda + dlambda).max(0.0);
        let applied = next - c.lambda;
        c.lambda = next;
        if applied != 0.0 {
            apply_position_correction(ba, 1.0, c.n, ra, applied);
            apply_position_correction(bb, -1.0, c.n, rb, applied);
        }
    }

    /// Position-level joint solve (equalities, one `λ` per scalar row;
    /// rope is the one-sided twin, spring the compliant twin of the
    /// distance row). A fully sleeping pair is frozen: no correction can
    /// move either side.
    fn solve_joint(&mut self, j: usize, alpha_tilde: f32, h: f32) {
        let joint = self.joints[j];
        if self.rigid_asleep[joint.a] && self.rigid_asleep[joint.b] {
            return;
        }
        match joint.kind {
            XpbdJointKind::Ball => {
                for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                    self.solve_position_row(
                        joint.a,
                        joint.b,
                        joint.la,
                        joint.lb,
                        axis,
                        0.0,
                        0.0,
                        alpha_tilde,
                    );
                }
            }
            XpbdJointKind::Distance => {
                // C = |pa − pb| − rest; λ state would need persistence
                // across iterations for exact compliant behavior — with the
                // default single iteration per substep λ starts at 0, which
                // is exactly the Small-Steps regime this engine implements.
                self.solve_anchor_distance(&joint, joint.rest_length, alpha_tilde);
            }
            XpbdJointKind::Rope => {
                // One-sided distance: only a stretched rope corrects —
                // slack never pushes (the position twin of the SI
                // one-sided velocity row).
                let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
                let pa = ba.position + ba.orientation * joint.la;
                let pb = bb.position + bb.orientation * joint.lb;
                let dist = (pa - pb).length();
                if dist <= joint.rest_length {
                    return;
                }
                self.solve_anchor_distance(&joint, joint.rest_length, alpha_tilde);
            }
            XpbdJointKind::Spring => {
                // Compliant distance about the rest length: per-joint
                // compliance from the stiffness (force-based `1/k`,
                // acceleration-based scaled by the live reduced mass),
                // damping runs as the velocity pass below.
                let Some(motor) = joint.spec.spring_motor() else {
                    return;
                };
                let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
                let pa = ba.position + ba.orientation * joint.la;
                let pb = bb.position + bb.orientation * joint.lb;
                let delta = pa - pb;
                let dist = delta.length();
                if dist < NEAR_ZERO || h <= 0.0 {
                    return;
                }
                let w = generalized_inverse_mass(
                    ba,
                    bb,
                    delta / dist,
                    pa - ba.position,
                    pb - bb.position,
                );
                if w <= 0.0 || !motor.stiffness.is_finite() || motor.stiffness <= 0.0 {
                    return;
                }
                let alpha = match motor.model {
                    crate::joint::MotorModel::ForceBased => 1.0 / motor.stiffness,
                    crate::joint::MotorModel::AccelerationBased => w / motor.stiffness,
                };
                self.solve_anchor_distance(&joint, joint.rest_length, alpha / (h * h));
            }
            XpbdJointKind::Fixed => {
                for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                    self.solve_position_row(
                        joint.a,
                        joint.b,
                        joint.la,
                        joint.lb,
                        axis,
                        0.0,
                        0.0,
                        alpha_tilde,
                    );
                }
                self.solve_angular_lock(&joint, alpha_tilde);
            }
            XpbdJointKind::Revolute | XpbdJointKind::Prismatic => {
                if joint.kind == XpbdJointKind::Prismatic {
                    // Anchor separation perpendicular to the slide axis.
                    let axis = {
                        let ba = &self.bodies[joint.a];
                        (ba.orientation * joint.ax_a).normalize_or(Vec3::Y)
                    };
                    let (t1, t2) = tangent_basis(axis);
                    for t in [t1, t2] {
                        self.solve_position_row(
                            joint.a,
                            joint.b,
                            joint.la,
                            joint.lb,
                            t,
                            0.0,
                            0.0,
                            alpha_tilde,
                        );
                    }
                } else {
                    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                        self.solve_position_row(
                            joint.a,
                            joint.b,
                            joint.la,
                            joint.lb,
                            axis,
                            0.0,
                            0.0,
                            alpha_tilde,
                        );
                    }
                }
                self.solve_axis_alignment(&joint, alpha_tilde);
            }
        }
    }

    /// Shared distance-row solve for distance/rope/spring joints:
    /// `C = |pa − pb| − rest` with compliance `alpha_tilde` (rigid when 0).
    /// Rope gates on the stretched side before calling; spring passes its
    /// own per-joint compliance. Stateless across iterations (see the
    /// distance arm above).
    fn solve_anchor_distance(&mut self, joint: &XpbdJoint, rest: f32, alpha_tilde: f32) {
        let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
        let pa = ba.position + ba.orientation * joint.la;
        let pb = bb.position + bb.orientation * joint.lb;
        let delta = pa - pb;
        let dist = delta.length();
        if dist < NEAR_ZERO {
            return;
        }
        let w = generalized_inverse_mass(ba, bb, delta / dist, pa - ba.position, pb - bb.position);
        if w <= 0.0 {
            return;
        }
        let dlambda = (rest - dist) / (w + alpha_tilde);
        if dlambda != 0.0 {
            let (ba, bb) = pair_mut(&mut self.bodies, joint.a, joint.b);
            let n = delta / dist;
            apply_position_correction(ba, 1.0, n, pa - ba.position, dlambda);
            apply_position_correction(bb, -1.0, n, pb - bb.position, dlambda);
        }
    }

    /// Velocity-level spring damping over the substep's spring joints
    /// (mirrors the friction velocity pass): a viscous impulse
    /// `−c·v_sep·h` along the anchor delta, with the force/acceleration
    /// interpretation folded into `c`. Position-level XPBD rows carry no
    /// damping, so without this a stiffness-only spring would ring —
    /// with it the oscillator settles. Explicit Euler: needs
    /// `damping · h · inverse_mass < 2` (same bound as the SI explicit
    /// spring); the motor budget clamps the impulse like everywhere else.
    fn solve_spring_damping(&mut self, h: f32) {
        if h <= 0.0 {
            return;
        }
        for j in 0..self.joints.len() {
            let joint = self.joints[j];
            if joint.kind != XpbdJointKind::Spring {
                continue;
            }
            let Some(motor) = joint.spec.spring_motor() else {
                continue;
            };
            if motor.damping <= 0.0 || motor.max_force < 0.0 {
                continue;
            }
            let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
            let pa = ba.position + ba.orientation * joint.la;
            let pb = bb.position + bb.orientation * joint.lb;
            let delta = pa - pb;
            let dist = delta.length();
            if dist < NEAR_ZERO {
                continue;
            }
            let n = delta / dist;
            let ra = pa - ba.position;
            let rb = pb - bb.position;
            let w = generalized_inverse_mass(ba, bb, n, ra, rb);
            if w <= 0.0 {
                continue;
            }
            // Separation rate of B away from A in the `pa − pb` frame:
            // `(va − vb)·n` grows as B recedes along `−n`. The damper
            // opposes it (`jt < 0` pulls the pair back together).
            let vrel = (point_velocity(ba, ra) - point_velocity(bb, rb)).dot(n);
            let c = match motor.model {
                crate::joint::MotorModel::ForceBased => motor.damping,
                crate::joint::MotorModel::AccelerationBased => motor.damping / w,
            };
            let cap = motor.max_force * h;
            let jt = (-c * vrel * h).clamp(-cap, cap);
            if jt != 0.0 {
                let (ba, bb) = pair_mut(&mut self.bodies, joint.a, joint.b);
                apply_velocity_impulse(ba, 1.0, n, ra, jt);
                apply_velocity_impulse(bb, -1.0, n, rb, jt);
            }
        }
    }

    /// One scalar positional row: `C = (pa − pb)·n − target`, equality.
    #[allow(clippy::too_many_arguments)]
    fn solve_position_row(
        &mut self,
        a: usize,
        b: usize,
        la: Vec3,
        lb: Vec3,
        n: Vec3,
        target: f32,
        lambda: f32,
        alpha_tilde: f32,
    ) {
        let (ba, bb) = pair_mut(&mut self.bodies, a, b);
        let pa = ba.position + ba.orientation * la;
        let pb = bb.position + bb.orientation * lb;
        let c = (pa - pb).dot(n) - target;
        let w = generalized_inverse_mass(ba, bb, n, pa - ba.position, pb - bb.position);
        if w <= 0.0 {
            return;
        }
        let dlambda = delta_lambda(c, lambda, w, alpha_tilde);
        if dlambda != 0.0 {
            apply_position_correction(ba, 1.0, n, pa - ba.position, dlambda);
            apply_position_correction(bb, -1.0, n, pb - bb.position, dlambda);
        }
    }

    /// Fixed-joint orientation lock: drive the relative-rotation vector to
    /// zero through three scalar angular rows (small-angle linearization).
    fn solve_angular_lock(&mut self, joint: &XpbdJoint, alpha_tilde: f32) {
        let theta = {
            let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
            let mut q_err = bb.orientation * (ba.orientation * joint.q_ref).inverse();
            if q_err.w < 0.0 {
                q_err = Quat::from_xyzw(-q_err.x, -q_err.y, -q_err.z, -q_err.w);
            }
            2.0 * q_err.xyz()
        };
        if theta.length_squared() < AXIS_REST_LEN2 {
            return;
        }
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let c = theta.dot(axis);
            if c.abs() < DEGENERATE_LEN2 {
                continue;
            }
            let w = {
                let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
                angular_inverse_mass(ba, bb, axis)
            };
            if w <= 0.0 {
                continue;
            }
            let dlambda = delta_lambda(c, 0.0, w, alpha_tilde);
            if dlambda != 0.0 {
                let (ba, bb) = pair_mut(&mut self.bodies, joint.a, joint.b);
                apply_angular_correction(ba, 1.0, axis, dlambda);
                apply_angular_correction(bb, -1.0, axis, dlambda);
            }
        }
    }

    /// Hinge/slide axis alignment: `a×b` with the twist about the axis
    /// projected out, so one rotational degree of freedom stays free.
    fn solve_axis_alignment(&mut self, joint: &XpbdJoint, alpha_tilde: f32) {
        let corr = {
            let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
            let axis_a = (ba.orientation * joint.ax_a).normalize_or(Vec3::Y);
            let axis_b = (bb.orientation * joint.ax_b).normalize_or(Vec3::Y);
            let mut corr = axis_a.cross(axis_b);
            corr -= axis_a * corr.dot(axis_a);
            corr
        };
        if corr.length_squared() < AXIS_REST_LEN2 {
            return;
        }
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let c = corr.dot(axis);
            if c.abs() < DEGENERATE_LEN2 {
                continue;
            }
            let w = {
                let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
                angular_inverse_mass(ba, bb, axis)
            };
            if w <= 0.0 {
                continue;
            }
            let dlambda = delta_lambda(c, 0.0, w, alpha_tilde);
            if dlambda != 0.0 {
                let (ba, bb) = pair_mut(&mut self.bodies, joint.a, joint.b);
                apply_angular_correction(ba, 1.0, axis, dlambda);
                apply_angular_correction(bb, -1.0, axis, dlambda);
            }
        }
    }

    /// Velocity pass over the substep contacts: restitution for fast
    /// approaches plus Coulomb friction clamped by the position-level
    /// normal impulse `λ/h`.
    fn solve_velocities(&mut self, contacts: &[Contact], h: f32) {
        for c in contacts {
            // Restitution first (may separate the pair); friction reads the
            // post-bounce velocities below.
            self.solve_restitution(c);
            let normal_impulse = c.lambda / h;
            if normal_impulse <= 0.0 {
                continue;
            }
            // Geometric-mean Coulomb combine (Box2D parity).
            let mu = (self.bodies[c.a].friction * self.bodies[c.b].friction).sqrt();
            if mu <= 0.0 {
                continue;
            }
            let (ba, bb) = (&self.bodies[c.a], &self.bodies[c.b]);
            let pa = ba.position + ba.orientation * c.la;
            let pb = bb.position + bb.orientation * c.lb;
            let ra = pa - ba.position;
            let rb = pb - bb.position;
            let vrel = point_velocity(ba, ra) - point_velocity(bb, rb);
            let vt = vrel - c.n * vrel.dot(c.n);
            let speed = vt.length();
            if speed < NEAR_ZERO {
                continue;
            }
            let t = vt / speed;
            let w = generalized_inverse_mass(ba, bb, t, ra, rb);
            if w <= 0.0 {
                continue;
            }
            let max_friction = mu * normal_impulse;
            let jt = (-speed / w).clamp(-max_friction, max_friction);
            if jt != 0.0 {
                let (ba, bb) = pair_mut(&mut self.bodies, c.a, c.b);
                apply_velocity_impulse(ba, 1.0, t, ra, jt);
                apply_velocity_impulse(bb, -1.0, t, rb, jt);
            }
        }
    }

    /// One-shot bounce for a single contact when the approach speed clears
    /// the restitution threshold.
    fn solve_restitution(&mut self, c: &Contact) {
        let (ba, bb) = (&self.bodies[c.a], &self.bodies[c.b]);
        let pa = ba.position + ba.orientation * c.la;
        let pb = bb.position + bb.orientation * c.lb;
        let ra = pa - ba.position;
        let rb = pb - bb.position;
        let vn = (point_velocity(ba, ra) - point_velocity(bb, rb)).dot(c.n);
        if vn >= -self.restitution_threshold {
            return;
        }
        let w = generalized_inverse_mass(ba, bb, c.n, ra, rb);
        if w <= 0.0 {
            return;
        }
        let e = HALF * (ba.restitution + bb.restitution);
        let j = -(1.0 + e) * vn / w;
        if j != 0.0 {
            let (ba, bb) = pair_mut(&mut self.bodies, c.a, c.b);
            apply_velocity_impulse(ba, 1.0, c.n, ra, j);
            apply_velocity_impulse(bb, -1.0, c.n, rb, j);
        }
    }

    /// Particle CCD clamp, once per substep: each awake, unpinned particle
    /// whose integrated displacement outruns its own contact radius is cast
    /// as a sphere from its substep-start pose along the displacement
    /// through [`cast_shape`] (conservative advancement, linear only).
    /// On a hit short of the path end the prediction is pulled back to the
    /// hit (minus [`CCD_BACKOFF`]); the discrete pass then owns the resting
    /// contact. Particles already touching report no hit (the sweep's
    /// contract) and particles moving less than one radius per substep
    /// cannot skip the discrete witness by construction.
    fn clamp_soft_ccd(bodies: &[RigidBody], soft: &mut SoftBody) {
        if soft.contact_radius <= 0.0 {
            return;
        }
        let sphere = Shape::Sphere {
            radius: soft.contact_radius,
        };
        // Index-based (not `iter_mut`): the per-target filter calls
        // `soft.can_couple_with`, which needs a clean immutable borrow.
        for pi in 0..soft.particles.len() {
            let (prev, delta) = {
                let p = &soft.particles[pi];
                if p.inv_mass <= 0.0 {
                    continue;
                }
                (p.prev_position, p.position - p.prev_position)
            };
            let len = delta.length();
            if len <= soft.contact_radius || !len.is_finite() {
                continue;
            }
            let targets = bodies.iter().enumerate().filter_map(|(b, body)| {
                if body.is_trigger || !soft.can_couple_with(body) {
                    return None;
                }
                Some((
                    BodyHandle::from(b),
                    ShapeRef {
                        shape: &body.shape,
                        pos: body.position,
                        rot: body.orientation,
                    },
                ))
            });
            let mover = ShapeRef {
                shape: &sphere,
                pos: prev,
                rot: Quat::IDENTITY,
            };
            if let Some(hit) = cast_shape(mover, delta, targets) {
                let t = (hit.t - CCD_BACKOFF).clamp(0.0, len);
                soft.particles[pi].position = prev + delta / len * t;
            }
        }
    }

    /// Impact wake pass over the freshly discovered contacts: a contact
    /// whose relative normal speed clears [`WAKE_IMPACT_SPEED`] wakes the
    /// sleeping member(s). Slow settling never wakes — that is what lets a
    /// stack fall asleep piece by piece without chatter.
    fn wake_on_impact(&mut self, contacts: &[Contact], soft_contacts: &[SoftContact]) {
        for c in contacts {
            let (ba, bb) = (&self.bodies[c.a], &self.bodies[c.b]);
            let ra = ba.orientation * c.la;
            let rb = bb.orientation * c.lb;
            let vn = (point_velocity(ba, ra) - point_velocity(bb, rb)).dot(c.n);
            if vn.abs() > WAKE_IMPACT_SPEED {
                self.wake_rigid(c.a);
                self.wake_rigid(c.b);
            }
        }
        for c in soft_contacts {
            let (Some(soft), Some(body)) = (self.soft_bodies.get(c.soft), self.bodies.get(c.body))
            else {
                continue;
            };
            let Some(particle) = soft.particles.get(c.particle) else {
                continue;
            };
            let rb = body.orientation * c.lb;
            let vn = (particle.velocity - point_velocity(body, rb)).dot(c.n);
            if vn.abs() > WAKE_IMPACT_SPEED {
                self.wake_soft(c.soft);
                self.wake_rigid(c.body);
            }
        }
    }

    /// Wake one rigid body: restore its sleep-zeroed inverse mass/inertia
    /// and reset the quiet timer. Non-dynamics and invalid indices are a
    /// no-op (statics are never asleep, kinematics never sleep).
    fn wake_rigid(&mut self, h: usize) {
        let Some(asleep) = self.rigid_asleep.get_mut(h) else {
            return;
        };
        if !*asleep {
            return;
        }
        *asleep = false;
        if let Some(t) = self.rigid_quiet.get_mut(h) {
            *t = 0.0;
        }
        if let Some(b) = self.bodies.get_mut(h) {
            b.wake_restore();
        }
    }

    /// Wake one soft body on impact (no baseline change — motion continues;
    /// explicit teleports use [`XpbdEngine::wake_soft_body`] instead).
    fn wake_soft(&mut self, s: usize) {
        if let Some(st) = self.soft_sleep.get_mut(s) {
            st.asleep = false;
            st.quiet = 0.0;
        }
    }

    /// Per-step sleep bookkeeping (sequential-impulse parity: 0.15 m/s,
    /// 0.5 s): a dynamic rigid body whose linear and angular speeds stay
    /// below the gates for [`SLEEP_TIME`] freezes (Jolt semantics — zeroed
    /// inverse mass/inertia, restored on wake); a soft body whose every
    /// particle stays below the linear gate freezes with zeroed velocities.
    /// Runs once per [`PhysicsEngine::step`], after the substeps.
    fn update_sleep(&mut self, dt: f32) {
        for (h, b) in self.bodies.iter_mut().enumerate() {
            if b.body_type != BodyType::Dynamic {
                self.rigid_asleep[h] = false;
                self.rigid_quiet[h] = 0.0;
                continue;
            }
            if self.rigid_asleep[h] {
                continue;
            }
            let slow = b.velocity.length() < SLEEP_LIN && b.angular_velocity.length() < SLEEP_ANG;
            if slow {
                self.rigid_quiet[h] += dt;
                if self.rigid_quiet[h] >= SLEEP_TIME {
                    self.rigid_asleep[h] = true;
                    b.sleep_staticify();
                }
            } else {
                self.rigid_quiet[h] = 0.0;
            }
        }
        for (s, soft) in self.soft_bodies.iter_mut().enumerate() {
            if self.soft_sleep[s].asleep {
                for p in &mut soft.particles {
                    p.velocity = Vec3::ZERO;
                }
                continue;
            }
            let slow = soft
                .particles
                .iter()
                .all(|p| p.velocity.length() < SLEEP_LIN);
            if slow {
                self.soft_sleep[s].quiet += dt;
                if self.soft_sleep[s].quiet >= SLEEP_TIME {
                    self.soft_sleep[s].asleep = true;
                    for p in &mut soft.particles {
                        p.velocity = Vec3::ZERO;
                    }
                }
            } else {
                self.soft_sleep[s].quiet = 0.0;
            }
        }
    }

    /// Reconcile the soft↔rigid touch set at the final poses and queue
    /// begin/end events per `(soft body, rigid body)` pair. Touching is
    /// overlap (`dist ≤ 0`) plus a [`CONTACT_BEGIN_SLOP`] grace band (only
    /// previous pairs are re-checked against the band, so resting jitter
    /// emits no flicker); frozen pairs keep their prior state (sleep emits
    /// no transitions — sequential-impulse parity).
    fn reconcile_soft_events(&mut self) {
        let n_bodies = self.bodies.len();
        let mut current = BTreeSet::new();
        for c in self.discover_soft_contacts() {
            current.insert((c.soft, c.particle, c.body));
        }
        // Grace band for previously touching triples missing from the fresh
        // overlap set (solver residuals live here).
        let missing: Vec<(usize, usize, usize)> =
            self.soft_touch.difference(&current).copied().collect();
        for (s, p, b) in missing {
            let (Some(soft), Some(body)) = (self.soft_bodies.get(s), self.bodies.get(b)) else {
                continue;
            };
            if soft.contact_radius <= 0.0 {
                continue;
            }
            let Some(particle) = soft.particles.get(p) else {
                continue;
            };
            let sphere = Shape::Sphere {
                radius: soft.contact_radius,
            };
            let d = shape_distance(
                ShapeRef {
                    shape: &sphere,
                    pos: particle.position,
                    rot: Quat::IDENTITY,
                },
                ShapeRef {
                    shape: &body.shape,
                    pos: body.position,
                    rot: body.orientation,
                },
            );
            if d.dist.is_finite() && d.dist <= CONTACT_BEGIN_SLOP {
                current.insert((s, p, b));
            }
        }
        // Frozen pairs keep their prior state.
        for &(s, p, b) in &self.soft_touch {
            if s < self.soft_bodies.len()
                && b < n_bodies
                && p < self.soft_bodies[s].particles.len()
                && self.soft_sleep[s].asleep
                && (self.bodies[b].body_type != BodyType::Dynamic || self.rigid_asleep[b])
                && !current.contains(&(s, p, b))
            {
                current.insert((s, p, b));
            }
        }
        let previous = std::mem::replace(&mut self.soft_touch, current);
        // The warm-start cache only holds live pairs from here on.
        self.soft_lambda_cache
            .retain(|k, _| self.soft_touch.contains(k));
        let pair_set = |set: &BTreeSet<(usize, usize, usize)>| {
            let mut pairs: Vec<(usize, usize)> = set.iter().map(|&(s, _, b)| (s, b)).collect();
            pairs.sort_unstable();
            pairs.dedup();
            pairs
        };
        let before = pair_set(&previous);
        let after = pair_set(&self.soft_touch);
        for (s, b) in after.iter().filter(|k| !before.contains(k)) {
            self.soft_contact_events.push(SoftContactEvent {
                soft: SoftHandle::from(*s),
                body: BodyHandle::from(*b),
                kind: ContactEventKind::Begin,
            });
        }
        for (s, b) in before.iter().filter(|k| !after.contains(k)) {
            self.soft_contact_events.push(SoftContactEvent {
                soft: SoftHandle::from(*s),
                body: BodyHandle::from(*b),
                kind: ContactEventKind::End,
            });
        }
    }

    /// Canonical key for the no-collide joint set.
    fn pair_key(a: usize, b: usize) -> (usize, usize) {
        (a.min(b), a.max(b))
    }

    /// Rebuild the no-collide set after structural edits (fixed joints are
    /// excluded: weld assemblies keep their structural contacts).
    fn rebuild_joint_pairs(&mut self) {
        self.joint_pairs.clear();
        for j in &self.joints {
            if j.kind != XpbdJointKind::Fixed {
                self.joint_pairs.insert(Self::pair_key(j.a, j.b));
            }
        }
    }

    /// Shared raycast kernel over all bodies (local-frame query, like the
    /// sequential-impulse engine, so hits agree by construction).
    fn raycast_body(&self, ray: &Ray, handle: usize, max_dist: f32) -> Option<RaycastHit> {
        if max_dist.is_nan() || max_dist < 0.0 || !ray.direction.is_finite() {
            return None;
        }
        let body = &self.bodies[handle];
        let inverse = body.orientation.inverse();
        let origin = inverse * (ray.origin - body.position);
        let direction = inverse * ray.direction;
        let (distance, local_normal) = raycast_shape_hit(&body.shape, origin, direction, max_dist)?;
        Some(RaycastHit {
            handle: BodyHandle::from(handle),
            point: ray.point_at(distance),
            normal: (body.orientation * local_normal).normalize_or(Vec3::Y),
            distance,
        })
    }
}

impl PhysicsEngine for XpbdEngine {
    fn step(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let h = dt / self.substeps as f32;
        for _ in 0..self.substeps {
            if self.bodies.is_empty() && self.soft_bodies.is_empty() {
                return;
            }
            self.substep(h);
        }
        self.reconcile_soft_events();
        self.update_sleep(dt);
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        self.bodies.push(body);
        self.rigid_asleep.push(false);
        self.rigid_quiet.push(0.0);
        BodyHandle::from(self.bodies.len() - 1)
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        let hi = handle.index();
        if hi >= self.bodies.len() {
            return;
        }
        let last = self.bodies.len() - 1;
        self.bodies.swap_remove(hi);
        self.rigid_asleep.swap_remove(hi);
        self.rigid_quiet.swap_remove(hi);
        let map = |h: usize| if h == last { hi } else { h };
        self.joints.retain(|j| j.a != hi && j.b != hi);
        for j in &mut self.joints {
            j.a = map(j.a);
            j.b = map(j.b);
        }
        self.rebuild_joint_pairs();
        // Indices shifted: soft caches keyed by body index go stale.
        self.soft_lambda_cache.clear();
        self.soft_touch.clear();
        self.soft_contact_events.clear();
    }

    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        self.bodies.get(usize::from(handle))
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        self.bodies.get_mut(usize::from(handle))
    }

    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Result<JointHandle, JointError> {
        validate_joint(&kind)?;
        let (ia, ib) = (usize::from(body_a), usize::from(body_b));
        if ia == ib {
            return Err(JointError::SelfJoint { handle: ia });
        }
        if ia >= self.bodies.len() || ib >= self.bodies.len() {
            return Err(JointError::InvalidHandles { a: ia, b: ib });
        }
        let model = match kind {
            JointKind::Ball { .. } => XpbdJointKind::Ball,
            JointKind::Revolute { .. } => XpbdJointKind::Revolute,
            JointKind::Prismatic { .. } => XpbdJointKind::Prismatic,
            JointKind::Fixed { .. } => XpbdJointKind::Fixed,
            JointKind::Distance { .. } => XpbdJointKind::Distance,
            JointKind::Rope { .. } => XpbdJointKind::Rope,
            JointKind::Spring { .. } => XpbdJointKind::Spring,
            // Wheel needs a suspension spring with axle lock, gear couples
            // other joints and six-DOF needs per-axis configs: all rejected
            // rather than silently mis-solved.
            JointKind::Wheel { .. } | JointKind::Gear { .. } | JointKind::SixDof { .. } => {
                return Err(JointError::Unsupported {
                    detail: "xpbd supports ball/revolute/prismatic/fixed/distance/rope/spring only"
                        .to_string(),
                });
            }
        };
        let resolved = resolve_joint(
            &kind,
            self.bodies[ia].position,
            self.bodies[ia].orientation,
            self.bodies[ib].position,
            self.bodies[ib].orientation,
        )
        .ok_or_else(|| JointError::BadAxis {
            detail: "unresolvable joint frames".to_string(),
        })?;
        // Distance rest length is the assembly separation; rope maximum and
        // spring rest ride in the spec (immutable after creation).
        let rest_length = match kind {
            JointKind::Rope { max_distance, .. } => max_distance,
            JointKind::Spring { motor, .. } => motor.target_position,
            _ => resolved.ref_distance,
        };
        self.joints.push(XpbdJoint {
            a: ia,
            b: ib,
            kind: model,
            spec: kind,
            reference: JointReference::from(resolved),
            la: resolved.la,
            lb: resolved.lb,
            ax_a: resolved.ax_a,
            ax_b: resolved.ax_b,
            rest_length,
            q_ref: resolved.ref_quat,
            servo: None,
        });
        self.rebuild_joint_pairs();
        // A new joint changes the constraint set: wake both members so a
        // frozen assembly cannot hold a stale pose (SI parity on edits).
        self.wake_rigid(ia);
        self.wake_rigid(ib);
        Ok(JointHandle::from(self.joints.len() - 1))
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        if handle.index() >= self.joints.len() {
            return;
        }
        self.joints.swap_remove(handle.index());
        self.rebuild_joint_pairs();
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Result<Option<RaycastHit>, QueryError> {
        crate::errors::check_ray_input(ray.origin, ray.direction, max_dist)?;
        let mut closest: Option<RaycastHit> = None;
        for handle in 0..self.bodies.len() {
            if let Some(hit) = self.raycast_body(&ray, handle, max_dist) {
                match &closest {
                    Some(best) if hit.distance < best.distance => closest = Some(hit),
                    None => closest = Some(hit),
                    _ => {}
                }
            }
        }
        Ok(closest)
    }

    fn shapecast(&self, shape: &Shape, from: Vec3, to: Vec3) -> Option<RaycastHit> {
        let mover = ShapeRef {
            shape,
            pos: from,
            rot: Quat::IDENTITY,
        };
        let targets = self.bodies.iter().enumerate().map(|(h, b)| {
            (
                BodyHandle::from(h),
                ShapeRef {
                    shape: &b.shape,
                    pos: b.position,
                    rot: b.orientation,
                },
            )
        });
        cast_shape(mover, to - from, targets).map(|h| RaycastHit {
            handle: h.handle,
            point: h.point,
            normal: h.normal,
            distance: h.t,
        })
    }

    fn wake_body(&mut self, handle: BodyHandle) {
        self.wake_rigid(handle.index());
    }
}

/// Whether the [`crate::Engine`] orchestrator can migrate this joint onto
/// the XPBD path: ball, revolute, prismatic, fixed, distance, rope and
/// spring solve structurally here (see [`XpbdEngine::add_joint`]). Wheel
/// needs a suspension spring with axle lock, gear couples other joints and
/// six-DOF needs per-axis configs — all rejected rather than silently
/// mis-solved, so the orchestrator parks them outside the migration
/// instead of panicking.
pub(crate) fn xpbd_supports_joint(kind: &JointKind) -> bool {
    matches!(
        kind,
        JointKind::Ball { .. }
            | JointKind::Revolute { .. }
            | JointKind::Prismatic { .. }
            | JointKind::Fixed { .. }
            | JointKind::Distance { .. }
            | JointKind::Rope { .. }
            | JointKind::Spring { .. }
    )
}

/// Drop XPBD-unsupported joints from a migration snapshot, keeping the
/// survivors in stable order with gear references remapped (same
/// [`crate::migration::joint_remap`] discipline as body-removal cleanup).
/// Bodies and soft bodies are untouched, so their handles never move —
/// only joint handles past a dropped joint compact.
pub(crate) fn retain_supported_joints(joints: &mut Vec<JointSnapshot>) {
    let kinds: Vec<JointKind> = joints.iter().map(|j| j.spec).collect();
    let removed: Vec<bool> = kinds.iter().map(|k| !xpbd_supports_joint(k)).collect();
    if removed.iter().all(|&r| !r) {
        return;
    }
    let remap = crate::migration::joint_remap(&kinds, removed);
    let mut old = 0;
    joints.retain_mut(|j| {
        let keep = remap[old].is_some();
        old += 1;
        if keep {
            crate::migration::remap_gear(&mut j.spec, &remap);
        }
        keep
    });
}

/// XPBD scalar update (Macklin et al. 2016, Eq. 18):
/// `Δλ = (−C − α̃·λ) / (w + α̃)` with the time-scaled compliance
/// `α̃ = α/h²` folded in by the caller. `α̃ = 0` is exactly PBD.
/// Shared with [`crate::soft`] (deformable distance rows reuse it 1:1).
pub(crate) fn delta_lambda(c: f32, lambda: f32, w_sum: f32, alpha_tilde: f32) -> f32 {
    (-c - alpha_tilde * lambda) / (w_sum + alpha_tilde)
}

/// World-space inverse inertia applied to `v`: `R·(D⁻¹·(Rᵀ·v))`.
/// Zero diagonal entries (locked/static axes) contribute nothing.
fn apply_inv_inertia(diag: Vec3, q: Quat, v: Vec3) -> Vec3 {
    let local = q.inverse() * v;
    let scaled = Vec3::new(
        if diag.x > 0.0 { local.x / diag.x } else { 0.0 },
        if diag.y > 0.0 { local.y / diag.y } else { 0.0 },
        if diag.z > 0.0 { local.z / diag.z } else { 0.0 },
    );
    q * scaled
}

/// World-space inertia applied to `v`: `R·(D·(Rᵀ·v))`.
fn world_inertia(diag: Vec3, q: Quat, v: Vec3) -> Vec3 {
    q * (diag * (q.inverse() * v))
}

/// Generalized inverse mass for a positional constraint along `n` through
/// the body-frame levers `ra`/`rb` (Müller et al., rigid-body XPBD):
/// `w = Σ (1/m + (r×n)ᵀ·I⁻¹·(r×n))`.
fn generalized_inverse_mass(a: &RigidBody, b: &RigidBody, n: Vec3, ra: Vec3, rb: Vec3) -> f32 {
    let mut w = a.inv_mass + b.inv_mass;
    let ta = ra.cross(n);
    w += ta.dot(apply_inv_inertia(a.inertia, a.orientation, ta));
    let tb = rb.cross(n);
    w += tb.dot(apply_inv_inertia(b.inertia, b.orientation, tb));
    w
}

/// Generalized inverse mass for a pure angular constraint about `n`:
/// `w = Σ nᵀ·I⁻¹·n`.
fn angular_inverse_mass(a: &RigidBody, b: &RigidBody, n: Vec3) -> f32 {
    n.dot(apply_inv_inertia(a.inertia, a.orientation, n))
        + n.dot(apply_inv_inertia(b.inertia, b.orientation, n))
}

/// Apply `Δλ` of a positional constraint to one body: translate by
/// `M⁻¹·∇Cᵀ·Δλ` and rotate by the resulting torque arm (the quaternion
/// sum is the linearized exponential map; renormalized after the solve).
fn apply_position_correction(body: &mut RigidBody, sign: f32, n: Vec3, r: Vec3, dlambda: f32) {
    if body.body_type != BodyType::Dynamic || body.inv_mass <= 0.0 {
        return;
    }
    let impulse = n * (sign * dlambda);
    body.position += impulse * body.inv_mass;
    let dtheta = apply_inv_inertia(body.inertia, body.orientation, r.cross(impulse));
    let q = body.orientation;
    let dq = Quat::from_xyzw(dtheta.x * HALF, dtheta.y * HALF, dtheta.z * HALF, 0.0) * q;
    body.orientation = Quat::from_xyzw(q.x + dq.x, q.y + dq.y, q.z + dq.z, q.w + dq.w);
}

/// Apply `Δλ` of an angular constraint: pure rotation, no translation.
fn apply_angular_correction(body: &mut RigidBody, sign: f32, n: Vec3, dlambda: f32) {
    if body.body_type != BodyType::Dynamic || body.inv_mass <= 0.0 {
        return;
    }
    let dtheta = apply_inv_inertia(body.inertia, body.orientation, n * (sign * dlambda));
    let q = body.orientation;
    let dq = Quat::from_xyzw(dtheta.x * HALF, dtheta.y * HALF, dtheta.z * HALF, 0.0) * q;
    body.orientation = Quat::from_xyzw(q.x + dq.x, q.y + dq.y, q.z + dq.z, q.w + dq.w);
}

/// Impulse-velocity update for one body at the lever `r`.
fn apply_velocity_impulse(body: &mut RigidBody, sign: f32, n: Vec3, r: Vec3, j: f32) {
    if body.body_type != BodyType::Dynamic || body.inv_mass <= 0.0 {
        return;
    }
    let impulse = n * (sign * j);
    body.velocity += impulse * body.inv_mass;
    body.angular_velocity += apply_inv_inertia(body.inertia, body.orientation, r.cross(impulse));
}

/// Point velocity `v + ω×r` at the world lever `r`.
fn point_velocity(body: &RigidBody, r: Vec3) -> Vec3 {
    body.velocity + body.angular_velocity.cross(r)
}

/// Exact exponential-map orientation integration, renormalized.
fn integrate_orientation(q: Quat, w: Vec3, h: f32) -> Quat {
    let half = h * HALF;
    let dq = Quat::from_xyzw(w.x * half, w.y * half, w.z * half, 0.0) * q;
    Quat::from_xyzw(q.x + dq.x, q.y + dq.y, q.z + dq.z, q.w + dq.w).normalize()
}

/// Angular velocity from the substep rotation `q·q_prev⁻¹` (linearized
/// angle-axis recovery with a rest deadband; positions are O(1) `f32`, so
/// sub-epsilon spin is dust).
fn angular_velocity_from_delta(q: Quat, q_prev: Quat, h: f32) -> Vec3 {
    let mut dq = q * q_prev.inverse();
    if dq.w < 0.0 {
        dq = Quat::from_xyzw(-dq.x, -dq.y, -dq.z, -dq.w);
    }
    let spin = 2.0 * dq.xyz();
    if spin.length_squared() < COINCIDENT_LEN2 {
        Vec3::ZERO
    } else {
        spin / h
    }
}

/// Two mutable body borrows by index (`a != b`, enforced by the caller —
/// joints reject self-pairs at creation and contacts pair distinct bodies).
fn pair_mut(bodies: &mut [RigidBody], a: usize, b: usize) -> (&mut RigidBody, &mut RigidBody) {
    debug_assert_ne!(a, b);
    if a < b {
        let (lo, hi) = bodies.split_at_mut(b);
        (&mut lo[a], &mut hi[0])
    } else {
        let (lo, hi) = bodies.split_at_mut(a);
        (&mut hi[0], &mut lo[b])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box dropped onto a static floor must come to rest on it: bounded
    /// penetration, near-zero velocity — the basic XPBD contact sanity check.
    #[test]
    fn box_settles_on_static_floor() {
        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        // Floor top face at y = 0; the falling box starts fully clear of it.
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -5.0, 0.0),
            Vec3::splat(5.0),
            0.0,
        ));
        let top = engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::splat(HALF),
            1.0,
        ));
        for _ in 0..180 {
            engine.step(1.0 / 60.0);
        }
        let b = engine.get_body(top).expect("box survives");
        // Floor top at y=0, box half-height 0.5: rest center ≈ 0.5.
        assert!(
            (b.position.y - HALF).abs() < 0.05,
            "settled height {}, want ~HALF",
            b.position.y
        );
        assert!(
            b.velocity.length() < 0.25,
            "rest velocity {} too high",
            b.velocity.length()
        );
    }

    /// A ball joint must hold a pendulum bob near its anchor length: the
    /// positional rows converge under substeps instead of drifting.
    #[test]
    fn ball_joint_holds_pendulum_length() {
        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let anchor = engine.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.1, 0.0));
        let bob = engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, -2.0, 0.0), 0.2, 1.0));
        engine
            .add_joint(
                anchor,
                bob,
                JointKind::Ball {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::new(0.0, 2.0, 0.0),
                },
            )
            .expect("ball joint accepted");
        for _ in 0..120 {
            engine.step(1.0 / 60.0);
        }
        let (pa, pb) = (
            engine.get_body(anchor).expect("anchor").position,
            engine.get_body(bob).expect("bob").position,
        );
        // Anchor at origin, bob anchor 2 m above its center: coincidence
        // means the bob center hangs 2 m below the origin.
        let separation = (pb - pa).length();
        assert!(
            (separation - 2.0).abs() < 0.05,
            "anchor separation {separation}, want ~2.0"
        );
    }

    /// Positive joint compliance must visibly soften a distance rod under a
    /// static axial load compared to the rigid default: a heavy ball hangs
    /// straight below its anchor, so the rod carries `m·g` of tension and
    /// the compliant equilibrium stretch is `α·m·g`.
    #[test]
    fn compliance_softens_distance_rod() {
        fn stretch(compliance: f32) -> f32 {
            let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
            engine.set_joint_compliance(compliance);
            let a = engine.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.1, 0.0));
            let b = engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, -3.0, 0.0), 0.1, 10.0));
            engine
                .add_joint(
                    a,
                    b,
                    JointKind::Distance {
                        local_anchor_a: Vec3::ZERO,
                        local_anchor_b: Vec3::ZERO,
                    },
                )
                .expect("distance joint accepted");
            for _ in 0..300 {
                engine.step(1.0 / 60.0);
            }
            let (pa, pb) = (
                engine.get_body(a).expect("a").position,
                engine.get_body(b).expect("b").position,
            );
            (pa - pb).length() - 3.0
        }
        let rigid = stretch(0.0).abs();
        let soft = stretch(2e-4).abs();
        assert!(rigid < 2e-3, "rigid rod stretch {rigid} too large");
        assert!(
            soft > 8e-3,
            "soft rod ({soft}) must stretch more than rigid ({rigid})"
        );
    }

    /// Unsupported joint models are rejected instead of mis-solved.
    #[test]
    fn unsupported_joints_return_none() {
        let mut engine = XpbdEngine::new(Vec3::ZERO);
        let a = engine.add_body(RigidBody::new_sphere(Vec3::ZERO, HALF, 1.0));
        let b = engine.add_body(RigidBody::new_sphere(Vec3::X, HALF, 1.0));
        assert!(
            engine
                .add_joint(
                    a,
                    b,
                    JointKind::Gear {
                        joint_a: JointHandle::from_raw(0),
                        joint_b: JointHandle::from_raw(0),
                        ratio: 1.0,
                    },
                )
                .is_err(),
            "gear must be rejected"
        );
        assert_eq!(engine.joint_count(), 0);
    }

    /// A hanging chain must keep its total length under gravity (D1.1):
    /// rigid structural rows converge instead of stretching like rubber.
    #[test]
    fn chain_holds_length_under_gravity() {
        use crate::soft::SoftBody;

        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let rope =
            engine.add_soft_body(SoftBody::chain(Vec3::ZERO, Vec3::NEG_Y, 6, HALF, 1.0, 0.0));
        for _ in 0..180 {
            engine.step(1.0 / 60.0);
        }
        let body = engine.get_soft_body(rope).expect("rope survives");
        // Pinned end at the origin, 5 links of 0.5: free end hangs at ≈ −2.5.
        let end = body.particles.last().expect("nonempty").position;
        assert!(
            (end.y + 2.5).abs() < 0.1,
            "free end height {}, want ~-2.5",
            end.y
        );
        let mut worst = 0.0f32;
        for c in &body.constraints {
            let d = (body.particles[c.a.index()].position - body.particles[c.b.index()].position)
                .length();
            worst = worst.max((d - c.rest).abs() / c.rest);
        }
        assert!(worst < 0.02, "max link stretch {worst}, want <2%");
        assert!(
            body.particles.iter().all(|p| p.position.is_finite()),
            "no NaN in rope"
        );
    }

    /// A top-pinned cloth grid must drape (D1.1): the free edge falls below
    /// the pins while structural stretch stays bounded — no PBD rubber.
    #[test]
    fn cloth_grid_drapes_with_bounded_stretch() {
        use crate::soft::{ClothPin, DeformKind, SoftBody};

        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let (cols, rows, spacing) = (6, 6, 0.25);
        let sheet = engine.add_soft_body(SoftBody::cloth_grid(
            Vec3::ZERO,
            cols,
            rows,
            spacing,
            1.0,
            0.0,
            0.0,
            1e-4,
            ClothPin::TopRow,
        ));
        for _ in 0..180 {
            engine.step(1.0 / 60.0);
        }
        let body = engine.get_soft_body(sheet).expect("sheet survives");
        // Free edge (last row) must hang well below the pinned top row.
        let edge_y = body.particles[(rows - 1) * cols..]
            .iter()
            .map(|p| p.position.y)
            .fold(f32::INFINITY, f32::min);
        assert!(
            edge_y < -0.8,
            "free edge at {edge_y}, want drape below -0.8"
        );
        let mut worst = 0.0f32;
        for c in body
            .constraints
            .iter()
            .filter(|c| c.kind == DeformKind::Structural)
        {
            let d = (body.particles[c.a.index()].position - body.particles[c.b.index()].position)
                .length();
            worst = worst.max((d - c.rest).abs() / c.rest);
        }
        assert!(worst < 0.08, "max structural stretch {worst}, want <8%");
        assert!(
            body.particles.iter().all(|p| p.position.is_finite()),
            "no NaN in cloth"
        );
    }

    /// A squashed soft cube must recover its volume (D1.3): with soft edges
    /// the shape alone would stay flat — the global volume row reinflates it.
    /// Edges stay softer than the volume row so the test isolates volume
    /// work; damping keeps the reinflation path quasi-static (an undamped
    /// rigid-volume snap overshoots into a tall limit cycle the soft edges
    /// cannot unwind).
    #[test]
    fn soft_cube_recovers_volume() {
        use crate::soft::SoftBody;

        let mut engine = XpbdEngine::new(Vec3::ZERO);
        let cube = engine.add_soft_body(SoftBody::soft_cube(Vec3::ZERO, 1.0, 1.0, 1e-5, 0.0));
        {
            let body = engine.get_soft_body(cube).expect("cube survives");
            assert!(
                (body.volume_rest - 1.0).abs() < 1e-5,
                "rest volume {}",
                body.volume_rest
            );
            assert_eq!(body.constraints.len(), 12, "cube edges");
            assert_eq!(body.triangles.len(), 12, "cube surface");
            // Squash along Y around the center: volume halves.
            let center = Vec3::splat(HALF);
            let b = engine.get_soft_body_mut(cube).expect("cube mut");
            b.damping = 8.0;
            for p in &mut b.particles {
                p.position = center + (p.position - center) * Vec3::new(1.0, HALF, 1.0);
            }
        }
        for _ in 0..180 {
            engine.step(1.0 / 60.0);
        }
        let body = engine.get_soft_body(cube).expect("cube survives");
        let ratio = body.volume() / body.volume_rest;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "volume ratio {ratio}, want ~1.0"
        );
        let (min_y, max_y) = body
            .particles
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p.position.y), hi.max(p.position.y))
            });
        assert!(
            (max_y - min_y - 1.0).abs() < 0.15,
            "height {}, want ~1.0",
            max_y - min_y
        );
        assert!(
            body.particles.iter().all(|p| p.position.is_finite()),
            "no NaN in cube"
        );
    }

    /// A soft cube dropped on a static floor must rest on it (D1.4): the
    /// particle↔rigid coupling holds the bottom layer at `contact_radius`
    /// without tunneling, while the volume row keeps the cube from
    /// crushing through.
    #[test]
    fn soft_cube_rests_on_static_floor() {
        use crate::soft::SoftBody;

        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -5.0, 0.0),
            Vec3::splat(5.0),
            0.0,
        ));
        let cube = engine.add_soft_body(SoftBody::soft_cube(
            Vec3::new(-HALF, 2.0, -HALF),
            1.0,
            1.0,
            0.0,
            0.0,
        ));
        engine.get_soft_body_mut(cube).expect("cube mut").damping = 4.0;
        for _ in 0..240 {
            engine.step(1.0 / 60.0);
        }
        let body = engine.get_soft_body(cube).expect("cube survives");
        // Floor top at y=0, particle radius 0.1: bottom layer rests at ≈0.1.
        let min_y = body
            .particles
            .iter()
            .map(|p| p.position.y)
            .fold(f32::INFINITY, f32::min);
        assert!(
            min_y > -0.05 && min_y < 0.25,
            "bottom layer at {min_y}, want ~0.1"
        );
        let ratio = body.volume() / body.volume_rest;
        assert!((ratio - 1.0).abs() < 0.1, "volume ratio {ratio}, want ~1.0");
        assert!(
            body.particles.iter().all(|p| p.position.is_finite()),
            "no NaN in landed cube"
        );
    }

    /// Layer/mask filtering on the soft side: a soft cube whose mask
    /// couples with nothing falls straight through the floor, while an
    /// identical default-filter cube rests on it.
    #[test]
    fn soft_layer_mask_filters_coupling() {
        use crate::soft::SoftBody;

        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -5.0, 0.0),
            Vec3::splat(5.0),
            0.0,
        ));
        let coupled = engine.add_soft_body(SoftBody::soft_cube(
            Vec3::new(-1.5, 2.0, -HALF),
            1.0,
            1.0,
            0.0,
            0.0,
        ));
        let mut ghost_body = SoftBody::soft_cube(Vec3::new(HALF, 2.0, -HALF), 1.0, 1.0, 0.0, 0.0);
        ghost_body.collision_mask = 0;
        let ghost = engine.add_soft_body(ghost_body);
        for h in [coupled, ghost] {
            engine.get_soft_body_mut(h).expect("cube mut").damping = 4.0;
        }
        for _ in 0..240 {
            engine.step(1.0 / 60.0);
        }
        let min_y = |engine: &XpbdEngine, h: SoftHandle| {
            engine
                .get_soft_body(h)
                .expect("cube survives")
                .particles
                .iter()
                .map(|p| p.position.y)
                .fold(f32::INFINITY, f32::min)
        };
        assert!(
            (min_y(&engine, coupled) - 0.1).abs() < 0.15,
            "coupled cube rests at ~0.1, got {}",
            min_y(&engine, coupled)
        );
        assert!(
            min_y(&engine, ghost) < -1.0,
            "masked-out cube must fall through, got {}",
            min_y(&engine, ghost)
        );
    }

    /// Coulomb friction in soft↔rigid contacts: the same cube on the same
    /// 12° incline sticks with high soft friction and slides with zero.
    #[test]
    fn soft_friction_holds_cube_on_incline() {
        use crate::soft::SoftBody;

        fn slide(mu: f32) -> f32 {
            let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
            engine.set_soft_friction(mu);
            let angle = 12.0f32.to_radians();
            let tilt = Quat::from_rotation_z(angle);
            let mut ramp = RigidBody::new_box(Vec3::ZERO, Vec3::new(5.0, 0.25, 5.0), 0.0);
            ramp.orientation = tilt;
            engine.add_body(ramp);
            // Drop a small cube just above the tilted top face.
            let surface = tilt * Vec3::new(0.0, 0.25, 0.0);
            let n = tilt * Vec3::Y;
            let cube = engine.add_soft_body(SoftBody::soft_cube(
                surface + n * 0.6 - Vec3::new(0.25, 0.0, 0.25),
                HALF,
                1.0,
                0.0,
                0.0,
            ));
            engine.get_soft_body_mut(cube).expect("cube mut").damping = 4.0;
            let com_x = |engine: &XpbdEngine| {
                let body = engine.get_soft_body(cube).expect("cube survives");
                body.particles.iter().map(|p| p.position.x).sum::<f32>()
                    / body.particles.len() as f32
            };
            for _ in 0..60 {
                engine.step(1.0 / 60.0);
            }
            let x0 = com_x(&engine);
            for _ in 0..240 {
                engine.step(1.0 / 60.0);
            }
            com_x(&engine) - x0
        }
        let stuck = slide(1.0);
        let loose = slide(0.0);
        assert!(
            stuck.abs() < 0.2,
            "high friction must hold the cube, slid {stuck}"
        );
        // Downhill is −X for a +12° Z-rotation (the +X edge rises).
        assert!(
            loose < -0.6,
            "zero friction must let the cube slide downhill, slid {loose}"
        );
    }

    /// Manifold stability through warm-started normal impulses: a rigid
    /// slab dropped onto a settled soft cube rests on it instead of
    /// sinking through, and the support cache stays live. (The slab spans
    /// the whole cube: a narrow box would fit between the corner
    /// particles — the D1 cube is a wireframe, not a solid.)
    #[test]
    fn soft_warm_start_keeps_stack_stable() {
        use crate::soft::SoftBody;

        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -5.0, 0.0),
            Vec3::splat(5.0),
            0.0,
        ));
        let cube = engine.add_soft_body(SoftBody::soft_cube(
            Vec3::new(-HALF, 2.0, -HALF),
            1.0,
            1.0,
            0.0,
            0.0,
        ));
        engine.get_soft_body_mut(cube).expect("cube mut").damping = 4.0;
        for _ in 0..240 {
            engine.step(1.0 / 60.0);
        }
        let top = engine
            .get_soft_body(cube)
            .expect("cube survives")
            .particles
            .iter()
            .map(|p| p.position.y)
            .fold(f32::NEG_INFINITY, f32::max);
        let slab = engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, top + 0.6, 0.0),
            Vec3::new(0.6, 0.2, 0.6),
            2.0,
        ));
        for _ in 0..240 {
            engine.step(1.0 / 60.0);
        }
        let b = engine.get_body(slab).expect("slab survives");
        assert!(
            b.position.y - 0.2 > top - HALF,
            "slab must rest on the cube, center at {} (cube top was {top})",
            b.position.y
        );
        assert!(
            b.velocity.length() < 0.4,
            "stack must calm, slab speed {}",
            b.velocity.length()
        );
        let body = engine.get_soft_body(cube).expect("cube survives");
        assert!(
            body.particles.iter().all(|p| p.position.is_finite()),
            "no NaN in loaded cube"
        );
        assert!(
            !engine.soft_lambda_cache.is_empty(),
            "warm-start cache must hold the resting support"
        );
    }

    /// Particle CCD: a 300 m/s particle (0.25 m per substep — past both
    /// faces of the 0.04 m wall in one jump) stops at the thin wall
    /// instead of tunneling through it.
    #[test]
    fn fast_particle_ccd_stops_at_thin_wall() {
        use crate::soft::SoftBody;

        let mut engine = XpbdEngine::new(Vec3::ZERO);
        engine.add_body(RigidBody::new_box(
            Vec3::ZERO,
            Vec3::new(0.02, 1.0, 1.0),
            0.0,
        ));
        let mut chain = SoftBody::chain(Vec3::new(1.0, 0.0, 0.0), Vec3::NEG_X, 1, 0.2, 1.0, 0.0);
        // Single-particle body: unpin the builder's anchor (soft_self test idiom).
        chain.particles[0].inv_mass = 1.0;
        let h = engine.add_soft_body(chain);
        engine.get_soft_body_mut(h).expect("particle mut").particles[0].velocity =
            Vec3::new(-300.0, 0.0, 0.0);
        for _ in 0..60 {
            engine.step(1.0 / 60.0);
        }
        let p = &engine
            .get_soft_body(h)
            .expect("particle survives")
            .particles[0];
        assert!(p.position.is_finite(), "no NaN in swept particle");
        // Wall faces at ±0.02, particle radius 0.04: rest at ≈ 0.06.
        assert!(
            p.position.x > -0.2 && p.position.x < HALF,
            "particle must stop at the wall, got x = {}",
            p.position.x
        );
    }

    /// Sleep parity (0.15 m/s + 0.5 s): settled rigid and soft bodies
    /// freeze with zeroed velocities, and a fast impact wakes the soft
    /// body back up.
    #[test]
    fn settled_bodies_sleep_and_impacts_wake() {
        use crate::soft::SoftBody;

        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -5.0, 0.0),
            Vec3::splat(5.0),
            0.0,
        ));
        let bx = engine.add_body(RigidBody::new_box(
            Vec3::new(3.0, 2.0, 0.0),
            Vec3::splat(HALF),
            1.0,
        ));
        let cube = engine.add_soft_body(SoftBody::soft_cube(
            Vec3::new(-HALF, 2.0, -HALF),
            1.0,
            1.0,
            0.0,
            0.0,
        ));
        engine.get_soft_body_mut(cube).expect("cube mut").damping = 4.0;
        for _ in 0..300 {
            engine.step(1.0 / 60.0);
        }
        assert!(engine.is_asleep(bx), "settled rigid box must sleep");
        assert!(engine.is_soft_asleep(cube), "settled soft cube must sleep");
        assert!(
            engine.get_body(bx).expect("box").velocity.length() < 1e-6,
            "sleep zeroes rigid velocity"
        );
        // Wide slab dropped flat onto the sleeping cube (spans the whole
        // wireframe, so the impact lands with full normal speed): it must
        // wake the cube back up.
        let slab = engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.6, 0.2, 0.6),
            2.0,
        ));
        engine.get_body_mut(slab).expect("slab mut").velocity = Vec3::new(0.0, -10.0, 0.0);
        for _ in 0..30 {
            engine.step(1.0 / 60.0);
        }
        assert!(
            !engine.is_soft_asleep(cube),
            "impact must wake the soft body"
        );
    }

    /// Soft↔rigid begin/end events: touchdown emits one Begin per pair,
    /// resting emits no flicker, teleporting away emits the End.
    #[test]
    fn soft_contact_begin_end_events() {
        use crate::soft::SoftBody;
        use crate::trigger::ContactEventKind;

        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let floor = engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -5.0, 0.0),
            Vec3::splat(5.0),
            0.0,
        ));
        let cube = engine.add_soft_body(SoftBody::soft_cube(
            Vec3::new(-HALF, 2.0, -HALF),
            1.0,
            1.0,
            0.0,
            0.0,
        ));
        engine.get_soft_body_mut(cube).expect("cube mut").damping = 4.0;
        let mut saw_begin = false;
        for _ in 0..240 {
            engine.step(1.0 / 60.0);
            for e in engine.drain_soft_contact_events() {
                if e.soft == cube && e.body == floor && e.kind == ContactEventKind::Begin {
                    saw_begin = true;
                }
            }
            if saw_begin {
                break;
            }
        }
        assert!(saw_begin, "touchdown must emit Begin");
        for _ in 0..30 {
            engine.step(1.0 / 60.0);
        }
        let rest = engine.drain_soft_contact_events();
        assert!(
            rest.is_empty(),
            "resting contact must not flicker, got {rest:?}"
        );
        {
            let body = engine.get_soft_body_mut(cube).expect("cube mut");
            for p in &mut body.particles {
                p.position += Vec3::new(0.0, 5.0, 0.0);
            }
        }
        engine.wake_soft_body(cube);
        engine.step(1.0 / 60.0);
        let ends = engine.drain_soft_contact_events();
        assert!(
            ends.iter()
                .any(|e| e.soft == cube && e.body == floor && e.kind == ContactEventKind::End),
            "teleport away must emit End, got {ends:?}"
        );
    }
}
