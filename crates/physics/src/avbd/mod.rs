//! Augmented Vertex Block Descent (Giles et al, SIGGRAPH'25) rigid-body engine
//! (AVBD): a second [`crate::engine::PhysicsEngine`]
//! implementation (M1, Genesis-style engine-level modularity).
//!
//! Ports the Augmented Vertex Block Descent update rules from
//! savant117/avbd-demo3d (MIT, Giles et al, SIGGRAPH'25): per-body 6x6 SPD
//! assembly (linear + angular + cross blocks) solved with dense LDL, plus a
//! per-constraint dual (lambda/penalty) update. Contacts use the Taylor
//! constraint `C = C0*(1-alpha) + J*dq` with push-only normal clamp and a
//! joint friction-cone clamp; joints are infinite-stiffness equalities of
//! the form `C = live - alpha*C0`.
//!
//! Validated by spike `spikes/001-avbd-stack` (4-box stack stands 300 steps,
//! rest heights +-6mm, settle on step 11). Spike lessons structural here:
//! dual rules are ported, never guessed; orientation uses the official exact
//! quaternion operators (naive xyz-add + linear-diff destabilizes levers);
//! joints carry the official geometric-stiffness term (without it ball
//! joints spin up through long levers).
//!
//! # M1/M2 scope (closed) + M3 at the Engine level
//!
//! - Shapes: all pairs query through `crate::distance::shape_distance`;
//!   face-stable multi-point manifolds are built for box-involved pairs
//!   (witness is fallback only — it slides and rocks stacks); everything
//!   else solves on the witness pair (functional but tippy).
//! - Contact frames are persistent per pair (witness normals flip sign at
//!   first touch; a flipped frame turns compressive lambda tensile).
//! - Anchors follow the official merge: frozen while static grip holds
//!   (`stick`), refreshed to live geometry on slide — but never while
//!   penetrating (support first) and never for resting pairs (empty shells
//!   beyond `GEN_MARGIN`, no stale push, no lambda burn).
//! - Friction is anisotropic (ODE `fdir1`/`mu`/`mu2` parity: per-axis
//!   Coulomb coefficients on a body-A-wins frame, elliptical cone
//!   projection) plus rolling/torsion resistance (MuJoCo triple: pure
//!   couples capped by mu × normal force). Zero coefficients skip the rows
//!   entirely, so default scenes pay nothing.
//! - Joints: [`crate::joint::JointKind::Ball`], free
//!   [`crate::joint::JointKind::Revolute`] (hinge axis via two angular rows)
//!   with travel limits (one-sided accumulator rows, Box2D order) and
//!   deadbeat impulse motors (exact clamped velocity step, no dual state),
//!   [`crate::joint::JointKind::Prismatic`] (2 perp + 2 angular rows, same
//!   limit/motor drive along the slide axis),
//!   [`crate::joint::JointKind::Fixed`] (weld: ball rows + locked assembly
//!   rotation), [`crate::joint::JointKind::Distance`] (single rod row),
//!   [`crate::joint::JointKind::Wheel`] (2 perp rows + 2 angular rows about
//!   the axle (spin about it is free) + a position-level suspension spring
//!   from Box2D frequency/damping (fixed penalty, sag holds the load;
//!   zero frequency degrades to a rigid slide row); deadbeat spin motor
//!   about the axle), [`crate::joint::JointKind::Gear`] (position-level
//!   equality `coord_a + ratio*coord_b = const` over hinge/slide
//!   coordinates, closer to Box2D than the builtin velocity-only pass)
//!   and [`crate::joint::JointKind::SixDof`] (per-axis free/locked/limited
//!   rows in body A's live assembly frame, mirroring the builtin position
//!   pass; limited forces in dual-owned per-axis slots).
//!   Sleep (M2, closed): quiet dynamics freeze per body (builtin 0.15 m/s
//!   + 0.5 s parity) and wake on fresh pairs/joint edits/velocity kicks;
//!   wake propagates through live pairs (1 cm backstop), so no per-engine
//!   island pass is needed here.
//!   Substeps (M2, closed): fixed-step accumulator — the tuned core always
//!   advances in exact 1/60 s increments (120 Hz hosts alternate sim/skip,
//!   hitches replay whole steps, debt past 4 steps clamps to slow motion).
//!   TOI (M2, closed at builtin linear parity — angular sweep stays
//!   discrete: a body whose step displacement exceeds half its smallest
//!   dimension sweeps `cast_shape` and clamps to the first hit; jointed
//!   partners are excluded so joint swings never self-clamp).
//!   No-collide for pin joints (builtin `joint_pairs` parity, narrowed to
//!   Ball/Revolute/Prismatic/Distance/Wheel): jointed bodies skip contact
//!   discovery — a hinge pin passes through its mount, and contact
//!   friction there is a phantom brake on the joint. Fixed/SixDof are
//!   excluded (weld-like assemblies whose tests bury boxes by
//!   construction; there the contact is structural).
//!   Fracture (M3, at the [`crate::Engine`] level, not in this engine):
//!   the orchestrator's contact-event pass splits bodies on hard hits,
//!   uniform for both solvers; reports wait in
//!   [`crate::Engine::drain_fracture_events`] (see `tests/fracture.rs`).
//!   Penalty joints show ~10cm dynamic stretch at swing bottom; motors droop
//!   under sustained load (velocity servo); ball position-servo rows damp
//!   fast orbital motion through long levers (all bounded, tested).
//! - Sphere piles settle on floors/boxes (rolling friction helps); the tall
//!   sphere tower is closed by measurement
//!   (`avbd_tall_tower_drop_settle_holds`: 6-sphere drop-settle holds,
//!   rolling multi-point contact was not needed).
//! - No islands inside this engine (deliberate): one implicit step of 10
//!   iterations; island routing lives at the orchestrator as
//!   [`crate::RoutingKind::Islands`] (`split.rs`). Broadphase stays an O(n2)
//!   bounding-sphere prefilter inside `AvbdEngine` (honest residual).
//!   Joint swings and spinning bodies rely on the discrete phase (angular
//!   sweep covered by measurement up to 60 rad/s, same as the builtin's
//!   linear-cast limit).
//!   The pre-touch pair band (points form up to `GEN_MARGIN` before contact)
//!   is load-bearing for fast impacts: their frozen-anchor C diverges on
//!   touchdown and the penalty ramp turns it into a projection catch.
//!   Gating point creation on touch was tried and reverted (fast bodies
//!   skip the window and tunnel); gating row force on live gap was tried
//!   and reverted (broke 4 settled scenes). Near-pass phantom force inside
//!   the band stays an M2 item (needs pressure-gated friction rows).
//! - The prismatic limit chase (dual/primal accumulator race, +5.7m
//!   attractor) is knife-edge sensitive: hot-loop code motion alone flips
//!   the slider outcome. Don't churn the limit rows cosmetically; M2 must
//!   robustify (damp) the chase itself.
//! - Orientation integrates with the exact exponential map
//!   (`exp(v/2)*q`, not chord Euler — the chord loses ~|w·dt|²/2 of angle
//!   per step, i.e. 0.25%/step spin decay at 10 rad/s) and differentiates
//!   with exact angle-axis recovery (`2*asin(|xyz|)`); per-step
//!   renormalization prevents long-term drift (the demo never
//!   renormalizes).

mod assembly;
mod joints_ext;
mod rows;
mod sleep;

#[cfg(test)]
mod contact_tests;
#[cfg(test)]
mod joint_tests;

pub use assembly::solve_6x6;

use self::rows::{
    inverse_symmetric, mat3_vec, pair_allowed, quat_diff_vec, quat_integrate, shape_min_dimension,
    world_inertia,
};

use glam::{Quat, Vec3};
use std::collections::BTreeSet;
use std::f32::consts::PI;

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::distance::{ShapeRef, cast_shape, shape_distance};
use crate::engine::{PhysicsEngine, raycast_shape_hit};
use crate::errors::{JointError, QueryError};
use crate::joint::{AxisConfig, JointHandle, JointKind};
use crate::math::{Ray, RaycastHit, tangent_basis};
use crate::migration::{JointReference, JointSnapshot};
use crate::shape::Shape;
use crate::trigger::{
    CONTACT_BEGIN_SLOP, CONTACT_HIT_THRESHOLD, ContactEvent, ContactEventKind, TriggerEvent,
    TriggerEventKind,
};
/// Solver iterations per step (official default).
const ITERS: usize = 10;
/// Stabilization: only `(1-alpha)` of the step-start violation enters `C`.
const ALPHA: f32 = 0.99;
/// Warmstart decay for duals and penalties (Eq. 19).
const GAMMA: f32 = 0.999;
/// Additive penalty ramp scale (Eq. 16, official `betaLin` for linear
/// constraints AND contact rows).
const BETA: f32 = 10000.0;
/// Additive penalty ramp scale for angular rows (official `betaAng`).
/// The old code used `BETA` for the hinge/limit angular rows too: at
/// 100x the official rate an angular penalty crosses the explicit
/// stability bound (pen*dt^2/I) in a handful of holding steps and the
/// row diverges — the prismatic slider (whose bob hangs 1m off-axis,
/// i.e. whose limit row IS an angular fight through the lever) held
/// ~200 steps, then snapped to +5.7m. Contacts keep `BETA` regardless
/// of axis (official `betaLin` covers all contact rows).
const BETA_ANG: f32 = 100.0;
/// Speculative contact margin folded into the normal `C0`.
const MARGIN: f32 = 0.01;
/// Penalty clamp range (official `PENALTY_MIN/MAX`).
const PENALTY_MIN: f32 = 1.0;
const PENALTY_MAX: f32 = 1.0e10;
/// Penalty of a freshly created contact row (official: clamped up from 0).
const PENALTY_INIT: f32 = 1.0;
/// Penalty of a freshly created joint row (official: constructed at 0,
/// Eq. 19 clamps up to `PENALTY_MIN`; stiffness comes from lambda first).
const JOINT_PENALTY_INIT: f32 = 1.0;
/// Pair creation distance: witness gap below this opens a contact pair.
const GEN_MARGIN: f32 = 0.05;
/// Box-corner manifold expansion around the witness plane.
const EXPAND_SLOP: f32 = 0.005;
/// Contact-point dedup distance when expanding manifolds.
const POINT_MATCH_DIST: f32 = 0.03;
/// Dual-memory match distance for persistent points across steps.
const LAMBDA_MATCH_DIST: f32 = 0.05;
/// Constraint satisfaction tolerance: a row with `|C|` below this carries
/// no fresh violation — no dual accumulation, no penalty ramp. Positions
/// are O(1) in f32 (eps ~1.2e-7), so anything smaller is rounding dust.
/// A dust violation with a live warm force still stamps (see `row_live`):
/// only dust force on dust violation is skipped, so idle rows cannot
/// ratchet lambda or penalty into a slow runaway.
const C_EPS: f32 = 1e-7;

/// Static-friction position threshold (official `STICK_THRESH`): a point
/// whose tangential violation is below this kept its grip last step.
const STICK_THRESH: f32 = 0.00001;
/// Sleep thresholds (builtin parity: islands sleep below 0.15 m/s and
/// 0.15 rad/s): a dynamic body slower than both for [`SLEEP_TIME`] seconds
/// freezes — skipped by the sweep and BDF1, static for the solver — and
/// wakes on new contact pairs, joint add/remove, or an externally set
/// velocity. No islands: AVBD wakes per body (a settled stack sleeps as
/// individual frozen supports; a bulldozer push through an *existing* pair
/// does not propagate wake — documented M3 gap).
const SLEEP_LIN: f32 = 0.15;
const SLEEP_ANG: f32 = 0.15;
const SLEEP_TIME: f32 = 0.5;
/// Gated-wake thresholds (builtin `wake_on_impact` + penetration-wake
/// parity): a NEW pair wakes its sleepers only on 0.5 m/s approach or a
/// fresh overlap deeper than 1 cm.
const WAKE_IMPACT_SPEED: f32 = 0.5;
const WAKE_PENETRATION: f32 = 0.01;

/// Angular travel slop for revolute limits (official `ANGULAR_SLOP`).
const LIMIT_SLOP_ANG: f32 = 0.005;
/// Slide travel slop for prismatic limits (official `LINEAR_SLOP`).
const LIMIT_SLOP_LIN: f32 = 0.002;
/// Cap on contact points per pair (official manifold holds 8).
const MAX_POINTS: usize = 8;
/// Cap on limit-row penalties (explicit-servo stability): a limit holds a
/// persistent bias (C = -F/pen, never converges to zero like contacts), so
/// an uncapped BETA ramp crosses the explicit stability bound pen*dt^2/m
/// (~1.4e4 for m=1) after ~150 holding steps and diverges into a growing
/// limit cycle (a slider held 150 steps, then fell through and snapped to
/// +5.7m). 1e4 holds 10N at 1mm penetration (inside the 2mm slop band, so
/// the row sleeps) with stability factor 2.8. Heavier loads sag deeper but
/// stay stable. Contacts need no cap (their C converges geometrically
/// within the step's iterations — a stable race).
const LIM_PEN_MAX: f32 = 1e4;

/// Fixed penalty of rolling/torsion rows (no dual ramp — the torque cap,
/// not the stiffness, shapes the resistance, mirroring the official
/// impulse clamp).
const ROLL_PEN: f32 = 100.0;
/// A single contact point: material anchors on both bodies plus the dual
/// state. Anchors are body-local, fixed at detection (spike lesson).
/// Penalty is per-axis (official `penalty` float3): a shared penalty lets
/// the loaded axis stiffen the idle axes into lever instability.
#[derive(Clone, Debug)]
struct AvbdPoint {
    ra: Vec3,
    rb: Vec3,
    lam: [f32; 3],
    pen: [f32; 3],
    /// Static grip held last step (official `stick`): anchors stay frozen.
    /// Rolling/sliding points refresh anchors to live geometry (otherwise a
    /// spinning body's frozen lever orbits and pumps energy).
    stuck: bool,
    /// Rolling/torsion dual memory `[roll_t1, roll_t2, spin_n]` (MuJoCo
    /// triple parity; fixed penalty, torque-capped, no ramp).
    roll_lam: [f32; 3],
}

/// Contact pair: two bodies, one normal frame, up to [`MAX_POINTS`] points.
#[derive(Clone, Debug)]
struct AvbdPair {
    a: usize,
    b: usize,
    n: Vec3,
    /// Coulomb coefficients along the frame tangents `[t1, t2]`
    /// (isotropic pairs carry `[mu, mu]`).
    mu: [f32; 2],
    /// Pre-step witness gap (`shape_distance`, signed: negative =
    /// penetrating). Drives the separated-damper rule: open pairs react
    /// with velocity only, touching pairs with the full Taylor spring.
    gap: f32,
    points: Vec<AvbdPoint>,
}

/// Joint row model selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AvbdJointKind {
    Ball,
    Revolute,
    Prismatic,
    Fixed,
    Distance,
    Wheel,
    Gear,
    SixDof,
}

/// One gear side: referenced bodies, world axis, anchor levers and the
/// live coordinate. `kind` selects torque (revolute) vs force
/// (prismatic) gradients (mirrors the builtin `GearSideData`, minus the
/// velocity terms AVBD doesn't need — BDF1 carries velocity).
struct GearSide {
    a: usize,
    b: usize,
    kind: crate::flags::CoordKind,
    axis: Vec3,
    ra: Vec3,
    rb: Vec3,
    coord: f32,
}

/// Assembly-frame axes for SixDof rows (body A's X/Y/Z, world-projected
/// per solve like the builtin `FRAME`).
const SIXDOF_FRAME: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

/// Equality-constraint joint (ball anchor rows + optional hinge axis rows).
/// Per-axis penalties (official float3s): sharing one penalty across rows
/// lets the loaded row drive idle rows (via long levers) unstable.
#[derive(Clone, Debug)]
struct AvbdJoint {
    a: usize,
    b: usize,
    la: Vec3,
    lb: Vec3,
    ax_a: Vec3,
    ax_b: Vec3,
    /// Wheel spin axle in each body's local frame (orthogonalized against
    /// the suspension at creation); unused by other kinds.
    bx_a: Vec3,
    bx_b: Vec3,
    /// Wheel spring `[frequency_hz, damping_ratio]` (Box2D semantics);
    /// unused by other kinds.
    susp: [f32; 2],
    kind: AvbdJointKind,
    /// Original joint spec (mirrors the builtin stored `kind`): lets the
    /// `Engine` orchestrator migrate joints across solvers without loss.
    spec: JointKind,
    /// Travel reference: revolute reference twist (rad), prismatic reference
    /// length (m), distance rest length (m); unused otherwise.
    ref_val: f32,
    /// Fixed-joint assembly relative rotation (`qa^-1 * qb` at creation).
    q_ref: Quat,
    /// Travel window `[min, max]` (revolute rad / prismatic m); `None` = free.
    lim: Option<[f32; 2]>,
    /// Servo drive `[target_speed, max_force_or_torque]`; `None` = unpowered.
    mot: Option<[f32; 2]>,
    /// One-sided limit accumulator (mirrors the official `acc_limit`).
    acc_lim: f32,
    /// Dual-side limit state (official joint `updateDual` discipline):
    /// the prismatic limit force lives here (`lambda = F`: recomputed from
    /// current positions, stored, warmstarted by the primal). The revolute
    /// limit keeps the legacy `acc_lim` accumulator (verified by the hinge
    /// test; unifying both rows onto this slot regressed it 0.50 → 0.63 —
    /// the hinge window is entered ballistically, not held under load, so
    /// the accumulator's faster bite wins there).
    lim_dual: f32,
    /// Gear references: indices into `joints` of the two coordinated
    /// (revolute/prismatic) joints; `ref_val` holds the assembly constant.
    gb: [usize; 2],
    /// Gear transmission ratio (`coord_a + ratio * coord_b = const`).
    gratio: f32,
    /// Continuous gear-side coordinates `(prev_raw, prev_cont)` per
    /// referenced joint: hinge twists wrap to `[-PI, PI]`, so a multi-turn
    /// drive would teleport the ratio residual by `2*PI` (measured:
    /// violent snap when arm A crosses PI, chaos-dependent recovery).
    /// Refreshed once per step in `update_gear_mem`; `gear_sides` unwraps
    /// the live raw coordinate against it (per-step motion is always a
    /// fraction of PI, so the unwrap stays correct all step).
    gear_mem: Option<([f32; 2], [f32; 2])>,
    /// SixDof per-axis config in body A's assembly frame (linear X/Y/Z).
    six_lin: [AxisConfig; 3],
    /// SixDof per-axis config in body A's assembly frame (angular X/Y/Z).
    six_ang: [AxisConfig; 3],
    /// SixDof/Fixed assembly anchor separation in A's frame
    /// (`qa^-1 * ((pb+rb)-(pa+ra))` at creation); locked linear axes
    /// measure drift relative to this.
    dref: Vec3,
    /// SixDof one-sided limit forces, dual-owned (slots 0..3 linear X/Y/Z,
    /// 3..6 angular): warmstarted by the primal, committed by the dual —
    /// same discipline as `lim_dual`, one slot per limited axis.
    sacc: [f32; 6],
    lam_l: [f32; 3],
    lam_a: [f32; 3],
    pen_l: [f32; 3],
    pen_a: [f32; 3],
}

/// Read-only discovery bundle for one candidate pair: everything the
/// sequential merge needs, nothing it writes. Discovery is a pure
/// function of the step-top state, so the thread schedule cannot leak
/// into the pairs (the M2 Strong-Confluence claim for AVBD).
struct Discovered {
    ia: usize,
    ib: usize,
    trigger: bool,
    exists: bool,
    normal: Vec3,
    mu: [f32; 2],
    signed: f32,
    fresh: Vec<(Vec3, Vec3)>,
    wake_vote: bool,
}
/// AVBD rigid-body engine; see the module docs for formulation and scope.
///
/// Bodies are stored in handle order (`swap_remove` on removal, exactly like
/// [`crate::engine::SequentialImpulseEngine`]); contacts and joints remap the
/// same way. Single-threaded, hence deterministic by construction.
#[derive(Clone, Debug)]
pub struct AvbdEngine {
    gravity: Vec3,
    bodies: Vec<RigidBody>,
    pairs: Vec<AvbdPair>,
    joints: Vec<AvbdJoint>,
    prev_touch: BTreeSet<(usize, usize)>,
    prev_trigger: BTreeSet<(usize, usize)>,
    contact_events: Vec<ContactEvent>,
    trigger_events: Vec<TriggerEvent>,
    pos0: Vec<Vec3>,
    rot0: Vec<Quat>,
    inertial: Vec<Vec3>,
    inertial_rot: Vec<Quat>,
    pre_vel: Vec<Vec3>,
    /// Per-body sleep timers (seconds below thresholds) and frozen flags.
    sleep_timer: Vec<f32>,
    asleep: Vec<bool>,
    /// Post-step poses of the previous substep (driver baseline): the
    /// engine never integrates kinematic bodies, so any change here is a
    /// driver teleport. `step_inner` rewinds kinematic `pos0`/`rot0` to
    /// these so contact/joint rows see driver motion as within-step `dq`
    /// at full strength.
    prev_pos: Vec<Vec3>,
    prev_rot: Vec<Quat>,
    /// Fixed-step accumulator (seconds of host time awaiting simulation):
    /// the core always advances in exact `DT_STEP` increments, so 120 Hz
    /// hosts alternate sim/skip and hitch frames catch up — wall speed is
    /// exact on average without touching the tuned single-step dynamics.
    time_debt: f32,
    /// Opt-in GPU AVBD dispatch (rung 1, `gpu` feature): attached via
    /// [`AvbdEngine::set_gpu_avbd`], default off. Device execution stays
    /// gated on the stub's `runs_on_device` (false until a real dispatch
    /// lands), so an attached stub transparently takes the CPU fallback.
    #[cfg(feature = "gpu")]
    gpu_avbd: Option<crate::gpu::GpuAvbdStub>,
    /// Completed steps that fell back to CPU while a GPU dispatch was
    /// attached (observability for the opt-in flag; never a physics input).
    #[cfg(feature = "gpu")]
    gpu_fallback_steps: u64,
}

impl AvbdEngine {
    /// Bodies in handle order, cloned for solver migration (`Engine`
    /// re-registers them 1:1, so handles stay valid across the switch).
    pub(crate) fn bodies_snapshot(&self) -> Vec<RigidBody> {
        // Sleepers migrate awake with their mass model restored (warm-start
        // state never migrates anyway, and a zeroed inverse mass must not
        // leak into the new engine).
        self.bodies
            .iter()
            .enumerate()
            .map(|(h, b)| {
                let mut c = b.clone();
                // `asleep` may lag `bodies` (it only grows on sleep
                // transitions) — a missing entry means awake.
                if self.asleep.get(h).copied().unwrap_or(false) && c.body_type == BodyType::Dynamic
                {
                    c.restore_sleep_triple();
                }
                c
            })
            .collect()
    }

    /// Driver poses at the previous completed step (new bodies use their pose).
    pub(crate) fn body_baselines(&self) -> Vec<crate::broadphase::PrevPose> {
        self.bodies
            .iter()
            .enumerate()
            .map(|(h, b)| crate::broadphase::PrevPose {
                pos: self.prev_pos.get(h).copied().unwrap_or(b.position),
                rot: self.prev_rot.get(h).copied().unwrap_or(b.orientation),
            })
            .collect()
    }

    /// Restore a driver's within-step motion baseline after rebuilding.
    pub(crate) fn restore_body_baseline(
        &mut self,
        h: crate::body::BodyHandle,
        pose: crate::broadphase::PrevPose,
    ) {
        self.ensure_scratch();
        let h = h.index();
        if h < self.bodies.len() {
            self.prev_pos[h] = pose.pos;
            self.prev_rot[h] = pose.rot;
        }
    }

    /// Completed-step event baseline for transparent solver migration.
    pub(crate) fn event_state(&self) -> crate::migration::EventState {
        crate::migration::EventState {
            contacts: self
                .prev_touch
                .iter()
                .copied()
                .map(|(a, b)| {
                    (
                        crate::body::BodyHandle::from(a),
                        crate::body::BodyHandle::from(b),
                    )
                })
                .collect(),
            triggers: self
                .prev_trigger
                .iter()
                .copied()
                .map(|(a, b)| {
                    (
                        crate::body::BodyHandle::from(a),
                        crate::body::BodyHandle::from(b),
                    )
                })
                .collect(),
        }
    }

    /// Seed a rebuilt solver without manufacturing a new contact/trigger begin.
    pub(crate) fn restore_event_state(&mut self, state: crate::migration::EventState) {
        self.prev_touch = state
            .contacts
            .into_iter()
            .map(|(a, b)| (a.index(), b.index()))
            .collect();
        self.prev_trigger = state
            .triggers
            .into_iter()
            .map(|(a, b)| (a.index(), b.index()))
            .collect();
    }

    /// Number of live joints in dense handle order.
    pub(crate) fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// Physical joint state in handle order; numerical warm starts stay local.
    pub(crate) fn joint_snapshots(&self) -> Vec<JointSnapshot> {
        self.joints
            .iter()
            .map(|j| {
                let mut reference = JointReference {
                    rotation: j.q_ref,
                    anchor_delta: j.dref,
                    ..JointReference::default()
                };
                match j.kind {
                    AvbdJointKind::Revolute => {
                        reference.angle = crate::invariants::Radians(j.ref_val);
                    }
                    AvbdJointKind::Prismatic | AvbdJointKind::Wheel => {
                        reference.length = crate::invariants::Meters(j.ref_val);
                    }
                    AvbdJointKind::Distance | AvbdJointKind::Gear => {
                        reference.distance = crate::invariants::Meters(j.ref_val);
                    }
                    _ => {}
                }
                if j.kind == AvbdJointKind::Gear
                    && let Some((a, b)) = self.gear_sides(j)
                {
                    // Rebase continuous gear phase into the new engine's raw
                    // chart without changing the current constraint error.
                    let raw = |r: usize| {
                        let side = &self.joints[r];
                        Self::joint_coordinate(side, &self.bodies[side.a], &self.bodies[side.b])
                            .unwrap_or(0.0)
                    };
                    reference.distance.0 -=
                        (a.coord - raw(j.gb[0])) + j.gratio * (b.coord - raw(j.gb[1]));
                }
                JointSnapshot {
                    a: crate::body::BodyHandle::from(j.a),
                    b: crate::body::BodyHandle::from(j.b),
                    spec: j.spec,
                    reference,
                }
            })
            .collect()
    }

    /// Restore the assembly pose after creating a migrated joint.
    pub(crate) fn restore_joint_reference(&mut self, h: JointHandle, r: JointReference) {
        let Some(j) = self.joints.get_mut(h.index()) else {
            return;
        };
        j.q_ref = r.rotation;
        j.dref = r.anchor_delta;
        j.ref_val = match j.kind {
            AvbdJointKind::Revolute => r.angle.0,
            AvbdJointKind::Prismatic | AvbdJointKind::Wheel => r.length.0,
            AvbdJointKind::Distance | AvbdJointKind::Gear => r.distance.0,
            _ => j.ref_val,
        };
    }

    /// Dense removal also removes dependent gears before remapping survivors.
    fn retain_joints(&mut self, removed: Vec<bool>) {
        let kinds: Vec<_> = self.joints.iter().map(|j| j.spec).collect();
        let remap = crate::migration::joint_remap(&kinds, removed);
        let mut old = 0;
        self.joints.retain_mut(|j| {
            let keep = remap[old].is_some();
            old += 1;
            if keep {
                crate::migration::remap_gear(&mut j.spec, &remap);
                if let JointKind::Gear {
                    joint_a, joint_b, ..
                } = j.spec
                {
                    j.gb = [joint_a.index(), joint_b.index()];
                }
            }
            keep
        });
    }

    /// Dynamic bodies currently frozen by sleep (observability for tests
    /// and the 16.7 ms budget: sleepers skip the sweep and BDF1).
    pub fn sleeping_count(&self) -> usize {
        self.asleep.iter().filter(|&&s| s).count()
    }

    /// Attach (`Some`) or detach (`None`) the opt-in GPU AVBD dispatch
    /// (rung 1, `gpu` feature). Default off. An attached stub still runs
    /// the CPU fallback until its `runs_on_device` turns true (no adapter
    /// path yet), so attaching never changes the trajectory — only the
    /// [`AvbdEngine::gpu_fallback_steps`] counter moves. The rung-2 device
    /// solver ([`crate::gpu::WgpuAvbdSolver`]) is not attachable here yet:
    /// it solves staged linear systems on the device, while engine stepping
    /// needs host discovery staging rows into that solve (rung 3).
    #[cfg(feature = "gpu")]
    pub fn set_gpu_avbd(&mut self, stub: Option<crate::gpu::GpuAvbdStub>) {
        self.gpu_avbd = stub;
    }

    /// Whether a GPU AVBD dispatch is attached (opt-in flag state, not
    /// device execution — see `runs_on_device` on the stub).
    #[cfg(feature = "gpu")]
    pub fn gpu_avbd_enabled(&self) -> bool {
        self.gpu_avbd.is_some()
    }

    /// Completed steps that fell back to CPU while a GPU dispatch was
    /// attached. Zero when detached; observability only, never an input.
    #[cfg(feature = "gpu")]
    pub fn gpu_fallback_steps(&self) -> u64 {
        self.gpu_fallback_steps
    }

    /// Freeze body `h` (builtin sleep semantics: static for the solver —
    /// zero inverse mass/inertia so contacts treat it as immovable and no
    /// invisible velocity can accumulate and detonate on wake).
    fn sleep_body(&mut self, h: usize) {
        if h >= self.asleep.len() || self.bodies[h].body_type != BodyType::Dynamic {
            return;
        }
        self.bodies[h].sleep_staticify();
        self.asleep[h] = true;
    }

    /// Wake body `h`, restoring its mass model (mirrors the builtin
    /// `wake_island` restore: `1/mass` + shape inertia). No-op for
    /// non-dynamics and awake bodies.
    fn wake_body(&mut self, h: usize) {
        // No-op before the first step (scratch not sized yet: everything is
        // awake with zeroed timers by construction) and for awake bodies.
        if h >= self.asleep.len() || !self.asleep[h] {
            return;
        }
        self.asleep[h] = false;
        self.sleep_timer[h] = 0.0;
        self.bodies[h].wake_restore();
    }

    /// One fixed `DT_STEP` advance. Hit/contact events emit only on the
    /// host step's last substep (earlier ones only advance state).
    fn step_inner(&mut self, emit: bool) {
        self.ensure_scratch();
        self.wake_joint_motion();
        self.update_gear_mem();
        let n = self.bodies.len();
        let event_start = self.contact_events.len();
        let mut ccd_hits = Vec::new();
        // Consume torques into angular velocity (cleared each step, like the
        // builtin engine); gravity folds into the inertial position below.
        for h in 0..n {
            if !self.solvable(h) {
                continue;
            }
            let b = &mut self.bodies[h];
            let iw = world_inertia(b.inertia, b.orientation);
            let iw_inv = inverse_symmetric(
                iw,
                Vec3::new(
                    if b.inertia.x > 0.0 { b.inertia.x } else { 0.0 },
                    if b.inertia.y > 0.0 { b.inertia.y } else { 0.0 },
                    if b.inertia.z > 0.0 { b.inertia.z } else { 0.0 },
                ),
            );
            let torque = std::mem::replace(&mut b.torque, Vec3::ZERO);
            b.angular_velocity += mat3_vec(iw_inv, torque) * DT_STEP;
        }
        self.motor_impulse();
        for (velocity, body) in self.pre_vel.iter_mut().zip(&self.bodies) {
            *velocity = body.velocity;
        }
        // Continuous clamp (builtin TOI parity, linear only): a dynamic body
        // whose step displacement exceeds half its smallest dimension sweeps
        // `cast_shape` along the motion and clamps to the first hit (+1mm),
        // with a one-shot bounce above the restitution threshold — otherwise
        // thin walls tunnel (10 m/s × 1/60 = 17 cm > 4 cm wall).
        for h in 0..n {
            if !self.solvable(h) || self.asleep[h] || self.bodies[h].is_trigger {
                continue;
            }
            let disp = self.bodies[h].velocity * DT_STEP;
            // TOI skip (builtin parity): the cast is only meaningful against
            // OTHER bodies — a jointed partner sweeping through the mover's
            // own swing (ball-pendulum anchors, hinge arcs) is not a wall.
            // Partners joined to the mover are excluded from the targets.
            if disp.length() <= 0.5 * shape_min_dimension(&self.bodies[h].shape) {
                continue;
            }
            let mover = ShapeRef {
                shape: &self.bodies[h].shape,
                pos: self.bodies[h].position,
                rot: self.bodies[h].orientation,
            };
            let hit = {
                let joined: Vec<usize> = self
                    .joints
                    .iter()
                    .filter(|j| j.kind != AvbdJointKind::Gear)
                    .filter_map(|j| {
                        if j.a == h {
                            Some(j.b)
                        } else if j.b == h {
                            Some(j.a)
                        } else {
                            None
                        }
                    })
                    .collect();
                let targets = (0..n)
                    .filter(|&o| {
                        o != h
                            && !self.bodies[o].is_trigger
                            && pair_allowed(&self.bodies[h], &self.bodies[o])
                            && !joined.contains(&o)
                    })
                    .map(|o| {
                        (
                            BodyHandle::from(o),
                            ShapeRef {
                                shape: &self.bodies[o].shape,
                                pos: self.bodies[o].position,
                                rot: self.bodies[o].orientation,
                            },
                        )
                    });
                cast_shape(mover, disp, targets)
            };
            if let Some(hit) = hit {
                let len = disp.length().max(1e-9);
                // `cast_shape` stops at first TOUCH (1 mm gap): stop AT the
                // reported pose — never step into it (the gap is the clean
                // touching contact the pair pass below owns) — and bound the
                // post-clamp travel to the generic creation margin, so a
                // far-wall report on a pass-through course still catches the
                // contact in the pair pass.
                let frac = (hit.t / len).clamp(0.0, 1.0);
                let e = self.bodies[h]
                    .restitution
                    .min(self.bodies[hit.handle.index()].restitution);
                let approach =
                    -(self.pre_vel[h] - self.pre_vel[hit.handle.index()]).dot(hit.normal);
                if approach > CONTACT_HIT_THRESHOLD {
                    let (a, b, normal) = if h < hit.handle.index() {
                        (BodyHandle::from(h), hit.handle, -hit.normal)
                    } else {
                        (hit.handle, BodyHandle::from(h), hit.normal)
                    };
                    ccd_hits.push(ContactEvent {
                        body_a: a,
                        body_b: b,
                        kind: ContactEventKind::Hit {
                            point: hit.point,
                            normal,
                            approach_speed: approach,
                        },
                    });
                }
                let b = &mut self.bodies[h];
                b.position += disp * frac;
                let vn = b.velocity.dot(hit.normal);
                if vn < 0.0 {
                    let bounce = if vn < -1.0 { 1.0 + e } else { 1.0 };
                    b.velocity -= hit.normal * (bounce * vn);
                }
                self.wake_body(hit.handle.index());
                // Any sleeper now touching this relocated body feels the
                // impact next pair pass (fresh-contact rule); its velocity
                // field is authoritative below, so freeze it now.
                self.sleep_timer[h] = 0.0;
            }
        }
        // Contacts, triggers, and pre-step velocities for hit events.
        let trigger_now = self.generate_pairs();
        self.wake_joint_motion();
        // External wake: torques, deadbeat motors and host edits write the
        // velocity fields directly, so a sleeper above thresholds wakes
        // before warmstart (teleports surface next step via BDF1).
        for h in 0..n {
            if self.asleep[h] {
                let b = &self.bodies[h];
                if b.velocity.length() > SLEEP_LIN || b.angular_velocity.length() > SLEEP_ANG {
                    self.wake_body(h);
                }
            }
        }
        for h in 0..n {
            // Driver baseline (Box2D-parity driver contract): the engine
            // never integrates kinematic bodies, so any pose change since
            // the last substep is a driver teleport — rewind its baseline
            // to the pre-teleport pose so contact/joint rows see driver
            // motion as within-step `dq` at full strength. Without this
            // only the (1-alpha)-diluted C0 residue couples, and a 2 m/s
            // pusher ghosts through its target (measured). TOI and pair
            // geometry above already ran on the true post-teleport poses.
            if self.bodies[h].body_type == BodyType::Kinematic {
                self.pos0[h] = self.prev_pos[h];
                self.rot0[h] = self.prev_rot[h];
            } else {
                self.pos0[h] = self.bodies[h].position;
                self.rot0[h] = self.bodies[h].orientation;
            }
            if !self.solvable(h) {
                self.inertial[h] = self.bodies[h].position;
                self.inertial_rot[h] = self.bodies[h].orientation;
                continue;
            }
            let b = &self.bodies[h];
            self.inertial[h] =
                b.position + b.velocity * DT_STEP + self.gravity * (DT_STEP * DT_STEP);
            self.inertial_rot[h] = quat_integrate(b.orientation, b.angular_velocity * DT_STEP);
        }
        // Eq. 19 warmstart decay.
        for pair in &mut self.pairs {
            for pt in &mut pair.points {
                pt.lam[0] *= ALPHA * GAMMA;
                pt.lam[1] *= ALPHA * GAMMA;
                pt.lam[2] *= ALPHA * GAMMA;
                pt.roll_lam[0] *= ALPHA * GAMMA;
                pt.roll_lam[1] *= ALPHA * GAMMA;
                pt.roll_lam[2] *= ALPHA * GAMMA;
                for k in 0..3 {
                    pt.pen[k] = (pt.pen[k] * GAMMA).clamp(PENALTY_MIN, PENALTY_MAX);
                }
            }
        }
        for j in &mut self.joints {
            for k in 0..3 {
                j.lam_l[k] *= ALPHA * GAMMA;
                j.lam_a[k] *= ALPHA * GAMMA;
                j.pen_l[k] = (j.pen_l[k] * GAMMA).clamp(PENALTY_MIN, PENALTY_MAX);
                j.pen_a[k] = (j.pen_a[k] * GAMMA).clamp(PENALTY_MIN, PENALTY_MAX);
            }
        }
        // Warmstarted positions (official adaptive weight is unity on free
        // fall; proper quaternion integration like the demo).
        for h in 0..n {
            if !self.solvable(h) || self.asleep[h] {
                continue;
            }
            self.bodies[h].position = self.inertial[h];
            self.bodies[h].orientation = self.inertial_rot[h];
        }
        // Main loop: primal sweep in reverse handle order (official sweeps
        // its body list head-first, i.e. reverse creation order), then duals.
        for _ in 0..ITERS {
            for h in (0..n).rev() {
                if self.solvable(h) && !self.asleep[h] {
                    self.solve_body(h);
                }
            }
            self.dual_update();
        }
        // BDF1 velocities + orientation renormalization (deviation from the
        // demo, which never renormalizes: prevents long-term quat drift).
        for h in 0..n {
            if !self.solvable(h) || self.asleep[h] {
                // Sleepers keep their zeroed velocity fields.
                continue;
            }
            let b = &mut self.bodies[h];
            b.velocity = (b.position - self.pos0[h]) / DT_STEP;
            // Official relative-rotation velocity `2*(q*q0^-1).xyz`, with a
            // rest deadband (positions are O(1) f32: sub-epsilon spin is dust).
            let spin = quat_diff_vec(b.orientation, self.rot0[h]);
            b.angular_velocity = if spin.length() < 1e-9 {
                Vec3::ZERO
            } else {
                spin / DT_STEP
            };
            b.orientation = b.orientation.normalize();
        }
        // Sleep bookkeeping (builtin parity, per body instead of per island):
        // slow dynamics accumulate quiet time and freeze; motion resets.
        // Support rule: only a body with a TOUCHING pair (signed gap) or a
        // joint may freeze. A damper shell is not support — without this a
        // caught arrival creeps to damper-terminal velocity (below the
        // sleep threshold) and freezes mid-air instead of touching down
        // (measured: 5 m/s drop froze +3cm up with zeroed velocity).
        // Joints count as support (a pendulum at rest hangs on its joint
        // with no contact pairs); empty shells do not.
        let mut supported = vec![false; n];
        for pair in &self.pairs {
            if pair.gap <= 0.0 && !pair.points.is_empty() {
                supported[pair.a] = true;
                supported[pair.b] = true;
            }
        }
        for j in &self.joints {
            supported[j.a] = true;
            supported[j.b] = true;
        }
        let joint_ready = self.joint_sleep_ready();
        let mut freeze = self.asleep.clone();
        for (h, &sup) in supported.iter().enumerate() {
            if !self.solvable(h) || self.asleep[h] {
                continue;
            }
            if !sup || !joint_ready[h] {
                self.sleep_timer[h] = 0.0;
                continue;
            }
            let b = &self.bodies[h];
            if b.velocity.length() < SLEEP_LIN && b.angular_velocity.length() < SLEEP_ANG {
                self.sleep_timer[h] += DT_STEP;
                freeze[h] = self.sleep_timer[h] >= SLEEP_TIME;
            } else {
                self.sleep_timer[h] = 0.0;
            }
        }
        self.propagate_joint_flags(&mut freeze, false);
        for (h, freeze) in freeze.into_iter().enumerate() {
            if freeze && !self.asleep[h] {
                self.sleep_body(h);
            }
        }
        if emit {
            self.emit_events(trigger_now);
        }
        // A solved TOI is an impact even when the separated contact shell
        // never acquired a discrete penetration. Preserve it for fracture.
        for hit in ccd_hits {
            if !self.contact_events[event_start..].iter().any(|e| {
                e.body_a == hit.body_a
                    && e.body_b == hit.body_b
                    && matches!(e.kind, ContactEventKind::Hit { .. })
            }) {
                self.contact_events.push(hit);
            }
        }
        // Refresh the driver baseline (end-of-step poses for the next
        // substep's teleport detection).
        for h in 0..n {
            self.prev_pos[h] = self.bodies[h].position;
            self.prev_rot[h] = self.bodies[h].orientation;
        }
    }

    /// Empty engine; `gravity` is a constant world-space acceleration
    /// applied to dynamic bodies each step.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            gravity,
            bodies: Vec::new(),
            pairs: Vec::new(),
            joints: Vec::new(),
            prev_touch: BTreeSet::new(),
            prev_trigger: BTreeSet::new(),
            contact_events: Vec::new(),
            trigger_events: Vec::new(),
            pos0: Vec::new(),
            rot0: Vec::new(),
            inertial: Vec::new(),
            inertial_rot: Vec::new(),
            pre_vel: Vec::new(),
            sleep_timer: Vec::new(),
            asleep: Vec::new(),
            prev_pos: Vec::new(),
            prev_rot: Vec::new(),
            time_debt: 0.0,
            #[cfg(feature = "gpu")]
            gpu_avbd: None,
            #[cfg(feature = "gpu")]
            gpu_fallback_steps: 0,
        }
    }

    /// Number of registered bodies.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    fn solvable(&self, h: usize) -> bool {
        self.bodies[h].body_type == BodyType::Dynamic && self.bodies[h].inv_mass > 0.0
    }

    fn ensure_scratch(&mut self) {
        let n = self.bodies.len();
        let old = self.pos0.len();
        self.pos0.resize(n, Vec3::ZERO);
        self.rot0.resize(n, Quat::IDENTITY);
        self.inertial.resize(n, Vec3::ZERO);
        self.inertial_rot.resize(n, Quat::IDENTITY);
        self.pre_vel.resize(n, Vec3::ZERO);
        self.sleep_timer.resize(n, 0.0);
        self.asleep.resize(n, false);
        // New indices start life with no teleport: baseline = current pose.
        self.prev_pos.resize(n, Vec3::ZERO);
        self.prev_rot.resize(n, Quat::IDENTITY);
        for h in old.min(n)..n {
            self.prev_pos[h] = self.bodies[h].position;
            self.prev_rot[h] = self.bodies[h].orientation;
        }
    }
}

/// Fixed step used by the solver tuning (the formulation is dt-parametric
/// through the mass terms; the iteration count is tuned for 1/60).
const DT_STEP: f32 = 1.0 / 60.0;
/// Fixed-step accumulator cap: at most this many 1/60 s substeps per
/// `step` call; larger host debts clamp (slow motion under extreme load
/// instead of the spiral of death). Matches the builtin spirit (12
/// substeps of its own loop) while keeping AVBD's core single-step.
const MAX_SUBSTEPS: usize = 4;
impl PhysicsEngine for AvbdEngine {
    fn step(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        // Opt-in GPU dispatch (rung 1): device execution stays gated on the
        // stub's `runs_on_device` (false until a real dispatch lands), so an
        // attached stub records the CPU fallback and the accumulator below
        // runs unchanged — attaching never changes the trajectory.
        #[cfg(feature = "gpu")]
        if self.gpu_avbd.as_ref().is_some_and(|s| !s.runs_on_device()) {
            self.gpu_fallback_steps += 1;
        }
        // Fixed-step accumulator: catch up in exact `DT_STEP` increments so
        // 120 Hz hosts alternate sim/skip and hitch frames replay whole
        // steps; past the cap the debt clamps (slow motion, never spiral).
        self.time_debt = (self.time_debt + dt).min(DT_STEP * MAX_SUBSTEPS as f32);
        let mut n = 0;
        while self.time_debt >= DT_STEP && n < MAX_SUBSTEPS {
            self.time_debt -= DT_STEP;
            n += 1;
            let last = self.time_debt < DT_STEP || n == MAX_SUBSTEPS;
            self.step_inner(last);
        }
    }
    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        self.bodies.push(body);
        BodyHandle::from(self.bodies.len() - 1)
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        let hi = handle.index();
        if hi >= self.bodies.len() {
            return;
        }
        let last = self.bodies.len() - 1;
        // Mirror the builtin: exiting trigger pairs report Exited.
        let mut exited: Vec<(usize, usize)> = self
            .prev_trigger
            .iter()
            .filter(|(a, b)| *a == hi || *b == hi)
            .map(|(a, b)| (*a, *b))
            .collect();
        exited.sort_unstable();
        for (a, b) in exited {
            self.trigger_events.push(TriggerEvent {
                body_a: crate::body::BodyHandle::from(a),
                body_b: crate::body::BodyHandle::from(b),
                kind: TriggerEventKind::Exited,
            });
        }
        self.bodies.swap_remove(hi);
        let map = |h: usize| if h == last { hi } else { h };
        self.pairs.retain_mut(|p| {
            if p.a == hi || p.b == hi {
                return false;
            }
            p.a = map(p.a);
            p.b = map(p.b);
            true
        });
        let removed = self.joints.iter().map(|j| j.a == hi || j.b == hi).collect();
        self.retain_joints(removed);
        for j in &mut self.joints {
            j.a = map(j.a);
            j.b = map(j.b);
        }
        // Handle-keyed state is stale after the swap (builtin parity).
        self.prev_touch.clear();
        self.prev_trigger.retain(|(a, b)| *a != hi && *b != hi);
        let mut remapped = BTreeSet::new();
        for (a, b) in &self.prev_trigger {
            remapped.insert((map(*a).min(map(*b)), map(*a).max(map(*b))));
        }
        self.prev_trigger = remapped;
        self.contact_events.clear();
        // Sleep vecs stay parallel; indices shifted, so wake everything
        // (removal is rare and correctness beats one quiet timer).
        if hi < self.sleep_timer.len() {
            self.sleep_timer.swap_remove(hi);
            self.asleep.swap_remove(hi);
        }
        if hi < self.prev_pos.len() {
            self.prev_pos.swap_remove(hi);
            self.prev_rot.swap_remove(hi);
        }
        for h in 0..self.bodies.len() {
            self.wake_body(h);
        }
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
        crate::migration::validate_joint(&kind)?;
        // A new constraint disturbs both assemblies (builtin wakes the
        // island; AVBD wakes per body). Spurious wake on rejected specs is
        // harmless — one quiet timer restarts.
        self.wake_body(usize::from(body_a));
        self.wake_body(usize::from(body_b));
        let (ia, ib) = (usize::from(body_a), usize::from(body_b));
        if ia == ib {
            return Err(JointError::SelfJoint { handle: ia });
        }
        if ia >= self.bodies.len() || ib >= self.bodies.len() {
            return Err(JointError::InvalidHandles { a: ia, b: ib });
        }
        // Gear holds no bodies of its own (coordinates other joints):
        // validate + capture the assembly constant up front; rows resolve
        // both sides directly (mirrors the builtin validation).
        if let JointKind::Gear {
            joint_a,
            joint_b,
            ratio,
        } = kind
        {
            if !ratio.is_finite() {
                return Err(JointError::NonFinite {
                    field: "ratio".to_string(),
                });
            }
            let (Some(ja), Some(jb)) = (
                self.joints.get(joint_a.index()),
                self.joints.get(joint_b.index()),
            ) else {
                return Err(JointError::UnknownRef {
                    handle: self.joints.len(),
                });
            };
            if !matches!(ja.kind, AvbdJointKind::Revolute | AvbdJointKind::Prismatic)
                || !matches!(jb.kind, AvbdJointKind::Revolute | AvbdJointKind::Prismatic)
            {
                return Err(JointError::Unsupported {
                    detail: "gear must coordinate revolute/prismatic joints".to_string(),
                });
            }
            // NOTE: coordinates read the REFERENCED joints' bodies.
            let ca = {
                let (x, y) = (&self.bodies[ja.a], &self.bodies[ja.b]);
                Self::joint_coordinate(ja, x, y).unwrap_or(0.0)
            };
            let cb = {
                let (x, y) = (&self.bodies[jb.a], &self.bodies[jb.b]);
                Self::joint_coordinate(jb, x, y).unwrap_or(0.0)
            };
            let joint = AvbdJoint {
                a: ia,
                b: ib,
                la: Vec3::ZERO,
                lb: Vec3::ZERO,
                ax_a: Vec3::X,
                ax_b: Vec3::X,
                bx_a: Vec3::X,
                bx_b: Vec3::X,
                susp: [0.0; 2],
                kind: AvbdJointKind::Gear,
                spec: kind,
                ref_val: ca + ratio * cb,
                q_ref: Quat::IDENTITY,
                lim: None,
                mot: None,
                acc_lim: 0.0,
                lim_dual: 0.0,
                gb: [joint_a.index(), joint_b.index()],
                gratio: ratio,
                gear_mem: None,
                six_lin: [AxisConfig::Free; 3],
                six_ang: [AxisConfig::Free; 3],
                dref: Vec3::ZERO,
                sacc: [0.0; 6],
                lam_l: [0.0; 3],
                lam_a: [0.0; 3],
                pen_l: [JOINT_PENALTY_INIT; 3],
                pen_a: [JOINT_PENALTY_INIT; 3],
            };
            self.joints.push(joint);
            return Ok(JointHandle::from(self.joints.len() - 1));
        }
        let ba = &self.bodies[ia];
        let bb = &self.bodies[ib];
        // Frames + assembly references, resolved once in the shared
        // `joint::resolve_joint` (same values the builtin engine captures).
        let r = crate::joint::resolve_joint(
            &kind,
            ba.position,
            ba.orientation,
            bb.position,
            bb.orientation,
        )
        .expect("non-gear/sixdof kinds resolve");
        let mut joint = AvbdJoint {
            a: ia,
            b: ib,
            la: r.la,
            lb: r.lb,
            ax_a: r.ax_a,
            ax_b: r.ax_b,
            bx_a: r.bx_a,
            bx_b: r.bx_b,
            susp: [0.0; 2],
            kind: AvbdJointKind::Ball,
            spec: kind,
            ref_val: 0.0,
            q_ref: r.ref_quat,
            lim: None,
            mot: None,
            acc_lim: 0.0,
            lim_dual: 0.0,
            gb: [0; 2],
            gratio: 0.0,
            gear_mem: None,
            six_lin: [AxisConfig::Free; 3],
            six_ang: [AxisConfig::Free; 3],
            dref: r.ref_anchor_delta,
            sacc: [0.0; 6],
            lam_l: [0.0; 3],
            lam_a: [0.0; 3],
            pen_l: [JOINT_PENALTY_INIT; 3],
            pen_a: [JOINT_PENALTY_INIT; 3],
        };
        match kind {
            JointKind::Ball { .. } => {}
            JointKind::Revolute { limit, motor, .. } => {
                // Admission policy: degenerate hinge axes reject (the
                // builtin substitutes a fallback instead).
                if r.degenerate {
                    return Err(JointError::BadAxis {
                        detail: "degenerate revolute hinge axis".to_string(),
                    });
                }
                joint.kind = AvbdJointKind::Revolute;
                joint.ref_val = r.ref_angle;
                joint.lim = limit.map(|l| [l.min, l.max]);
                joint.mot = motor.map(|m| [m.target_speed, m.max_torque]);
            }
            JointKind::Prismatic { limit, motor, .. } => {
                if r.degenerate {
                    return Err(JointError::BadAxis {
                        detail: "degenerate prismatic slide axis".to_string(),
                    });
                }
                joint.kind = AvbdJointKind::Prismatic;
                joint.ref_val = r.ref_length;
                joint.lim = limit.map(|l| [l.min, l.max]);
                joint.mot = motor.map(|m| [m.target_speed, m.max_force]);
            }
            JointKind::Fixed { .. } => {
                joint.kind = AvbdJointKind::Fixed;
                joint.q_ref = r.ref_quat;
            }
            JointKind::Distance { .. } => {
                joint.kind = AvbdJointKind::Distance;
                joint.ref_val = r.ref_distance;
            }
            JointKind::Wheel {
                suspension, motor, ..
            } => {
                joint.kind = AvbdJointKind::Wheel;
                joint.ref_val = r.ref_length;
                joint.susp = [suspension.frequency_hz, suspension.damping_ratio];
                joint.mot = motor.map(|m| [m.target_speed, m.max_torque]);
            }
            JointKind::Gear { .. } => {
                // Handled by the early gear block above; unreachable here.
                return Err(JointError::Unsupported {
                    detail: "gear handled above".to_string(),
                });
            }
            JointKind::SixDof {
                linear, angular, ..
            } => {
                joint.kind = AvbdJointKind::SixDof;
                joint.six_lin = linear;
                joint.six_ang = angular;
                joint.q_ref = r.ref_quat;
                joint.dref = r.ref_anchor_delta;
            }
        };
        self.joints.push(joint);
        Ok(JointHandle::from(self.joints.len() - 1))
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        if handle.index() < self.joints.len() {
            let (a, b) = (self.joints[handle.index()].a, self.joints[handle.index()].b);
            self.wake_body(a);
            self.wake_body(b);
            let mut removed = vec![false; self.joints.len()];
            removed[handle.index()] = true;
            self.retain_joints(removed);
        }
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Result<Option<RaycastHit>, QueryError> {
        crate::errors::check_ray_input(ray.origin, ray.direction, max_dist)?;
        let mut closest: Option<RaycastHit> = None;
        for (h, body) in self.bodies.iter().enumerate() {
            let inverse = body.orientation.inverse();
            let origin = inverse * (ray.origin - body.position);
            let direction = inverse * ray.direction;
            let Some((distance, local_normal)) =
                raycast_shape_hit(&body.shape, origin, direction, max_dist)
            else {
                continue;
            };
            let nearer = closest.as_ref().is_none_or(|c| distance < c.distance);
            if nearer {
                closest = Some(RaycastHit {
                    handle: BodyHandle::from(h),
                    point: ray.point_at(distance),
                    normal: (body.orientation * local_normal).normalize_or(Vec3::Y),
                    distance,
                });
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
                crate::body::BodyHandle::from(h),
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

    fn drain_trigger_events(&mut self) -> Vec<TriggerEvent> {
        std::mem::take(&mut self.trigger_events)
    }

    fn drain_contact_events(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.contact_events)
    }

    fn wake_body(&mut self, handle: BodyHandle) {
        self.wake_body(handle.index());
    }
}

impl AvbdEngine {
    /// Reconcile trigger and solid-contact transitions after a step.
    fn emit_events(&mut self, trigger_now: BTreeSet<(usize, usize)>) {
        for (a, b) in trigger_now.symmetric_difference(&self.prev_trigger) {
            let (a, b) = (*a, *b);
            let entered = trigger_now.contains(&(a, b));
            self.trigger_events.push(TriggerEvent {
                body_a: crate::body::BodyHandle::from(a),
                body_b: crate::body::BodyHandle::from(b),
                kind: if entered {
                    TriggerEventKind::Entered
                } else {
                    TriggerEventKind::Exited
                },
            });
        }
        self.prev_trigger = trigger_now;
        let mut touch_now = BTreeSet::new();
        let mut hits: Vec<ContactEvent> = Vec::new();
        for p in &self.pairs {
            let a = &self.bodies[p.a];
            let b = &self.bodies[p.b];
            let mut deepest = 0.0f32;
            let mut witness = (a.position + b.position) * 0.5;
            for pt in &p.points {
                let pa = a.position + a.orientation * pt.ra;
                let pb = b.position + b.orientation * pt.rb;
                let gap = p.n.dot(pa - pb);
                if -gap > deepest {
                    deepest = -gap;
                    witness = pa;
                }
            }
            if deepest > CONTACT_BEGIN_SLOP {
                touch_now.insert((p.a, p.b));
                if !self.prev_touch.contains(&(p.a, p.b)) {
                    let approach = -((self.pre_vel[p.a] - self.pre_vel[p.b]).dot(p.n));
                    if approach > CONTACT_HIT_THRESHOLD {
                        hits.push(ContactEvent {
                            body_a: crate::body::BodyHandle::from(p.a),
                            body_b: crate::body::BodyHandle::from(p.b),
                            kind: ContactEventKind::Hit {
                                point: witness,
                                normal: -p.n,
                                approach_speed: approach,
                            },
                        });
                    }
                }
            }
        }
        for (a, b) in touch_now.symmetric_difference(&self.prev_touch) {
            let (a, b) = (*a, *b);
            self.contact_events.push(ContactEvent {
                body_a: crate::body::BodyHandle::from(a),
                body_b: crate::body::BodyHandle::from(b),
                kind: if touch_now.contains(&(a, b)) {
                    ContactEventKind::Begin
                } else {
                    ContactEventKind::End
                },
            });
        }
        self.contact_events.append(&mut hits);
        self.prev_touch = touch_now;
    }
}
