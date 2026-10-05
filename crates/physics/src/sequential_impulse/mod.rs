//! Sequential-impulse (projected Gauss-Seidel) solver: thin step orchestrator.
//! Owns [`SequentialImpulseEngine`] (structure, constructors/setters and the
//! [`PhysicsEngine::step`] pipeline); substep passes live in `step`, the
//! narrowphase in `narrow`, caches in `caches`, island sleep in `sleep`,
//! queries in `queries`, events in `events` and scalar helpers in `math`.

mod caches;
mod contacts;
mod events;
pub mod hooks;
mod islands;
pub mod joints;
mod math;
mod mover;
mod narrow;
mod queries;
mod sleep;
mod step;

pub(crate) use crate::engine::{Manifold, ManifoldPoint, PhysicsEngine};
pub use caches::{NarrowShardPool, SatCache, SatCacheEntry};
pub(crate) use hooks::HookOverride;
pub use hooks::{
    ContactHooks, ContactPointView, ContactView, ModifyContext, OneWayPlatform, PairFilterContext,
    SolverFlags,
};
pub use math::{
    apply_impulse, effective_mass, inv_inertia_axis, mul_inv_inertia, point_velocity,
    solve_normal_block, solve_small,
};
pub use mover::{KinematicMover, MoverHandle};
pub use narrow::{box_manifold, detect_collisions_into, obb_sat};
pub(crate) use queries::DEFAULT_MAX_CCD_SUBSTEPS;
pub use queries::{
    ContinuousHit, ccd_impact_velocity, find_angular_continuous_hit, kinematic_cast,
    remove_angular_approach, sweep_gap,
};
pub(crate) use queries::{raycast_body_hit, raycast_shape_hit};
pub use step::ManifoldState;
pub(crate) use step::{WarmCache, WarmPoint};

use self::caches::*;
use self::events::*;
use self::math::*;
use self::narrow::*;
use self::step::*;

use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use dashmap::DashMap;

use glam::{Quat, Vec3};

/// Seconds → milliseconds for step-timing telemetry.
const MS_PER_SEC: f64 = 1000.0;
/// Max spread of per-body required substeps that still skips the filter.
const SUBSTEP_FILTER_SPREAD: u32 = 4;

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::broadphase::{
    BroadPhase, BroadPhaseBackend, BroadPhaseKind, BroadPhaseStats, PrevPose, StepBudget,
    StepTiming,
};
use crate::distance;
use crate::errors::{QueryError, SnapshotError, StepError};
#[cfg(feature = "gpu")]
use crate::gpu::GpuSequentialImpulse;
use crate::joint::{Joint, JointHandle, JointKind};
use crate::math::{Ray, RaycastHit};
use crate::migration::{JointReference, JointSnapshot};
use crate::shape::Shape;
use crate::trigger::{
    CONTACT_BEGIN_SLOP, ContactEvent, ContactEventKind, ContactForceEvent, TriggerEvent,
    TriggerEventKind,
};
use crate::wide::{SolverStep, build_solver_steps};

/// Manifold point capacity: every per-point parallel array (`acc`,
/// `target`, `pen0`, warm-start tables) and the `1..=N` count invariant
/// are sized by this. Box2D-class 4-point cap: one face contact (2) plus
/// margin for speculative/rolling rows sharing the same lanes.
pub(crate) const MAX_MANIFOLD_POINTS: usize = 4;

/// Minimum substep count accepted by
/// [`step_with_substeps`](SequentialImpulseEngine::step_with_substeps)
/// (R7, Box3D `b3World_Step(dt, subStepCount)` parity).
pub const MIN_SUBSTEP_COUNT: u32 = 1;
/// Maximum substep count accepted by
/// [`step_with_substeps`](SequentialImpulseEngine::step_with_substeps)
/// (R7, Box3D `b3World_Step(dt, subStepCount)` parity).
pub const MAX_SUBSTEP_COUNT: u32 = 64;

/// Admission check for an explicit per-call substep count (R7): `1..=64`
/// passes, anything else is a typed refusal — never a silent clamp (a
/// clamped count would change solver quality behind the caller's back).
pub(crate) fn check_substep_count(n: u32) -> Result<(), StepError> {
    if (MIN_SUBSTEP_COUNT..=MAX_SUBSTEP_COUNT).contains(&n) {
        Ok(())
    } else {
        Err(StepError::BadSubstepCount {
            got: n,
            min: MIN_SUBSTEP_COUNT,
            max: MAX_SUBSTEP_COUNT,
        })
    }
}

/// The CPU reference physics engine: sequential-impulse solver with a
/// selectable broadphase, manifold generation, island-coherent sleeping,
/// warm-started contacts and joints, and optional SIMD-wide / GPU contact
/// solving. Sweep-and-Prune is the default; UniformGrid is opt-in while its
/// workload tradeoffs are benchmarked.
///
/// Step pipeline per [`PhysicsEngine::step`]: rebuild AABBs and broadphase
/// pairs → narrowphase manifolds → union-find islands (contacts + joints) →
/// `substeps` × (warm start, velocity iterations with friction/restitution,
/// positional Baumgarte pass) → integration. Bodies outside active islands
/// sleep as a whole island and wake together.
#[allow(missing_docs)]
pub struct SequentialImpulseEngine {
    pub bodies: Vec<RigidBody>,
    broadphase: BroadPhaseBackend,
    gravity: Vec3,
    substeps: u32,
    velocity_iterations: u32,
    position_iterations: u32,
    /// CFM softness scale for the positional pass (G3): 0 = rigid, larger =
    /// softer, smoother corrections spread over more iterations.
    contact_softness: f32,
    /// Accumulated normal impulses per matched contact point, keyed by sorted
    /// body pair. Applied in a dedicated WarmStart stage (G2b).
    warm_impulses: WarmCache,
    /// Island (constraint-graph component) per body, rebuilt every step from
    /// the contact graph (G4). Sleep and wake are island-coherent, exactly
    /// like Jolt/Box3D: a resting stack can only sleep as a whole, otherwise
    /// an awake neighbour's contact immediately re-wakes a per-body sleeper.
    island: Vec<u32>,
    /// Per-island sleep timers, keyed by island root handle.
    island_timers: FxHashMap<u32, f32>,
    /// Whether each body ever exceeded the frozen speed gate since it was
    /// added (parallel to `bodies`; see `is_body_frozen`). Monotonic until
    /// the body is removed: an island whose members never moved qualifies
    /// for the frozen fast track, one that did keeps the legacy timers, so
    /// settling scenes (including the determinism snapshot) are untouched.
    body_moved: Vec<bool>,
    /// Post-wake grace before the frozen fast track may fire, keyed by
    /// island root handle (see `update_sleep`). A woken island — even one
    /// that settles back to numerical stillness immediately (zero-g
    /// teleport overlaps) — accumulates sleep at the normal rate for a few
    /// steps, preserving the pre-fast-track wakefulness floor the
    /// penetration-wake tests pin. Never-woken exact-rest islands skip the
    /// grace and sleep in ~2 steps.
    island_grace: FxHashMap<u32, u32>,
    asleep: Vec<bool>,
    /// Step-start pose per body, parallel to `bodies`. Drivers move kinematic
    /// bodies by setting positions directly, so the velocity field alone does
    /// not describe their motion: the broadphase sweep, the speculative
    /// margin and the kinematic sweep read the step displacement
    /// (`position - prev`) instead. Updated at the top of every `step`
    /// (even on the fully-sleeping fast path, so the delta stays one step).
    prev_pose: Vec<PrevPose>,
    /// Driver velocity save area, parallel to `bodies` conceptually: above-
    /// gate teleports temporarily install the implied motion into the
    /// kinematic velocity fields for the step (margins, CCD, contacts all
    /// read fields), then restore the driver values at step end. Only
    /// entries for touched bodies are pushed; empty in steady state when
    /// nobody teleports.
    saved_driver_vel: Vec<(usize, Vec3, Vec3)>,
    /// Attached kinematic platform movers (R2, dense handle order — see
    /// [`MoverHandle`]): transient driver state, not simulation state.
    /// Snapshots, restores and solver migrations do not carry them;
    /// re-attach after a restore. Invalidated by `remove_body` (dropped
    /// with the platform, remapped with the swapped tail).
    movers: Vec<KinematicMover>,
    /// Persistent joint constraints with warm-start state (G5). Joints also
    /// feed the island union-find: jointed bodies sleep and wake together.
    joints: Vec<Joint>,
    /// Sorted body pairs connected by a joint. Jointed bodies never collide
    /// (Box2D `collide_connected = false` default): a hinge pin passes through
    /// the arm, so the parts legitimately sweep through each other's space,
    /// and contact friction there would act as a phantom brake on the joint.
    joint_pairs: FxHashSet<(usize, usize)>,
    /// Diagnostics: (body_a, body_b) of the last substep's manifolds.
    debug_pairs: Vec<(usize, usize)>,
    /// Trigger pairs overlapping on the previous completed step.
    trigger_pairs: FxHashSet<(usize, usize)>,
    /// Solid-contact pairs touching on the previous completed step
    /// (canonical min/max keys). Frozen pairs retain their state — sleep
    /// emits no transitions.
    contact_touch: FxHashSet<(usize, usize)>,
    /// Contact transitions waiting for the caller to drain (hits recorded
    /// pre-solve every substep with per-step dedupe, begin/end reconciled
    /// at step end).
    contact_events: Vec<ContactEvent>,
    /// Per-pair contact-force reports waiting for the caller to drain
    /// (emitted at step end from the tracked step peak; empty unless a
    /// body opts in via its contact-force threshold).
    contact_force_events: Vec<ContactForceEvent>,
    /// Running per-pair peak of per-substep contact force (total normal
    /// impulse over the substep length), tracked only while at least one
    /// body enables contact-force reports. Cleared at every step start;
    /// step-end emission reads this, never the solver state directly.
    force_peak: FxHashMap<(usize, usize), f32>,
    /// Pairs that already emitted a hit this step (dedupe for sustained
    /// crushes); cleared at s==0 of every step.
    scratch_hit_pairs: FxHashSet<(usize, usize)>,
    /// Wall-clock breakdown of the last completed `step` (diagnostics only).
    last_step_timing: StepTiming,
    /// Worst-case budget: deterministic substep shedding (see
    /// [`StepBudget`]). `None` disables shedding. Default: on.
    step_budget: Option<StepBudget>,
    /// Substeps shed by the budget on the last completed `step` (0 when the
    /// full speed-requested count ran). Observable marker for the fallback,
    /// read via [`SequentialImpulseEngine::last_substep_shed`].
    last_shed: u32,
    /// Maximum conservative-advancement iterations per pair in the angular
    /// (nonlinear) CCD sweep — the Rapier `max_ccd_substeps` analog.
    /// `0` disables the angular sweep entirely. Default matches the legacy
    /// fixed loop, so default scenes are bit-identical.
    max_ccd_substeps: usize,
    /// Angular sweeps that exhausted the iteration cap on the last completed
    /// `step` (0 when every applied clamp carried the tunnel-free proof).
    /// Observable marker for the best-effort fallback, read via
    /// [`SequentialImpulseEngine::last_ccd_caps`].
    last_ccd_caps: u32,
    /// Enter/exit transitions waiting for the caller to drain.
    trigger_events: Vec<TriggerEvent>,
    /// G7: SIMD-wide contact solver path for single-point manifolds.
    /// Default [`SolvePath::Wide`]. Select [`SolvePath::Scalar`] for
    /// bit-exact scalar reproduction.
    wide_solver: crate::flags::SolvePath,
    /// Scratch buffers reused across substeps to avoid per-frame allocations.
    scratch_manifolds: Vec<Manifold>,
    pub scratch_pairs: Vec<(usize, usize)>,
    /// Per-substep bucket boundaries over `scratch_pairs` (stable counting
    /// sort by required substeps, rebuilt once per step when filtering).
    scratch_bucket_edges: Vec<usize>,
    scratch_clamped: Vec<bool>,
    scratch_parent: Vec<usize>,
    /// Pooled narrowphase shard buffers for scheduler dispatch, taken and
    /// restored around the substep loop like the other scratch state.
    scratch_narrow_shards: NarrowShardPool,
    /// Generation stamps for the flat singleton eligibility scan (one entry
    /// per body; see `flat_singleton_eligible`). Reused every step, never
    /// reallocated after the first large scene.
    flat_marks: Vec<u32>,
    /// Current generation for `flat_marks` (0 reserved as the cleared state).
    flat_gen: u32,
    /// Reused flat-path solve shards (one per worker chunk, not per
    /// manifold), taken and restored around the substep loop like the
    /// other scratch state.
    scratch_flat_shards: Vec<IslandWork>,
    /// G7: optional GPU contact solver (gpu feature). When attached,
    /// single-point manifolds are solved on the GPU instead of the CPU
    /// wide path; multi-point manifolds stay on the CPU island path.
    #[cfg(feature = "gpu")]
    gpu_solver: Option<GpuSequentialImpulse>,
    narrow_cache: FxHashMap<(usize, usize), NarrowCacheEntry>,
    sat_cache: SatCache,
    /// Optional Rapier-style contact hooks (`filter_pair` at the narrow
    /// input, `filter_intersection_pair` over the trigger overlaps,
    /// `modify_contact` pre-solve). `None` (default) is the legacy
    /// bit-exact path; see the [`hooks`](crate::sequential_impulse::hooks)
    /// module docs for the determinism contract.
    contact_hooks: Option<Box<dyn ContactHooks>>,
    /// Canonical pair keys whose [`SolverFlags`](hooks::SolverFlags) say
    /// `READ_ONLY` this step: manifolds are built and events emitted, but
    /// the islands never solve them. Rebuilt by [`apply_hook_filter`](hooks)
    /// every step; empty without hooks (or with all-`COMPUTE` hooks).
    hook_read_only: FxHashSet<(usize, usize)>,
}

/// Dense joint rebuild after removals: drops the marked joints, remaps
/// surviving gear references so they keep pointing at the same joints (a
/// gear whose reference cannot be remapped is dropped — sound, never
/// dangling), and recomputes the no-collide pair set (gears carry none —
/// the underlying joints already do).
fn rebuild_joints(
    joints: &mut Vec<Joint>,
    joint_pairs: &mut FxHashSet<(usize, usize)>,
    drop_j: Vec<bool>,
) {
    let kinds: Vec<_> = joints.iter().map(|j| j.kind).collect();
    let remap = crate::migration::joint_remap(&kinds, drop_j);
    let mut old = 0;
    joints.retain_mut(|j| {
        let keep = remap[old].is_some();
        old += 1;
        if keep {
            crate::migration::remap_gear(&mut j.kind, &remap);
        }
        keep
    });
    *joint_pairs = joints
        .iter()
        .filter(|j| !matches!(j.kind, JointKind::Gear { .. }))
        .map(|j| {
            let (a, b) = (j.body_a.index(), j.body_b.index());
            (a.min(b), a.max(b))
        })
        .collect();
}

impl SequentialImpulseEngine {
    /// Bodies in handle order, cloned for solver migration (`Engine`
    /// re-registers them 1:1, so handles stay valid across the switch).
    pub(crate) fn bodies_snapshot(&self) -> Vec<RigidBody> {
        self.bodies.clone()
    }

    /// Number of registered bodies (dense handles).
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Driver baselines are physical step history, not discardable warm impulses.
    pub(crate) fn body_baselines(&self) -> Vec<PrevPose> {
        self.prev_pose.clone()
    }

    /// Restore a driver's within-step motion baseline after rebuilding.
    pub(crate) fn restore_body_baseline(&mut self, h: BodyHandle, pose: PrevPose) {
        if let Some(old) = self.prev_pose.get_mut(h.index()) {
            *old = pose;
        }
    }

    /// Completed-step event baseline for transparent solver migration.
    pub(crate) fn event_state(&self) -> crate::migration::EventState {
        crate::migration::EventState {
            contacts: self
                .contact_touch
                .iter()
                .copied()
                .map(|(a, b)| (BodyHandle::from(a), BodyHandle::from(b)))
                .collect(),
            triggers: self
                .trigger_pairs
                .iter()
                .copied()
                .map(|(a, b)| (BodyHandle::from(a), BodyHandle::from(b)))
                .collect(),
        }
    }

    /// Seed a rebuilt solver without manufacturing a new contact/trigger begin.
    pub(crate) fn restore_event_state(&mut self, state: crate::migration::EventState) {
        self.contact_touch = state
            .contacts
            .into_iter()
            .map(|(a, b)| (a.index(), b.index()))
            .collect();
        self.trigger_pairs = state
            .triggers
            .into_iter()
            .map(|(a, b)| (a.index(), b.index()))
            .collect();
    }

    /// Captures a versioned plain-data [`WorldSnapshot`](crate::snapshot::WorldSnapshot)
    /// of this engine: tuning, every body field verbatim, sleep/island
    /// state, driver and event baselines, pending queues, joint specs with
    /// assembly references and warm accumulators, and the contact
    /// warm-start cache. Reads state only — the engine is untouched.
    ///
    /// # Errors
    ///
    /// [`SnapshotError::Unsupported`] while contact hooks or a GPU solver
    /// are attached (host callbacks and device state are not serializable —
    /// detach them first; the no-hooks path is the bit-exact baseline).
    pub(crate) fn capture_world_snapshot(
        &self,
    ) -> Result<crate::snapshot::WorldSnapshot, SnapshotError> {
        use crate::snapshot::{
            BodySnapshot, BroadPhaseSnapshot, ContactEventSnapshot, ContactForceEventSnapshot,
            JointMotorSnapshot, JointReferenceSnapshot, JointSnapshot as SnapshotJoint,
            PoseSnapshot, TriggerEventSnapshot, TuningSnapshot, WORLD_SNAPSHOT_VERSION,
            WarmPairSnapshot, WarmPointSnapshot, WorldSnapshot,
        };
        if self.contact_hooks.is_some() {
            return Err(SnapshotError::Unsupported {
                detail: "cannot snapshot with contact hooks attached".to_string(),
            });
        }
        #[cfg(feature = "gpu")]
        if self.gpu_solver.is_some() {
            return Err(SnapshotError::Unsupported {
                detail: "cannot snapshot with a GPU solver attached".to_string(),
            });
        }
        let mut island_timers: Vec<(u32, f32)> =
            self.island_timers.iter().map(|(&k, &v)| (k, v)).collect();
        island_timers.sort_by_key(|&(k, _)| k);
        let mut island_grace: Vec<(u32, u32)> =
            self.island_grace.iter().map(|(&k, &v)| (k, v)).collect();
        island_grace.sort_unstable();
        let mut event_contacts: Vec<(u32, u32)> = self
            .contact_touch
            .iter()
            .map(|&(a, b)| (a as u32, b as u32))
            .collect();
        event_contacts.sort_unstable();
        let mut event_triggers: Vec<(u32, u32)> = self
            .trigger_pairs
            .iter()
            .map(|&(a, b)| (a as u32, b as u32))
            .collect();
        event_triggers.sort_unstable();
        let mut warm: Vec<WarmPairSnapshot> = self
            .warm_impulses
            .iter()
            .map(|(&pair, (pts, count))| WarmPairSnapshot {
                pair,
                points: pts
                    .iter()
                    .take(*count)
                    .map(WarmPointSnapshot::from_point)
                    .collect(),
            })
            .collect();
        warm.sort_by_key(|w| w.pair);
        Ok(WorldSnapshot {
            version: WORLD_SNAPSHOT_VERSION,
            gravity: self.gravity.to_array(),
            tuning: TuningSnapshot {
                substeps: self.substeps,
                velocity_iterations: self.velocity_iterations,
                position_iterations: self.position_iterations,
                contact_softness: self.contact_softness,
                step_budget: self
                    .step_budget
                    .map(|b| (b.max_pair_substeps, b.min_substeps)),
                max_ccd_substeps: self.max_ccd_substeps,
                broadphase: match self.broadphase_kind() {
                    BroadPhaseKind::SweepAndPrune => BroadPhaseSnapshot::SweepAndPrune,
                    BroadPhaseKind::UniformGrid => BroadPhaseSnapshot::UniformGrid,
                    BroadPhaseKind::DynamicAabbTree => BroadPhaseSnapshot::DynamicAabbTree,
                    BroadPhaseKind::Auto => BroadPhaseSnapshot::Auto,
                },
                wide_solver: self.wide_solver.use_wide(),
            },
            bodies: self.bodies.iter().map(BodySnapshot::from_body).collect(),
            asleep: self.asleep.clone(),
            moved: self.body_moved.clone(),
            islands: self.island.clone(),
            island_timers,
            island_grace,
            prev: self
                .prev_pose
                .iter()
                .map(|p| PoseSnapshot::from_parts(p.pos, p.rot))
                .collect(),
            event_contacts,
            event_triggers,
            pending_contacts: self
                .contact_events
                .iter()
                .map(ContactEventSnapshot::from_event)
                .collect(),
            pending_triggers: self
                .trigger_events
                .iter()
                .map(TriggerEventSnapshot::from_event)
                .collect(),
            pending_forces: self
                .contact_force_events
                .iter()
                .map(ContactForceEventSnapshot::from_event)
                .collect(),
            warm,
            joints: self
                .joints
                .iter()
                .map(|j| SnapshotJoint {
                    a: j.body_a.as_u32(),
                    b: j.body_b.as_u32(),
                    spec: crate::snapshot::JointKindSnapshot::from_spec(j.kind),
                    reference: JointReferenceSnapshot {
                        angle: j.reference_angle,
                        length: j.reference_length,
                        distance: j.reference_distance,
                        rotation: [
                            j.reference_quat.x,
                            j.reference_quat.y,
                            j.reference_quat.z,
                            j.reference_quat.w,
                        ],
                        anchor_delta: j.reference_anchor_delta.to_array(),
                    },
                    servo: j.servo.map(JointMotorSnapshot::from_motor),
                    acc_lin: j.acc_lin,
                    acc_ang: j.acc_ang,
                    acc_limit: j.acc_limit,
                    acc_dist: j.acc_dist,
                    acc_rope: j.acc_rope,
                    acc_gear: j.acc_gear,
                    gear_mem: j.gear_mem,
                    acc_6dof: j.acc_6dof,
                })
                .collect(),
            soft_bodies: Vec::new(),
            soft_touch: Vec::new(),
            soft_friction: None,
            soft_contact_events: Vec::new(),
            fracture_events: Vec::new(),
        })
    }

    /// Replaces this engine's state with a captured [`WorldSnapshot`](crate::snapshot::WorldSnapshot):
    /// fresh engine from the snapshot gravity and tuning, bodies and
    /// driver/sleep/event state restored verbatim, joints pushed directly
    /// (no assembly re-solve, so axes and references keep their exact
    /// bits), warm caches and pending queues requeued. Stepping afterwards
    /// continues the captured trajectory bit-identically.
    ///
    /// # Errors
    ///
    /// [`SnapshotError::InvalidData`] on length skew, dangling handles,
    /// non-finite scalars or rejected shapes/joints;
    /// [`SnapshotError::Unsupported`] when the snapshot carries soft
    /// bodies (a bare sequential-impulse engine cannot own them).
    pub(crate) fn restore_world(
        &mut self,
        snap: &crate::snapshot::WorldSnapshot,
    ) -> Result<(), SnapshotError> {
        use crate::snapshot::{check_joint_motor, check_joint_spec};
        let bad = |detail: &str| SnapshotError::InvalidData {
            detail: detail.to_string(),
        };
        let n = snap.bodies.len();
        for (name, len) in [
            ("asleep", snap.asleep.len()),
            ("moved", snap.moved.len()),
            ("islands", snap.islands.len()),
            ("prev", snap.prev.len()),
        ] {
            if len != n {
                return Err(bad(&format!("`{name}` length {len} != bodies {n}")));
            }
        }
        if !snap.soft_bodies.is_empty() {
            return Err(SnapshotError::Unsupported {
                detail:
                    "snapshot holds soft bodies: restore through Engine::restore_world_snapshot"
                        .to_string(),
            });
        }
        // Validate everything before mutating: a failed restore must not
        // leave a half-built engine behind.
        let mut bodies = Vec::with_capacity(n);
        for b in &snap.bodies {
            bodies.push(b.to_body()?);
        }
        for (i, j) in snap.joints.iter().enumerate() {
            let spec = j.spec.to_spec();
            check_joint_spec(&spec, j.a as usize, j.b as usize, n, i)?;
            if let JointKind::Gear {
                joint_a, joint_b, ..
            } = spec
            {
                for r in [joint_a, joint_b] {
                    if !matches!(
                        snap.joints[r.index()].spec,
                        crate::snapshot::JointKindSnapshot::Revolute { .. }
                            | crate::snapshot::JointKindSnapshot::Prismatic { .. }
                    ) {
                        return Err(bad("gear coordinates a non-hinge/slider joint"));
                    }
                }
            }
            if let Some(m) = j.servo {
                check_joint_motor(&m.to_motor())?;
            }
            for (name, v) in [
                ("reference.angle", j.reference.angle),
                ("reference.length", j.reference.length),
                ("reference.distance", j.reference.distance),
            ] {
                if !v.is_finite() {
                    return Err(bad(&format!("non-finite joint `{name}`")));
                }
            }
        }
        for &(a, b) in snap.event_contacts.iter().chain(snap.event_triggers.iter()) {
            if a as usize >= n || b as usize >= n {
                return Err(bad("event baseline references out-of-range bodies"));
            }
        }
        for e in &snap.pending_contacts {
            if e.a as usize >= n || e.b as usize >= n {
                return Err(bad("pending contact references out-of-range bodies"));
            }
        }
        for e in snap
            .pending_triggers
            .iter()
            .map(|e| (e.a, e.b))
            .chain(snap.pending_forces.iter().map(|e| (e.a, e.b)))
        {
            if e.0 as usize >= n || e.1 as usize >= n {
                return Err(bad("pending event references out-of-range bodies"));
            }
        }
        for f in &snap.pending_forces {
            if !(f.force.is_finite() && f.force >= 0.0) {
                return Err(bad("pending force event is not finite non-negative"));
            }
        }
        for w in &snap.warm {
            if w.pair.0 >= n || w.pair.1 >= n {
                return Err(bad("warm cache references out-of-range bodies"));
            }
            if w.points.is_empty() || w.points.len() > MAX_MANIFOLD_POINTS {
                return Err(bad("warm cache point count outside 1..=4"));
            }
        }
        for (root, _) in snap
            .island_timers
            .iter()
            .map(|&(r, t)| (r, t))
            .chain(snap.island_grace.iter().map(|&(r, g)| (r, g as f32)))
        {
            if root as usize >= n {
                return Err(bad("sleep map references out-of-range island root"));
            }
        }
        // Rebuild from scratch: tuning first (backend switches clear the
        // warm cache, so state lands after), then bodies, joints, and the
        // event/sleep/warm-start state verbatim.
        *self = Self::new(Vec3::new(snap.gravity[0], snap.gravity[1], snap.gravity[2]));
        self.set_substeps(snap.tuning.substeps);
        self.set_velocity_iterations(snap.tuning.velocity_iterations);
        self.set_position_iterations(snap.tuning.position_iterations);
        self.set_contact_softness(snap.tuning.contact_softness);
        self.set_step_budget(
            snap.tuning
                .step_budget
                .map(|(max_pair_substeps, min_substeps)| StepBudget {
                    max_pair_substeps,
                    min_substeps,
                }),
        );
        self.set_max_ccd_substeps(snap.tuning.max_ccd_substeps);
        self.set_broadphase(match snap.tuning.broadphase {
            crate::snapshot::BroadPhaseSnapshot::SweepAndPrune => BroadPhaseKind::SweepAndPrune,
            crate::snapshot::BroadPhaseSnapshot::UniformGrid => BroadPhaseKind::UniformGrid,
            crate::snapshot::BroadPhaseSnapshot::DynamicAabbTree => BroadPhaseKind::DynamicAabbTree,
            crate::snapshot::BroadPhaseSnapshot::Auto => BroadPhaseKind::Auto,
        });
        self.set_solve_path(crate::flags::SolvePath::from(snap.tuning.wide_solver));
        for (i, body) in bodies.into_iter().enumerate() {
            let h = self.add_body(body);
            debug_assert_eq!(h.index(), i);
            self.asleep[i] = snap.asleep[i];
            self.body_moved[i] = snap.moved[i];
            self.island[i] = snap.islands[i];
            self.prev_pose[i] = PrevPose {
                pos: Vec3::new(
                    snap.prev[i].position[0],
                    snap.prev[i].position[1],
                    snap.prev[i].position[2],
                ),
                rot: Quat::from_xyzw(
                    snap.prev[i].rotation[0],
                    snap.prev[i].rotation[1],
                    snap.prev[i].rotation[2],
                    snap.prev[i].rotation[3],
                ),
            };
        }
        for j in &snap.joints {
            let spec = j.spec.to_spec();
            let (a, b) = (j.a as usize, j.b as usize);
            let is_gear = matches!(spec, JointKind::Gear { .. });
            self.joints.push(Joint {
                body_a: BodyHandle::from(a),
                body_b: BodyHandle::from(b),
                kind: spec,
                acc_lin: j.acc_lin,
                acc_ang: j.acc_ang,
                reference_angle: j.reference.angle,
                reference_length: j.reference.length,
                reference_distance: j.reference.distance,
                reference_quat: Quat::from_xyzw(
                    j.reference.rotation[0],
                    j.reference.rotation[1],
                    j.reference.rotation[2],
                    j.reference.rotation[3],
                ),
                reference_anchor_delta: Vec3::new(
                    j.reference.anchor_delta[0],
                    j.reference.anchor_delta[1],
                    j.reference.anchor_delta[2],
                ),
                acc_limit: j.acc_limit,
                acc_dist: j.acc_dist,
                acc_rope: j.acc_rope,
                servo: j.servo.map(crate::snapshot::JointMotorSnapshot::to_motor),
                acc_gear: j.acc_gear,
                gear_mem: j.gear_mem,
                acc_6dof: j.acc_6dof,
            });
            if !is_gear {
                self.joint_pairs.insert((a.min(b), a.max(b)));
            }
        }
        self.contact_touch = snap
            .event_contacts
            .iter()
            .map(|&(a, b)| ((a as usize).min(b as usize), (a as usize).max(b as usize)))
            .collect();
        self.trigger_pairs = snap
            .event_triggers
            .iter()
            .map(|&(a, b)| ((a as usize).min(b as usize), (a as usize).max(b as usize)))
            .collect();
        self.contact_events = snap.pending_contacts.iter().map(|e| e.to_event()).collect();
        self.trigger_events = snap.pending_triggers.iter().map(|e| e.to_event()).collect();
        self.contact_force_events = snap.pending_forces.iter().map(|e| e.to_event()).collect();
        self.warm_impulses = snap
            .warm
            .iter()
            .map(|w| {
                let mut pts = [crate::sequential_impulse::WarmPoint {
                    la: Vec3::ZERO,
                    lb: Vec3::ZERO,
                    normal: Vec3::ZERO,
                    impulse: 0.0,
                }; MAX_MANIFOLD_POINTS];
                for (k, p) in w.points.iter().enumerate() {
                    pts[k] = p.to_point();
                }
                (w.pair, (pts, w.points.len()))
            })
            .collect();
        self.island_timers = snap.island_timers.iter().copied().collect();
        self.island_grace = snap.island_grace.iter().copied().collect();
        Ok(())
    }

    /// Physical joint state in handle order, independent of warm impulses.
    pub(crate) fn joint_snapshots(&self) -> Vec<JointSnapshot> {
        self.joints
            .iter()
            .map(|j| {
                let mut reference = JointReference {
                    angle: crate::invariants::Radians(j.reference_angle),
                    length: crate::invariants::Meters(j.reference_length),
                    distance: crate::invariants::Meters(j.reference_distance),
                    rotation: j.reference_quat,
                    anchor_delta: j.reference_anchor_delta,
                };
                if let JointKind::Gear {
                    joint_a,
                    joint_b,
                    ratio,
                } = j.kind
                {
                    let offset = |h: JointHandle, k: usize| {
                        let side = &self.joints[h.index()];
                        let raw = joints::joint_coordinate(&self.bodies, side).unwrap_or(0.0);
                        let kind = if matches!(side.kind, JointKind::Revolute { .. }) {
                            crate::flags::CoordKind::Angular
                        } else {
                            crate::flags::CoordKind::Linear
                        };
                        let memory = j.gear_mem.map(|(r, c)| (r[k], c[k]));
                        crate::migration::gear_coordinate(raw, kind, memory) - raw
                    };
                    reference.distance.0 -= offset(joint_a, 0) + ratio * offset(joint_b, 1);
                }
                JointSnapshot {
                    a: j.body_a,
                    b: j.body_b,
                    spec: j.kind,
                    reference,
                    servo: j.servo,
                }
            })
            .collect()
    }

    /// Restore physical assembly references after a solver migration.
    pub(crate) fn restore_joint_reference(&mut self, h: JointHandle, r: JointReference) {
        let Some(j) = self.joints.get_mut(h.index()) else {
            return;
        };
        j.reference_angle = r.angle.0;
        j.reference_length = r.length.0;
        j.reference_distance = r.distance.0;
        j.reference_quat = r.rotation;
        j.reference_anchor_delta = r.anchor_delta;
    }

    /// Local-space body read: `h` is an SI-table index, not a global handle.
    /// Thin reinterpretation over [`PhysicsEngine::get_body`]; same lookup.
    pub(crate) fn get_body_local(&self, h: crate::body::LocalSiBody) -> Option<&RigidBody> {
        self.bodies.get(h.index())
    }

    /// Local-space body write: `h` is an SI-table index, not a global handle.
    pub(crate) fn get_body_mut_local(
        &mut self,
        h: crate::body::LocalSiBody,
    ) -> Option<&mut RigidBody> {
        self.bodies.get_mut(h.index())
    }

    /// Local-space wake: `h` is an SI-table index, not a global handle.
    pub(crate) fn wake_body_local(&mut self, h: crate::body::LocalSiBody) {
        if h.index() < self.bodies.len() {
            self.wake_island(h.index());
        }
    }

    /// Local-space baseline restore: `h` is an SI-table index.
    pub(crate) fn restore_body_baseline_local(
        &mut self,
        h: crate::body::LocalSiBody,
        pose: PrevPose,
    ) {
        self.restore_body_baseline(BodyHandle::from(h), pose);
    }

    /// Local-space joint restore: `h` is an SI-table index.
    pub(crate) fn restore_joint_reference_local(
        &mut self,
        h: crate::joint::LocalSiJoint,
        r: JointReference,
    ) {
        self.restore_joint_reference(JointHandle::from(h), r);
    }

    /// Generalized motor override on an assembled joint (see
    /// [`crate::joint::JointMotor`]): `Some` replaces the spec motor for
    /// the drive, `None` clears the override. Only revolute, prismatic and
    /// wheel joints take a motor (the wheel spin is the revolute special
    /// case); every other kind is refused explicitly. Wakes the island so
    /// the drive takes effect immediately.
    ///
    /// # Errors
    ///
    /// [`crate::errors::JointError::UnknownRef`] for a stale handle,
    /// `Unsupported` for a joint kind without a driven axis, `NonFinite`
    /// for an invalid motor (see [`crate::migration::validate_motor`]).
    pub fn set_joint_motor(
        &mut self,
        handle: JointHandle,
        motor: Option<crate::joint::JointMotor>,
    ) -> Result<(), crate::errors::JointError> {
        use crate::errors::JointError;
        if let Some(m) = motor {
            crate::migration::validate_motor(&m)?;
        }
        let Some(j) = self.joints.get(handle.index()) else {
            return Err(JointError::UnknownRef {
                handle: handle.index(),
            });
        };
        if !matches!(
            j.kind,
            JointKind::Revolute { .. } | JointKind::Prismatic { .. } | JointKind::Wheel { .. }
        ) {
            return Err(JointError::Unsupported {
                detail: "set_joint_motor needs a revolute, prismatic or wheel joint".to_string(),
            });
        }
        self.joints[handle.index()].servo = motor;
        let (a, b) = (
            self.joints[handle.index()].body_a,
            self.joints[handle.index()].body_b,
        );
        for h in [a, b] {
            if self.bodies[h.index()].body_type == BodyType::Dynamic {
                self.wake_island(h.index());
            }
        }
        Ok(())
    }

    /// Current motor override of a joint (`None` = spec motor applies, if
    /// any). `None` for an invalid handle.
    pub fn joint_motor(&self, handle: JointHandle) -> Option<crate::joint::JointMotor> {
        self.joints.get(handle.index())?.servo
    }

    /// Restore the motor override after a solver migration (verbatim —
    /// like the assembly references, the drive rides along).
    pub(crate) fn restore_joint_motor(
        &mut self,
        h: JointHandle,
        motor: Option<crate::joint::JointMotor>,
    ) {
        if let Some(j) = self.joints.get_mut(h.index()) {
            j.servo = motor;
        }
    }

    /// Local-space motor restore: `h` is an SI-table index.
    pub(crate) fn restore_joint_motor_local(
        &mut self,
        h: crate::joint::LocalSiJoint,
        motor: Option<crate::joint::JointMotor>,
    ) {
        self.restore_joint_motor(JointHandle::from(h), motor);
    }

    /// Local-space motor override: `h` is an SI-table index.
    pub(crate) fn set_joint_motor_local(
        &mut self,
        h: crate::joint::LocalSiJoint,
        motor: Option<crate::joint::JointMotor>,
    ) -> Result<(), crate::errors::JointError> {
        self.set_joint_motor(JointHandle::from(h), motor)
    }

    /// Local-space joint creation: inputs and output are SI-table indices.
    /// Returns the local joint handle (a lossless `u32` reinterpretation of
    /// the engine's dense [`JointHandle`]).
    pub(crate) fn add_joint_local(
        &mut self,
        a: crate::body::LocalSiBody,
        b: crate::body::LocalSiBody,
        spec: JointKind,
    ) -> Result<crate::joint::LocalSiJoint, crate::errors::JointError> {
        self.add_joint(BodyHandle::from(a), BodyHandle::from(b), spec)
            .map(crate::joint::LocalSiJoint::from)
    }

    /// Empty engine with the default tuning: 12 substeps, 8 velocity
    /// iterations, 4 position iterations, rigid contacts, SIMD-wide solver
    /// on, no gravity until set here. `gravity` is a constant world-space
    /// acceleration (m/s²) applied to dynamic bodies each step.
    pub fn new(gravity: Vec3) -> Self {
        /// Default substeps per `step` call.
        const DEFAULT_SUBSTEPS: u32 = 12;
        /// Default velocity Gauss-Seidel iterations per substep.
        const DEFAULT_VELOCITY_ITERS: u32 = 8;
        /// Default NGS position iterations per substep.
        const DEFAULT_POSITION_ITERS: u32 = 4;
        Self {
            bodies: Vec::new(),
            broadphase: BroadPhaseBackend::new(BroadPhaseKind::UniformGrid),
            gravity,
            substeps: DEFAULT_SUBSTEPS,
            velocity_iterations: DEFAULT_VELOCITY_ITERS,
            position_iterations: DEFAULT_POSITION_ITERS,
            contact_softness: 0.0,
            warm_impulses: FxHashMap::default(),
            island: Vec::new(),
            island_timers: FxHashMap::default(),
            island_grace: FxHashMap::default(),
            body_moved: Vec::new(),
            asleep: Vec::new(),
            prev_pose: Vec::new(),
            saved_driver_vel: Vec::new(),
            movers: Vec::new(),
            joints: Vec::new(),
            joint_pairs: FxHashSet::default(),
            debug_pairs: Vec::new(),
            trigger_pairs: FxHashSet::default(),
            contact_touch: FxHashSet::default(),
            contact_events: Vec::new(),
            contact_force_events: Vec::new(),
            force_peak: FxHashMap::default(),
            scratch_hit_pairs: FxHashSet::default(),
            last_step_timing: StepTiming::default(),
            step_budget: Some(StepBudget::default()),
            last_shed: 0,
            max_ccd_substeps: DEFAULT_MAX_CCD_SUBSTEPS,
            last_ccd_caps: 0,
            trigger_events: Vec::new(),
            wide_solver: crate::flags::SolvePath::Wide,
            scratch_manifolds: Vec::new(),
            scratch_pairs: Vec::new(),
            scratch_bucket_edges: Vec::new(),
            scratch_clamped: Vec::new(),
            scratch_parent: Vec::new(),
            scratch_narrow_shards: NarrowShardPool::default(),
            flat_marks: Vec::new(),
            flat_gen: 0,
            scratch_flat_shards: Vec::new(),
            narrow_cache: FxHashMap::default(),
            sat_cache: DashMap::default(),
            contact_hooks: None,
            hook_read_only: FxHashSet::default(),
            #[cfg(feature = "gpu")]
            gpu_solver: None,
        }
    }

    /// Select the broadphase candidate-pair backend.
    ///
    /// The default is [`BroadPhaseKind::UniformGrid`] (wins the local 10k-body
    /// scene matrix: tiled / giant_floor / sparse / islands / heterogeneous).
    /// [`BroadPhaseKind::SweepAndPrune`] is retained as the compatibility
    /// baseline; [`BroadPhaseKind::DynamicAabbTree`] is experimental.
    /// [`BroadPhaseKind::Auto`] routes analytically between sweep, grid and
    /// tree with hysteresis; see [`BroadPhaseKind::Auto`] for the contract.
    pub fn set_broadphase(&mut self, kind: BroadPhaseKind) {
        if self.broadphase.kind() != kind {
            self.broadphase = BroadPhaseBackend::new(kind);
            self.warm_impulses.clear();
        }
    }

    /// Returns the currently selected broadphase backend.
    pub fn broadphase_kind(&self) -> BroadPhaseKind {
        self.broadphase.kind()
    }

    /// Backend that served the latest update when
    /// [`BroadPhaseKind::Auto`] is selected (any of the three backends,
    /// never `Auto` itself); `None` for explicit backend
    /// selections. Diagnostics for tuning and benchmarks, not part of the
    /// simulation contract.
    pub fn auto_active_broadphase(&self) -> Option<BroadPhaseKind> {
        self.broadphase.auto_active_kind()
    }

    /// Live dynamic AABB tree for read-only query acceleration (R5):
    /// `Some` only while the broadphase is tree-backed (explicit
    /// [`BroadPhaseKind::DynamicAabbTree`] or [`BroadPhaseKind::Auto`]
    /// currently routed there). The reference borrows the trees the pair
    /// pipeline maintains — no copy, no rebuild — and query passes hold it
    /// shared, so no insert/rebalance can interleave with a traversal.
    pub(crate) fn broadphase_tree_for_query(
        &self,
    ) -> Option<&crate::broadphase_tree::DynamicAabbTree> {
        self.broadphase.as_query_tree()
    }

    /// Selects the uniform-grid backend and configures its cell size.
    ///
    /// Smaller cells reduce false candidate pairs at the cost of more cell
    /// bookkeeping. The default grid size is 2.0 world units. This method
    /// resets the warm-start cache because changing the backend is a
    /// diagnostic/configuration boundary between simulation runs.
    pub fn set_uniform_grid_cell_size(&mut self, cell_size: f32) {
        self.broadphase = BroadPhaseBackend::uniform_grid(cell_size);
        self.warm_impulses.clear();
    }

    /// Returns counters from the latest broadphase update.
    ///
    /// The values are diagnostics for tuning and benchmarks; they are not
    /// part of the simulation contract.
    pub fn broadphase_stats(&self) -> BroadPhaseStats {
        self.broadphase.stats()
    }

    /// Wall-clock breakdown of the last completed `step`.
    ///
    /// Diagnostic only: per-substep phases are summed across the substep loop.
    /// Zeroed until the first step runs.
    pub fn step_timing(&self) -> StepTiming {
        self.last_step_timing
    }

    /// Selects the single-point contact-solver path (default: wide).
    /// [`SolvePath::Scalar`] forces the scalar path, which is bit-exact
    /// with the pre-G7 solver for scenes that predate the wide batches.
    pub fn set_solve_path(&mut self, path: crate::flags::SolvePath) {
        self.wide_solver = path;
    }

    /// Boolean-compat wrapper for [`Self::set_solve_path`] (kept for tests).
    pub fn set_wide_solver(&mut self, enabled: bool) {
        self.set_solve_path(crate::flags::SolvePath::from(enabled));
    }

    /// Current single-point solver path.
    pub fn solve_path(&self) -> crate::flags::SolvePath {
        self.wide_solver
    }

    /// Attach a GPU contact solver (G7, `gpu` feature). When the GPU solver
    /// is present, single-point contacts are solved on the GPU and the CPU
    /// wide-path is unused. The GPU solver is a Jacobi/GS hybrid (not
    /// bit-identical to the CPU path); see the `gpu` module docs.
    #[cfg(feature = "gpu")]
    pub fn set_gpu_solver(&mut self, solver: GpuSequentialImpulse) {
        self.gpu_solver = Some(solver);
    }

    /// Number of sub-iterations the solver splits each `step(dt)` into
    /// (default 12). More substeps = more stable stacks, linearly more cost.
    pub fn set_substeps(&mut self, n: u32) {
        self.substeps = n;
    }

    /// Current substep cap (default 12): what `step(dt)` splits the step
    /// into before the adaptive pass and the step budget trim it down.
    pub fn substeps(&self) -> u32 {
        self.substeps
    }

    /// Box3D `b3World_Step(dt, subStepCount)` parity (R7): advance by `dt`
    /// with exactly `n` substeps as the cap. The override is temporary —
    /// restored before return, never stored or snapshotted — so `step(dt)`
    /// keeps the configured default bit-identically afterwards. The
    /// per-body adaptive pass and the step budget trim work on top of `n`
    /// exactly as they work on top of
    /// [`set_substeps`](Self::set_substeps).
    ///
    /// # Errors
    ///
    /// [`StepError::BadSubstepCount`] unless `n` is in `1..=64`
    /// ([`MIN_SUBSTEP_COUNT`]..=[`MAX_SUBSTEP_COUNT`]): out-of-range
    /// counts are refused explicitly, never clamped silently.
    pub fn step_with_substeps(&mut self, dt: f32, substeps: u32) -> Result<(), StepError> {
        check_substep_count(substeps)?;
        let saved = self.substeps;
        self.substeps = substeps;
        PhysicsEngine::step(self, dt);
        self.substeps = saved;
        Ok(())
    }

    /// Worst-case step budget (default: on, see [`StepBudget::default`]).
    /// `None` disables shedding and always runs the speed-requested count.
    /// Some budget sheds substeps — never pairs — when the candidate-pair
    /// count times the requested count exceeds the budget. The shed count of
    /// the last step is observable via [`Self::last_substep_shed`].
    pub fn set_step_budget(&mut self, budget: Option<StepBudget>) {
        self.step_budget = budget;
    }

    /// Substeps shed by the budget on the last completed `step`: requested
    /// minus applied (0 when the full count ran, or when the world slept
    /// through the step). Diagnostics for tuning; not part of the
    /// simulation contract.
    pub fn last_substep_shed(&self) -> u32 {
        self.last_shed
    }

    /// Maximum conservative-advancement iterations per pair in the angular
    /// (nonlinear) CCD sweep (Rapier `max_ccd_substeps` analog). `0`
    /// disables the angular sweep entirely. Exhausted sweeps fall back to
    /// a best-effort clamp and are counted in
    /// [`Self::last_ccd_caps`].
    pub fn set_max_ccd_substeps(&mut self, n: usize) {
        self.max_ccd_substeps = n;
    }

    /// Current angular-sweep iteration budget (default matches the legacy
    /// fixed loop).
    pub fn max_ccd_substeps(&self) -> usize {
        self.max_ccd_substeps
    }

    /// Angular sweeps that exhausted the iteration cap on the last completed
    /// `step`: accepted clamps without the tunnel-free TOI proof (0 when
    /// every applied clamp was proven, or when the world slept through the
    /// step). Diagnostics for tuning; not part of the simulation contract.
    pub fn last_ccd_caps(&self) -> u32 {
        self.last_ccd_caps
    }

    /// Sequential-impulse velocity iterations per substep (default 8).
    pub fn set_velocity_iterations(&mut self, n: u32) {
        self.velocity_iterations = n;
    }

    /// Baumgarte positional-correction iterations per substep (default 4).
    pub fn set_position_iterations(&mut self, n: u32) {
        self.position_iterations = n;
    }

    /// CFM softness scale for the positional pass: 0 (default) = rigid,
    /// larger values spread corrections over more iterations for smoother
    /// but softer penetration recovery.
    pub fn set_contact_softness(&mut self, softness: f32) {
        self.contact_softness = softness;
    }

    /// How many joint constraints are currently live (diagnostics for
    /// joint creation/removal bookkeeping, including gear dependents).
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// (Diagnostics) how many contact manifolds touched the body on the last
    /// substep of the previous step.
    pub fn debug_contact_count(&self, handle: BodyHandle) -> usize {
        let hi = handle.index();
        self.debug_pairs
            .iter()
            .filter(|&&(a, b)| a == hi || b == hi)
            .count()
    }
}

impl SequentialImpulseEngine {
    /// Folds one substep's merged warm impulses into the running per-pair
    /// force peak (`impulse / sub_dt` per pair, maximum wins). Read-only:
    /// the solver never observes this pass.
    fn track_force_peak(&mut self, sub_dt: f32) {
        if !(sub_dt.is_finite() && sub_dt > 0.0) {
            return;
        }
        for (&key, (pts, count)) in self.warm_impulses.iter() {
            let impulse: f32 = pts.iter().take(*count).map(|p| p.impulse).sum();
            let force = impulse / sub_dt;
            if force.is_finite() && force >= 0.0 {
                let peak = self.force_peak.entry(key).or_insert(0.0);
                if force > *peak {
                    *peak = force;
                }
            }
        }
    }

    /// Contact-force emission for one completed step (called at the end of
    /// [`PhysicsEngine::step`](crate::engine::PhysicsEngine::step)): one
    /// report per opted-in pair whose tracked step peak reaches the
    /// smaller enabled threshold, in canonical pair order. Reads the peak
    /// map and the last manifolds only — the solver never observes this
    /// pass, so untracked scenes stay bit-identical.
    fn emit_contact_force_events(&mut self, manifolds: &[Manifold], track: bool) {
        // Opt-in gate (mirrors the per-step decision in `step`): the
        // default scene pays one linear flag scan per step and nothing else.
        if !track {
            return;
        }
        let n = self.bodies.len();
        for m in manifolds {
            let (a, b) = (m.body_a.index(), m.body_b.index());
            if a >= n || b >= n {
                continue;
            }
            let (lo, hi) = (a.min(b), a.max(b));
            let threshold = self.bodies[lo]
                .contact_force_threshold
                .min(self.bodies[hi].contact_force_threshold);
            if !threshold.is_finite() {
                continue;
            }
            // Read-only hook pairs carry zeroed impulses by construction —
            // reporting them would manufacture zero-force events.
            if self.hook_read_only.contains(&(lo, hi)) {
                continue;
            }
            let force = self.force_peak.get(&(lo, hi)).copied().unwrap_or(0.0);
            // NaN forces never report: emission is finite-only.
            if force.is_nan() || force < threshold {
                continue;
            }
            // Deepest point wins (first maximum on ties: manifold order).
            let mut point = m.points[0].world_point;
            let mut deepest = f32::NEG_INFINITY;
            for k in 0..m.point_count.min(MAX_MANIFOLD_POINTS) {
                if m.points[k].penetration > deepest {
                    deepest = m.points[k].penetration;
                    point = m.points[k].world_point;
                }
            }
            self.contact_force_events.push(ContactForceEvent {
                a: BodyHandle::from(lo),
                b: BodyHandle::from(hi),
                force,
                point,
            });
        }
        self.contact_force_events.sort_by_key(|e| (e.a, e.b));
    }
}

impl PhysicsEngine for SequentialImpulseEngine {
    fn step(&mut self, dt: f32) {
        // Same contract as the other engines and the orchestrator: a
        // non-finite or non-positive host delta is a no-op. Integrating it
        // poisons every awake velocity (`gravity * dt` is NaN) once the
        // debug assertions in the integrate path are compiled out.
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        // Driver snapshot FIRST (even on the fast path below): the kinematic
        // step displacement must span exactly one step, and a zero-velocity
        // teleport still counts as driven motion for the wake check.
        let teleported_kinematic = self.snapshot_driver_motion();
        // R2 movers before the fast-path check: engine-driven platforms
        // displace here (carrying and waking passengers through the mover
        // pass), so a displacing mover keeps the world awake exactly like
        // a driver teleport — and a parked one costs nothing.
        let mover_motion = self.apply_movers();
        // G7: a fully sleeping world with no trigger state cannot change —
        // skip the whole substep loop (broadphase re-sort included) instead
        // of paying to rediscover that nothing moves. Trigger-only worlds
        // still run the overlap reconciliation pass below.
        let has_awake_dynamic = self
            .bodies
            .iter()
            .enumerate()
            .any(|(h, b)| b.body_type == BodyType::Dynamic && !self.asleep[h]);
        // Driven kinematics are never asleep by construction: a MOVING one
        // (nonzero velocity field) keeps the world awake, otherwise a
        // kinematic platform would ghost through sleepers with the loop
        // skipped. Parked kinematics (zero velocity, unmoved) cost nothing.
        // Driver contract: above-gate teleports count as driven motion (the
        // implied motion is installed into the fields for the step), and any
        // pose change counts via `teleported_kinematic` below.
        let has_driven_kinematic = self.bodies.iter().any(|b| {
            b.body_type == BodyType::Kinematic
                && (b.velocity.length_squared() + b.angular_velocity.length_squared() > 0.0)
        });
        let has_trigger = self.bodies.iter().any(|body| body.is_trigger);
        if !has_awake_dynamic
            && !has_driven_kinematic
            && !teleported_kinematic
            && !mover_motion
            && !has_trigger
            && self.trigger_pairs.is_empty()
        {
            // Fully sleeping world: no substep loop ran, so no phase work
            // happened this step.
            self.last_step_timing = StepTiming::default();
            self.last_shed = 0;
            self.last_ccd_caps = 0;
            return;
        }
        let eff_substeps_before_budget = self.effective_substeps(dt);
        // Above-gate teleports become the implied motion in the kinematic
        // fields for the rest of the step (margins, CCD, contacts); the
        // driver values come back at step end.
        self.apply_driver_velocity(dt);
        // Diagnostics: contact-manifold partners per body from the last
        // Reuse scratch buffers across substeps; keep capacity across frames.
        self.scratch_manifolds.clear();
        // last_manifolds will alias scratch_manifolds after loop; keep separate copy for debug
        let mut last_manifolds_snapshot: Vec<Manifold> = Vec::new();
        // Broadphase once per step using full dt swept AABBs (conservative
        // superset for all substeps). This cuts 12x broadphase cost for the
        // tiled 10k case: 70ms -> 6ms. Narrow still per substep because
        // contacts depend on moving poses, but candidate list is reused.
        let t0 = Instant::now();
        self.broadphase
            .update(&self.bodies, dt, Some(&self.prev_pose));
        let broad_phase_ms = t0.elapsed().as_secs_f64() * MS_PER_SEC;
        let mut broad_active: Vec<(usize, usize)> = self.broadphase.active().to_vec();
        // H1 contact-hooks filter: cheap pair veto after the broadphase,
        // before the narrowphase (no-op without hooks, order-preserving).
        self.apply_hook_filter(&mut broad_active);
        // Kinematic sweep BEFORE the substep loop: teleported/fast drivers
        // cast their step segment against dynamics, wake victims and
        // transfer the normal approach. Runs on final driver poses, so the
        // woken victims join the substep solves below with adjusted
        // velocities.
        self.solve_kinematic_sweep(dt);
        // Worst-case budget (deterministic): shed substeps against the known
        // pair count — never pairs. `timing.substeps` records the applied
        // count; the shed count is kept in `last_shed` for observability.
        // The count is taken BEFORE the frozen-pair filter below, so the
        // shedding decision never depends on sleep state.
        let (eff_substeps, shed) =
            self.apply_step_budget(eff_substeps_before_budget, broad_active.len());
        self.last_shed = shed;
        // Fresh CCD-cap window for this step: `solve_continuous`
        // accumulates capped sweeps across all substeps below.
        self.last_ccd_caps = 0;
        let sub_dt = dt / eff_substeps as f32;
        let mut timing = StepTiming {
            substeps: eff_substeps,
            broad_phase_ms,
            ..StepTiming::default()
        };
        // B: per-body required substeps for filtering extra substeps.
        let body_needed = self.body_required_substeps(dt);
        let body_needed_opt: Option<&[u32]> = {
            let min_needed = body_needed.iter().copied().min().unwrap_or(0);
            let max_needed = body_needed.iter().copied().max().unwrap_or(0);
            if max_needed - min_needed < SUBSTEP_FILTER_SPREAD {
                None
            } else {
                Some(&body_needed)
            }
        };
        let needs_filter = body_needed_opt.is_some() || !self.joint_pairs.is_empty();
        // Per-substep pair buckets: stable counting sort of the candidate
        // pairs by required substeps, once per step. Slow pairs are then
        // visited only on the substeps that need them instead of being
        // re-checked (and re-rejected) 12x per step. Bucket s holds exactly
        // the pairs the old per-substep filter kept at s, in the same
        // relative order — bit-identical narrowphase input per substep.
        let mut sorted_pairs = std::mem::take(&mut self.scratch_pairs);
        sorted_pairs.clear();
        let mut bucket_edges = std::mem::take(&mut self.scratch_bucket_edges);
        bucket_edges.clear();
        if needs_filter {
            let eff = eff_substeps as usize;
            sorted_pairs.reserve(broad_active.len());
            // Count required-substep classes, clamped to eff (over-budget
            // pairs are needed on every substep that runs, like before).
            // bucket_edges doubles as the counter array, then the placement
            // cursors, then the final boundary table — no per-step
            // allocation. Class of a pair: jointed pairs are excluded on
            // every substep (class 0 = never visited); without per-body
            // data every pair needs every substep (class eff).
            bucket_edges.resize(eff + 1, 0);
            // Fast path: no joints — skip the per-pair set lookup entirely.
            if self.joint_pairs.is_empty() {
                if let Some(req) = body_needed_opt {
                    for &(a, b) in &broad_active {
                        bucket_edges[(req[a].max(req[b]) as usize).min(eff)] += 1;
                    }
                } else {
                    bucket_edges[eff] = broad_active.len();
                }
            } else {
                for &(a, b) in &broad_active {
                    let c = if self.joint_pairs.contains(&(a, b)) {
                        0
                    } else if let Some(req) = body_needed_opt {
                        (req[a].max(req[b]) as usize).min(eff)
                    } else {
                        eff
                    };
                    bucket_edges[c] += 1;
                }
            }
            // Exclusive prefix sums: starts[c] = #{k < c}. Forward stable
            // placement advances them into end offsets #{k <= c} — which
            // are exactly the bucket boundaries S_s (bucket s = suffix
            // [S_s..N) of pairs with class > s, original relative order).
            let mut cursor = 0;
            for slot in bucket_edges.iter_mut() {
                let n = *slot;
                *slot = cursor;
                cursor += n;
            }
            sorted_pairs.resize(broad_active.len(), (0, 0));
            if self.joint_pairs.is_empty() {
                if let Some(req) = body_needed_opt {
                    for &(a, b) in &broad_active {
                        let c = (req[a].max(req[b]) as usize).min(eff);
                        let w = bucket_edges[c];
                        sorted_pairs[w] = (a, b);
                        bucket_edges[c] = w + 1;
                    }
                } else {
                    // No per-body data, no joints: every pair needs every
                    // substep. Counting put the whole list in class eff at
                    // offset 0; the prefix starts are already the correct
                    // boundaries (S_s = 0 for s < eff).
                    sorted_pairs.copy_from_slice(&broad_active);
                }
            } else {
                // Jointed pairs (class 0) are skipped: their slots stay
                // holes, so the skipped count must become bucket boundary
                // S_0 below — otherwise substep 0 reads the holes as (0, 0)
                // self-pairs (a hard crash in the island solver whenever
                // body 0 is dynamic).
                let mut skipped = 0;
                for &(a, b) in &broad_active {
                    let c = if self.joint_pairs.contains(&(a, b)) {
                        0
                    } else if let Some(req) = body_needed_opt {
                        (req[a].max(req[b]) as usize).min(eff)
                    } else {
                        eff
                    };
                    if c == 0 {
                        skipped += 1;
                        continue;
                    }
                    let w = bucket_edges[c];
                    sorted_pairs[w] = (a, b);
                    bucket_edges[c] = w + 1;
                }
                bucket_edges[0] = skipped;
            }
        }
        // Scheduler narrowphase shard pool, taken once for the whole substep
        // loop and restored after (same discipline as the other scratch).
        let mut narrow_shards = std::mem::take(&mut self.scratch_narrow_shards);
        // Flat singleton fast path (100k-tiled): when every dynamic body
        // appears in at most one candidate pair, islands are all
        // single-manifold by construction — solve over coarse shards with
        // the same kernels (see `solve_flat_velocity`). Decided once per
        // step from the broadphase pairs; snapshot/confluence scenes stay
        // below the pair gate and keep the island path exactly.
        // Eligibility runs on the hook-filtered pairs (hook vetoes produce
        // no contacts, so they cannot break singleton disjointness), before
        // the frozen-pair filter below, so the path decision never depends
        // on sleep state. Contact hooks force the island path: flat shards
        // and the GPU batch path do not implement overrides (see the
        // `hooks` module docs).
        let use_flat = self.flat_singleton_eligible(&broad_active) && !self.has_contact_hooks();
        // Frozen-pair filter: both-asleep pairs are skipped by the
        // narrowphase unconditionally (see `narrow_pair`), so dropping them
        // here only removes per-substep rejections — the manifold stream
        // (and its order) is bit-identical. `retain` preserves the relative
        // order, and the budget + flat decisions above already ran on the
        // full set, so shedding and the solve path are sleep-independent.
        broad_active.retain(|&(a, b)| !(self.asleep[a] && self.asleep[b]));
        let mut flat_shards = std::mem::take(&mut self.scratch_flat_shards);
        // Contact-force tracking gate, decided once per step: the default
        // scene (no finite threshold) pays one linear flag scan per step
        // and skips every per-substep/per-pair read below.
        let track_forces = self.bodies.iter().any(|b| b.contact_force_events_enabled());
        self.force_peak.clear();
        for s in 0..eff_substeps {
            // Per-step hit dedupe window opens here (see collect_active).
            if s == 0 {
                self.scratch_hit_pairs.clear();
            }
            // Box3D stage order: solve velocities BEFORE moving positions, so
            // a resting contact kills gravity's velocity gain in the same
            // substep instead of letting the body free-fall and snapping it
            // back (the snap is an inelastic collision and bleeds energy).
            self.integrate_velocities(sub_dt);
            // Narrowphase input: the full candidate list on homogeneous
            // scenes, the pre-bucketed class suffix on mixed ones (see the
            // counting sort above the substep loop).
            let t0 = Instant::now();
            let mut manifolds_buf = std::mem::take(&mut self.scratch_manifolds);
            manifolds_buf.clear();
            if needs_filter {
                // Bucket s = suffix of pairs with class > s: exactly the
                // set the old per-substep filter kept, same order.
                let start = bucket_edges[s as usize];
                detect_collisions_into_with_cache(
                    &self.bodies,
                    &sorted_pairs[start..],
                    &self.asleep,
                    sub_dt,
                    &mut manifolds_buf,
                    None,
                    s,
                    &mut self.narrow_cache,
                    Some(&self.sat_cache),
                    &mut narrow_shards,
                );
            } else {
                detect_collisions_into_with_cache(
                    &self.bodies,
                    &broad_active,
                    &self.asleep,
                    sub_dt,
                    &mut manifolds_buf,
                    None,
                    s,
                    &mut self.narrow_cache,
                    Some(&self.sat_cache),
                    &mut narrow_shards,
                );
            }
            timing.narrow_phase_ms += t0.elapsed().as_secs_f64() * MS_PER_SEC;
            // Restitution is one-shot per step, evaluated on the first substep.
            let t0 = Instant::now();
            let gate = crate::flags::RestitutionGate::from(s == 0);
            let mut islands: Vec<IslandWork> = Vec::new();
            if use_flat {
                self.solve_flat_velocity(&manifolds_buf, gate, sub_dt, dt, &mut flat_shards);
            } else {
                islands = self.solve_contacts_velocity(&mut manifolds_buf, gate, sub_dt, dt);
            }
            self.solve_joints_velocity(sub_dt);
            // Continuous pass on the solver-adjusted velocities: clamp fast
            // movers to their first impact and keep them there this substep.
            // Reuse a local clamped buffer (capacity kept across substeps via scratch).
            let mut clamped_buf = std::mem::take(&mut self.scratch_clamped);
            clamped_buf.clear();
            clamped_buf.resize(self.bodies.len(), false);
            clamped_buf.fill(false);
            self.solve_continuous(sub_dt, &mut clamped_buf);
            self.integrate_positions(sub_dt, &clamped_buf);
            if use_flat {
                self.solve_flat_position(&mut flat_shards, dt);
            } else {
                self.solve_contacts_position(&mut islands, dt);
            }
            self.solve_joints_position();
            timing.solver_ms += t0.elapsed().as_secs_f64() * MS_PER_SEC;
            // Snapshot last manifolds for island rebuild; clone only once per frame
            if s + 1 == eff_substeps {
                last_manifolds_snapshot.clear();
                last_manifolds_snapshot.extend_from_slice(&manifolds_buf);
            }
            self.scratch_clamped = clamped_buf;
            self.scratch_manifolds = manifolds_buf;
            // Contact-force peak tracking (read-only over the merged warm
            // cache): the step peak of per-substep force, so a transient
            // impact that resolves mid-step still reports. Gated above —
            // untracked scenes never enter this loop body.
            if track_forces {
                self.track_force_peak(sub_dt);
            }
        }
        self.scratch_narrow_shards = narrow_shards;
        self.scratch_pairs = sorted_pairs;
        self.scratch_bucket_edges = bucket_edges;
        self.scratch_flat_shards = flat_shards;
        // Diagnostics: contact-manifold partners per body from the last
        // substep (drives sleep/island debugging; tiny flat copy).
        self.debug_pairs.clear();
        self.debug_pairs.extend(
            last_manifolds_snapshot
                .iter()
                .map(|m| (m.body_a.index(), m.body_b.index())),
        );
        let t_island = Instant::now();
        self.rebuild_islands(&last_manifolds_snapshot);
        self.update_sleep(dt);
        timing.island_ms += t_island.elapsed().as_secs_f64() * MS_PER_SEC;
        self.last_step_timing = timing;

        // Rebuild the broadphase at the completed poses so trigger events
        // describe the state visible after this whole physics step, not the
        // state from before the final substep's integration. The rebuild
        // runs only when something can observe it (a trigger body is live
        // or a stale trigger overlap is pending): every backend keys its
        // incremental baseline off exact swept-box equality, so skipping
        // the rebuild on triggerless worlds only moves the baseline from
        // the tight end-of-step boxes to the swept start-of-step boxes —
        // unmoved bodies still compare clean (same boxes, retained pairs
        // stay valid) and moved bodies still compare dirty. The next step's
        // swept update emits the same pair set either way.
        let t_trigger = Instant::now();
        if has_trigger || !self.trigger_pairs.is_empty() {
            self.broadphase
                .update(&self.bodies, 0.0, Some(&self.prev_pose));
            let current_triggers = detect_trigger_overlaps(&self.bodies, self.broadphase.active());
            // Intersection-pair filter: sensor veto before event
            // reconciliation (no-op without hooks, order-preserving).
            let current_triggers = self.apply_hook_intersection_filter(current_triggers);
            let previous_triggers = std::mem::take(&mut self.trigger_pairs);
            self.trigger_pairs = update_trigger_events(
                &previous_triggers,
                current_triggers,
                &mut self.trigger_events,
            );
        }
        // Solid-contact begin/end reconcile (gameplay events): touching =
        // penetration above the begin slop in the last substep's manifolds.
        // Frozen pairs keep their prior state (sleep emits no transitions); stale handles
        // (removed bodies) are dropped — removal clears the set anyway.
        let n = self.bodies.len();
        let mut current_touch: FxHashSet<(usize, usize)> = FxHashSet::default();
        for m in &last_manifolds_snapshot {
            let (a, b) = (m.body_a.index(), m.body_b.index());
            if a >= n || b >= n {
                continue;
            }
            let mut touching = false;
            for k in 0..m.point_count {
                if m.points[k].penetration > -CONTACT_BEGIN_SLOP {
                    touching = true;
                    break;
                }
            }
            if touching {
                current_touch.insert((a.min(b), a.max(b)));
            }
        }
        for &(a, b) in &self.contact_touch {
            if a < n && b < n && self.asleep[a] && self.asleep[b] {
                current_touch.insert((a, b));
            }
        }
        let previous_touch = std::mem::replace(&mut self.contact_touch, current_touch);
        let mut begins: Vec<(usize, usize)> = self
            .contact_touch
            .difference(&previous_touch)
            .copied()
            .collect();
        let mut ends: Vec<(usize, usize)> = previous_touch
            .difference(&self.contact_touch)
            .copied()
            .collect();
        begins.sort_unstable();
        ends.sort_unstable();
        for (a, b) in begins {
            self.contact_events.push(ContactEvent {
                body_a: BodyHandle::from(a),
                body_b: BodyHandle::from(b),
                kind: ContactEventKind::Begin,
            });
        }
        for (a, b) in ends {
            self.contact_events.push(ContactEvent {
                body_a: BodyHandle::from(a),
                body_b: BodyHandle::from(b),
                kind: ContactEventKind::End,
            });
        }
        self.contact_events
            .sort_by_key(|e| (e.body_a.min(e.body_b), e.body_a.max(e.body_b)));
        // Contact-force reports (Rapier `CONTACT_FORCE_EVENTS` parity):
        // one report per opted-in pair whose tracked step peak reaches
        // its threshold, in canonical pair order. The opt-in scan is the
        // only work on the default path (no body enabled): per-pair peak
        // reads happen strictly below it, so untracked scenes pay one
        // linear flag scan per step and nothing else.
        self.emit_contact_force_events(&last_manifolds_snapshot, track_forces);
        self.last_step_timing.trigger_ms += t_trigger.elapsed().as_secs_f64() * MS_PER_SEC;
        // Hand the velocity fields back to the driver: solver impulses must
        // never corrupt driver-owned kinematic state across steps. Then
        // refresh the step-start baseline for the next step's teleport cover.
        self.restore_driver_velocity();
        self.sync_prev_pose();
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        let handle = BodyHandle::from(self.bodies.len());
        let island_id = if body.body_type == BodyType::Dynamic {
            handle.as_u32()
        } else {
            u32::MAX
        };
        // World-scale sleep foundation: statics are born asleep (they never
        // move, so every frozen-pair skip in narrowphase/scheduler applies
        // to them from step one). Kinematics stay awake — driven bodies
        // must wake sleepers on contact, never ghost through them.
        let born_asleep = body.body_type == BodyType::Static;
        self.prev_pose.push(PrevPose {
            pos: body.position,
            rot: body.orientation,
        });
        self.bodies.push(body);
        self.island.push(island_id);
        self.asleep.push(born_asleep);
        // Born pristine: the frozen fast track (see `update_sleep`) may
        // only fire for bodies that never moved, so the flag starts clear
        // and is set on the first above-gate motion.
        self.body_moved.push(false);
        handle
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        if handle.index() < self.bodies.len() {
            let last_idx = self.bodies.len() - 1;
            let last = BodyHandle::from(last_idx);
            let hi = handle.index();
            let previous_triggers = std::mem::take(&mut self.trigger_pairs);
            let mut removed_triggers: Vec<(usize, usize)> = Vec::new();
            let mut remapped_triggers: FxHashSet<(usize, usize)> = FxHashSet::default();
            for (body_a, body_b) in previous_triggers {
                if body_a == hi || body_b == hi {
                    removed_triggers.push((body_a, body_b));
                    continue;
                }
                let map = |body: usize| if body == last_idx { hi } else { body };
                let a = map(body_a);
                let b = map(body_b);
                remapped_triggers.insert((a.min(b), a.max(b)));
            }
            removed_triggers.sort_unstable();
            for (body_a, body_b) in removed_triggers {
                self.trigger_events.push(TriggerEvent {
                    body_a: BodyHandle::from(body_a),
                    body_b: BodyHandle::from(body_b),
                    kind: TriggerEventKind::Exited,
                });
            }
            self.trigger_pairs = remapped_triggers;
            self.bodies.swap_remove(handle.index());
            self.island.swap_remove(handle.index());
            self.asleep.swap_remove(handle.index());
            self.body_moved.swap_remove(handle.index());
            self.prev_pose.swap_remove(handle.index());
            // Mover handles are body indices: drop movers driving the
            // removed body, remap the swapped-in tail onto the freed slot
            // (mover handles past a dropped mover shift — same convention
            // as bodies).
            let mut mi = 0;
            while mi < self.movers.len() {
                if self.movers[mi].body == handle {
                    self.movers.swap_remove(mi);
                } else {
                    if self.movers[mi].body == last {
                        self.movers[mi].body = handle;
                    }
                    mi += 1;
                }
            }
            // Drop joints touching the removed body (gears die with their
            // referenced joints inside the rebuild — dangling joint indices
            // are never kept); remap the swapped-in body's index in the
            // survivors first.
            let drop_j: Vec<bool> = self
                .joints
                .iter()
                .map(|j| j.body_a == handle || j.body_b == handle)
                .collect();
            for j in self.joints.iter_mut() {
                if j.body_a == last {
                    j.body_a = handle;
                }
                if j.body_b == last {
                    j.body_b = handle;
                }
            }
            rebuild_joints(&mut self.joints, &mut self.joint_pairs, drop_j);
            // Body handles are identities: the swap remaps the tail body's
            // index, so contact state keyed by handles is no longer valid.
            // Drop it (Box2D parity: events may invalidate on destroy) —
            // surviving contacts re-begin on the next step.
            self.contact_touch.clear();
            self.contact_events.clear();
            self.contact_force_events.clear();
            self.force_peak.clear();
            // swap_remove shifts the last body's index; warm-start keys are
            // body indices, so the cache is no longer valid.
            self.warm_impulses.clear();
        }
    }

    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Result<JointHandle, crate::errors::JointError> {
        use crate::errors::JointError;
        crate::migration::validate_joint(&kind)?;
        let (ia, ib) = (usize::from(body_a), usize::from(body_b));
        if ia == ib {
            return Err(JointError::SelfJoint { handle: ia });
        }
        if ia >= self.bodies.len() || ib >= self.bodies.len() {
            return Err(JointError::InvalidHandles { a: ia, b: ib });
        }
        // Resolve frames + assembly references once (shared with AVBD —
        // see `joint::resolve_joint`): axis normalization, axle
        // orthogonalization and Box2D-style reference capture live there.
        // Gear coordinates other joints (needs the joint list) and is
        // handled below; it resolves to `None` here.
        let resolved = if matches!(kind, JointKind::Gear { .. }) {
            None
        } else {
            crate::joint::resolve_joint(
                &kind,
                self.bodies[ia].position,
                self.bodies[ia].orientation,
                self.bodies[ib].position,
                self.bodies[ib].orientation,
            )
        };
        // Write the normalized frames back into the stored kind (the
        // solvers read axes from `JointKind`).
        let kind = match kind {
            JointKind::Revolute {
                local_anchor_a,
                local_anchor_b,
                limit,
                motor,
                ..
            } => {
                let Some(r) = resolved.as_ref() else {
                    return Err(JointError::BadAxis {
                        detail: "joint frame resolve failed".into(),
                    });
                };
                JointKind::Revolute {
                    local_anchor_a,
                    local_anchor_b,
                    local_axis_a: r.ax_a,
                    local_axis_b: r.ax_b,
                    limit,
                    motor,
                }
            }
            JointKind::Prismatic {
                local_anchor_a,
                local_anchor_b,
                limit,
                motor,
                ..
            } => {
                let Some(r) = resolved.as_ref() else {
                    return Err(JointError::BadAxis {
                        detail: "joint frame resolve failed".into(),
                    });
                };
                JointKind::Prismatic {
                    local_anchor_a,
                    local_anchor_b,
                    local_axis_a: r.ax_a,
                    local_axis_b: r.ax_b,
                    limit,
                    motor,
                }
            }
            JointKind::Wheel {
                local_anchor_a,
                local_anchor_b,
                suspension,
                motor,
                ..
            } => {
                let Some(r) = resolved.as_ref() else {
                    return Err(JointError::BadAxis {
                        detail: "joint frame resolve failed".into(),
                    });
                };
                JointKind::Wheel {
                    local_anchor_a,
                    local_anchor_b,
                    local_suspension_a: r.ax_a,
                    local_suspension_b: r.ax_b,
                    local_axle_a: r.bx_a,
                    local_axle_b: r.bx_b,
                    suspension,
                    motor,
                }
            }
            other => other,
        };
        // Gear validation: both references must exist and coordinate a
        // revolute or prismatic joint. The gear holds no bodies of its own —
        // keep the caller's distinct metadata endpoints. The gear pass
        // and island union resolve all four coordinate participants.
        let is_gear = matches!(kind, JointKind::Gear { .. });
        if let JointKind::Gear {
            joint_a,
            joint_b,
            ratio,
        } = &kind
        {
            use crate::errors::JointError;
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
            if !matches!(
                ja.kind,
                JointKind::Revolute { .. } | JointKind::Prismatic { .. }
            ) || !matches!(
                jb.kind,
                JointKind::Revolute { .. } | JointKind::Prismatic { .. }
            ) {
                return Err(JointError::Unsupported {
                    detail: "gear must coordinate revolute/prismatic joints".to_string(),
                });
            }
        }
        // A new joint on a sleeping island changes its constraint set — wake
        // it so the joint state can settle coherently.
        for (h, i) in [(body_a, ia), (body_b, ib)] {
            if self.bodies[i].body_type == BodyType::Dynamic
                && self.asleep.get(i).copied().unwrap_or(false)
            {
                self.wake_island(usize::from(h));
            }
        }
        // Gears coordinate other joints instead of constraining a body pair —
        // the underlying joints already carry the no-collide entry.
        if !is_gear {
            self.joint_pairs.insert((ia.min(ib), ia.max(ib)));
        }
        // Limits/motors measure travel from the assembly pose (Box2D
        // `m_referenceAngle`): captured in `resolved` above.
        // Wheels capture the same twist about their axle for the motor.
        let reference_angle = resolved.map(|r| r.ref_angle).unwrap_or(0.0);
        // Prismatic limits and the wheel spring measure anchor separation
        // along the slide/suspension axis from the assembly pose.
        let reference_length = resolved.map(|r| r.ref_length).unwrap_or(0.0);
        // The distance rod keeps its assembly anchor distance; the gear
        // captures its constraint constant.
        let reference_distance = match &kind {
            JointKind::Distance { .. } | JointKind::Rope { .. } | JointKind::Spring { .. } => {
                resolved.map(|r| r.ref_distance).unwrap_or(0.0)
            }
            JointKind::Gear {
                joint_a,
                joint_b,
                ratio,
            } => {
                let ca = crate::engine::joints::joint_coordinate(
                    &self.bodies,
                    &self.joints[joint_a.index()],
                )
                .unwrap_or(0.0);
                let cb = crate::engine::joints::joint_coordinate(
                    &self.bodies,
                    &self.joints[joint_b.index()],
                )
                .unwrap_or(0.0);
                ca + ratio * cb
            }
            _ => 0.0,
        };
        // Fixed, wheel and six-DOF angular locks measure drift from the
        // assembly relative orientation.
        let reference_quat = resolved.map(|r| r.ref_quat).unwrap_or(Quat::IDENTITY);
        // Fixed, wheel and six-DOF locked-axis position steps measure the
        // anchor separation from the assembly one (in A's frame), so offset
        // assemblies hold instead of collapsing into coincidence.
        let reference_anchor_delta = resolved.map(|r| r.ref_anchor_delta).unwrap_or(Vec3::ZERO);
        let mut joint = Joint::new(body_a, body_b, kind);
        joint.reference_angle = reference_angle;
        joint.reference_length = reference_length;
        joint.reference_distance = reference_distance;
        joint.reference_quat = reference_quat;
        joint.reference_anchor_delta = reference_anchor_delta;
        self.joints.push(joint);
        Ok(JointHandle::from(self.joints.len() - 1))
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        if handle.index() >= self.joints.len() {
            return;
        }
        // Joint indices shift on removal, so gear references (joint indices)
        // are remapped through a dense rebuild: the removed joint goes, gears
        // pointing at it go with it (dangling references are never kept),
        // survivors keep pointing at the same joints.
        let old_len = self.joints.len();
        let mut drop_j = vec![false; old_len];
        drop_j[handle.index()] = true;
        for (oi, j) in self.joints.iter().enumerate() {
            if drop_j[oi] {
                continue;
            }
            if let JointKind::Gear {
                joint_a, joint_b, ..
            } = &j.kind
                && (*joint_a == handle || *joint_b == handle)
            {
                drop_j[oi] = true;
            }
        }
        rebuild_joints(&mut self.joints, &mut self.joint_pairs, drop_j);
    }

    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        self.bodies.get(usize::from(handle))
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        self.bodies.get_mut(usize::from(handle))
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Result<Option<RaycastHit>, QueryError> {
        crate::errors::check_ray_input(ray.origin, ray.direction, max_dist)?;
        let mut closest: Option<RaycastHit> = None;
        for handle in 0..self.bodies.len() {
            if let Some(hit) = self.raycast_body(&ray, BodyHandle::from(handle), max_dist) {
                match &closest {
                    Some(best) if hit.distance < best.distance => closest = Some(hit),
                    None => closest = Some(hit),
                    _ => {}
                }
            }
        }
        Ok(closest)
    }

    /// Honest shapecast (G6): conservative advancement over exact pairwise
    /// shape distances (`distance.rs`). Tunnel-free for any cast length and
    /// any target thickness; the hit distance is the true first touch, not
    /// the nearest fixed sample. Rotation of the cast shape is fixed during
    /// the sweep (linear cast).
    fn shapecast(&self, shape: &Shape, from: Vec3, to: Vec3) -> Option<RaycastHit> {
        let mover = distance::ShapeRef {
            shape,
            pos: from,
            rot: Quat::IDENTITY,
        };
        let targets = self.bodies.iter().enumerate().map(|(h, b)| {
            (
                BodyHandle::from(h),
                distance::ShapeRef {
                    shape: &b.shape,
                    pos: b.position,
                    rot: b.orientation,
                },
            )
        });
        distance::cast_shape(mover, to - from, targets).map(|h| RaycastHit {
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

    fn drain_contact_force_events(&mut self) -> Vec<ContactForceEvent> {
        std::mem::take(&mut self.contact_force_events)
    }

    fn wake_body(&mut self, handle: BodyHandle) {
        if handle.index() < self.bodies.len() {
            self.wake_island(handle.index());
        }
    }
}
