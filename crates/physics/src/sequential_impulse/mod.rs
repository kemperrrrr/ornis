//! Sequential-impulse (projected Gauss-Seidel) solver: thin step orchestrator.
//! Owns [`SequentialImpulseEngine`] (structure, constructors/setters and the
//! [`PhysicsEngine::step`] pipeline); substep passes live in `step`, the
//! narrowphase in `narrow`, caches in `caches`, island sleep in `sleep`,
//! queries in `queries`, events in `events` and scalar helpers in `math`.

mod caches;
mod contacts;
mod events;
mod islands;
pub mod joints;
mod math;
mod narrow;
mod queries;
mod sleep;
mod step;

pub(crate) use crate::engine::{Manifold, ManifoldPoint, PhysicsEngine};
pub use caches::{NarrowShardPool, SatCache, SatCacheEntry};
pub use math::{
    apply_impulse, effective_mass, inv_inertia_axis, mul_inv_inertia, point_velocity,
    solve_normal_block, solve_small,
};
pub use narrow::{box_manifold, detect_collisions_into, obb_sat};
pub(crate) use queries::raycast_shape_hit;
pub use queries::{
    ContinuousHit, ccd_impact_velocity, find_angular_continuous_hit, kinematic_cast,
    remove_angular_approach, sweep_gap,
};
pub use step::ManifoldState;

use self::caches::*;
use self::events::*;
use self::math::*;
use self::narrow::*;
use self::step::*;

use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use dashmap::DashMap;

use glam::{Quat, Vec3};

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::broadphase::{
    BroadPhase, BroadPhaseBackend, BroadPhaseKind, BroadPhaseStats, PrevPose, StepBudget,
    StepTiming,
};
use crate::distance;
#[cfg(feature = "gpu")]
use crate::gpu::GpuSequentialImpulse;
use crate::joint::{Joint, JointHandle, JointKind};
use crate::math::{Ray, RaycastHit};
use crate::migration::{JointReference, JointSnapshot};
use crate::shape::Shape;
use crate::trigger::{
    CONTACT_BEGIN_SLOP, ContactEvent, ContactEventKind, TriggerEvent, TriggerEventKind,
};
use crate::wide::{SolverStep, build_solver_steps};

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
    /// Enter/exit transitions waiting for the caller to drain.
    trigger_events: Vec<TriggerEvent>,
    /// G7: enable SIMD-wide contact solver for single-point manifolds.
    /// Default true. Set to false for bit-exact scalar reproduction.
    wide_solver: bool,
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
    /// G7: optional GPU contact solver (gpu feature). When attached,
    /// single-point manifolds are solved on the GPU instead of the CPU
    /// wide path; multi-point manifolds stay on the CPU island path.
    #[cfg(feature = "gpu")]
    gpu_solver: Option<GpuSequentialImpulse>,
    narrow_cache: FxHashMap<(usize, usize), NarrowCacheEntry>,
    sat_cache: SatCache,
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
        .map(|j| (j.body_a.min(j.body_b), j.body_a.max(j.body_b)))
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
    pub(crate) fn restore_body_baseline(&mut self, h: usize, pose: PrevPose) {
        if let Some(old) = self.prev_pose.get_mut(h) {
            *old = pose;
        }
    }

    /// Completed-step event baseline for transparent solver migration.
    pub(crate) fn event_state(&self) -> crate::migration::EventState {
        crate::migration::EventState {
            contacts: self.contact_touch.iter().copied().collect(),
            triggers: self.trigger_pairs.iter().copied().collect(),
        }
    }

    /// Seed a rebuilt solver without manufacturing a new contact/trigger begin.
    pub(crate) fn restore_event_state(&mut self, state: crate::migration::EventState) {
        self.contact_touch = state.contacts.into_iter().collect();
        self.trigger_pairs = state.triggers.into_iter().collect();
    }

    /// Physical joint state in handle order, independent of warm impulses.
    pub(crate) fn joint_snapshots(&self) -> Vec<JointSnapshot> {
        self.joints
            .iter()
            .map(|j| {
                let mut reference = JointReference {
                    angle: j.reference_angle,
                    length: j.reference_length,
                    distance: j.reference_distance,
                    rotation: j.reference_quat,
                    anchor_delta: j.reference_anchor_delta,
                };
                if let JointKind::Gear {
                    joint_a,
                    joint_b,
                    ratio,
                } = j.kind
                {
                    let offset = |h: usize, k: usize| {
                        let side = &self.joints[h];
                        let raw = joints::joint_coordinate(&self.bodies, side).unwrap_or(0.0);
                        let angular = matches!(side.kind, JointKind::Revolute { .. });
                        let memory = j.gear_mem.map(|(r, c)| (r[k], c[k]));
                        crate::migration::gear_coordinate(raw, angular, memory) - raw
                    };
                    reference.distance -= offset(joint_a, 0) + ratio * offset(joint_b, 1);
                }
                JointSnapshot {
                    a: j.body_a,
                    b: j.body_b,
                    spec: j.kind,
                    reference,
                }
            })
            .collect()
    }

    /// Restore physical assembly references after a solver migration.
    pub(crate) fn restore_joint_reference(&mut self, h: JointHandle, r: JointReference) {
        let Some(j) = self.joints.get_mut(h) else {
            return;
        };
        j.reference_angle = r.angle;
        j.reference_length = r.length;
        j.reference_distance = r.distance;
        j.reference_quat = r.rotation;
        j.reference_anchor_delta = r.anchor_delta;
    }

    /// Empty engine with the default tuning: 12 substeps, 8 velocity
    /// iterations, 4 position iterations, rigid contacts, SIMD-wide solver
    /// on, no gravity until set here. `gravity` is a constant world-space
    /// acceleration (m/s²) applied to dynamic bodies each step.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            bodies: Vec::new(),
            broadphase: BroadPhaseBackend::new(BroadPhaseKind::UniformGrid),
            gravity,
            substeps: 12,
            velocity_iterations: 8,
            position_iterations: 4,
            contact_softness: 0.0,
            warm_impulses: FxHashMap::default(),
            island: Vec::new(),
            island_timers: FxHashMap::default(),
            asleep: Vec::new(),
            prev_pose: Vec::new(),
            saved_driver_vel: Vec::new(),
            joints: Vec::new(),
            joint_pairs: FxHashSet::default(),
            debug_pairs: Vec::new(),
            trigger_pairs: FxHashSet::default(),
            contact_touch: FxHashSet::default(),
            contact_events: Vec::new(),
            scratch_hit_pairs: FxHashSet::default(),
            last_step_timing: StepTiming::default(),
            step_budget: Some(StepBudget::default()),
            last_shed: 0,
            trigger_events: Vec::new(),
            wide_solver: true,
            scratch_manifolds: Vec::new(),
            scratch_pairs: Vec::new(),
            scratch_bucket_edges: Vec::new(),
            scratch_clamped: Vec::new(),
            scratch_parent: Vec::new(),
            scratch_narrow_shards: NarrowShardPool::default(),
            narrow_cache: FxHashMap::default(),
            sat_cache: DashMap::default(),
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

    /// Toggle the G7 SIMD-wide contact solver (default: enabled). Disabling
    /// it forces the scalar single-point path, which is bit-exact with the
    /// pre-G7 solver for scenes that predate the wide batches.
    pub fn set_wide_solver(&mut self, enabled: bool) {
        self.wide_solver = enabled;
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
        self.debug_pairs
            .iter()
            .filter(|&&(a, b)| a == handle || b == handle)
            .count()
    }
}

impl PhysicsEngine for SequentialImpulseEngine {
    fn step(&mut self, dt: f32) {
        // Driver snapshot FIRST (even on the fast path below): the kinematic
        // step displacement must span exactly one step, and a zero-velocity
        // teleport still counts as driven motion for the wake check.
        let teleported_kinematic = self.snapshot_driver_motion();
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
            && !has_trigger
            && self.trigger_pairs.is_empty()
        {
            // Fully sleeping world: no substep loop ran, so no phase work
            // happened this step.
            self.last_step_timing = StepTiming::default();
            self.last_shed = 0;
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
        let broad_phase_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let broad_active: Vec<(usize, usize)> = self.broadphase.active().to_vec();
        // Kinematic sweep BEFORE the substep loop: teleported/fast drivers
        // cast their step segment against dynamics, wake victims and
        // transfer the normal approach. Runs on final driver poses, so the
        // woken victims join the substep solves below with adjusted
        // velocities.
        self.solve_kinematic_sweep(dt);
        // Worst-case budget (deterministic): shed substeps against the known
        // pair count — never pairs. `timing.substeps` records the applied
        // count; the shed count is kept in `last_shed` for observability.
        let (eff_substeps, shed) =
            self.apply_step_budget(eff_substeps_before_budget, broad_active.len());
        self.last_shed = shed;
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
            if max_needed - min_needed < 4 {
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
            timing.narrow_phase_ms += t0.elapsed().as_secs_f64() * 1000.0;
            // Restitution is one-shot per step, evaluated on the first substep.
            let t0 = Instant::now();
            let mut islands = self.solve_contacts_velocity(&manifolds_buf, s == 0, sub_dt, dt);
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
            self.solve_contacts_position(&mut islands, dt);
            self.solve_joints_position();
            timing.solver_ms += t0.elapsed().as_secs_f64() * 1000.0;
            // Snapshot last manifolds for island rebuild; clone only once per frame
            if s + 1 == eff_substeps {
                last_manifolds_snapshot.clear();
                last_manifolds_snapshot.extend_from_slice(&manifolds_buf);
            }
            self.scratch_clamped = clamped_buf;
            self.scratch_manifolds = manifolds_buf;
        }
        self.scratch_narrow_shards = narrow_shards;
        self.scratch_pairs = sorted_pairs;
        self.scratch_bucket_edges = bucket_edges;
        // Diagnostics: contact-manifold partners per body from the last
        // substep (drives sleep/island debugging; tiny flat copy).
        self.debug_pairs.clear();
        self.debug_pairs
            .extend(last_manifolds_snapshot.iter().map(|m| (m.body_a, m.body_b)));
        let t_island = Instant::now();
        self.rebuild_islands(&last_manifolds_snapshot);
        self.update_sleep(dt);
        timing.island_ms += t_island.elapsed().as_secs_f64() * 1000.0;
        self.last_step_timing = timing;

        // Rebuild the broadphase at the completed poses so trigger events
        // describe the state visible after this whole physics step, not the
        // state from before the final substep's integration. The rebuild
        // itself always runs (backends key their incremental baseline off
        // it); only the overlap detect + reconcile is gated — triggerless
        // worlds skip it instead of scanning every pair for nothing.
        let t_trigger = Instant::now();
        self.broadphase
            .update(&self.bodies, 0.0, Some(&self.prev_pose));
        if has_trigger || !self.trigger_pairs.is_empty() {
            let current_triggers = detect_trigger_overlaps(&self.bodies, self.broadphase.active());
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
            let (a, b) = (m.body_a, m.body_b);
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
                body_a: a,
                body_b: b,
                kind: ContactEventKind::Begin,
            });
        }
        for (a, b) in ends {
            self.contact_events.push(ContactEvent {
                body_a: a,
                body_b: b,
                kind: ContactEventKind::End,
            });
        }
        self.contact_events
            .sort_by_key(|e| (e.body_a.min(e.body_b), e.body_a.max(e.body_b)));
        self.last_step_timing.trigger_ms += t_trigger.elapsed().as_secs_f64() * 1000.0;
        // Hand the velocity fields back to the driver: solver impulses must
        // never corrupt driver-owned kinematic state across steps. Then
        // refresh the step-start baseline for the next step's teleport cover.
        self.restore_driver_velocity();
        self.sync_prev_pose();
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        let handle = self.bodies.len();
        let island_id = if body.body_type == BodyType::Dynamic {
            handle as u32
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
        handle
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        if handle < self.bodies.len() {
            let last = self.bodies.len() - 1;
            let previous_triggers = std::mem::take(&mut self.trigger_pairs);
            let mut removed_triggers = Vec::new();
            let mut remapped_triggers = FxHashSet::default();
            for (body_a, body_b) in previous_triggers {
                if body_a == handle || body_b == handle {
                    removed_triggers.push((body_a, body_b));
                    continue;
                }
                let map = |body: usize| if body == last { handle } else { body };
                let a = map(body_a);
                let b = map(body_b);
                remapped_triggers.insert((a.min(b), a.max(b)));
            }
            removed_triggers.sort_unstable();
            for (body_a, body_b) in removed_triggers {
                self.trigger_events.push(TriggerEvent {
                    body_a,
                    body_b,
                    kind: TriggerEventKind::Exited,
                });
            }
            self.trigger_pairs = remapped_triggers;
            self.bodies.swap_remove(handle);
            self.island.swap_remove(handle);
            self.asleep.swap_remove(handle);
            self.prev_pose.swap_remove(handle);
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
    ) -> Option<JointHandle> {
        if !crate::migration::valid_joint(&kind) {
            return None;
        }
        if body_a == body_b || body_a >= self.bodies.len() || body_b >= self.bodies.len() {
            return None;
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
                self.bodies[body_a].position,
                self.bodies[body_a].orientation,
                self.bodies[body_b].position,
                self.bodies[body_b].orientation,
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
                let r = resolved.as_ref().expect("non-gear kinds resolve");
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
                let r = resolved.as_ref().expect("non-gear kinds resolve");
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
                let r = resolved.as_ref().expect("non-gear kinds resolve");
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
            if !ratio.is_finite() {
                return None;
            }
            let (Some(ja), Some(jb)) = (self.joints.get(*joint_a), self.joints.get(*joint_b))
            else {
                return None;
            };
            if !matches!(
                ja.kind,
                JointKind::Revolute { .. } | JointKind::Prismatic { .. }
            ) || !matches!(
                jb.kind,
                JointKind::Revolute { .. } | JointKind::Prismatic { .. }
            ) {
                return None;
            }
        }
        // A new joint on a sleeping island changes its constraint set — wake
        // it so the joint state can settle coherently.
        for h in [body_a, body_b] {
            if self.bodies[h].body_type == BodyType::Dynamic
                && self.asleep.get(h).copied().unwrap_or(false)
            {
                self.wake_island(h);
            }
        }
        // Gears coordinate other joints instead of constraining a body pair —
        // the underlying joints already carry the no-collide entry.
        if !is_gear {
            self.joint_pairs
                .insert((body_a.min(body_b), body_a.max(body_b)));
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
            JointKind::Distance { .. } => resolved.map(|r| r.ref_distance).unwrap_or(0.0),
            JointKind::Gear {
                joint_a,
                joint_b,
                ratio,
            } => {
                let ca =
                    crate::engine::joints::joint_coordinate(&self.bodies, &self.joints[*joint_a])
                        .unwrap_or(0.0);
                let cb =
                    crate::engine::joints::joint_coordinate(&self.bodies, &self.joints[*joint_b])
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
        Some(self.joints.len() - 1)
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        if handle >= self.joints.len() {
            return;
        }
        // Joint indices shift on removal, so gear references (joint indices)
        // are remapped through a dense rebuild: the removed joint goes, gears
        // pointing at it go with it (dangling references are never kept),
        // survivors keep pointing at the same joints.
        let old_len = self.joints.len();
        let mut drop_j = vec![false; old_len];
        drop_j[handle] = true;
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
        self.bodies.get(handle)
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        self.bodies.get_mut(handle)
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit> {
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
        closest
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
                h,
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

    fn wake_body(&mut self, handle: BodyHandle) {
        if handle < self.bodies.len() {
            self.wake_island(handle);
        }
    }
}
