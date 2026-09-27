//! Substep driver: warm-start and solver-stage state, substep counts,
//! integration, driver-motion plumbing and the continuous-collision passes.

use glam::{Quat, Vec3};
use rustc_hash::FxHashMap;

use crate::body::{BodyType, RigidBody};
use crate::broadphase::PrevPose;
use crate::distance;
use crate::engine::Manifold;

use super::SequentialImpulseEngine;
use super::math::vec3_finite;
use super::queries::{
    ccd_impact_velocity, find_continuous_hit, kinematic_cast, shape_min_dimension,
    swept_orientation,
};
use super::*;

/// Cached contact point for warm starting (G2b): world-space point plus the
/// accumulated normal impulse from the previous substep.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WarmPoint {
    /// Body-frame anchors of the contact point on both bodies. Unlike the
    /// world position, these are stable while the same surface feature stays
    /// in contact (Jolt persists contacts by feature id the same way).
    pub(crate) la: Vec3,
    pub(crate) lb: Vec3,
    /// Contact normal at cache time: a corner rolling from one face to the
    /// next is a DIFFERENT feature and must not inherit the impulse.
    pub(crate) normal: Vec3,
    pub(crate) impulse: f32,
}

/// Warm-start cache: per body pair, up to MAX_MANIFOLD_POINTS matched contact points.
pub(crate) type WarmCache = FxHashMap<(usize, usize), ([WarmPoint; MAX_MANIFOLD_POINTS], usize)>;

/// Per-manifold solver state shared between the velocity and position stages
/// of a substep (G6 stage split: velocities solve BEFORE positions move, so
/// the NGS pass needs the detection-time anchors/penetrations carried over).
#[derive(Clone)]
pub struct ManifoldState {
    /// Manifold index in the step's manifold list.
    pub mi: usize,
    /// First body handle.
    pub i: usize,
    /// Second body handle.
    pub j: usize,
    /// Active point count (1..=MAX_MANIFOLD_POINTS): indexes every parallel array below.
    /// Enforced at the single construction site; direct writes bypass the
    /// invariant (see [`ManifoldState::has_valid_count`]).
    pub count: usize,
    /// Accumulated normal impulse per point (warm start).
    pub acc: [f32; MAX_MANIFOLD_POINTS],
    /// Accumulated friction impulse per point (first tangent).
    pub acc_friction: [f32; MAX_MANIFOLD_POINTS],
    /// Accumulated friction impulse per point (second tangent).
    pub acc_friction2: [f32; MAX_MANIFOLD_POINTS],
    /// Restitution bias per point.
    pub bias: [f32; MAX_MANIFOLD_POINTS],
    /// G6 speculative per-point approach-speed limit (negative of the
    /// remaining gap / sub_dt; 0 for touching points). The velocity
    /// solve drives vn to this target instead of 0, so a separated
    /// point may close its gap within the substep but never more.
    pub target: [f32; MAX_MANIFOLD_POINTS],
    /// Coulomb friction coefficient.
    pub mu: f32,
    /// Transverse Coulomb coefficient (ODE `mu2` parity): cap along `t2`
    /// while `mu` caps `t1`. Bitwise-equal to `mu` for all default bodies,
    /// which routes the solve through the legacy circular-cone path —
    /// existing scenes stay bit-identical (snapshot-guarded).
    pub mu2: f32,
    /// Pair rolling-resistance coefficient in meters (`max` of both
    /// bodies; MuJoCo `rolling` parity). Zero disables rolling solve.
    pub mu_roll: f32,
    /// Pair torsional coefficient in meters (`max`; MuJoCo parity).
    pub mu_spin: f32,
    /// Accumulated rolling impulses about `t1`/`t2` per point, capped by
    /// `mu_roll × acc[k]` (torque-cap × normal-impulse, MuJoCo units).
    pub acc_roll: [f32; MAX_MANIFOLD_POINTS],
    /// Accumulated rolling impulse about `t2` per point.
    pub acc_roll2: [f32; MAX_MANIFOLD_POINTS],
    /// Accumulated torsional impulse about `n` per point, capped by
    /// `mu_spin × acc[k]`.
    pub acc_spin: [f32; MAX_MANIFOLD_POINTS],
    /// Fixed tangent basis (Box2D-style): friction is solved along
    /// directions derived from the contact normal ONCE, not from the
    /// instantaneous slip velocity — velocity-aligned friction walks
    /// the contact and lets resting stacks drift sideways.
    pub t1: Vec3,
    /// Second fixed tangent axis.
    pub t2: Vec3,
    /// G3 body-frame anchors and detection-time penetration per point,
    /// so the positional pass can re-measure live separation.
    pub la: [Vec3; MAX_MANIFOLD_POINTS],
    /// Body-B body-frame anchors per point.
    pub lb: [Vec3; MAX_MANIFOLD_POINTS],
    /// Detection-time penetration per point.
    pub pen0: [f32; MAX_MANIFOLD_POINTS],
}

impl ManifoldState {
    /// Whether `count` satisfies the 1..=MAX_MANIFOLD_POINTS parallel-array invariant.
    /// The single construction site
    /// (`prepare_manifold_state`, via [`Manifold::has_valid_count`](crate::engine::Manifold::has_valid_count))
    /// admits only valid counts; every parallel array below is indexed
    /// `0..count`.
    pub fn has_valid_count(&self) -> bool {
        (1..=MAX_MANIFOLD_POINTS).contains(&self.count)
    }

    /// Live lane count, clamped to the buffer size (defensive: construction
    /// admits only `1..=MAX_MANIFOLD_POINTS`, so this equals `count` on valid states).
    pub fn live_count(&self) -> usize {
        self.count.min(MAX_MANIFOLD_POINTS)
    }

    /// Live normal impulses (`acc[..count]`).
    pub fn acc_slice(&self) -> &[f32] {
        &self.acc[..self.live_count()]
    }

    /// Live detection-time penetrations (`pen0[..count]`).
    pub fn pen0_slice(&self) -> &[f32] {
        &self.pen0[..self.live_count()]
    }
}

/// Per-island work item for the G7 parallel solver: an island-local shard of
/// the world. Body indices inside `manifolds` and `states` are LOCAL
/// (positions in `body_idx`/`bodies`); `keys` maps each local manifold to its
/// global body-pair key for the warm-start cache. Islands are disjoint over
/// dynamic bodies by construction (union-find over the fresh manifolds), so
/// solving them concurrently is race-free and bit-identical for any thread
/// count: each island runs its manifolds in the original global order, and
/// Gauss-Seidel updates on disjoint state commute exactly.
pub(crate) struct IslandWork {
    /// Sorted global body handles; local index = position in this vec.
    pub(crate) body_idx: Vec<usize>,
    /// Gathered body shard (statics included; never written back).
    pub(crate) bodies: Vec<RigidBody>,
    /// Manifolds cloned with LOCAL body indices.
    pub(crate) manifolds: Vec<Manifold>,
    /// Global sorted body-pair key per local manifold (warm cache I/O).
    pub(crate) keys: Vec<(usize, usize)>,
    /// Velocity-stage output, consumed by the position stage.
    pub(crate) states: Vec<ManifoldState>,
    /// This island's updated warm-cache entries (merged after the join).
    pub(crate) warm: WarmCache,
}

/// Context for building a ManifoldState (packs the per-manifold parameters,
/// keeping `build_manifold_state` below the structural gate's nargs limit).
// Only consumed by the `gpu` feature's GPU contact path today.
#[allow(dead_code)]
pub(crate) struct ManifoldCtx<'a> {
    pub(crate) bodies: &'a mut [RigidBody],
    pub(crate) warm_in: &'a WarmCache,
    pub(crate) gate: crate::flags::RestitutionGate,
    pub(crate) sub_dt: f32,
    pub(crate) mi: usize,
    pub(crate) i: usize,
    pub(crate) j: usize,
}

impl SequentialImpulseEngine {
    /// Applies the worst-case budget to a speed-requested substep count,
    /// given the broadphase candidate-pair count. Returns
    /// `(applied, shed)`. Pure counts, no wall-clock: deterministic for a
    /// given scene trajectory.
    pub fn apply_step_budget(&self, requested: u32, pairs: usize) -> (u32, u32) {
        let Some(budget) = self.step_budget else {
            return (requested, 0);
        };
        if requested <= budget.min_substeps || pairs == 0 {
            return (requested, 0);
        }
        let allowed = u32::try_from(budget.max_pair_substeps / pairs).unwrap_or(u32::MAX);
        let applied = requested.min(allowed.max(budget.min_substeps));
        (applied, requested - applied)
    }

    /// Adaptive substep count for this step: keep each sub-dt at or below a
    /// target so fast bodies get enough solver passes to settle without
    /// tunnelling, while resting/low-speed scenes drop to `MIN_SUBSTEPS` so the
    /// world can sleep cheaply. `self.substeps` is the upper bound (set by the
    /// caller); the solver never runs more than that.
    ///
    /// ponytail: global heuristic clamped to a fixed min — per-island or
    /// penetration-driven substepping is the upgrade path when heterogeneous
    /// scenes show regressions.
    pub(crate) fn effective_substeps(&self, dt: f32) -> u32 {
        const MIN_SUBSTEPS: u32 = 4;
        const SUB_DT_TARGET: f32 = 1.0 / 240.0;
        let mut max_speed = 0.0f32;
        for (h, b) in self.bodies.iter().enumerate() {
            if b.body_type == BodyType::Dynamic && !self.asleep[h] {
                max_speed = max_speed.max(b.velocity.length());
            }
        }
        // Adaptive substepping only *lowers* the caller's cap (self.substeps)
        // on low-speed scenes so the world can sleep cheaply; it never raises
        // above the cap (so explicit set_substeps(1) for CCD tests is kept).
        let lower = MIN_SUBSTEPS.min(self.substeps);
        let wanted = (max_speed * dt / SUB_DT_TARGET).ceil() as u32;
        wanted.clamp(lower, self.substeps)
    }

    /// Per-body required substeps for the current frame (B: per-island
    /// sub-dt splitting). Slow bodies need MIN, fast need up to `self.substeps`.
    pub fn body_required_substeps(&self, dt: f32) -> Vec<u32> {
        const MIN_SUBSTEPS: u32 = 4;
        const SUB_DT_TARGET: f32 = 1.0 / 240.0;
        let lower = MIN_SUBSTEPS.min(self.substeps);
        self.bodies
            .iter()
            .enumerate()
            .map(|(h, b)| {
                if b.body_type != BodyType::Dynamic || self.asleep[h] {
                    lower
                } else {
                    let wanted = (b.velocity.length() * dt / SUB_DT_TARGET).ceil() as u32;
                    wanted.clamp(lower, self.substeps)
                }
            })
            .collect()
    }

    /// Per-island iteration scaling: fast islands get the full budget,
    /// slow islands are solved cheaply. Derived from the same target sub-dt
    /// as `effective_substeps` so the two heuristics stay coherent.
    ///
    /// ponytail: 2 iters minimum for velocity, 1 for position — per-island
    /// sub-dt splitting (instead of iteration scaling) is the upgrade path
    /// if heterogeneous scenes still show solver jitter.
    #[allow(dead_code)]
    pub fn adaptive_iters_for_island(&self, max_speed: f32, dt: f32, base_iters: u32) -> u32 {
        self.adaptive_iters_for_island_with_pen(max_speed, 0.0, dt, base_iters)
    }

    /// Penetration-aware variant of [`SequentialImpulseEngine::adaptive_iters_for_island`]:
    /// deep penetrations scale iterations even at zero speed.
    pub fn adaptive_iters_for_island_with_pen(
        &self,
        max_speed: f32,
        max_pen: f32,
        dt: f32,
        base_iters: u32,
    ) -> u32 {
        const MIN_SUBSTEPS: u32 = 4;
        const SUB_DT_TARGET: f32 = 1.0 / 240.0;
        const PEN_SLOP: f32 = 0.01;
        // Keep at least 2 velocity / 1 position iteration so even resting
        // islands still correct residual penetration.
        let min_iters = if base_iters > 4 { 2 } else { 1 };
        let max_sub = self.substeps.max(1);
        let lower = MIN_SUBSTEPS.min(max_sub);
        let wanted_vel = (max_speed * dt / SUB_DT_TARGET).ceil() as u32;
        let wanted_pen = (max_pen / PEN_SLOP).ceil() as u32;
        let wanted = wanted_vel.max(wanted_pen).clamp(lower, max_sub);
        let scaled = ((wanted as f32 / max_sub as f32) * base_iters as f32).ceil() as u32;
        scaled.clamp(min_iters, base_iters)
    }

    /// Velocity half of the integration (Box3D `IntegrateVelocities`): apply
    /// gravity and pending torque so the constraint solvers below act on the
    /// velocities that the upcoming position integration will actually use.
    /// Free spin also gains the gyroscopic correction
    /// ([`apply_gyroscopic`]): without it anisotropic bodies cannot precess,
    /// and spin about the intermediate inertia axis stays (wrongly) stable.
    pub(crate) fn integrate_velocities(&mut self, dt: f32) {
        debug_assert!(
            dt.is_finite() && dt > 0.0,
            "dt must be positive finite, got {dt}"
        );
        for (h, body) in self.bodies.iter_mut().enumerate() {
            if body.body_type != BodyType::Dynamic || self.asleep.get(h).copied().unwrap_or(false) {
                continue;
            }
            body.velocity += self.gravity * dt;

            // Angular integrate from applied torque.
            if body.torque != Vec3::ZERO {
                let torque_delta = body.torque * dt;
                debug_assert!(
                    vec3_finite(torque_delta),
                    "torque*dt overflowed: torque={:?} dt={dt}",
                    body.torque
                );
                body.angular_velocity +=
                    mul_inv_inertia(body.inertia, body.orientation, torque_delta);
                body.torque = Vec3::ZERO;
            }
            // Gyroscopic correction, torque or not: free spinners need it.
            Self::apply_gyroscopic(
                body.inertia,
                body.orientation,
                &mut body.angular_velocity,
                dt,
            );
        }
    }

    /// Gyroscopic correction, Box3D-`b3IntegrateVelocitiesTask` idea with an own
    /// diagonal-only implementation: Newton-Raphson on
    /// `I·(w2−w1) + h·(w2×(I·w2)) = 0` in the body frame (our inertia is a body
    /// diagonal, theirs a full local 3×3 — same equation, cheaper Jacobian).
    /// Fixed [`GYROSCOPIC_ITERATIONS`] iterations, so the result is a pure
    /// function of the inputs (deterministic); on non-convergence the last
    /// iterate is kept, never NaN (the Jacobian stays diagonally dominant for
    /// small `h`, and `solve_small` reports singularity instead of dividing).
    ///
    /// Skips, both exact: isotropic inertia (the term is identically zero, so
    /// every cube/sphere test scene pays nothing) and zero spin.
    ///
    /// Without this, spin about the intermediate inertia axis is (wrongly)
    /// stable and free asymmetric bodies never tumble — see the Dzhanibekov
    /// test below.
    const GYROSCOPIC_ITERATIONS: usize = 3;

    pub(crate) fn apply_gyroscopic(inertia: Vec3, orientation: Quat, omega: &mut Vec3, h: f32) {
        if *omega == Vec3::ZERO {
            return;
        }
        let imax = inertia.x.max(inertia.y).max(inertia.z);
        if !imax.is_finite() || imax <= 0.0 {
            return;
        }
        let spread = (inertia.x - inertia.y)
            .abs()
            .max((inertia.y - inertia.z).abs())
            .max((inertia.z - inertia.x).abs());
        if spread <= 1e-6 * imax {
            return;
        }
        let (a, b, c) = (inertia.x, inertia.y, inertia.z);
        // To the body frame (diagonal inertia) and back with the same rotation.
        let q = orientation;
        let w1 = q.conjugate() * *omega;
        let mut w2 = w1;
        for _ in 0..Self::GYROSCOPIC_ITERATIONS {
            // u = I·w2; residual r = I·(w2−w1) + h·(w2×u).
            let u = Vec3::new(a * w2.x, b * w2.y, c * w2.z);
            let cross = w2.cross(u);
            let r = Vec3::new(
                a * (w2.x - w1.x) + h * cross.x,
                b * (w2.y - w1.y) + h * cross.y,
                c * (w2.z - w1.z) + h * cross.z,
            );
            // Jacobian J = I + h·(skew(w2)·I − skew(u)), diagonal inertia:
            // J[i][i] is the bare inertia (the motion term has no self-part),
            // off-diagonals mix the spin with the inertia-weighted spin.
            let j = [
                [a, h * (u.z - b * w2.z), h * (c * w2.y - u.y), 0.0],
                [h * (a * w2.z - u.z), b, h * (u.x - c * w2.x), 0.0],
                [h * (u.y - a * w2.y), h * (b * w2.x - u.x), c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let Some(d) = solve_small(&j, &[-r.x, -r.y, -r.z, 0.0], 3) else {
                break;
            };
            w2 += Vec3::new(d[0], d[1], d[2]);
        }
        let w2_world = q * w2;
        debug_assert!(
            vec3_finite(w2_world),
            "gyroscopic correction diverged: inertia={inertia:?} w1={w1:?}"
        );
        *omega = w2_world;
    }

    /// Position half of the integration (Box3D `IntegratePositions`): move
    /// bodies along the solver-adjusted velocities. Bodies flagged in `skip`
    /// were already clamped to their time of impact by the continuous pass
    /// and must not move again this substep.
    pub(crate) fn integrate_positions(&mut self, dt: f32, skip: &[bool]) {
        for (h, body) in self.bodies.iter_mut().enumerate() {
            if body.body_type != BodyType::Dynamic || self.asleep.get(h).copied().unwrap_or(false) {
                continue;
            }
            if skip.get(h).copied().unwrap_or(false) {
                continue;
            }
            // Linear integrate (semi-implicit, post-solve velocities).
            body.position += body.velocity * dt;

            // Rotation: exact small-step quaternion update (exp of angular velocity * dt).
            if body.angular_velocity != Vec3::ZERO {
                let dwq = Quat::from_scaled_axis(body.angular_velocity * dt);
                body.orientation = (dwq * body.orientation).normalize();
            }
        }
    }

    /// Step-start driver check: reports whether any kinematic body moved
    /// since the previous step end (`prev_pose`). The wake fast-path must
    /// count those, or a zero-velocity teleport ghosts while the substep
    /// loop is skipped.
    ///
    /// Runs at the very top of `step` (even on the fully-sleeping fast
    /// path) and does NOT touch `prev_pose`: the broadphase union and the
    /// kinematic sweep below both read the step-start poses, which
    /// `sync_prev_pose` refreshes at step end.
    pub(crate) fn snapshot_driver_motion(&self) -> bool {
        for (h, b) in self.bodies.iter().enumerate() {
            if b.body_type == BodyType::Kinematic
                && self
                    .prev_pose
                    .get(h)
                    .is_some_and(|p| p.pos != b.position || p.rot != b.orientation)
            {
                return true;
            }
        }
        false
    }

    /// Refreshes `prev_pose` from the current poses. Runs at the very end of
    /// every full `step` (never on the fast path — there no kinematic moved,
    /// so there is nothing to refresh), keeping the step-start baseline
    /// exactly one step behind.
    pub(crate) fn sync_prev_pose(&mut self) {
        let n = self.bodies.len();
        self.prev_pose.resize(
            n,
            PrevPose {
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY,
            },
        );
        for (h, b) in self.bodies.iter().enumerate() {
            self.prev_pose[h].pos = b.position;
            self.prev_pose[h].rot = b.orientation;
        }
    }

    /// Above-gate teleports act as the implied motion everywhere for one
    /// step: installs (`position - prev`) / `dt` (plus the orientation-delta
    /// spin) into the kinematic velocity fields, saving the driver values
    /// into `saved_driver_vel`. Margins, CCD, wake and contact responses all
    /// read fields, so this single write keeps every phase consistent;
    /// `restore_driver_velocity` at step end puts the driver values back, so
    /// the solver never corrupts driver-owned state.
    ///
    /// Below-gate teleports keep their fields (positional settle, Box2D
    /// parity: a small nudge imparts no momentum); disciplined drivers whose
    /// per-step segment sits below the gate see bit-identical behavior.
    pub(crate) fn apply_driver_velocity(&mut self, dt: f32) {
        self.saved_driver_vel.clear();
        if dt <= 0.0 {
            return;
        }
        for h in 0..self.bodies.len() {
            let b = &self.bodies[h];
            if b.body_type != BodyType::Kinematic {
                continue;
            }
            let Some(prev) = self.prev_pose.get(h) else {
                continue;
            };
            let displacement = b.position - prev.pos;
            // Same travel gate as the sweep: the discrete phase owns short
            // segments, the implied motion owns long ones.
            if displacement.length() <= 0.5 * shape_min_dimension(&b.shape) {
                continue;
            }
            let dq = b.orientation * prev.rot.conjugate();
            let mut spin = b.angular_velocity;
            if dq.w < 1.0 - 1e-6 {
                let angle = 2.0 * dq.w.clamp(-1.0, 1.0).acos();
                let axis = dq.xyz() / (1.0 - dq.w * dq.w).sqrt().max(1e-9);
                if axis.is_finite() {
                    spin = axis * (angle / dt);
                }
            }
            self.saved_driver_vel
                .push((h, b.velocity, b.angular_velocity));
            let m = &mut self.bodies[h];
            m.velocity = displacement / dt;
            m.angular_velocity = spin;
        }
    }

    /// Puts back the driver velocity fields saved by
    /// [`apply_driver_velocity`](Self::apply_driver_velocity). Runs at the
    /// end of every full `step` (after sleep), so externally observed fields
    /// are always driver-owned.
    pub(crate) fn restore_driver_velocity(&mut self) {
        for (h, lin, ang) in self.saved_driver_vel.drain(..) {
            if let Some(b) = self.bodies.get_mut(h) {
                b.velocity = lin;
                b.angular_velocity = ang;
            }
        }
    }

    /// Kinematic sweep (teleport CCD): each kinematic body whose step
    /// displacement outruns the linear travel gate casts it against the
    /// dynamic bodies via conservative advancement. A hit wakes the victim
    /// and transfers the normal approach one-shot (kinematic-vs-dynamic
    /// contact response with the driver's implied velocity); tangential
    /// motion and resting penetration stay with the discrete solver.
    ///
    /// The driver keeps full ownership of the kinematic pose — the sweep
    /// never moves the mover, only the victims. Rotation teleports are NOT
    /// swept (linear cast only): a spinning driver must still carry a
    /// matching `angular_velocity` field, same contract as before.
    pub(crate) fn solve_kinematic_sweep(&mut self, dt: f32) {
        struct SweepHit {
            target: usize,
            normal: Vec3,
            driver_vel: Vec3,
            restitution: f32,
        }
        let mut hits: Vec<SweepHit> = Vec::new();
        {
            let bodies = &self.bodies;
            for (h, mover) in bodies.iter().enumerate() {
                if mover.body_type != BodyType::Kinematic || mover.is_trigger {
                    continue;
                }
                let Some(prev) = self.prev_pose.get(h) else {
                    continue;
                };
                let displacement = mover.position - prev.pos;
                // Travel gate, mirror of the linear CCD one: below half the
                // thinnest feature the discrete phase + speculative margin
                // own the contact, no sweep needed.
                if displacement.length() <= 0.5 * shape_min_dimension(&mover.shape) {
                    continue;
                }
                let mover_layer = mover.collision_layer;
                let mover_mask = mover.collision_mask;
                // Earliest hit across all dynamic targets (linear-CCD order:
                // nearest wins, deterministic by body index on ties).
                let mut best: Option<(f32, Vec3, usize)> = None;
                for (t, target) in bodies.iter().enumerate() {
                    if t == h
                        || target.is_trigger
                        || target.body_type != BodyType::Dynamic
                        || mover_mask & target.collision_layer == 0
                        || target.collision_mask & mover_layer == 0
                    {
                        continue;
                    }
                    let target_ref = distance::ShapeRef {
                        shape: &target.shape,
                        pos: target.position,
                        rot: target.orientation,
                    };
                    if let Some((dist, normal)) = kinematic_cast(
                        &mover.shape,
                        mover.orientation,
                        prev.pos,
                        displacement,
                        target_ref,
                    ) && best.is_none_or(|(b, _, _)| dist < b)
                    {
                        best = Some((dist, normal, t));
                    }
                }
                // No end-overlap guard: the discrete phase reads the same
                // implied velocity (installed by `apply_driver_velocity`),
                // so sweep and discrete responses agree — like the linear
                // CCD clamp coexisting with the discrete contact solve.
                if let Some((_, normal, t)) = best {
                    let driver_vel = if dt > 0.0 {
                        displacement / dt
                    } else {
                        mover.velocity
                    };
                    hits.push(SweepHit {
                        target: t,
                        normal,
                        driver_vel,
                        restitution: mover.restitution.min(bodies[t].restitution),
                    });
                }
            }
        }
        for hit in hits {
            self.wake_island(hit.target);
            let t = &mut self.bodies[hit.target];
            // Mirror image of the linear CCD response (which acts on the
            // mover): here the mover is immovable, so the victim absorbs
            // the approach. `u > 0` means closing along the sweep normal.
            let u = (t.velocity - hit.driver_vel).dot(hit.normal);
            if u > 0.0 {
                let bounce = if u > 1.0 { 1.0 + hit.restitution } else { 1.0 };
                t.velocity -= hit.normal * (bounce * u);
            }
        }
    }

    /// Time-of-impact pass (G6, b3SolveContinuous analog): runs after the
    /// velocity solve, before positions move. Linear movers use conservative
    /// advancement; rotating boxes/capsules use the fully analytic swept-volume
    /// conservative advancement (exact distance + `|disp|+r*angle` bound, no
    /// 5° sampling). The body is clamped to the first detected impact and
    /// flagged in `skip` so `integrate_positions` does not move it a second
    /// time.
    // The loop indexes bodies/asleep/skip in parallel; a range loop is the
    // clearest form here (same policy as the solver loops above).
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn solve_continuous(&mut self, sub_dt: f32, skip: &mut [bool]) {
        debug_assert!(
            sub_dt.is_finite() && sub_dt > 0.0,
            "sub_dt must be positive finite, got {sub_dt}"
        );
        for h in 0..self.bodies.len() {
            if self.bodies[h].body_type != BodyType::Dynamic || self.asleep[h] {
                continue;
            }
            let disp = self.bodies[h].velocity * sub_dt;
            debug_assert!(vec3_finite(disp), "velocity*sub_dt overflowed for body {h}");
            let Some(hit) = find_continuous_hit(&self.bodies, h, disp, sub_dt) else {
                continue;
            };
            let orientation = swept_orientation(&self.bodies[h], sub_dt, hit.fraction);
            let e = self.bodies[h]
                .restitution
                .min(self.bodies[hit.handle.index()].restitution);
            let b = &mut self.bodies[h];
            // Back off a hair so the discrete narrow phase sees a clean
            // touching contact next substep, not a zero-gap flicker.
            b.position += disp * hit.fraction + hit.normal * 1e-3;
            b.orientation = orientation;
            skip[h] = true;
            if hit.kind.is_angular() {
                // Unified contact impulse: contact-point speed (spin counts),
                // restitution-aware, tangential preserved. Subsumes the old
                // split linear-bounce + spin-kill for angular hits.
                let lever = hit.contact.map(|c| c - b.position).unwrap_or(Vec3::ZERO);
                let (v, w) = ccd_impact_velocity(
                    b.velocity,
                    b.angular_velocity,
                    b.inv_mass,
                    b.inertia,
                    b.orientation,
                    lever,
                    hit.normal,
                    e,
                );
                b.velocity = v;
                b.angular_velocity = w;
            } else {
                let vn = b.velocity.dot(hit.normal);
                if vn < 0.0 {
                    // Inelastic below the shared restitution threshold; a
                    // genuine impact bounces (one-shot, like the discrete
                    // restitution stage).
                    let bounce = if vn < -1.0 { 1.0 + e } else { 1.0 };
                    b.velocity -= hit.normal * (bounce * vn);
                }
            }
        }
    }
}
