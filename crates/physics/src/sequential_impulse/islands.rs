//! Island management for `SequentialImpulseEngine` (G4/G7): union-find island
//! rebuild, island-coherent sleep/wake, partitioning into parallel work items,
//! and the per-island velocity dispatch. Split out of `engine.rs` to keep each
//! type's method count within the structural gate's thresholds.

use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Mutex;

use ornis_schedule::run_levels;

use super::*;
use crate::flags::{Dispatch, RestitutionGate, SolvePath};

impl SequentialImpulseEngine {
    /// Rebuild the constraint-graph islands (union-find over dynamic bodies
    /// connected by a contact manifold). Static bodies never join islands —
    /// they anchor them, like in Jolt.
    pub(super) fn rebuild_islands(&mut self, manifolds: &[Manifold]) {
        let n = self.bodies.len();
        // Reuse scratch_parent to avoid Vec alloc per rebuild (called once per step,
        // plus per substep in partition). Capacity kept across frames.
        self.scratch_parent.clear();
        self.scratch_parent.extend(0..n);
        let mut parent = std::mem::take(&mut self.scratch_parent);
        union_contact_edges(&mut parent, &self.bodies, manifolds);
        // Joints are constraint-graph edges too (G5): jointed dynamic bodies
        // belong to one island and sleep/wake together. Gears coordinate two
        // other joints — union all four bodies through the (validated)
        // references so the geared assembly sleeps and wakes as one.
        for joint in &self.joints {
            union_dynamic_pair(
                &mut parent,
                &self.bodies,
                joint.body_a.index(),
                joint.body_b.index(),
            );
            if let JointKind::Gear {
                joint_a, joint_b, ..
            } = &joint.kind
            {
                for r in [*joint_a, *joint_b] {
                    if let Some(referenced) = self.joints.get(r.index()) {
                        union_dynamic_pair(
                            &mut parent,
                            &self.bodies,
                            referenced.body_a.index(),
                            referenced.body_b.index(),
                        );
                    }
                }
            }
        }
        merge_sleeping_representations(&mut parent, &self.asleep, &self.island, n);
        // Canonicalize island ids to the MINIMUM member index. The raw
        // union-find root depends on manifold order, which varies step to
        // step; a root that flips identity resets the island's sleep timer
        // forever and the island never sleeps (measured on a 1025-body grid:
        // half of the perfectly quiet scene stayed awake at ~200 ms/frame).
        assign_canonical_islands(self, &mut parent, n);
        self.scratch_parent = parent;
    }

    /// Island-coherent sleep bookkeeping, run once per step (G4): an island
    /// whose bodies ALL stay slow for SLEEP_TIME seconds is frozen as a
    /// whole; islands are woken as a whole by contact with an awake body
    /// (see `wake_on_impact` in `engine/contacts.rs`).
    ///
    /// Frozen fast track: an island whose every member is numerically
    /// stationary AND never moved since it was added (see
    /// [`Self::is_body_frozen`] and `body_moved`) accumulates sleep six
    /// times faster, so exact-rest scenes (tiled grids spawned at their
    /// rest pose: solver residues ~1e-8) sleep in ~2 steps instead of ~13.
    /// Anything that ever moved keeps the legacy timers bit-exactly —
    /// settling scenes, including the determinism snapshot, are untouched
    /// (a fast-settling box still drifts ~1e-4 after converging below the
    /// frozen gate, which would move the frozen point). The fast track
    /// also stays disarmed for [`Self::WAKE_GRACE_STEPS`] steps after any
    /// wake, so teleport overlaps resolved in place keep the old wakefulness
    /// floor, and still needs two steps, so step-one-active pins hold.
    pub(super) fn update_sleep(&mut self, dt: f32) {
        // Age the post-wake grace first: a wake armed during this step's
        // substep loop still counts this step as graced.
        for g in self.island_grace.values_mut() {
            *g = g.saturating_sub(1);
        }
        self.island_grace.retain(|_, &mut g| g > 0);
        let (quiet, frozen, disturbed) = Self::collect_sleep_flags(self);
        let island_size = Self::collect_island_sizes(self);
        let mut to_sleep: Vec<u32> = Vec::new();
        for (root, q) in quiet {
            let sleep_time =
                Self::sleep_time_for_size(island_size.get(&root).copied().unwrap_or(1));
            let timer = self.island_timers.entry(root).or_insert(0.0);
            if q {
                // Frozen islands converge 6x faster; the size scaling is
                // kept (large islands still need more frozen steps). Only
                // pristine islands (nothing ever moved) with expired grace
                // qualify — everything else keeps the legacy rate.
                let fast = frozen.get(&root).copied().unwrap_or(false)
                    && !disturbed.get(&root).copied().unwrap_or(false)
                    && !self.island_grace.contains_key(&root);
                *timer += dt * if fast { Self::FROZEN_SLEEP_RATE } else { 1.0 };
                if *timer >= sleep_time {
                    to_sleep.push(root);
                }
            } else {
                *timer = 0.0;
            }
        }
        if to_sleep.is_empty() {
            return;
        }
        // Single pass over bodies (was: one full scan per sleeping island —
        // the 100k settled transition showed a 6.9 s spike there). Same set
        // of bodies freezes; only the loop shape changed.
        let to_sleep: FxHashSet<u32> = to_sleep.into_iter().collect();
        for h in 0..self.bodies.len() {
            if to_sleep.contains(&self.island[h]) {
                self.asleep[h] = true;
                // A sleeping body is STATIC for the solver (Jolt
                // semantics): zero inverse mass/inertia makes every
                // impulse and effective-mass computation treat it as
                // immovable, so a resting contact with an awake body
                // can never accumulate invisible velocity in the
                // sleeper and detonate it on wake. Restored on wake.
                self.bodies[h].sleep_staticify();
            }
        }
    }

    /// Wake the whole island containing body `h` (contact with an awake body
    /// propagates motion through the island, so partial wake is incoherent).
    /// Non-dynamic bodies have no island (statics are asleep from birth and
    /// never wake anything) — waking them is a no-op by construction.
    ///
    /// Every wake also arms the post-wake grace (see [`Self::update_sleep`]):
    /// the island accumulates sleep at the normal rate for the next
    /// [`WAKE_GRACE_STEPS`] steps even if it is numerically frozen, so a
    /// teleport overlap resolved in place cannot re-sleep before the old
    /// 0.2 s floor elapses.
    pub fn wake_island(&mut self, h: usize) {
        if self.bodies[h].body_type != BodyType::Dynamic {
            return;
        }
        let root = self.island[h];
        for h2 in 0..self.bodies.len() {
            if self.island[h2] == root {
                self.asleep[h2] = false;
                // Undo the sleep-time staticification (see update_sleep).
                self.bodies[h2].wake_restore();
            }
        }
        self.island_timers.insert(root, 0.0);
        self.island_grace.insert(root, Self::WAKE_GRACE_STEPS);
    }

    /// Post-wake steps during which the frozen fast track stays disarmed.
    /// The penetration-wake tests pin 5 awake steps after a zero-velocity
    /// teleport overlap; 8 keeps a 3-step margin while still sleeping
    /// never-woken rest scenes in ~2.
    const WAKE_GRACE_STEPS: u32 = 8;
    /// Sleep-timer multiplier for pristine frozen islands (vs legacy 1×).
    const FROZEN_SLEEP_RATE: f32 = 6.0;
    /// Base sleep dwell (s) before an island may freeze.
    const SLEEP_TIME_BASE: f32 = 0.2;
    /// Extra sleep dwell (s) per body in the island.
    const SLEEP_TIME_PER_BODY: f32 = 0.02;
    /// Upper clamp on size-scaled sleep dwell (s).
    const SLEEP_TIME_MAX: f32 = 0.6;

    fn sleep_time_for_size(size: usize) -> f32 {
        (Self::SLEEP_TIME_BASE + Self::SLEEP_TIME_PER_BODY * size as f32)
            .clamp(Self::SLEEP_TIME_BASE, Self::SLEEP_TIME_MAX)
    }

    fn is_body_slow(b: &RigidBody) -> bool {
        const LIN_SLEEP: f32 = 0.15;
        const ANG_SLEEP: f32 = 0.15;
        b.velocity.length() < LIN_SLEEP && b.angular_velocity.length() < ANG_SLEEP
    }

    /// Numerically stationary: the solver has converged to a fixed point
    /// (tiled-rest residues sit at ~1e-8 for both axes). The gate alone
    /// does not protect settling scenes — a fast-settling box still drifts
    /// ~1e-4 after converging below it, which would move the frozen point —
    /// so the fast track additionally requires a pristine island (see
    /// `body_moved`): anything that ever moved keeps the legacy timers
    /// bit-exactly.
    fn is_body_frozen(b: &RigidBody) -> bool {
        const FROZEN_EPS: f32 = 1e-5;
        b.velocity.length() < FROZEN_EPS && b.angular_velocity.length() < FROZEN_EPS
    }

    /// Per-island quiet, frozen and disturbed flags. `quiet`/`frozen` are
    /// ANDs over the members (see [`Self::is_body_slow`] and
    /// [`Self::is_body_frozen`]); `disturbed` is the OR over the monotonic
    /// per-body `body_moved` marks, updated here on the first above-gate
    /// motion. Single pass. A disturbed island never fast-tracks again —
    /// the conservative direction (legacy timers), so merges, removals and
    /// solver migrations can only miss the optimization, never break a
    /// settled trajectory.
    fn collect_sleep_flags(
        engine: &mut Self,
    ) -> (
        FxHashMap<u32, bool>,
        FxHashMap<u32, bool>,
        FxHashMap<u32, bool>,
    ) {
        let mut quiet: FxHashMap<u32, bool> = FxHashMap::default();
        let mut frozen: FxHashMap<u32, bool> = FxHashMap::default();
        let mut disturbed: FxHashMap<u32, bool> = FxHashMap::default();
        for h in 0..engine.bodies.len() {
            if engine.island[h] == u32::MAX || engine.asleep[h] {
                continue;
            }
            let still = Self::is_body_frozen(&engine.bodies[h]);
            if !still {
                engine.body_moved[h] = true;
            }
            let slow = still || Self::is_body_slow(&engine.bodies[h]);
            quiet
                .entry(engine.island[h])
                .and_modify(|q| *q &= slow)
                .or_insert(slow);
            frozen
                .entry(engine.island[h])
                .and_modify(|q| *q &= still)
                .or_insert(still);
            let moved = engine.body_moved[h];
            disturbed
                .entry(engine.island[h])
                .and_modify(|q| *q |= moved)
                .or_insert(moved);
        }
        (quiet, frozen, disturbed)
    }

    fn collect_island_sizes(engine: &Self) -> FxHashMap<u32, usize> {
        let mut m: FxHashMap<u32, usize> = FxHashMap::default();
        for &r in &engine.island {
            if r != u32::MAX {
                *m.entry(r).or_insert(0) += 1;
            }
        }
        m
    }

    /// Partition `active` (manifold indices) into islands and build work
    /// items. Extracted so both the CPU path and the GPU hybrid path reuse
    /// the same island-building logic. `hook` carries the validated
    /// contact-hooks overrides aligned with the global `manifolds` slice
    /// (missing entries read as `None`); each island keeps its own aligned
    /// copy for the velocity preamble.
    pub(super) fn partition_into_islands(
        &mut self,
        active: &[usize],
        manifolds: &[Manifold],
        hook: &[Option<HookOverride>],
    ) -> Vec<IslandWork> {
        let n = self.bodies.len();
        self.scratch_parent.clear();
        self.scratch_parent.extend(0..n);
        let mut parent = std::mem::take(&mut self.scratch_parent);
        for &mi in active {
            let m = &manifolds[mi];
            let (a, b) = (m.body_a.index(), m.body_b.index());
            if self.bodies[a].body_type == BodyType::Dynamic
                && self.bodies[b].body_type == BodyType::Dynamic
            {
                let (ra, rb) = (union_find(&mut parent, a), union_find(&mut parent, b));
                if ra != rb {
                    parent[rb] = ra;
                }
            }
        }
        let mut group_of: FxHashMap<usize, usize> = FxHashMap::default();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for &mi in active {
            let m = &manifolds[mi];
            let d = if self.bodies[m.body_a.index()].body_type == BodyType::Dynamic {
                m.body_a.index()
            } else {
                m.body_b.index()
            };
            let root = union_find(&mut parent, d);
            match group_of.entry(root) {
                std::collections::hash_map::Entry::Occupied(e) => groups[*e.get()].push(mi),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(groups.len());
                    groups.push(vec![mi]);
                }
            }
        }
        // Return parent buffer before heavy island building so group loop can reuse it later if needed.
        self.scratch_parent = parent;

        let mut islands: Vec<IslandWork> = Vec::with_capacity(groups.len());
        for group in groups {
            let mut body_idx: Vec<usize> = Vec::new();
            for &mi in &group {
                body_idx.push(manifolds[mi].body_a.index());
                body_idx.push(manifolds[mi].body_b.index());
            }
            body_idx.sort_unstable();
            body_idx.dedup();
            let shard: Vec<RigidBody> = body_idx.iter().map(|&g| self.bodies[g].clone()).collect();
            let local_of: FxHashMap<usize, usize> =
                body_idx.iter().enumerate().map(|(i, &g)| (g, i)).collect();
            let island_manifolds: Vec<Manifold> = group
                .iter()
                .filter_map(|&mi| {
                    let a = *local_of.get(&manifolds[mi].body_a.index())?;
                    let b = *local_of.get(&manifolds[mi].body_b.index())?;
                    let mut mc = manifolds[mi].clone();
                    mc.body_a = crate::body::BodyHandle::from(a);
                    mc.body_b = crate::body::BodyHandle::from(b);
                    Some(mc)
                })
                .collect();
            let keys: Vec<(usize, usize)> = group
                .iter()
                .map(|&mi| {
                    let m = &manifolds[mi];
                    let (a, b) = (m.body_a.index(), m.body_b.index());
                    (a.min(b), a.max(b))
                })
                .collect();
            // Contact-hooks overrides aligned with the island manifolds
            // (same group order as `keys` above; missing global entries
            // read as `None` = legacy preamble).
            let island_hook: Vec<Option<HookOverride>> = group
                .iter()
                .map(|&mi| hook.get(mi).cloned().unwrap_or(None))
                .collect();
            islands.push(IslandWork {
                body_idx,
                bodies: shard,
                manifolds: island_manifolds,
                keys,
                states: Vec::new(),
                warm: FxHashMap::default(),
                hook: island_hook,
            });
        }
        islands
    }

    /// Runs `f` over island work items: sequentially when `Sequential`, else
    /// through one scheduler level with one node per island. Islands are
    /// disjoint over dynamic bodies, so concurrent execution is race-free;
    /// node order is fixed, so results stay deterministic for any thread
    /// count. Each node locks only its own island reference (uncontended);
    /// the single guard allocation is per dispatch, not per island.
    pub(super) fn dispatch_islands<F>(islands: &mut [IslandWork], mode: Dispatch, f: F)
    where
        F: Fn(usize, &mut IslandWork) + Sync,
    {
        if mode != Dispatch::Parallel {
            for (idx, isl) in islands.iter_mut().enumerate() {
                f(idx, isl);
            }
            return;
        }
        let guarded: Vec<Mutex<&mut IslandWork>> = islands.iter_mut().map(Mutex::new).collect();
        let level = vec![(0..guarded.len()).collect::<Vec<usize>>()];
        run_levels(&level, guarded.len(), true, |idx| {
            f(
                idx,
                &mut guarded[idx].lock().unwrap_or_else(|e| e.into_inner()),
            );
        });
    }

    /// Dispatch the island velocity solves (via the scheduler when wide
    /// enough), scatter bodies back, and merge warm caches.
    pub(super) fn dispatch_islands_velocity(
        &mut self,
        islands: &mut [IslandWork],
        gate: RestitutionGate,
        sub_dt: f32,
        dt: f32,
    ) {
        const PAR_MIN_ISLANDS: usize = 2;
        const PAR_MIN_MANIFOLDS: usize = 24;
        if islands.is_empty() {
            return;
        }
        let mode = Dispatch::from(
            islands.len() >= PAR_MIN_ISLANDS
                && islands.iter().map(|i| i.manifolds.len()).sum::<usize>() >= PAR_MIN_MANIFOLDS,
        );
        let warm_in = &self.warm_impulses;
        let base_iters = self.velocity_iterations;
        // Overrides force the scalar solver; a no-op hook keeps `wide_solver`.
        let path = velocity_solve_path(self.wide_solver, islands);
        // per-island adaptive iters: precompute outside the dispatched closure
        // so we don't borrow `self` inside it (borrow checker). Tall-stack
        // islands never scale below the full base budget (the resting
        // downscale starves O(depth) support propagation) plus one sweep
        // per chain level above the gate, capped — see `stack_velocity_iters`.
        let iters_per_island: Vec<u32> = islands
            .iter()
            .map(|isl| {
                let max_speed = isl
                    .bodies
                    .iter()
                    .filter(|b| b.body_type == BodyType::Dynamic)
                    .map(|b| b.velocity.length().max(b.angular_velocity.length()))
                    .fold(0.0f32, f32::max);
                let max_pen = isl
                    .manifolds
                    .iter()
                    .flat_map(|m| m.points[..m.point_count].iter().map(|p| p.penetration))
                    .fold(0.0f32, f32::max);
                let adaptive =
                    self.adaptive_iters_for_island_with_pen(max_speed, max_pen, dt, base_iters);
                if super::contacts::stack_path_for_island(isl.manifolds.len()) {
                    adaptive.max(super::contacts::stack_velocity_iters(
                        base_iters,
                        isl.manifolds.len(),
                    ))
                } else {
                    adaptive
                }
            })
            .collect();
        Self::dispatch_islands(islands, mode, |idx, isl| {
            let iters = iters_per_island[idx];
            let (states, warm) = Self::solve_island_velocity(
                &mut isl.bodies,
                &isl.manifolds,
                &isl.keys,
                warm_in,
                iters,
                gate,
                sub_dt,
                path,
                &isl.hook,
            );
            isl.states = states;
            isl.warm = warm;
        });
        let mut next: WarmCache = FxHashMap::default();
        for isl in islands.iter() {
            for (l, &g) in isl.body_idx.iter().enumerate() {
                if self.bodies[g].body_type == BodyType::Dynamic {
                    self.bodies[g] = isl.bodies[l].clone();
                }
            }
            next.extend(isl.warm.iter().map(|(k, v)| (*k, *v)));
        }
        self.warm_impulses = next;
    }
}

/// Scalar path when any manifold on these islands carries a hook override.
/// A no-op hook stores only `None` and stays on `configured`, so it matches
/// a run with no hooks. Wide and scalar single-point impulses are not the
/// same bits once a real impulse is applied.
fn velocity_solve_path(configured: SolvePath, islands: &[IslandWork]) -> SolvePath {
    let hooks_override = islands
        .iter()
        .any(|isl| isl.hook.iter().any(Option::is_some));
    if hooks_override {
        SolvePath::Scalar
    } else {
        configured
    }
}

/// Union two bodies into one island iff both are dynamic.
fn union_dynamic_pair(parent: &mut [usize], bodies: &[RigidBody], a: usize, b: usize) {
    if bodies[a].body_type == BodyType::Dynamic && bodies[b].body_type == BodyType::Dynamic {
        let (ra, rb) = (union_find(parent, a), union_find(parent, b));
        if ra != rb {
            parent[rb] = ra;
        }
    }
}

/// Union-find over the contact-graph edges of the fresh manifolds.
fn union_contact_edges(parent: &mut [usize], bodies: &[RigidBody], manifolds: &[Manifold]) {
    for m in manifolds {
        union_dynamic_pair(parent, bodies, m.body_a.index(), m.body_b.index());
    }
}

/// A fully sleeping island keeps its composition even if the contact detection
/// blinks for a step: its members are not integrated, so their relative
/// geometry cannot change — dissolving the island would let one member wake
/// while its support stays asleep. (One representative per old island, not an
/// O(n²) pair scan.)
#[allow(clippy::needless_range_loop)]
fn merge_sleeping_representations(parent: &mut [usize], asleep: &[bool], island: &[u32], n: usize) {
    let mut asleep_rep: FxHashMap<u32, usize> = FxHashMap::default();
    for h in 0..n {
        if !asleep.get(h).copied().unwrap_or(false) {
            continue;
        }
        let old = island[h];
        if old == u32::MAX {
            continue;
        }
        match asleep_rep.entry(old) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(h);
            }
            std::collections::hash_map::Entry::Occupied(e) => {
                let (ra, rb) = (union_find(parent, *e.get()), union_find(parent, h));
                if ra != rb {
                    parent[rb] = ra;
                }
            }
        }
    }
}

/// Write the canonical island ids (minimum member index per component) and
/// drop timers of roots that no longer exist.
// Island arrays are indexed in parallel by body handle.
#[allow(clippy::needless_range_loop)]
fn assign_canonical_islands(engine: &mut SequentialImpulseEngine, parent: &mut [usize], n: usize) {
    let mut canonical: FxHashMap<usize, usize> = FxHashMap::default();
    for h in 0..n {
        if engine.bodies[h].body_type != BodyType::Dynamic {
            continue;
        }
        let r = union_find(parent, h);
        canonical
            .entry(r)
            .and_modify(|m| *m = (*m).min(h))
            .or_insert(h);
    }
    for h in 0..n {
        engine.island[h] = if engine.bodies[h].body_type == BodyType::Dynamic {
            canonical[&union_find(parent, h)] as u32
        } else {
            u32::MAX
        };
    }
    // Drop timers of roots that no longer exist.
    let roots: FxHashSet<u32> = engine.island.iter().copied().collect();
    engine.island_timers.retain(|r, _| roots.contains(r));
    engine.island_grace.retain(|r, _| roots.contains(r));
}

/// Minimum candidate-pair count for the flat singleton fast path
/// ([`SequentialImpulseEngine::flat_singleton_eligible`]). Snapshot,
/// confluence and sleep scenes are orders of magnitude smaller, so they
/// keep the island path exactly; only visual-scale scenes route here.
pub(super) const FLAT_MIN_PAIRS: usize = 2048;

/// Shard count rule for the flat singleton path: enough coarse shards to
/// feed every worker without the 100k-node dispatch the island path pays
/// (same shape as the narrowphase rule, so both stages scale together).
/// Order-preserving chunking keeps the assignment deterministic.
/// Fallback worker hint when `available_parallelism` is unavailable.
const DEFAULT_WORKER_HINT: usize = 4;
/// Coarse shards per worker on the flat singleton path.
const FLAT_SHARDS_PER_WORKER: usize = 4;
/// Upper clamp on flat-path shard count.
const MAX_FLAT_SHARDS: usize = 64;

fn flat_shard_count(pairs: usize) -> usize {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(DEFAULT_WORKER_HINT);
    (threads * FLAT_SHARDS_PER_WORKER)
        .clamp(DEFAULT_WORKER_HINT, MAX_FLAT_SHARDS)
        .min(pairs.max(1))
}

impl SequentialImpulseEngine {
    /// Flat-path eligibility: no joints, at least [`FLAT_MIN_PAIRS`]
    /// candidates, and every dynamic body in at most one candidate pair
    /// (statics and kinematics may repeat — they anchor islands, like in
    /// Jolt). Then every contact island is a single manifold by
    /// construction, and the union-find / group / clone / dispatch
    /// scaffolding can be skipped in favor of a direct solve with the
    /// same per-manifold kernels in an equivalent order (see
    /// [`SequentialImpulseEngine::solve_flat_velocity`]).
    ///
    /// Single linear scan with generation stamps (`flat_marks`/`flat_gen`):
    /// no hash map, no allocation after the first large scene. Decided
    /// once per step from the broadphase pairs (a superset of every
    /// substep's narrow-active set, so eligibility is stable for the step).
    pub(super) fn flat_singleton_eligible(&mut self, active: &[(usize, usize)]) -> bool {
        if active.len() < FLAT_MIN_PAIRS || !self.joint_pairs.is_empty() {
            return false;
        }
        let n = self.bodies.len();
        if self.flat_marks.len() != n {
            self.flat_marks.resize(n, 0);
        }
        self.flat_gen = self.flat_gen.wrapping_add(1);
        if self.flat_gen == 0 {
            // Wrapped past the reserved cleared state: reset and restart.
            self.flat_marks.fill(0);
            self.flat_gen = 1;
        }
        let stamp = self.flat_gen;
        let (bodies, marks) = (&self.bodies, &mut self.flat_marks);
        for &(a, b) in active {
            for h in [a, b] {
                if bodies[h].body_type == BodyType::Dynamic {
                    if marks[h] == stamp {
                        return false;
                    }
                    marks[h] = stamp;
                }
            }
        }
        true
    }

    /// Coarsen the narrow-active manifolds into solve shards: contiguous
    /// chunks in manifold order (deterministic), each a self-contained
    /// [`IslandWork`] with cloned bodies and island-local manifold indices
    /// — the same work-item shape the island dispatch solves, but `S`
    /// shards instead of one work item per manifold. Singleton eligibility
    /// guarantees shards are disjoint over dynamic bodies, so the parallel
    /// solve commutes exactly (Strong Confluence).
    pub(super) fn build_flat_shards(
        &self,
        active: &[usize],
        manifolds: &[Manifold],
    ) -> Vec<IslandWork> {
        let count = flat_shard_count(active.len());
        let mut shards: Vec<IslandWork> = Vec::with_capacity(count);
        for part in active.chunks(active.len().div_ceil(count)) {
            let mut body_idx: Vec<usize> = Vec::with_capacity(part.len() * 2);
            for &gmi in part {
                body_idx.push(manifolds[gmi].body_a.index());
                body_idx.push(manifolds[gmi].body_b.index());
            }
            body_idx.sort_unstable();
            body_idx.dedup();
            let bodies: Vec<RigidBody> = body_idx.iter().map(|&g| self.bodies[g].clone()).collect();
            let local_of: FxHashMap<usize, usize> =
                body_idx.iter().enumerate().map(|(i, &g)| (g, i)).collect();
            let shard_manifolds: Vec<Manifold> = part
                .iter()
                .filter_map(|&gmi| {
                    let a = *local_of.get(&manifolds[gmi].body_a.index())?;
                    let b = *local_of.get(&manifolds[gmi].body_b.index())?;
                    let mut mc = manifolds[gmi].clone();
                    mc.body_a = crate::body::BodyHandle::from(a);
                    mc.body_b = crate::body::BodyHandle::from(b);
                    Some(mc)
                })
                .collect();
            let keys: Vec<(usize, usize)> = part
                .iter()
                .map(|&gmi| {
                    let m = &manifolds[gmi];
                    let (a, b) = (m.body_a.index(), m.body_b.index());
                    (a.min(b), a.max(b))
                })
                .collect();
            shards.push(IslandWork {
                body_idx,
                bodies,
                manifolds: shard_manifolds,
                keys,
                states: Vec::new(),
                warm: FxHashMap::default(),
                hook: Vec::new(),
            });
        }
        shards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::BodyType;
    use crate::engine::PhysicsEngine;

    /// Interior rest grid (the `settled_grid` shape, no edge overhang):
    /// pristine frozen islands must sleep in a couple of steps, not after
    /// the full legacy 0.2 s — while the very first step stays fully awake.
    #[test]
    fn pristine_rest_grid_sleeps_fast_but_first_step_stays_active() {
        let mut physics = SequentialImpulseEngine::new(glam::Vec3::new(0.0, -9.81, 0.0));
        for tx in -1..=1 {
            for tz in -1..=1 {
                physics.add_body(RigidBody::new_box(
                    glam::Vec3::new(tx as f32 * 10.0, -0.5, tz as f32 * 10.0),
                    glam::Vec3::new(5.0, 0.5, 5.0),
                    0.0,
                ));
            }
        }
        let mut dynamics = Vec::new();
        for gx in -2..=2 {
            for gz in -2..=2 {
                dynamics.push(physics.add_body(RigidBody::new_box(
                    glam::Vec3::new(gx as f32 * 2.0, 0.4, gz as f32 * 2.0),
                    glam::Vec3::splat(0.4),
                    1.0,
                )));
            }
        }
        physics.step(1.0 / 60.0);
        assert!(
            dynamics.iter().all(|&h| !physics.is_asleep(h)),
            "first step is active: nothing sleeps yet"
        );
        for _ in 0..3 {
            physics.step(1.0 / 60.0);
        }
        assert!(
            dynamics.iter().all(|&h| physics.is_asleep(h)),
            "pristine rest grid must sleep within 4 steps"
        );
        // Rest heights hold: the fast track froze a fixed point, not a fall.
        for &h in &dynamics {
            let b = physics.get_body(h).unwrap();
            assert!(
                (b.position.y - 0.4).abs() < 0.05,
                "settled body rest height, got {}",
                b.position.y
            );
        }
    }

    /// Anything that ever moved keeps the legacy timers: a dropped box is
    /// still awake 6 steps after landing, even once it is numerically
    /// frozen — only pristine islands fast-track.
    #[test]
    fn moved_island_keeps_legacy_sleep_latency() {
        let mut physics = SequentialImpulseEngine::new(glam::Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            glam::Vec3::new(0.0, -1.0, 0.0),
            glam::Vec3::new(10.0, 1.0, 10.0),
            0.0,
        ));
        let klein = physics.add_body(RigidBody::new_box(
            glam::Vec3::new(0.0, 2.0, 0.0),
            glam::Vec3::splat(0.5),
            1.0,
        ));
        // Fall 1.5 m to the floor (~33 steps), then 6 settled steps: past
        // the 2-step fast-track latency, well short of the legacy ~13.
        let mut landed = false;
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
            if physics.get_body(klein).unwrap().position.y < 0.6 {
                landed = true;
                break;
            }
        }
        assert!(landed, "box must land within 120 steps");
        assert!(
            physics.body_moved[klein.index()],
            "fallen box is marked moved"
        );
        for _ in 0..6 {
            physics.step(1.0 / 60.0);
        }
        assert!(
            !physics.is_asleep(klein),
            "a box that fell 1.5 m must not fast-track: legacy latency applies"
        );
    }

    /// Post-wake grace: a pristine sleeper woken by a zero-velocity deep
    /// overlap is still awake 5 steps later (legacy floor), even though it
    /// never exceeds the frozen gate.
    #[test]
    fn woken_pristine_sleeper_keeps_wakefulness_floor() {
        let mut physics = SequentialImpulseEngine::new(glam::Vec3::ZERO);
        let sleeper = physics.add_body(RigidBody::new_box(
            glam::Vec3::new(0.0, 0.5, 0.0),
            glam::Vec3::splat(0.5),
            1.0,
        ));
        for _ in 0..10 {
            physics.step(1.0 / 60.0);
        }
        assert!(physics.is_asleep(sleeper), "box must sleep in zero-g");
        physics.add_body(RigidBody::new_box(
            glam::Vec3::new(0.0, 1.45, 0.0),
            glam::Vec3::splat(0.5),
            1.0,
        ));
        for _ in 0..5 {
            physics.step(1.0 / 60.0);
        }
        assert!(
            !physics.is_asleep(sleeper),
            "deep overlap must keep the sleeper awake for 5 steps"
        );
        assert!(
            physics
                .bodies
                .iter()
                .any(|b| b.body_type == BodyType::Dynamic),
            "dynamics present"
        );
    }
}
