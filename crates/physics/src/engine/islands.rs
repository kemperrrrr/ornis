//! Island management for `BuiltinPhysicsEngine` (G4/G7): union-find island
//! rebuild, island-coherent sleep/wake, partitioning into parallel work items,
//! and the per-island velocity dispatch. Split out of `engine.rs` to keep each
//! type's method count within the structural gate's thresholds.

use std::collections::HashMap;

use rayon::prelude::*;

use super::*;

impl BuiltinPhysicsEngine {
    /// Rebuild the constraint-graph islands (union-find over dynamic bodies
    /// connected by a contact manifold). Static bodies never join islands —
    /// they anchor them, like in Jolt.
    pub(super) fn rebuild_islands(&mut self, manifolds: &[Manifold]) {
        let n = self.bodies.len();
        let mut parent: Vec<usize> = (0..n).collect();

        union_contact_edges(&mut parent, &self.bodies, manifolds);
        // Joints are constraint-graph edges too (G5): jointed dynamic bodies
        // belong to one island and sleep/wake together.
        for joint in &self.joints {
            union_dynamic_pair(&mut parent, &self.bodies, joint.body_a, joint.body_b);
        }
        merge_sleeping_representations(&mut parent, &self.asleep, &self.island, n);
        // Canonicalize island ids to the MINIMUM member index. The raw
        // union-find root depends on manifold order, which varies step to
        // step; a root that flips identity resets the island's sleep timer
        // forever and the island never sleeps (measured on a 1025-body grid:
        // half of the perfectly quiet scene stayed awake at ~200 ms/frame).
        assign_canonical_islands(self, &mut parent, n);
    }

    /// Island-coherent sleep bookkeeping, run once per step (G4): an island
    /// whose bodies ALL stay slow for SLEEP_TIME seconds is frozen as a
    /// whole; islands are woken as a whole by contact with an awake body
    /// (see `wake_on_impact` in `engine/contacts.rs`).
    pub(super) fn update_sleep(&mut self, dt: f32) {
        let quiet = Self::collect_quiet_islands(self);
        let island_size = Self::collect_island_sizes(self);
        let mut to_sleep: Vec<u32> = Vec::new();
        for (root, q) in quiet {
            let sleep_time =
                Self::sleep_time_for_size(island_size.get(&root).copied().unwrap_or(1));
            let timer = self.island_timers.entry(root).or_insert(0.0);
            if q {
                *timer += dt;
                if *timer >= sleep_time {
                    to_sleep.push(root);
                }
            } else {
                *timer = 0.0;
            }
        }
        for root in to_sleep {
            for h in 0..self.bodies.len() {
                if self.island[h] == root {
                    self.asleep[h] = true;
                    let b = &mut self.bodies[h];
                    b.velocity = Vec3::ZERO;
                    b.angular_velocity = Vec3::ZERO;
                    // A sleeping body is STATIC for the solver (Jolt
                    // semantics): zero inverse mass/inertia makes every
                    // impulse and effective-mass computation treat it as
                    // immovable, so a resting contact with an awake body
                    // can never accumulate invisible velocity in the
                    // sleeper and detonate it on wake. Restored on wake.
                    b.inv_mass = 0.0;
                    b.inertia = Vec3::ZERO;
                }
            }
        }
    }

    /// Wake the whole island containing body `h` (contact with an awake body
    /// propagates motion through the island, so partial wake is incoherent).
    pub(super) fn wake_island(&mut self, h: usize) {
        let root = self.island[h];
        for h2 in 0..self.bodies.len() {
            if self.island[h2] == root {
                self.asleep[h2] = false;
                // Undo the sleep-time staticification (see update_sleep).
                let b = &mut self.bodies[h2];
                if b.body_type == BodyType::Dynamic {
                    b.inv_mass = 1.0 / b.mass;
                    b.inertia = b.shape.inertia(b.mass);
                }
            }
        }
        self.island_timers.insert(root, 0.0);
    }

    fn sleep_time_for_size(size: usize) -> f32 {
        (0.2 + 0.02 * size as f32).clamp(0.2, 0.6)
    }

    fn is_body_slow(b: &RigidBody) -> bool {
        const LIN_SLEEP: f32 = 0.15;
        const ANG_SLEEP: f32 = 0.15;
        b.velocity.length() < LIN_SLEEP && b.angular_velocity.length() < ANG_SLEEP
    }

    fn collect_quiet_islands(engine: &Self) -> HashMap<u32, bool> {
        let mut quiet: HashMap<u32, bool> = HashMap::new();
        for h in 0..engine.bodies.len() {
            if engine.island[h] == u32::MAX || engine.asleep[h] {
                continue;
            }
            let slow = Self::is_body_slow(&engine.bodies[h]);
            quiet
                .entry(engine.island[h])
                .and_modify(|q| *q &= slow)
                .or_insert(slow);
        }
        quiet
    }

    fn collect_island_sizes(engine: &Self) -> HashMap<u32, usize> {
        let mut m: HashMap<u32, usize> = HashMap::new();
        for &r in &engine.island {
            if r != u32::MAX {
                *m.entry(r).or_insert(0) += 1;
            }
        }
        m
    }

    /// Partition `active` (manifold indices) into islands and build work
    /// items. Extracted so both the CPU path and the GPU hybrid path reuse
    /// the same island-building logic.
    pub(super) fn partition_into_islands(
        &self,
        active: &[usize],
        manifolds: &[Manifold],
    ) -> Vec<IslandWork> {
        let n = self.bodies.len();
        let mut parent: Vec<usize> = (0..n).collect();
        for &mi in active {
            let m = &manifolds[mi];
            let (a, b) = (m.body_a, m.body_b);
            if self.bodies[a].body_type == BodyType::Dynamic
                && self.bodies[b].body_type == BodyType::Dynamic
            {
                let (ra, rb) = (union_find(&mut parent, a), union_find(&mut parent, b));
                if ra != rb {
                    parent[rb] = ra;
                }
            }
        }
        let mut group_of: HashMap<usize, usize> = HashMap::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for &mi in active {
            let m = &manifolds[mi];
            let d = if self.bodies[m.body_a].body_type == BodyType::Dynamic {
                m.body_a
            } else {
                m.body_b
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

        let mut islands: Vec<IslandWork> = Vec::with_capacity(groups.len());
        for group in groups {
            let mut body_idx: Vec<usize> = Vec::new();
            for &mi in &group {
                body_idx.push(manifolds[mi].body_a);
                body_idx.push(manifolds[mi].body_b);
            }
            body_idx.sort_unstable();
            body_idx.dedup();
            let shard: Vec<RigidBody> = body_idx.iter().map(|&g| self.bodies[g].clone()).collect();
            let local = |g: usize| body_idx.binary_search(&g).expect("island body");
            let island_manifolds: Vec<Manifold> = group
                .iter()
                .map(|&mi| {
                    let mut mc = manifolds[mi].clone();
                    mc.body_a = local(manifolds[mi].body_a);
                    mc.body_b = local(manifolds[mi].body_b);
                    mc
                })
                .collect();
            let keys: Vec<(usize, usize)> = group
                .iter()
                .map(|&mi| {
                    let m = &manifolds[mi];
                    (m.body_a.min(m.body_b), m.body_a.max(m.body_b))
                })
                .collect();
            islands.push(IslandWork {
                body_idx,
                bodies: shard,
                manifolds: island_manifolds,
                keys,
                states: Vec::new(),
                warm: HashMap::new(),
            });
        }
        islands
    }

    /// Dispatch the island velocity solves (parallel via rayon when wide
    /// enough), scatter bodies back, and merge warm caches.
    pub(super) fn dispatch_islands_velocity(
        &mut self,
        islands: &mut Vec<IslandWork>,
        allow_restitution: bool,
        sub_dt: f32,
        dt: f32,
    ) {
        const PAR_MIN_ISLANDS: usize = 2;
        const PAR_MIN_MANIFOLDS: usize = 24;
        if islands.is_empty() {
            return;
        }
        let parallel = islands.len() >= PAR_MIN_ISLANDS
            && islands.iter().map(|i| i.manifolds.len()).sum::<usize>() >= PAR_MIN_MANIFOLDS;
        let warm_in = &self.warm_impulses;
        let base_iters = self.velocity_iterations;
        let wide_on = self.wide_solver;
        // per-island adaptive iters: precompute outside the parallel closure
        // so we don't borrow `self` inside `par_iter_mut` (borrow checker).
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
                self.adaptive_iters_for_island_with_pen(max_speed, max_pen, dt, base_iters)
            })
            .collect();
        if parallel {
            islands.par_iter_mut().enumerate().for_each(|(idx, isl)| {
                let iters = iters_per_island[idx];
                let (states, warm) = Self::solve_island_velocity(
                    &mut isl.bodies,
                    &isl.manifolds,
                    &isl.keys,
                    warm_in,
                    iters,
                    allow_restitution,
                    sub_dt,
                    wide_on,
                );
                isl.states = states;
                isl.warm = warm;
            });
        } else {
            islands.iter_mut().enumerate().for_each(|(idx, isl)| {
                let iters = iters_per_island[idx];
                let (states, warm) = Self::solve_island_velocity(
                    &mut isl.bodies,
                    &isl.manifolds,
                    &isl.keys,
                    warm_in,
                    iters,
                    allow_restitution,
                    sub_dt,
                    wide_on,
                );
                isl.states = states;
                isl.warm = warm;
            });
        }
        let mut next: WarmCache = HashMap::new();
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
        union_dynamic_pair(parent, bodies, m.body_a, m.body_b);
    }
}

/// A fully sleeping island keeps its composition even if the contact detection
/// blinks for a step: its members are not integrated, so their relative
/// geometry cannot change — dissolving the island would let one member wake
/// while its support stays asleep. (One representative per old island, not an
/// O(n²) pair scan.)
#[allow(clippy::needless_range_loop)]
fn merge_sleeping_representations(parent: &mut [usize], asleep: &[bool], island: &[u32], n: usize) {
    let mut asleep_rep: HashMap<u32, usize> = HashMap::new();
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
fn assign_canonical_islands(engine: &mut BuiltinPhysicsEngine, parent: &mut [usize], n: usize) {
    let mut canonical: HashMap<usize, usize> = HashMap::new();
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
    let roots: std::collections::HashSet<u32> = engine.island.iter().copied().collect();
    engine.island_timers.retain(|r, _| roots.contains(r));
}
