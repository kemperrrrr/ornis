//! Contact solver stages for `BuiltinPhysicsEngine` (G2b-G7): warm-started
//! Gauss-Seidel velocity solve with block-LCP normals, one-shot restitution,
//! NGS position pass, and the island dispatch plumbing. Split out of
//! `engine.rs` to keep each type's method count within the structural
//! gate's thresholds.

#[cfg(feature = "gpu")]
use crate::gpu::{pack_single_point_batches, write_back_acc};

use super::*;

/// Warm-start / restitution policy constants, shared by the CPU island path
/// and the GPU single-point path (identical preamble semantics).
const MATCH_TOL_SQ: f32 = 0.05 * 0.05;
const RESTITUTION_THRESHOLD: f32 = 1.0;
const RESTITUTION_MAX_PEN: f32 = 0.05;

/// Best matching unused cached point for warm point `k` (feature persistence,
/// Jolt-style): nearest anchor within tolerance with a compatible normal.
#[allow(clippy::needless_range_loop)]
fn best_cached_point(
    cached_points: &[WarmPoint],
    cached_count: usize,
    used: &[bool; 4],
    la_k: Vec3,
    lb_k: Vec3,
    n: Vec3,
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (c, cp) in cached_points.iter().enumerate().take(cached_count) {
        if used[c] {
            continue;
        }
        // Feature compatibility: same surface region AND a compatible contact
        // normal (rolling over an edge changes the feature, dot < 0.7 => no match).
        if cp.normal.dot(n) < 0.7 {
            continue;
        }
        let d2 = (cp.la - la_k).length_squared() + (cp.lb - lb_k).length_squared();
        if d2 < MATCH_TOL_SQ && best.is_none_or(|(_, bd)| d2 < bd) {
            best = Some((c, d2));
        }
    }
    best.map(|(c, _)| c)
}

/// Match cached impulses by body-frame anchors: stable while the same surface
/// feature stays in contact, even when the bodies move fast in world space.
#[allow(clippy::needless_range_loop)]
fn match_warm_points(
    la: &[Vec3; 4],
    lb: &[Vec3; 4],
    n: Vec3,
    key: (usize, usize),
    warm_in: &WarmCache,
    count: usize,
) -> ([f32; 4], [bool; 4]) {
    let mut warm = [0.0f32; 4];
    let mut matched = [false; 4];
    if let Some((cached_points, cached_count)) = warm_in.get(&key) {
        let mut used = [false; 4];
        for k in 0..count {
            if let Some(c) = best_cached_point(cached_points, *cached_count, &used, la[k], lb[k], n)
            {
                used[c] = true;
                warm[k] = cached_points[c].impulse;
                matched[k] = true;
            }
        }
    }
    (warm, matched)
}

/// Speculative approach-speed target per point (G6): a separated point may
/// close its gap within this substep, but no more (Box2D speculative distance).
#[allow(clippy::needless_range_loop)]
fn speculative_targets(pen0: &[f32; 4], count: usize, sub_dt: f32) -> [f32; 4] {
    let mut target = [0.0f32; 4];
    for k in 0..count {
        if pen0[k] < 0.0 {
            target[k] = pen0[k] / sub_dt;
        }
    }
    target
}

/// Restitution bias from the pre-solve approach velocity — only on the first
/// substep of a step and only for NEW (unmatched) points: one bounce per
/// impact event. A persistent contact must never re-restitute — the NGS
/// position pass would feed it fresh approach velocity every step and the
/// bounce becomes an energy pump (Box3D applies restitution as a one-shot,
/// never cached).
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn compute_restitution_bias(
    bodies: &[RigidBody],
    m: &Manifold,
    matched: &[bool; 4],
    pen0: &[f32; 4],
    n: Vec3,
    e: f32,
    allow_restitution: bool,
    sub_dt: f32,
) -> [f32; 4] {
    let mut bias = [0.0f32; 4];
    if !allow_restitution {
        return bias;
    }
    let (i, j) = (m.body_a, m.body_b);
    for k in 0..m.point_count {
        if matched[k] || pen0[k] > RESTITUTION_MAX_PEN {
            continue;
        }
        let p = m.points[k].world_point;
        let ra = p - bodies[i].position;
        let rb = p - bodies[j].position;
        let vn0 = (point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra)).dot(n);
        if vn0 >= -RESTITUTION_THRESHOLD {
            continue;
        }
        // A speculative point restitutes only if the approach is fast enough
        // to actually land within this substep — otherwise the bounce would
        // fire in mid-air.
        if pen0[k] < 0.0 && -pen0[k] > -vn0 * sub_dt {
            continue;
        }
        bias[k] = -e * vn0;
    }
    bias
}

/// WarmStart stage: apply cached impulses once (Box2D pattern). Capped so the
/// warm impulse can never push the pair APART faster than they currently
/// approach: a stale cached impulse applied to a separating (or nearly static)
/// contact is pure energy injection, repeated 240x/s (this was the high-spin
/// pump).
#[allow(clippy::needless_range_loop)]
fn apply_warm_start(
    bodies: &mut [RigidBody],
    m: &Manifold,
    i: usize,
    j: usize,
    n: Vec3,
    warm: &[f32; 4],
    target: &[f32; 4],
) -> [f32; 4] {
    let mut warm_applied = *warm;
    for k in 0..m.point_count {
        if warm[k] > 0.0 {
            let p = m.points[k].world_point;
            let ra = p - bodies[i].position;
            let rb = p - bodies[j].position;
            let k_eff = effective_mass(bodies, i, j, n, ra, rb);
            if k_eff < 1e-10 {
                warm_applied[k] = 0.0;
                continue;
            }
            let vn_pre = (point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra)).dot(n);
            // Cap against the speculative target too: a separated point may
            // keep approaching up to its gap limit.
            let applied = warm[k].min(((target[k] - vn_pre) / k_eff).max(0.0));
            warm_applied[k] = applied;
            if applied > 0.0 {
                apply_impulse(bodies, i, j, n * applied, ra, rb);
            }
        }
    }
    warm_applied
}

/// Build one ManifoldState for a manifold at global body indices taken from
/// `m`. Shared preamble of the CPU island path (`solve_island_velocity`) and
/// the GPU single-point path (`build_manifold_state`). `key` is the sorted
/// global body-pair for warm-cache lookup.
#[allow(clippy::needless_range_loop)]
fn prepare_manifold_state(
    bodies: &mut [RigidBody],
    m: &Manifold,
    key: (usize, usize),
    warm_in: &WarmCache,
    allow_restitution: bool,
    sub_dt: f32,
    mi: usize,
) -> Option<ManifoldState> {
    let (i, j) = (m.body_a, m.body_b);
    let total_inv = bodies[i].inv_mass + bodies[j].inv_mass;
    if total_inv < 1e-10 {
        return None;
    }
    let n = m.normal;
    let count = m.point_count;

    // --- Body-frame anchors first: matching and G3 both need them ---
    let mut la = [Vec3::ZERO; 4];
    let mut lb = [Vec3::ZERO; 4];
    let mut pen0 = [0.0f32; 4];
    for k in 0..count {
        let p = m.points[k].world_point;
        la[k] = bodies[i].orientation.inverse() * (p - bodies[i].position);
        lb[k] = bodies[j].orientation.inverse() * (p - bodies[j].position);
        pen0[k] = m.points[k].penetration;
    }

    let (warm, matched) = match_warm_points(&la, &lb, n, key, warm_in, count);

    let e = bodies[i].restitution.min(bodies[j].restitution);
    let mu = bodies[i].friction.max(bodies[j].friction);
    let target = speculative_targets(&pen0, count, sub_dt);
    let bias =
        compute_restitution_bias(bodies, m, &matched, &pen0, n, e, allow_restitution, sub_dt);
    let warm_applied = apply_warm_start(bodies, m, i, j, n, &warm, &target);

    Some(ManifoldState {
        mi,
        i,
        j,
        count,
        acc: warm_applied,
        acc_friction: [0.0; 4],
        acc_friction2: [0.0; 4],
        bias,
        target,
        mu,
        t1: tangent_basis(n),
        t2: tangent_basis(n).cross(n),
        la,
        lb,
        pen0,
    })
}

impl BuiltinPhysicsEngine {
    /// Build one ManifoldState entry for a manifold at global body indices
    /// `i`/`j`. Thin wrapper over the shared `prepare_manifold_state`
    /// preamble; today only the GPU single-point path calls it.
    #[allow(clippy::needless_range_loop)]
    #[allow(dead_code)]
    pub(super) fn build_manifold_state(
        ctx: &mut ManifoldCtx,
        m: &Manifold,
        key: (usize, usize),
    ) -> Option<ManifoldState> {
        prepare_manifold_state(
            &mut *ctx.bodies,
            m,
            key,
            ctx.warm_in,
            ctx.allow_restitution,
            ctx.sub_dt,
            ctx.mi,
        )
    }

    /// Contact velocity solve using the GPU for single-point manifolds
    /// (G7, `gpu` feature). Multi-point manifolds are dispatched on CPU
    /// islands. This is a Jacobi/GS hybrid (not bit-identical).
    #[cfg(feature = "gpu")]
    // Warm-point packing indexes parallel per-point arrays; a range loop is
    // the clearest form here (same style as the scalar solver).
    #[allow(clippy::needless_range_loop)]
    pub(super) fn solve_contacts_velocity_gpu(
        &mut self,
        active: Vec<usize>,
        manifolds: &[Manifold],
        allow_restitution: bool,
        sub_dt: f32,
        dt: f32,
    ) -> Vec<IslandWork> {
        // Build global states for ALL active manifolds (shared preamble).
        let mut global_states: Vec<ManifoldState> = Vec::with_capacity(active.len());
        for &mi in &active {
            let m = &manifolds[mi];
            let (i, j) = (m.body_a, m.body_b);
            let key = (i.min(j), i.max(j));
            let mut ctx = ManifoldCtx {
                bodies: &mut self.bodies,
                warm_in: &self.warm_impulses,
                allow_restitution,
                sub_dt,
                mi,
                i,
                j,
            };
            if let Some(st) = Self::build_manifold_state(&mut ctx, m, key) {
                global_states.push(st);
            }
        }

        // Split: single-point → GPU, multi-point → CPU islands.
        let mut single_si: Vec<usize> = Vec::new();
        let mut multi_mi: Vec<usize> = Vec::new();
        for (si, st) in global_states.iter().enumerate() {
            if st.count == 1 {
                single_si.push(si);
            } else {
                multi_mi.push(st.mi);
            }
        }

        // GPU solve single-point contacts.
        let mut gpu_warm: WarmCache = HashMap::new();
        if !single_si.is_empty() {
            let gpu = self.gpu_solver.as_mut().unwrap();
            let (batches, num_batches) =
                pack_single_point_batches(&self.bodies, &global_states, manifolds, &single_si);
            if num_batches > 0 {
                gpu.upload_bodies(&self.bodies);
                gpu.upload_batches(&batches);
                gpu.solve(num_batches, self.velocity_iterations, allow_restitution);
                gpu.download_bodies(&mut self.bodies);
                let mut dl_batches = batches;
                gpu.download_acc(&mut dl_batches);
                write_back_acc(&mut global_states, &single_si, &dl_batches);
                // Persist warm cache for single-point manifolds.
                for &si in &single_si {
                    let st = &global_states[si];
                    let m = &manifolds[st.mi];
                    let key = (m.body_a.min(m.body_b), m.body_a.max(m.body_b));
                    let mut pts = [WarmPoint {
                        la: Vec3::ZERO,
                        lb: Vec3::ZERO,
                        normal: Vec3::ZERO,
                        impulse: 0.0,
                    }; 4];
                    for k in 0..st.count {
                        pts[k] = WarmPoint {
                            la: st.la[k],
                            lb: st.lb[k],
                            normal: m.normal,
                            impulse: st.acc[k],
                        };
                    }
                    gpu_warm.insert(key, (pts, st.count));
                }
            }
        }

        // CPU islands for multi-point manifolds. The island dispatch replaces
        // `warm_impulses` with the multi-point cache, so the GPU entries are
        // merged back in afterwards.
        let islands = if multi_mi.is_empty() {
            Vec::new()
        } else {
            let mut islands = self.partition_into_islands(&multi_mi, manifolds);
            self.dispatch_islands_velocity(&mut islands, allow_restitution, sub_dt, dt);
            islands
        };
        self.warm_impulses.extend(gpu_warm);
        islands
    }

    /// Velocity stage of the contact solver (G6 stage order, G7 island
    /// dispatch). Orchestrator: a sequential sleep/wake pre-pass (the only
    /// part mutating island state), a union-find partition over the FRESH
    /// manifolds, then each island solved independently by
    /// `solve_island_velocity` — in parallel via rayon when the scene is wide
    /// enough. Islands are disjoint over dynamic bodies by construction, so
    /// concurrent solves are race-free and bit-identical for any thread
    /// count (Strong Confluence). Returns the island work items; the position
    /// stage reuses them (states + remapped manifolds) after integration.
    pub(super) fn solve_contacts_velocity(
        &mut self,
        manifolds: &[Manifold],
        allow_restitution: bool,
        sub_dt: f32,
        dt: f32,
    ) -> Vec<IslandWork> {
        let active = self.collect_active_manifolds(manifolds);
        if active.is_empty() {
            self.warm_impulses.clear();
            return Vec::new();
        }

        // --- GPU single-point path (G7) ---
        // When a GPU solver is attached, the whole velocity solve for
        // single-point manifolds moves to the GPU (`solve_contacts_velocity_gpu`),
        // and multi-point manifolds keep the CPU island path. The GPU and CPU
        // passes are NOT interleaved per Gauss-Seidel iteration (they run
        // sequentially per substep) — a Jacobi/GS hybrid that is physically
        // correct but not bit-identical to the pure CPU path (see PLAN.md).
        #[cfg(feature = "gpu")]
        if self.gpu_solver.is_some() {
            return self.solve_contacts_velocity_gpu(
                active,
                manifolds,
                allow_restitution,
                sub_dt,
                dt,
            );
        }

        // --- Partition into islands + dispatch (G7) ---
        // Islands are disjoint over dynamic bodies by construction, so
        // concurrent solves are race-free and bit-identical for any thread
        // count (Strong Confluence).
        let mut islands = self.partition_into_islands(&active, manifolds);
        self.dispatch_islands_velocity(&mut islands, allow_restitution, sub_dt, dt);
        islands
    }

    /// Position stage (G3 split impulse, G6 order, G7 island dispatch): runs
    /// AFTER positions are integrated. Iterated NGS per island;
    /// pseudo-motion only — real velocities are never touched. Each iteration
    /// re-measures the LIVE separation at the stored body-frame anchors, so
    /// corrections distribute evenly across the manifold set instead of
    /// one-shot rigid pushes. β is kept low (0.2): stronger pseudo-correction
    /// resonates with the velocity solve on rocking contacts.
    pub(super) fn solve_contacts_position(&mut self, islands: &mut [IslandWork], dt: f32) {
        const PAR_MIN_ISLANDS: usize = 2;
        const PAR_MIN_MANIFOLDS: usize = 24;
        if islands.is_empty() {
            return;
        }
        // Re-gather: integration, TOI clamps and the joint velocity pass all
        // moved the main array since the velocity stage ran.
        for isl in islands.iter_mut() {
            for (l, &g) in isl.body_idx.iter().enumerate() {
                isl.bodies[l] = self.bodies[g].clone();
            }
        }
        let base_iters = self.position_iterations;
        let softness = self.contact_softness;
        let total_manifolds: usize = islands.iter().map(|i| i.manifolds.len()).sum();
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
        if islands.len() >= PAR_MIN_ISLANDS && total_manifolds >= PAR_MIN_MANIFOLDS {
            islands.par_iter_mut().enumerate().for_each(|(idx, isl)| {
                let iters = iters_per_island[idx];
                Self::solve_island_position(
                    &mut isl.bodies,
                    &isl.manifolds,
                    &isl.states,
                    iters,
                    softness,
                );
            });
        } else {
            islands.iter_mut().enumerate().for_each(|(idx, isl)| {
                let iters = iters_per_island[idx];
                Self::solve_island_position(
                    &mut isl.bodies,
                    &isl.manifolds,
                    &isl.states,
                    iters,
                    softness,
                );
            });
        }
        for isl in islands.iter() {
            for (l, &g) in isl.body_idx.iter().enumerate() {
                if self.bodies[g].body_type == BodyType::Dynamic {
                    self.bodies[g] = isl.bodies[l].clone();
                }
            }
        }
    }

    /// Wake the sleeper of a manifold pair iff the awake partner closes in faster
    /// than `threshold` (impact hysteresis; see `collect_active_manifolds`).
    fn wake_on_impact(&mut self, m: &Manifold, threshold: f32) {
        let (i, j) = (m.body_a, m.body_b);
        let (s, o) = if self.asleep[i] { (i, j) } else { (j, i) };
        let p = m.points[0].world_point;
        let rs = p - self.bodies[s].position;
        let ro = p - self.bodies[o].position;
        let approach = (point_velocity(&self.bodies[o], ro) - point_velocity(&self.bodies[s], rs))
            .dot(m.normal)
            * if self.asleep[i] { -1.0 } else { 1.0 };
        // m.normal points i → j; `approach` is the speed at which the
        // awake partner closes in on the sleeper (sleep velocities are
        // zeroed, so this is just the partner's normal speed).
        if approach > threshold {
            self.wake_island(s);
        }
    }

    /// Sequential pre-pass of the contact velocity stage: sleep/wake policy +
    /// active-manifold filtering. The only part of the stage that mutates
    /// island state; runs before any parallel island work.
    fn collect_active_manifolds(&mut self, manifolds: &[Manifold]) -> Vec<usize> {
        // Sleep: a contact needs work only if at least one side is an AWAKE
        // DYNAMIC body. Static geometry never wakes anything (a body asleep
        // on the floor must stay asleep).
        const WAKE_IMPACT_SPEED: f32 = 0.5;
        let mut active: Vec<usize> = Vec::with_capacity(manifolds.len());
        for (mi, m) in manifolds.iter().enumerate() {
            let (i, j) = (m.body_a, m.body_b);
            let ai = self.asleep[i] || self.bodies[i].body_type != BodyType::Dynamic;
            let aj = self.asleep[j] || self.bodies[j].body_type != BodyType::Dynamic;
            if ai && aj {
                continue;
            }
            // Wake hysteresis (G7): a sleeping island is woken only by a
            // genuine IMPACT — approach speed above the threshold. A resting
            // micro-jitter contact (vn ≈ 0) must NOT wake it.
            if (self.asleep[i] && !aj) || (self.asleep[j] && !ai) {
                self.wake_on_impact(m, WAKE_IMPACT_SPEED);
            }
            // A still-sleeping body is static for the solver (its inv_mass is
            // zeroed at sleep), so sleeper+static pairs carry no work.
            if self.bodies[i].inv_mass + self.bodies[j].inv_mass < 1e-10 {
                continue;
            }
            active.push(mi);
        }
        active
    }

    /// Per-island velocity solve: the G2b–G6 inner solver (warm start,
    /// Gauss-Seidel with block-LCP normals and fixed-basis friction, one-shot
    /// restitution, cache persist), operating on an island-local body shard.
    /// All body indices in `manifolds` and the returned states are LOCAL;
    /// `keys` maps each local manifold to its global body-pair warm-cache key.
    /// When `use_wide` is true, single-point manifolds are solved in
    /// SIMD-wide batches (G7); multi-point (block LCP) stays scalar.
    // The 8th parameter (`use_wide`, G7) tips this over clippy's default
    // 7-argument limit; packing them into a struct would only add churn.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve_island_velocity(
        bodies: &mut [RigidBody],
        manifolds: &[Manifold],
        keys: &[(usize, usize)],
        warm_in: &WarmCache,
        velocity_iterations: u32,
        allow_restitution: bool,
        sub_dt: f32,
        use_wide: bool,
    ) -> (Vec<ManifoldState>, WarmCache) {
        let mut states =
            prepare_island_states(bodies, manifolds, keys, warm_in, allow_restitution, sub_dt);
        // --- Velocity solve: Gauss-Seidel iterations over ALL manifolds ---
        // G7: single-point manifolds are packed into SIMD-wide batches
        // (disjoint body sets, original GS order preserved — every contact
        // stays in its place in the sequence); multi-point manifolds keep
        // the scalar block-LCP path. Steps run in manifold order, so the
        // computation is the same sequence either way.
        let mut steps = if use_wide {
            build_solver_steps(bodies, manifolds, &states)
        } else {
            Vec::new()
        };
        run_velocity_iterations(
            bodies,
            manifolds,
            &mut states,
            &mut steps,
            velocity_iterations,
            use_wide,
        );
        // Wide batches own the accumulated impulses of their lanes during
        // the iterations; write them back so the cache persist below sees
        // the final values.
        if use_wide {
            for step in &steps {
                if let SolverStep::Wide(b) = step {
                    b.write_back_acc(&mut states);
                }
            }
        }
        // --- Restitution stage (Box3D b3SolverStage_Restitution analog) ---
        // One-shot per step: push the normal point velocity up to the stored
        // bounce target. NOT accumulated, NOT warm-started — this is what
        // keeps spinning bodies from pumping energy through the bounce.
        if allow_restitution {
            run_restitution_stage(bodies, manifolds, &states, &mut steps, use_wide);
        }
        let next = persist_warm_cache(manifolds, keys, &states);
        (states, next)
    }

    /// One scalar Gauss-Seidel velocity step for a single manifold: the
    /// block-LCP normal solve (multi-point) or the projected scalar solve
    /// (single-point), then Coulomb friction along the fixed tangent basis.
    /// Extracted from the iteration loop so the G7 step sequence (wide
    /// batches interleaved with scalar manifolds) reuses the exact same code.
    // Solver loops index several parallel per-point arrays; range loops are
    // the clearest form here.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn solve_scalar_velocity_step(
        bodies: &mut [RigidBody],
        manifolds: &[Manifold],
        st: &mut ManifoldState,
    ) {
        let m = &manifolds[st.mi];
        let (i, j) = (st.i, st.j);
        let n = m.normal;
        let total_inv = bodies[i].inv_mass + bodies[j].inv_mass;

        // ---- Normal direction ----
        // G4: multi-point manifolds are solved as an exact LCP block
        // (scalar per-point GS oscillates between coupled points of one
        // manifold — the rocking pump); single points keep the scalar
        // projected update.
        if st.count >= 2 {
            let mut pts = [Vec3::ZERO; 4];
            for k in 0..st.count {
                pts[k] = m.points[k].world_point;
            }
            solve_normal_block(bodies, i, j, n, &pts, &mut st.acc, &st.target, st.count);
        } else {
            let k = 0;
            let p = m.points[k].world_point;
            let ra = p - bodies[i].position;
            let rb = p - bodies[j].position;
            let k_eff = effective_mass(bodies, i, j, n, ra, rb);
            if k_eff >= 1e-10 {
                let rel = point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra);
                let vn = rel.dot(n);
                // Inelastic contact: restitution is a separate one-shot stage
                // (below), never accumulated. G6: the target is the
                // speculative approach limit (0 when touching), not
                // necessarily a full stop.
                let lambda = (st.target[k] - vn) / k_eff;
                let new_acc = (st.acc[k] + lambda).max(0.0);
                let delta = new_acc - st.acc[k];
                st.acc[k] = new_acc;
                if delta.abs() > 1e-12 {
                    apply_impulse(bodies, i, j, n * delta, ra, rb);
                }
            }
        }

        // Friction (Coulomb) along the FIXED tangent basis (extracted helper).
        Self::solve_scalar_friction(bodies, i, j, st, m, total_inv);
    }

    /// One-shot restitution step for a single manifold (the scalar half of
    /// the post-iteration stage; the wide half lives in `WideBatch`).
    #[allow(clippy::needless_range_loop)]
    pub(super) fn solve_scalar_restitution_step(
        bodies: &mut [RigidBody],
        manifolds: &[Manifold],
        st: &ManifoldState,
    ) {
        let m = &manifolds[st.mi];
        let (i, j) = (st.i, st.j);
        let n = m.normal;
        let total_inv = bodies[i].inv_mass + bodies[j].inv_mass;
        for k in 0..st.count {
            if st.bias[k] <= 0.0 {
                continue;
            }
            let p = m.points[k].world_point;
            let ra = p - bodies[i].position;
            let rb = p - bodies[j].position;
            let ra_n = ra.cross(n);
            let rb_n = rb.cross(n);
            let k_eff = total_inv
                + ra_n.dot(mul_inv_inertia(
                    bodies[i].inertia,
                    bodies[i].orientation,
                    ra_n,
                ))
                + rb_n.dot(mul_inv_inertia(
                    bodies[j].inertia,
                    bodies[j].orientation,
                    rb_n,
                ));
            if k_eff < 1e-10 {
                continue;
            }
            let vn = (point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra)).dot(n);
            let lambda = (st.bias[k] - vn) / k_eff;
            if lambda > 0.0 {
                apply_impulse(bodies, i, j, n * lambda, ra, rb);
            }
        }
    }

    /// Friction step for a single manifold (extracted to reduce
    /// cognitive complexity of solve_scalar_velocity_step).
    #[allow(clippy::needless_range_loop)]
    pub(super) fn solve_scalar_friction(
        bodies: &mut [RigidBody],
        i: usize,
        j: usize,
        st: &mut ManifoldState,
        m: &Manifold,
        total_inv: f32,
    ) {
        for k in 0..st.count {
            debug_assert!(st.acc[k].is_finite(), "acc must be finite");
            debug_assert!(
                st.mu.is_finite() && st.mu >= 0.0,
                "mu must be non-negative, got {}",
                st.mu
            );
            let p = m.points[k].world_point;
            let ra = p - bodies[i].position;
            let rb = p - bodies[j].position;
            let rel = point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra);
            let max_friction = st.mu * st.acc[k];
            let mut f_imp = Vec3::ZERO;
            for axis in 0..2 {
                let t = if axis == 0 { st.t1 } else { st.t2 };
                let k_t = total_inv + tangent_effective_mass(bodies, i, j, ra, rb, t);
                if k_t < 1e-10 {
                    continue;
                }
                let vt = rel.dot(t);
                let lambda_t = -vt / k_t;
                let (cur, other) = if axis == 0 {
                    (st.acc_friction[k], st.acc_friction2[k])
                } else {
                    (st.acc_friction2[k], st.acc_friction[k])
                };
                let new_t = clamp_friction_impulse(cur + lambda_t, other, max_friction);
                debug_assert!(new_t.is_finite(), "friction impulse overflowed");
                if axis == 0 {
                    f_imp += t * (new_t - st.acc_friction[k]);
                    st.acc_friction[k] = new_t;
                } else {
                    f_imp += t * (new_t - st.acc_friction2[k]);
                    st.acc_friction2[k] = new_t;
                }
            }
            if f_imp.length_squared() > 1e-24 {
                apply_impulse(bodies, i, j, f_imp, ra, rb);
            }
        }
    }

    /// Per-island NGS position solve on the local shard (stage doc lives on
    /// `solve_contacts_position`). Pseudo-motion only: real velocities and
    /// the warm cache are never touched here.
    // Live anchors are re-measured per iteration; range loops over the
    // per-point arrays are the clearest form here.
    #[allow(clippy::needless_range_loop)]
    pub(super) fn solve_island_position(
        bodies: &mut [RigidBody],
        manifolds: &[Manifold],
        states: &[ManifoldState],
        position_iterations: u32,
        contact_softness: f32,
    ) {
        const SLOP: f32 = 0.02;
        const MAX_CORRECTION: f32 = 0.25;
        const BETA_POS: f32 = 0.2;
        for _ in 0..position_iterations {
            for st in states {
                let m = &manifolds[st.mi];
                let (i, j) = (st.i, st.j);
                let n = m.normal;
                let inv_mass_a = bodies[i].inv_mass;
                let inv_mass_b = bodies[j].inv_mass;
                let total_inv = inv_mass_a + inv_mass_b;
                let cfm = contact_softness * total_inv;
                for k in 0..st.count {
                    let (pos_a, rot_a) = (bodies[i].position, bodies[i].orientation);
                    let (pos_b, rot_b) = (bodies[j].position, bodies[j].orientation);
                    // Live world anchors; at detection they coincided, so the
                    // separation along n started at -pen0.
                    let wa = pos_a + rot_a * st.la[k];
                    let wb = pos_b + rot_b * st.lb[k];
                    let separation = (wb - wa).dot(n) - st.pen0[k];
                    let c = (-separation - SLOP).clamp(0.0, MAX_CORRECTION);
                    if c <= 0.0 {
                        continue;
                    }
                    let ra = wa - pos_a;
                    let rb = wb - pos_b;
                    let ra_n = ra.cross(n);
                    let rb_n = rb.cross(n);
                    let k_pos = total_inv
                        + ra_n.dot(mul_inv_inertia(bodies[i].inertia, rot_a, ra_n))
                        + rb_n.dot(mul_inv_inertia(bodies[j].inertia, rot_b, rb_n));
                    let k_soft = make_soft(k_pos, cfm);
                    if k_soft < 1e-10 {
                        continue;
                    }
                    let lam = BETA_POS * c / k_soft;
                    apply_positional_impulse(bodies, i, j, n * lam, ra, rb);
                }
            }
        }
    }
}

/// Build the per-manifold solver states for every solvable manifold of an
/// island (warm-start matching, restitution bias, capped warm start). Runs
/// before any iteration; identical preamble for wide and scalar paths.
fn prepare_island_states(
    bodies: &mut [RigidBody],
    manifolds: &[Manifold],
    keys: &[(usize, usize)],
    warm_in: &WarmCache,
    allow_restitution: bool,
    sub_dt: f32,
) -> Vec<ManifoldState> {
    // G2b: warm-start cache matches points by proximity, not by index —
    // manifold point order changes frame to frame (sorted by depth).
    let mut states: Vec<ManifoldState> = Vec::with_capacity(manifolds.len());
    for (mi, m) in manifolds.iter().enumerate() {
        let key = keys[mi];
        if let Some(st) =
            prepare_manifold_state(bodies, m, key, warm_in, allow_restitution, sub_dt, mi)
        {
            states.push(st);
        }
    }
    states
}

/// Gauss-Seidel iterations over ALL manifolds: wide batches interleaved with
/// scalar manifolds when `use_wide`, plain scalar sequence otherwise. Both
/// orders visit the contacts in manifold order, so results are identical.
fn run_velocity_iterations(
    bodies: &mut [RigidBody],
    manifolds: &[Manifold],
    states: &mut [ManifoldState],
    steps: &mut [SolverStep],
    velocity_iterations: u32,
    use_wide: bool,
) {
    for _ in 0..velocity_iterations {
        if use_wide {
            for step in steps.iter_mut() {
                match step {
                    SolverStep::Wide(b) => {
                        b.gather(bodies);
                        b.solve_iteration();
                        b.scatter(bodies);
                    }
                    SolverStep::Scalar(si) => {
                        BuiltinPhysicsEngine::solve_scalar_velocity_step(
                            bodies,
                            manifolds,
                            &mut states[*si],
                        );
                    }
                }
            }
        } else {
            for st in states.iter_mut() {
                BuiltinPhysicsEngine::solve_scalar_velocity_step(bodies, manifolds, st);
            }
        }
    }
}

/// One-shot restitution stage across the island (wide + scalar halves).
fn run_restitution_stage(
    bodies: &mut [RigidBody],
    manifolds: &[Manifold],
    states: &[ManifoldState],
    steps: &mut [SolverStep],
    use_wide: bool,
) {
    if use_wide {
        for step in steps.iter_mut() {
            match step {
                SolverStep::Wide(b) => {
                    b.gather(bodies);
                    b.solve_restitution();
                    b.scatter(bodies);
                }
                SolverStep::Scalar(si) => {
                    BuiltinPhysicsEngine::solve_scalar_restitution_step(
                        bodies,
                        manifolds,
                        &states[*si],
                    );
                }
            }
        }
    } else {
        for st in states {
            BuiltinPhysicsEngine::solve_scalar_restitution_step(bodies, manifolds, st);
        }
    }
}

/// Persist this island's cache entries for the next substep (st.i/st.j are
/// island-LOCAL indices; the cache is keyed globally).
#[allow(clippy::needless_range_loop)]
fn persist_warm_cache(
    manifolds: &[Manifold],
    keys: &[(usize, usize)],
    states: &[ManifoldState],
) -> WarmCache {
    let mut next: WarmCache = HashMap::new();
    for st in states {
        let m = &manifolds[st.mi];
        let mut pts = [WarmPoint {
            la: Vec3::ZERO,
            lb: Vec3::ZERO,
            normal: Vec3::ZERO,
            impulse: 0.0,
        }; 4];
        for k in 0..st.count {
            pts[k] = WarmPoint {
                la: st.la[k],
                lb: st.lb[k],
                normal: m.normal,
                impulse: st.acc[k],
            };
        }
        next.insert(keys[st.mi], (pts, st.count));
    }
    next
}

/// Effective inverse mass of a tangent-direction friction constraint at one
/// contact point with levers `ra`/`rb` along tangent `t`.
fn tangent_effective_mass(
    bodies: &[RigidBody],
    i: usize,
    j: usize,
    ra: Vec3,
    rb: Vec3,
    t: Vec3,
) -> f32 {
    let ra_t = ra.cross(t);
    let rb_t = rb.cross(t);
    ra_t.dot(mul_inv_inertia(
        bodies[i].inertia,
        bodies[i].orientation,
        ra_t,
    )) + rb_t.dot(mul_inv_inertia(
        bodies[j].inertia,
        bodies[j].orientation,
        rb_t,
    ))
}

/// Coulomb cone projection for one friction axis given the accumulated
/// impulse on the perpendicular axis.
fn clamp_friction_impulse(new_t: f32, other: f32, max_friction: f32) -> f32 {
    let len = (new_t * new_t + other * other).sqrt();
    if len > max_friction && len > 1e-12 {
        new_t * (max_friction / len)
    } else {
        new_t
    }
}
