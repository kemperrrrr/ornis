//! Contact solver stages for `SequentialImpulseEngine` (G2b-G7): warm-started
//! Gauss-Seidel velocity solve with block-LCP normals, one-shot restitution,
//! NGS position pass, and the island dispatch plumbing. Split out of
//! `engine.rs` to keep each type's method count within the structural
//! gate's thresholds.

#[cfg(feature = "gpu")]
use crate::gpu::{pack_single_point_batches, write_back_acc};

use rustc_hash::FxHashMap;

use super::*;
use crate::constants::{
    DEGENERATE_LEN2, FEATURE_NORMAL_DOT_MIN, FRICTION_IMPULSE_DUST, MIN_EFFECTIVE_MASS,
};
use crate::contact_math::{contact_friction_clamp, contact_normal_step};
use crate::flags::{Dispatch, RestitutionGate, RollAxis, SolvePath};

/// Warm-start / restitution policy constants, shared by the CPU island path
/// and the GPU single-point path (identical preamble semantics).
const MATCH_TOL_SQ: f32 = 0.05 * 0.05;
const RESTITUTION_THRESHOLD: f32 = 1.0;
/// Cached normal impulse above this means the point already carries load,
/// so it must not bounce again. A speculative preview caches zero.
const LOADED_IMPULSE: f32 = 1e-3;
/// Min total manifolds before flat/island contact stages go parallel.
const PARALLEL_MIN_MANIFOLDS: usize = 24;

/// Best matching unused cached point for warm point `k` (feature persistence,
/// Jolt-style): nearest anchor within tolerance with a compatible normal.
#[allow(clippy::needless_range_loop)]
fn best_cached_point(
    cached_points: &[WarmPoint],
    cached_count: usize,
    used: &[bool; MAX_MANIFOLD_POINTS],
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
        if cp.normal.dot(n) < FEATURE_NORMAL_DOT_MIN {
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
    la: &[Vec3; MAX_MANIFOLD_POINTS],
    lb: &[Vec3; MAX_MANIFOLD_POINTS],
    n: Vec3,
    key: (usize, usize),
    warm_in: &WarmCache,
    count: usize,
) -> [f32; MAX_MANIFOLD_POINTS] {
    let mut warm = [0.0f32; MAX_MANIFOLD_POINTS];
    if let Some((cached_points, cached_count)) = warm_in.get(&key) {
        let mut used = [false; MAX_MANIFOLD_POINTS];
        for k in 0..count {
            if let Some(c) = best_cached_point(cached_points, *cached_count, &used, la[k], lb[k], n)
            {
                used[c] = true;
                warm[k] = cached_points[c].impulse;
            }
        }
    }
    warm
}

/// Speculative approach-speed target per point (G6): a separated point may
/// close its gap within this substep, but no more (Box2D speculative distance).
#[allow(clippy::needless_range_loop)]
fn speculative_targets(
    pen0: &[f32; MAX_MANIFOLD_POINTS],
    count: usize,
    sub_dt: f32,
) -> [f32; MAX_MANIFOLD_POINTS] {
    let mut target = [0.0f32; MAX_MANIFOLD_POINTS];
    for k in 0..count {
        if pen0[k] < 0.0 {
            target[k] = pen0[k] / sub_dt;
        }
    }
    target
}

/// Restitution bias from the pre-solve approach velocity: one bounce per
/// impact. A pair that already carries any cached normal impulse must
/// never re-restitute — the position pass would feed it fresh approach
/// velocity and the bounce becomes an energy pump (Box3D applies
/// restitution as a one-shot, never cached). Point identity is not the
/// test: a tall stack shuffles manifold points every substep, and
/// treating each shuffle as a new impact launches the tower.
///
/// A cache whose impulses are all ~0 is only a speculative preview of
/// the same feature; the real landing still bounces. Approach slower
/// than [`RESTITUTION_THRESHOLD`] never bounces, so a body spawned already
/// buried (contact speed ~0) stays put even when the overlap is deep.
/// Penetration depth is not a separate cap: a fast point can sink past
/// 5 cm in one substep and must still rebound.
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn compute_restitution_bias(
    bodies: &[RigidBody],
    m: &Manifold,
    pair_loaded: bool,
    pen0: &[f32; MAX_MANIFOLD_POINTS],
    n: Vec3,
    e: f32,
    gate: RestitutionGate,
    sub_dt: f32,
) -> [f32; MAX_MANIFOLD_POINTS] {
    let mut bias = [0.0f32; MAX_MANIFOLD_POINTS];
    if !gate.is_enabled() || pair_loaded {
        return bias;
    }
    let (i, j) = (m.body_a.index(), m.body_b.index());
    for k in 0..m.point_count {
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

/// Minimum island manifold count for the tall-stack solver path (see
/// [`STACK_PATH_MIN_MANIFOLDS`]): islands at or above this size solve
/// resting contacts with full warm support and unscaled iteration budgets.
/// Every pinned scene stays below it — the determinism snapshot (4-stack
/// island) and `tall_stack_stands_still` (5) — so those trajectories stay
/// bit-identical while deeper chains (whose support must propagate O(depth)
/// levels per sweep; 6-box towers already scatter on the downscaled path)
/// get the budget they need. The tiled 10k probe routes through the flat
/// singleton path, never through here.
pub(crate) const STACK_PATH_MIN_MANIFOLDS: usize = 6;

/// Tall-stack path predicate: true for islands whose constraint chain is
/// deep enough that per-island iteration downscaling starves support
/// propagation. Kept in one place so the velocity, warm-start and position
/// stages gate identically.
#[inline]
pub(crate) fn stack_path_for_island(manifold_count: usize) -> bool {
    manifold_count >= STACK_PATH_MIN_MANIFOLDS
}

/// Cap on stack-path velocity iters as a multiple of the base count.
const STACK_ITERS_CAP_MUL: u32 = 3;

/// Tall-stack velocity budget for an island with `manifold_count` manifolds:
/// the full base budget (never the resting downscale) plus one extra sweep
/// per two chain levels above the gate — support propagates a few levels per
/// Gauss-Seidel sweep, and the steady-state warm start (see
/// [`apply_warm_start`]) already carries the bulk load, so the iterations
/// only chase the per-substep delta. Capped at 3× base: tall islands are
/// body-count small, and an unbounded count would let one pathological
/// island eat the frame.
#[inline]
pub(crate) fn stack_velocity_iters(base_iters: u32, manifold_count: usize) -> u32 {
    let extra = manifold_count.saturating_sub(STACK_PATH_MIN_MANIFOLDS) as u32 / 2;
    base_iters.saturating_add(extra).min(
        base_iters
            .saturating_mul(STACK_ITERS_CAP_MUL)
            .max(base_iters),
    )
}

/// WarmStart stage: apply cached impulses once (Box2D pattern). Capped so the
/// warm impulse can never push the pair APART faster than they currently
/// approach: a stale cached impulse applied to a separating (or nearly static)
/// contact is pure energy injection, repeated 240x/s (this was the high-spin
/// pump).
///
/// `full_support` (tall-stack path only): a contact that is NOT separating
/// (`vn_pre <= target`, i.e. resting or approaching) applies the FULL cached
/// impulse instead of the capped one. The cap neuters steady-state support —
/// gravity loads both bodies equally, so a resting pair's relative approach
/// is ~0 and the cap admits ~0 every substep, forcing the solver to rebuild
/// the whole chain load from scratch via Gauss-Seidel (O(depth) sweeps for a
/// stack). Overshoot from a stale cache is cheap to trim (one local sweep
/// clamps it down), while rebuild is not — so the asymmetric choice is full
/// apply on approach, capped apply on separation (pump protection stays).
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn apply_warm_start(
    bodies: &mut [RigidBody],
    m: &Manifold,
    i: usize,
    j: usize,
    n: Vec3,
    warm: &[f32; MAX_MANIFOLD_POINTS],
    target: &[f32; MAX_MANIFOLD_POINTS],
    full_support: bool,
) -> [f32; MAX_MANIFOLD_POINTS] {
    let mut warm_applied = *warm;
    for k in 0..m.point_count {
        if warm[k] > 0.0 {
            let p = m.points[k].world_point;
            let ra = p - bodies[i].position;
            let rb = p - bodies[j].position;
            let k_eff = effective_mass(bodies, i, j, n, ra, rb);
            if k_eff < MIN_EFFECTIVE_MASS {
                warm_applied[k] = 0.0;
                continue;
            }
            let vn_pre = (point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra)).dot(n);
            // Cap against the speculative target too: a separated point may
            // keep approaching up to its gap limit.
            let applied = if full_support && vn_pre <= target[k] {
                warm[k]
            } else {
                warm[k].min(((target[k] - vn_pre) / k_eff).max(0.0))
            };
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
/// global body-pair for warm-cache lookup. `full_support` selects the
/// tall-stack warm-start mode (see [`apply_warm_start`]); flat/GPU paths
/// pass false.
/// Anisotropic contact frame (ODE `fdir1`/`mu`/`mu2` parity): returns the
/// pair tangent basis plus the per-axis Coulomb coefficients.
///
/// Each body maps to an (along, transverse) coefficient pair — a body with
/// `friction_dir` contributes `(friction, friction_transverse)`, one
/// without contributes `(friction, friction)` — and each axis takes the
/// `max` across the pair (same combining as isotropic `mu`). The first
/// body (A) with a set direction wins `t1`: its local dir goes to world
/// and is projected onto the plane ⊥ `n`. Degenerate projections (dir ∥
/// `n`, zero-length) fall back to the default normal-derived basis, and
/// with no directions anywhere the basis is exactly `tangent_basis(n)`.
///
/// With all defaults (`friction_dir: None`, transverse mirrored from
/// `friction`) the returned `mu == mu2`, routing the solver through the
/// legacy circular-cone path bit-identically.
pub(super) fn anisotropic_frame(
    bodies: &[RigidBody],
    i: usize,
    j: usize,
    n: Vec3,
) -> (Vec3, f32, f32) {
    let pick_dir = |b: &RigidBody| -> Option<Vec3> {
        let frame = b.friction_frame().ok()?;
        let axis = match frame {
            crate::invariants::FrictionFrame::Isotropic => return None,
            crate::invariants::FrictionFrame::Aniso(u) => u.get(),
        };
        let world = b.orientation * axis;
        let proj = world - n * world.dot(n);
        crate::invariants::UnitVec3::normalize_checked(proj).map(|u| u.get())
    };
    // Body A wins (documented priority); B only if A sets nothing.
    let t1 = pick_dir(&bodies[i])
        .or_else(|| pick_dir(&bodies[j]))
        .unwrap_or_else(|| crate::math::tangent_basis(n).0);
    let axis = |b: &RigidBody| -> (f32, f32) {
        if b.friction_dir.is_some() {
            (b.friction, b.friction_transverse)
        } else {
            (b.friction, b.friction)
        }
    };
    let (a1, a2) = axis(&bodies[i]);
    let (b1, b2) = axis(&bodies[j]);
    (t1, a1.max(b1), a2.max(b2))
}

#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn prepare_manifold_state(
    bodies: &mut [RigidBody],
    m: &Manifold,
    key: (usize, usize),
    warm_in: &WarmCache,
    gate: RestitutionGate,
    sub_dt: f32,
    mi: usize,
    full_support: bool,
    hook: Option<&HookOverride>,
) -> Option<ManifoldState> {
    let (i, j) = (m.body_a.index(), m.body_b.index());
    let total_inv = bodies[i].inv_mass + bodies[j].inv_mass;
    if total_inv < MIN_EFFECTIVE_MASS {
        return None;
    }
    let n = m.normal;
    if !m.has_valid_count() {
        return None;
    }
    let count = m.point_count;

    // --- Body-frame anchors first: matching and G3 both need them ---
    let mut la = [Vec3::ZERO; MAX_MANIFOLD_POINTS];
    let mut lb = [Vec3::ZERO; MAX_MANIFOLD_POINTS];
    let mut pen0 = [0.0f32; MAX_MANIFOLD_POINTS];
    for k in 0..count {
        let p = m.points[k].world_point;
        la[k] = bodies[i].orientation.inverse() * (p - bodies[i].position);
        lb[k] = bodies[j].orientation.inverse() * (p - bodies[j].position);
        pen0[k] = m.points[k].penetration;
    }

    let warm = match_warm_points(&la, &lb, n, key, warm_in, count);
    // Any cached normal impulse on the pair, matched or not. A speculative
    // preview stores ~0 and must still bounce; a shuffled stack point must not.
    let pair_loaded = warm_in.get(&key).is_some_and(|(pts, n_cached)| {
        pts.iter()
            .take(*n_cached)
            .any(|p| p.impulse > LOADED_IMPULSE)
    });

    let e = hook
        .and_then(|h| h.restitution)
        .unwrap_or_else(|| bodies[i].restitution.min(bodies[j].restitution));
    let (t1, legacy_mu, legacy_mu2) = anisotropic_frame(bodies, i, j, n);
    // Hook friction is an isotropic override (both Coulomb axes take the
    // hook value, replacing body-level anisotropy for this pair); without
    // one the legacy pair frame stands exactly.
    let (mu, mu2) = hook
        .and_then(|h| h.friction)
        .map_or((legacy_mu, legacy_mu2), |f| (f, f));
    let surface_velocity = hook.and_then(|h| h.surface_velocity).unwrap_or(Vec3::ZERO);
    let mu_roll = bodies[i].rolling_friction.max(bodies[j].rolling_friction);
    let mu_spin = bodies[i].torsion_friction.max(bodies[j].torsion_friction);
    let target = speculative_targets(&pen0, count, sub_dt);
    let bias = compute_restitution_bias(bodies, m, pair_loaded, &pen0, n, e, gate, sub_dt);
    let warm_applied = apply_warm_start(bodies, m, i, j, n, &warm, &target, full_support);

    Some(ManifoldState {
        mi,
        i,
        j,
        count,
        acc: warm_applied,
        acc_friction: [0.0; MAX_MANIFOLD_POINTS],
        acc_friction2: [0.0; MAX_MANIFOLD_POINTS],
        bias,
        target,
        mu,
        mu2,
        mu_roll,
        mu_spin,
        acc_roll: [0.0; MAX_MANIFOLD_POINTS],
        acc_roll2: [0.0; MAX_MANIFOLD_POINTS],
        acc_spin: [0.0; MAX_MANIFOLD_POINTS],
        t1,
        t2: t1.cross(n),
        surface_velocity,
        la,
        lb,
        pen0,
    })
}

impl SequentialImpulseEngine {
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
            ctx.gate,
            ctx.sub_dt,
            ctx.mi,
            false,
            None,
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
        gate: RestitutionGate,
        sub_dt: f32,
        dt: f32,
    ) -> Vec<IslandWork> {
        // Build global states for ALL active manifolds (shared preamble).
        let mut global_states: Vec<ManifoldState> = Vec::with_capacity(active.len());
        for &mi in &active {
            let m = &manifolds[mi];
            let (i, j) = (m.body_a.index(), m.body_b.index());
            let key = (i.min(j), i.max(j));
            let mut ctx = ManifoldCtx {
                bodies: &mut self.bodies,
                warm_in: &self.warm_impulses,
                gate,
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
        let mut gpu_warm: WarmCache = FxHashMap::default();
        if !single_si.is_empty()
            && let Some(gpu) = self.gpu_solver.as_mut()
        {
            let (batches, num_batches) =
                pack_single_point_batches(&self.bodies, &global_states, manifolds, &single_si);
            if num_batches > 0 {
                gpu.upload_bodies(&self.bodies);
                gpu.upload_batches(&batches);
                gpu.solve(num_batches, self.velocity_iterations, gate);
                gpu.download_bodies(&mut self.bodies);
                let mut dl_batches = batches;
                gpu.download_acc(&mut dl_batches);
                write_back_acc(&mut global_states, &single_si, &dl_batches);
                // Persist warm cache for single-point manifolds.
                for &si in &single_si {
                    let st = &global_states[si];
                    let m = &manifolds[st.mi];
                    let key = (
                        m.body_a.index().min(m.body_b.index()),
                        m.body_a.index().max(m.body_b.index()),
                    );
                    let mut pts = [WarmPoint {
                        la: Vec3::ZERO,
                        lb: Vec3::ZERO,
                        normal: Vec3::ZERO,
                        impulse: 0.0,
                    }; MAX_MANIFOLD_POINTS];
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
            let mut islands = self.partition_into_islands(&multi_mi, manifolds, &[]);
            self.dispatch_islands_velocity(&mut islands, gate, sub_dt, dt);
            islands
        };
        self.warm_impulses.extend(gpu_warm);
        islands
    }

    /// Velocity stage of the contact solver (G6 stage order, G7 island
    /// dispatch). Orchestrator: a sequential sleep/wake pre-pass (the only
    /// part mutating island state), a union-find partition over the FRESH
    /// manifolds, then each island solved independently by
    /// `solve_island_velocity` — via the scheduler when the scene is wide
    /// enough. Islands are disjoint over dynamic bodies by construction, so
    /// concurrent solves are race-free and bit-identical for any thread
    /// count (Strong Confluence). Returns the island work items; the position
    /// stage reuses them (states + remapped manifolds) after integration.
    pub(super) fn solve_contacts_velocity(
        &mut self,
        manifolds: &mut [Manifold],
        gate: RestitutionGate,
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
        // Contact hooks stay on the CPU island path, which alone implements
        // overrides (see the `hooks` module docs).
        #[cfg(feature = "gpu")]
        if self.gpu_solver.is_some() && !self.has_contact_hooks() {
            return self.solve_contacts_velocity_gpu(active, manifolds, gate, sub_dt, dt);
        }

        // --- H2 contact-hooks pre-solve (sequential, manifold order) ---
        // Runs after the sleep/wake pre-pass, before island partitioning,
        // so the call order is deterministic and the overrides travel with
        // the manifolds into the islands.
        let overrides = self.apply_hook_modify(manifolds, &active, sub_dt);

        // --- H1 solver flags + H2 reduction (hooks attached only) ---
        // `READ_ONLY` pairs keep their manifolds and the events recorded
        // above, but never enter the islands (zero impulses); manifolds a
        // hook fully cleared (`point_count == 0`) carry nothing to solve.
        // Without hooks every manifold is valid and the list passes
        // through untouched.
        let active: Vec<usize> = if self.contact_hooks.is_some() {
            let read_only = &self.hook_read_only;
            active
                .into_iter()
                .filter(|&mi| {
                    let m = &manifolds[mi];
                    if !m.has_valid_count() {
                        return false;
                    }
                    let (a, b) = (m.body_a.index(), m.body_b.index());
                    !read_only.contains(&(a.min(b), a.max(b)))
                })
                .collect()
        } else {
            active
        };
        if active.is_empty() {
            self.warm_impulses.clear();
            return Vec::new();
        }

        // --- Partition into islands + dispatch (G7) ---
        // Islands are disjoint over dynamic bodies by construction, so
        // concurrent solves are race-free and bit-identical for any thread
        // count (Strong Confluence).
        let mut islands = self.partition_into_islands(&active, manifolds, &overrides);
        self.dispatch_islands_velocity(&mut islands, gate, sub_dt, dt);
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
        let mode = Dispatch::from(
            islands.len() >= PAR_MIN_ISLANDS && total_manifolds >= PAR_MIN_MANIFOLDS,
        );
        Self::dispatch_islands(islands, mode, |idx, isl| {
            let iters = iters_per_island[idx];
            Self::solve_island_position(
                &mut isl.bodies,
                &isl.manifolds,
                &isl.states,
                iters,
                softness,
            );
        });
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
        let (i, j) = (m.body_a.index(), m.body_b.index());
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
    ///
    /// Sleep rule in one place: the `asleep` flag alone decides. Statics are
    /// asleep from birth (never wake anything); kinematics are never asleep
    /// (driven bodies wake sleepers on contact instead of ghosting through).
    /// Pairs with no dynamic member are the driver's business, never the
    /// solver's — skipped exactly as before.
    fn collect_active_manifolds(&mut self, manifolds: &[Manifold]) -> Vec<usize> {
        // Sleep: a contact needs work only if at least one side is an AWAKE
        // body. Static geometry never wakes anything (a body asleep
        // on the floor must stay asleep).
        const WAKE_IMPACT_SPEED: f32 = 0.5;
        let mut active: Vec<usize> = Vec::with_capacity(manifolds.len());
        for (mi, m) in manifolds.iter().enumerate() {
            // Reduction point: a hook may have cleared every point of this
            // manifold (narrowphase output is always valid, so without
            // hooks this never fires).
            if !m.has_valid_count() {
                continue;
            }
            let (i, j) = (m.body_a.index(), m.body_b.index());
            let ai = self.asleep[i];
            let aj = self.asleep[j];
            if ai && aj {
                continue;
            }
            if self.bodies[i].body_type != BodyType::Dynamic
                && self.bodies[j].body_type != BodyType::Dynamic
            {
                continue;
            }
            // Wake hysteresis (G7): a sleeping island is woken only by a
            // genuine IMPACT — approach speed above the threshold. A resting
            // micro-jitter contact (vn ≈ 0) must NOT wake it.
            if (self.asleep[i] && !aj) || (self.asleep[j] && !ai) {
                self.wake_on_impact(m, WAKE_IMPACT_SPEED);
            }
            // Penetration wake: teleporting drivers (and spawns) move bodies
            // with a zero velocity field, so the approach test above is
            // blind to them — but deepening overlap is unambiguous motion.
            // Resting contacts sit orders of magnitude below this (NGS
            // residuals), so sleep stays stable.
            const WAKE_PENETRATION: f32 = 0.01;
            if self.asleep[i] || self.asleep[j] {
                let mut deep = false;
                for k in 0..m.point_count {
                    if m.points[k].penetration > WAKE_PENETRATION {
                        deep = true;
                        break;
                    }
                }
                if deep {
                    if self.asleep[i] {
                        self.wake_island(i);
                    }
                    if self.asleep[j] {
                        self.wake_island(j);
                    }
                }
            }
            // A still-sleeping body is static for the solver (its inv_mass is
            // zeroed at sleep), so sleeper+static pairs carry no work.
            if self.bodies[i].inv_mass + self.bodies[j].inv_mass < MIN_EFFECTIVE_MASS {
                continue;
            }
            // Hit events (Gameplay, Box3D parity): the hardest-approaching
            // point of a solved contact faster than the threshold. Recorded
            // pre-solve on every substep (each substep's approach is physical
            // pre-solve data); one hit per pair per step — the scratch set
            // cleared at s==0 dedupes sustained crushes.
            {
                let key = (i.min(j), i.max(j));
                if !self.scratch_hit_pairs.contains(&key) {
                    let mut best = 0.0f32;
                    let mut best_point = Vec3::ZERO;
                    for k in 0..m.point_count {
                        let p = m.points[k].world_point;
                        let ra = p - self.bodies[i].position;
                        let rb = p - self.bodies[j].position;
                        let vrel = point_velocity(&self.bodies[j], rb)
                            - point_velocity(&self.bodies[i], ra);
                        let closing = -vrel.dot(m.normal);
                        if closing > best {
                            best = closing;
                            best_point = p;
                        }
                    }
                    if best > crate::trigger::CONTACT_HIT_THRESHOLD {
                        self.scratch_hit_pairs.insert(key);
                        self.contact_events.push(crate::trigger::ContactEvent {
                            body_a: crate::body::BodyHandle::from(i),
                            body_b: crate::body::BodyHandle::from(j),
                            kind: crate::trigger::ContactEventKind::Hit {
                                point: best_point,
                                normal: m.normal,
                                approach_speed: best,
                            },
                        });
                    }
                }
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
    /// `hook` carries the validated contact-hooks overrides aligned with the
    /// local manifolds (missing entries read as `None` = legacy preamble).
    /// When `path` is [`SolvePath::Wide`], single-point manifolds are solved
    /// in SIMD-wide batches (G7); multi-point (block LCP) stays scalar.
    // The 9th parameter (`path`, G7) tips this over clippy's default
    // 7-argument limit; packing them into a struct would only add churn.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve_island_velocity(
        bodies: &mut [RigidBody],
        manifolds: &[Manifold],
        keys: &[(usize, usize)],
        warm_in: &WarmCache,
        velocity_iterations: u32,
        gate: RestitutionGate,
        sub_dt: f32,
        path: SolvePath,
        hook: &[Option<HookOverride>],
    ) -> (Vec<ManifoldState>, WarmCache) {
        let mut states =
            prepare_island_states(bodies, manifolds, keys, warm_in, gate, sub_dt, hook);
        // --- Velocity solve: Gauss-Seidel iterations over ALL manifolds ---
        // G7: single-point manifolds are packed into SIMD-wide batches
        // (disjoint body sets, original GS order preserved — every contact
        // stays in its place in the sequence); multi-point manifolds keep
        // the scalar block-LCP path. Steps run in manifold order, so the
        // computation is the same sequence either way.
        let mut steps = if path.use_wide() {
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
            path,
        );
        // Wide batches own the accumulated impulses of their lanes during
        // the iterations; write them back so the cache persist below sees
        // the final values.
        if path.use_wide() {
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
        if gate.is_enabled() {
            run_restitution_stage(bodies, manifolds, &states, &mut steps, path);
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
            let mut pts = [Vec3::ZERO; MAX_MANIFOLD_POINTS];
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
            if k_eff >= MIN_EFFECTIVE_MASS {
                let rel = point_velocity(&bodies[j], rb) - point_velocity(&bodies[i], ra);
                let vn = rel.dot(n);
                // Inelastic contact: restitution is a separate one-shot stage
                // (below), never accumulated. G6: the target is the
                // speculative approach limit (0 when touching), not
                // necessarily a full stop. Shared isotropic row
                // (contact_math): the reciprocal folds the division into
                // the kernel's multiply form.
                let new_acc = contact_normal_step::eval(vn, st.target[k], 1.0 / k_eff, st.acc[k]);
                let delta = new_acc - st.acc[k];
                st.acc[k] = new_acc;
                if delta.abs() > DEGENERATE_LEN2 {
                    apply_impulse(bodies, i, j, n * delta, ra, rb);
                }
            }
        }

        // Friction (Coulomb) along the FIXED tangent basis (extracted helper).
        Self::solve_scalar_friction(bodies, i, j, st, m, total_inv);
    }

    /// One-shot restitution step for a single manifold. The wide velocity
    /// iterations do not apply this impulse: both paths call this function
    /// so a no-op hook and the default solver share one bounce.
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
            if k_eff < MIN_EFFECTIVE_MASS {
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
    /// cognitive complexity of solve_scalar_velocity_step). Slide friction
    /// is isotropic (legacy circular cone, verbatim) while `mu == mu2`;
    /// anisotropic pairs take the elliptical-cone projection instead.
    /// Rolling/torsional resistance follows in the same pass.
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
            // Conveyor hook: friction drives the slip toward the belt
            // velocity instead of zero. The zero branch keeps the legacy
            // rows bit-exact (no subtract is issued at all without hooks).
            let rel = if st.surface_velocity == Vec3::ZERO {
                rel
            } else {
                rel - st.surface_velocity
            };
            if st.mu == st.mu2 {
                // Legacy circular-cone path (isotropic): sequential
                // per-axis clamp, bit-identical to the pre-anisotropy code.
                let max_friction = st.mu * st.acc[k];
                let mut f_imp = Vec3::ZERO;
                for axis in 0..2 {
                    let t = if axis == 0 { st.t1 } else { st.t2 };
                    let k_t = total_inv + tangent_effective_mass(bodies, i, j, ra, rb, t);
                    if k_t < MIN_EFFECTIVE_MASS {
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
                if f_imp.length_squared() > FRICTION_IMPULSE_DUST {
                    apply_impulse(bodies, i, j, f_imp, ra, rb);
                }
            } else {
                // Elliptical cone: joint projection of both axes in the
                // normalized constraint space (T1/mu1, T2/mu2), capped by
                // the normal impulse. Deliberate deviation from the ODE
                // friction pyramid (independent per-axis box clamps): the
                // ellipse is the exact Coulomb generalization.
                let k_t1 = total_inv + tangent_effective_mass(bodies, i, j, ra, rb, st.t1);
                let k_t2 = total_inv + tangent_effective_mass(bodies, i, j, ra, rb, st.t2);
                let lam1 = if k_t1 >= MIN_EFFECTIVE_MASS {
                    -rel.dot(st.t1) / k_t1
                } else {
                    0.0
                };
                let lam2 = if k_t2 >= MIN_EFFECTIVE_MASS {
                    -rel.dot(st.t2) / k_t2
                } else {
                    0.0
                };
                let mut u1 = st.acc_friction[k] + lam1;
                let mut u2 = st.acc_friction2[k] + lam2;
                // A zero coefficient carries nothing along its axis.
                if st.mu <= 0.0 {
                    u1 = 0.0;
                }
                if st.mu2 <= 0.0 {
                    u2 = 0.0;
                }
                let r1 = if st.mu > 0.0 { u1 / st.mu } else { 0.0 };
                let r2 = if st.mu2 > 0.0 { u2 / st.mu2 } else { 0.0 };
                let r_len = r1.hypot(r2);
                let cap = st.acc[k];
                if r_len > cap && r_len > DEGENERATE_LEN2 {
                    let s = cap / r_len;
                    u1 *= s;
                    u2 *= s;
                }
                debug_assert!(u1.is_finite() && u2.is_finite(), "aniso impulse overflowed");
                let f_imp = st.t1 * (u1 - st.acc_friction[k]) + st.t2 * (u2 - st.acc_friction2[k]);
                st.acc_friction[k] = u1;
                st.acc_friction2[k] = u2;
                if f_imp.length_squared() > FRICTION_IMPULSE_DUST {
                    apply_impulse(bodies, i, j, f_imp, ra, rb);
                }
            }
            // Rolling + torsional resistance (MuJoCo slide/torsion/rolling
            // triple parity): pure couples opposing relative spin, capped
            // by mu × normal impulse. Zero coefficients skip everything.
            if st.mu_roll > 0.0 || st.mu_spin > 0.0 {
                // Each axis re-reads ω. An impulse about t1 changes ω·t2
                // when inertia is not aligned with the contact frame, so a
                // wrel captured once couples the two tangents and the normal.
                Self::solve_scalar_rolling(bodies, i, j, st, k, st.t1, st.mu_roll, RollAxis::RollU);
                Self::solve_scalar_rolling(bodies, i, j, st, k, st.t2, st.mu_roll, RollAxis::RollV);
                Self::solve_scalar_rolling(
                    bodies,
                    i,
                    j,
                    st,
                    k,
                    m.normal,
                    st.mu_spin,
                    RollAxis::Spin,
                );
            }
        }
    }

    /// One rolling/torsional axis for a single contact point: opposes the
    /// relative spin about `axis` with a pure couple (no linear part —
    /// MuJoCo contact-frame torque model), accumulated per point and
    /// capped by `mu_axis × normal impulse`. `axis_kind` selects the
    /// accumulator: [`RollAxis::RollU`] / [`RollAxis::RollV`] / [`RollAxis::Spin`].
    /// Relative spin is read from the live angular velocities so a previous
    /// axis in the same sweep is visible.
    // Eight parameters mirror the neighboring friction helpers (bodies,
    // pair, point, axis, cap, slot); packing them would hide the
    // call-site symmetry of the three axes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve_scalar_rolling(
        bodies: &mut [RigidBody],
        i: usize,
        j: usize,
        st: &mut ManifoldState,
        k: usize,
        axis: Vec3,
        mu_axis: f32,
        axis_kind: RollAxis,
    ) {
        if mu_axis <= 0.0 {
            return;
        }
        let wrel = bodies[j].angular_velocity - bodies[i].angular_velocity;
        let k_rot = axis.dot(mul_inv_inertia(
            bodies[i].inertia,
            bodies[i].orientation,
            axis,
        )) + axis.dot(mul_inv_inertia(
            bodies[j].inertia,
            bodies[j].orientation,
            axis,
        ));
        if k_rot < MIN_EFFECTIVE_MASS {
            return;
        }
        let cap = mu_axis * st.acc[k];
        let cur = match axis_kind {
            RollAxis::RollU => st.acc_roll[k],
            RollAxis::RollV => st.acc_roll2[k],
            RollAxis::Spin => st.acc_spin[k],
        };
        let delta = -wrel.dot(axis) / k_rot;
        let new = (cur + delta).clamp(-cap, cap);
        debug_assert!(new.is_finite(), "rolling impulse overflowed");
        if new != cur {
            apply_angular_impulse(bodies, i, j, axis * (new - cur));
            match axis_kind {
                RollAxis::RollU => st.acc_roll[k] = new,
                RollAxis::RollV => st.acc_roll2[k] = new,
                RollAxis::Spin => st.acc_spin[k] = new,
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
                    if k_soft < MIN_EFFECTIVE_MASS {
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
/// `hook` carries the validated contact-hooks overrides aligned with the
/// island-local manifolds (missing entries read as `None` = legacy).
fn prepare_island_states(
    bodies: &mut [RigidBody],
    manifolds: &[Manifold],
    keys: &[(usize, usize)],
    warm_in: &WarmCache,
    gate: RestitutionGate,
    sub_dt: f32,
    hook: &[Option<HookOverride>],
) -> Vec<ManifoldState> {
    // G2b: warm-start cache matches points by proximity, not by index —
    // manifold point order changes frame to frame (sorted by depth).
    // Tall-stack islands apply full resting support (see `apply_warm_start`).
    let full_support = stack_path_for_island(manifolds.len());
    let mut states: Vec<ManifoldState> = Vec::with_capacity(manifolds.len());
    for (mi, m) in manifolds.iter().enumerate() {
        let key = keys[mi];
        let hook_m = hook.get(mi).and_then(|o| o.as_ref());
        if let Some(st) = prepare_manifold_state(
            bodies,
            m,
            key,
            warm_in,
            gate,
            sub_dt,
            mi,
            full_support,
            hook_m,
        ) {
            states.push(st);
        }
    }
    states
}

/// Gauss-Seidel iterations over ALL manifolds: wide batches interleaved with
/// scalar manifolds when `path` is wide, plain scalar sequence otherwise. Both
/// orders visit the contacts in manifold order, so results are identical.
fn run_velocity_iterations(
    bodies: &mut [RigidBody],
    manifolds: &[Manifold],
    states: &mut [ManifoldState],
    steps: &mut [SolverStep],
    velocity_iterations: u32,
    path: SolvePath,
) {
    for _ in 0..velocity_iterations {
        if path.use_wide() {
            for step in steps.iter_mut() {
                match step {
                    SolverStep::Wide(b) => {
                        b.gather(bodies);
                        b.solve_iteration();
                        b.scatter(bodies);
                    }
                    SolverStep::Scalar(si) => {
                        SequentialImpulseEngine::solve_scalar_velocity_step(
                            bodies,
                            manifolds,
                            &mut states[*si],
                        );
                    }
                }
            }
        } else {
            for st in states.iter_mut() {
                SequentialImpulseEngine::solve_scalar_velocity_step(bodies, manifolds, st);
            }
        }
    }
}

/// One-shot restitution stage. Always this scalar impulse, including after
/// wide velocity iterations: the lane form multiplied by a precomputed
/// inverse mass and did not match the division here.
fn run_restitution_stage(
    bodies: &mut [RigidBody],
    manifolds: &[Manifold],
    states: &[ManifoldState],
    _steps: &mut [SolverStep],
    _path: SolvePath,
) {
    for st in states {
        SequentialImpulseEngine::solve_scalar_restitution_step(bodies, manifolds, st);
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
    let mut next: WarmCache = FxHashMap::default();
    for st in states {
        let m = &manifolds[st.mi];
        let mut pts = [WarmPoint {
            la: Vec3::ZERO,
            lb: Vec3::ZERO,
            normal: Vec3::ZERO,
            impulse: 0.0,
        }; MAX_MANIFOLD_POINTS];
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
/// impulse on the perpendicular axis. Thin alias over the shared isotropic
/// row ([`crate::contact_math`]); kept so solver call sites stay unchanged.
fn clamp_friction_impulse(new_t: f32, other: f32, max_friction: f32) -> f32 {
    contact_friction_clamp::eval(new_t, other, max_friction)
}

impl SequentialImpulseEngine {
    /// Flat singleton velocity solve (parallel coarse shards): the same
    /// per-manifold kernels as the island path
    /// ([`prepare_manifold_state`], wide/scalar iterations, one-shot
    /// restitution, warm persist) over coarse solve shards instead of one
    /// island-work per manifold — no union-find, no per-manifold
    /// Vecs/sorts/clones, no 100k-node dispatch, no tiny-map merge churn.
    ///
    /// Bit-identical to the island path when
    /// [`SequentialImpulseEngine::flat_singleton_eligible`] holds:
    /// singleton manifolds are disjoint over dynamic bodies, statics ride
    /// each shard as read-only clones (discarded on scatter, like the
    /// island path), and the warm cache is keyed per disjoint pair — so
    /// cross-manifold order never affects the result (Strong Confluence).
    /// `shards_out` persists across the substep for the position stage.
    pub(super) fn solve_flat_velocity(
        &mut self,
        manifolds: &[Manifold],
        gate: RestitutionGate,
        sub_dt: f32,
        dt: f32,
        shards_out: &mut Vec<IslandWork>,
    ) {
        shards_out.clear();
        let active = self.collect_active_manifolds(manifolds);
        if active.is_empty() {
            self.warm_impulses.clear();
            return;
        }
        shards_out.extend(self.build_flat_shards(&active, manifolds));
        let base_iters = self.velocity_iterations;
        let path = self.wide_solver;
        let total_manifolds: usize = shards_out.iter().map(|s| s.manifolds.len()).sum();
        let mode =
            Dispatch::from(shards_out.len() >= 2 && total_manifolds >= PARALLEL_MIN_MANIFOLDS);
        let this = &*self;
        let warm_in = &this.warm_impulses;
        Self::dispatch_islands(shards_out, mode, |_, shard| {
            this.solve_one_flat_shard(shard, warm_in, base_iters, gate, sub_dt, dt, path);
        });
        // Scatter dynamics back + merge warm caches (reserved upfront; the
        // island path regrows its map by repeated per-island extend).
        let mut next: WarmCache = FxHashMap::default();
        next.reserve(active.len());
        for shard in shards_out.iter() {
            for (l, &g) in shard.body_idx.iter().enumerate() {
                if self.bodies[g].body_type == BodyType::Dynamic {
                    self.bodies[g] = shard.bodies[l].clone();
                }
            }
            next.extend(shard.warm.iter().map(|(k, v)| (*k, *v)));
        }
        self.warm_impulses = next;
    }

    /// One flat shard's velocity solve: per-manifold adaptive iteration
    /// counts (the same penetration-aware heuristic as the island
    /// dispatch), stable-bucketed to amortize the step-sequence build.
    /// States persist on the shard for the position stage.
    #[allow(clippy::needless_range_loop)]
    #[allow(clippy::too_many_arguments)]
    fn solve_one_flat_shard(
        &self,
        shard: &mut IslandWork,
        warm_in: &WarmCache,
        base_iters: u32,
        gate: RestitutionGate,
        sub_dt: f32,
        dt: f32,
        path: SolvePath,
    ) {
        shard.states.clear();
        shard.warm.clear();
        shard.warm.reserve(shard.manifolds.len());
        let mut buckets: Vec<Vec<usize>> = Vec::new();
        buckets.resize_with(base_iters as usize + 1, Vec::new);
        for (li, m) in shard.manifolds.iter().enumerate() {
            let (a, b) = (m.body_a.index(), m.body_b.index());
            let max_speed = [a, b]
                .into_iter()
                .filter(|&h| shard.bodies[h].body_type == BodyType::Dynamic)
                .map(|h| {
                    shard.bodies[h]
                        .velocity
                        .length()
                        .max(shard.bodies[h].angular_velocity.length())
                })
                .fold(0.0f32, f32::max);
            let max_pen = m.points[..m.point_count]
                .iter()
                .map(|p| p.penetration)
                .fold(0.0f32, f32::max);
            let iters = self.adaptive_iters_for_island_with_pen(max_speed, max_pen, dt, base_iters);
            buckets[iters as usize].push(li);
        }
        for (iters, bucket) in buckets.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let base = shard.states.len();
            for &li in bucket {
                let key = shard.keys[li];
                // Split the shard borrow: the manifold/key are read-only,
                // the bodies are solved in place (disjoint fields).
                let (bodies, manifolds) = (&mut shard.bodies, &shard.manifolds);
                if let Some(st) = prepare_manifold_state(
                    bodies,
                    &manifolds[li],
                    key,
                    warm_in,
                    gate,
                    sub_dt,
                    li,
                    false,
                    None,
                ) {
                    shard.states.push(st);
                }
            }
            if shard.states.len() == base {
                continue;
            }
            // Split again: states grow while manifolds stay read-only.
            let (states, manifolds) = (&mut shard.states, &shard.manifolds);
            let states = &mut states[base..];
            let mut steps = build_solver_steps(&shard.bodies, manifolds, states);
            run_velocity_iterations(
                &mut shard.bodies,
                manifolds,
                states,
                &mut steps,
                iters as u32,
                path,
            );
            if path.use_wide() {
                for step in &steps {
                    if let SolverStep::Wide(b) = step {
                        b.write_back_acc(states);
                    }
                }
            }
            if gate.is_enabled() {
                run_restitution_stage(&mut shard.bodies, manifolds, states, &mut steps, path);
            }
            for st in states.iter() {
                let m = &manifolds[st.mi];
                let mut pts = [WarmPoint {
                    la: Vec3::ZERO,
                    lb: Vec3::ZERO,
                    normal: Vec3::ZERO,
                    impulse: 0.0,
                }; MAX_MANIFOLD_POINTS];
                for k in 0..st.count {
                    pts[k] = WarmPoint {
                        la: st.la[k],
                        lb: st.lb[k],
                        normal: m.normal,
                        impulse: st.acc[k],
                    };
                }
                shard.warm.insert(shard.keys[st.mi], (pts, st.count));
            }
        }
    }

    /// Flat singleton position (NGS) solve: re-gathers the shards from the
    /// integrated bodies (like [`SequentialImpulseEngine::solve_contacts_position`]),
    /// then per-manifold iteration counts from the same heuristic,
    /// dispatched in parallel and scattered back. Same disjointness
    /// argument as [`SequentialImpulseEngine::solve_flat_velocity`].
    pub(super) fn solve_flat_position(&mut self, shards: &mut [IslandWork], dt: f32) {
        if shards.is_empty() {
            return;
        }
        // Re-gather: integration moved the main array since the velocity
        // stage ran (same discipline as the island position stage).
        for shard in shards.iter_mut() {
            for (l, &g) in shard.body_idx.iter().enumerate() {
                shard.bodies[l] = self.bodies[g].clone();
            }
        }
        let base_iters = self.position_iterations;
        let softness = self.contact_softness;
        let total_manifolds: usize = shards.iter().map(|s| s.manifolds.len()).sum();
        let mode = Dispatch::from(shards.len() >= 2 && total_manifolds >= PARALLEL_MIN_MANIFOLDS);
        let this = &*self;
        Self::dispatch_islands(shards, mode, |_, shard| {
            for st in shard.states.iter() {
                let m = &shard.manifolds[st.mi];
                let max_speed = [st.i, st.j]
                    .into_iter()
                    .filter(|&h| shard.bodies[h].body_type == BodyType::Dynamic)
                    .map(|h| {
                        shard.bodies[h]
                            .velocity
                            .length()
                            .max(shard.bodies[h].angular_velocity.length())
                    })
                    .fold(0.0f32, f32::max);
                let max_pen = m.points[..m.point_count]
                    .iter()
                    .map(|p| p.penetration)
                    .fold(0.0f32, f32::max);
                let iters =
                    this.adaptive_iters_for_island_with_pen(max_speed, max_pen, dt, base_iters);
                Self::solve_island_position(
                    &mut shard.bodies,
                    &shard.manifolds,
                    std::slice::from_ref(st),
                    iters,
                    softness,
                );
            }
        });
        for shard in shards.iter() {
            for (l, &g) in shard.body_idx.iter().enumerate() {
                if self.bodies[g].body_type == BodyType::Dynamic {
                    self.bodies[g] = shard.bodies[l].clone();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One Gauss-Seidel sweep, inertia rotated 45° about Y so an impulse
    /// about the contact tangent X also changes ω·Z. Capturing `wrel` once
    /// leaves |ω_z| ≈ 0.98; re-reading it after the first axis cancels Z
    /// and puts the residual back on X.
    #[test]
    fn rolling_axes_reread_spin_under_anisotropic_inertia() {
        let floor = RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 0.0);
        let mut body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0);
        body.inertia = Vec3::new(1.0, 1.0, 0.01);
        body.orientation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        body.angular_velocity = Vec3::X;
        let mut bodies = vec![floor, body];
        let mut points = [ManifoldPoint {
            world_point: Vec3::ZERO,
            penetration: 0.0,
        }; MAX_MANIFOLD_POINTS];
        points[0].penetration = 0.01;
        let manifold = Manifold::from_parts(
            BodyHandle::from(0usize),
            BodyHandle::from(1usize),
            Vec3::Y,
            points,
            1,
        )
        .expect("one-point manifold");
        let mut st = ManifoldState {
            mi: 0,
            i: 0,
            j: 1,
            count: 1,
            acc: [1.0e3, 0.0, 0.0, 0.0],
            acc_friction: [0.0; MAX_MANIFOLD_POINTS],
            acc_friction2: [0.0; MAX_MANIFOLD_POINTS],
            bias: [0.0; MAX_MANIFOLD_POINTS],
            target: [0.0; MAX_MANIFOLD_POINTS],
            mu: 0.0,
            mu2: 0.0,
            mu_roll: 1.0,
            mu_spin: 1.0,
            acc_roll: [0.0; MAX_MANIFOLD_POINTS],
            acc_roll2: [0.0; MAX_MANIFOLD_POINTS],
            acc_spin: [0.0; MAX_MANIFOLD_POINTS],
            t1: Vec3::X,
            t2: Vec3::Z,
            surface_velocity: Vec3::ZERO,
            la: [Vec3::ZERO; MAX_MANIFOLD_POINTS],
            lb: [Vec3::ZERO; MAX_MANIFOLD_POINTS],
            pen0: [0.01, 0.0, 0.0, 0.0],
        };
        SequentialImpulseEngine::solve_scalar_velocity_step(&mut bodies, &[manifold], &mut st);
        let w = bodies[1].angular_velocity;
        assert!(
            w.z.abs() < 0.15,
            "Z stayed coupled to the X couple, ω={w:?}"
        );
        assert!(
            w.x.abs() > 0.5,
            "second axis must put the residual back on X, ω={w:?}"
        );
    }
}
