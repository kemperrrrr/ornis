//! Broadphase backends for the builtin physics engine.
//!
//! The module keeps the candidate-pair contract separate from any one
//! spatial data structure. Sweep-and-Prune remains the compatibility baseline;
//! the opt-in uniform grid targets scenes where one large static AABB would
//! otherwise make the baseline quadratic.

use std::collections::{HashMap, HashSet};

use glam::Vec3;

use crate::body::{BodyType, RigidBody};
use crate::broadphase_tree::DynamicAabbTree;
use crate::math::AABB;

pub(crate) const HALF_SPEC_MARGIN: f32 = 0.025;
const DEFAULT_GRID_CELL_SIZE: f32 = 2.0;
const DEFAULT_MAX_CELLS_PER_BODY: usize = 4096;

/// Summary of the last broadphase update.
///
/// The counters describe candidate generation before narrowphase/solver work;
/// they are intended for benchmark diagnostics rather than gameplay logic.
///
/// `pair_tests`, `filter_rejections`, `static_static_skips` and
/// `aabb_rejections` are honest totals comparable to a full rebuild and to
/// the `SweepAndPrune` baseline: after an incremental [`UniformGrid`] update
/// they equal `retained clean-clean` + `new dirty-involved` counts, while
/// `candidate_pairs` is always derived from the final de-duplicated active
/// set. If the backend ever emitted incremental-only counters, it would be
/// explicitly documented; the current implementation preserves totals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BroadPhaseStats {
    /// Number of bodies included in the update.
    pub body_count: usize,
    /// Number of raw pair checks before filtering and AABB rejection (honest total).
    pub pair_tests: usize,
    /// Pair checks rejected by mutual collision layers/masks (honest total).
    pub filter_rejections: usize,
    /// Raw pair checks skipped because both bodies are ordinary static bodies (honest total).
    pub static_static_skips: usize,
    /// Pair checks whose swept AABBs do not overlap (honest total).
    pub aabb_rejections: usize,
    /// Unique candidate pairs emitted for narrowphase processing.
    pub candidate_pairs: usize,
    /// Number of occupied cells for the grid backend.
    pub occupied_cells: usize,
    /// Number of bodies routed through the grid's large-body escape path.
    pub large_bodies: usize,
}

/// Wall-clock breakdown of a single [`crate::BuiltinPhysicsEngine::step`].
///
/// Diagnostics for benchmarks only; not part of the simulation contract.
/// Per-substep phases are summed across the substep loop, so totals reflect
/// the whole step rather than the last substep.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepTiming {
    /// Time spent rebuilding swept AABBs and candidate pairs.
    pub broad_phase_ms: f64,
    /// Time spent in narrowphase manifold generation over the pair list.
    pub narrow_phase_ms: f64,
    /// Time spent in velocity/position solving, joints and continuous pass.
    pub solver_ms: f64,
    /// Number of substeps the timings were summed over.
    pub substeps: u32,
}

impl StepTiming {
    /// Per-substep average for a phase (0.0 when no substeps ran).
    pub fn per_substep_ms(&self) -> f64 {
        if self.substeps == 0 {
            0.0
        } else {
            (self.broad_phase_ms + self.narrow_phase_ms + self.solver_ms) / self.substeps as f64
        }
    }
}

/// Available candidate-pair backends for [`crate::BuiltinPhysicsEngine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadPhaseKind {
    /// Axis sweep baseline used by default for compatibility.
    SweepAndPrune,
    /// Uniform spatial grid with a large-body escape path.
    UniformGrid,
    /// Experimental persistent dynamic AABB tree.
    DynamicAabbTree,
}

/// Internal broadphase contract: update world AABBs and expose deterministic
/// candidate pairs for the narrowphase. Backends do not solve contacts.
pub(crate) trait BroadPhase {
    /// Rebuild candidate pairs for the current body poses and substep motion.
    fn update(&mut self, bodies: &[RigidBody], sub_dt: f32);
    /// Candidate body pairs, canonicalized as `(lower_handle, higher_handle)`.
    fn active(&self) -> &[(usize, usize)];
    /// Diagnostics for the latest update.
    fn stats(&self) -> BroadPhaseStats;
}

/// Runtime-selected broadphase backend.
pub(crate) enum BroadPhaseBackend {
    /// Compatibility Sweep-and-Prune implementation.
    SweepAndPrune(SweepAndPrune),
    /// Experimental uniform grid implementation.
    UniformGrid(UniformGrid),
    /// Experimental persistent dynamic AABB tree implementation.
    DynamicAabbTree(DynamicAabbTree),
}

impl BroadPhaseBackend {
    /// Creates the selected backend with its default tuning.
    pub(crate) fn new(kind: BroadPhaseKind) -> Self {
        match kind {
            BroadPhaseKind::SweepAndPrune => Self::SweepAndPrune(SweepAndPrune::new()),
            BroadPhaseKind::UniformGrid => Self::UniformGrid(UniformGrid::new()),
            BroadPhaseKind::DynamicAabbTree => Self::DynamicAabbTree(DynamicAabbTree::new()),
        }
    }

    /// Creates a uniform grid with an explicit cell size.
    pub(crate) fn uniform_grid(cell_size: f32) -> Self {
        Self::UniformGrid(UniformGrid::with_cell_size(cell_size))
    }

    /// Returns the selected backend kind.
    pub(crate) fn kind(&self) -> BroadPhaseKind {
        match self {
            Self::SweepAndPrune(_) => BroadPhaseKind::SweepAndPrune,
            Self::UniformGrid(_) => BroadPhaseKind::UniformGrid,
            Self::DynamicAabbTree(_) => BroadPhaseKind::DynamicAabbTree,
        }
    }
}

impl BroadPhase for BroadPhaseBackend {
    fn update(&mut self, bodies: &[RigidBody], sub_dt: f32) {
        match self {
            Self::SweepAndPrune(backend) => backend.update(bodies, sub_dt),
            Self::UniformGrid(backend) => backend.update(bodies, sub_dt),
            Self::DynamicAabbTree(backend) => backend.update(bodies, sub_dt),
        }
    }

    fn active(&self) -> &[(usize, usize)] {
        match self {
            Self::SweepAndPrune(backend) => backend.active(),
            Self::UniformGrid(backend) => backend.active(),
            Self::DynamicAabbTree(backend) => backend.active(),
        }
    }

    fn stats(&self) -> BroadPhaseStats {
        match self {
            Self::SweepAndPrune(backend) => backend.stats(),
            Self::UniformGrid(backend) => backend.stats(),
            Self::DynamicAabbTree(backend) => backend.stats(),
        }
    }
}

pub(crate) fn swept_aabbs(bodies: &[RigidBody], sub_dt: f32) -> Vec<AABB> {
    bodies
        .iter()
        .map(|body| {
            let mut aabb = body.shape.aabb(body.position, body.orientation);
            if body.body_type == BodyType::Dynamic {
                let displacement = body.velocity * sub_dt;
                aabb.expand(aabb.min + displacement);
                aabb.expand(aabb.max + displacement);
            }
            let margin = Vec3::splat(HALF_SPEC_MARGIN);
            aabb.expand(aabb.min - margin);
            aabb.expand(aabb.max + margin);
            aabb
        })
        .collect()
}

fn candidate_allowed(
    bodies: &[RigidBody],
    aabbs: &[AABB],
    stats: &mut BroadPhaseStats,
    first: usize,
    second: usize,
) -> bool {
    if !bodies[first].can_collide_with(&bodies[second]) {
        stats.filter_rejections += 1;
        return false;
    }
    if bodies[first].body_type == BodyType::Static
        && bodies[second].body_type == BodyType::Static
        && !bodies[first].is_trigger
        && !bodies[second].is_trigger
    {
        stats.static_static_skips += 1;
        return false;
    }
    if !aabbs[first].overlaps(&aabbs[second]) {
        stats.aabb_rejections += 1;
        return false;
    }
    true
}

fn add_pair(
    pairs: &mut HashSet<(usize, usize)>,
    bodies: &[RigidBody],
    aabbs: &[AABB],
    stats: &mut BroadPhaseStats,
    first: usize,
    second: usize,
) {
    if first == second {
        return;
    }
    stats.pair_tests += 1;
    let (a, b) = if first < second {
        (first, second)
    } else {
        (second, first)
    };
    if candidate_allowed(bodies, aabbs, stats, a, b) {
        pairs.insert((a, b));
    }
}

/// Baseline axis sweep-and-prune broadphase.
pub(crate) struct SweepAndPrune {
    aabbs: Vec<AABB>,
    active: Vec<(usize, usize)>,
    sort_axis: usize,
    stats: BroadPhaseStats,
}

impl SweepAndPrune {
    fn new() -> Self {
        Self {
            aabbs: Vec::new(),
            active: Vec::new(),
            sort_axis: 0,
            stats: BroadPhaseStats::default(),
        }
    }
}

impl BroadPhase for SweepAndPrune {
    fn update(&mut self, bodies: &[RigidBody], sub_dt: f32) {
        self.aabbs = swept_aabbs(bodies, sub_dt);
        self.sort_axis = (self.sort_axis + 1) % 3;
        self.active.clear();

        self.stats = BroadPhaseStats {
            body_count: bodies.len(),
            ..BroadPhaseStats::default()
        };
        let n = self.aabbs.len();
        let mut starts: Vec<(f32, usize)> = self
            .aabbs
            .iter()
            .enumerate()
            .map(|(index, aabb)| match self.sort_axis {
                0 => (aabb.min.x, index),
                1 => (aabb.min.y, index),
                _ => (aabb.min.z, index),
            })
            .collect();
        starts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for sweep_position in 0..n {
            let first = starts[sweep_position].1;
            let sweep_aabb = &self.aabbs[first];
            let end = match self.sort_axis {
                0 => sweep_aabb.max.x,
                1 => sweep_aabb.max.y,
                _ => sweep_aabb.max.z,
            };
            for &(_, second) in &starts[(sweep_position + 1)..] {
                let start = match self.sort_axis {
                    0 => self.aabbs[second].min.x,
                    1 => self.aabbs[second].min.y,
                    _ => self.aabbs[second].min.z,
                };
                if start > end {
                    break;
                }
                self.stats.pair_tests += 1;
                // Canonicalize to (lower, higher) — do NOT skip on `first < second`:
                // `first`/`second` are body indices, not sweep positions, so a
                // higher-index body that sorts earlier on the axis would otherwise
                // lose its pair (e.g. a large static floor vs lower-index dynamics).
                let (a, b) = if first < second {
                    (first, second)
                } else {
                    (second, first)
                };
                if candidate_allowed(bodies, &self.aabbs, &mut self.stats, a, b) {
                    self.active.push((a, b));
                }
            }
        }
    }

    fn active(&self) -> &[(usize, usize)] {
        &self.active
    }

    fn stats(&self) -> BroadPhaseStats {
        let mut stats = self.stats;
        stats.candidate_pairs = self.active.len();
        stats
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CellKey {
    x: i32,
    y: i32,
    z: i32,
}

impl CellKey {
    fn from_point(point: Vec3, cell_size: f32) -> Self {
        Self {
            x: (point.x / cell_size).floor() as i32,
            y: (point.y / cell_size).floor() as i32,
            z: (point.z / cell_size).floor() as i32,
        }
    }
}

/// Uniform grid broadphase. Bodies spanning too many cells use `large` and
/// are compared against all bodies once, avoiding an enormous cell expansion
/// while retaining correctness for large floors and environment pieces.
pub(crate) struct UniformGrid {
    aabbs: Vec<AABB>,
    active: Vec<(usize, usize)>,
    cells: HashMap<CellKey, Vec<usize>>,
    large: Vec<usize>,
    cell_size: f32,
    max_cells_per_body: usize,
    stats: BroadPhaseStats,
    scratch_pairs: HashSet<(usize, usize)>,
    body_cells: Vec<Vec<CellKey>>,
    body_is_large: Vec<bool>,
    prev_meta: Vec<(bool, u32, u32, BodyType)>,
}

impl UniformGrid {
    fn new() -> Self {
        Self::with_cell_size(DEFAULT_GRID_CELL_SIZE)
    }

    fn with_cell_size(cell_size: f32) -> Self {
        assert!(
            cell_size.is_finite() && cell_size > 0.0,
            "uniform grid cell size must be finite and positive, got {cell_size}"
        );
        Self {
            aabbs: Vec::new(),
            active: Vec::new(),
            cells: HashMap::new(),
            large: Vec::new(),
            cell_size,
            max_cells_per_body: DEFAULT_MAX_CELLS_PER_BODY,
            stats: BroadPhaseStats::default(),
            scratch_pairs: HashSet::new(),
            body_cells: Vec::new(),
            body_is_large: Vec::new(),
            prev_meta: Vec::new(),
        }
    }

    fn cell_bounds(&self, aabb: &AABB) -> (CellKey, CellKey, u64) {
        let minimum = CellKey::from_point(aabb.min, self.cell_size);
        let maximum = CellKey::from_point(aabb.max, self.cell_size);
        let span_x = (i64::from(maximum.x) - i64::from(minimum.x) + 1) as u64;
        let span_y = (i64::from(maximum.y) - i64::from(minimum.y) + 1) as u64;
        let span_z = (i64::from(maximum.z) - i64::from(minimum.z) + 1) as u64;
        (
            minimum,
            maximum,
            span_x.saturating_mul(span_y).saturating_mul(span_z),
        )
    }

    fn insert_body(&mut self, body: usize) {
        let (minimum, maximum, cell_count) = self.cell_bounds(&self.aabbs[body]);
        if cell_count > self.max_cells_per_body as u64 {
            self.large.push(body);
            if body < self.body_is_large.len() {
                self.body_is_large[body] = true;
            }
            if body < self.body_cells.len() {
                self.body_cells[body].clear();
            }
            return;
        }
        if body < self.body_is_large.len() {
            self.body_is_large[body] = false;
        }
        let mut keys = Vec::new();
        for x in minimum.x..=maximum.x {
            for y in minimum.y..=maximum.y {
                for z in minimum.z..=maximum.z {
                    let key = CellKey { x, y, z };
                    self.cells.entry(key).or_default().push(body);
                    keys.push(key);
                }
            }
        }
        if body < self.body_cells.len() {
            self.body_cells[body] = keys;
        }
    }

    fn full_rebuild(&mut self, bodies: &[RigidBody], new_aabbs: Vec<AABB>) {
        self.aabbs = new_aabbs;
        self.cells.clear();
        self.large.clear();
        self.scratch_pairs.clear();
        let n = self.aabbs.len();
        self.body_cells.resize(n, Vec::new());
        self.body_is_large.resize(n, false);
        for c in &mut self.body_cells {
            c.clear();
        }
        self.prev_meta.resize(n, (false, 0, 0, BodyType::Static));
        for (i, b) in bodies.iter().enumerate() {
            self.prev_meta[i] = (
                b.is_trigger,
                b.collision_layer,
                b.collision_mask,
                b.body_type,
            );
        }
        self.stats = BroadPhaseStats {
            body_count: bodies.len(),
            ..BroadPhaseStats::default()
        };
        for body in 0..n {
            self.insert_body(body);
        }
        self.stats.occupied_cells = self.cells.len();
        self.stats.large_bodies = self.large.len();
        for occupants in self.cells.values() {
            for (offset, &first) in occupants.iter().enumerate() {
                for &second in &occupants[(offset + 1)..] {
                    add_pair(
                        &mut self.scratch_pairs,
                        bodies,
                        &self.aabbs,
                        &mut self.stats,
                        first,
                        second,
                    );
                }
            }
        }
        for &large in &self.large.clone() {
            for body in 0..self.aabbs.len() {
                add_pair(
                    &mut self.scratch_pairs,
                    bodies,
                    &self.aabbs,
                    &mut self.stats,
                    large,
                    body,
                );
            }
        }
        self.active.clear();
        self.active.extend(self.scratch_pairs.iter().copied());
        self.active.sort_unstable();
    }
}

impl BroadPhase for UniformGrid {
    fn update(&mut self, bodies: &[RigidBody], sub_dt: f32) {
        let new_aabbs = swept_aabbs(bodies, sub_dt);
        // First run or topology change -> full rebuild.
        if self.aabbs.len() != new_aabbs.len()
            || self.body_cells.len() != new_aabbs.len()
            || self.aabbs.is_empty()
        {
            self.full_rebuild(bodies, new_aabbs);
            return;
        }
        // Find dirty bodies (AABB or filter changed).
        let mut dirty = Vec::new();
        for i in 0..new_aabbs.len() {
            let meta = (
                bodies[i].is_trigger,
                bodies[i].collision_layer,
                bodies[i].collision_mask,
                bodies[i].body_type,
            );
            if new_aabbs[i].min != self.aabbs[i].min
                || new_aabbs[i].max != self.aabbs[i].max
                || self.prev_meta[i] != meta
            {
                dirty.push(i);
            }
        }
        if dirty.is_empty() {
            self.aabbs = new_aabbs;
            self.stats.body_count = bodies.len();
            self.stats.occupied_cells = self.cells.len();
            self.stats.large_bodies = self.large.len();
            return;
        }
        // Heuristic: >50% dirty -> full rebuild cheaper than incremental bookkeeping.
        if dirty.len() * 2 > new_aabbs.len() {
            self.full_rebuild(bodies, new_aabbs);
            return;
        }
        // Update prev_meta for dirty bodies before retaining pairs.
        for &d in &dirty {
            self.prev_meta[d] = (
                bodies[d].is_trigger,
                bodies[d].collision_layer,
                bodies[d].collision_mask,
                bodies[d].body_type,
            );
        }
        let dirty_set: HashSet<usize> = dirty.iter().copied().collect();
        // Remove dirty bodies from spatial index.
        for &d in &dirty {
            if self.body_is_large[d] {
                if let Some(pos) = self.large.iter().position(|&x| x == d) {
                    self.large.swap_remove(pos);
                }
            } else {
                // Clone keys to avoid borrow conflict.
                let keys = self.body_cells[d].clone();
                for key in keys {
                    if let Some(vec) = self.cells.get_mut(&key) {
                        if let Some(pos) = vec.iter().position(|&x| x == d) {
                            vec.swap_remove(pos);
                        }
                        if vec.is_empty() {
                            self.cells.remove(&key);
                        }
                    }
                }
            }
            self.body_cells[d].clear();
            self.body_is_large[d] = false;
        }
        // Keep only clean-clean pairs.
        self.scratch_pairs
            .retain(|(a, b)| !dirty_set.contains(a) && !dirty_set.contains(b));
        // Reset per-frame counters but keep honest totals: retained clean-clean
        // counts + new dirty-involved counts = total comparable to full rebuild.
        let retained_tests = self.stats.pair_tests;
        let retained_filter = self.stats.filter_rejections;
        let retained_static = self.stats.static_static_skips;
        let retained_aabb = self.stats.aabb_rejections;
        self.stats = BroadPhaseStats {
            body_count: bodies.len(),
            pair_tests: retained_tests,
            filter_rejections: retained_filter,
            static_static_skips: retained_static,
            aabb_rejections: retained_aabb,
            candidate_pairs: 0,
            occupied_cells: 0,
            large_bodies: 0,
        };
        // Update AABBs before reinsertion so cell_bounds uses new bounds.
        self.aabbs = new_aabbs;
        for &d in &dirty {
            self.insert_body(d);
        }
        self.stats.occupied_cells = self.cells.len();
        self.stats.large_bodies = self.large.len();
        // Generate pairs involving dirty bodies.
        for &d in &dirty {
            if self.body_is_large[d] {
                for o in 0..bodies.len() {
                    add_pair(
                        &mut self.scratch_pairs,
                        bodies,
                        &self.aabbs,
                        &mut self.stats,
                        d,
                        o,
                    );
                }
            } else {
                // Pairs within dirty's cells.
                let keys = self.body_cells[d].clone();
                for key in keys {
                    if let Some(occupants) = self.cells.get(&key).cloned() {
                        for o in occupants {
                            if o == d {
                                continue;
                            }
                            add_pair(
                                &mut self.scratch_pairs,
                                bodies,
                                &self.aabbs,
                                &mut self.stats,
                                d,
                                o,
                            );
                        }
                    }
                }
                // Pairs against large bodies.
                for &large in &self.large.clone() {
                    add_pair(
                        &mut self.scratch_pairs,
                        bodies,
                        &self.aabbs,
                        &mut self.stats,
                        d,
                        large,
                    );
                }
            }
        }
        // stats now holds honest total = retained clean-clean + new dirty-involved;
        // `candidate_pairs` derived from active len.
        self.active.clear();
        self.active.extend(self.scratch_pairs.iter().copied());
        self.active.sort_unstable();
    }

    fn active(&self) -> &[(usize, usize)] {
        &self.active
    }

    fn stats(&self) -> BroadPhaseStats {
        let mut stats = self.stats;
        stats.candidate_pairs = self.active.len();
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> Vec<RigidBody> {
        vec![
            RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 0.0),
            RigidBody::new_sphere(Vec3::new(0.5, 0.0, 0.0), 0.75, 1.0),
            RigidBody::new_sphere(Vec3::new(4.0, 0.0, 0.0), 0.5, 1.0),
            RigidBody::new_box(Vec3::new(-4.0, 0.0, 0.0), Vec3::splat(0.5), 1.0),
        ]
    }

    #[test]
    fn sweep_and_prune_keeps_pairs_where_higher_index_sorts_first() {
        // Regression: SAP must not skip a pair just because the body with the
        // higher index sorts earlier on the sweep axis. A large static floor
        // (index 2) sorts first by min.x but must still pair with lower-index
        // dynamics that overlap it.
        let bodies = vec![
            RigidBody::new_box(Vec3::new(0.0, 0.5, 0.0), Vec3::splat(0.5), 1.0),
            RigidBody::new_box(Vec3::new(0.0, -0.5, 0.0), Vec3::splat(0.5), 1.0),
            RigidBody::new_box(Vec3::new(0.0, -10.0, 0.0), Vec3::splat(20.0), 0.0),
        ];
        let mut sweep = SweepAndPrune::new();
        sweep.update(&bodies, 0.0);
        let mut pairs = sweep.active().to_vec();
        pairs.sort_unstable();
        assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn uniform_grid_matches_sweep_and_prune_candidate_pairs() {
        let bodies = scene();
        let mut sweep = SweepAndPrune::new();
        let mut grid = UniformGrid::new();
        sweep.update(&bodies, 1.0 / 60.0);
        grid.update(&bodies, 1.0 / 60.0);
        let mut sweep_pairs = sweep.active().to_vec();
        sweep_pairs.sort_unstable();
        assert_eq!(grid.active(), sweep_pairs.as_slice());
    }

    #[test]
    fn uniform_grid_deduplicates_bodies_spanning_multiple_cells() {
        let bodies = vec![
            RigidBody::new_box(Vec3::ZERO, Vec3::splat(4.0), 0.0),
            RigidBody::new_sphere(Vec3::new(0.5, 0.0, 0.0), 0.75, 1.0),
        ];
        let mut grid = UniformGrid::new();
        grid.update(&bodies, 0.0);
        assert_eq!(grid.active(), &[(0, 1)]);
        let stats = grid.stats();
        assert_eq!(stats.body_count, 2);
        assert_eq!(stats.candidate_pairs, 1);
        assert!(stats.occupied_cells > 1);
    }

    #[test]
    fn grid_cell_size_keeps_pairs_while_changing_cell_occupancy() {
        let bodies = scene();
        let mut fine = UniformGrid::with_cell_size(1.0);
        let mut coarse = UniformGrid::with_cell_size(4.0);
        fine.update(&bodies, 0.0);
        coarse.update(&bodies, 0.0);
        assert_eq!(fine.active(), coarse.active());
        assert_ne!(fine.stats().occupied_cells, coarse.stats().occupied_cells);
    }

    #[test]
    fn static_static_pairs_are_skipped_but_static_triggers_are_kept() {
        let ordinary = vec![
            RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 0.0),
            RigidBody::new_box(Vec3::new(0.5, 0.0, 0.0), Vec3::splat(1.0), 0.0),
        ];
        let mut grid = UniformGrid::new();
        grid.update(&ordinary, 0.0);
        assert!(grid.active().is_empty());
        assert!(grid.stats().static_static_skips > 0);

        let trigger = vec![
            RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 0.0).with_trigger(true),
            RigidBody::new_box(Vec3::new(0.5, 0.0, 0.0), Vec3::splat(1.0), 0.0),
        ];
        grid.update(&trigger, 0.0);
        assert_eq!(grid.active(), &[(0, 1)]);
    }

    #[test]
    fn large_body_escape_path_keeps_candidate_pairs_correct() {
        let bodies = vec![
            RigidBody::new_box(Vec3::ZERO, Vec3::splat(100.0), 0.0),
            RigidBody::new_sphere(Vec3::new(1.0, 0.0, 1.0), 0.5, 1.0),
            RigidBody::new_sphere(Vec3::new(300.0, 0.0, 0.0), 0.5, 1.0),
        ];
        let mut grid = UniformGrid::new();
        grid.update(&bodies, 0.0);
        assert_eq!(grid.active(), &[(0, 1)]);
    }

    #[test]
    fn filtered_pairs_are_not_emitted_by_either_backend() {
        let bodies = vec![
            RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0).with_collision_filter(0b0001, 0b0010),
            RigidBody::new_sphere(Vec3::new(0.5, 0.0, 0.0), 1.0, 1.0)
                .with_collision_filter(0b0010, 0b0100),
        ];
        let mut sweep = SweepAndPrune::new();
        let mut grid = UniformGrid::new();
        sweep.update(&bodies, 0.0);
        grid.update(&bodies, 0.0);
        assert!(sweep.active().is_empty());
        assert!(grid.active().is_empty());
    }
}
