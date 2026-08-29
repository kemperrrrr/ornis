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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BroadPhaseStats {
    /// Number of bodies included in the update.
    pub body_count: usize,
    /// Number of raw pair checks before filtering and AABB rejection.
    pub pair_tests: usize,
    /// Pair checks rejected by mutual collision layers/masks.
    pub filter_rejections: usize,
    /// Raw pair checks skipped because both bodies are ordinary static bodies.
    pub static_static_skips: usize,
    /// Pair checks whose swept AABBs do not overlap.
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
                if first < second {
                    self.stats.pair_tests += 1;
                    if candidate_allowed(bodies, &self.aabbs, &mut self.stats, first, second) {
                        self.active.push((first, second));
                    }
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
            return;
        }
        for x in minimum.x..=maximum.x {
            for y in minimum.y..=maximum.y {
                for z in minimum.z..=maximum.z {
                    self.cells
                        .entry(CellKey { x, y, z })
                        .or_default()
                        .push(body);
                }
            }
        }
    }
}

impl BroadPhase for UniformGrid {
    fn update(&mut self, bodies: &[RigidBody], sub_dt: f32) {
        self.aabbs = swept_aabbs(bodies, sub_dt);
        self.active.clear();
        self.cells.clear();
        self.large.clear();
        self.stats = BroadPhaseStats {
            body_count: bodies.len(),
            ..BroadPhaseStats::default()
        };

        for body in 0..self.aabbs.len() {
            self.insert_body(body);
        }
        self.stats.occupied_cells = self.cells.len();
        self.stats.large_bodies = self.large.len();

        let mut pairs = HashSet::new();
        for occupants in self.cells.values() {
            for (offset, &first) in occupants.iter().enumerate() {
                for &second in &occupants[(offset + 1)..] {
                    add_pair(
                        &mut pairs,
                        bodies,
                        &self.aabbs,
                        &mut self.stats,
                        first,
                        second,
                    );
                }
            }
        }
        for &large in &self.large {
            for body in 0..self.aabbs.len() {
                add_pair(
                    &mut pairs,
                    bodies,
                    &self.aabbs,
                    &mut self.stats,
                    large,
                    body,
                );
            }
        }
        self.active = pairs.into_iter().collect();
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
