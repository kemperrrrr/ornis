//! Parallelism level scheduling mechanics — engine crate
//! (Phase A of the unification plan, audit 2026-08-22 §7).
//!
//! This crate contains only the domain-agnostic part: nodes with access sets
//! (keys `K` in `reads`/`writes`), explicit ordering edges with validation
//! and a unified [`OrderError`], parallelism levels ([`compute_levels`] and
//! the bitset variant [`bitset_level_plan`]), a plan cache with diagnostics
//! ([`PlanCache`]), a level executor ([`run_levels`]: sequential / rayon /
//! wasm-sequential) and a mermaid projector for plan debug diagrams
//! ([`MermaidDiagram`]).
//!
//! Consumers keep domain data on their side and assemble key slices:
//! `ornis-core::schedule::Schedule` schedules systems by singleton
//! resources (key — `TypeId`), `ornis-render::FramePlan` — passes by
//! texture resources (key — `ResourceId`). Texture pools, lifetimes,
//! S4 budget, `Resources`/ECS stay in domain crates (Phase A
//! anti-goals: no wgpu or ECS in this crate).
//!
//! Conflict semantics: nodes `i < j` are ordered if `i` writes what
//! `j` reads or writes (RaW/WaW), or `j` writes what `i` reads
//! (anti-dependency, WaR); registration order is the tie-break, explicit
//! edges only split levels. Bitset plan equivalence with the naive
//! Vec model is pinned by the differential test
//! `bitset_plan_matches_reference_model` below; frontends exercise the same
//! contract on their own types (`core::schedule` — with the eponymous test,
//! `render` — with golden `levels` partitions including the production graph).

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use fixedbitset::FixedBitSet;
#[cfg(not(target_family = "wasm"))]
use rayon::prelude::*;

/// Parallelism level engine — unified for all schedulers in the project
/// (IDEAS §28.2: "one scheduler for everything"). `ordered(i, j)` = there is
/// a dependency i→j (i < j, registration order is the tie-break); level j =
/// 1 + max predecessor levels, groups are emitted in ascending order,
/// and nodes within a group keep registration order.
pub fn compute_levels(n: usize, ordered: impl Fn(usize, usize) -> bool) -> Vec<Vec<usize>> {
    let mut level = vec![0usize; n];
    for j in 1..n {
        let mut best = 0usize;
        for (i, prev_level) in level.iter().enumerate().take(j) {
            if ordered(i, j) {
                best = best.max(prev_level + 1);
            }
        }
        level[j] = best;
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, l) in level.iter().enumerate() {
        if groups.len() <= *l {
            groups.resize_with(*l + 1, Vec::new);
        }
        groups[*l].push(i);
    }
    groups
}

/// Bitset-backed level plan: dense index over distinct access keys,
/// `FixedBitSet` intersections instead of linear `Vec::contains` in the
/// O(n²) loop, explicit edges as an adjacency matrix. Conflict semantics
/// are identical to the naive Vec model (RaW/WaW/WaR); equivalence is
/// pinned by the `bitset_plan_matches_reference_model` test.
///
/// `reads`/`writes` — parallel slices by node (index = registration node).
/// Edges outside the node range are ignored (edge validation lives at the
/// API layer, see [`resolve_named_edge`]).
pub fn bitset_level_plan<K: Copy + Eq + Hash>(
    reads: &[Vec<K>],
    writes: &[Vec<K>],
    ordering: &[(usize, usize)],
) -> Vec<Vec<usize>> {
    assert_eq!(reads.len(), writes.len(), "parallel access slices");
    let nodes = reads.len();
    let mut resource_ids: HashMap<K, u32> = HashMap::new();
    let mut dense = |id: K| -> usize {
        let next = resource_ids.len() as u32;
        *resource_ids.entry(id).or_insert(next) as usize
    };
    let reads: Vec<FixedBitSet> = reads
        .iter()
        .map(|a| a.iter().map(|&id| dense(id)).collect())
        .collect();
    let writes: Vec<FixedBitSet> = writes
        .iter()
        .map(|a| a.iter().map(|&id| dense(id)).collect())
        .collect();
    let mut successors = vec![FixedBitSet::with_capacity(nodes); nodes];
    for &(before, after) in ordering {
        if before < nodes && after < nodes {
            successors[before].insert(after);
        }
    }
    compute_levels(nodes, |i, j| {
        let disjoint_writes_read = writes[i].is_disjoint(&reads[j]);
        let disjoint_writes_writes = writes[i].is_disjoint(&writes[j]);
        let disjoint_read_writes = reads[i].is_disjoint(&writes[j]);
        !(disjoint_writes_read && disjoint_writes_writes && disjoint_read_writes)
            || successors[i].contains(j)
    })
}

/// Error adding an explicit ordering edge — unified type for all
/// frontends (previously near-verbatim duplicates: `OrderError` in core and
/// `GraphOrderError` in render, audit §4.2). Node names are `String`:
/// systems and passes are equivalent at this level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderError {
    /// No node with this name exists (name uniqueness is the caller's responsibility).
    UnknownNode { name: String },
    /// `after` is registered before `before`: execution order is
    /// registration order, explicit edges only split levels.
    BackwardEdge { before: String, after: String },
}

impl fmt::Display for OrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderError::UnknownNode { name } => write!(f, "no node named '{name}'"),
            OrderError::BackwardEdge { before, after } => write!(
                f,
                "'{after}' is registered earlier than '{before}' — register nodes in \
                 execution order; explicit edges only split parallel levels"
            ),
        }
    }
}

impl std::error::Error for OrderError {}

/// Name-based edge validation: names → registration indices via
/// `index_of`; unknown name → [`OrderError::UnknownNode`], backward
/// direction (b ≥ a) → [`OrderError::BackwardEdge`]. Shared part of the
/// name-API of both frontends (audit §4.2).
pub fn resolve_named_edge(
    before: &str,
    after: &str,
    index_of: impl Fn(&str) -> Option<usize>,
) -> Result<(usize, usize), OrderError> {
    let Some(b) = index_of(before) else {
        return Err(OrderError::UnknownNode {
            name: before.to_owned(),
        });
    };
    let Some(a) = index_of(after) else {
        return Err(OrderError::UnknownNode {
            name: after.to_owned(),
        });
    };
    if b >= a {
        return Err(OrderError::BackwardEdge {
            before: before.to_owned(),
            after: after.to_owned(),
        });
    }
    Ok((b, a))
}

/// Index-based edge validation: registration indices are already in hand,
/// `name_of` returns the node name for diagnostics (`None` — no node, the
/// name in the error is synthesized as `"#<index>"`). Shared part of the
/// id-API of both frontends.
pub fn validate_indexed_edge(
    before: usize,
    after: usize,
    name_of: impl Fn(usize) -> Option<String>,
) -> Result<(), OrderError> {
    let Some(before_name) = name_of(before) else {
        return Err(OrderError::UnknownNode {
            name: format!("#{before}"),
        });
    };
    let Some(after_name) = name_of(after) else {
        return Err(OrderError::UnknownNode {
            name: format!("#{after}"),
        });
    };
    if before >= after {
        return Err(OrderError::BackwardEdge {
            before: before_name,
            after: after_name,
        });
    }
    Ok(())
}

/// Level plan cache with diagnostics (project S1-cache style):
/// invalidated by node-graph mutations, recomputed lazily;
/// the [`PlanCache::computations`] counter does not grow in steady state.
/// A poisoned Mutex is recovered silently — the plan stays correct
/// (it is a pure function of graph state).
#[derive(Default)]
pub struct PlanCache {
    cached: Mutex<Option<Vec<Vec<usize>>>>,
    computations: AtomicUsize,
}

impl PlanCache {
    /// Empty (dirty) cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Invalidates the cache (called by frontend mutations).
    pub fn invalidate(&mut self) {
        *self
            .cached
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    /// Returns the cached plan or recomputes `compute` and remembers it.
    pub fn get_or_compute(&self, compute: impl FnOnce() -> Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let mut guard = self
            .cached
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(levels) = guard.as_ref() {
            return levels.clone();
        }
        let levels = compute();
        *guard = Some(levels.clone());
        self.computations.fetch_add(1, Ordering::Relaxed);
        levels
    }

    /// How many times the plan was recomputed — cache diagnostics.
    pub fn computations(&self) -> usize {
        self.computations.load(Ordering::Relaxed)
    }
}

/// Level executor: levels run strictly sequentially, nodes within a level
/// run in parallel (rayon, so `run` must be `Sync`).
/// `parallel = false` — strict registration order 0..nodes (note: it may
/// differ from the level-traversal order, hence nodes are counted via a
/// separate parameter).
#[cfg(not(target_family = "wasm"))]
pub fn run_levels(levels: &[Vec<usize>], nodes: usize, parallel: bool, run: impl Fn(usize) + Sync) {
    if parallel {
        for group in levels {
            group.par_iter().for_each(|&i| run(i));
        }
        return;
    }
    for i in 0..nodes {
        run(i);
    }
}

/// wasm variant of [`run_levels`]: no rayon threads — always strict
/// registration order 0..nodes, so the `Sync` bound on `run` is lifted
/// (web-backend wgpu GPU types are not `Sync`; consumer —
/// `ormis-render::frame_exec` pass recording, backlog #19).
#[cfg(target_family = "wasm")]
pub fn run_levels(levels: &[Vec<usize>], nodes: usize, parallel: bool, run: impl Fn(usize)) {
    let _ = (levels, parallel);
    for i in 0..nodes {
        run(i);
    }
}

/// Debug mermaid projection of the level plan — shared projector
/// (S6 projection of the render shell, generalized in slice 1b of the
/// graph-elimination approach, PLAN.md App. C). Domain-agnostic: nodes and
/// edges are string identifiers/labels assembled by the frontend
/// (`ornis-render::FrameLayout` — `P{i}`/`R{j}` by pass and resource
/// indices, `ornis-core::schedule::Schedule` — `S{i}` by system indices).
/// Levels are rendered as subgraphs, flows as edges; GitHub renders
/// ```mermaid blocks natively, so a plan dump in PR review becomes an
/// image. Labels are inserted verbatim — escaping mermaid syntax (quotes,
/// brackets) stays with the frontend, as it was in the render shell.
#[derive(Debug, Clone)]
pub struct MermaidDiagram {
    out: String,
}

impl Default for MermaidDiagram {
    fn default() -> Self {
        Self {
            out: String::from("flowchart TD\n"),
        }
    }
}

impl MermaidDiagram {
    /// Empty diagram with header `flowchart TD`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Level as subgraph `subgraph {id}["{title}"]` containing nodes
    /// `{node_id}["{label}"]`.
    pub fn level(&mut self, id: &str, title: &str, nodes: &[(String, String)]) -> &mut Self {
        self.out
            .push_str(&format!("  subgraph {id}[\"{title}\"]\n"));
        for (node_id, label) in nodes {
            self.out.push_str(&format!("    {node_id}[\"{label}\"]\n"));
        }
        self.out.push_str("  end\n");
        self
    }

    /// Free top-level node: `{id}["{label}"]`.
    pub fn node(&mut self, id: &str, label: &str) -> &mut Self {
        self.out.push_str(&format!("  {id}[\"{label}\"]\n"));
        self
    }

    /// Flow edge `{from} --> {to}` (direction semantics are defined by the
    /// frontend: resource read/write, explicit ordering edge...).
    pub fn edge(&mut self, from: &str, to: &str) -> &mut Self {
        self.out.push_str(&format!("  {from} --> {to}\n"));
        self
    }

    /// Diagram text (without surrounding ```mermaid — added at the insertion site).
    pub fn render(&self) -> String {
        self.out.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive Vec conflict model — differential test oracle:
    /// i<j are ordered on RaW/WaW (i writes into reads/writes of j) or WaR
    /// (j writes into reads of i); explicit edges are overlaid on top.
    fn naive_conflicts(a_reads: &[u8], a_writes: &[u8], b_reads: &[u8], b_writes: &[u8]) -> bool {
        a_writes
            .iter()
            .any(|w| b_reads.contains(w) || b_writes.contains(w))
            || b_writes.iter().any(|w| a_reads.contains(w))
    }

    fn plan_via_closure(
        reads: &[Vec<u8>],
        writes: &[Vec<u8>],
        ordering: &[(usize, usize)],
    ) -> Vec<Vec<usize>> {
        compute_levels(reads.len(), |i, j| {
            naive_conflicts(&reads[i], &writes[i], &reads[j], &writes[j])
                || ordering.contains(&(i, j))
        })
    }

    #[test]
    fn independent_writers_share_one_level() {
        let reads: Vec<Vec<u8>> = vec![vec![], vec![], vec![]];
        let writes = vec![vec![0], vec![1], vec![2]];
        assert_eq!(plan_via_closure(&reads, &writes, &[]), vec![vec![0, 1, 2]]);
        assert_eq!(bitset_level_plan(&reads, &writes, &[]), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn chain_is_linear_and_anti_dependency_orders_reader_first() {
        // 0 writes k0; 1 reads k0 and writes k1; 2 reads k1.
        let reads: Vec<Vec<u8>> = vec![vec![], vec![0], vec![1]];
        let writes = vec![vec![0], vec![1], vec![]];
        assert_eq!(
            bitset_level_plan(&reads, &writes, &[]),
            vec![vec![0], vec![1], vec![2]]
        );
        // i reads k0, j writes k0 → reader first (anti-dependency).
        let reads: Vec<Vec<u8>> = vec![vec![0], vec![]];
        let writes = vec![vec![], vec![0]];
        assert_eq!(
            bitset_level_plan(&reads, &writes, &[]),
            vec![vec![0], vec![1]]
        );
    }

    #[test]
    fn explicit_edges_only_split_levels() {
        let reads: Vec<Vec<u8>> = vec![vec![], vec![]];
        let writes = vec![vec![0], vec![1]];
        assert_eq!(bitset_level_plan(&reads, &writes, &[]), vec![vec![0, 1]]);
        assert_eq!(
            bitset_level_plan(&reads, &writes, &[(0, 1)]),
            vec![vec![0], vec![1]]
        );
    }

    #[test]
    fn out_of_range_edges_are_ignored() {
        let reads: Vec<Vec<u8>> = vec![vec![]];
        let writes = vec![vec![0]];
        assert_eq!(bitset_level_plan(&reads, &writes, &[(3, 7)]), vec![vec![0]]);
        assert_eq!(bitset_level_plan(&reads, &writes, &[(0, 0)]), vec![vec![0]]);
    }

    /// Differential test (Phase A exit criterion): pseudo-random access
    /// sets (LCG) — the bitset plan must match the naive Vec model, with
    /// and without explicit edges.
    #[test]
    fn bitset_plan_matches_reference_model() {
        let mut lcg = 0x5EED_600Du64;
        let mut next = move || {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lcg
        };
        let mut reads: Vec<Vec<u8>> = Vec::new();
        let mut writes: Vec<Vec<u8>> = Vec::new();
        for _ in 0..12 {
            let mut r = Vec::new();
            let mut w = Vec::new();
            if next() % 2 == 0 {
                r.push((next() % 5) as u8);
            }
            if next() % 3 == 0 {
                r.push((next() % 5) as u8);
            }
            if next() % 2 == 0 {
                w.push((next() % 5) as u8);
            }
            if next() % 3 == 0 {
                w.push((next() % 5) as u8);
            }
            reads.push(r);
            writes.push(w);
        }
        assert_eq!(
            bitset_level_plan(&reads, &writes, &[]),
            plan_via_closure(&reads, &writes, &[]),
            "bitset plan must match the reference model without explicit edges"
        );
        let edges = [(2usize, 8usize), (0, 11)];
        assert_eq!(
            bitset_level_plan(&reads, &writes, &edges),
            plan_via_closure(&reads, &writes, &edges),
            "bitset plan must match the reference model with explicit edges"
        );
    }

    #[test]
    fn named_edge_validates_names_and_direction() {
        let names = ["a", "b"];
        let index_of = |name: &str| names.iter().position(|n| *n == name);
        assert_eq!(resolve_named_edge("a", "b", index_of), Ok((0, 1)));
        assert_eq!(
            resolve_named_edge("b", "a", index_of),
            Err(OrderError::BackwardEdge {
                before: "b".to_owned(),
                after: "a".to_owned(),
            })
        );
        assert_eq!(
            resolve_named_edge("a", "ghost", index_of),
            Err(OrderError::UnknownNode {
                name: "ghost".to_owned(),
            })
        );
    }

    #[test]
    fn indexed_edge_validates_ids_and_direction() {
        let name_of = |i: usize| (i < 2).then(|| format!("n{i}"));
        assert_eq!(validate_indexed_edge(0, 1, name_of), Ok(()));
        assert!(matches!(
            validate_indexed_edge(1, 0, name_of),
            Err(OrderError::BackwardEdge { .. })
        ));
        assert_eq!(
            validate_indexed_edge(0, 99, name_of),
            Err(OrderError::UnknownNode {
                name: "#99".to_owned(),
            })
        );
    }

    #[test]
    fn order_error_display_is_actionable() {
        let unknown = OrderError::UnknownNode {
            name: "ghost".into(),
        };
        assert_eq!(unknown.to_string(), "no node named 'ghost'");
        let backward = OrderError::BackwardEdge {
            before: "a".into(),
            after: "b".into(),
        };
        assert!(backward.to_string().contains("registered earlier"));
    }

    #[test]
    fn plan_cache_hits_and_invalidates() {
        let mut cache = PlanCache::new();
        let compute = || vec![vec![0usize]];
        assert_eq!(cache.computations(), 0);
        assert_eq!(cache.get_or_compute(compute), vec![vec![0]]);
        assert_eq!(cache.get_or_compute(compute), vec![vec![0]]);
        assert_eq!(cache.computations(), 1, "second call is a cache hit");
        cache.invalidate();
        let _ = cache.get_or_compute(compute);
        assert_eq!(cache.computations(), 2, "invalidate forces recompute");
    }

    #[test]
    fn run_levels_sequential_is_registration_order() {
        // Deliberately non-trivial case where level traversal ≠ registration
        // order: sequential mode must keep the latter.
        let trace = Mutex::new(Vec::new());
        run_levels(&[vec![0, 2], vec![1]], 3, false, |i| {
            trace.lock().unwrap().push(i);
        });
        assert_eq!(*trace.lock().unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn run_levels_parallel_covers_same_set() {
        let levels = vec![vec![0, 2], vec![1, 3]];
        let par_trace = Mutex::new(Vec::new());
        run_levels(&levels, 4, true, |i| {
            par_trace.lock().unwrap().push(i);
        });
        let mut par = par_trace.lock().unwrap().clone();
        par.sort_unstable();
        assert_eq!(par, vec![0, 1, 2, 3]);
    }

    #[test]
    fn mermaid_diagram_renders_flowchart() {
        // Byte-level golden for the shared projector (slice 1b): level as
        // subgraph, free node, edge. The render shell pins the same format
        // with its own mermaid_is_a_valid_projection.
        let mut d = MermaidDiagram::new();
        d.level("L0", "level 0", &[("N0".into(), "alpha".into())])
            .node("R0", "shared")
            .edge("R0", "N0");
        assert_eq!(
            d.render(),
            "flowchart TD\n  subgraph L0[\"level 0\"]\n    N0[\"alpha\"]\n  end\n  R0[\"shared\"]\n  R0 --> N0\n"
        );
    }
}
