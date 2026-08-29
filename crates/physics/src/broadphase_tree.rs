//! Experimental persistent dynamic AABB tree broadphase.
//!
//! Implements the ideas from [`docs/quality/broadphase-reference-2026-08-29.md`]
//! (Box3D/Jolt review) without copying their C/C++ source: a persistent proxy
//! per body, a fat AABB margin, separate static/dynamic trees and an
//! active/moved-body list so a step only re-queries bodies that actually
//! moved. The candidate-pair contract is identical to the other backends
//! (`(lower_handle, higher_handle)`, deterministic after sorting).
//!
//! This is an experimental backend. Compare it against `SweepAndPrune` and
//! `UniformGrid` through the physics benchmarks before choosing a default.

use glam::Vec3;

use crate::body::{BodyType, RigidBody};
use crate::broadphase::{BroadPhase, BroadPhaseStats, HALF_SPEC_MARGIN};
use crate::math::AABB;

/// Which tree a proxy belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeKind {
    Static,
    Dynamic,
}

/// Per-body proxy. The node index in its tree is stable across moves; removal
/// uses a free list so body handles (== body indices) stay valid after a
/// `swap_remove` shrinks the body slice.
#[derive(Debug, Clone)]
struct Proxy {
    kind: TreeKind,
    node: usize,
    /// Fat AABB: the inserted box, expanded so small moves need no re-insert.
    fat: AABB,
    /// Set when the body moved this update; cleared after the pair query.
    moved: bool,
}

/// Binary AABB tree node (Box3D `b3DynamicTree` shape): leaf carries a body
/// index, internal nodes carry the union box of their children.
#[derive(Debug, Clone)]
struct Node {
    aabb: AABB,
    parent: Option<usize>,
    child1: Option<usize>,
    child2: Option<usize>,
    /// Leaf nodes reference a body index; internal nodes are `None`.
    body: Option<usize>,
}

/// One AABB tree over a set of body proxies, stored in a node pool with a free list.
#[derive(Debug, Clone, Default)]
struct Tree {
    nodes: Vec<Node>,
    free: Vec<usize>,
    root: Option<usize>,
}

impl Tree {
    fn alloc_node(&mut self, aabb: AABB, body: Option<usize>) -> usize {
        if let Some(i) = self.free.pop() {
            self.nodes[i] = Node {
                aabb,
                parent: None,
                child1: None,
                child2: None,
                body,
            };
            i
        } else {
            self.nodes.push(Node {
                aabb,
                parent: None,
                child1: None,
                child2: None,
                body,
            });
            self.nodes.len() - 1
        }
    }

    fn free_node(&mut self, i: usize) {
        self.nodes[i].parent = None;
        self.nodes[i].child1 = None;
        self.nodes[i].child2 = None;
        self.nodes[i].body = None;
        self.free.push(i);
    }

    /// Insert a leaf node, growing the tree with a greedy SAH-like sibling
    /// choice (Box3D `b3DynamicTree` insert, no rotations yet).
    fn insert_leaf(&mut self, leaf: usize) {
        // ponytail: greedy sibling selection, no tree rotations. Ceiling: no
        // periodic rebalance, so bounds drift for long runs; add rotations when
        // a measured workload shows degradation.
        if self.root.is_none() {
            self.root = Some(leaf);
            self.nodes[leaf].parent = None;
            return;
        }
        let leaf_aabb = self.nodes[leaf].aabb;
        let mut node = self.root.unwrap();
        while self.nodes[node].child1.is_some() {
            let c1 = self.nodes[node].child1.unwrap();
            let c2 = self.nodes[node].child2.unwrap();
            let area = self.nodes[node].aabb.union_area();
            let combined_area = self.nodes[node].aabb.union(&leaf_aabb).union_area();
            let cost = 2.0 * combined_area;
            let descent = 2.0 * (combined_area - area);
            let cost1 = self.nodes[c1].aabb.union(&leaf_aabb).union_area() + descent;
            let cost2 = self.nodes[c2].aabb.union(&leaf_aabb).union_area() + descent;
            if cost < cost1 && cost < cost2 {
                break;
            }
            node = if cost1 < cost2 { c1 } else { c2 };
        }

        let old_parent = self.nodes[node].parent;
        let new_parent = self.alloc_node(self.nodes[node].aabb.union(&leaf_aabb), None);
        self.nodes[new_parent].parent = old_parent;
        self.nodes[new_parent].child1 = Some(node);
        self.nodes[new_parent].child2 = Some(leaf);
        self.nodes[node].parent = Some(new_parent);
        self.nodes[leaf].parent = Some(new_parent);
        if let Some(p) = old_parent {
            if self.nodes[p].child1 == Some(node) {
                self.nodes[p].child1 = Some(new_parent);
            } else {
                self.nodes[p].child2 = Some(new_parent);
            }
        } else {
            self.root = Some(new_parent);
        }
        let mut current = self.nodes[leaf].parent;
        while let Some(c) = current {
            self.nodes[c].aabb = Self::refit(&self.nodes, c);
            current = self.nodes[c].parent;
        }
    }

    fn remove_leaf(&mut self, leaf: usize) {
        let parent = self.nodes[leaf].parent;
        self.nodes[leaf].parent = None;
        let Some(parent) = parent else {
            self.root = None;
            return;
        };
        let grandparent = self.nodes[parent].parent;
        let sibling = if self.nodes[parent].child1 == Some(leaf) {
            self.nodes[parent].child2.unwrap()
        } else {
            self.nodes[parent].child1.unwrap()
        };
        if let Some(gp) = grandparent {
            if self.nodes[gp].child1 == Some(parent) {
                self.nodes[gp].child1 = Some(sibling);
            } else {
                self.nodes[gp].child2 = Some(sibling);
            }
            self.nodes[sibling].parent = Some(gp);
            self.free_node(parent);
            let mut current = Some(gp);
            while let Some(c) = current {
                self.nodes[c].aabb = Self::refit(&self.nodes, c);
                current = self.nodes[c].parent;
            }
        } else {
            self.root = Some(sibling);
            self.nodes[sibling].parent = None;
            self.free_node(parent);
        }
    }

    fn refit(nodes: &[Node], node: usize) -> AABB {
        let n = &nodes[node];
        match (n.child1, n.child2) {
            (Some(c1), Some(c2)) => n.aabb.union(&nodes[c1].aabb).union(&nodes[c2].aabb),
            _ => n.aabb,
        }
    }

    /// Collect every leaf body whose AABB overlaps `target`.
    fn query(&self, target: &AABB, out: &mut Vec<usize>) {
        if let Some(root) = self.root {
            Self::query_recursive(self, root, target, out);
        }
    }

    fn query_recursive(tree: &Tree, node: usize, target: &AABB, out: &mut Vec<usize>) {
        if !tree.nodes[node].aabb.overlaps(target) {
            return;
        }
        if tree.nodes[node].child1.is_none() {
            if let Some(body) = tree.nodes[node].body {
                out.push(body);
            }
            return;
        }
        let c1 = tree.nodes[node].child1.unwrap();
        let c2 = tree.nodes[node].child2.unwrap();
        Self::query_recursive(tree, c1, target, out);
        Self::query_recursive(tree, c2, target, out);
    }
}

/// Persistent dynamic AABB tree broadphase.
pub(crate) struct DynamicAabbTree {
    proxies: Vec<Option<Proxy>>,
    static_tree: Tree,
    dynamic_tree: Tree,
    active: Vec<(usize, usize)>,
    stats: BroadPhaseStats,
}

impl DynamicAabbTree {
    /// Creates an empty tree backend.
    pub(crate) fn new() -> Self {
        Self {
            proxies: Vec::new(),
            static_tree: Tree::default(),
            dynamic_tree: Tree::default(),
            active: Vec::new(),
            stats: BroadPhaseStats::default(),
        }
    }

    fn fat_aabb(base: AABB) -> AABB {
        let margin = Vec3::splat(HALF_SPEC_MARGIN * 4.0);
        AABB {
            min: base.min - margin,
            max: base.max + margin,
        }
    }

    fn tree_of(&mut self, kind: TreeKind) -> &mut Tree {
        match kind {
            TreeKind::Static => &mut self.static_tree,
            TreeKind::Dynamic => &mut self.dynamic_tree,
        }
    }
}

impl BroadPhase for DynamicAabbTree {
    fn update(&mut self, bodies: &[RigidBody], sub_dt: f32) {
        self.active.clear();
        self.stats = BroadPhaseStats {
            body_count: bodies.len(),
            ..BroadPhaseStats::default()
        };
        let swept = crate::broadphase::swept_aabbs(bodies, sub_dt);
        self.sync_proxies(bodies.len());
        // In Ornis every `update` runs on already-integrated poses and most
        // awake bodies move each substep, so we re-query ALL dynamic proxies
        // (a persistent tree still beats a full O(n) sweep on query cost).
        // `moved` is tracked per proxy but not used to skip queries yet — it is
        // the hook for a future sleeping-body fast path.
        let dynamic = self.refresh_proxies(bodies);
        self.query_dynamic(bodies, &swept, &dynamic);
        // Drop duplicate pairs (a body may appear in several leaves) and
        // sort for deterministic output matching the other backends.
        self.active.sort_unstable();
        self.active.dedup();
        self.stats.candidate_pairs = self.active.len();
        for proxy in self.proxies.iter_mut().flatten() {
            proxy.moved = false;
        }
    }

    fn active(&self) -> &[(usize, usize)] {
        &self.active
    }

    fn stats(&self) -> BroadPhaseStats {
        self.stats
    }
}

impl DynamicAabbTree {
    /// Drop proxies whose body index fell off the end of the slice (a
    /// `swap_remove` upstream remapped/shrunk the body list).
    fn sync_proxies(&mut self, body_count: usize) {
        while self.proxies.len() > body_count {
            if let Some(proxy) = self.proxies.pop().flatten() {
                let tree = match proxy.kind {
                    TreeKind::Static => &mut self.static_tree,
                    TreeKind::Dynamic => &mut self.dynamic_tree,
                };
                tree.remove_leaf(proxy.node);
                tree.free_node(proxy.node);
            }
        }
    }

    /// Insert new bodies and re-insert moved dynamic bodies into their tree.
    /// Returns the indices of every dynamic body (the set re-queried each
    /// update — see `update` for why a moved-only set is not yet used).
    fn refresh_proxies(&mut self, bodies: &[RigidBody]) -> Vec<usize> {
        let mut dynamic = Vec::new();
        for (i, body) in bodies.iter().enumerate() {
            let base = body.shape.aabb(body.position, body.orientation);
            let fat = Self::fat_aabb(base);
            if self.proxies.get(i).and_then(|p| p.as_ref()).is_none() {
                self.insert_new(i, body.body_type, fat);
                if body.body_type != BodyType::Static {
                    dynamic.push(i);
                }
                continue;
            }
            let proxy = self.proxies[i].as_mut().unwrap();
            if !proxy.fat.contains_aabb(&base) {
                proxy.fat = fat;
                proxy.moved = true;
                if proxy.kind == TreeKind::Dynamic {
                    let node = proxy.node;
                    let tree = self.tree_of(TreeKind::Dynamic);
                    tree.remove_leaf(node);
                    tree.nodes[node].aabb = fat;
                    tree.insert_leaf(node);
                }
            }
            if body.body_type != BodyType::Static {
                dynamic.push(i);
            }
        }
        dynamic
    }

    fn insert_new(&mut self, i: usize, kind: BodyType, fat: AABB) {
        let is_static = kind == BodyType::Static;
        let tree = if is_static {
            &mut self.static_tree
        } else {
            &mut self.dynamic_tree
        };
        let node = tree.alloc_node(fat, Some(i));
        tree.insert_leaf(node);
        let proxy = Proxy {
            kind: if is_static {
                TreeKind::Static
            } else {
                TreeKind::Dynamic
            },
            node,
            fat,
            moved: true,
        };
        if i >= self.proxies.len() {
            self.proxies.push(Some(proxy));
        } else {
            self.proxies[i] = Some(proxy);
        }
    }

    /// Query every dynamic body against both trees, filter and canonicalize
    /// pairs deterministically into `self.active`.
    fn query_dynamic(&mut self, bodies: &[RigidBody], swept: &[AABB], dynamic: &[usize]) {
        for &a in dynamic {
            let mut candidates = Vec::new();
            let fat = self.proxies[a].as_ref().unwrap().fat;
            self.dynamic_tree.query(&fat, &mut candidates);
            self.static_tree.query(&fat, &mut candidates);
            for &b in &candidates {
                if a == b {
                    continue;
                }
                self.stats.pair_tests += 1;
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if pair_allowed(bodies, swept, &mut self.stats, lo, hi) {
                    self.active.push((lo, hi));
                }
            }
        }
    }
}

impl AABB {
    fn union_area(&self) -> f32 {
        let e = self.max - self.min;
        let (x, y, z) = (e.x.max(0.0), e.y.max(0.0), e.z.max(0.0));
        2.0 * (x * y + y * z + z * x)
    }

    fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Whether `other` is fully inside this (fat) box.
    fn contains_aabb(&self, other: &AABB) -> bool {
        other.min.x >= self.min.x
            && other.max.x <= self.max.x
            && other.min.y >= self.min.y
            && other.max.y <= self.max.y
            && other.min.z >= self.min.z
            && other.max.z <= self.max.z
    }
}

/// Whether the canonical pair `(first, second)` survives broadphase filtering:
/// mutual collision layers/masks, static-static skip (except triggers) and
/// final swept-AABB overlap. Mirrors `candidate_allowed` in `broadphase.rs`
/// but updates this backend's own stats.
fn pair_allowed(
    bodies: &[RigidBody],
    swept: &[AABB],
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
    if !swept[first].overlaps(&swept[second]) {
        stats.aabb_rejections += 1;
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broadphase::{BroadPhaseBackend, BroadPhaseKind, swept_aabbs};

    fn scene() -> Vec<RigidBody> {
        vec![
            RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 0.0),
            RigidBody::new_sphere(Vec3::new(0.5, 0.0, 0.0), 0.75, 1.0),
            RigidBody::new_sphere(Vec3::new(4.0, 0.0, 0.0), 0.5, 1.0),
            RigidBody::new_box(Vec3::new(-4.0, 0.0, 0.0), Vec3::splat(0.5), 1.0),
            // Static floor: must NOT pair with other static bodies, but must
            // pair with every dynamic body overlapping it (exercises the
            // static-tree query path).
            RigidBody::new_box(Vec3::new(0.0, -10.0, 0.0), Vec3::splat(20.0), 0.0),
        ]
    }

    /// Brute-force oracle: every canonical pair whose swept AABBs overlap and
    /// that survives collision filtering. Independent of any backend, so it
    /// catches tree errors that a SAP comparison would mask (SAP itself drops
    /// pairs where a higher body index sorts earlier on the sweep axis).
    fn brute_force_pairs(bodies: &[RigidBody]) -> Vec<(usize, usize)> {
        let swept = swept_aabbs(bodies, 1.0 / 60.0);
        let mut pairs = Vec::new();
        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                if !bodies[i].can_collide_with(&bodies[j]) {
                    continue;
                }
                if bodies[i].body_type == BodyType::Static
                    && bodies[j].body_type == BodyType::Static
                    && !bodies[i].is_trigger
                    && !bodies[j].is_trigger
                {
                    continue;
                }
                if !swept[i].overlaps(&swept[j]) {
                    continue;
                }
                pairs.push((i, j));
            }
        }
        pairs.sort_unstable();
        pairs
    }

    #[test]
    fn dynamic_tree_matches_brute_force_oracle() {
        let bodies = scene();
        let mut tree = BroadPhaseBackend::new(BroadPhaseKind::DynamicAabbTree);
        tree.update(&bodies, 1.0 / 60.0);
        let mut tree_pairs = tree.active().to_vec();
        tree_pairs.sort_unstable();
        assert_eq!(tree_pairs, brute_force_pairs(&bodies));
    }

    #[test]
    fn dynamic_tree_reports_new_pairs_after_a_move() {
        let mut bodies = scene();
        let mut tree = BroadPhaseBackend::new(BroadPhaseKind::DynamicAabbTree);
        tree.update(&bodies, 1.0 / 60.0);
        let mut first = tree.active().to_vec();
        first.sort_unstable();
        // Move a dynamic body into overlap with the far static floor.
        bodies[1].position = Vec3::new(0.0, -9.0, 0.0);
        tree.update(&bodies, 1.0 / 60.0);
        let mut second = tree.active().to_vec();
        second.sort_unstable();
        assert_ne!(first, second);
        assert!(second.contains(&(1, 4)));
    }
}
