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
//!
//! Matrix 2026-09-09 (`probe_100k`, cold): candidate pairs are identical to
//! `UniformGrid` on all five 10k scenes (tiled/giant_floor/sparse/
//! islands/heterogeneous). AVL-style single rotations on the insert climb
//! (plus `height` maintenance in `refit_node`) brought depth from 87 to
//! ≤24 on 2000 grid-ordered bodies; the 10k matrix now beats the grid on
//! tiled (68 vs 121 ms), giant_floor (97 vs 141 ms) and islands (13.5 vs
//! 21.7 ms) and ties on sparse/hetero. 100k tiled: tree 759 ms vs grid-8
//! 604 ms — same band, ~10x under the historical 8 s grid number.
//!
//! Stays explicit opt-in for dense worlds; `Auto` routes sparse worlds here
//! since pair buffering. The remaining gap vs the incremental grid is
//! settled dense scenes (the tree rebuilds its sorted vec each update).

use rustc_hash::FxHashSet;

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
    /// Leaf height is 0; internal height is one plus the taller child.
    /// Maintained by `refit_node`; stale values only cost query depth,
    /// never correctness (queries prune on AABBs, not heights).
    height: i32,
}

/// One AABB tree over a set of body proxies, stored in a node pool with a free list.
#[derive(Debug, Clone, Default)]
struct Tree {
    nodes: Vec<Node>,
    free: Vec<usize>,
    root: Option<usize>,
}

impl Tree {
    fn alloc_node(&mut self, aabb: AABB, body: Option<usize>, height: i32) -> usize {
        if let Some(i) = self.free.pop() {
            self.nodes[i] = Node {
                aabb,
                parent: None,
                child1: None,
                child2: None,
                body,
                height,
            };
            i
        } else {
            self.nodes.push(Node {
                aabb,
                parent: None,
                child1: None,
                child2: None,
                body,
                height,
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
    /// choice (Box3D `b3DynamicTree` insert) and single rotations on the
    /// climb back up, so grid-order insertion cannot degenerate the tree.
    fn insert_leaf(&mut self, leaf: usize) {
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
        let new_parent = self.alloc_node(
            self.nodes[node].aabb.union(&leaf_aabb),
            None,
            1 + self.nodes[node].height,
        );
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
            Self::refit_node(&mut self.nodes, c);
            let balanced = self.rotate(c);
            current = self.nodes[balanced].parent;
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
                Self::refit_node(&mut self.nodes, c);
                current = self.nodes[c].parent;
            }
        } else {
            self.root = Some(sibling);
            self.nodes[sibling].parent = None;
            self.free_node(parent);
        }
    }

    /// Recomputes an internal node's union box and height from its children.
    /// Leaves are left alone (their box/height are set at alloc/insert).
    fn refit_node(nodes: &mut [Node], node: usize) {
        if let (Some(c1), Some(c2)) = (nodes[node].child1, nodes[node].child2) {
            let aabb = nodes[c1].aabb.union(&nodes[c2].aabb);
            let height = 1 + nodes[c1].height.max(nodes[c2].height);
            nodes[node].aabb = aabb;
            nodes[node].height = height;
        }
    }

    /// Sets both children of an internal node (and their parent links).
    fn link(&mut self, parent: usize, child1: usize, child2: usize) {
        self.nodes[parent].child1 = Some(child1);
        self.nodes[parent].child2 = Some(child2);
        self.nodes[child1].parent = Some(parent);
        self.nodes[child2].parent = Some(parent);
    }

    /// Moves `new_root` into the tree slot `old_root` occupied. The parent
    /// must be captured *before* any `link` calls (they overwrite parent
    /// pointers, so reading it after would self-loop the tree).
    fn relink_above(&mut self, parent: Option<usize>, old_root: usize, new_root: usize) {
        self.nodes[new_root].parent = parent;
        if let Some(p) = parent {
            if self.nodes[p].child1 == Some(old_root) {
                self.nodes[p].child1 = Some(new_root);
            } else {
                debug_assert_eq!(self.nodes[p].child2, Some(old_root));
                self.nodes[p].child2 = Some(new_root);
            }
        } else {
            self.root = Some(new_root);
        }
    }

    /// Single AVL-style rotation when one child outweighs the other by more
    /// than one level. The shorter grandchild joins the lighter side, so the
    /// subtree height drops by one in every branch except exact ties (which
    /// stay put in height but keep a canonical shape). Returns the node now
    /// at this position. All choices are strict/deterministic, and boxes are
    /// refit bottom-up, so queries stay exact and runs stay reproducible.
    fn rotate(&mut self, a: usize) -> usize {
        let (Some(b), Some(c)) = (self.nodes[a].child1, self.nodes[a].child2) else {
            return a;
        };
        let (bh, ch) = (self.nodes[b].height, self.nodes[c].height);
        let parent = self.nodes[a].parent;
        if ch > bh + 1 {
            let (Some(f), Some(g)) = (self.nodes[c].child1, self.nodes[c].child2) else {
                return a;
            };
            if self.nodes[f].height >= self.nodes[g].height {
                // C' = (A'(B, G), F).
                self.link(a, b, g);
                self.link(c, a, f);
            } else {
                // C' = (A'(B, F), G).
                self.link(a, b, f);
                self.link(c, a, g);
            }
            self.relink_above(parent, a, c);
            Self::refit_node(&mut self.nodes, a);
            Self::refit_node(&mut self.nodes, c);
            c
        } else if bh > ch + 1 {
            let (Some(f), Some(g)) = (self.nodes[b].child1, self.nodes[b].child2) else {
                return a;
            };
            if self.nodes[g].height >= self.nodes[f].height {
                // B' = (A'(F, C), G).
                self.link(a, f, c);
                self.link(b, a, g);
            } else {
                // B' = (A'(G, C), F).
                self.link(a, g, c);
                self.link(b, a, f);
            }
            self.relink_above(parent, a, b);
            Self::refit_node(&mut self.nodes, a);
            Self::refit_node(&mut self.nodes, b);
            b
        } else {
            a
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
    /// Reused query output buffer: one allocation total instead of one per
    /// dynamic body per update.
    scratch: Vec<usize>,
    /// Persistent pair buffer: pairs where neither endpoint was dirty are
    /// retained across updates without re-query (see `refresh_proxies` for
    /// the exact dirty rule). Cleared on any body-count change, where
    /// `swap_remove` remapping could otherwise alias identities.
    active_set: FxHashSet<(usize, usize)>,
    /// Swept box seen per body on the previous update. Dirty means the
    /// swept box (or the filter/type meta) changed; clean pairs are
    /// *exactly* reusable because the filter is a pure function of current
    /// swept boxes and meta.
    prev_swept: Vec<AABB>,
    /// `(is_trigger, collision_layer, collision_mask)` per body, mirroring
    /// the grid's `prev_meta` (body type lives in the proxy kind).
    prev_filter: Vec<(bool, u32, u32)>,
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
            scratch: Vec::new(),
            active_set: FxHashSet::default(),
            prev_swept: Vec::new(),
            prev_filter: Vec::new(),
        }
    }

    fn fat_aabb(swept: AABB) -> AABB {
        // The fat box must cover the swept box: queries run against fats but
        // pairs are filtered by swept overlap, so a fat narrower than the
        // swept motion path would silently drop fast-body pairs that the
        // other backends emit. The extra margin absorbs small velocity
        // changes between updates without forcing a re-insert.
        let margin = Vec3::splat(HALF_SPEC_MARGIN * 4.0);
        AABB {
            min: swept.min - margin,
            max: swept.max + margin,
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
        // `pair_tests` counts checks executed by THIS update (0 on a fully
        // clean one); `candidate_pairs` always reflects the full buffered
        // set, retained or freshly queried.
        self.stats = BroadPhaseStats {
            body_count: bodies.len(),
            ..BroadPhaseStats::default()
        };
        let swept = crate::broadphase::swept_aabbs(bodies, sub_dt);
        // Any count change can remap body indices (`swap_remove`), which
        // would alias buffered pair identities: drop the buffer and re-query
        // everything, exactly like the grid's length-mismatch full rebuild.
        let full = bodies.len() != self.proxies.len();
        self.sync_proxies(bodies.len());
        let (dynamic, dirty, static_changed) = self.refresh_proxies(bodies, &swept);
        if full || static_changed {
            self.active_set.clear();
            self.query_dynamic(bodies, &swept, &dynamic);
        } else if !dirty.is_empty() {
            let dirty_set: FxHashSet<usize> = dirty.iter().copied().collect();
            self.active_set
                .retain(|(a, b)| !dirty_set.contains(a) && !dirty_set.contains(b));
            self.query_dynamic(bodies, &swept, &dirty);
        }
        // Else: nothing moved — the buffer already holds the exact set.
        // Drop duplicate pairs (a body may appear in several leaves) and
        // sort for deterministic output matching the other backends. (The
        // set already dedups; the sorted vec is the contract order.)
        self.active.clear();
        self.active.extend(self.active_set.iter().copied());
        self.active.sort_unstable();
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

    /// Insert new bodies, re-insert escaped proxies into their own tree and
    /// migrate proxies whose body changed type. Returns the indices of every
    /// dynamic body, the dirty dynamic bodies needing a (re-)query, and
    /// whether the static set changed (new / flipped / moved statics force
    /// a full re-query, since unmoved dynamics would otherwise never be
    /// tested against them). Fats are built from swept boxes so every query
    /// covers the full motion path (see `fat_aabb`).
    ///
    /// Dirty is exact, not tolerant: a body is dirty when its swept box or
    /// its filter/type meta differs from the previous update. Clean pairs
    /// are therefore exactly reusable — the pair filter is a pure function
    /// of current swept boxes and meta.
    fn refresh_proxies(
        &mut self,
        bodies: &[RigidBody],
        swept: &[AABB],
    ) -> (Vec<usize>, Vec<usize>, bool) {
        let mut dynamic = Vec::new();
        let mut dirty = Vec::new();
        let mut static_changed = false;
        for (i, body) in bodies.iter().enumerate() {
            let filter = (body.is_trigger, body.collision_layer, body.collision_mask);
            let want_static = body.body_type == BodyType::Static;
            if self.proxies.get(i).and_then(|p| p.as_ref()).is_none() {
                self.insert_new(i, body.body_type, Self::fat_aabb(swept[i]));
                if want_static {
                    static_changed = true;
                } else {
                    dynamic.push(i);
                    dirty.push(i);
                }
                continue;
            }
            let known = i < self.prev_swept.len() && i < self.prev_filter.len();
            let swept_same = known
                && self.prev_swept[i].min == swept[i].min
                && self.prev_swept[i].max == swept[i].max;
            let filter_same = known && self.prev_filter[i] == filter;
            // Copy out before any tree call so the proxy borrow ends first.
            let (kind, node) = {
                let proxy = self.proxies[i].as_ref().unwrap();
                (proxy.kind, proxy.node)
            };
            if (kind == TreeKind::Static) != want_static {
                // Body changed type: drop from the old tree, insert into the
                // matching one. Without this a proxy would linger in the
                // wrong tree after a static/dynamic flip.
                self.tree_of(kind).remove_leaf(node);
                self.tree_of(kind).free_node(node);
                self.proxies[i] = None;
                self.insert_new(i, body.body_type, Self::fat_aabb(swept[i]));
                if want_static {
                    static_changed = true;
                } else {
                    dynamic.push(i);
                    dirty.push(i);
                }
                continue;
            }
            if !swept_same || !filter_same {
                if want_static {
                    // Same pose but new filters: unmoved dynamics would never
                    // be re-tested against it — take the full path instead of
                    // risking a stale buffered pair.
                    static_changed = true;
                } else {
                    dirty.push(i);
                }
            }
            if !self.proxies[i]
                .as_ref()
                .unwrap()
                .fat
                .contains_aabb(&swept[i])
            {
                let fat = Self::fat_aabb(swept[i]);
                let proxy = self.proxies[i].as_mut().unwrap();
                proxy.fat = fat;
                proxy.moved = true;
                let tree = self.tree_of(kind);
                tree.remove_leaf(node);
                tree.nodes[node].aabb = fat;
                tree.insert_leaf(node);
            }
            if !want_static {
                dynamic.push(i);
            }
        }
        // Publish current history for the next update's dirty check.
        self.prev_swept.resize(
            bodies.len(),
            AABB {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
            },
        );
        self.prev_filter.resize(bodies.len(), (false, 0, 0));
        for (i, body) in bodies.iter().enumerate() {
            self.prev_swept[i] = swept[i];
            self.prev_filter[i] = (body.is_trigger, body.collision_layer, body.collision_mask);
        }
        (dynamic, dirty, static_changed)
    }

    fn insert_new(&mut self, i: usize, kind: BodyType, fat: AABB) {
        let is_static = kind == BodyType::Static;
        let tree = if is_static {
            &mut self.static_tree
        } else {
            &mut self.dynamic_tree
        };
        let node = tree.alloc_node(fat, Some(i), 0);
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

    /// Query the given dynamic bodies against both trees and insert passing
    /// pairs into the persistent buffer (duplicates collapse in the set;
    /// `update` rebuilds the sorted contract vec from it).
    fn query_dynamic(&mut self, bodies: &[RigidBody], swept: &[AABB], dynamic: &[usize]) {
        for &a in dynamic {
            // Take the scratch buffer so the tree queries can borrow `self`
            // mutably; capacity is restored at the end of the iteration.
            let mut candidates = std::mem::take(&mut self.scratch);
            candidates.clear();
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
                    self.active_set.insert((lo, hi));
                }
            }
            self.scratch = candidates;
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

    fn moving_scene() -> Vec<RigidBody> {
        // A fast body (30 m/s covers 0.5 m per step) heading at a static
        // wall, plus a falling body and a fast-fast crossing pair. The wall
        // sits at base.min 0.8: inside the swept path (swept max ~0.95) but
        // outside any base-derived fat box (fat max 0.5), so a fat-only
        // query would silently drop pair (0, 1).
        let mut fast = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.4), 1.0);
        fast.velocity = Vec3::new(30.0, 0.0, 0.0);
        let wall = RigidBody::new_box(Vec3::new(1.2, 0.0, 0.0), Vec3::splat(0.4), 0.0);
        let mut falling = RigidBody::new_sphere(Vec3::new(-3.0, 5.0, 0.0), 0.5, 1.0);
        falling.velocity = Vec3::new(0.0, -20.0, 0.0);
        let mut left = RigidBody::new_sphere(Vec3::new(-6.0, 0.0, 0.0), 0.5, 1.0);
        left.velocity = Vec3::new(25.0, 0.0, 0.0);
        let mut right = RigidBody::new_sphere(Vec3::new(-4.0, 0.0, 0.0), 0.5, 1.0);
        right.velocity = Vec3::new(-25.0, 0.0, 0.0);
        vec![fast, wall, falling, left, right]
    }

    fn tree_pairs(bodies: &[RigidBody], sub_dt: f32) -> Vec<(usize, usize)> {
        let mut tree = BroadPhaseBackend::new(BroadPhaseKind::DynamicAabbTree);
        tree.update(bodies, sub_dt);
        let mut pairs = tree.active().to_vec();
        pairs.sort_unstable();
        pairs
    }

    #[test]
    fn dynamic_tree_matches_oracle_with_velocities() {
        let bodies = moving_scene();
        // Twice: persistence state must not change the contract, and output
        // must be deterministic across updates.
        assert_eq!(tree_pairs(&bodies, 1.0 / 60.0), brute_force_pairs(&bodies));
        assert_eq!(tree_pairs(&bodies, 1.0 / 60.0), brute_force_pairs(&bodies));
    }

    #[test]
    fn dynamic_tree_keeps_fast_body_pairs_past_the_old_fat_margin() {
        // 30 m/s * 1/60 s = 0.5 m of swept path: the wall at base.min 0.8 is
        // inside the swept path but outside any base-derived fat box.
        let bodies = moving_scene();
        assert!(tree_pairs(&bodies, 1.0 / 60.0).contains(&(0, 1)));
    }

    #[test]
    fn dynamic_tree_reinserts_teleported_static_proxies() {
        let mut bodies = scene();
        let mut tree = BroadPhaseBackend::new(BroadPhaseKind::DynamicAabbTree);
        tree.update(&bodies, 1.0 / 60.0);
        // Teleport the static floor (index 4) far away: stale fats must not
        // keep emitting its old pairs.
        bodies[4].position = Vec3::new(500.0, -10.0, 0.0);
        tree.update(&bodies, 1.0 / 60.0);
        let mut pairs = tree.active().to_vec();
        pairs.sort_unstable();
        assert_eq!(pairs, brute_force_pairs(&bodies));
        assert!(!pairs.iter().any(|&(a, b)| a == 4 || b == 4));
    }

    #[test]
    fn dynamic_tree_survives_static_dynamic_type_flips() {
        let mut bodies = scene();
        // Flip the floor to dynamic and back: no panic, pairs track the
        // oracle through the proxy migration.
        bodies[4] = RigidBody::new_box(Vec3::new(0.0, -10.0, 0.0), Vec3::splat(20.0), 1.0);
        let mut tree = BroadPhaseBackend::new(BroadPhaseKind::DynamicAabbTree);
        tree.update(&bodies, 1.0 / 60.0);
        let mut pairs = tree.active().to_vec();
        pairs.sort_unstable();
        assert_eq!(pairs, brute_force_pairs(&bodies));
        bodies[4] = RigidBody::new_box(Vec3::new(0.0, -10.0, 0.0), Vec3::splat(20.0), 0.0);
        tree.update(&bodies, 1.0 / 60.0);
        let mut pairs = tree.active().to_vec();
        pairs.sort_unstable();
        assert_eq!(pairs, brute_force_pairs(&bodies));
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

    /// Iterative max depth of a tree (explicit stack: a degenerate chain
    /// must not overflow the test thread while being measured).
    fn max_depth(tree: &Tree) -> usize {
        let Some(root) = tree.root else {
            return 0;
        };
        let mut deepest = 0usize;
        let mut stack = vec![(root, 0usize)];
        while let Some((node, depth)) = stack.pop() {
            deepest = deepest.max(depth);
            if let Some(c1) = tree.nodes[node].child1 {
                stack.push((c1, depth + 1));
            }
            if let Some(c2) = tree.nodes[node].child2 {
                stack.push((c2, depth + 1));
            }
        }
        deepest
    }

    fn grid_scene_2k() -> Vec<RigidBody> {
        // 2000 dynamic boxes in index order: the insertion pattern the
        // rotation-less tree serves worst.
        let mut bodies = Vec::with_capacity(2000);
        for i in 0..2000u32 {
            bodies.push(RigidBody::new_box(
                Vec3::new((i % 45) as f32 * 2.0, 1.0, (i / 45) as f32 * 2.0),
                Vec3::splat(0.4),
                1.0,
            ));
        }
        bodies
    }

    #[test]
    fn dynamic_tree_stays_bounded_under_grid_insertion_order() {
        let bodies = grid_scene_2k();
        let mut backend = DynamicAabbTree::new();
        backend.update(&bodies, 1.0 / 60.0);
        let depth = max_depth(&backend.dynamic_tree).max(max_depth(&backend.static_tree));
        assert!(
            depth <= 24,
            "dynamic tree degenerated to depth {depth} on 2000 grid bodies"
        );
        // Balanced or not, the contract must hold exactly.
        let mut pairs = backend.active().to_vec();
        pairs.sort_unstable();
        assert_eq!(pairs, brute_force_pairs(&bodies));
    }

    #[test]
    fn dynamic_tree_skips_fully_clean_updates() {
        let bodies = grid_scene_2k();
        let mut backend = DynamicAabbTree::new();
        backend.update(&bodies, 1.0 / 60.0);
        let first = backend.active().to_vec();
        backend.update(&bodies, 1.0 / 60.0);
        // Nothing moved: zero pair checks, identical pairs.
        assert_eq!(backend.stats().pair_tests, 0);
        assert_eq!(backend.active(), first.as_slice());
    }

    #[test]
    fn dynamic_tree_partial_updates_match_oracle() {
        let mut bodies = grid_scene_2k();
        let mut backend = DynamicAabbTree::new();
        backend.update(&bodies, 1.0 / 60.0);
        // Move three bodies (one into overlap, two across the grid).
        bodies[0].position = Vec3::new(2.0, 1.0, 0.0);
        bodies[100].position += Vec3::new(0.0, 3.0, 0.0);
        bodies[1000].position += Vec3::new(5.0, 0.0, 0.0);
        backend.update(&bodies, 1.0 / 60.0);
        let mut pairs = backend.active().to_vec();
        pairs.sort_unstable();
        assert_eq!(pairs, brute_force_pairs(&bodies));
        // Only the dirty bodies were queried — far below a full re-query of
        // all ~2000 dynamics.
        assert!(
            backend.stats().pair_tests < 2000,
            "partial update tested {} pairs",
            backend.stats().pair_tests
        );
    }

    #[test]
    fn dynamic_tree_tracks_filter_changes_without_motion() {
        let mut bodies = grid_scene_2k();
        // Overlap bodies 0 and 1 so they pair under default filters.
        bodies[1].position = Vec3::new(0.5, 1.0, 0.0);
        let mut backend = DynamicAabbTree::new();
        backend.update(&bodies, 1.0 / 60.0);
        let mut first = backend.active().to_vec();
        first.sort_unstable();
        assert!(first.contains(&(0, 1)));
        // Same poses, incompatible filters: the pair must vanish.
        bodies[0] = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.4), 1.0)
            .with_collision_filter(0b0001, 0b0010);
        bodies[1] = RigidBody::new_box(Vec3::new(0.5, 1.0, 0.0), Vec3::splat(0.4), 1.0)
            .with_collision_filter(0b0100, 0b1000);
        backend.update(&bodies, 1.0 / 60.0);
        let mut pairs = backend.active().to_vec();
        pairs.sort_unstable();
        assert_eq!(pairs, brute_force_pairs(&bodies));
        assert!(!pairs.contains(&(0, 1)));
    }

    #[test]
    fn dynamic_tree_handles_add_and_swap_remove() {
        let mut bodies = grid_scene_2k();
        let mut backend = DynamicAabbTree::new();
        backend.update(&bodies, 1.0 / 60.0);
        bodies.push(RigidBody::new_box(
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::splat(0.4),
            1.0,
        ));
        backend.update(&bodies, 1.0 / 60.0);
        let mut pairs = backend.active().to_vec();
        pairs.sort_unstable();
        assert_eq!(pairs, brute_force_pairs(&bodies));
        // Engine-style removal remaps indices: the buffer must not alias.
        bodies.swap_remove(0);
        backend.update(&bodies, 1.0 / 60.0);
        let mut pairs = backend.active().to_vec();
        pairs.sort_unstable();
        assert_eq!(pairs, brute_force_pairs(&bodies));
    }
}
