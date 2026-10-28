// qual:allow(srp, file_length=390) reason: "One dense-set store: slots, sparse index and change log share the same per-entity invariants, and the log is only reachable through the store's mark_* paths"
//! Dense sparse-set component storage.
//!
//! [`ComponentStore<T>`] keeps every live component of one type contiguous
//! for cache-linear iteration: a [`PageTable`] maps entity id → dense index
//! and a bitset marks live ids. Lookups verify the entity generation, so a
//! stale handle can never observe or delete another entity's component.
//! Removal is O(1) swap-with-last — dense order is not stable across
//! removals; [`ComponentStore::defrag`] restores id order.
//!
//! Opt-in mutation logging: [`ComponentStore::enable_change_tracking`] plus
//! [`ComponentStore::take_changes`] record which entity ids were touched
//! since the last take, so consumers (render culling) can skip entities
//! that never changed. Stores that never enable tracking pay one relaxed
//! atomic load per mutating call.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use fixedbitset::FixedBitSet;
use rayon::prelude::*;

use crate::entity::{DenseIndex, Entity};
use crate::layout_clock::next_layout;
use crate::page_table::PageTable;
use crate::prefetch::{PREFETCH_STRIDE, prefetch_iter};

/// Upper bound on logged ids between takes: when the list would grow past
/// it, the ids are dropped and the log goes exhaustive instead, so a hot
/// lane cannot grow the change log without bound.
const CHANGE_ID_CAP: usize = 16 * 1024;

/// Entities mutated in one [`ComponentStore`] since the previous
/// [`ComponentStore::take_changes`], plus a conservative fallback flag.
///
/// [`Self::ids`] may repeat an entity (several mutations between takes)
/// and is meant to be consumed as a set. Whole [`Entity`] handles (id plus
/// generation) are logged, so a recycled id never looks like the removed
/// entity that held the slot before it. [`Self::exhaustive`] short-circuits
/// the set: when `true`, every entity of the store counts as changed —
/// either the id list hit its cap (16k ids) or tracking was enabled after
/// writes that were never logged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentChanges {
    /// Entities whose components were inserted, mutated or removed since
    /// the previous take. Duplicates possible; empty when nothing happened
    /// or when [`Self::exhaustive`] already covers everything.
    pub ids: Vec<Entity>,
    /// `true` when the id list is incomplete and *every* entity of the
    /// store must be treated as changed (overflow, late enable, clone).
    pub exhaustive: bool,
}

/// Bounded entity list behind the store's change-log mutex.
struct ChangeLog {
    ids: Vec<Entity>,
    exhaustive: bool,
}

/// Dense single-type component storage (sparse-set style).
///
/// Classic three-structure sparse set: a [`PageTable`] maps an entity id to
/// its index in the dense `data` array, and a [`FixedBitSet`] records which
/// ids are live. Components of all live entities sit contiguously in
/// `data`, giving cache-linear iteration — the core read path for systems.
///
/// Invariants kept by every operation:
/// * `data[i]` belongs to `entities[i]`, and
///   `sparse.get(entities[i].id()) == i` for every live slot;
/// * lookups verify the entity's generation, so a handle to a destroyed
///   (and possibly recycled) entity can never observe or delete another
///   entity's component.
///
/// The 64-byte alignment keeps the hot header on its own cache line when
/// stores are embedded in larger lane objects. Removal is O(1) via
/// swap-with-last (dense order is not stable across removals; call
/// [`defrag`](ComponentStore::defrag) to restore id order).
#[repr(align(64))]
pub struct ComponentStore<T> {
    /// Dense component values, one per live entity, in insertion order
    /// (modulo swap-on-remove). Iterate this slice for cache-friendly reads.
    pub data: Vec<T>,
    /// Entity owning the component at the same index in [`Self::data`].
    /// Change order or membership only through the store's methods: they
    /// keep [`Self::layout`] in step with this sequence.
    pub entities: Vec<Entity>,
    sparse: PageTable<usize>,
    bitset: FixedBitSet,
    layout: u64,
    /// Mutation-logging gate: off until [`ComponentStore::enable_change_tracking`];
    /// a relaxed load keeps untracked stores on the plain write path.
    track: AtomicBool,
    /// Ids changed since the last [`ComponentStore::take_changes`].
    /// Interior mutability so enable/take work through a shared lane guard.
    changes: Mutex<ChangeLog>,
}

impl<T> Default for ComponentStore<T> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            entities: Vec::new(),
            sparse: PageTable::default(),
            bitset: FixedBitSet::new(),
            layout: next_layout(),
            track: AtomicBool::new(false),
            changes: Mutex::new(ChangeLog {
                ids: Vec::new(),
                exhaustive: false,
            }),
        }
    }
}

impl<T: Clone> Clone for ComponentStore<T> {
    fn clone(&self) -> Self {
        let track = self.track.load(Ordering::Relaxed);
        Self {
            data: self.data.clone(),
            entities: self.entities.clone(),
            sparse: self.sparse.clone(),
            bitset: self.bitset.clone(),
            layout: self.layout,
            track: AtomicBool::new(track),
            // The clone never witnessed the source's logged writes: while
            // tracking is on, its first take must report everything changed.
            changes: Mutex::new(ChangeLog {
                ids: Vec::new(),
                exhaustive: track,
            }),
        }
    }
}

impl<T> ComponentStore<T> {
    /// Creates an empty store with no allocations beyond the defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Version of the entity sequence in [`Self::entities`]. Changes whenever
    /// membership or order changes, and never repeats for a different
    /// sequence in the process, so equal values mean identical `entities`.
    pub fn layout(&self) -> u64 {
        self.layout
    }

    /// Starts logging mutations of this store for change detection.
    ///
    /// Idempotent: the first call flags the log exhaustive, so writes made
    /// before tracking began count as changes of every entity; later calls
    /// are a single relaxed atomic swap. Render extraction enables tracking
    /// lazily on the lanes it drains via [`take_changes`](Self::take_changes).
    pub fn enable_change_tracking(&self) {
        if !self.track.swap(true, Ordering::Relaxed) {
            let mut log = self.change_log();
            log.exhaustive = true;
        }
    }

    /// Drains the change log accumulated since the previous call and
    /// resets it to empty / non-exhaustive.
    ///
    /// [`ComponentChanges::exhaustive`] `true` means the id list hit its
    /// cap or tracking began (or a clone was taken) late — the consumer
    /// must treat every entity of this store as changed.
    pub fn take_changes(&self) -> ComponentChanges {
        let mut log = self.change_log();
        ComponentChanges {
            ids: std::mem::take(&mut log.ids),
            exhaustive: std::mem::take(&mut log.exhaustive),
        }
    }

    /// Recovers from a poisoned change-log mutex: a panic while marking
    /// leaves the ids it already wrote, which is safe to over-report.
    fn change_log(&self) -> std::sync::MutexGuard<'_, ChangeLog> {
        self.changes.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Records `entity` as changed while tracking is on. Over the id cap
    /// the list is dropped and replaced by the exhaustive flag.
    fn mark_changed(&self, entity: Entity) {
        if !self.track.load(Ordering::Relaxed) {
            return;
        }
        let mut log = self.change_log();
        if log.exhaustive {
            return;
        }
        if log.ids.len() >= CHANGE_ID_CAP {
            log.ids.clear();
            log.exhaustive = true;
            return;
        }
        log.ids.push(entity);
    }

    /// Records every live entity as changed — for entry points handing out
    /// `&mut T` over the whole dense slice ([`iter_mut`](Self::iter_mut)
    /// and friends), where any element may be written. One lock for the
    /// loop; over the cap this becomes exhaustive exactly like
    /// [`mark_changed`](Self::mark_changed).
    fn mark_all(&self) {
        if !self.track.load(Ordering::Relaxed) {
            return;
        }
        let mut log = self.change_log();
        if log.exhaustive {
            return;
        }
        for entity in &self.entities {
            if log.ids.len() >= CHANGE_ID_CAP {
                log.ids.clear();
                log.exhaustive = true;
                return;
            }
            log.ids.push(*entity);
        }
    }

    /// Inserts or replaces the component for `entity`.
    ///
    /// Re-insertion overwrites the existing value in place and refreshes
    /// the stored handle (a newer generation must be able to read its own
    /// write). A first insertion appends to the dense arrays. Marks
    /// `entity` changed while tracking is on.
    pub fn insert(&mut self, entity: Entity, component: T) {
        self.mark_changed(entity);
        let id = entity.entity_id().index();
        if self.bitset.contains(id)
            && let Some(&dense_idx) = self.sparse.get(id)
        {
            self.data[dense_idx] = component;
            // A handle may carry a newer generation: without updating the entry,
            // a fresh handle could not read its own component
            // (found by proptest store_matches_hashmap_model).
            if self.entities[dense_idx] != entity {
                self.entities[dense_idx] = entity;
                self.layout = next_layout();
            }
            return;
        }
        let dense_idx = self.data.len();
        self.data.push(component);
        self.entities.push(entity);
        self.layout = next_layout();
        self.sparse.set(id, dense_idx);
        self.bitset.grow(id + 1);
        self.bitset.set(id, true);
    }

    /// Removes the component for `entity`, returning it.
    ///
    /// Uses swap-with-last, so relative order of the remaining components
    /// changes (the moved component's sparse mapping is updated). Returns
    /// `None` if the entity has no component here or the handle is stale.
    /// A successful removal marks the entity changed while tracking is on.
    pub fn remove(&mut self, entity: Entity) -> Option<T> {
        let id = entity.entity_id().index();
        if !self.bitset.contains(id) {
            return None;
        }
        let dense_idx = *self.sparse.get(id)?;
        // A stale handle (different generation) must not remove another
        // entity's component — the same check as in get/contains.
        if entity.generation() != self.entities[dense_idx].generation() {
            return None;
        }
        self.mark_changed(entity);
        self.bitset.set(id, false);

        let last = self.data.len() - 1;
        if dense_idx != last {
            self.data.swap(dense_idx, last);
            self.entities.swap(dense_idx, last);
            let moved_entity = self.entities[dense_idx];
            self.sparse.set(moved_entity.entity_id().index(), dense_idx);
        }

        let component = self.data.pop();
        self.entities.pop();
        self.layout = next_layout();
        component
    }

    /// Returns the component for `entity`, or `None` if absent or the
    /// handle refers to a destroyed generation.
    pub fn get(&self, entity: Entity) -> Option<&T> {
        let id = entity.entity_id().index();
        if !self.bitset.contains(id) {
            return None;
        }
        let dense_idx = *self.sparse.get(id)?;
        if entity.generation() != self.entities[dense_idx].generation() {
            return None;
        }
        Some(&self.data[dense_idx])
    }

    /// Mutable variant of [`get`](ComponentStore::get) with the same
    /// generation-checked semantics. Marks `entity` changed while
    /// tracking is on, whether or not the caller writes through the
    /// returned reference.
    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        let id = entity.entity_id().index();
        if !self.bitset.contains(id) {
            return None;
        }
        let dense_idx = *self.sparse.get(id)?;
        if entity.generation() != self.entities[dense_idx].generation() {
            return None;
        }
        self.mark_changed(entity);
        Some(&mut self.data[dense_idx])
    }

    /// Returns `true` if `entity` currently owns a component here (with
    /// a matching generation).
    pub fn contains(&self, entity: Entity) -> bool {
        let id = entity.entity_id().index();
        if !self.bitset.contains(id) {
            return false;
        }
        let Some(&dense_idx) = self.sparse.get(id) else {
            return false;
        };
        dense_idx < self.data.len() && entity.generation() == self.entities[dense_idx].generation()
    }

    /// Iterates components in dense order with software prefetch hints.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        prefetch_iter!(self.data.iter(), PREFETCH_STRIDE)
    }

    /// Mutably iterates components in dense order with prefetch hints.
    /// Marks every entity changed while tracking is on: the iteration
    /// may write any element through the `&mut T` handles it yields.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.mark_all();
        prefetch_iter!(self.data.iter_mut(), PREFETCH_STRIDE)
    }

    /// Parallel read-only iteration over the dense slice (rayon-backed).
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T>
    where
        T: Sync,
    {
        self.data[..].par_iter()
    }

    /// Parallel mutable iteration over the dense slice (rayon-backed).
    /// Marks every entity changed while tracking is on (see
    /// [`iter_mut`](Self::iter_mut)).
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T>
    where
        T: Send,
    {
        self.mark_all();
        self.data[..].par_iter_mut()
    }

    /// Number of live components (length of the dense array).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if no components are stored.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read-only view of the liveness bitset (one bit per entity id).
    /// Used for set algebra between lanes (see
    /// [`iter_without`](ComponentStore::iter_without)).
    pub fn bitset(&self) -> &FixedBitSet {
        &self.bitset
    }

    /// Maps an entity handle to its index in the dense array, verifying
    /// both presence and generation. Exposed for callers that need to pair
    /// this store with packed GPU-side layouts by index.
    pub fn dense_index(&self, entity: Entity) -> Option<DenseIndex> {
        let id = entity.entity_id().index();
        if !self.bitset.contains(id) {
            return None;
        }
        let dense_idx = *self.sparse.get(id)?;
        if entity.generation() != self.entities[dense_idx].generation() {
            return None;
        }
        Some(DenseIndex::from_raw(dense_idx))
    }

    /// Dense index of `entity` as a plain `usize` (generation-checked).
    pub fn index_of(&self, entity: Entity) -> Option<usize> {
        self.dense_index(entity).map(usize::from)
    }

    /// [`index_of`](ComponentStore::index_of) with a first guess: when
    /// `entities[hint]` already is `entity`, the sparse lookup is skipped. A
    /// wrong hint only costs that lookup, never a wrong answer.
    pub fn index_hinted(&self, entity: Entity, hint: usize) -> Option<usize> {
        if self.entities.get(hint) == Some(&entity) {
            return Some(hint);
        }
        self.index_of(entity)
    }

    /// Whether the dense array lists entities in ascending entity-id order,
    /// so iterating `data` visits them in the order of
    /// [`iter_entities`](ComponentStore::iter_entities).
    pub fn is_id_ordered(&self) -> bool {
        self.entities
            .windows(2)
            .all(|pair| pair[0].entity_id() < pair[1].entity_id())
    }

    /// Iterates `(entity, &A, &B)` pairs for every entity present in both
    /// stores, skipping entities whose generations disagree between them.
    pub fn iter_zip<'a, 'b, U>(&'a self, other: &'b ComponentStore<U>) -> ZipIter<'a, 'b, T, U> {
        let entity_ids: Vec<u32> = self
            .bitset
            .intersection(&other.bitset)
            .map(|i| i as u32)
            .collect();
        ZipIter {
            entity_ids,
            cursor: 0,
            store_a: self,
            store_b: other,
        }
    }

    /// Iterates `(entity, component)` pairs for entities NOT marked in
    /// `exclude` — the "join with absence" pattern used by systems that
    /// require one component but forbid another.
    pub fn iter_without<'a>(
        &'a self,
        exclude: &'a FixedBitSet,
    ) -> impl Iterator<Item = (Entity, &'a T)> + 'a {
        let bits = self.bitset.difference(exclude);
        bits.filter_map(move |id| {
            // A set bit without a sparse entry is unreachable via the public
            // API (insert/remove keep both structures in sync); if that
            // invariant ever breaks, skip the id rather than truncate.
            let dense_idx = (self.sparse.get(id)).copied()?;
            Some((self.entities[dense_idx], &self.data[dense_idx]))
        })
    }

    /// Iterate over all live entities currently stored, without their
    /// component values. Used by packed-component traversal (`Pack::
    /// for_each_packed`) and any caller that needs the entity set of a lane.
    pub fn iter_entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.bitset.ones().filter_map(move |id| {
            let dense_idx = self.sparse.get(id).copied()?;
            let entity = self.entities[dense_idx];
            // Re-check generation so a recycled id is not yielded.
            if entity.generation() == self.entities[dense_idx].generation() {
                Some(entity)
            } else {
                None
            }
        })
    }

    /// Mutably iterates the dense array in chunks of exactly 4 elements
    /// (SIMD-friendly width); any remainder is reachable via
    /// [`ChunkedIterMut::into_tail`]. Marks every entity changed while
    /// tracking is on (see [`iter_mut`](Self::iter_mut)).
    pub fn chunked_iter_mut(&mut self) -> ChunkedIterMut<'_, T> {
        /// SIMD-friendly dense-lane chunk width.
        const CHUNK_WIDTH: usize = 4;
        self.mark_all();
        let n = self.data.len();
        let chunk_end = (n / CHUNK_WIDTH) * CHUNK_WIDTH;
        let (head, tail) = self.data.split_at_mut(chunk_end);
        ChunkedIterMut {
            chunks: head.chunks_exact_mut(CHUNK_WIDTH),
            tail: if tail.is_empty() { None } else { Some(tail) },
        }
    }

    /// Parallel mutable iteration in chunks of 4 elements, flattened back
    /// into individual items. Marks every entity changed while tracking
    /// is on (see [`iter_mut`](Self::iter_mut)).
    pub fn chunked_par_iter_mut(&mut self) -> impl ParallelIterator<Item = &mut T>
    where
        T: Send,
    {
        /// SIMD-friendly dense-lane chunk width.
        const CHUNK_WIDTH: usize = 4;
        self.mark_all();
        self.data.par_chunks_exact_mut(CHUNK_WIDTH).flatten()
    }

    /// Reorders the dense array by ascending entity id, restoring a
    /// canonical layout after swap-on-remove churn. O(n log n) and
    /// allocation-heavy — call at explicit maintenance points, not per frame.
    /// Marks every entity changed while tracking is on: dense order feeds
    /// downstream consumers that pair slots with this sequence.
    pub fn defrag(&mut self)
    where
        T: Clone,
    {
        if self.data.len() < 2 {
            return;
        }
        self.mark_all();
        let mut sorted: Vec<(Entity, T)> = self
            .entities
            .iter()
            .zip(self.data.iter())
            .map(|(e, d)| (*e, d.clone()))
            .collect();
        sorted.sort_by_key(|(e, _)| e.id());

        self.data.clear();
        self.entities.clear();
        for (entity, component) in sorted {
            self.sparse.set(entity.entity_id().index(), self.data.len());
            self.data.push(component);
            self.entities.push(entity);
        }
        self.layout = next_layout();
    }
}

/// Iterator produced by [`ComponentStore::iter_zip`]: yields the shared
/// entities of two stores together with references to both components.
pub struct ZipIter<'a, 'b, A, B> {
    entity_ids: Vec<u32>,
    cursor: usize,
    store_a: &'a ComponentStore<A>,
    store_b: &'b ComponentStore<B>,
}

impl<'a, 'b, A, B> ZipIter<'a, 'b, A, B> {
    /// Looks up the component pair for one entity id. `None` means "skip
    /// this id" — never "stop iterating":
    ///
    /// * a set bit without a sparse entry is unreachable via the public
    ///   API (insert/remove keep both structures in sync), but if that
    ///   invariant ever breaks, the id is skipped defensively instead of
    ///   silently aborting the whole iteration;
    /// * a recycled slot may hold different generations in the two stores
    ///   (one store missed the component cleanup on destroy): no live
    ///   entity owns both halves, so the pair is skipped.
    fn lookup(&self, id: usize) -> Option<(Entity, &'a A, &'b B)> {
        let dense_a = self.store_a.sparse.get(id).copied()?;
        let entity = self.store_a.entities[dense_a];
        let dense_b = self.store_b.sparse.get(id).copied()?;
        if entity.generation() != self.store_b.entities[dense_b].generation() {
            return None;
        }
        Some((
            entity,
            &self.store_a.data[dense_a],
            &self.store_b.data[dense_b],
        ))
    }
}

impl<'a, 'b, A, B> Iterator for ZipIter<'a, 'b, A, B> {
    type Item = (Entity, &'a A, &'b B);

    fn next(&mut self) -> Option<Self::Item> {
        while self.cursor < self.entity_ids.len() {
            let id = self.entity_ids[self.cursor] as usize;
            self.cursor += 1;
            if let Some(item) = self.lookup(id) {
                return Some(item);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Skips (generation mismatches, defensive sparse misses) make the
        // remaining id count an upper bound only — the lower bound is 0.
        let remaining = self.entity_ids.len() - self.cursor;
        (0, Some(remaining))
    }
}

/// Chunk-of-4 mutable iterator from
/// [`ComponentStore::chunked_iter_mut`]; the leftover `< 4` elements are
/// available through [`into_tail`](Self::into_tail).
pub struct ChunkedIterMut<'a, T> {
    chunks: std::slice::ChunksExactMut<'a, T>,
    tail: Option<&'a mut [T]>,
}

impl<'a, T> Iterator for ChunkedIterMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.chunks.size_hint()
    }
}

impl<'a, T> ChunkedIterMut<'a, T> {
    /// Consumes the iterator, returning the trailing remainder shorter
    /// than one full 4-element chunk (if any).
    pub fn into_tail(self) -> Option<&'a mut [T]> {
        self.tail
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityAllocator;

    /// The two lines every change-tracking test opens with, in one place:
    /// spelled out per test they read as a duplicated code fragment.
    fn tracked() -> (EntityAllocator, ComponentStore<i32>) {
        (EntityAllocator::new(), ComponentStore::new())
    }

    /// `n` freshly allocated entities, in allocation order.
    fn entities(alloc: &mut EntityAllocator, n: usize) -> Vec<Entity> {
        (0..n).map(|_| alloc.allocate()).collect()
    }

    #[test]
    fn insert_get_remove() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<f32> = ComponentStore::new();

        let e1 = alloc.allocate();
        let e2 = alloc.allocate();

        store.insert(e1, 1.0);
        store.insert(e2, 2.0);

        assert_eq!(store.get(e1), Some(&1.0));
        assert_eq!(store.get(e2), Some(&2.0));
        assert_eq!(store.len(), 2);

        assert_eq!(store.remove(e1), Some(1.0));
        assert_eq!(store.len(), 1);
        assert!(store.get(e1).is_none());
        assert_eq!(store.get(e2), Some(&2.0));
    }

    #[test]
    fn generation_check() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<u32> = ComponentStore::new();

        let e = alloc.allocate();
        store.insert(e, 42);
        assert_eq!(store.get(e), Some(&42));

        store.remove(e);
        alloc.deallocate(e);
        assert!(store.get(e).is_none());

        let e2 = alloc.allocate();
        assert_eq!(e2.id(), e.id());
        assert_ne!(e2.generation(), e.generation());
        assert!(store.get(e2).is_none());

        store.insert(e2, 99);
        assert_eq!(store.get(e2), Some(&99));
    }

    #[test]
    fn dense_iter() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        for i in 0..5 {
            store.insert(alloc.allocate(), i);
        }

        let collected: Vec<_> = store.iter().copied().collect();
        assert_eq!(collected, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn rayon_par_iter() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        for i in 0..1000 {
            store.insert(alloc.allocate(), i);
        }

        store.par_iter_mut().for_each(|x| *x *= 2);

        let mut expected: Vec<_> = (0..1000).map(|i| i * 2).collect();
        let mut actual: Vec<_> = store.iter().copied().collect();
        actual.sort();
        expected.sort();
        assert_eq!(actual, expected);
    }

    #[test]
    fn insert_overwrite() {
        let mut alloc = EntityAllocator::new();
        let mut store = ComponentStore::new();

        let e = alloc.allocate();
        store.insert(e, 10);
        store.insert(e, 20);
        assert_eq!(store.get(e), Some(&20));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn zip_iter() {
        let mut alloc = EntityAllocator::new();
        let mut pos = ComponentStore::new();
        let mut vel = ComponentStore::new();

        for i in 0..10 {
            let e = alloc.allocate();
            pos.insert(e, i as f32);
            if i % 2 == 0 {
                vel.insert(e, (i * 10) as f32);
            }
        }

        let results: Vec<_> = pos.iter_zip(&vel).collect();
        assert_eq!(results.len(), 5);
        for &(_, p, v) in &results {
            assert!((v - p * 10.0).abs() < 1e-6);
        }
    }

    #[test]
    fn zip_iter_skips_generation_mismatch() {
        let mut pos: ComponentStore<f32> = ComponentStore::new();
        let mut vel: ComponentStore<f32> = ComponentStore::new();

        // Recycled slot: id 0 is gen 0 in `vel` but gen 1 in `pos` (the
        // destroy path re-seated the entity in pos without ever removing
        // the stale component from vel). No live entity owns both halves,
        // so the zip must skip id 0 entirely.
        vel.insert(Entity::new_with_gen(0, 0), 1.0);
        pos.insert(Entity::new_with_gen(0, 1), 2.0);

        // A genuine pair at a higher id: a skip must not abort the
        // iteration — ids come from the bitset intersection in ascending
        // order, so the stale id 0 is met first.
        let pair = Entity::new_with_gen(5, 0);
        pos.insert(pair, 10.0);
        vel.insert(pair, 20.0);

        let iter = pos.iter_zip(&vel);
        // The upper bound counts both ids, including the one about to be
        // skipped on the generation mismatch.
        assert_eq!(iter.size_hint(), (0, Some(2)));

        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, pair);
        assert_eq!(*results[0].1, 10.0);
        assert_eq!(*results[0].2, 20.0);
    }

    #[test]
    fn without_filter() {
        let mut alloc = EntityAllocator::new();
        let mut store = ComponentStore::new();
        let mut exclude_bits = FixedBitSet::new();

        let mut excluded_entity = None;
        for i in 0..10 {
            let e = alloc.allocate();
            store.insert(e, i);
            if i == 3 {
                excluded_entity = Some(e);
                exclude_bits.grow(e.id() as usize + 1);
                exclude_bits.set(e.id() as usize, true);
            }
        }

        let results: Vec<_> = store.iter_without(&exclude_bits).collect();
        assert_eq!(results.len(), 9);
        for (e, _) in &results {
            assert_ne!(Some(*e), excluded_entity);
        }
    }

    #[test]
    fn chunked_iter() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        for i in 0..12 {
            store.insert(alloc.allocate(), i);
        }

        let mut iter = store.chunked_iter_mut();
        for chunk in &mut iter {
            for val in chunk.iter_mut() {
                *val += 1;
            }
        }
        if let Some(tail) = iter.into_tail() {
            for val in tail.iter_mut() {
                *val += 1;
            }
        }

        let total: i32 = store.iter().sum();
        // Type annotation is required: serde_json (an ornis-core dependency) adds
        // `impl PartialEq<serde_json::Value> for i32` — without an explicit type the sum
        // in assert_eq! is ambiguous (E0283).
        assert_eq!(total, (0..12).map(|x| x + 1).sum::<i32>());
    }

    #[test]
    fn defrag_sorts_by_id() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        let mut entities = Vec::new();
        for _ in 0..5 {
            entities.push(alloc.allocate());
        }

        store.insert(entities[2], 20);
        store.insert(entities[0], 10);
        store.insert(entities[4], 40);
        store.insert(entities[1], 15);
        store.insert(entities[3], 30);

        store.defrag();

        let dense_ids: Vec<u32> = store.entities.iter().map(|e| e.id()).collect();
        assert_eq!(dense_ids, vec![0, 1, 2, 3, 4]);

        assert_eq!(store.get(entities[0]), Some(&10));
        assert_eq!(store.get(entities[1]), Some(&15));
        assert_eq!(store.get(entities[2]), Some(&20));
        assert_eq!(store.get(entities[3]), Some(&30));
        assert_eq!(store.get(entities[4]), Some(&40));
    }

    #[test]
    fn defrag_two_unsorted_elements() {
        // Exactly two elements, inserted out of id order: the early-return
        // guard `len < 2` must not swallow this case (mutants replacing
        // `<` with `==`/`<=` return without sorting and leave the store
        // in unsorted order).
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        // allocate() hands out ids in order, so insert them in reverse
        // id order to make the dense array genuinely unsorted.
        let first = alloc.allocate(); // id 0
        let second = alloc.allocate(); // id 1
        store.insert(second, 20);
        store.insert(first, 10);

        store.defrag();

        let dense_ids: Vec<u32> = store.entities.iter().map(|e| e.id()).collect();
        assert_eq!(dense_ids, vec![0, 1]);
        assert_eq!(store.get(first), Some(&10));
        assert_eq!(store.get(second), Some(&20));
    }

    #[test]
    fn is_empty_true_and_false() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        assert!(store.is_empty());

        store.insert(alloc.allocate(), 1);
        assert!(!store.is_empty());

        // Empty again after removal — the counter must be exact.
        store.remove(Entity::new(0));
        assert!(store.is_empty());
    }

    #[test]
    fn dense_index_returns_index() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        let a = alloc.allocate();
        let b = alloc.allocate();
        store.insert(a, 1);
        store.insert(b, 2);

        assert_eq!(store.dense_index(a), Some(DenseIndex::from_raw(0)));
        assert_eq!(store.dense_index(b), Some(DenseIndex::from_raw(1)));

        // A stale generation must not map to the live entry.
        let stale = Entity::new_with_gen(a.id(), a.generation() + 1);
        assert_eq!(store.dense_index(stale), None);
    }

    #[test]
    fn bitset_tracks_insert() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        let a = alloc.allocate();
        let b = alloc.allocate();
        store.insert(a, 1);
        store.insert(b, 2);

        assert!(store.bitset().contains(a.id() as usize));
        assert!(store.bitset().contains(b.id() as usize));

        store.remove(a);
        assert!(!store.bitset().contains(a.id() as usize));
        assert!(store.bitset().contains(b.id() as usize));
    }

    #[test]
    fn clone_preserves_data() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        let a = alloc.allocate();
        let b = alloc.allocate();
        store.insert(a, 10);
        store.insert(b, 20);

        let copy = store.clone();
        assert_eq!(copy.len(), 2);
        assert_eq!(copy.get(a), Some(&10));
        assert_eq!(copy.get(b), Some(&20));

        // The clone is independent: mutating it must not touch the source.
        let mut copy = copy;
        copy.insert(a, 99);
        assert_eq!(store.get(a), Some(&10));
    }

    #[test]
    fn zip_iter_size_hint_tracks_remaining() {
        let mut alloc = EntityAllocator::new();
        let mut pos = ComponentStore::new();
        let mut vel = ComponentStore::new();

        for _ in 0..5 {
            let e = alloc.allocate();
            pos.insert(e, 0);
            vel.insert(e, 0);
        }

        let mut iter = pos.iter_zip(&vel);
        // The upper bound tracks the remaining entity ids; the lower bound
        // is 0 because entries may be skipped (see
        // zip_iter_skips_generation_mismatch).
        assert_eq!(iter.size_hint(), (0, Some(5)));

        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert_eq!(iter.size_hint(), (0, Some(3)));
    }

    #[test]
    fn chunked_iter_mut_sizes_and_tail() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        for i in 0..13 {
            store.insert(alloc.allocate(), i);
        }

        // 13 elements: three exact 4-element chunks plus a 1-element tail.
        let mut iter = store.chunked_iter_mut();
        let mut chunk_sizes = Vec::new();
        let mut seen = 0usize;
        for chunk in &mut iter {
            chunk_sizes.push(chunk.len());
            for v in chunk.iter_mut() {
                *v += 1;
                seen += 1;
            }
        }
        let tail = iter.into_tail().expect("13 elements must leave a tail");
        assert_eq!(tail.len(), 1);
        tail[0] += 1;
        seen += 1;

        assert_eq!(chunk_sizes, vec![4, 4, 4]);
        assert_eq!(seen, 13);
        assert_eq!(
            store.iter().sum::<i32>(),
            (0..13).map(|x| x + 1).sum::<i32>()
        );
    }

    #[test]
    fn chunked_iter_mut_exact_multiple_has_no_tail() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        for i in 0..8 {
            store.insert(alloc.allocate(), i);
        }

        let mut iter = store.chunked_iter_mut();
        let mut chunk_sizes = Vec::new();
        for chunk in &mut iter {
            chunk_sizes.push(chunk.len());
        }
        assert_eq!(chunk_sizes, vec![4, 4]);
        assert!(iter.into_tail().is_none());
    }

    #[test]
    fn chunked_iter_mut_size_hint() {
        let mut alloc = EntityAllocator::new();
        let mut store: ComponentStore<i32> = ComponentStore::new();

        for i in 0..12 {
            store.insert(alloc.allocate(), i);
        }

        let mut iter = store.chunked_iter_mut();
        assert_eq!(iter.size_hint(), (3, Some(3)));

        assert!(iter.next().is_some());
        assert_eq!(iter.size_hint(), (2, Some(2)));
    }

    #[test]
    fn change_tracking_stays_off_until_enabled() {
        let (mut alloc, mut store) = tracked();
        let [e1, e2] = entities(&mut alloc, 2)[..] else {
            unreachable!("two entities")
        };

        store.insert(e1, 1);
        *store.get_mut(e1).unwrap() = 2;
        store.remove(e1);
        store.insert(e2, 3);
        store.iter_mut().for_each(|value| *value += 1);
        store.par_iter_mut().for_each(|value| *value += 1);
        store.defrag();

        // Without enable the log stays silent — the gate is the atomic flag.
        assert_eq!(
            store.take_changes(),
            ComponentChanges {
                ids: Vec::new(),
                exhaustive: false
            }
        );
    }

    #[test]
    fn first_enable_is_exhaustive_then_muts_logged() {
        let (mut alloc, mut store) = tracked();
        let before = alloc.allocate();
        store.insert(before, 1); // written before tracking: never logged

        store.enable_change_tracking();
        let changes = store.take_changes();
        assert!(changes.exhaustive, "pre-enable writes must count as all");
        assert!(changes.ids.is_empty());

        // A second enable is a no-op: no fresh exhaustive snapshot.
        store.enable_change_tracking();
        assert!(!store.take_changes().exhaustive);

        // From here only real mutations appear, non-exhaustively.
        let after = alloc.allocate();
        store.insert(after, 2);
        let changes = store.take_changes();
        assert!(!changes.exhaustive);
        assert_eq!(changes.ids, vec![after]);
    }

    #[test]
    fn take_drains_and_resets() {
        let (mut alloc, mut store) = tracked();
        store.enable_change_tracking();
        let _ = store.take_changes(); // consume the first-enable snapshot

        let e = alloc.allocate();
        store.insert(e, 1);
        assert_eq!(store.take_changes().ids, vec![e]);

        // Nothing new: the log comes back empty, not stale, not exhaustive.
        let changes = store.take_changes();
        assert!(changes.ids.is_empty());
        assert!(!changes.exhaustive);
    }

    #[test]
    fn get_mut_remove_and_stale_handles_mark() {
        let (mut alloc, mut store) = tracked();
        let [e1, e2] = entities(&mut alloc, 2)[..] else {
            unreachable!("two entities")
        };
        store.insert(e1, 1);
        store.insert(e2, 2);
        store.enable_change_tracking();
        let _ = store.take_changes();

        // get_mut marks even when the caller never writes.
        let _ = store.get_mut(e1);
        assert_eq!(store.take_changes().ids, vec![e1]);

        // A stale generation neither removes nor marks.
        let stale = Entity::new_with_gen(e1.id(), e1.generation() + 1);
        assert_eq!(store.remove(stale), None);
        assert!(store.take_changes().ids.is_empty());

        // A successful removal marks the removed id.
        assert_eq!(store.remove(e1), Some(1));
        assert_eq!(store.take_changes().ids, vec![e1]);
    }

    #[test]
    fn iter_mut_family_marks_all_entities() {
        let (mut alloc, mut store) = tracked();
        let expected = entities(&mut alloc, 5);
        for (i, &e) in expected.iter().enumerate() {
            store.insert(e, i as i32);
        }
        store.enable_change_tracking();
        let _ = store.take_changes();

        // No element is written: the entry points still mark everything.
        store.iter_mut().for_each(|_| {});
        let mut ids = store.take_changes().ids;
        ids.sort_unstable();
        assert_eq!(ids, expected);

        store.par_iter_mut().for_each(|_| {});
        assert_eq!(store.take_changes().ids, expected);

        let mut iter = store.chunked_iter_mut();
        for chunk in &mut iter {
            let _ = chunk;
        }
        let _ = iter.into_tail();
        ids = store.take_changes().ids;
        ids.sort_unstable();
        assert_eq!(ids, expected);

        store.chunked_par_iter_mut().for_each(|_| {});
        ids = store.take_changes().ids;
        ids.sort_unstable();
        assert_eq!(ids, expected);

        store.defrag();
        ids = store.take_changes().ids;
        ids.sort_unstable();
        assert_eq!(ids, expected);
    }

    #[test]
    fn id_overflow_becomes_exhaustive() {
        let (mut alloc, mut store) = tracked();
        store.enable_change_tracking();
        let _ = store.take_changes();

        // CAP distinct ids fill the list; the next mark overflows it.
        for i in 0..=CHANGE_ID_CAP {
            store.insert(alloc.allocate(), i as i32);
        }
        let changes = store.take_changes();
        assert!(changes.exhaustive, "cap overflow must go exhaustive");
        assert!(changes.ids.is_empty());

        // Resets like everything else once drained.
        store.insert(alloc.allocate(), 1);
        assert!(!store.take_changes().exhaustive);
    }

    #[test]
    fn bulk_iter_overflows_to_exhaustive() {
        let (mut alloc, mut store) = tracked();
        // Built while tracking is off, so the log is still empty below.
        for i in 0..=CHANGE_ID_CAP {
            store.insert(alloc.allocate(), i as i32);
        }
        store.enable_change_tracking();
        let _ = store.take_changes();

        // One bulk iteration over the oversized store exhausts the list.
        store.iter_mut().for_each(|_| {});
        let changes = store.take_changes();
        assert!(changes.exhaustive);
        assert!(changes.ids.is_empty());
    }

    #[test]
    fn marks_skip_while_exhaustive() {
        let (mut alloc, mut store) = tracked();
        let e1 = alloc.allocate();
        store.insert(e1, 1);

        // The first-enable snapshot already says "everything changed":
        // neither a per-id mark nor a bulk mark may refill the id list.
        store.enable_change_tracking();
        store.insert(alloc.allocate(), 2);
        store.iter_mut().for_each(|_| {});
        let changes = store.take_changes();
        assert!(changes.exhaustive);
        assert!(
            changes.ids.is_empty(),
            "marks while exhaustive must not refill ids"
        );
    }

    #[test]
    fn clone_reports_exhaustive_while_tracking() {
        let (mut alloc, mut store) = tracked();
        store.enable_change_tracking();
        let _ = store.take_changes();
        let e = alloc.allocate();
        store.insert(e, 1);

        // The clone missed the source's pending write.
        assert!(store.clone().take_changes().exhaustive);

        // An untracked clone still goes exhaustive on its first enable.
        let mut plain: ComponentStore<i32> = ComponentStore::new();
        plain.insert(e, 2);
        let plain_clone = plain.clone();
        plain_clone.enable_change_tracking();
        assert!(plain_clone.take_changes().exhaustive);
    }
}
