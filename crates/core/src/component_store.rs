use fixedbitset::FixedBitSet;
use rayon::prelude::*;

use crate::entity::Entity;
use crate::page_table::PageTable;
use crate::prefetch::{PREFETCH_STRIDE, prefetch_iter};

#[repr(align(64))]
pub struct ComponentStore<T> {
    pub data: Vec<T>,
    pub entities: Vec<Entity>,
    sparse: PageTable<usize>,
    bitset: FixedBitSet,
}

impl<T> Default for ComponentStore<T> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            entities: Vec::new(),
            sparse: PageTable::default(),
            bitset: FixedBitSet::new(),
        }
    }
}

impl<T: Clone> Clone for ComponentStore<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            entities: self.entities.clone(),
            sparse: self.sparse.clone(),
            bitset: self.bitset.clone(),
        }
    }
}

impl<T> ComponentStore<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, entity: Entity, component: T) {
        let id = entity.id() as usize;
        if self.bitset.contains(id)
            && let Some(&dense_idx) = self.sparse.get(id)
        {
            self.data[dense_idx] = component;
            // A handle may carry a newer generation: without updating the entry,
            // a fresh handle could not read its own component
            // (found by proptest store_matches_hashmap_model).
            self.entities[dense_idx] = entity;
            return;
        }
        let dense_idx = self.data.len();
        self.data.push(component);
        self.entities.push(entity);
        self.sparse.set(id, dense_idx);
        self.bitset.grow(id + 1);
        self.bitset.set(id, true);
    }

    pub fn remove(&mut self, entity: Entity) -> Option<T> {
        let id = entity.id() as usize;
        if !self.bitset.contains(id) {
            return None;
        }
        let dense_idx = *self.sparse.get(id)?;
        // A stale handle (different generation) must not remove another
        // entity's component — the same check as in get/contains.
        if entity.generation() != self.entities[dense_idx].generation() {
            return None;
        }
        self.bitset.set(id, false);

        let last = self.data.len() - 1;
        if dense_idx != last {
            self.data.swap(dense_idx, last);
            self.entities.swap(dense_idx, last);
            let moved_entity = self.entities[dense_idx];
            self.sparse.set(moved_entity.id() as usize, dense_idx);
        }

        let component = self.data.pop();
        self.entities.pop();
        component
    }

    pub fn get(&self, entity: Entity) -> Option<&T> {
        let id = entity.id() as usize;
        if !self.bitset.contains(id) {
            return None;
        }
        let dense_idx = *self.sparse.get(id)?;
        if entity.generation() != self.entities[dense_idx].generation() {
            return None;
        }
        Some(&self.data[dense_idx])
    }

    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        let id = entity.id() as usize;
        if !self.bitset.contains(id) {
            return None;
        }
        let dense_idx = *self.sparse.get(id)?;
        if entity.generation() != self.entities[dense_idx].generation() {
            return None;
        }
        Some(&mut self.data[dense_idx])
    }

    pub fn contains(&self, entity: Entity) -> bool {
        let id = entity.id() as usize;
        if !self.bitset.contains(id) {
            return false;
        }
        let Some(&dense_idx) = self.sparse.get(id) else {
            return false;
        };
        dense_idx < self.data.len() && entity.generation() == self.entities[dense_idx].generation()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        prefetch_iter!(self.data.iter(), PREFETCH_STRIDE)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        prefetch_iter!(self.data.iter_mut(), PREFETCH_STRIDE)
    }

    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T>
    where
        T: Sync,
    {
        self.data[..].par_iter()
    }

    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T>
    where
        T: Send,
    {
        self.data[..].par_iter_mut()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn bitset(&self) -> &FixedBitSet {
        &self.bitset
    }

    pub fn dense_index(&self, entity: Entity) -> Option<usize> {
        let id = entity.id() as usize;
        if !self.bitset.contains(id) {
            return None;
        }
        let dense_idx = *self.sparse.get(id)?;
        if entity.generation() != self.entities[dense_idx].generation() {
            return None;
        }
        Some(dense_idx)
    }

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

    pub fn iter_without<'a>(
        &'a self,
        exclude: &'a FixedBitSet,
    ) -> impl Iterator<Item = (Entity, &'a T)> + 'a {
        let bits = self.bitset.difference(exclude);
        bits.filter_map(move |id| {
            let dense_idx = (self.sparse.get(id)).copied()?;
            let entity = self.entities[dense_idx];
            if entity.generation() != self.entities[dense_idx].generation() {
                return None;
            }
            Some((entity, &self.data[dense_idx]))
        })
    }

    pub fn chunked_iter_mut(&mut self) -> ChunkedIterMut<'_, T> {
        let n = self.data.len();
        let chunk_end = (n / 4) * 4;
        let (head, tail) = self.data.split_at_mut(chunk_end);
        ChunkedIterMut {
            chunks: head.chunks_exact_mut(4),
            tail: if tail.is_empty() { None } else { Some(tail) },
        }
    }

    pub fn chunked_par_iter_mut(&mut self) -> impl ParallelIterator<Item = &mut T>
    where
        T: Send,
    {
        self.data.par_chunks_exact_mut(4).flatten()
    }

    pub fn defrag(&mut self)
    where
        T: Clone,
    {
        if self.data.len() < 2 {
            return;
        }
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
            self.sparse.set(entity.id() as usize, self.data.len());
            self.data.push(component);
            self.entities.push(entity);
        }
    }
}

pub struct ZipIter<'a, 'b, A, B> {
    entity_ids: Vec<u32>,
    cursor: usize,
    store_a: &'a ComponentStore<A>,
    store_b: &'b ComponentStore<B>,
}

impl<'a, 'b, A, B> Iterator for ZipIter<'a, 'b, A, B> {
    type Item = (Entity, &'a A, &'b B);

    fn next(&mut self) -> Option<Self::Item> {
        while self.cursor < self.entity_ids.len() {
            let id = self.entity_ids[self.cursor] as usize;
            self.cursor += 1;
            let dense_a = (self.store_a.sparse.get(id)).copied()?;
            let entity = self.store_a.entities[dense_a];
            if entity.generation() != self.store_a.entities[dense_a].generation() {
                continue;
            }
            let dense_b = (self.store_b.sparse.get(id)).copied()?;
            if entity.generation() != self.store_b.entities[dense_b].generation() {
                continue;
            }
            return Some((
                entity,
                &self.store_a.data[dense_a],
                &self.store_b.data[dense_b],
            ));
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entity_ids.len() - self.cursor;
        (remaining, Some(remaining))
    }
}

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
    pub fn into_tail(self) -> Option<&'a mut [T]> {
        self.tail
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityAllocator;

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
        // Аннотация обязательна: serde_json (зависимость ornis-core) добавляет
        // `impl PartialEq<serde_json::Value> for i32` — без явного типа сумма
        // в assert_eq! неоднозначна (E0283).
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

        assert_eq!(store.dense_index(a), Some(0));
        assert_eq!(store.dense_index(b), Some(1));

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
        assert_eq!(iter.size_hint(), (5, Some(5)));

        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert_eq!(iter.size_hint(), (3, Some(3)));
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
        assert_eq!(store.iter().sum::<i32>(), (0..13).map(|x| x + 1).sum::<i32>());
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
}
