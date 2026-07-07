use fixedbitset::FixedBitSet;
use rayon::prelude::*;
use crate::entity::Entity;
use crate::page_table::PageTable;

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

impl<T> ComponentStore<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, entity: Entity, component: T) {
        let id = entity.id() as usize;
        if self.bitset.contains(id) {
            if let Some(&dense_idx) = self.sparse.get(id) {
                self.data[dense_idx] = component;
                return;
            }
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
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
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
}
