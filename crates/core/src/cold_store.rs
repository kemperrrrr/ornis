//! Storage for infrequently accessed ("cold") components.
//!
//! Wraps [`ComponentStore`] for component types that are rarely read or
//! written (save data, editor-only metadata, ...). Keeping them in a
//! separate store keeps hot archetype/smart stores dense and cache
//! friendly while still offering full CRUD access.

use crate::component_store::ComponentStore;
use crate::entity::Entity;

/// A [`ComponentStore`] dedicated to cold (rarely touched) components of
/// type `T`. API mirrors [`ComponentStore`] exactly; the split exists
/// purely to keep hot data paths compact.
pub struct ColdComponentStore<T> {
    inner: ComponentStore<T>,
}

impl<T> Default for ColdComponentStore<T> {
    fn default() -> Self {
        Self {
            inner: ComponentStore::new(),
        }
    }
}

impl<T> ColdComponentStore<T> {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces the component for `entity`.
    pub fn insert(&mut self, entity: Entity, component: T) {
        self.inner.insert(entity, component);
    }

    /// Removes and returns the component for `entity`, if present.
    pub fn remove(&mut self, entity: Entity) -> Option<T> {
        self.inner.remove(entity)
    }

    /// Returns a shared reference to the component, if present.
    pub fn get(&self, entity: Entity) -> Option<&T> {
        self.inner.get(entity)
    }

    /// Returns an exclusive reference to the component, if present.
    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        self.inner.get_mut(entity)
    }

    /// Returns `true` if `entity` has a component in this store.
    pub fn contains(&self, entity: Entity) -> bool {
        self.inner.contains(entity)
    }

    /// Iterates over all stored components in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter()
    }

    /// Mutably iterates over all stored components in unspecified order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.inner.iter_mut()
    }

    /// Returns the number of stored components.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if no components are stored.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityAllocator;

    #[test]
    fn insert_get_contains_len() {
        let mut alloc = EntityAllocator::new();
        let mut store = ColdComponentStore::new();

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        let a = alloc.allocate();
        let b = alloc.allocate();
        store.insert(a, 1u32);
        store.insert(b, 2);

        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
        assert!(store.contains(a));
        assert!(store.contains(b));
        assert_eq!(store.get(a), Some(&1));
        assert_eq!(store.get(b), Some(&2));
    }

    #[test]
    fn get_mut_updates_in_place() {
        let mut alloc = EntityAllocator::new();
        let mut store = ColdComponentStore::new();

        let e = alloc.allocate();
        store.insert(e, 10u32);
        *store.get_mut(e).unwrap() += 5;

        assert_eq!(store.get(e), Some(&15));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn remove_returns_value_and_shrinks() {
        let mut alloc = EntityAllocator::new();
        let mut store = ColdComponentStore::new();

        let a = alloc.allocate();
        let b = alloc.allocate();
        store.insert(a, 1u32);
        store.insert(b, 2);

        assert_eq!(store.remove(a), Some(1));
        assert_eq!(store.len(), 1);
        assert!(!store.contains(a));
        assert!(store.contains(b));

        // Removing the last element empties the store.
        assert_eq!(store.remove(b), Some(2));
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn missing_entity_operations() {
        let mut alloc = EntityAllocator::new();
        let mut store: ColdComponentStore<u32> = ColdComponentStore::new();

        let e = alloc.allocate();
        assert!(!store.contains(e));
        assert!(store.get(e).is_none());
        assert!(store.get_mut(e).is_none());
        assert_eq!(store.remove(e), None);
    }

    #[test]
    fn iter_visits_all_elements() {
        let mut alloc = EntityAllocator::new();
        let mut store = ColdComponentStore::new();

        for i in 0..5 {
            store.insert(alloc.allocate(), i);
        }

        let mut collected: Vec<_> = store.iter().copied().collect();
        collected.sort();
        assert_eq!(collected, vec![0, 1, 2, 3, 4]);

        for v in store.iter_mut() {
            *v *= 10;
        }
        let mut scaled: Vec<_> = store.iter().copied().collect();
        scaled.sort();
        assert_eq!(scaled, vec![0, 10, 20, 30, 40]);
    }
}
