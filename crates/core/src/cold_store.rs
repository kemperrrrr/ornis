use crate::component_store::ComponentStore;
use crate::entity::Entity;

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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, entity: Entity, component: T) {
        self.inner.insert(entity, component);
    }

    pub fn remove(&mut self, entity: Entity) -> Option<T> {
        self.inner.remove(entity)
    }

    pub fn get(&self, entity: Entity) -> Option<&T> {
        self.inner.get(entity)
    }

    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        self.inner.get_mut(entity)
    }

    pub fn contains(&self, entity: Entity) -> bool {
        self.inner.contains(entity)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.inner.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
