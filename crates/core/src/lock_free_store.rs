use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

use crossbeam_epoch::{Atomic, Guard, Owned};

use crate::component_store::ComponentStore;
use crate::entity::{Entity, EntityAllocator};

trait LockFreeLane: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn remove_entity(&self, entity: Entity);
}

struct LaneInner<T: Clone + Send + Sync> {
    store: Atomic<ComponentStore<T>>,
}

impl<T: 'static + Clone + Send + Sync> LaneInner<T> {
    fn new() -> Self {
        Self {
            store: Atomic::new(ComponentStore::new()),
        }
    }

    fn read<'g>(&'g self, guard: &'g Guard) -> &'g ComponentStore<T> {
        let shared = self.store.load(Ordering::Acquire, guard);
        unsafe { shared.deref() }
    }

    fn write(&self, f: impl FnOnce(&mut ComponentStore<T>)) {
        let guard = crossbeam_epoch::pin();
        let shared = self.store.load(Ordering::Acquire, &guard);
        let mut new_store = unsafe { (*shared.deref()).clone() };
        f(&mut new_store);
        self.store.store(Owned::new(new_store), Ordering::Release);
        unsafe { guard.defer_destroy(shared) };
    }
}

impl<T: 'static + Clone + Send + Sync> LockFreeLane for LaneInner<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn remove_entity(&self, entity: Entity) {
        self.write(|store| {
            store.remove(entity);
        });
    }
}

pub struct LockFreeStore {
    lanes: HashMap<TypeId, Box<dyn LockFreeLane>>,
    allocator: std::sync::Mutex<EntityAllocator>,
}

impl Default for LockFreeStore {
    fn default() -> Self {
        Self {
            lanes: HashMap::new(),
            allocator: std::sync::Mutex::new(EntityAllocator::new()),
        }
    }
}

impl LockFreeStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: 'static + Clone + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.lanes
            .entry(tid)
            .or_insert_with(|| Box::new(LaneInner::<T>::new()));
    }

    fn ensure_lane<T: 'static + Clone + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.lanes
            .entry(tid)
            .or_insert_with(|| Box::new(LaneInner::<T>::new()));
    }

    pub fn create_entity(&self) -> Entity {
        self.allocator.lock().unwrap().allocate()
    }

    pub fn destroy_entity(&self, entity: Entity) {
        for (_, lane) in self.lanes.iter() {
            lane.remove_entity(entity);
        }
        self.allocator.lock().unwrap().deallocate(entity);
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.allocator.lock().unwrap().is_alive(entity)
    }

    pub fn insert<T: 'static + Clone + Send + Sync>(&mut self, entity: Entity, component: T) {
        self.ensure_lane::<T>();
        let tid = TypeId::of::<T>();
        let lane = self.lanes.get(&tid).unwrap();
        lane.as_any()
            .downcast_ref::<LaneInner<T>>()
            .unwrap()
            .write(|store| {
                store.insert(entity, component);
            });
    }

    pub fn read_lane<T: 'static + Clone + Send + Sync>(&self) -> Option<LockFreeReadGuard<'_, T>> {
        let tid = TypeId::of::<T>();
        let guard = crossbeam_epoch::pin();
        let lane = self.lanes.get(&tid)?;
        let inner = lane.as_any().downcast_ref::<LaneInner<T>>().unwrap();
        // Safety: `guard` is moved into the returned LockFreeReadGuard,
        // so the epoch pin outlives the reference loaded from the lane.
        let store_ptr: *const ComponentStore<T> = inner.read(&guard);
        let store_ref: &ComponentStore<T> = unsafe { &*store_ptr };
        Some(LockFreeReadGuard {
            store: store_ref,
            _guard: guard,
        })
    }
}

pub struct LockFreeReadGuard<'g, T> {
    store: &'g ComponentStore<T>,
    _guard: Guard,
}

impl<'g, T> std::ops::Deref for LockFreeReadGuard<'g, T> {
    type Target = ComponentStore<T>;

    fn deref(&self) -> &Self::Target {
        self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_insert() {
        let mut store = LockFreeStore::new();
        let entity = store.create_entity();
        store.insert::<f32>(entity, 1.0);
        let guard = store.read_lane::<f32>().unwrap();
        assert_eq!(guard.get(entity), Some(&1.0));
    }

    #[test]
    fn entity_lifecycle() {
        let mut store = LockFreeStore::new();
        let e = store.create_entity();
        assert!(store.is_alive(e));
        store.insert::<f32>(e, 10.0);
        store.destroy_entity(e);
        assert!(!store.is_alive(e));
        let guard = store.read_lane::<f32>().unwrap();
        assert!(guard.get(e).is_none());
    }

    #[test]
    fn register_creates_readable_lane() {
        let mut store = LockFreeStore::new();
        store.register::<u32>();
        // A registered lane must be readable even before any insert.
        let guard = store.read_lane::<u32>();
        assert!(guard.is_some());
    }

    #[test]
    fn insert_overwrite_read_back() {
        let mut store = LockFreeStore::new();
        let e = store.create_entity();

        store.insert::<f32>(e, 1.0);
        store.insert::<f32>(e, 2.0);

        let guard = store.read_lane::<f32>().unwrap();
        assert_eq!(guard.len(), 1);
        assert_eq!(guard.get(e), Some(&2.0));
    }

    #[test]
    fn deref_exposes_lane_data() {
        let mut store = LockFreeStore::new();
        let e = store.create_entity();
        store.insert::<f32>(e, 7.5);

        let guard = store.read_lane::<f32>().unwrap();
        // Deref must reach the real lane, not an empty default store.
        assert_eq!((*guard).get(e), Some(&7.5));
    }

    #[test]
    fn destroy_entity_removes_from_all_lanes() {
        let mut store = LockFreeStore::new();
        let e = store.create_entity();
        store.insert::<f32>(e, 1.0);
        store.insert::<u64>(e, 2);

        store.destroy_entity(e);

        let f32_lane = store.read_lane::<f32>().unwrap();
        assert!(f32_lane.get(e).is_none());
        let u64_lane = store.read_lane::<u64>().unwrap();
        assert!(u64_lane.get(e).is_none());
        assert_eq!(f32_lane.len(), 0);
    }
}
