use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{RwLock, atomic::Ordering};

use crossbeam_epoch::{Atomic, Guard, Owned, pin as epoch_pin};

use crate::cold_store::ColdComponentStore;
use crate::entity::{Entity, EntityAllocator};
use crate::component_store::ComponentStore;

trait Lane: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn remove_entity(&self, entity: Entity);
}

impl<T: 'static + Send + Sync> Lane for RwLock<ComponentStore<T>> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn remove_entity(&self, entity: Entity) {
        if let Ok(mut guard) = self.write() {
            guard.remove(entity);
        }
    }
}

impl<T: 'static + Send + Sync> Lane for RwLock<ColdComponentStore<T>> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn remove_entity(&self, entity: Entity) {
        if let Ok(mut guard) = self.write() {
            guard.remove(entity);
        }
    }
}

struct LockFreeLaneInner<T: Clone + Send + Sync> {
    store: Atomic<ComponentStore<T>>,
}

impl<T: 'static + Clone + Send + Sync> LockFreeLaneInner<T> {
    fn new() -> Self {
        Self { store: Atomic::new(ComponentStore::new()) }
    }

    fn read<'g>(&'g self, guard: &'g Guard) -> &'g ComponentStore<T> {
        let shared = self.store.load(Ordering::Acquire, guard);
        unsafe { shared.deref() }
    }

    fn write(&self, f: impl FnOnce(&mut ComponentStore<T>)) {
        let guard = epoch_pin();
        let shared = self.store.load(Ordering::Acquire, &guard);
        let mut new_store = unsafe { (*shared.deref()).clone() };
        f(&mut new_store);
        self.store.store(Owned::new(new_store), Ordering::Release);
        unsafe { guard.defer_destroy(shared) };
    }
}

impl<T: 'static + Clone + Send + Sync> Lane for LockFreeLaneInner<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn remove_entity(&self, entity: Entity) {
        self.write(|store| { store.remove(entity); });
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

pub struct SmartStore {
    lanes: HashMap<TypeId, Box<dyn Lane>>,
    cold_lanes: HashMap<TypeId, Box<dyn Lane>>,
    allocator: RwLock<EntityAllocator>,
}

impl Default for SmartStore {
    fn default() -> Self {
        Self {
            lanes: HashMap::new(),
            cold_lanes: HashMap::new(),
            allocator: RwLock::new(EntityAllocator::new()),
        }
    }
}

impl SmartStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: 'static + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.lanes.entry(tid).or_insert_with(|| {
            Box::new(RwLock::new(ComponentStore::<T>::new()))
        });
    }

    pub fn register_lock_free<T: 'static + Clone + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.lanes.entry(tid).or_insert_with(|| {
            Box::new(LockFreeLaneInner::<T>::new())
        });
    }

    fn ensure_lane<T: 'static + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.lanes.entry(tid).or_insert_with(|| {
            Box::new(RwLock::new(ComponentStore::<T>::new()))
        });
    }

    fn ensure_lock_free_lane<T: 'static + Clone + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.lanes.entry(tid).or_insert_with(|| {
            Box::new(LockFreeLaneInner::<T>::new())
        });
    }

    pub fn register_cold<T: 'static + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.cold_lanes.entry(tid).or_insert_with(|| {
            Box::new(RwLock::new(ColdComponentStore::<T>::new()))
        });
    }

    fn ensure_cold_lane<T: 'static + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.cold_lanes.entry(tid).or_insert_with(|| {
            Box::new(RwLock::new(ColdComponentStore::<T>::new()))
        });
    }

    pub fn insert_cold<T: 'static + Send + Sync>(&mut self, entity: Entity, component: T) {
        self.ensure_cold_lane::<T>();
        let tid = TypeId::of::<T>();
        if let Some(lane) = self.cold_lanes.get(&tid) {
            if let Some(store) = lane.as_any()
                .downcast_ref::<RwLock<ColdComponentStore<T>>>()
            {
                store.write().unwrap().insert(entity, component);
            }
        }
    }

    pub fn read_cold_lane<T: 'static + Send + Sync>(&self) -> Option<std::sync::RwLockReadGuard<'_, ColdComponentStore<T>>> {
        let tid = TypeId::of::<T>();
        let lane = self.cold_lanes.get(&tid)?;
        Some(lane.as_any()
            .downcast_ref::<RwLock<ColdComponentStore<T>>>()
            .unwrap()
            .read()
            .unwrap())
    }

    pub fn write_cold_lane<T: 'static + Send + Sync>(&self) -> Option<std::sync::RwLockWriteGuard<'_, ColdComponentStore<T>>> {
        let tid = TypeId::of::<T>();
        let lane = self.cold_lanes.get(&tid)?;
        Some(lane.as_any()
            .downcast_ref::<RwLock<ColdComponentStore<T>>>()
            .unwrap()
            .write()
            .unwrap())
    }

    pub fn create_entity(&self) -> Entity {
        self.allocator.write().unwrap().allocate()
    }

    pub fn destroy_entity(&self, entity: Entity) {
        for (_, lane) in self.lanes.iter() {
            lane.remove_entity(entity);
        }
        for (_, lane) in self.cold_lanes.iter() {
            lane.remove_entity(entity);
        }
        self.allocator.write().unwrap().deallocate(entity);
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.allocator.read().unwrap().is_alive(entity)
    }

    pub fn insert<T: 'static + Clone + Send + Sync>(&mut self, entity: Entity, component: T) {
        self.ensure_lane::<T>();
        let tid = TypeId::of::<T>();
        let lane = self.lanes.get(&tid).unwrap();
        if let Some(rwlock) = lane.as_any().downcast_ref::<RwLock<ComponentStore<T>>>() {
            rwlock.write().unwrap().insert(entity, component);
        } else if let Some(lf) = lane.as_any().downcast_ref::<LockFreeLaneInner<T>>() {
            lf.write(|store| store.insert(entity, component));
        }
    }

    pub fn read_lane<T: 'static + Send + Sync>(&self) -> Option<std::sync::RwLockReadGuard<'_, ComponentStore<T>>> {
        let tid = TypeId::of::<T>();
        let lane = self.lanes.get(&tid)?;
        Some(lane.as_any()
            .downcast_ref::<RwLock<ComponentStore<T>>>()
            .unwrap()
            .read()
            .unwrap())
    }

    pub fn write_lane<T: 'static + Send + Sync>(&self) -> Option<std::sync::RwLockWriteGuard<'_, ComponentStore<T>>> {
        let tid = TypeId::of::<T>();
        let lane = self.lanes.get(&tid)?;
        Some(lane.as_any()
            .downcast_ref::<RwLock<ComponentStore<T>>>()
            .unwrap()
            .write()
            .unwrap())
    }

    pub fn with_lock_free_lane<T: 'static + Clone + Send + Sync, R>(&self, f: impl FnOnce(&ComponentStore<T>) -> R) -> Option<R> {
        let tid = TypeId::of::<T>();
        let lane = self.lanes.get(&tid)?;
        let guard = epoch_pin();
        let inner = lane.as_any().downcast_ref::<LockFreeLaneInner<T>>()?;
        let store_ref = inner.read(&guard);
        Some(f(store_ref))
    }

    pub fn write_lock_free_lane<T: 'static + Clone + Send + Sync>(&self, f: impl FnOnce(&mut ComponentStore<T>)) {
        let tid = TypeId::of::<T>();
        let lane = self.lanes.get(&tid).unwrap();
        if let Some(lf) = lane.as_any().downcast_ref::<LockFreeLaneInner<T>>() {
            lf.write(f);
        }
    }
}

pub trait Pack: Clone + Send + Sync + 'static {
    type PackMut<'a>
    where
        Self: 'a;
    
    fn pack_register(store: &mut SmartStore);
    fn pack_insert(&self, store: &mut SmartStore, entity: Entity);
    fn pack_get(store: &SmartStore, entity: Entity) -> Option<Self>;
    fn pack_get_mut<'a>(store: &'a mut SmartStore, entity: Entity) -> Option<Self::PackMut<'a>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_insert() {
        let mut store = SmartStore::new();

        let entity = store.create_entity();
        store.insert::<f32>(entity, 1.0);
        store.insert::<u32>(entity, 42);

        let pos_lane = store.read_lane::<f32>().unwrap();
        assert_eq!(pos_lane.get(entity), Some(&1.0));
        drop(pos_lane);

        let hp_lane = store.read_lane::<u32>().unwrap();
        assert_eq!(hp_lane.get(entity), Some(&42));
    }

    #[test]
    fn entity_lifecycle() {
        let mut store = SmartStore::new();

        let e = store.create_entity();
        assert!(store.is_alive(e));

        store.insert::<f32>(e, 10.0);
        store.destroy_entity(e);
        assert!(!store.is_alive(e));

        let lane = store.read_lane::<f32>().unwrap();
        assert!(lane.get(e).is_none());
    }

    #[test]
    fn lock_free_lane_basic() {
        let mut store = SmartStore::new();
        store.register_lock_free::<f32>();

        let e = store.create_entity();
        store.insert::<f32>(e, 3.14);

        let val = store.with_lock_free_lane::<f32, _>(|store| {
            store.get(e).copied()
        }).unwrap();
        assert_eq!(val, Some(3.14));
    }

    #[test]
    fn lock_free_write() {
        let mut store = SmartStore::new();
        store.register_lock_free::<f32>();

        let e = store.create_entity();
        store.insert::<f32>(e, 1.0);

        store.write_lock_free_lane::<f32>(|store| {
            store.insert(e, 2.0);
        });

        let val = store.with_lock_free_lane::<f32, _>(|store| {
            store.get(e).copied()
        }).unwrap();
        assert_eq!(val, Some(2.0));
    }

    #[test]
    fn lock_free_concurrent_read() {
        use std::sync::Arc;
        use std::thread;

        let mut store = SmartStore::new();
        store.register_lock_free::<f32>();

        let e = store.create_entity();
        store.insert::<f32>(e, 1.0);

        let store = Arc::new(store);
        let mut handles = vec![];

        for _ in 0..8 {
            let s = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    let val = s.with_lock_free_lane::<f32, _>(|store| {
                        store.get(e).copied()
                    }).unwrap();
                    assert_eq!(val, Some(1.0));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn benchmark_speed() {
        let mut store = SmartStore::new();
        let start = std::time::Instant::now();
        let count = 100_000;

        let mut entities = Vec::with_capacity(count);
        for _ in 0..count {
            entities.push(store.create_entity());
        }

        for &e in &entities {
            store.insert::<f32>(e, 1.0);
        }

        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 500, "took {} ms", elapsed.as_millis());
    }
}