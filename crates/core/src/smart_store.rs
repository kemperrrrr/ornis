use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::RwLock;

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

pub struct SmartStore {
    lanes: HashMap<TypeId, Box<dyn Lane>>,
    hot_lanes: HashMap<TypeId, Box<dyn Lane>>,
    cold_lanes: HashMap<TypeId, Box<dyn Lane>>,
    allocator: RwLock<EntityAllocator>,
}

impl Default for SmartStore {
    fn default() -> Self {
        Self {
            lanes: HashMap::new(),
            hot_lanes: HashMap::new(),
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

    fn ensure_lane<T: 'static + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.lanes.entry(tid).or_insert_with(|| {
            Box::new(RwLock::new(ComponentStore::<T>::new()))
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
        for (_, lane) in self.hot_lanes.iter() {
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

    pub fn insert<T: 'static + Send + Sync>(&mut self, entity: Entity, component: T) {
        self.ensure_lane::<T>();
        let tid = TypeId::of::<T>();
        let lane = self.lanes.get(&tid).unwrap();
        lane.as_any()
            .downcast_ref::<RwLock<ComponentStore<T>>>()
            .unwrap()
            .write()
            .unwrap()
            .insert(entity, component);
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
