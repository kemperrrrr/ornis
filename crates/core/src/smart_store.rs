//! The smart store: type-erased component lanes plus entity lifecycle.
//!
//! [`SmartStore`] is the engine's central component database. Each
//! component type lives in its own lane - an `RwLock`-guarded
//! [`ComponentStore`] by default, or an epoch-reclaimed lock-free clone-on-
//! write store for the experimental `lock-free` feature. Cold (rarely
//! accessed) lanes are kept in a separate map so hot data stays compact.
//! The [`Pack`] trait extends the store with multi-component packed
//! access used by GPU upload paths.
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{RwLock, atomic::Ordering};

use crossbeam_epoch::{Atomic, Guard, Owned, pin as epoch_pin};

use crate::cold_store::ColdComponentStore;
use crate::component_store::ComponentStore;
use crate::entity::{Entity, EntityAllocator};

/// Internal per-lane interface: erase storage details behind `SmartStore`.
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
        Self {
            store: Atomic::new(ComponentStore::new()),
        }
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
        self.write(|store| {
            store.remove(entity);
        });
    }
}

// Reserved: RAII read guard for lock-free lanes (experimental
// "lock-free" feature). Holds an epoch guard alive while exposing the
// snapshot of the store captured at read time.
#[allow(dead_code)]
pub struct LockFreeReadGuard<'g, T> {
    store: &'g ComponentStore<T>,
    _guard: Guard,
}

// Reserved: a lock-free read guard mirrored from `lock_free_store`.
// Not wired into `SmartStore` yet (read_lane returns an RwLock guard);
// kept for the experimental "lock-free" feature. Mutants are skipped:
// nothing calls `deref` today, so a mutation here is untestable dead code.
#[mutants::skip]
impl<'g, T> std::ops::Deref for LockFreeReadGuard<'g, T> {
    type Target = ComponentStore<T>;

    fn deref(&self) -> &Self::Target {
        self.store
    }
}

/// Central ECS storage: one lane per component type plus entity
/// allocation.
///
/// Hot lanes hold frequently touched components behind per-type `RwLock`s
/// (readers of different lanes run fully in parallel; rayon systems can
/// share `&SmartStore`). Lock-free lanes trade write cost (full snapshot
/// + swap) for wait-free reads via crossbeam epochs.
///
/// Cold lanes isolate rarely used components from the hot working set.
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
    /// Creates an empty store with no lanes and a fresh entity allocator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Eagerly creates the hot lane for component type `T` (no-op if it
    /// already exists). Registration up front avoids surprise allocations
    /// inside systems.
    pub fn register<T: 'static + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.lanes
            .entry(tid)
            .or_insert_with(|| Box::new(RwLock::new(ComponentStore::<T>::new())));
    }

    /// Registers `T` as a lock-free (clone-on-write, epoch-reclaimed)
    /// hot lane instead of the default `RwLock` lane. Requires `T: Clone`
    /// because every write publishes a fresh snapshot.
    pub fn register_lock_free<T: 'static + Clone + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.lanes
            .entry(tid)
            .or_insert_with(|| Box::new(LockFreeLaneInner::<T>::new()));
    }

    fn ensure_lane<T: 'static + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.lanes
            .entry(tid)
            .or_insert_with(|| Box::new(RwLock::new(ComponentStore::<T>::new())));
    }

    // reserved: lock-free lane registration (experimental "lock-free" feature)
    // Private and never called — `register_lock_free` inlines the same body.
    #[allow(dead_code)]
    #[mutants::skip]
    fn ensure_lock_free_lane<T: 'static + Clone + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.lanes
            .entry(tid)
            .or_insert_with(|| Box::new(LockFreeLaneInner::<T>::new()));
    }

    /// Eagerly creates the cold lane for `T`. No-op if already present.
    pub fn register_cold<T: 'static + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.cold_lanes
            .entry(tid)
            .or_insert_with(|| Box::new(RwLock::new(ColdComponentStore::<T>::new())));
    }

    fn ensure_cold_lane<T: 'static + Send + Sync>(&mut self) {
        let tid = TypeId::of::<T>();
        self.cold_lanes
            .entry(tid)
            .or_insert_with(|| Box::new(RwLock::new(ColdComponentStore::<T>::new())));
    }

    /// Inserts a cold component for `entity`, creating the cold lane on
    /// first use.
    pub fn insert_cold<T: 'static + Send + Sync>(&mut self, entity: Entity, component: T) {
        self.ensure_cold_lane::<T>();
        let tid = TypeId::of::<T>();
        if let Some(lane) = self.cold_lanes.get(&tid)
            && let Some(store) = lane
                .as_any()
                .downcast_ref::<RwLock<ColdComponentStore<T>>>()
        {
            store.write().unwrap().insert(entity, component);
        }
    }

    /// Shared read guard over the cold lane of `T`; `None` if the lane
    /// was never registered/populated.
    pub fn read_cold_lane<T: 'static + Send + Sync>(
        &self,
    ) -> Option<std::sync::RwLockReadGuard<'_, ColdComponentStore<T>>> {
        let tid = TypeId::of::<T>();
        let lane = self.cold_lanes.get(&tid)?;
        Some(
            lane.as_any()
                .downcast_ref::<RwLock<ColdComponentStore<T>>>()
                .unwrap()
                .read()
                .unwrap(),
        )
    }

    /// Exclusive write guard over the cold lane of `T`; `None` if the
    /// lane was never registered/populated.
    pub fn write_cold_lane<T: 'static + Send + Sync>(
        &self,
    ) -> Option<std::sync::RwLockWriteGuard<'_, ColdComponentStore<T>>> {
        let tid = TypeId::of::<T>();
        let lane = self.cold_lanes.get(&tid)?;
        Some(
            lane.as_any()
                .downcast_ref::<RwLock<ColdComponentStore<T>>>()
                .unwrap()
                .write()
                .unwrap(),
        )
    }

    /// Allocates a new live entity handle (recycling freed ids with a
    /// bumped generation).
    pub fn create_entity(&self) -> Entity {
        self.allocator.write().unwrap().allocate()
    }

    /// Destroys `entity`: removes its components from every hot and cold
    /// lane (including lock-free ones), then returns the id to the
    /// allocator for reuse.
    pub fn destroy_entity(&self, entity: Entity) {
        for lane in self.lanes.values() {
            lane.remove_entity(entity);
        }
        for lane in self.cold_lanes.values() {
            lane.remove_entity(entity);
        }
        self.allocator.write().unwrap().deallocate(entity);
    }

    /// Returns `true` if the handle matches the allocator's current
    /// generation - i.e. the entity was created and not yet destroyed.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.allocator.read().unwrap().is_alive(entity)
    }

    /// Inserts or replaces the hot component `T` for `entity`, creating
    /// the lane on first use. Dispatches to the RwLock or lock-free
    /// implementation depending on how `T` was registered.
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

    /// Reads the hot lane of component `T` as a shared guard.
    ///
    /// # Panics
    /// When schedule enforcement is enabled: if this lane was not declared
    /// as read ([`SystemAccess::reads_lane`](crate::SystemAccess)) or
    /// written (`writes_lane`) by the running system.
    pub fn read_lane<T: 'static + Send + Sync>(
        &self,
    ) -> Option<std::sync::RwLockReadGuard<'_, ComponentStore<T>>> {
        crate::schedule::assert_lane_access_declared::<T>(false);
        let tid = TypeId::of::<T>();
        let lane = self.lanes.get(&tid)?;
        Some(
            lane.as_any()
                .downcast_ref::<RwLock<ComponentStore<T>>>()
                .unwrap()
                .read()
                .unwrap(),
        )
    }

    /// Writes to the hot lane of component `T` behind an exclusive guard.
    ///
    /// # Panics
    /// When schedule enforcement is enabled: if this lane was not declared
    /// strictly for writing
    /// ([`SystemAccess::writes_lane`](crate::SystemAccess)); a read-only
    /// declaration does not cover writes.
    pub fn write_lane<T: 'static + Send + Sync>(
        &self,
    ) -> Option<std::sync::RwLockWriteGuard<'_, ComponentStore<T>>> {
        crate::schedule::assert_lane_access_declared::<T>(true);
        let tid = TypeId::of::<T>();
        let lane = self.lanes.get(&tid)?;
        Some(
            lane.as_any()
                .downcast_ref::<RwLock<ComponentStore<T>>>()
                .unwrap()
                .write()
                .unwrap(),
        )
    }

    /// Runs `f` against an epoch-pinned snapshot of the lock-free lane of
    /// `T`. Readers never block writers (and vice versa); the snapshot is
    /// consistent for the duration of the call. Returns `None` if `T` has
    /// no lock-free lane.
    pub fn with_lock_free_lane<T: 'static + Clone + Send + Sync, R>(
        &self,
        f: impl FnOnce(&ComponentStore<T>) -> R,
    ) -> Option<R> {
        let tid = TypeId::of::<T>();
        let lane = self.lanes.get(&tid)?;
        let guard = epoch_pin();
        let inner = lane.as_any().downcast_ref::<LockFreeLaneInner<T>>()?;
        let store_ref = inner.read(&guard);
        Some(f(store_ref))
    }

    /// Applies a mutation to the lock-free lane of `T`: clones the current
    /// snapshot, runs the mutator on the clone, then atomically publishes
    /// it. Old snapshots are reclaimed by the epoch garbage collector once
    /// readers drain. No-op if `T` has no lock-free lane.
    pub fn write_lock_free_lane<T: 'static + Clone + Send + Sync>(
        &self,
        f: impl FnOnce(&mut ComponentStore<T>),
    ) {
        let tid = TypeId::of::<T>();
        let lane = self.lanes.get(&tid).unwrap();
        if let Some(lf) = lane.as_any().downcast_ref::<LockFreeLaneInner<T>>() {
            lf.write(f);
        }
    }
}

/// A bundle of components stored contiguously ("packed") for GPU upload
/// and bulk traversal.
///
/// Implementations are usually generated by
/// [`#[derive(Pack)]`](ornis_macros::derive_pack) over a plain struct; the
/// derive registers one lane per field and implements gather/scatter of
/// whole bundles per entity. This is the Rust-side half of the engine's
/// Component Packing scheme (see PLAN.md): systems mutate packed bundles,
/// which are then uploaded as a single buffer rather than field by field.
pub trait Pack: Clone + Send + Sync + 'static {
    /// Mutable handle produced by [`Pack::pack_get_mut`]; writes go back
    /// into the store when dropped or flushed.
    type PackMut<'a>
    where
        Self: 'a;

    /// Creates/registers all lanes this bundle requires on `store`.
    fn pack_register(store: &mut SmartStore);

    /// Gathers the bundle from existing lanes for `entity` and inserts it
    /// into its own packed lane.
    fn pack_insert(&self, store: &mut SmartStore, entity: Entity);

    /// Reconstructs the bundle for `entity`, or `None` if it lacks any of
    /// the components.
    fn pack_get(store: &SmartStore, entity: Entity) -> Option<Self>;

    /// Returns a mutable view of the bundle for `entity` allowing in-place
    /// edits of the underlying component data.
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
        store.insert::<f32>(e, 3.5);

        let val = store
            .with_lock_free_lane::<f32, _>(|store| store.get(e).copied())
            .unwrap();
        assert_eq!(val, Some(3.5));
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

        let val = store
            .with_lock_free_lane::<f32, _>(|store| store.get(e).copied())
            .unwrap();
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
                    let val = s
                        .with_lock_free_lane::<f32, _>(|store| store.get(e).copied())
                        .unwrap();
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
        // Benchmark test - threshold generous to account for CI variance
        assert!(
            elapsed.as_millis() < 2000,
            "took {} ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn cold_lane_roundtrip() {
        let mut store = SmartStore::new();
        let e = store.create_entity();

        store.insert_cold::<f32>(e, 3.25);

        let lane = store.read_cold_lane::<f32>().unwrap();
        assert_eq!(lane.get(e), Some(&3.25));
        assert_eq!(lane.len(), 1);
    }

    #[test]
    fn cold_lane_write_guard_updates() {
        let mut store = SmartStore::new();
        let e = store.create_entity();
        store.insert_cold::<f32>(e, 1.0);

        {
            let mut lane = store.write_cold_lane::<f32>().unwrap();
            lane.insert(e, 9.0);
        }

        let lane = store.read_cold_lane::<f32>().unwrap();
        assert_eq!(lane.get(e), Some(&9.0));
    }

    #[test]
    fn destroy_entity_removes_from_cold_lanes() {
        let mut store = SmartStore::new();
        let e = store.create_entity();
        store.insert_cold::<f32>(e, 1.0);
        store.insert_cold::<u32>(e, 2);

        store.destroy_entity(e);

        let f32_lane = store.read_cold_lane::<f32>().unwrap();
        assert!(f32_lane.get(e).is_none());
        assert_eq!(f32_lane.len(), 0);
        let u32_lane = store.read_cold_lane::<u32>().unwrap();
        assert!(u32_lane.get(e).is_none());
    }

    #[test]
    fn lock_free_remove_entity_clears_component() {
        let mut store = SmartStore::new();
        store.register_lock_free::<f32>();
        let e = store.create_entity();
        store.insert::<f32>(e, 1.0);

        store.destroy_entity(e);

        let val = store
            .with_lock_free_lane::<f32, _>(|store| store.get(e).copied())
            .unwrap();
        assert_eq!(val, None);
        // The lane must be empty, not just the entity filtered out.
        let len = store
            .with_lock_free_lane::<f32, _>(|store| store.len())
            .unwrap();
        assert_eq!(len, 0);
    }

    #[test]
    fn register_creates_hot_lane() {
        let mut store = SmartStore::new();
        store.register::<f32>();
        // `register` must create the lane eagerly; `read_lane` then
        // succeeds even before any insert.
        let lane = store.read_lane::<f32>().unwrap();
        assert_eq!(lane.len(), 0);
        drop(lane); // release the read guard before taking the write guard
        assert!(store.write_lane::<f32>().is_some());
    }

    #[test]
    fn register_cold_creates_cold_lane() {
        let mut store = SmartStore::new();
        store.register_cold::<f32>();
        let lane = store.read_cold_lane::<f32>().unwrap();
        assert_eq!(lane.len(), 0);
        drop(lane);
        assert!(store.write_cold_lane::<f32>().is_some());
    }
}
