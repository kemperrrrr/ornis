//! Concurrent component storage via epoch-based copy-on-write lanes.
//!
//! [`LockFreeStore`] holds one lane per component type. Reads load the
//! current [`ComponentStore`] snapshot under a crossbeam-epoch guard without
//! locking; writes clone-modify-swap the snapshot and defer destruction of
//! the old one to the epoch reclaimer. Entity allocation stays under a
//! mutex. Invariant: a published snapshot is never mutated in place.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard};

use crossbeam_epoch::{Atomic, Guard, Owned};

use crate::component_store::ComponentStore;
use crate::entity::{Entity, EntityAllocator};

/// Mutex guard that recovers from poison instead of panicking.
fn mutex_lock<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|e| e.into_inner())
}

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

    fn write(&self, mut f: impl FnMut(&mut ComponentStore<T>)) {
        // CAS publish loop: `load -> clone -> store` without a CAS lets two
        // writers read the same snapshot and both `defer_destroy` it (double
        // free, UB). Only the thread whose `compare_exchange` succeeds owns
        // the replaced snapshot and may retire it; losers drop their private
        // clone inline and retry on the fresh snapshot. Success ordering is
        // AcqRel (acquire the latest snapshot, release the publish);
        // failure ordering is Acquire (reload the current pointer).
        loop {
            let guard = crossbeam_epoch::pin();
            let shared = self.store.load(Ordering::Acquire, &guard);
            // Safety: `shared` is pinned by `guard` for this iteration.
            let mut new_store = unsafe { (*shared.deref()).clone() };
            f(&mut new_store);
            match self.store.compare_exchange(
                shared,
                Owned::new(new_store),
                Ordering::AcqRel,
                Ordering::Acquire,
                &guard,
            ) {
                Ok(_) => {
                    // Safety: this thread replaced `shared`; no other thread
                    // can retire it because its CAS on the same `shared`
                    // pointer now fails. Epoch reclamation frees it once
                    // readers drain.
                    unsafe { guard.defer_destroy(shared) };
                    break;
                }
                Err(_) => {
                    // CAS lost: our `Owned` clone is dropped with the error
                    // and we retry against the winning snapshot.
                    continue;
                }
            }
        }
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
    allocator: Mutex<EntityAllocator>,
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
        mutex_lock(&self.allocator).allocate()
    }

    pub fn destroy_entity(&self, entity: Entity) {
        for (_, lane) in self.lanes.iter() {
            lane.remove_entity(entity);
        }
        mutex_lock(&self.allocator).deallocate(entity);
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        mutex_lock(&self.allocator).is_alive(entity)
    }

    pub fn insert<T: 'static + Clone + Send + Sync>(&mut self, entity: Entity, component: T) {
        self.ensure_lane::<T>();
        let tid = TypeId::of::<T>();
        if let Some(inner) = self
            .lanes
            .get(&tid)
            .and_then(|lane| lane.as_any().downcast_ref::<LaneInner<T>>())
        {
            // `component` is cloned per CAS attempt: the write loop may
            // re-apply the closure after contention, so it must be `FnMut`.
            inner.write(|store| {
                store.insert(entity, component.clone());
            });
        }
    }

    pub fn read_lane<T: 'static + Clone + Send + Sync>(&self) -> Option<LockFreeReadGuard<'_, T>> {
        let tid = TypeId::of::<T>();
        let guard = crossbeam_epoch::pin();
        let lane = self.lanes.get(&tid)?;
        let inner = lane.as_any().downcast_ref::<LaneInner<T>>()?;
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

    /// Stress: concurrent writers racing on one lane must not lose updates
    /// or retire one snapshot twice. Under the old `load -> clone -> store`
    /// code two writers could publish over each other (lost inserts) and
    /// both `defer_destroy` the same snapshot (double free, UB).
    #[test]
    fn concurrent_writes_no_lost_updates_or_double_free() {
        use std::sync::Arc;
        use std::thread;

        const THREADS: usize = 8;
        const PER_THREAD: usize = 250;
        // Ids below this are hammered by removals; only higher ids are
        // asserted so the test is deterministic under any interleaving.
        const SHARED_REMOVE_IDS: u32 = 16;

        let lane = Arc::new(LaneInner::<u64>::new());
        let mut handles = Vec::new();
        for t in 0..THREADS {
            let lane = Arc::clone(&lane);
            handles.push(thread::spawn(move || {
                for i in 0..PER_THREAD {
                    let id = (t * PER_THREAD + i) as u32;
                    let e = Entity::new(id);
                    lane.write(|store| {
                        store.insert(e, u64::from(id));
                    });
                }
                // Every thread removes the same shared ids, maximizing the
                // chance that two writers load one snapshot at once.
                for id in 0..SHARED_REMOVE_IDS {
                    lane.write(|store| {
                        store.remove(Entity::new(id));
                    });
                }
            }));
        }
        // A concurrent reader pins old snapshots while writers retire them,
        // exercising epoch reclamation under contention.
        let reader_lane = Arc::clone(&lane);
        let reader = thread::spawn(move || {
            for _ in 0..2000 {
                let guard = crossbeam_epoch::pin();
                let _ = reader_lane.read(&guard).len();
            }
        });

        for h in handles {
            h.join().unwrap();
        }
        reader.join().unwrap();

        let guard = crossbeam_epoch::pin();
        let snapshot = lane.read(&guard);
        for t in 0..THREADS {
            for i in 0..PER_THREAD {
                let id = (t * PER_THREAD + i) as u32;
                if id >= SHARED_REMOVE_IDS {
                    assert_eq!(
                        snapshot.get(Entity::new(id)),
                        Some(&u64::from(id)),
                        "lost update for entity {id}"
                    );
                }
            }
        }
    }

    /// Stress through the public API: overlapping `destroy_entity` calls
    /// from several threads retire snapshots concurrently; readers observe
    /// a consistent (eventually empty) lane.
    #[test]
    fn concurrent_destroy_no_double_free() {
        use std::sync::Arc;
        use std::thread;

        let mut store = LockFreeStore::new();
        let entities: Vec<Entity> = (0..256).map(|_| store.create_entity()).collect();
        for &e in &entities {
            store.insert::<u64>(e, 1);
        }
        let store = Arc::new(store);

        let mut handles = Vec::new();
        for _ in 0..4 {
            let s = Arc::clone(&store);
            let all = entities.clone();
            handles.push(thread::spawn(move || {
                // Every thread destroys every entity: maximum snapshot
                // contention on the same lane.
                for &e in &all {
                    s.destroy_entity(e);
                }
            }));
        }
        let s = Arc::clone(&store);
        let reader = thread::spawn(move || {
            for _ in 0..1000 {
                let _ = s.read_lane::<u64>().map(|g| g.len());
            }
        });

        for h in handles {
            h.join().unwrap();
        }
        reader.join().unwrap();

        let guard = store.read_lane::<u64>().unwrap();
        assert_eq!(guard.len(), 0);
    }
}
