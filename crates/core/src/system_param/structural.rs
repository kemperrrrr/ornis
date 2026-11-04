//! Deferred structural operations as a typed system parameter (R4, IDEAS §32).
//!
//! [`StructuralWriter`] is the `#[smart_system]` parameter for `add` /
//! `remove` / `spawn`: a body sees shared storage only, so those operations
//! enqueue onto [`StructuralQueue`] and are applied at the frame boundary
//! ([`apply_structural_queue`](crate::mutation::apply_structural_queue),
//! before [`Stage::Last`](crate::Stage::Last)). The parameter declares a
//! write to the queue, which serializes every queueing system across
//! parallel levels in registration order — the source of the queue's
//! deterministic FIFO (see [`StructuralQueue`] docs).

use crate::Entity;
use crate::mutation::StructuralQueue;
use crate::schedule::{Resources, SystemAccess};
use crate::smart_store::{Pack, SmartStore};

use super::SystemParam;

/// Deferred structural-ops parameter: queues `add`/`remove`/`spawn` for the
/// frame boundary instead of applying them inline.
///
/// Its type always declares `writes::<StructuralQueue>()`, independent of
/// which helpers are called in the function body — so all queueing systems
/// conflict and run in registration order across parallel levels, and the
/// FIFO apply reproduces that order bitwise for any thread count.
///
/// `Copy` (shared borrows only): the generated per-entity loop passes
/// parameters by value, so they must survive more than one iteration.
pub struct StructuralWriter<'a> {
    queue: &'a StructuralQueue,
    store: &'a SmartStore,
}

impl Clone for StructuralWriter<'_> {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for StructuralWriter<'_> {}

impl<'a> StructuralWriter<'a> {
    /// Fetches the queue and store singletons; panics when
    /// [`register_structural_queue`](crate::mutation::register_structural_queue)
    /// was not called (same missing-resource contract as
    /// [`Res`](super::Res)). The store is fetched here so [`spawn`](Self::spawn)
    /// can allocate the id up front and hand back a live [`Entity`].
    // qual:allow(complexity, error_handling) reason: "deliberate missing-resource contract — the queue must be registered before the schedule runs; mirrors the Res/EventWriter fetch panics"
    pub fn fetch(resources: &'a Resources) -> Self {
        let queue = resources.get::<StructuralQueue>().unwrap_or_else(|| {
            panic!(
                "smart_system: missing StructuralQueue resource — call \
                 register_structural_queue() (or run inside a World/Engine) \
                 before running the schedule"
            )
        });
        let store = resources
            .get::<SmartStore>()
            .unwrap_or_else(|| panic!("smart_system: missing SmartStore resource"));
        Self { queue, store }
    }

    /// Queues creation of an entity carrying `bundle` and returns its live
    /// handle immediately.
    ///
    /// The id is allocated now through the store's interior allocator (the
    /// only allocation route available under shared access), so handle
    /// ordering follows *call* order, not apply order.
    /// [`StructuralWriter`](Self) declares a queue write, which serializes
    /// every queueing system in registration order — so all writers (and
    /// every system on their levels) observe call order == id order. A
    /// system that allocates directly (`SmartStore::create_entity` under a
    /// plain read declaration) does not conflict with the queue and may
    /// interleave ids; declare a write on the store from such systems when
    /// id order matters.
    ///
    /// The bundle lands at apply time — see
    /// [`apply_structural_queue`](crate::mutation::apply_structural_queue).
    /// A handle to an entity despawned before the apply is skipped there,
    /// so a queued spawn can never resurrect a dead id.
    pub fn spawn<P: Pack>(&self, bundle: P) -> Entity {
        let entity = self.store.create_entity();
        self.queue.push(Box::new(move |store: &mut SmartStore| {
            if store.is_alive(entity) {
                bundle.pack_insert(store, entity);
            }
        }));
        entity
    }

    /// Queues `component` for `entity` (upsert; the hot lane for `T` is
    /// created on first use). Skipped at apply when the entity is no longer
    /// alive.
    pub fn insert<T: 'static + Clone + Send + Sync>(&self, entity: Entity, component: T) {
        self.queue.push(Box::new(move |store: &mut SmartStore| {
            if store.is_alive(entity) {
                store.insert(entity, component);
            }
        }));
    }

    /// Queues removal of component `T` from `entity` (no-op when the entity
    /// or the lane entry is already gone).
    ///
    /// # Panics
    /// At apply time, if `T` was registered as a lock-free lane:
    /// [`SmartStore::write_lane`](crate::SmartStore::write_lane) rejects
    /// those, and this deferred path cannot use the snapshot API (it needs
    /// `remove` under `&mut`). `#[smart_system]` has the same constraint
    /// for field writes, so packed/system components stay `RwLock` lanes.
    pub fn remove<T: 'static + Send + Sync>(&self, entity: Entity) {
        self.queue.push(Box::new(move |store: &mut SmartStore| {
            if store.is_alive(entity)
                && let Some(mut lane) = store.write_lane::<T>()
            {
                lane.remove(entity);
            }
        }));
    }

    /// Queues destruction of `entity`: components leave every hot and cold
    /// lane and the id returns to the allocator, in queue order.
    pub fn despawn(&self, entity: Entity) {
        self.queue.push(Box::new(move |store: &mut SmartStore| {
            store.destroy_entity(entity);
        }));
    }

    /// Declares exclusive access to the structural queue (the ordering key
    /// that serializes queueing systems in registration order).
    pub fn access() -> SystemAccess {
        SystemAccess::new().writes::<StructuralQueue>()
    }
}

impl SystemParam for StructuralWriter<'_> {
    type Item<'a> = StructuralWriter<'a>;

    fn access() -> SystemAccess {
        StructuralWriter::access()
    }

    fn fetch(resources: &Resources) -> Self::Item<'_> {
        StructuralWriter::fetch(resources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutation::register_structural_queue;
    use crate::schedule::Schedule;
    use std::any::TypeId;

    /// Minimal hand-written `Pack` bundle (the derive lives in
    /// `ornis-macros`, which core unit tests cannot use).
    #[derive(Clone)]
    struct One(u32);

    impl Pack for One {
        type PackMut<'a> = ();

        fn pack_register(store: &mut SmartStore) {
            store.register::<u32>();
        }

        fn pack_insert(&self, store: &mut SmartStore, entity: Entity) {
            store.insert(entity, self.0);
        }

        fn pack_get(store: &SmartStore, entity: Entity) -> Option<Self> {
            store
                .read_lane::<u32>()?
                .get(entity)
                .map(|value| One(*value))
        }

        fn pack_get_mut<'a>(
            _store: &'a mut SmartStore,
            _entity: Entity,
        ) -> Option<Self::PackMut<'a>> {
            None
        }

        fn pack_put(&self, store: &SmartStore, entity: Entity) {
            if let Some(mut lane) = store.write_lane::<u32>() {
                lane.insert(entity, self.0);
            }
        }

        fn pack_entities(store: &SmartStore) -> Vec<Entity> {
            store
                .read_lane::<u32>()
                .map(|lane| lane.entities.clone())
                .unwrap_or_default()
        }

        fn pack_lane_ids() -> Vec<TypeId> {
            vec![TypeId::of::<u32>()]
        }
    }

    fn resources_with_store() -> Resources {
        let mut resources = Resources::new();
        resources.insert(SmartStore::new());
        register_structural_queue(&mut resources);
        resources
            .get_mut::<SmartStore>()
            .expect("store")
            .register::<u32>();
        resources
    }

    #[test]
    fn structural_writer_declares_queue_write() {
        assert_eq!(
            StructuralWriter::access(),
            SystemAccess::new().writes::<StructuralQueue>()
        );
    }

    #[test]
    fn fetch_and_spawn_insert_despawn_queue_until_applied() {
        let mut resources = resources_with_store();
        let writer = StructuralWriter::fetch(&resources);
        assert!(
            resources
                .get::<StructuralQueue>()
                .expect("queue registered")
                .is_empty()
        );

        let spawned = writer.spawn(One(7));
        let untouched = resources
            .get::<SmartStore>()
            .expect("store")
            .create_entity();
        writer.insert(untouched, 9_u32);
        writer.remove::<u32>(spawned);
        writer.despawn(untouched);

        {
            let store = resources.get::<SmartStore>().expect("store");
            assert_eq!(store.read_lane::<u32>().map(|lane| lane.len()), Some(0));
        }

        crate::mutation::apply_structural_queue(&mut resources);

        let store = resources.get::<SmartStore>().expect("store");
        // spawn+remove of `spawned`: the lane entry was removed again but
        // the entity itself stays alive; `untouched` was despawned after
        // its insert, so its entry went with it.
        assert_eq!(store.read_lane::<u32>().map(|lane| lane.len()), Some(0));
        assert!(store.is_alive(spawned));
        assert!(!store.is_alive(untouched));
    }

    #[test]
    fn spawn_returns_live_handle_and_lands_bundle_at_apply() {
        let mut resources = resources_with_store();
        let entity = StructuralWriter::fetch(&resources).spawn(One(11));
        {
            let store = resources.get::<SmartStore>().expect("store");
            assert!(store.is_alive(entity));
            assert_eq!(store.read_lane::<u32>().map(|lane| lane.len()), Some(0));
        }

        crate::mutation::apply_structural_queue(&mut resources);

        let store = resources.get::<SmartStore>().expect("store");
        assert_eq!(store.read_lane::<u32>().map(|lane| lane.len()), Some(1));
        assert_eq!(
            store.read_lane::<u32>().expect("lane").get(entity).copied(),
            Some(11)
        );
    }

    #[test]
    #[should_panic(expected = "missing StructuralQueue")]
    fn fetch_panics_without_queue() {
        let resources = Resources::new();
        let _ = StructuralWriter::fetch(&resources);
    }

    #[test]
    fn systems_declaring_the_writer_conflict_across_levels() {
        struct Probe;
        impl crate::System for Probe {
            fn name(&self) -> &'static str {
                "probe"
            }
            fn access(&self) -> SystemAccess {
                StructuralWriter::access()
            }
            fn run(&self, resources: &Resources) {
                StructuralWriter::fetch(resources).despawn(Entity::new(0));
            }
        }
        let mut schedule = Schedule::new();
        schedule.add_system(Probe);
        schedule.add_system(Probe);
        assert_eq!(schedule.levels(), vec![vec![0], vec![1]]);
    }

    /// Pins the documented `remove` constraint: the deferred path goes
    /// through `write_lane`, which rejects lock-free lanes at apply time.
    #[test]
    #[should_panic(expected = "cannot access a lock-free lane")]
    fn remove_panics_on_lock_free_lane() {
        let mut resources = Resources::new();
        resources.insert(SmartStore::new());
        register_structural_queue(&mut resources);
        resources
            .get_mut::<SmartStore>()
            .expect("store")
            .register_lock_free::<u32>();
        let entity = resources
            .get::<SmartStore>()
            .expect("store")
            .create_entity();

        StructuralWriter::fetch(&resources).remove::<u32>(entity);
        crate::mutation::apply_structural_queue(&mut resources);
    }
}
