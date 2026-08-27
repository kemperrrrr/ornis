//! The logical engine world: shared ECS storage and singleton resources.
//!
//! [`World`] is the first integration layer above [`SmartStore`] and
//! [`Resources`]. It gives systems, physics, rendering and tooling one
//! authoritative container without imposing an archetype layout: components
//! remain in Ornis' independent sparse-set lanes. Domain runtimes should be
//! registered as resources and consumed through the common [`Schedule`]
//! contract.

use std::any::Any;

use crate::schedule::{Resources, Schedule};
use crate::smart_store::SmartStore;

/// The authoritative logical state container for one engine instance.
///
/// The world stores the ECS [`SmartStore`] as the conventional `SmartStore`
/// resource and keeps all other singleton state in the same resource map.
/// This is a logical unification boundary: CPU and GPU representations may
/// still be separate and are coordinated by their respective runtime
/// resources and residency trackers.
pub struct World {
    resources: Resources,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// Creates a world with an empty authoritative [`SmartStore`].
    pub fn new() -> Self {
        let mut resources = Resources::new();
        resources.insert(SmartStore::new());
        Self { resources }
    }

    /// Returns the world's singleton resource container.
    ///
    /// Systems normally receive this value through [`Schedule::run`]. This
    /// accessor is intended for setup, inspection and platform integration.
    pub fn resources(&self) -> &Resources {
        &self.resources
    }

    /// Returns the world's mutable singleton resource container.
    ///
    /// Use this during setup or between schedule runs. Mutating resources
    /// while systems are executing would bypass the scheduler's access
    /// contract.
    pub fn resources_mut(&mut self) -> &mut Resources {
        &mut self.resources
    }

    /// Returns the authoritative sparse-set ECS store, if it is registered.
    ///
    /// A caller can remove or replace the conventional `SmartStore` through
    /// [`World::resources_mut`], so this method reports absence instead of
    /// hiding that state behind a panic.
    pub fn store(&self) -> Option<&SmartStore> {
        self.resources.get::<SmartStore>()
    }

    /// Returns the mutable authoritative sparse-set ECS store, if registered.
    ///
    /// Use this for setup and command application between schedule runs.
    pub fn store_mut(&mut self) -> Option<&mut SmartStore> {
        self.resources.get_mut::<SmartStore>()
    }

    /// Inserts or replaces a singleton domain resource.
    ///
    /// Physics, rendering, asset and platform runtimes should be exposed to
    /// systems this way rather than through separate world containers.
    pub fn insert<R: Any + Send + Sync>(&mut self, resource: R) -> Option<R> {
        self.resources.insert(resource)
    }

    /// Removes a singleton domain resource between schedule runs.
    pub fn remove<R: Any + Send + Sync>(&mut self) -> Option<R> {
        self.resources.remove::<R>()
    }

    /// Runs the common system schedule against this world.
    ///
    /// The schedule receives the same resource map that setup and domain
    /// runtimes use. Components remain sparse-set lanes inside the
    /// `SmartStore` resource; no archetype migration is introduced here.
    pub fn run(&self, schedule: &Schedule) {
        schedule.run(&self.resources);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Entity, System, SystemAccess};
    use std::sync::{Arc, Mutex};

    struct ReadStore;

    impl System for ReadStore {
        fn name(&self) -> &'static str {
            "read_store"
        }

        fn access(&self) -> SystemAccess {
            SystemAccess::new()
                .reads::<SmartStore>()
                .reads_lane::<u32>()
        }

        fn run(&self, resources: &Resources) {
            let store = resources.get::<SmartStore>().expect("world store");
            assert_eq!(store.read_lane::<u32>().map(|lane| lane.len()), Some(1));
        }
    }

    struct ObserveCount(Arc<Mutex<usize>>);

    impl System for ObserveCount {
        fn name(&self) -> &'static str {
            "observe_count"
        }

        fn access(&self) -> SystemAccess {
            SystemAccess::new()
                .reads::<SmartStore>()
                .reads_lane::<u32>()
        }

        fn run(&self, resources: &Resources) {
            let store = resources.get::<SmartStore>().expect("world store");
            *self.0.lock().expect("count lock") = store
                .read_lane::<u32>()
                .map(|lane| lane.len())
                .unwrap_or_default();
        }
    }

    #[test]
    fn world_owns_sparse_set_store() {
        let mut world = World::new();
        let entity: Entity = world.store_mut().expect("store").create_entity();
        world.store_mut().expect("store").insert(entity, 7_u32);

        assert_eq!(
            world
                .store()
                .expect("store")
                .read_lane::<u32>()
                .map(|lane| lane.len()),
            Some(1)
        );
    }

    #[test]
    fn schedule_runs_against_world_resources() {
        let mut world = World::new();
        let entity = world.store_mut().expect("store").create_entity();
        world.store_mut().expect("store").insert(entity, 7_u32);

        let mut schedule = Schedule::new();
        let seen = Arc::new(Mutex::new(0));
        schedule.add_system(ObserveCount(seen.clone()));
        world.run(&schedule);

        assert_eq!(*seen.lock().expect("count lock"), 1);
    }

    #[test]
    fn domain_resources_share_world_with_ecs() {
        let mut world = World::new();
        world.insert(42_u32);
        assert_eq!(world.resources().get::<u32>(), Some(&42));
        assert!(world.store().is_some());
    }

    #[test]
    fn replacing_conventional_store_is_explicit() {
        let mut world = World::new();
        let previous = world.insert(SmartStore::new());
        assert!(previous.is_some());
        assert!(world.store().is_some());
        assert!(world.remove::<SmartStore>().is_some());
        assert!(world.store().is_none());
    }

    #[test]
    fn schedule_can_read_store_resource_directly() {
        let mut world = World::new();
        let entity = world.store_mut().expect("store").create_entity();
        world.store_mut().expect("store").insert(entity, 7_u32);

        let mut schedule = Schedule::new();
        schedule.add_system(ReadStore);
        world.run(&schedule);
    }
}
