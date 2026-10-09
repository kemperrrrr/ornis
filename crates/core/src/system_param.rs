//! Typed system parameters for [`#[smart_system]`](crate::smart_system)
//! (IDEAS §32, RAT-31).
//!
//! Access follows from **parameter types** (Bevy-style `SystemParam`),
//! never from syntactic body analysis: [`Res`] declares a shared read,
//! [`ResMut`] an exclusive-scheduler write (mutation itself goes through
//! the resource's interior mutability), [`Events`] declares event-queue
//! access. [`PlainResource`] keeps interior mutability out of [`Res`]:
//! a `Mutex`/`RwLock`/`Atomic*` behind a shared read would be
//! memory-safe but non-deterministic (see the commutativity contract in
//! [`crate::schedule`]), so it must be declared as [`ResMut`] instead.
//!
//! [`SmartStore`] is deliberately *not* a parameter: systems generated
//! by `#[smart_system]` reach the store through the first (`Pack`)
//! parameter, and a raw `&Resources` parameter would bypass declared
//! accesses entirely — both are compile errors in the macro.

use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Mutex;
use std::time::Duration;

use crate::schedule::{Resources, SystemAccess};
use crate::{Entity, FixedTime, SmartStore, Time};

/// Marker for resources without interior mutability.
///
/// A `PlainResource` can be shared between systems on the same parallel
/// level through [`Res`]: every byte reachable from it is either
/// immutable or mutated only between scheduler runs. Types with interior
/// mutability (`Mutex`, `RwLock`, `Cell`/`RefCell`, atomics, and anything
/// containing them) must **not** implement this trait — they go through
/// [`ResMut`], which declares a scheduler-visible write.
///
/// Custom resources get the bound structurally via
/// `#[derive(PlainResource)]` (from `ornis-macros`): the derive adds a
/// `Field: PlainResource` bound per field, so a struct holding a `Mutex`
/// fails with the missing-bound error below instead of silently sharing
/// mutable state. Primitives, plain std containers of plain data, and
/// `glam` vectors implement it out of the box.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be shared as `Res`: it may hold interior mutability; \
               use `ResMut` (declares a write) or `#[derive(PlainResource)]` on a plain-data struct"
)]
pub trait PlainResource: Send + Sync + 'static {}

macro_rules! plain_impl {
    ($($ty:ty),*) => {
        $(impl PlainResource for $ty {})*
    };
}

plain_impl!(
    bool, char, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, String,
    Duration
);
plain_impl!(glam::Vec2, glam::Vec3, glam::Vec4, glam::Quat);
plain_impl!(
    Time,
    FixedTime,
    crate::Seconds,
    crate::FixedSteps,
    crate::Position,
    crate::Meters
);

impl<T: PlainResource> PlainResource for Option<T> {}
impl<T: PlainResource> PlainResource for Vec<T> {}
impl<T: PlainResource> PlainResource for Box<T> {}
impl<T: PlainResource, E: PlainResource> PlainResource for Result<T, E> {}
impl<T: PlainResource> PlainResource for [T; 0] {}
impl<T: PlainResource> PlainResource for [T; 1] {}
impl<T: PlainResource> PlainResource for [T; 2] {}
impl<T: PlainResource> PlainResource for [T; 3] {}
impl<T: PlainResource> PlainResource for [T; 4] {}
impl PlainResource for () {}
impl<A: PlainResource> PlainResource for (A,) {}
impl<A: PlainResource, B: PlainResource> PlainResource for (A, B) {}
impl<A: PlainResource, B: PlainResource, C: PlainResource> PlainResource for (A, B, C) {}
impl<K: PlainResource, V: PlainResource> PlainResource for HashMap<K, V> {}

/// Shared resource parameter: declares `reads::<T>`.
///
/// `T` must be [`PlainResource`] (no interior mutability), so sharing it
/// across a parallel level is deterministic. For interior-mutable state
/// use [`ResMut`].
pub struct Res<'a, T: PlainResource> {
    value: &'a T,
}

impl<'a, T: PlainResource> Res<'a, T> {
    /// Fetches the singleton; panics with the type name when absent
    /// (same contract as Bevy's `Res`: a declared resource must exist).
    pub fn fetch(resources: &'a Resources) -> Self {
        let value = resources.get::<T>().unwrap_or_else(|| {
            panic!(
                "smart_system: missing resource '{}'",
                std::any::type_name::<T>()
            )
        });
        Self { value }
    }

    /// Contributes `reads::<T>` to the system access set.
    pub fn access() -> SystemAccess {
        SystemAccess::new().reads::<T>()
    }
}

impl<T: PlainResource> Deref for Res<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}

/// Exclusive (scheduler-visible) resource parameter: declares
/// `writes::<T>`.
///
/// Mutation itself goes through the resource's interior mutability
/// (`Mutex`, atomics — the `PhysicsRuntime` pattern); the write
/// declaration keeps conflicting systems on separate levels in
/// registration order, so the mutation order stays deterministic.
/// Unlike [`Res`], `T` is *not* required to be [`PlainResource`].
pub struct ResMut<'a, T: Send + Sync + 'static> {
    value: &'a T,
}

impl<'a, T: Send + Sync + 'static> ResMut<'a, T> {
    /// Fetches the singleton; panics with the type name when absent.
    pub fn fetch(resources: &'a Resources) -> Self {
        let value = resources.get::<T>().unwrap_or_else(|| {
            panic!(
                "smart_system: missing resource '{}'",
                std::any::type_name::<T>()
            )
        });
        Self { value }
    }

    /// Contributes `writes::<T>` to the system access set.
    pub fn access() -> SystemAccess {
        SystemAccess::new().writes::<T>()
    }
}

impl<T: Send + Sync + 'static> Deref for ResMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}

/// Deterministic event queue resource: `(entity, event)` pairs sent by
/// one system and applied by another.
///
/// Sends from any thread/level append in arrival order; draining sorts by
/// entity id, so application order is deterministic regardless of which
/// parallel level produced the events (Strong Confluence, IDEAS §24.1).
/// A system that only sends still declares a write (see [`Events`]).
pub struct EventStore<E: Send + 'static> {
    queue: Mutex<Vec<(Entity, E)>>,
}

impl<E: Send + 'static> Default for EventStore<E> {
    fn default() -> Self {
        Self {
            queue: Mutex::new(Vec::new()),
        }
    }
}

impl<E: Send + 'static> EventStore<E> {
    /// Appends `(to, event)`; callable from any thread.
    pub fn send(&self, to: Entity, event: E) {
        self.queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push((to, event));
    }

    /// Takes all queued events sorted by entity id (stable for equal ids,
    /// so same-entity sends keep arrival order).
    pub fn drain_sorted(&self) -> Vec<(Entity, E)> {
        let mut taken = self
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .split_off(0);
        taken.sort_by_key(|(entity, _)| entity.id());
        taken
    }

    /// Peeks all queued events sorted by entity id without consuming
    /// them (per-entity consumers over a shared snapshot; the queue is
    /// cleared explicitly with [`clear`](Self::clear)).
    pub fn read_sorted(&self) -> Vec<(Entity, E)>
    where
        E: Clone,
    {
        let mut snapshot = self
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        snapshot.sort_by_key(|(entity, _)| entity.id());
        snapshot
    }

    /// Drops all queued events (explicit end-of-frame lifecycle; automatic
    /// consume-once application is a `#[smart_system]` v2b follow-up).
    pub fn clear(&self) {
        self.queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    /// Number of queued events.
    pub fn len(&self) -> usize {
        self.queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Registers the [`EventStore`] singleton for `E` (systems fetch it
/// through [`Events`]; the store must exist before the schedule runs).
pub fn register_events<E: Send + Sync + 'static>(resources: &mut Resources) {
    if resources.get::<EventStore<E>>().is_none() {
        resources.insert(EventStore::<E>::default());
    }
}

/// Event parameter: `send` in one system, drain in another.
///
/// Declares `writes::<EventStore<E>>` when the body sends, otherwise
/// `reads::<EventStore<E>>` (a write declaration also covers the read
/// for enforcement), so producers and consumers never share a level.
/// Draining is sorted by entity id — deterministic across parallel and
/// sequential runs.
pub struct Events<'a, E: Send + Sync + 'static> {
    store: &'a EventStore<E>,
    _marker: PhantomData<&'a ()>,
}

impl<'a, E: Send + Sync + 'static> Events<'a, E> {
    /// Fetches the queue singleton; panics when [`register_events`] was
    /// not called (same missing-resource contract as [`Res`]).
    pub fn fetch(resources: &'a Resources) -> Self {
        let store = resources.get::<EventStore<E>>().unwrap_or_else(|| {
            panic!(
                "smart_system: missing EventStore<'{}'> — call register_events() before running the schedule",
                std::any::type_name::<E>()
            )
        });
        Self {
            store,
            _marker: PhantomData,
        }
    }

    /// Queues `event` for `to` (arrival order; application order is fixed
    /// later by [`EventStore::drain_sorted`]).
    pub fn send(&self, to: Entity, event: E) {
        self.store.send(to, event);
    }

    /// Takes all queued events sorted by entity id.
    pub fn drain_sorted(&self) -> Vec<(Entity, E)> {
        self.store.drain_sorted()
    }

    /// Peeks all queued events sorted by entity id without consuming.
    pub fn read_sorted(&self) -> Vec<(Entity, E)>
    where
        E: Clone,
    {
        self.store.read_sorted()
    }

    /// Drops all queued events.
    pub fn clear(&self) {
        self.store.clear();
    }

    /// Contributes the event-queue access: write when the system sends
    /// or clears, read when it only peeks/drains.
    pub fn access(writes_queue: bool) -> SystemAccess {
        if writes_queue {
            SystemAccess::new().writes::<EventStore<E>>()
        } else {
            SystemAccess::new().reads::<EventStore<E>>()
        }
    }
}

/// Uniform fetch + access contribution for `#[smart_system]` parameters.
///
/// The macro calls `<P as SystemParam>::access()` for the static
/// `System::access()` and `P::fetch(resources)` inside `run`; adding a
/// new parameter kind means implementing this trait plus a macro arm —
/// no scheduler changes.
pub trait SystemParam: Sized {
    /// Borrowed value handed to the per-entity body.
    type Item<'a>;

    /// Access contribution of this parameter.
    fn access() -> SystemAccess;

    /// Fetches the parameter from the world (panics on missing
    /// resources, like [`Res::fetch`]).
    fn fetch(resources: &Resources) -> Self::Item<'_>;
}

impl<T: PlainResource> SystemParam for Res<'_, T> {
    type Item<'a> = Res<'a, T>;

    fn access() -> SystemAccess {
        Self::access()
    }

    fn fetch(resources: &Resources) -> Self::Item<'_> {
        Res::fetch(resources)
    }
}

impl<T: Send + Sync + 'static> SystemParam for ResMut<'_, T> {
    type Item<'a> = ResMut<'a, T>;

    fn access() -> SystemAccess {
        Self::access()
    }

    fn fetch(resources: &Resources) -> Self::Item<'_> {
        ResMut::fetch(resources)
    }
}

/// Canonical lane-capture order: [`TypeId`]s sorted ascending.
///
/// Lanes are `RwLock`s; acquiring several in one system in different
/// orders across systems deadlocks. Generated `#[smart_system]` code
/// (and hand-written multi-lane systems) sort lane ids through this
/// helper before locking, so every system captures in the same order.
pub fn canonical_lane_order(ids: &mut [TypeId]) {
    ids.sort();
}

/// Marker asserting at macro-expansion time that [`SmartStore`] never
/// appears as a typed parameter: systems reach components through the
/// first (`Pack`) parameter, and a raw store handle would bypass the
/// derived access set. The `#[smart_system]` macro rejects
/// `Res<SmartStore>` with a dedicated diagnostic; this alias exists so
/// the rule is also greppable next to the parameter types.
pub type NoRawStore = SmartStore;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Plain {
        _x: f32,
    }

    impl PlainResource for Plain {}

    #[test]
    fn res_declares_read_and_fetches() {
        let mut resources = Resources::new();
        resources.insert(Plain { _x: 1.0 });
        let access = Res::<Plain>::access();
        assert_eq!(access, SystemAccess::new().reads::<Plain>());
        assert_eq!(*Res::<Plain>::fetch(&resources), Plain { _x: 1.0 });
    }

    #[test]
    fn res_mut_declares_write_without_plain_bound() {
        let mut resources = Resources::new();
        resources.insert(Mutex::new(1u32));
        let access = ResMut::<Mutex<u32>>::access();
        assert_eq!(access, SystemAccess::new().writes::<Mutex<u32>>());
        assert_eq!(*ResMut::<Mutex<u32>>::fetch(&resources).lock().unwrap(), 1);
    }

    #[test]
    fn events_drain_in_entity_id_order() {
        let store = EventStore::<&'static str>::default();
        store.send(Entity::new(9), "nine");
        store.send(Entity::new(2), "two");
        store.send(Entity::new(5), "five");
        let drained = store.drain_sorted();
        let ids: Vec<u32> = drained.iter().map(|(e, _)| e.id()).collect();
        assert_eq!(ids, vec![2, 5, 9]);
        assert!(store.is_empty());
    }

    #[test]
    fn events_access_send_is_write_drain_is_read() {
        assert_eq!(
            Events::<u32>::access(true),
            SystemAccess::new().writes::<EventStore<u32>>()
        );
        assert_eq!(
            Events::<u32>::access(false),
            SystemAccess::new().reads::<EventStore<u32>>()
        );
    }

    #[test]
    fn canonical_lane_order_sorts_type_ids() {
        let mut ids = vec![TypeId::of::<u32>(), TypeId::of::<u8>()];
        canonical_lane_order(&mut ids);
        let mut expected = vec![TypeId::of::<u8>(), TypeId::of::<u32>()];
        expected.sort();
        assert_eq!(ids, expected);
    }
}
