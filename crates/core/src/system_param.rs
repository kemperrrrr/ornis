//! Typed system parameters for [`#[smart_system]`](crate::smart_system)
//! (IDEAS §32, RAT-31).
//!
//! Access follows from **parameter types** (Bevy-style `SystemParam`),
//! never from syntactic body analysis: [`Res`] declares a shared read,
//! [`ResMut`] an exclusive-scheduler write (mutation itself goes through
//! the resource's interior mutability), and [`EventWriter`]/[`EventReader`]
//! declare event-queue access by role. [`PlainResource`] keeps interior
//! mutability out of [`Res`]:
//! a `Mutex`/`RwLock`/`Atomic*` behind a shared read would be
//! memory-safe but non-deterministic (see the commutativity contract in
//! [`crate::schedule`]), so it must be declared as [`ResMut`] instead.
//!
//! [`SmartStore`] is deliberately *not* a parameter: systems generated
//! by `#[smart_system]` reach the store through the first (`Pack`)
//! parameter. Raw `&Resources`, `Res<SmartStore>`, and `ResMut` handles for
//! `SmartStore`/`Resources` would bypass declared accesses, so they are
//! rejected by the macro or the `ResMutResource` bound.

use std::any::TypeId;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::{
    AtomicBool, AtomicI8, AtomicI16, AtomicI32, AtomicI64, AtomicIsize, AtomicU8, AtomicU16,
    AtomicU32, AtomicU64, AtomicUsize,
};
use std::sync::{Mutex, RwLock};
use std::time::Duration;

use crate::schedule::{Resources, SystemAccess};
use crate::{Entity, FixedTime, SmartStore, Time};

/// Marker for types that may be accessed through [`ResMut`].
///
/// This bound is deliberately opt-in: `SmartStore` and [`Resources`] do not
/// implement it, so aliases cannot smuggle either scheduler handle into a
/// `#[smart_system]` body. Ordinary resources can implement it when their
/// state is covered by the access key for the concrete type.
///
/// # Safety
/// Implementors must not expose scheduler internals such as `SmartStore` or
/// `Resources` through this resource. Mutable state must remain encapsulated
/// by the resource so `ResMut<T>`'s `writes::<T>()` declaration is complete;
/// in particular, state must not be aliased through a different resource type
/// with an independent access key. Shared-pointer wrappers such as `Arc<T>`
/// therefore need an explicit, audited newtype opt-in rather than a blanket
/// implementation.
pub unsafe trait ResMutResource: Send + Sync + 'static {}

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
///
/// # Safety
/// Every value reachable from an implementation must be immutable through
/// shared references, or changed only between scheduler runs. Use
/// [`ResMutResource`] for interior-mutable state.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be shared as `Res`: it may hold interior mutability; \
               use `ResMut` (declares a write) or `#[derive(PlainResource)]` on a plain-data struct"
)]
pub unsafe trait PlainResource: ResMutResource {}

macro_rules! plain_impl {
    ($($ty:ty),*) => {
        $(
            unsafe impl ResMutResource for $ty {}
            unsafe impl PlainResource for $ty {}
        )*
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

unsafe impl<T: ResMutResource> ResMutResource for Option<T> {}
unsafe impl<T: PlainResource> PlainResource for Option<T> {}
unsafe impl<T: ResMutResource> ResMutResource for Vec<T> {}
unsafe impl<T: PlainResource> PlainResource for Vec<T> {}
unsafe impl<T: ResMutResource> ResMutResource for Box<T> {}
unsafe impl<T: PlainResource> PlainResource for Box<T> {}
unsafe impl<T: ResMutResource, E: ResMutResource> ResMutResource for Result<T, E> {}
unsafe impl<T: PlainResource, E: PlainResource> PlainResource for Result<T, E> {}
unsafe impl<T: ResMutResource> ResMutResource for [T; 0] {}
unsafe impl<T: PlainResource> PlainResource for [T; 0] {}
unsafe impl<T: ResMutResource> ResMutResource for [T; 1] {}
unsafe impl<T: PlainResource> PlainResource for [T; 1] {}
unsafe impl<T: ResMutResource> ResMutResource for [T; 2] {}
unsafe impl<T: PlainResource> PlainResource for [T; 2] {}
unsafe impl<T: ResMutResource> ResMutResource for [T; 3] {}
unsafe impl<T: PlainResource> PlainResource for [T; 3] {}
unsafe impl<T: ResMutResource> ResMutResource for [T; 4] {}
unsafe impl<T: PlainResource> PlainResource for [T; 4] {}
unsafe impl ResMutResource for () {}
unsafe impl PlainResource for () {}
unsafe impl<A: ResMutResource> ResMutResource for (A,) {}
unsafe impl<A: PlainResource> PlainResource for (A,) {}
unsafe impl<A: ResMutResource, B: ResMutResource> ResMutResource for (A, B) {}
unsafe impl<A: PlainResource, B: PlainResource> PlainResource for (A, B) {}
unsafe impl<A: ResMutResource, B: ResMutResource, C: ResMutResource>
    ResMutResource for (A, B, C)
{
}
unsafe impl<A: PlainResource, B: PlainResource, C: PlainResource> PlainResource for (A, B, C) {}
unsafe impl<K: ResMutResource, V: ResMutResource> ResMutResource for HashMap<K, V> {}
unsafe impl<K: PlainResource, V: PlainResource> PlainResource for HashMap<K, V> {}

// Synchronization primitives are valid mutable resources only when the
// protected value is itself an approved resource type. In particular this
// excludes `Mutex<SmartStore>` and `Mutex<Resources>` by default.
unsafe impl<T: ResMutResource> ResMutResource for Mutex<T> {}
unsafe impl<T: ResMutResource> ResMutResource for RwLock<T> {}
unsafe impl<T: ResMutResource> ResMutResource for std::sync::OnceLock<T> {}
unsafe impl ResMutResource for AtomicBool {}
unsafe impl ResMutResource for AtomicI8 {}
unsafe impl ResMutResource for AtomicI16 {}
unsafe impl ResMutResource for AtomicI32 {}
unsafe impl ResMutResource for AtomicI64 {}
unsafe impl ResMutResource for AtomicIsize {}
unsafe impl ResMutResource for AtomicU8 {}
unsafe impl ResMutResource for AtomicU16 {}
unsafe impl ResMutResource for AtomicU32 {}
unsafe impl ResMutResource for AtomicU64 {}
unsafe impl ResMutResource for AtomicUsize {}

/// Shared resource parameter: declares `reads::<T>`.
///
/// `T` must be [`PlainResource`] (no interior mutability), so sharing it
/// across a parallel level is deterministic. For interior-mutable state
/// use [`ResMut`].
///
/// `Copy` (a shared borrow only): the generated per-entity loop passes
/// parameters by value, so they must survive more than one iteration.
/// Manual impls: a derive would add a `T: Copy` bound the resource types
/// do not satisfy.
pub struct Res<'a, T: PlainResource> {
    value: &'a T,
}

impl<T: PlainResource> Clone for Res<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: PlainResource> Copy for Res<'_, T> {}

impl<'a, T: PlainResource> Res<'a, T> {
    /// Fetches the singleton; panics with the type name when absent
    /// (same contract as Bevy's `Res`: a declared resource must exist).
    // qual:allow(complexity, error_handling) reason: "deliberate missing-resource contract — a declared resource must exist; mirrors Bevy's Res fetch panic with the type name attached"
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
/// Unlike [`Res`], `T` may contain interior mutability, but it must opt in
/// to [`ResMutResource`] so its resource access remains scheduler-visible.
///
/// `Copy` (a shared borrow only): see [`Res`]. Manual impls for the
/// same derive-bound reason.
pub struct ResMut<'a, T: ResMutResource> {
    value: &'a T,
}

impl<T: ResMutResource> Clone for ResMut<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ResMutResource> Copy for ResMut<'_, T> {}

impl<'a, T: ResMutResource> ResMut<'a, T> {
    /// Fetches the singleton; panics with the type name when absent.
    // qual:allow(complexity, error_handling) reason: "deliberate missing-resource contract — a declared resource must exist; mirrors Bevy's Res fetch panic with the type name attached"
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

impl<T: ResMutResource> Deref for ResMut<'_, T> {
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
/// Writers and readers both declare a write through [`EventWriter`] and
/// [`EventReader`]; readers may drain the queue.
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
/// through [`EventWriter`] or [`EventReader`]; the store must exist before
/// the schedule runs).
pub fn register_events<E: Send + Sync + 'static>(resources: &mut Resources) {
    if resources.get::<EventStore<E>>().is_none() {
        resources.insert(EventStore::<E>::default());
    }
}

/// Event producer parameter. Its type always declares a write to the queue,
/// independent of which helpers are called in the function body.
pub struct EventWriter<'a, E: Send + Sync + 'static> {
    store: &'a EventStore<E>,
}

impl<E: Send + Sync + 'static> Clone for EventWriter<'_, E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Send + Sync + 'static> Copy for EventWriter<'_, E> {}

impl<'a, E: Send + Sync + 'static> EventWriter<'a, E> {
    /// Fetches the queue singleton; panics when [`register_events`] was
    /// not called (same missing-resource contract as [`Res`]).
    // qual:allow(complexity, error_handling) reason: "deliberate missing-resource contract — register_events must run before the schedule; same rationale as Res::fetch above"
    pub fn fetch(resources: &'a Resources) -> Self {
        let store = resources.get::<EventStore<E>>().unwrap_or_else(|| {
            panic!(
                "smart_system: missing EventStore<'{}'> — call register_events() before running the schedule",
                std::any::type_name::<E>()
            )
        });
        Self { store }
    }

    /// Queues `event` for `to` (arrival order; application order is fixed
    /// later by [`EventReader::drain_sorted`]).
    pub fn send(&self, to: Entity, event: E) {
        self.store.send(to, event);
    }

    /// Declares exclusive access to the event queue.
    pub fn access() -> SystemAccess {
        SystemAccess::new().writes::<EventStore<E>>()
    }
}

/// Event consumer parameter.
///
/// It declares a write because [`drain_sorted`](Self::drain_sorted) consumes
/// queue contents. This is conservative even for a body that only calls
/// [`read_sorted`](Self::read_sorted), ensuring consumers never share a
/// scheduler level with producers or other consumers.
pub struct EventReader<'a, E: Send + Sync + 'static> {
    store: &'a EventStore<E>,
}

impl<E: Send + Sync + 'static> Clone for EventReader<'_, E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Send + Sync + 'static> Copy for EventReader<'_, E> {}

impl<'a, E: Send + Sync + 'static> EventReader<'a, E> {
    /// Fetches the queue singleton; panics when [`register_events`] was
    /// not called (same missing-resource contract as [`Res`]).
    // qual:allow(complexity, error_handling) reason: "deliberate missing-resource contract — register_events must run before the schedule; same rationale as Res::fetch above"
    pub fn fetch(resources: &'a Resources) -> Self {
        let store = resources.get::<EventStore<E>>().unwrap_or_else(|| {
            panic!(
                "smart_system: missing EventStore<'{}'> — call register_events() before running the schedule",
                std::any::type_name::<E>()
            )
        });
        Self { store }
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

    /// Declares exclusive access because this reader can drain the queue.
    pub fn access() -> SystemAccess {
        SystemAccess::new().writes::<EventStore<E>>()
    }
}

/// Uniform fetch + access contract for typed system parameters.
///
/// `#[smart_system]` emits the same wrapper access/fetch operations into its
/// generated `System`; this trait also lets generic hosts use those operations
/// without duplicating parameter bounds. Adding a macro parameter kind means
/// implementing this trait plus parser/emitter arms — no scheduler changes.
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

impl<T: ResMutResource> SystemParam for ResMut<'_, T> {
    type Item<'a> = ResMut<'a, T>;

    fn access() -> SystemAccess {
        Self::access()
    }

    fn fetch(resources: &Resources) -> Self::Item<'_> {
        ResMut::fetch(resources)
    }
}

impl<E: Send + Sync + 'static> SystemParam for EventWriter<'_, E> {
    type Item<'a> = EventWriter<'a, E>;

    fn access() -> SystemAccess {
        EventWriter::<E>::access()
    }

    fn fetch(resources: &Resources) -> Self::Item<'_> {
        EventWriter::fetch(resources)
    }
}

impl<E: Send + Sync + 'static> SystemParam for EventReader<'_, E> {
    type Item<'a> = EventReader<'a, E>;

    fn access() -> SystemAccess {
        EventReader::<E>::access()
    }

    fn fetch(resources: &Resources) -> Self::Item<'_> {
        EventReader::fetch(resources)
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

/// Legacy type alias naming the store handle forbidden as a system resource.
/// Systems reach components through the first (`Pack`) parameter; a raw
/// store handle would bypass the derived lane access set. The macro rejects
/// direct `Res<SmartStore>` and `ResMutResource` is deliberately not
/// implemented for `SmartStore` or [`Resources`].
pub type NoRawStore = SmartStore;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Plain {
        _x: f32,
    }

    unsafe impl ResMutResource for Plain {}
    unsafe impl PlainResource for Plain {}

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
        assert_eq!(
            *ResMut::<Mutex<u32>>::fetch(&resources).lock().unwrap(),
            1
        );
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
    fn event_writer_and_reader_both_declare_exclusive_access() {
        let write = SystemAccess::new().writes::<EventStore<u32>>();
        assert_eq!(EventWriter::<u32>::access(), write);
        // Readers may drain, so their parameter type is conservatively a write.
        assert_eq!(EventReader::<u32>::access(), write);
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
