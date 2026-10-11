//! Event queues for typed system parameters (IDEAS §24.1, §32).
//!
//! [`EventStore`] is the singleton resource holding `(entity, event)` pairs;
//! [`EventWriter`] and [`EventReader`] are the parameters a `#[smart_system]`
//! declares, and both contribute an exclusive (`writes`) access because a
//! reader may drain the queue. Draining sorts by entity id, so application
//! order is deterministic regardless of the parallel level that produced the
//! events.

use std::sync::Mutex;

use crate::Entity;
use crate::schedule::{Resources, SystemAccess};

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
    /// not called (same missing-resource contract as
    /// [`Res`](super::Res)).
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
    /// not called (same missing-resource contract as
    /// [`Res`](super::Res)).
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
