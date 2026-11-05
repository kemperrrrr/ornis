//! Deferred structural operations as a typed system parameter (R4, IDEAS §32).
//!
//! [`StructuralWriter`] is the `#[smart_system]` parameter for `add` /
//! `remove` / `spawn`: a body sees shared storage only, so those operations
//! enqueue onto a queue instead of applying inline.
//!
//! The type parameter selects *when* the queue drains, and that choice is
//! what keeps simulation deterministic:
//!
//! - [`FixedStep`] targets [`FixedStructuralQueue`], drained after **every**
//!   fixed tick. Use it from [`Stage::Gameplay`](crate::Stage::Gameplay)
//!   systems: tick `k+1` then observes tick `k`'s ops no matter how the host
//!   sliced frames into ticks, so results cannot depend on FPS or catch-up.
//! - [`FrameStep`] targets [`FrameStructuralQueue`], drained once per frame before
//!   [`Stage::Last`](crate::Stage::Last). Use it from the once-per-frame
//!   stages (`PostFrame` and friends).
//!
//! The engine validates the pairing at registration: a fixed writer outside
//! the `Gameplay` plan, or a frame writer inside it, is a configuration
//! error ([`Engine::add_stage_system`](crate::Engine::add_stage_system)).
//!
//! Each writer declares a write to its queue, which serializes every
//! queueing system across parallel levels in registration order — the source
//! of the queues' deterministic FIFO.

use std::sync::{Arc, OnceLock};

use crate::Entity;
use crate::mutation::{FixedStructuralQueue, FrameStructuralQueue, StructuralQueue};
use crate::schedule::{Resources, SystemAccess};
use crate::smart_store::{Pack, SmartStore};

use super::SystemParam;

/// Queue kind marker: ops drain after every fixed tick.
///
/// A unit type with no state: it exists so the compiler — not a runtime flag —
/// decides which queue a [`StructuralWriter`] addresses and which
/// [`Stage`](crate::Stage) it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FixedStep;

/// Queue kind marker: ops drain once per frame, before
/// [`Stage::Last`](crate::Stage::Last).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameStep;

/// Maps a [`StructuralWriter`]'s kind parameter to its queue singleton.
///
/// Sealed: [`FixedStep`] and [`FrameStep`] are the only kinds, so the engine's
/// stage/writer validation covers every writer that can exist.
pub trait Kind: sealed::Sealed {
    /// The queue resource this kind writes to.
    type Queue: StructuralQueue + 'static + Send + Sync;

    /// Whether writer systems of this kind belong to the fixed-step
    /// (`Stage::Gameplay`) plan. The engine rejects a mismatch at
    /// registration.
    const IS_FIXED: bool;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::FixedStep {}
    impl Sealed for super::FrameStep {}
}

impl Kind for FixedStep {
    type Queue = FixedStructuralQueue;
    const IS_FIXED: bool = true;
}

impl Kind for FrameStep {
    type Queue = FrameStructuralQueue;
    const IS_FIXED: bool = false;
}

/// Deferred structural-ops parameter, generic over *when* it applies.
///
/// `StructuralWriter<FixedStep>`/`StructuralWriter<FrameStep>` queue `add` /
/// `remove` / `spawn` for the matching boundary instead of applying them
/// inline. The type always declares a write to that queue, independent of
/// which helpers the body calls, so all queueing systems conflict and run in
/// registration order — and the FIFO apply reproduces that order for any
/// thread count.
///
/// `Copy` (shared borrows only): the generated per-entity loop passes
/// parameters by value, so they must survive more than one iteration.
pub struct StructuralWriter<'a, K: Kind = FixedStep> {
    queue: &'a K::Queue,
    store: &'a SmartStore,
}

impl<K: Kind> Clone for StructuralWriter<'_, K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: Kind> Copy for StructuralWriter<'_, K> {}

impl<'a, K: Kind> StructuralWriter<'a, K> {
    /// Fetches the queue and store singletons; panics when the queue was not
    /// registered (same missing-resource contract as [`Res`](super::Res)).
    /// [`World`](crate::World) registers both queues at construction.
    ///
    /// # Panics
    /// When the matching queue or the [`SmartStore`] is absent — a
    /// registration bug, not a recoverable state.
    // qual:allow(complexity, error_handling) reason: "deliberate missing-resource contract — the queue must be registered before the schedule runs; mirrors the Res/EventWriter fetch panics"
    pub fn fetch(resources: &'a Resources) -> Self {
        let queue = resources.get::<K::Queue>().unwrap_or_else(|| {
            panic!(
                "smart_system: missing structural queue — call \
                 register_structural_queues() (or run inside a World/Engine) \
                 before running the schedule; also check the writer kind matches \
                 the stage (`FixedStep` for Gameplay, `FrameStep` for once-per-frame stages)"
            )
        });
        let store = resources
            .get::<SmartStore>()
            .expect("smart_system: missing SmartStore resource");
        Self { queue, store }
    }

    /// Queues creation of an entity carrying `bundle` and returns its handle.
    ///
    /// The id is reserved now through the store's interior allocator (the
    /// only allocation route available under shared access) and the bundle
    /// lands at apply. Between the call and the apply the handle is **live
    /// but empty**: `is_alive` says `true` while no lane has an entry for it.
    /// Use [`spawn_deferred`](Self::spawn_deferred) when nothing may observe
    /// the entity before its components exist.
    ///
    /// Handle ordering follows *call* order. Because the writer serializes
    /// all queueing systems, every writer observes call order == id order; a
    /// system that allocates directly (`SmartStore::create_entity`) does not
    /// conflict with the queue and may interleave ids — give such a system a
    /// store write when id order matters.
    ///
    /// A handle to an entity despawned before the apply is skipped there, so
    /// a queued spawn can never resurrect a dead id.
    pub fn spawn<P: Pack>(&self, bundle: P) -> Entity {
        let entity = self.store.create_entity();
        self.queue.enqueue(Box::new(move |store: &mut SmartStore| {
            if store.is_alive(entity) {
                bundle.pack_insert(store, entity);
            }
        }));
        entity
    }

    /// Queues creation of an entity and resolves its handle at apply time.
    ///
    /// Unlike [`spawn`](Self::spawn) the id is drawn *when the op runs*, so
    /// nothing is observable between the call and the boundary: no live
    /// entity without components, and ids follow apply (FIFO) order rather
    /// than call order. The returned [`PendingEntity`] resolves after the
    /// boundary; calling [`resolve`](PendingEntity::resolve) before it panics
    /// rather than handing out a nonexistent handle.
    // qual:api — the non-observable spawn alternative for hosts that cannot use a live handle.
    pub fn spawn_deferred<P: Pack>(&self, bundle: P) -> PendingEntity {
        let slot = Arc::new(OnceLock::new());
        let handle = PendingEntity(slot.clone());
        self.queue.enqueue(Box::new(move |store: &mut SmartStore| {
            let entity = store.create_entity();
            bundle.pack_insert(store, entity);
            let _ = slot.set(entity);
        }));
        handle
    }

    /// Queues `component` for `entity` (upsert; the hot lane for `T` is
    /// created on first use). Skipped at apply when the entity is no longer
    /// alive.
    pub fn insert<T: 'static + Clone + Send + Sync>(&self, entity: Entity, component: T) {
        self.queue.enqueue(Box::new(move |store: &mut SmartStore| {
            if store.is_alive(entity) {
                store.insert(entity, component);
            }
        }));
    }

    /// Queues removal of component `T` from `entity` (no-op when the entity
    /// or the lane entry is already gone).
    ///
    /// # Panics
    /// Immediately, at the call site, when `T` is a lock-free lane: the
    /// deferred apply needs [`SmartStore::write_lane`], which rejects those,
    /// and the snapshot API has no `remove`. Failing here instead of at the
    /// boundary keeps the error next to the offending system.
    pub fn remove<T: 'static + Send + Sync>(&self, entity: Entity) {
        assert!(
            !self.store.is_lock_free_lane::<T>(),
            "smart_system: `remove::<{}>()` cannot target a lock-free lane — \
             lock-free lanes expose only the closure snapshot API, which has no \
             deferred remove; register `{}` as an RwLock lane",
            std::any::type_name::<T>(),
            std::any::type_name::<T>()
        );
        self.queue.enqueue(Box::new(move |store: &mut SmartStore| {
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
        self.queue.enqueue(Box::new(move |store: &mut SmartStore| {
            store.destroy_entity(entity);
        }));
    }

    /// Declares exclusive access to this kind's queue — the ordering key that
    /// serializes queueing systems in registration order.
    pub fn access() -> SystemAccess {
        SystemAccess::new().writes::<K::Queue>()
    }
}

impl<K: Kind> SystemParam for StructuralWriter<'_, K> {
    type Item<'a> = StructuralWriter<'a, K>;

    fn access() -> SystemAccess {
        StructuralWriter::<K>::access()
    }

    fn fetch(resources: &Resources) -> Self::Item<'_> {
        StructuralWriter::<K>::fetch(resources)
    }
}

/// Handle reserved by [`StructuralWriter::spawn_deferred`], resolved at apply.
///
/// Cheap to clone; every clone observes the same resolution, mirroring how an
/// `Entity` handle refers to one entity.
#[derive(Clone, Debug)]
pub struct PendingEntity(Arc<OnceLock<Entity>>);

impl PendingEntity {
    /// The resolved entity.
    ///
    /// # Panics
    /// Before the queue applies the spawn — an unresolved handle must never
    /// reach the world, so this is a hard error rather than a silent `None`.
    // qual:allow(complexity, error_handling) reason: "the panic IS the contract: resolving an unapplied handle must fail loudly, and `# Panics` documents it; `try_resolve` is the soft variant"
    pub fn resolve(&self) -> Entity {
        self.0
            .get()
            .copied()
            .expect("PendingEntity resolved before its structural op applied")
    }

    /// The resolved entity, or `None` while the spawn is still queued.
    // qual:api — the non-panicking half of `resolve`, for hosts that may run before the boundary.
    pub fn try_resolve(&self) -> Option<Entity> {
        self.0.get().copied()
    }
}
