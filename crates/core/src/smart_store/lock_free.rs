//! Lock-free (clone-on-write, epoch-reclaimed) lane storage for
//! [`SmartStore`](super::SmartStore).
//!
//! [`LockFreeLaneInner`] trades write cost (full snapshot + atomic swap) for
//! wait-free reads via crossbeam epochs. It backs the experimental
//! `lock-free` registration path; the default `RwLock` lanes live in the
//! parent module. Split out so the store's hot/cold lane lifecycle stays
//! separate from the epoch machinery.

use std::any::Any;
use std::sync::atomic::Ordering;

use crossbeam_epoch::{Atomic, Guard, Owned, pin as epoch_pin};

use crate::component_store::ComponentStore;
use crate::entity::Entity;

use super::Lane;

pub(crate) struct LockFreeLaneInner<T: Clone + Send + Sync> {
    store: Atomic<ComponentStore<T>>,
}

impl<T: 'static + Clone + Send + Sync> LockFreeLaneInner<T> {
    pub(super) fn new() -> Self {
        Self {
            store: Atomic::new(ComponentStore::new()),
        }
    }

    pub(super) fn read<'g>(&'g self, guard: &'g Guard) -> &'g ComponentStore<T> {
        let shared = self.store.load(Ordering::Acquire, guard);
        // SAFETY: `shared` was loaded under `guard`, which the caller holds
        // for `'g`; the epoch reclaimer cannot free the snapshot while pinned.
        unsafe { shared.deref() }
    }

    pub(super) fn write(&self, mut f: impl FnMut(&mut ComponentStore<T>)) {
        // CAS publish loop: `load -> clone -> store` without a CAS lets two
        // writers read the same snapshot and both `defer_destroy` it (double
        // free, UB). Only the thread whose `compare_exchange` succeeds owns
        // the replaced snapshot and may retire it; losers drop their private
        // clone inline and retry on the fresh snapshot. Success ordering is
        // AcqRel (acquire the latest snapshot, release the publish);
        // failure ordering is Acquire (reload the current pointer).
        loop {
            let guard = epoch_pin();
            let shared = self.store.load(Ordering::Acquire, &guard);
            // SAFETY: `shared` is pinned by `guard` for this iteration.
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
                    // SAFETY: this thread replaced `shared`; no other thread
                    // can retire it because its CAS on the same `shared`
                    // pointer now fails. Epoch reclamation frees it once
                    // readers drain.
                    unsafe { guard.defer_destroy(shared) };
                    break;
                }
                Err(_) => {
                    // CAS lost: our `Owned` clone is dropped with the error;
                    // the loop retries against the winning snapshot.
                }
            }
        }
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

    fn is_lock_free(&self) -> bool {
        true
    }

    fn clone_lane(&self) -> Box<dyn Lane> {
        let guard = epoch_pin();
        let snapshot = self.read(&guard).clone();
        Box::new(LockFreeLaneInner {
            store: Atomic::new(snapshot),
        })
    }
}

impl<T: Clone + Send + Sync> Drop for LockFreeLaneInner<T> {
    // qual:allow(unsafe) reason: "Drop-only forwarder; the unsafe operations live in free_live_snapshot, which is the counted warning"
    fn drop(&mut self) {
        // SAFETY: `&mut self` proves exclusive access, and lock-free reads
        // go through `with_lock_free_lane(&self, ..)`, whose snapshot cannot
        // outlive the store — so no live reader of this snapshot exists and
        // none can appear afterwards. `Drop` is the only caller allowed by
        // `free_live_snapshot`.
        unsafe { super::free_live_snapshot(&mut self.store) };
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
