//! Typed system parameters for [`#[smart_system]`](crate::smart_system)
//! (IDEAS §32, RAT-31).
//!
//! Access follows from **parameter types** (Bevy-style `SystemParam`),
//! never from syntactic body analysis: [`Res`] declares a shared read,
//! [`ResMut`] an exclusive-scheduler write (mutation itself goes through
//! the resource's interior mutability), and [`EventWriter`]/[`EventReader`]
//! declare event-queue access by role ([`events`] owns the queue itself).
//! [`PlainResource`] keeps interior mutability out of [`Res`]:
//! a `Mutex`/`RwLock`/`Atomic*` behind a shared read would be
//! memory-safe but non-deterministic (see the commutativity contract in
//! [`crate::schedule`]), so it must be declared as [`ResMut`] instead.
//!
//! [`crate::SmartStore`] is deliberately *not* a parameter: systems generated
//! by `#[smart_system]` reach the store through the first (`Pack`)
//! parameter. Raw `&Resources`, `Res<SmartStore>`, and `ResMut` handles for
//! `SmartStore`/`Resources` would bypass declared accesses, so they are
//! rejected by the macro or the `ResMutResource` bound.

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::{
    AtomicBool, AtomicI8, AtomicI16, AtomicI32, AtomicI64, AtomicIsize, AtomicU8, AtomicU16,
    AtomicU32, AtomicU64, AtomicUsize,
};
use std::sync::{Mutex, RwLock};
use std::time::Duration;

use crate::schedule::{Resources, SystemAccess};
use crate::{FixedTime, Time};

mod events;

pub use events::{EventReader, EventStore, EventWriter, register_events};

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

// SAFETY (both traits): every `plain_impl!` type is owned, `'static` plain
// data — `&T` reaches no interior mutability, so sharing it across a level
// through `Res` is deterministic, and `ResMut`'s `writes::<T>()` key covers
// all mutation (the type is only mutable through `&mut`, which the scheduler
// serializes).
macro_rules! plain_impl {
    ($($ty:ty),*) => {
        $(
            // SAFETY: every expanded type is owned `'static` plain data —
            // `&T` reaches no interior mutability, so sharing it across a
            // level through `Res` is deterministic, and `ResMut`'s
            // `writes::<T>()` key covers all mutation (only via `&mut`,
            // which the scheduler serializes).
            unsafe impl ResMutResource for $ty {}
            // SAFETY: same plain-data argument as above — no interior
            // mutability is reachable through a shared reference.
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

// SAFETY (container/tuple impls): purely structural forwarding — a container
// is an approved resource only when the bound holds for everything reachable
// from it, so no new state and no hidden interior mutability is introduced.
// (Each impl below repeats the one-line justification so the
// `undocumented_unsafe_blocks` gate sees it directly on the item.)
// SAFETY: structural forwarding — `Option` adds no state beyond `T`.
unsafe impl<T: ResMutResource> ResMutResource for Option<T> {}
// SAFETY: structural forwarding — `Option` adds no state beyond `T`.
unsafe impl<T: PlainResource> PlainResource for Option<T> {}
// SAFETY: structural forwarding — `Vec` adds no state beyond `T`.
unsafe impl<T: ResMutResource> ResMutResource for Vec<T> {}
// SAFETY: structural forwarding — `Vec` adds no state beyond `T`.
unsafe impl<T: PlainResource> PlainResource for Vec<T> {}
// SAFETY: structural forwarding — `Box` adds no state beyond `T`.
unsafe impl<T: ResMutResource> ResMutResource for Box<T> {}
// SAFETY: structural forwarding — `Box` adds no state beyond `T`.
unsafe impl<T: PlainResource> PlainResource for Box<T> {}
// SAFETY: structural forwarding — `Result` adds no state beyond `T`/`E`.
unsafe impl<T: ResMutResource, E: ResMutResource> ResMutResource for Result<T, E> {}
// SAFETY: structural forwarding — `Result` adds no state beyond `T`/`E`.
unsafe impl<T: PlainResource, E: PlainResource> PlainResource for Result<T, E> {}
// SAFETY: structural forwarding — `[T; 0]` holds no values at all.
unsafe impl<T: ResMutResource> ResMutResource for [T; 0] {}
// SAFETY: structural forwarding — `[T; 0]` holds no values at all.
unsafe impl<T: PlainResource> PlainResource for [T; 0] {}
// SAFETY: structural forwarding — the array adds no state beyond `T`.
unsafe impl<T: ResMutResource> ResMutResource for [T; 1] {}
// SAFETY: structural forwarding — the array adds no state beyond `T`.
unsafe impl<T: PlainResource> PlainResource for [T; 1] {}
// SAFETY: structural forwarding — the array adds no state beyond `T`.
unsafe impl<T: ResMutResource> ResMutResource for [T; 2] {}
// SAFETY: structural forwarding — the array adds no state beyond `T`.
unsafe impl<T: PlainResource> PlainResource for [T; 2] {}
// SAFETY: structural forwarding — the array adds no state beyond `T`.
unsafe impl<T: ResMutResource> ResMutResource for [T; 3] {}
// SAFETY: structural forwarding — the array adds no state beyond `T`.
unsafe impl<T: PlainResource> PlainResource for [T; 3] {}
// SAFETY: structural forwarding — the array adds no state beyond `T`.
unsafe impl<T: ResMutResource> ResMutResource for [T; 4] {}
// SAFETY: structural forwarding — the array adds no state beyond `T`.
unsafe impl<T: PlainResource> PlainResource for [T; 4] {}
// SAFETY: `()` reaches no state, so sharing it is always sound.
unsafe impl ResMutResource for () {}
// SAFETY: `()` reaches no state, so sharing it is always sound.
unsafe impl PlainResource for () {}
// SAFETY: structural forwarding — the tuple adds no state beyond `A`.
unsafe impl<A: ResMutResource> ResMutResource for (A,) {}
// SAFETY: structural forwarding — the tuple adds no state beyond `A`.
unsafe impl<A: PlainResource> PlainResource for (A,) {}
// SAFETY: structural forwarding — the tuple adds no state beyond `A`/`B`.
unsafe impl<A: ResMutResource, B: ResMutResource> ResMutResource for (A, B) {}
// SAFETY: structural forwarding — the tuple adds no state beyond `A`/`B`.
unsafe impl<A: PlainResource, B: PlainResource> PlainResource for (A, B) {}
// SAFETY: structural forwarding — the tuple adds no state beyond `A`/`B`/`C`.
unsafe impl<A: ResMutResource, B: ResMutResource, C: ResMutResource> ResMutResource for (A, B, C) {}
// SAFETY: structural forwarding — the tuple adds no state beyond `A`/`B`/`C`.
unsafe impl<A: PlainResource, B: PlainResource, C: PlainResource> PlainResource for (A, B, C) {}
// SAFETY: structural forwarding — the map adds no state beyond `K`/`V`.
unsafe impl<K: ResMutResource, V: ResMutResource> ResMutResource for HashMap<K, V> {}
// SAFETY: structural forwarding — the map adds no state beyond `K`/`V`.
unsafe impl<K: PlainResource, V: PlainResource> PlainResource for HashMap<K, V> {}

// Synchronization primitives are valid mutable resources only when the
// protected value is itself an approved resource type. In particular this
// excludes `Mutex<SmartStore>` and `Mutex<Resources>` by default.
//
// SAFETY: `T: ResMutResource` excludes scheduler handles
// (`Mutex<SmartStore>`, `Mutex<Resources>`): the state belongs solely to
// this `Mutex`, and the `writes::<Mutex<T>>()` key covers all mutation.
// Reachable exclusively through `ResMut` (a declared `writes` access) —
// never through `Res`/`PlainResource`.
// SAFETY: interior-mutable, hence `ResMut`-only (see above).
unsafe impl<T: ResMutResource> ResMutResource for Mutex<T> {}
// SAFETY: interior-mutable, hence `ResMut`-only (see above).
unsafe impl<T: ResMutResource> ResMutResource for RwLock<T> {}
// SAFETY: interior-mutable, hence `ResMut`-only (see above).
unsafe impl<T: ResMutResource> ResMutResource for std::sync::OnceLock<T> {}
// SAFETY: atomics carry their own state with no reachable aliases.
unsafe impl ResMutResource for AtomicBool {}
// SAFETY: atomics carry their own state with no reachable aliases.
unsafe impl ResMutResource for AtomicI8 {}
// SAFETY: atomics carry their own state with no reachable aliases.
unsafe impl ResMutResource for AtomicI16 {}
// SAFETY: atomics carry their own state with no reachable aliases.
unsafe impl ResMutResource for AtomicI32 {}
// SAFETY: atomics carry their own state with no reachable aliases.
unsafe impl ResMutResource for AtomicI64 {}
// SAFETY: atomics carry their own state with no reachable aliases.
unsafe impl ResMutResource for AtomicIsize {}
// SAFETY: atomics carry their own state with no reachable aliases.
unsafe impl ResMutResource for AtomicU8 {}
// SAFETY: atomics carry their own state with no reachable aliases.
unsafe impl ResMutResource for AtomicU16 {}
// SAFETY: atomics carry their own state with no reachable aliases.
unsafe impl ResMutResource for AtomicU32 {}
// SAFETY: atomics carry their own state with no reachable aliases.
unsafe impl ResMutResource for AtomicU64 {}
// SAFETY: atomics carry their own state with no reachable aliases.
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[derive(Debug, PartialEq)]
    struct Plain {
        _x: f32,
    }

    // SAFETY: test-only plain-data struct — `f32` field, no interior
    // mutability reachable through a shared reference.
    unsafe impl ResMutResource for Plain {}
    // SAFETY: same plain-data argument as above.
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
        assert_eq!(*ResMut::<Mutex<u32>>::fetch(&resources).lock().unwrap(), 1);
    }
}
