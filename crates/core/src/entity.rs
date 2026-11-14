//! Entity identifiers and allocation.
//!
//! An [`Entity`] is a lightweight handle (index + generation) that systems
//! use to reference a game object in the ECS. The [`EntityAllocator`]
//! recycles freed indices and bumps their generation, so stale handles
//! referring to destroyed entities are detected instead of silently
//! aliasing a newly created one.

/// Slot index of an [`Entity`] in the allocator table.
///
/// Newtype over `u32` so entity ids never mix with generations, dense
/// indices or physics handles at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityId(u32);

impl EntityId {
    /// Wraps a raw `u32` entity slot.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` slot index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Slot index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for EntityId {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for EntityId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<EntityId> for u32 {
    fn from(h: EntityId) -> Self {
        h.0
    }
}

impl From<EntityId> for usize {
    fn from(h: EntityId) -> Self {
        h.0 as usize
    }
}

/// Generation guard of an [`Entity`]: bumped on every id recycle so stale
/// handles fail liveness checks instead of aliasing a new entity.
///
/// Newtype over `u32` so generations never mix with ids at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Generation(u32);

impl Generation {
    /// Wraps a raw `u32` generation counter.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` generation counter.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Generation as `usize` (rarely needed; prefer [`Self::as_u32`]).
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for Generation {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<Generation> for u32 {
    fn from(h: Generation) -> Self {
        h.0
    }
}

/// Dense index into a [`crate::component_store::ComponentStore`]'s packed
/// data array (position in insertion order, modulo swap-on-remove).
///
/// Newtype over `usize` so dense positions never mix with entity ids at
/// the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DenseIndex(usize);

impl DenseIndex {
    /// Wraps a raw `usize` dense position.
    pub const fn from_raw(raw: usize) -> Self {
        Self(raw)
    }

    /// Raw dense position.
    pub const fn as_usize(self) -> usize {
        self.0
    }

    /// Dense position as `usize` for slice lookups.
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for DenseIndex {
    fn from(v: usize) -> Self {
        Self(v)
    }
}

impl From<DenseIndex> for usize {
    fn from(h: DenseIndex) -> Self {
        h.0
    }
}

/// A stable handle to an entity in the ECS.
///
/// `Entity` is just a plain identifier: it carries no data itself. The
/// pair `(id, generation)` lets stores distinguish a live entity from a
/// recycled one — when the id is reused after deallocation its
/// generation is bumped, so old handles fail liveness checks.
///
/// The derived order is by slot id, then generation: a total order over
/// handles, so change feeds and tests can sort without a key function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Entity {
    pub(crate) id: EntityId,
    pub(crate) generation: Generation,
}

impl Entity {
    /// Creates an entity handle with generation 0.
    ///
    /// Only meaningful for tests or deserialization: entities produced by
    /// an [`EntityAllocator`](crate::entity::EntityAllocator) may carry a
    /// higher generation if their id was recycled before.
    pub fn new(id: u32) -> Self {
        Self {
            id: EntityId::from_raw(id),
            generation: Generation::from_raw(0),
        }
    }

    /// Creates an entity handle with an explicit generation.
    pub fn new_with_gen(id: u32, generation: u32) -> Self {
        Self {
            id: EntityId::from_raw(id),
            generation: Generation::from_raw(generation),
        }
    }

    /// Returns the slot index of this entity. Recycled ids reuse indices.
    pub fn id(&self) -> u32 {
        self.id.as_u32()
    }

    /// Returns the generation guarding against stale-handle reuse.
    pub fn generation(&self) -> u32 {
        self.generation.as_u32()
    }

    /// Typed slot index of this entity.
    pub fn entity_id(&self) -> EntityId {
        self.id
    }

    /// Typed generation guarding against stale-handle reuse.
    pub fn generation_id(&self) -> Generation {
        self.generation
    }
}

/// Hands out [`Entity`] handles and tracks which ones are alive.
///
/// Freed ids go onto a free list and are recycled on the next allocate;
/// each recycle bumps the stored generation so previously issued handles
/// for that id stop reporting as alive ([`is_alive`](Self::is_alive)).
/// Generations are `u32` and wrap at `u32::MAX`, so a handle can only be
/// re-issued after 2^32 recycles of the same slot.
#[derive(Default, Clone)]
pub struct EntityAllocator {
    next_id: u32,
    free_list: Vec<u32>,
    generations: Vec<u32>,
    /// Per-slot lower bound on the generation [`Self::allocate`] may issue,
    /// indexed like `generations` but sparse: empty (and unallocated) until
    /// [`Self::advance_generations_past`] raises a floor. A slot alive in a
    /// restored allocator keeps its current generation, so the bound that
    /// protects its *next* recycle cannot live in `generations`.
    min_next_generation: Vec<u32>,
}

impl EntityAllocator {
    /// Creates an empty allocator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates a fresh entity, recycling a freed id when available.
    ///
    /// The generation issued is at least the slot's floor, so a restored
    /// allocator never re-issues a handle a newer one already handed out
    /// (see [`Self::advance_generations_past`]).
    pub fn allocate(&mut self) -> Entity {
        let id = if let Some(recycled) = self.free_list.pop() {
            recycled
        } else {
            let id = self.next_id;
            self.next_id += 1;
            if id as usize >= self.generations.len() {
                self.generations.push(0);
            }
            id
        };
        let idx = id as usize;
        let floor = self.min_next_generation.get(idx).copied().unwrap_or(0);
        let generation = self.generations[idx].max(floor);
        self.generations[idx] = generation;
        Entity {
            id: EntityId::from_raw(id),
            generation: Generation::from_raw(generation),
        }
    }

    /// Marks an entity as dead and queues its id for reuse.
    ///
    /// Bumps the id's generation, invalidating every outstanding handle
    /// to it. Deallocating an unknown or already-dead id is a no-op.
    pub fn deallocate(&mut self, entity: Entity) {
        // Idempotent: concurrent `destroy_entity` calls for the same handle
        // must not push one id onto the free list twice — duplicates would
        // hand the same id out twice on the next allocates. (`is_alive`
        // bounds-checks the index, so the explicit length check is covered.)
        if !self.is_alive(entity) {
            return;
        }
        let idx = entity.id.index();
        self.generations[idx] = entity.generation.as_u32().wrapping_add(1);
        self.free_list.push(entity.id.as_u32());
    }

    /// Returns `true` if the handle matches the current live generation
    /// for its id — i.e. the entity was allocated and not yet freed.
    ///
    /// Ids at or past the allocation frontier are never alive, even when a
    /// restored allocator reserved a generation floor for them.
    pub fn is_alive(&self, entity: Entity) -> bool {
        let idx = entity.id.index();
        if idx >= self.next_id as usize {
            return false;
        }
        self.generations[idx] == entity.generation.as_u32()
    }

    /// Raises the generation floor of every slot `live` has used past every
    /// generation `live` issued for it, so restoring an older allocator
    /// (E4 Stop) can never re-issue an `(id, generation)` pair that already
    /// named a different entity.
    ///
    /// Handles alive here stay alive: the floor bounds what the *next*
    /// allocate of a slot may issue, not its current generation. That covers
    /// all three cases — a slot alive here that `live` recycled, a slot freed
    /// here, and a slot `live` created beyond this allocator's frontier
    /// (which stays unallocated, so it never reports as alive).
    ///
    /// A slot already freed here has no handle left to preserve, so its
    /// generation moves to the floor outright: a handle `live` issued for it
    /// stops reporting as alive immediately, not just on the next allocate.
    ///
    /// Generations wrap at `u32::MAX` just as [`Self::deallocate`] does, so
    /// the guarantee holds until a single slot has been recycled 2^32 times.
    pub fn advance_generations_past(&mut self, live: &Self) {
        let len = self.min_next_generation.len().max(live.generations.len());
        self.min_next_generation.resize(len, 0);
        for (idx, &issued) in live.generations.iter().enumerate() {
            let floor = issued.wrapping_add(1);
            self.min_next_generation[idx] = self.min_next_generation[idx].max(floor);
        }
        for &id in &self.free_list {
            let idx = id as usize;
            self.generations[idx] = self.generations[idx].max(self.min_next_generation[idx]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_recycle() {
        let mut alloc = EntityAllocator::new();
        let a = alloc.allocate();
        let b = alloc.allocate();
        assert_eq!(a.id.as_u32(), 0);
        assert_eq!(b.id.as_u32(), 1);
        assert!(alloc.is_alive(a));
        alloc.deallocate(a);
        assert!(!alloc.is_alive(a));
        let c = alloc.allocate();
        assert_eq!(c.id.as_u32(), 0);
        assert_ne!(c.generation, a.generation);
        assert!(alloc.is_alive(c));
    }

    /// Double and stale `deallocate` must be no-ops: the id lands on the
    /// free list exactly once (never handed out twice) and the stored
    /// generation only moves forward.
    #[test]
    fn deallocate_twice_and_stale_is_noop() {
        let mut alloc = EntityAllocator::new();
        let a = alloc.allocate();
        alloc.deallocate(a);
        // Same handle again: the generation already bumped, so `is_alive`
        // fails and the call must not queue the id a second time.
        alloc.deallocate(a);
        let c = alloc.allocate();
        assert_eq!(c.id(), a.id());
        assert_eq!(c.generation(), a.generation() + 1);
        // The id was queued exactly once: the next allocate hands out a
        // fresh id instead of the same one a second time.
        let d = alloc.allocate();
        assert_ne!(d.id(), a.id());
        // A stale handle (pre-recycle generation) must neither roll the
        // generation back nor re-queue the id while `c` is live.
        alloc.deallocate(a);
        assert!(alloc.is_alive(c));
        assert_eq!(c.generation(), a.generation() + 1);
        // Retiring the live handle once kills it; retiring it twice still
        // queues the id only once.
        alloc.deallocate(c);
        assert!(!alloc.is_alive(c));
        alloc.deallocate(c);
        let e = alloc.allocate();
        assert_eq!(e.id(), a.id());
        assert_eq!(e.generation(), a.generation() + 2);
    }

    /// Restoring an older allocator must not re-issue a handle the newer
    /// one already handed out (E4 Stop): freed ids advance past every
    /// generation the live allocator used, and ids the live allocator
    /// created are reserved above it instead of restarting at 0.
    #[test]
    fn advance_generations_past_never_reissues_a_live_handle() {
        let mut snapshot = EntityAllocator::new();
        let kept = snapshot.allocate();
        let recycled = snapshot.allocate();
        snapshot.deallocate(recycled);

        // Play runs on a clone: it recycles the freed id twice and creates
        // two entities the snapshot never saw.
        let mut live = snapshot.clone();
        let first = live.allocate();
        live.deallocate(first);
        let second = live.allocate();
        let created = live.allocate();
        let also_created = live.allocate();
        live.deallocate(also_created);

        snapshot.advance_generations_past(&live);
        assert!(
            snapshot.is_alive(kept),
            "a restored live handle must stay valid"
        );
        let mut issued = vec![kept, recycled, first, second, created, also_created];
        for _ in 0..4 {
            let next = snapshot.allocate();
            assert!(
                !issued.contains(&next),
                "{next:?} was already issued during Play"
            );
            issued.push(next);
        }
    }

    /// A slot alive in the snapshot needs a floor too (E4 review): Play
    /// despawns it and recycles the id, and after Stop the restored handle
    /// is live again — retiring it in Edit must not hand out the recycled
    /// handle Play used.
    #[test]
    fn advance_generations_past_floors_a_slot_alive_in_the_snapshot() {
        let mut snapshot = EntityAllocator::new();
        let hero = snapshot.allocate();

        let mut live = snapshot.clone();
        live.deallocate(hero);
        let recycled = live.allocate();
        assert_eq!(recycled.id(), hero.id());
        assert_ne!(recycled.generation(), hero.generation());

        snapshot.advance_generations_past(&live);
        assert!(
            snapshot.is_alive(hero),
            "the restored handle must stay valid"
        );
        snapshot.deallocate(hero);
        let next = snapshot.allocate();
        assert_eq!(next.id(), hero.id());
        assert_ne!(next, recycled, "{next:?} already named a Play entity");
        assert_ne!(next, hero);
    }

    /// A Play handle for a slot the snapshot had already freed must be dead
    /// after the restore (E4 review): such a slot has no live handle to
    /// preserve, so its generation advances outright, not just its floor.
    #[test]
    fn advance_generations_past_kills_play_handles_of_freed_slots() {
        let mut snapshot = EntityAllocator::new();
        let kept = snapshot.allocate();
        let freed = snapshot.allocate();
        snapshot.deallocate(freed);

        let mut live = snapshot.clone();
        let play_handle = live.allocate();
        assert_eq!(play_handle.id(), freed.id());

        snapshot.advance_generations_past(&live);
        assert!(snapshot.is_alive(kept));
        assert!(
            !snapshot.is_alive(play_handle),
            "{play_handle:?} named a Play entity, not a restored one"
        );
        let next = snapshot.allocate();
        assert_eq!(next.id(), freed.id());
        assert_ne!(next, play_handle);
    }

    /// Ids `live` created past the snapshot frontier are reserved, not
    /// allocated: no handle for them may report as alive until this
    /// allocator actually hands the id out.
    #[test]
    fn reserved_ids_past_the_frontier_are_not_alive() {
        let mut snapshot = EntityAllocator::new();
        let kept = snapshot.allocate();
        let mut live = snapshot.clone();
        let created = live.allocate();

        snapshot.advance_generations_past(&live);
        assert!(snapshot.is_alive(kept));
        for generation in 0..4 {
            let phantom = Entity::new_with_gen(created.id(), generation);
            assert!(
                !snapshot.is_alive(phantom),
                "{phantom:?} was never allocated here"
            );
        }
        let issued = snapshot.allocate();
        assert_eq!(issued.id(), created.id());
        assert!(snapshot.is_alive(issued));
        assert_ne!(issued, created);
    }
}
