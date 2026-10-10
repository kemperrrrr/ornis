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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Default)]
pub struct EntityAllocator {
    next_id: u32,
    free_list: Vec<u32>,
    generations: Vec<u32>,
}

impl EntityAllocator {
    /// Creates an empty allocator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates a fresh entity, recycling a freed id when available.
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
        let generation = self.generations[id as usize];
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
    pub fn is_alive(&self, entity: Entity) -> bool {
        let idx = entity.id.index();
        if idx >= self.generations.len() {
            return false;
        }
        self.generations[idx] == entity.generation.as_u32()
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
}
