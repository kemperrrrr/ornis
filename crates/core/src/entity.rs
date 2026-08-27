//! Entity identifiers and allocation.
//!
//! An [`Entity`] is a lightweight handle (index + generation) that systems
//! use to reference a game object in the ECS. The [`EntityAllocator`]
//! recycles freed indices and bumps their generation, so stale handles
//! referring to destroyed entities are detected instead of silently
//! aliasing a newly created one.

/// A stable handle to an entity in the ECS.
///
/// `Entity` is just a plain identifier: it carries no data itself. The
/// pair `(id, generation)` lets stores distinguish a live entity from a
/// recycled one — when the id is reused after deallocation its
/// generation is bumped, so old handles fail liveness checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    pub(crate) id: u32,
    pub(crate) generation: u32,
}

impl Entity {
    /// Creates an entity handle with generation 0.
    ///
    /// Only meaningful for tests or deserialization: entities produced by
    /// an [`EntityAllocator`](crate::entity::EntityAllocator) may carry a
    /// higher generation if their id was recycled before.
    pub fn new(id: u32) -> Self {
        Self { id, generation: 0 }
    }

    /// Creates an entity handle with an explicit generation.
    pub fn new_with_gen(id: u32, generation: u32) -> Self {
        Self { id, generation }
    }

    /// Returns the slot index of this entity. Recycled ids reuse indices.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the generation guarding against stale-handle reuse.
    pub fn generation(&self) -> u32 {
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
        Entity { id, generation }
    }

    /// Marks an entity as dead and queues its id for reuse.
    ///
    /// Bumps the id's generation, invalidating every outstanding handle
    /// to it. Deallocating an unknown or already-dead id is a no-op.
    pub fn deallocate(&mut self, entity: Entity) {
        if entity.id as usize >= self.generations.len() {
            return;
        }
        self.generations[entity.id as usize] = entity.generation.wrapping_add(1);
        self.free_list.push(entity.id);
    }

    /// Returns `true` if the handle matches the current live generation
    /// for its id — i.e. the entity was allocated and not yet freed.
    pub fn is_alive(&self, entity: Entity) -> bool {
        if (entity.id as usize) >= self.generations.len() {
            return false;
        }
        self.generations[entity.id as usize] == entity.generation
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
        assert_eq!(a.id, 0);
        assert_eq!(b.id, 1);
        assert!(alloc.is_alive(a));
        alloc.deallocate(a);
        assert!(!alloc.is_alive(a));
        let c = alloc.allocate();
        assert_eq!(c.id, 0);
        assert_ne!(c.generation, a.generation);
        assert!(alloc.is_alive(c));
    }
}
