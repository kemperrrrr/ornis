#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    pub(crate) id: u32,
    pub(crate) generation: u32,
}

impl Entity {
    pub fn new(id: u32) -> Self {
        Self { id, generation: 0 }
    }

    pub fn new_with_gen(id: u32, generation: u32) -> Self {
        Self { id, generation }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }
}

#[derive(Default)]
pub struct EntityAllocator {
    next_id: u32,
    free_list: Vec<u32>,
    generations: Vec<u32>,
}

impl EntityAllocator {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn deallocate(&mut self, entity: Entity) {
        if entity.id as usize >= self.generations.len() {
            return;
        }
        self.generations[entity.id as usize] = entity.generation.wrapping_add(1);
        self.free_list.push(entity.id);
    }

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
