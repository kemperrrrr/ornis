//! Sparse paged storage backing the smart store's virtual slots.
//!
//! A [`PageTable`] maps dense indices to values in fixed-size pages that
//! are allocated lazily on first write. This keeps memory proportional to
//! actual occupancy while preserving O(1) indexed access — the classic
//! sparse-set/page-table trade-off used by packed component lanes.
/// Number of slots per page. Chosen to match a typical 4 KiB arena so a
/// fully populated page of small components stays cache and allocator
/// friendly.
pub const PAGE_SIZE: usize = 4096;

/// Sparse, lazily paged vector indexed by dense `usize` slots.
///
/// Pages of `PAGE_SIZE` slots are allocated only when a slot inside them
/// is first written via [`set`](PageTable::set); reads of untouched
/// regions return `None` without allocating. This gives O(1) access with
/// memory proportional to occupancy - used as backing storage for packed
/// component lanes in [`SmartStore`](crate::SmartStore).
///
/// `T: Clone + Default` is required: untouched slots conceptually hold
/// `T::default()` and pages are materialized by filling with defaults.
pub struct PageTable<T> {
    pages: Vec<Option<Box<[T; PAGE_SIZE]>>>,
}

impl<T: Clone + Default> Default for PageTable<T> {
    fn default() -> Self {
        Self { pages: Vec::new() }
    }
}

impl<T: Clone + Default> Clone for PageTable<T> {
    fn clone(&self) -> Self {
        Self {
            pages: self
                .pages
                .iter()
                .map(|opt| {
                    opt.as_ref().map(|b| {
                        let mut v: Vec<T> = Vec::with_capacity(PAGE_SIZE);
                        v.extend_from_slice(&b[..]);
                        v.into_boxed_slice().try_into().ok().unwrap()
                    })
                })
                .collect(),
        }
    }
}

impl<T: Clone + Default> PageTable<T> {
    /// Creates an empty table with no pages allocated.
    pub fn new() -> Self {
        Self::default()
    }

    fn page_mut(&mut self, index: usize) -> &mut Option<Box<[T; PAGE_SIZE]>> {
        let page_id = index / PAGE_SIZE;
        if page_id >= self.pages.len() {
            self.pages.resize_with(page_id + 1, || None);
        }
        &mut self.pages[page_id]
    }

    fn page(&self, index: usize) -> Option<&[T; PAGE_SIZE]> {
        let page_id = index / PAGE_SIZE;
        self.pages.get(page_id)?.as_ref().map(|b| &**b)
    }

    /// Returns a reference to the slot at `index`, or `None` if the
    /// containing page was never allocated.
    pub fn get(&self, index: usize) -> Option<&T> {
        let offset = index % PAGE_SIZE;
        self.page(index).map(|p| &p[offset])
    }

    /// Returns an exclusive reference to the slot at `index`. Unlike
    /// [`set`](PageTable::set), this does NOT allocate a missing page:
    /// returning `None` preserves the "untouched" semantics of sparse slots.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let offset = index % PAGE_SIZE;
        self.page_mut(index).as_mut().map(|p| &mut p[offset])
    }

    /// Writes `value` into the slot at `index`, allocating its page
    /// (filled with `T::default()`) if this is the first touch.
    pub fn set(&mut self, index: usize, value: T) {
        let page = self.page_mut(index);
        let offset = index % PAGE_SIZE;
        let p = page.get_or_insert_with(|| Box::new(std::array::from_fn(|_| T::default())));
        p[offset] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_get() {
        let mut pt: PageTable<usize> = PageTable::new();
        pt.set(0, 42);
        pt.set(PAGE_SIZE, 100);
        assert_eq!(pt.get(0), Some(&42));
        assert_eq!(pt.get(PAGE_SIZE), Some(&100));
    }

    #[test]
    fn lazy_allocation() {
        let mut pt: PageTable<usize> = PageTable::new();
        pt.set(9999, 7);
        assert_eq!(pt.get(9999), Some(&7));
        assert!(pt.get(0).is_none());
    }

    #[test]
    fn get_mut_offsets_within_page() {
        // The page offset must be `index % PAGE_SIZE`, not `/` or `+`:
        // both alternatives would either alias another slot or panic.
        let mut pt: PageTable<usize> = PageTable::new();
        pt.set(PAGE_SIZE + 7, 42);
        pt.set(2 * PAGE_SIZE + 3, 99);

        assert_eq!(pt.get_mut(PAGE_SIZE + 7), Some(&mut 42));
        assert_eq!(pt.get_mut(2 * PAGE_SIZE + 3), Some(&mut 99));

        // Mutations through get_mut must be visible via get.
        *pt.get_mut(PAGE_SIZE + 7).unwrap() = 1;
        assert_eq!(pt.get(PAGE_SIZE + 7), Some(&1));
        // A different slot in the same page is untouched.
        assert_eq!(pt.get(2 * PAGE_SIZE + 3), Some(&99));
    }

    #[test]
    fn get_mut_missing_page_is_none() {
        let mut pt: PageTable<usize> = PageTable::new();
        // No page allocated yet: get_mut must return None (not a leaked default).
        assert!(pt.get_mut(0).is_none());
        assert!(pt.get_mut(PAGE_SIZE).is_none());
    }

    #[test]
    fn clone_preserves_and_is_independent() {
        let mut pt: PageTable<usize> = PageTable::new();
        pt.set(0, 10);
        pt.set(PAGE_SIZE + 1, 20);

        let copy = pt.clone();
        assert_eq!(copy.get(0), Some(&10));
        assert_eq!(copy.get(PAGE_SIZE + 1), Some(&20));

        // Mutating the clone must not touch the source.
        let mut copy = copy;
        *copy.get_mut(0).unwrap() = 99;
        assert_eq!(pt.get(0), Some(&10));
    }
}
