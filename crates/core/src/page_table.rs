pub const PAGE_SIZE: usize = 4096;

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

    pub fn get(&self, index: usize) -> Option<&T> {
        let offset = index % PAGE_SIZE;
        self.page(index).map(|p| &p[offset])
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let offset = index % PAGE_SIZE;
        self.page_mut(index).as_mut().map(|p| &mut p[offset])
    }

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
}
