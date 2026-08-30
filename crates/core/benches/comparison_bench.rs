//! Benchmarks comparing `ComponentStore` against a naive sparse `HashMap` store.

use std::collections::HashMap;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use ornis_core::{ComponentStore, Entity, EntityAllocator};

const COUNT: usize = 10_000;

struct PureSparseStore<T> {
    data: Vec<T>,
    entities: Vec<Entity>,
    sparse: Vec<Option<usize>>,
}

impl<T> PureSparseStore<T> {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            entities: Vec::new(),
            sparse: Vec::new(),
        }
    }

    fn insert(&mut self, entity: Entity, component: T) {
        let id = entity.id() as usize;
        if id >= self.sparse.len() {
            self.sparse.resize_with(id + 1, || None);
        }
        if let Some(idx) = self.sparse[id] {
            self.data[idx] = component;
            return;
        }
        let idx = self.data.len();
        self.data.push(component);
        self.entities.push(entity);
        self.sparse[id] = Some(idx);
    }

    fn get(&self, entity: Entity) -> Option<&T> {
        let id = entity.id() as usize;
        let dense_idx = self.sparse.get(id).copied().flatten()?;
        if entity.generation() != self.entities[dense_idx].generation() {
            return None;
        }
        self.data.get(dense_idx)
    }

    fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }
}

struct ArchetypeStore<T> {
    chunks: Vec<Vec<T>>,
    entity_to_chunk: Vec<Option<(usize, usize)>>,
}

impl<T> ArchetypeStore<T> {
    fn new() -> Self {
        Self {
            chunks: vec![Vec::new()],
            entity_to_chunk: Vec::new(),
        }
    }

    fn insert(&mut self, entity: Entity, component: T) {
        let id = entity.id() as usize;
        if id >= self.entity_to_chunk.len() {
            self.entity_to_chunk.resize_with(id + 1, || None);
        }
        let idx = self.chunks[0].len();
        self.chunks[0].push(component);
        self.entity_to_chunk[id] = Some((0, idx));
    }

    fn get(&self, entity: Entity) -> Option<&T> {
        let id = entity.id() as usize;
        let (chunk, idx) = self.entity_to_chunk.get(id).copied().flatten()?;
        self.chunks[chunk].get(idx)
    }

    fn iter(&self) -> impl Iterator<Item = &T> {
        self.chunks[0].iter()
    }
}

fn setup_hybrid(count: usize) -> (Vec<Entity>, ComponentStore<f32>) {
    let mut alloc = EntityAllocator::new();
    let mut store = ComponentStore::new();
    let entities: Vec<_> = (0..count).map(|_| alloc.allocate()).collect();
    for &e in &entities {
        store.insert(e, 1.0);
    }
    (entities, store)
}

fn setup_pure(count: usize) -> (Vec<Entity>, PureSparseStore<f32>) {
    let mut alloc = EntityAllocator::new();
    let mut store = PureSparseStore::new();
    let entities: Vec<_> = (0..count).map(|_| alloc.allocate()).collect();
    for &e in &entities {
        store.insert(e, 1.0);
    }
    (entities, store)
}

fn setup_hashmap(count: usize) -> (Vec<Entity>, HashMap<Entity, f32>) {
    let mut alloc = EntityAllocator::new();
    let mut store = HashMap::new();
    let entities: Vec<_> = (0..count).map(|_| alloc.allocate()).collect();
    for &e in &entities {
        store.insert(e, 1.0);
    }
    (entities, store)
}

fn setup_archetype(count: usize) -> (Vec<Entity>, ArchetypeStore<f32>) {
    let mut alloc = EntityAllocator::new();
    let mut store = ArchetypeStore::new();
    let entities: Vec<_> = (0..count).map(|_| alloc.allocate()).collect();
    for &e in &entities {
        store.insert(e, 1.0);
    }
    (entities, store)
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    group.sample_size(100);

    group.bench_function("hybrid", |b| {
        let mut alloc = EntityAllocator::new();
        b.iter(|| {
            let mut store = ComponentStore::new();
            for _ in 0..COUNT {
                store.insert(alloc.allocate(), 1.0);
            }
            black_box(store.len());
        });
    });

    group.bench_function("pure_sparse", |b| {
        let mut alloc = EntityAllocator::new();
        b.iter(|| {
            let mut store = PureSparseStore::new();
            for _ in 0..COUNT {
                store.insert(alloc.allocate(), 1.0);
            }
            black_box(store.data.len());
        });
    });

    group.bench_function("hashmap", |b| {
        let mut alloc = EntityAllocator::new();
        b.iter(|| {
            let mut store = HashMap::new();
            for _ in 0..COUNT {
                store.insert(alloc.allocate(), 1.0);
            }
            black_box(store.len());
        });
    });

    group.bench_function("archetype", |b| {
        let mut alloc = EntityAllocator::new();
        b.iter(|| {
            let mut store = ArchetypeStore::new();
            for _ in 0..COUNT {
                store.insert(alloc.allocate(), 1.0);
            }
            black_box(store.chunks[0].len());
        });
    });

    group.finish();
}

fn bench_iterate(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterate");
    group.sample_size(100);

    let (_, hybrid) = setup_hybrid(COUNT);
    let (_, pure) = setup_pure(COUNT);
    let (_, hash) = setup_hashmap(COUNT);
    let (_, arch) = setup_archetype(COUNT);

    group.bench_function("hybrid", |b| {
        b.iter(|| {
            for v in hybrid.iter() {
                black_box(v);
            }
        });
    });

    group.bench_function("pure_sparse", |b| {
        b.iter(|| {
            for v in pure.iter() {
                black_box(v);
            }
        });
    });

    group.bench_function("hashmap", |b| {
        b.iter(|| {
            for v in hash.values() {
                black_box(v);
            }
        });
    });

    group.bench_function("archetype", |b| {
        b.iter(|| {
            for v in arch.iter() {
                black_box(v);
            }
        });
    });

    group.finish();
}

fn bench_random_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_access");
    group.sample_size(100);

    let (entities, hybrid) = setup_hybrid(COUNT);
    let (_, pure) = setup_pure(COUNT);
    let (_, hash) = setup_hashmap(COUNT);
    let (_, arch) = setup_archetype(COUNT);

    group.bench_function("hybrid", |b| {
        b.iter(|| {
            for &e in &entities {
                black_box(hybrid.get(e));
            }
        });
    });

    group.bench_function("pure_sparse", |b| {
        b.iter(|| {
            for &e in &entities {
                black_box(pure.get(e));
            }
        });
    });

    group.bench_function("hashmap", |b| {
        b.iter(|| {
            for &e in &entities {
                black_box(hash.get(&e));
            }
        });
    });

    group.bench_function("archetype", |b| {
        b.iter(|| {
            for &e in &entities {
                black_box(arch.get(e));
            }
        });
    });

    group.finish();
}

fn bench_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");
    group.sample_size(50);

    let count = COUNT;

    let (_, hybrid) = setup_hybrid(count);
    let (_, pure) = setup_pure(count);
    let (_, hash) = setup_hashmap(count);
    let (_, arch) = setup_archetype(count);

    group.bench_function("hybrid", |b| {
        b.iter(|| {
            let size = hybrid.data.capacity() * std::mem::size_of::<f32>()
                + hybrid.entities.capacity() * std::mem::size_of::<Entity>();
            black_box(size);
        });
    });

    group.bench_function("pure_sparse", |b| {
        b.iter(|| {
            let size = pure.data.capacity() * std::mem::size_of::<f32>()
                + pure.sparse.capacity() * std::mem::size_of::<Option<usize>>();
            black_box(size);
        });
    });

    group.bench_function("hashmap", |b| {
        b.iter(|| {
            let size = hash.capacity();
            black_box(size);
        });
    });

    group.bench_function("archetype", |b| {
        b.iter(|| {
            let size: usize = arch
                .chunks
                .iter()
                .map(|c| c.capacity() * std::mem::size_of::<f32>())
                .sum();
            black_box(size);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_iterate,
    bench_random_access,
    bench_memory
);
criterion_main!(benches);
