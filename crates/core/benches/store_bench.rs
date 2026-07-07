use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ornis_core::{ComponentStore, Entity, EntityAllocator};

fn setup(count: usize) -> (EntityAllocator, Vec<Entity>, ComponentStore<f32>) {
    let mut alloc = EntityAllocator::new();
    let mut store = ComponentStore::new();
    let mut entities = Vec::with_capacity(count);
    for _ in 0..count {
        let e = alloc.allocate();
        entities.push(e);
        store.insert(e, 1.0);
    }
    (alloc, entities, store)
}

fn bench_insert(c: &mut Criterion) {
    c.bench_function("insert_100k", |b| {
        let count = 100_000;
        let (mut alloc, _, _) = setup(count);
        b.iter(|| {
            let e = alloc.allocate();
            let mut store = ComponentStore::new();
            store.insert(black_box(e), black_box(1.0f32));
        });
    });
}

fn bench_iterate(c: &mut Criterion) {
    c.bench_function("iterate_100k", |b| {
        let (_, _, store) = setup(100_000);
        b.iter(|| {
            for val in store.iter() {
                black_box(val);
            }
        });
    });
}

fn bench_random_access(c: &mut Criterion) {
    c.bench_function("random_access_100k", |b| {
        let (_, entities, store) = setup(100_000);
        b.iter(|| {
            for &e in &entities {
                black_box(store.get(e));
            }
        });
    });
}

criterion_group!(benches, bench_insert, bench_iterate, bench_random_access);
criterion_main!(benches);
