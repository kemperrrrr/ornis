use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rayon::iter::ParallelIterator;

use ornis_core::{ComponentStore, Entity, EntityAllocator};

fn setup(
    count: usize,
) -> (
    EntityAllocator,
    Vec<Entity>,
    ComponentStore<f32>,
    ComponentStore<f32>,
) {
    let mut alloc = EntityAllocator::new();
    let mut store_a = ComponentStore::new();
    let mut store_b = ComponentStore::new();
    let mut entities = Vec::with_capacity(count);
    for i in 0..count {
        let e = alloc.allocate();
        entities.push(e);
        store_a.insert(e, 1.0);
        if i % 2 == 0 {
            store_b.insert(e, 2.0);
        }
    }
    (alloc, entities, store_a, store_b)
}

fn bench_insert(c: &mut Criterion) {
    c.bench_function("insert_100k", |b| {
        let (mut alloc, _, _, _) = setup(100_000);
        b.iter(|| {
            let e = alloc.allocate();
            let mut store = ComponentStore::new();
            store.insert(black_box(e), black_box(1.0f32));
        });
    });
}

fn bench_iterate(c: &mut Criterion) {
    c.bench_function("iterate_100k", |b| {
        let (_, _, store, _) = setup(100_000);
        b.iter(|| {
            for val in store.iter() {
                black_box(val);
            }
        });
    });
}

fn bench_random_access(c: &mut Criterion) {
    c.bench_function("random_access_100k", |b| {
        let (_, entities, store, _) = setup(100_000);
        b.iter(|| {
            for &e in &entities {
                black_box(store.get(e));
            }
        });
    });
}

fn bench_intersection(c: &mut Criterion) {
    c.bench_function("intersection_100k", |b| {
        let (_, _, store_a, store_b) = setup(100_000);
        b.iter(|| {
            for (_, val_a, val_b) in store_a.iter_zip(&store_b) {
                black_box((val_a, val_b));
            }
        });
    });
}

fn bench_par_iterate(c: &mut Criterion) {
    c.bench_function("par_iterate_100k", |b| {
        let (_, _, store, _) = setup(100_000);
        b.iter(|| {
            store.par_iter().for_each(|val| {
                black_box(val);
            });
        });
    });
}

criterion_group!(
    benches,
    bench_insert,
    bench_iterate,
    bench_par_iterate,
    bench_random_access,
    bench_intersection,
);
criterion_main!(benches);
