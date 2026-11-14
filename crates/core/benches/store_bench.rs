//! Benchmarks for `ComponentStore` insert and iteration throughput, and for
//! the whole-store deep copy the editor takes when Play starts.

use criterion::{Criterion, criterion_group, criterion_main};
use rayon::iter::ParallelIterator;

use ornis_core::{ComponentStore, Entity, EntityAllocator, SmartStore};

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
            store.insert(std::hint::black_box(e), std::hint::black_box(1.0f32));
        });
    });
}

fn bench_iterate(c: &mut Criterion) {
    c.bench_function("iterate_100k", |b| {
        let (_, _, store, _) = setup(100_000);
        b.iter(|| {
            for val in store.iter() {
                std::hint::black_box(val);
            }
        });
    });
}

fn bench_random_access(c: &mut Criterion) {
    c.bench_function("random_access_100k", |b| {
        let (_, entities, store, _) = setup(100_000);
        b.iter(|| {
            for &e in &entities {
                std::hint::black_box(store.get(e));
            }
        });
    });
}

fn bench_intersection(c: &mut Criterion) {
    c.bench_function("intersection_100k", |b| {
        let (_, _, store_a, store_b) = setup(100_000);
        b.iter(|| {
            for (_, val_a, val_b) in store_a.iter_zip(&store_b) {
                std::hint::black_box((val_a, val_b));
            }
        });
    });
}

fn bench_par_iterate(c: &mut Criterion) {
    c.bench_function("par_iterate_100k", |b| {
        let (_, _, store, _) = setup(100_000);
        b.iter(|| {
            store.par_iter().for_each(|val| {
                std::hint::black_box(val);
            });
        });
    });
}

/// Lane widths of the editor-shaped store: a 40-byte TRS, two 16-byte
/// vectors and a body-sized 64-byte record.
const TRS_WORDS: usize = 10;
const VECTOR_WORDS: usize = 4;
const BODY_WORDS: usize = 16;

/// Scene sizes the Play-snapshot copy is measured at.
const SNAPSHOT_SIZES: &[usize] = &[1_000, 10_000, 100_000];

/// Cost of the deep copy `EditorSession::play` takes on the Edit→Play edge
/// (E4 review): one clone per Play keypress, never on the tick path. The
/// store is editor-shaped — six hot lanes covering the widths the real
/// editor registers (a 40-byte TRS, two 16-byte vectors, a body-sized
/// 64-byte record, two scalars), one cold lane and one lock-free lane, every
/// lane populated for every entity.
fn bench_play_snapshot_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("play_snapshot_clone");
    for &count in SNAPSHOT_SIZES {
        let mut store = SmartStore::new();
        store.register::<[f32; TRS_WORDS]>();
        store.register::<[f32; VECTOR_WORDS]>();
        store.register::<[u32; VECTOR_WORDS]>();
        store.register::<[f32; BODY_WORDS]>();
        store.register::<u64>();
        store.register::<f32>();
        store.register_cold::<u32>();
        store.register_lock_free::<u16>();
        for i in 0..count {
            let e = store.create_entity();
            store.insert(e, [1.0f32; TRS_WORDS]);
            store.insert(e, [1.0f32; VECTOR_WORDS]);
            store.insert(e, [1u32; VECTOR_WORDS]);
            store.insert(e, [1.0f32; BODY_WORDS]);
            store.insert(e, i as u64);
            store.insert(e, i as f32);
            store.insert_cold(e, i as u32);
            store.write_lock_free_lane::<u16>(|lane| lane.insert(e, i as u16));
        }
        group.throughput(criterion::Throughput::Elements(count as u64));
        group.bench_function(format!("entities_{count}"), |b| {
            b.iter(|| std::hint::black_box(&store).clone());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_iterate,
    bench_par_iterate,
    bench_random_access,
    bench_intersection,
    bench_play_snapshot_clone,
);
criterion_main!(benches);
