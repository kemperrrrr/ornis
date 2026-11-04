//! Strong-confluence gate for R4 deferred structural ops (RAT-57, #115).
//!
//! Enqueue order and the applied state must be bitwise identical at 1 and
//! 32 rayon threads: the `StructuralWriter` parameter declares a write to
//! the queue, so every queueing system serializes in registration order and
//! the FIFO apply reproduces that order regardless of the thread count.
//!
//! Split from `confluence_tests.rs` to keep that file under the SRP
//! `[tests].file_length` limit.

use ornis_core::{
    Pack, Resources, Schedule, SmartStore, StructuralWriter,
    mutation::{apply_structural_queue, register_structural_queue},
};
use ornis_macros::smart_system;

/// Point rayon's global pool at `threads` before the first pool init.
fn set_rayon_threads(threads: usize) {
    // SAFETY: no other test in this binary touches the process environment,
    // and rayon reads `RAYON_NUM_THREADS` once at first pool init; no
    // concurrent env reader exists while these tests run.
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }
}

/// Bundle written by the queued structural path (dense field lane).
#[derive(Clone, Copy, Debug, PartialEq, ornis_macros::Pack)]
struct Gauge {
    level: f32,
}

/// Component queued through `StructuralWriter::insert`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Marker {
    id: u32,
}

/// Lane-only sibling: disjoint from the queue, so it shares a level with
/// the first queue writer and runs on rayon while ops are enqueued.
#[derive(Clone, Copy, Debug, PartialEq, ornis_macros::Pack)]
struct Drift {
    dx: f32,
}

#[smart_system]
fn bump_gauge(u: &mut Gauge, #[entity] e: Entity, ops: StructuralWriter) {
    u.level += 1.0;
    ops.insert(e, Marker { id: u.level as u32 });
}

#[smart_system]
fn drift_lane(u: &mut Drift) {
    u.dx += 0.25;
}

#[smart_system]
fn bump_gauge_again(u: &mut Gauge, #[entity] e: Entity, ops: StructuralWriter) {
    u.level += 2.0;
    ops.insert(
        e,
        Marker {
            id: 100 + u.level as u32,
        },
    );
}

/// Runs the structural schedule at `threads` rayon threads and returns
/// `(level bits, marker id)` per seeded entity in id order. The second
/// writer conflicts with the first (queue + `Gauge` lane), while the
/// lane-only sibling shares the first writer's level — so the enqueue
/// races a rayon-parallel system, and the queue must still apply in
/// registration order.
fn run_structural_with_threads(threads: usize) -> Vec<(u32, u32)> {
    set_rayon_threads(threads);

    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    Gauge::pack_register(&mut store);
    Drift::pack_register(&mut store);
    store.register::<Marker>();
    let mut entities = Vec::with_capacity(64);
    for i in 0..64 {
        let entity = store.create_entity();
        Gauge { level: i as f32 }.pack_insert(&mut store, entity);
        Drift { dx: i as f32 }.pack_insert(&mut store, entity);
        entities.push(entity);
    }
    register_structural_queue(&mut resources);
    resources.insert(store);

    let mut schedule = Schedule::new();
    schedule.add_system(BumpGaugeSystem::new());
    schedule.add_system(DriftLaneSystem::new());
    schedule.add_system(BumpGaugeAgainSystem::new());
    schedule.run(&resources);
    apply_structural_queue(&mut resources);

    let store = resources.get::<SmartStore>().expect("store");
    entities
        .iter()
        .map(|&entity| {
            let level = Gauge::pack_get(store, entity).expect("gauge").level;
            let marker = store
                .read_lane::<Marker>()
                .and_then(|lane| lane.get(entity).copied())
                .expect("queued insert applied");
            (level.to_bits(), marker.id)
        })
        .collect()
}

/// RAT-57 / issue #115: deferred structural ops are confluence-neutral —
/// enqueue order (queue-conflict levels) and apply state must be bitwise
/// identical at 1 and 32 rayon threads.
#[test]
fn strong_confluence_structural_queue() {
    let threads_1 = run_structural_with_threads(1);
    let threads_32 = run_structural_with_threads(32);

    assert_eq!(threads_1.len(), 64, "structural_queue");
    // Sanity: both writers' lane touches and the queued insert landed —
    // seed + 1 + 2, marker from the last writer (100 + level).
    assert_eq!(threads_1[0].0, (3.0f32).to_bits());
    assert_eq!(threads_1[0].1, 103);
    assert_eq!(threads_1[63].0, (66.0f32).to_bits());
    assert_eq!(threads_1[63].1, 166);

    for ((l1, m1), (l2, m2)) in threads_1.iter().zip(&threads_32) {
        assert_eq!(l1, l2, "structural_queue: level bits diverge");
        assert_eq!(m1, m2, "structural_queue: marker id diverge");
    }
}
