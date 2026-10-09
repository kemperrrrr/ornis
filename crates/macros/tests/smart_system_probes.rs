//! Probe tests for `#[smart_system]` access inference: interior mutability,
//! non-plain lane bindings, parameter order, `&T` parameters, same-lane
//! read+write, parallel levels, store-alias forms, macro invocations, and
//! parity with a hand-written system.

use std::sync::{Arc, Mutex};

use ornis_core::{Resources, Schedule, SmartStore, System};

use ornis_macros::smart_system;

mod common;

use common::{Cfg, Counter, P, V, run_enforced, world};

// ---- п.1: interior mutability must be declared as a write ----

#[smart_system(writes(Mutex<Counter>))]
fn bump(resources: &Resources) {
    resources
        .get::<Mutex<Counter>>()
        .expect("counter")
        .lock()
        .expect("lock")
        .0 += 1;
}

#[test]
fn interior_mutability_write_is_declared_as_write() {
    let entry = BumpSystem::new().access();
    assert!(
        entry
            .writes
            .iter()
            .any(|id| *id == std::any::TypeId::of::<Mutex<Counter>>())
    );
    assert!(
        !entry
            .reads
            .iter()
            .any(|id| *id == std::any::TypeId::of::<Mutex<Counter>>())
    );
    assert_eq!(run_enforced(BumpSystem::new()), Ok(()));
    // Bump twice through the schedule to prove the write really lands.
    let resources = world();
    let mut schedule = Schedule::new();
    schedule.set_parallel(false).set_enforce_accesses(true);
    schedule.add_system(BumpSystem::new());
    schedule.run(&resources);
    schedule.run(&resources);
    assert_eq!(
        resources.get::<Mutex<Counter>>().unwrap().lock().unwrap().0,
        2
    );
}

// ---- п.2a: let-else lane binding ----

#[smart_system]
fn let_else_lane(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    let Some(mut pos) = store.write_lane::<P>() else {
        return;
    };
    for p in pos.iter_mut() {
        p.x += 1.0;
    }
}

#[test]
fn let_else_lane_declared() {
    assert_eq!(run_enforced(LetElseLaneSystem::new()), Ok(()));
    assert!(
        LetElseLaneSystem::new()
            .access()
            .writes_lanes
            .contains(&std::any::TypeId::of::<P>())
    );
}

// ---- п.2b: type-annotated lane binding ----

#[smart_system]
fn typed_let_lane(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    let vel: std::sync::RwLockReadGuard<'_, ornis_core::ComponentStore<V>> =
        store.read_lane::<V>().expect("vel");
    let _ = vel.len();
}

#[test]
fn typed_let_lane_declared() {
    assert_eq!(run_enforced(TypedLetLaneSystem::new()), Ok(()));
}

// ---- п.2c: lane call inline in the loop header ----

// The loop intentionally stays sequential (inline lane call): allow the
// macro's left-sequential warning, the probe asserts the access set.
#[allow(deprecated)]
#[smart_system]
fn stmt_inline(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    for v in store.write_lane::<V>().expect("vel").iter_mut() {
        v.x += 1.0;
    }
}

#[test]
fn stmt_inline_lane_declared() {
    assert_eq!(run_enforced(StmtInlineSystem::new()), Ok(()));
}

// ---- п.5: `&Resources` position is preserved ----

#[smart_system]
fn order(speed: f32, resources: &Resources) {
    let _ = (speed, resources.len());
}

#[smart_system]
fn order_mid(first: f32, resources: &Resources, last: f32) {
    let _ = (first, resources.len(), last);
}

#[test]
fn param_order_is_preserved() {
    let _ = OrderSystem::new(1.0);
    let _ = OrderMidSystem::new(1.0, 2.0);
    assert_eq!(run_enforced(OrderSystem::new(1.0)), Ok(()));
    assert_eq!(run_enforced(OrderMidSystem::new(1.0, 2.0)), Ok(()));
}

// ---- п.12: `&T` parameter becomes an owned field, no per-frame clone ----

// `&Vec<u8>` (not `&[u8]`) on purpose: the probe needs a sized owned
// target for the struct field.
#[allow(clippy::ptr_arg)]
#[smart_system]
fn table_user(resources: &Resources, table: &Vec<u8>) {
    let store = resources.get::<SmartStore>().expect("store");
    let mut pos = store.write_lane::<P>().expect("pos");
    let bonus = table.len() as f32;
    for p in pos.iter_mut() {
        p.x += bonus;
    }
}

#[test]
fn ref_param_becomes_owned_field() {
    let resources = world();
    let mut schedule = Schedule::new();
    schedule.set_parallel(false).set_enforce_accesses(true);
    schedule.add_system(TableUserSystem::new(vec![1, 2, 3]));
    schedule.run(&resources);
    let store = resources.get::<SmartStore>().expect("store");
    let lane = store.read_lane::<P>().expect("pos");
    for pos in lane.iter() {
        assert_eq!(pos.x, 3.0);
    }
}

// ---- п.13: read + write of one lane is a single `writes_lane` ----

#[smart_system]
fn read_write_same_lane(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    // Scoped: the read guard must drop before the write guard is taken
    // (`RwLock` deadlocks otherwise — store semantics, not macro business).
    let sum: f32 = {
        let pre = store.read_lane::<P>().expect("pre");
        pre.iter().map(|p| p.x).sum()
    };
    let mut pos = store.write_lane::<P>().expect("pos");
    for p in pos.iter_mut() {
        p.x += sum;
    }
}

#[test]
fn same_lane_read_write_is_single_write_entry() {
    let entry = ReadWriteSameLaneSystem::new().access();
    assert_eq!(entry.reads_lanes.len(), 0);
    assert_eq!(entry.writes_lanes.len(), 1);
    assert_eq!(run_enforced(ReadWriteSameLaneSystem::new()), Ok(()));
}

// ---- п.13: parallel levels from derived accesses ----

#[smart_system]
fn writes_p(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    let mut pos = store.write_lane::<P>().expect("pos");
    for p in pos.iter_mut() {
        p.x += 1.0;
    }
}

#[smart_system]
fn writes_v(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    let mut vel = store.write_lane::<V>().expect("vel");
    for v in vel.iter_mut() {
        v.x += 1.0;
    }
}

#[test]
fn disjoint_writers_share_a_level() {
    let mut schedule = Schedule::new();
    schedule
        .add_system(WritesPSystem::new())
        .add_system(WritesVSystem::new());
    assert_eq!(schedule.levels().len(), 1, "disjoint lanes share one level");
}

#[test]
fn conflicting_writers_split_levels() {
    let mut schedule = Schedule::new();
    schedule
        .add_system(WritesPSystem::new())
        .add_system(WritesPSystem::new());
    assert_eq!(
        schedule.levels().len(),
        2,
        "write/write conflict splits levels"
    );
}

// ---- п.3: explicit `reads(...)` exempts a helper escape ----

fn cfg_helper(resources: &Resources) -> f32 {
    resources.get::<Cfg>().map(|c| c.k).unwrap_or(0.0)
}

#[smart_system(opaque, reads(Cfg))]
fn via_helper_allowed(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    let mut pos = store.write_lane::<P>().expect("pos");
    let boost = cfg_helper(resources);
    for p in pos.iter_mut() {
        p.x += boost;
    }
}

#[test]
fn explicit_reads_exempt_helper_escape() {
    assert_eq!(run_enforced(ViaHelperAllowedSystem::new()), Ok(()));
}

// ---- п.9: raw identifier system ----

#[smart_system]
fn r#loop(resources: &Resources) {
    let _ = resources.len();
}

#[test]
fn raw_ident_system_compiles_with_unrawed_name() {
    assert_eq!(LoopSystem::new().name(), "loop");
    assert_eq!(run_enforced(LoopSystem::new()), Ok(()));
}

// ---- parity with a hand-written system ----

/// Hand-written twin of `machine_integrate` below: same per-element
/// arithmetic in the same order (bitwise parity), deliberately different
/// structure (explicit iterator + `while let`) so the pair is a parity
/// check and not a copy.
fn hand_integrate(resources: &Resources, speed: f32) {
    let cfg = resources.get::<Cfg>().expect("cfg");
    let store = resources.get::<SmartStore>().expect("store");
    let gain = (speed, cfg.k);
    let vel = store.read_lane::<V>().expect("vel");
    let mut pos = store.write_lane::<P>().expect("pos");
    let mut pairs = pos.iter_mut().zip(vel.iter());
    // Deliberately not a `for` loop: this is the hand-written twin of
    // `machine_integrate`, kept structurally distinct for the parity test.
    #[allow(clippy::while_let_on_iterator)]
    while let Some((p, v)) = pairs.next() {
        p.x += (v.x + gain.1) * gain.0;
    }
}

#[smart_system]
fn machine_integrate(resources: &Resources, speed: f32) {
    let store = resources.get::<SmartStore>().expect("store");
    let cfg = resources.get::<Cfg>().expect("cfg");
    let mut pos = store.write_lane::<P>().expect("pos");
    let vel = store.read_lane::<V>().expect("vel");
    for (p, v) in pos.iter_mut().zip(vel.iter()) {
        p.x += (v.x + cfg.k) * speed;
    }
}

fn run_hand(resources: &Resources, speed: f32) {
    hand_integrate(resources, speed);
}

#[test]
fn generated_matches_handwritten_bitwise() {
    let left = world();
    let right = world();
    let mut schedule = Schedule::new();
    schedule.set_parallel(false).set_enforce_accesses(true);
    schedule.add_system(MachineIntegrateSystem::new(0.5));
    schedule.run(&left);
    run_hand(&right, 0.5);
    let left_lane = left
        .get::<SmartStore>()
        .expect("store")
        .read_lane::<P>()
        .expect("pos");
    let right_lane = right
        .get::<SmartStore>()
        .expect("store")
        .read_lane::<P>()
        .expect("pos");
    for (generated, hand) in left_lane.iter().zip(right_lane.iter()) {
        assert_eq!(generated.x.to_bits(), hand.x.to_bits());
    }
}

// ---- п.13: Strong Confluence 1-vs-32 for a `#[smart_system]` system ----

#[smart_system]
fn wide_map(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    let cfg = resources.get::<Cfg>().expect("cfg");
    let mut pos = store.write_lane::<P>().expect("pos");
    let vel = store.read_lane::<V>().expect("vel");
    for (p, v) in pos.iter_mut().zip(vel.iter()) {
        // Map-only workload: per-element results are order-independent.
        p.x += (v.x + cfg.k) * 0.5;
    }
}

fn confluence_world(n: usize) -> Resources {
    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    store.register::<P>();
    store.register::<V>();
    for i in 0..n {
        let e = store.create_entity();
        store.insert(e, P { x: 0.0 });
        store.insert(
            e,
            V {
                x: if i % 2 == 0 { 0.1 } else { -0.1 },
            },
        );
    }
    resources.insert(store);
    resources.insert(Cfg { k: 2.0 });
    resources
}

fn collect_bits(resources: &Resources) -> Vec<u32> {
    let store = resources.get::<SmartStore>().expect("store");
    let lane = store.read_lane::<P>().expect("pos");
    // Entity insertion order, no sorting: a permutation would hide here.
    lane.iter().map(|p| p.x.to_bits()).collect()
}

#[test]
fn smart_system_confluence_1_vs_32() {
    let run = |threads: usize| {
        ornis_core::rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool")
            .install(|| {
                assert_eq!(
                    ornis_core::rayon::current_num_threads(),
                    threads,
                    "pool really runs on {threads} threads"
                );
                let resources = confluence_world(512);
                let mut schedule = Schedule::new();
                schedule.add_system(WideMapSystem::new());
                schedule.run(&resources);
                collect_bits(&resources)
            })
    };
    assert_eq!(run(1), run(32));
}

// ---- N5: wrapped interior mutability is declared as a write ----

#[smart_system(writes(Arc<Mutex<Counter>>))]
fn arc_mutex(resources: &Resources) {
    resources
        .get::<Arc<Mutex<Counter>>>()
        .expect("counter")
        .lock()
        .expect("lock")
        .0 += 1;
}

#[test]
fn n5_arc_mutex_write_is_declared_as_write() {
    let entry = ArcMutexSystem::new().access();
    assert!(
        entry
            .writes
            .iter()
            .any(|id| *id == std::any::TypeId::of::<Arc<Mutex<Counter>>>())
    );
    assert!(
        !entry
            .reads
            .iter()
            .any(|id| *id == std::any::TypeId::of::<Arc<Mutex<Counter>>>())
    );
    assert_eq!(run_enforced(ArcMutexSystem::new()), Ok(()));
}

// ---- N2: a store passed to a helper is rejected; `reads_lane` declares it ----

fn lane_helper(store: &SmartStore) -> usize {
    store.read_lane::<V>().map(|lane| lane.len()).unwrap_or(0)
}

#[smart_system(reads_lane(V))]
fn store_declared(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    let _ = lane_helper(store);
}

#[test]
fn n2_store_escape_with_lane_declared() {
    let entry = StoreDeclaredSystem::new().access();
    assert!(entry.reads_lanes.contains(&std::any::TypeId::of::<V>()));
    assert_eq!(run_enforced(StoreDeclaredSystem::new()), Ok(()));
}

// ---- N6: a re-aliased store is still tracked ----

#[smart_system]
fn realias(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    let s2 = store;
    let _ = s2.read_lane::<V>().expect("vel").len();
}

#[test]
fn n6_realiased_store_lane_declared() {
    assert_eq!(run_enforced(RealiasSystem::new()), Ok(()));
    assert!(
        RealiasSystem::new()
            .access()
            .reads_lanes
            .contains(&std::any::TypeId::of::<V>())
    );
}

// ---- N3: a lane off a `resources.get::<SmartStore>()` chain counts ----

#[smart_system]
fn chain_lane(resources: &Resources) {
    let n = resources
        .get::<SmartStore>()
        .expect("store")
        .read_lane::<V>()
        .expect("vel")
        .len();
    let _ = n;
}

#[test]
fn n3_chain_lane_declared() {
    assert_eq!(run_enforced(ChainLaneSystem::new()), Ok(()));
    assert!(
        ChainLaneSystem::new()
            .access()
            .reads_lanes
            .contains(&std::any::TypeId::of::<V>())
    );
}

// ---- N1: accesses inside known std macros are derived ----

#[smart_system]
fn in_macro(resources: &Resources) {
    assert!(resources.get::<Cfg>().is_some());
}

#[test]
fn n1_resource_in_assert_declared() {
    assert_eq!(run_enforced(InMacroSystem::new()), Ok(()));
    assert!(
        InMacroSystem::new()
            .access()
            .reads
            .contains(&std::any::TypeId::of::<Cfg>())
    );
}

#[smart_system]
fn lane_in_macro(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    println!("{}", store.read_lane::<V>().expect("vel").len());
}

#[test]
fn n1_lane_in_println_declared() {
    assert_eq!(run_enforced(LaneInMacroSystem::new()), Ok(()));
    assert!(
        LaneInMacroSystem::new()
            .access()
            .reads_lanes
            .contains(&std::any::TypeId::of::<V>())
    );
}

// ---- N4: explicit `writes` never blanket-approves an escape; `opaque` does ----

#[smart_system(opaque, reads(Cfg), writes(Mutex<Counter>))]
fn partial_explicit_allowed(resources: &Resources) {
    resources
        .get::<Mutex<Counter>>()
        .expect("counter")
        .lock()
        .expect("lock")
        .0 += 1;
    let _ = cfg_helper(resources);
}

#[test]
fn n4_opaque_with_full_declaration_passes() {
    assert_eq!(run_enforced(PartialExplicitAllowedSystem::new()), Ok(()));
}
